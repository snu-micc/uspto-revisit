"""Asynchronous SMILES lookup utilities."""

from __future__ import annotations

import asyncio
import csv
import html
import json
import logging
import pickle
import random
import re
import unicodedata
from pathlib import Path
from urllib.parse import quote

import nest_asyncio
from rdkit import Chem
from tqdm import tqdm

from uspto_revisit.chemspider import get_smiles_from_chemspider_async
from uspto_revisit.json_utils import fix_json_string, fix_name, parse_json_object

nest_asyncio.apply()

DEFAULT_KNOWN_SMILES = {
    "ice": "O",
    "DCM": "C(Cl)Cl",
    "DMA": "CC(=O)N",
    "DMF": "CN(C)C=O",
    "THF": "C1CCOC1",
    "ether": "CCOCC",
    "absolute ethanol": "CCO",
    "dry acetone": "CC(=O)C",
    "dry ether": "CCOCC",
    "dry tetrahydrofuran": "C1CCOC1",
    "dry toluene": "Cc1ccccc1",
    "ultrapure water": "O",
    "sodium methylate": "C[O-].[Na+]",
    "lithiumaluminium hydride": "[Li+].[AlH4-]",
    "racemic phenylalanine": "NC(Cc1ccccc1)C(=O)O",
    "brine": "O.[Na+].[Cl-]",
    "Pd/C": "Pd",
    "DMSO": "CS(=O)C",
    "LiAlH4": "[Li+].[AlH4-]",
}

# Only aliases that preserve a specific chemical identity belong here. Mixtures,
# supported catalysts, and patent-local labels are deliberately not aliased.
SAFE_NAME_ALIASES = {
    "1,1'-carbonylbis(imidazole)": "1,1'-carbonyldiimidazole",
    "n,n'-carbonyldiimidazole": "1,1'-carbonyldiimidazole",
    "bis-(triphenylphosphine)palladium(ii) dichloride": (
        "bis(triphenylphosphine)palladium(II) dichloride"
    ),
    "bis(triphenylphosphine)palladium(ii) chloride": (
        "bis(triphenylphosphine)palladium(II) dichloride"
    ),
    "diea": "DIPEA",
    "mo(co)6": "molybdenum hexacarbonyl",
    "palladium tetrakis(triphenylphosphine)": (
        "tetrakis(triphenylphosphine)palladium(0)"
    ),
    "phosphorous pentachloride": "phosphorus pentachloride",
    "ruphos": "2-dicyclohexylphosphino-2',6'-diisopropoxybiphenyl",
    "tetrakis(triphenylphosphine)palladium": (
        "tetrakis(triphenylphosphine)palladium(0)"
    ),
    "tri(dibenzylidenacetone)dipalladium(0)": (
        "tris(dibenzylideneacetone)dipalladium(0)"
    ),
    "tri(dibenzylideneacetone)dipalladium(0)": (
        "tris(dibenzylideneacetone)dipalladium(0)"
    ),
    "tris(dibenzylideneacetone)dipalladium": (
        "tris(dibenzylideneacetone)dipalladium(0)"
    ),
}

AMBIGUOUS_EXACT_NAMES = {
    "4a molecular sieve",
    "4å molecular sieve",
    "ad-7 resin",
    "ad-mix-α",
    "activated charcoal",
    "ala-cholesterol",
    "alcohol",
    "alcoholic solution",
    "charcoal",
    "cell lysate",
    "celite",
    "diatomaceous earth",
    "filtering agent",
    "glucose dehydrogenase",
    "hexanes",
    "ketoreductase",
    "mineral oil",
    "organometallic catalyst",
    "oxone",
    "oxone®",
    "palladium hydroxide on carbon",
    "palladium on carbon",
    "palladium on charcoal",
    "paraformaldehyde",
    "petroleum ether",
    "petrolether",
    "polyphosphoric acid",
    "raney nickel",
    "saline",
    "sephadex lh-20",
    "sephadex lh20",
    "silica",
    "silica gel",
    "white solid",
    "xylenes",
    "z-ala-cholesterol",
    "zygosaccharomyces rouxii atcc 14462",
}

AMBIGUOUS_NAME_PATTERNS = (
    re.compile(r"^(?:compound|example|intermediate)(?:\s|$)", re.I),
    re.compile(r"^(?:product|reagents?)\s+(?:from|for|of|analogous\s+to)(?:\s|$)", re.I),
    re.compile(r"^(?:title compound|acid chloride|acetylating agent|alkylating agent)$", re.I),
    re.compile(r"^(?:n-acetylating|n-methylating) agent$", re.I),
    re.compile(r"^(?:ketone|silyl ether)\s+[A-Za-z0-9.-]+$", re.I),
    re.compile(r"^[A-Za-z]?\d+(?:[-.]\d+)*$"),
    re.compile(r"\b(?:PAMAM|resin|copolymer|co-polymer|molecular sieve)\b", re.I),
    re.compile(r"^benzhydrol\s+.*benzoate$", re.I),
    re.compile(r"^sephadex\b", re.I),
    re.compile(r"^(?:crude product|activated ester|mixed anhydride)\b", re.I),
    re.compile(
        r"\b(?:filter aid|mixture|solution in|aqueous solution|on charcoal|on carbon)\b",
        re.I,
    ),
)

smiles_cache = {}
resolution_overrides = {}


def _clean_compound_name(compound_name: str) -> str:
    return " ".join(str(compound_name).replace("′", "'").split())


def normalize_name_key(compound_name: str) -> str:
    """Return a stable key for exact, punctuation-insensitive name matching."""
    cleaned = html.unescape(re.sub(r"<[^>]+>", "", _clean_compound_name(compound_name)))
    cleaned = (
        cleaned.replace("α", "alpha")
        .replace("β", "beta")
        .replace("γ", "gamma")
        .replace("δ", "delta")
    )
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", "", cleaned)


def load_resolution_overrides(path: str | Path | None) -> dict:
    """Load auditable, dataset-specific aliases or active-component choices."""
    global resolution_overrides
    resolution_overrides = {}
    if not path:
        return resolution_overrides

    override_path = Path(path)
    with override_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    entries = payload.get("overrides", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("NoSmi overrides must be a list or an object with 'overrides'.")

    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError("Every NoSmi override must contain a non-empty 'name'.")
        if not entry.get("lookup_name") and not entry.get("smiles"):
            raise ValueError(
                f"Override for {entry['name']!r} needs 'lookup_name' or 'smiles'."
            )
        normalized = normalize_name_key(entry["name"])
        if not normalized:
            raise ValueError(f"Override name cannot be normalized: {entry['name']!r}")
        if normalized in resolution_overrides:
            raise ValueError(f"Duplicate NoSmi override name: {entry['name']!r}")
        copied = dict(entry)
        if copied.get("smiles"):
            copied["smiles"] = canonicalize_smiles(copied["smiles"])
            if not copied["smiles"]:
                raise ValueError(f"Invalid override SMILES for {entry['name']!r}")
        resolution_overrides[normalized] = copied
    return resolution_overrides


def resolve_name_alias(compound_name: str) -> str:
    """Return a conservative, identity-preserving lookup alias."""
    cleaned = _clean_compound_name(compound_name)
    return SAFE_NAME_ALIASES.get(cleaned.casefold(), cleaned)


def should_skip_automatic_resolution(compound_name: str) -> bool:
    """Reject mixtures, materials, and underspecified patent-local labels."""
    cleaned = _clean_compound_name(compound_name)
    if cleaned.casefold() in AMBIGUOUS_EXACT_NAMES:
        return True
    return any(pattern.search(cleaned) for pattern in AMBIGUOUS_NAME_PATTERNS)


def canonicalize_smiles(smiles: str | None) -> str | None:
    """Validate and canonicalize one SMILES value with RDKit."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        molecule = Chem.MolFromSmiles(smiles.strip())
        if molecule is None:
            return None
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    except Exception as exc:
        logging.warning("RDKit rejected SMILES %r: %s", smiles, exc)
        return None


def _known_smiles(compound_name: str) -> str | None:
    cleaned = _clean_compound_name(compound_name).casefold()
    for name, smiles in DEFAULT_KNOWN_SMILES.items():
        if name.casefold() == cleaned:
            return canonicalize_smiles(smiles)
    return None


def extract_pubchem_smiles(response_text: str) -> str | None:
    """Read current and legacy PubChem PUG REST SMILES property names."""
    data = json.loads(response_text)
    properties = data.get("PropertyTable", {}).get("Properties", [])
    if not properties:
        return None
    record = properties[0]
    for key in ("SMILES", "IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES"):
        value = canonicalize_smiles(record.get(key))
        if value:
            return value
    return None


def load_cache(cache_path: str | Path = "smiles_cache.pkl") -> None:
    global smiles_cache
    path = Path(cache_path)
    if path.exists():
        with path.open("rb") as handle:
            smiles_cache = pickle.load(handle)


def save_cache(cache_path: str | Path = "smiles_cache.pkl") -> None:
    with Path(cache_path).open("wb") as handle:
        pickle.dump(smiles_cache, handle)


def audit_name_smiles_consistency(
    cache_path: str | Path,
    output_path: str | Path,
) -> int:
    """Write same-normalized-name cache entries that resolve to different structures."""
    path = Path(cache_path)
    rows = []
    if path.is_file():
        with path.open("rb") as handle:
            cache = pickle.load(handle)
        grouped = {}
        for name, smiles in cache.items():
            canonical = canonicalize_smiles(smiles)
            normalized = normalize_name_key(str(name))
            if canonical and normalized:
                grouped.setdefault(normalized, {}).setdefault(canonical, []).append(str(name))
        for normalized, structures in grouped.items():
            if len(structures) > 1:
                for smiles, names in structures.items():
                    rows.append({"normalized_name": normalized, "smiles": smiles, "cache_names": " | ".join(sorted(names))})
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("normalized_name", "smiles", "cache_names"))
        writer.writeheader()
        writer.writerows(rows)
    return len({row["normalized_name"] for row in rows})


async def exponential_backoff(attempt: int, max_delay: int = 60) -> None:
    delay = min(max_delay, (2**attempt) + random.uniform(0, 1))
    await asyncio.sleep(delay)


async def fetch_smiles(session, url: str, semaphore, max_retries: int) -> str | None:
    import aiohttp

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.get(url) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        return response_text
                    if response.status == 429 or "ServerBusy" in response_text:
                        logging.warning(
                            "[Busy] Server busy, retrying... (Attempt %s/%s) for URL: %s",
                            attempt + 1,
                            max_retries,
                            url,
                        )
                        await exponential_backoff(attempt)
                    elif 400 <= response.status < 500:
                        logging.info(
                            "[NotFound] Lookup returned status %s for URL: %s",
                            response.status,
                            url,
                        )
                        return None
                    else:
                        logging.error(
                            "[Error] Failed with status code: %s for URL: %s",
                            response.status,
                            url,
                        )
                        response.raise_for_status()
            except asyncio.TimeoutError:
                logging.error("[Timeout] Failed to fetch SMILES from %s", url)
            except aiohttp.ClientError as exc:
                logging.error("[ClientError] Failed to fetch SMILES from %s [Error] %s", url, exc)
                if attempt < max_retries - 1:
                    await exponential_backoff(attempt)
            except Exception as exc:
                logging.error("[Error] Failed to fetch SMILES from %s [Error] %s", url, exc)
                if attempt < max_retries - 1:
                    await exponential_backoff(attempt)
    return None


async def get_smiles_from_pubchem(session, compound_name, semaphore, max_retries):
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(compound_name)}/property/SMILES/JSON"
    )
    try:
        response_text = await fetch_smiles(session, url, semaphore, max_retries)
        if response_text:
            smiles = extract_pubchem_smiles(response_text)
            if smiles:
                return smiles, "PubChem"
    except Exception as exc:
        logging.error("[PubChem] Failed for %s from URL: %s [Error] %s", compound_name, url, exc)
    return None, None


async def get_smiles_from_cir(session, compound_name, semaphore, max_retries):
    url = f"http://cactus.nci.nih.gov/chemical/structure/{quote(compound_name)}/smiles"
    try:
        smiles = canonicalize_smiles(
            await fetch_smiles(session, url, semaphore, max_retries)
        )
        return (smiles, "CIR") if smiles else (None, None)
    except Exception as exc:
        logging.error("[CIR] Failed for %s from URL: %s [Error] %s", compound_name, url, exc)
    return None, None


async def get_smiles_from_opsin(session, compound_name, semaphore, max_retries):
    url = f"https://opsin.ch.cam.ac.uk/opsin/{quote(compound_name)}.smi"
    try:
        smiles = canonicalize_smiles(
            await fetch_smiles(session, url, semaphore, max_retries)
        )
        return (smiles, "OPSIN") if smiles else (None, None)
    except Exception as exc:
        logging.error("[OPSIN] Failed for %s from URL: %s [Error] %s", compound_name, url, exc)
    return None, None


def _chebi_detail_names(detail: dict) -> list[str]:
    names = [detail.get("name", ""), detail.get("ascii_name", "")]
    for group in (detail.get("names") or {}).values():
        if not isinstance(group, list):
            continue
        for item in group:
            if isinstance(item, dict):
                names.extend((item.get("name", ""), item.get("ascii_name", "")))
    return [name for name in names if name]


async def get_smiles_from_chebi(session, compound_name, semaphore, max_retries):
    """Resolve an exact ChEBI primary name or curated synonym to SMILES."""
    search_url = (
        "https://www.ebi.ac.uk/chebi/backend/api/public/es_search"
        f"?term={quote(compound_name)}&page=1&size=3"
    )
    try:
        response_text = await fetch_smiles(
            session,
            search_url,
            semaphore,
            max_retries,
        )
        if not response_text:
            return None, None
        data = json.loads(response_text)
        query_key = normalize_name_key(compound_name)
        hits = data.get("results", [])

        # Primary-name matches can be accepted directly from the search record.
        for hit in hits:
            source = hit.get("_source", {}) if isinstance(hit, dict) else {}
            primary_names = (source.get("name", ""), source.get("ascii_name", ""))
            if query_key in {normalize_name_key(name) for name in primary_names if name}:
                smiles = canonicalize_smiles(source.get("smiles"))
                if smiles:
                    return smiles, "ChEBI"

        # A short acronym can be an exact synonym for unrelated compounds (for
        # example BOP). Require at least five normalized characters before using
        # synonym matches, and inspect the full ChEBI record.
        if len(query_key) < 5:
            return None, None
        for hit in hits:
            chebi_id = str(hit.get("_id", "")) if isinstance(hit, dict) else ""
            if not chebi_id:
                continue
            detail_url = (
                "https://www.ebi.ac.uk/chebi/backend/api/public/compound/"
                f"{quote(chebi_id)}/"
            )
            detail_text = await fetch_smiles(
                session,
                detail_url,
                semaphore,
                max_retries,
            )
            if not detail_text:
                continue
            detail = json.loads(detail_text)
            detail_keys = {
                normalize_name_key(name) for name in _chebi_detail_names(detail)
            }
            if query_key not in detail_keys:
                continue
            smiles = canonicalize_smiles(
                (detail.get("default_structure") or {}).get("smiles")
            )
            if smiles:
                return smiles, "ChEBI"
    except Exception as exc:
        logging.error(
            "[ChEBI] Failed for %s from URL: %s [Error] %s",
            compound_name,
            search_url,
            exc,
        )
    return None, None


async def get_smiles(session, compound_name, fix_name_bool, semaphore):
    if not isinstance(compound_name, str):
        logging.warning(
            "compound_name is not a string: %s (type: %s)",
            compound_name,
            type(compound_name),
        )
        compound_name = str(compound_name)

    original_name = _clean_compound_name(compound_name)
    known_smiles = _known_smiles(original_name)
    if known_smiles:
        return known_smiles, "KnownAlias"

    compound_name = resolve_name_alias(original_name)
    if fix_name_bool:
        compound_name = fix_name(compound_name)
        compound_name = resolve_name_alias(compound_name)
        known_smiles = _known_smiles(compound_name)
        if known_smiles:
            return known_smiles, "KnownAlias"

    for cache_key in (original_name, compound_name):
        cached_smiles = canonicalize_smiles(smiles_cache.get(cache_key))
        if cached_smiles:
            return cached_smiles, "Cache"

    result, source = await get_smiles_from_opsin(
        session,
        compound_name,
        semaphore,
        max_retries=2,
    )
    if result:
        smiles_cache[original_name] = result
        smiles_cache[compound_name] = result
        logging.info("Found SMILES for %s from %s: %s", compound_name, source, result)
        return result, source

    logging.warning("Failed to find SMILES for %s in all sources.", compound_name)
    return None, None


async def get_smiles_dict(response, session, fix_name_bool, semaphore):
    smiles_dict = {}
    problem_chemicals = []

    async def process_chemicals(chemicals_dict, category):
        tasks = {
            code: get_smiles(session, compound_name, fix_name_bool, semaphore)
            for code, compound_name in chemicals_dict.items()
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for code, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logging.error("SMILES lookup failed for %s: %s", code, result)
                result = (None, None)
            smiles, source = result
            if smiles:
                smiles_dict[code] = smiles
                logging.info("Found SMILES for %s from %s: %s", chemicals_dict[code], source, smiles)
                continue

            fixed_name = fix_name(chemicals_dict[code])
            fixed_smiles, fixed_source = await get_smiles(session, fixed_name, True, semaphore)
            if fixed_smiles:
                smiles_dict[code] = fixed_smiles
                logging.info(
                    "Found SMILES for %s (Fixed Name) from %s: %s",
                    chemicals_dict[code],
                    fixed_source,
                    fixed_smiles,
                )
            else:
                problem_chemicals.append(f"{chemicals_dict[code]} ({category})")
                smiles_dict[code] = f"[{chemicals_dict[code]} (NoSmi)]"

    if "Reactants, Solvents, Catalysts" in response:
        await process_chemicals(response["Reactants, Solvents, Catalysts"], "Reactant/Solvent/Catalyst")
    product_key = "Product" if "Product" in response else "Products" if "Products" in response else None
    if product_key:
        await process_chemicals(response[product_key], "Product")
    if problem_chemicals:
        logging.info("Problem chemicals: %s", problem_chemicals)

    return smiles_dict


async def process_batch(json_responses, fix_name_bool, semaphore):
    import aiohttp
    import certifi
    import ssl

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_context)
    ) as session:
        tasks = []
        task_positions = []
        results = [{} for _ in json_responses]
        for idx, json_response in enumerate(json_responses):
            parsed_response = parse_json_object(json_response)
            if parsed_response is None:
                fixed_json = fix_json_string(json_response)
                parsed_response = parse_json_object(fixed_json) if fixed_json else None
                if parsed_response is None:
                    logging.error("Skipping invalid JSON response: %s", json_response)
                    continue

            task_positions.append(idx)
            tasks.append(get_smiles_dict(parsed_response, session, fix_name_bool, semaphore))

        task_results = await asyncio.gather(*tasks)
        for idx, result in zip(task_positions, task_results):
            results[idx] = result
        return results


async def resolve_no_smi_name(compound_name, session, semaphore):
    """Resolve one previously missing name with aliases and source validation."""
    original_name = _clean_compound_name(compound_name)
    override = resolution_overrides.get(normalize_name_key(original_name))
    if override and override.get("smiles"):
        return override["smiles"], f"Curated:{override.get('kind', 'override')}"

    known_smiles = _known_smiles(original_name)
    if known_smiles:
        return known_smiles, "KnownAlias"
    if not override and should_skip_automatic_resolution(original_name):
        logging.info(
            "Skipping ambiguous or non-molecular name during automatic resolution: %s",
            original_name,
        )
        return None, None

    lookup_name = (
        _clean_compound_name(override["lookup_name"])
        if override and override.get("lookup_name")
        else resolve_name_alias(original_name)
    )
    for cache_key in (original_name, lookup_name):
        cached_smiles = canonicalize_smiles(smiles_cache.get(cache_key))
        if cached_smiles:
            source = f"Curated:{override.get('kind', 'alias')}+Cache" if override else "Cache"
            return cached_smiles, source

    results = await asyncio.gather(
        get_smiles_from_pubchem(session, lookup_name, semaphore, max_retries=3),
        get_smiles_from_opsin(session, lookup_name, semaphore, max_retries=2),
        get_smiles_from_cir(session, lookup_name, semaphore, max_retries=2),
        get_smiles_from_chebi(session, lookup_name, semaphore, max_retries=2),
        get_smiles_from_chemspider_async(lookup_name),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            logging.error("Error processing %s: %s", original_name, result)
            continue
        smiles, source = result
        smiles = canonicalize_smiles(smiles)
        if smiles:
            smiles_cache[original_name] = smiles
            smiles_cache[lookup_name] = smiles
            if override:
                source = f"Curated:{override.get('kind', 'alias')}+{source}"
            return smiles, source
    return None, None


async def process_no_smi_entry(code, compound_name, session, semaphore, smiles_dict, idx):
    smiles, source = await resolve_no_smi_name(compound_name, session, semaphore)
    if smiles:
        logging.info("[%s] Found SMILES for %s from %s: %s", idx, compound_name, source, smiles)
        smiles_dict[code] = smiles
        return

    logging.info("[%s] No SMILES found for %s", idx, compound_name)
    smiles_dict[code] = f"[{compound_name} (NoSmi)]"


async def process_batch_final(
    smiles_dict_list,
    session,
    semaphore,
    resolution_cache=None,
):
    if resolution_cache is None:
        resolution_cache = {}

    targets = {}
    for idx, smiles_dict_item in enumerate(smiles_dict_list):
        no_smi_entries = {
            key: value for key, value in smiles_dict_item.items() if "(NoSmi)" in value
        }
        for code, compound_name_with_no_smi in no_smi_entries.items():
            compound_name = compound_name_with_no_smi.replace("(NoSmi)", "").strip("[] ")
            cache_key = normalize_name_key(compound_name)
            targets.setdefault(cache_key, {"name": compound_name, "entries": []})[
                "entries"
            ].append((smiles_dict_item, code, idx, compound_name))

    pending_keys = [key for key in targets if key not in resolution_cache]
    pending_results = await asyncio.gather(
        *[
            resolve_no_smi_name(targets[key]["name"], session, semaphore)
            for key in pending_keys
        ]
    )
    resolution_cache.update(zip(pending_keys, pending_results))

    for cache_key, target in targets.items():
        smiles, source = resolution_cache[cache_key]
        for smiles_dict_item, code, idx, compound_name in target["entries"]:
            if smiles:
                logging.info(
                    "[%s] Found SMILES for %s from %s: %s",
                    idx,
                    compound_name,
                    source,
                    smiles,
                )
                smiles_dict_item[code] = smiles
            else:
                logging.info("[%s] No SMILES found for %s", idx, compound_name)
                smiles_dict_item[code] = f"[{compound_name} (NoSmi)]"


def calculate_no_smi_percentage(smiles_dict_list):
    total_entries = sum(len(smiles_dict_item) for smiles_dict_item in smiles_dict_list)
    no_smi_entries = sum(
        1
        for smiles_dict_item in smiles_dict_list
        for value in smiles_dict_item.values()
        if "(NoSmi)" in value
    )
    no_smi_percentage = (no_smi_entries / total_entries) * 100 if total_entries > 0 else 0
    logging.info("Found %s entries with NoSmi [%.2f%%].", no_smi_entries, no_smi_percentage)
    return no_smi_percentage


async def reprocess_no_smi(smiles_dict_file, output_file, session, semaphore, batch_size):
    with Path(smiles_dict_file).open("r", encoding="utf-8-sig") as handle:
        smiles_dict_list = json.load(handle)

    total_batches = (len(smiles_dict_list) + batch_size - 1) // batch_size
    resolution_cache = {}
    with tqdm(total=total_batches, desc="Processing Batches", unit="batch") as pbar:
        for idx in range(0, len(smiles_dict_list), batch_size):
            batch = smiles_dict_list[idx : idx + batch_size]
            await process_batch_final(
                batch,
                session,
                semaphore,
                resolution_cache=resolution_cache,
            )
            no_smi_percentage = calculate_no_smi_percentage(smiles_dict_list)
            batch_number = idx // batch_size + 1
            print(
                f"[BATCH {batch_number}] {round(no_smi_percentage, 2)}% of entries "
                "do not have a corresponding SMILES representation"
            )
            with Path(output_file).open("w", encoding="utf-8-sig") as handle:
                json.dump(smiles_dict_list, handle, ensure_ascii=False, indent=2)
            logging.info("Batch %s/%s processed.", batch_number, total_batches)
            pbar.update(1)

        logging.info("Processing completed")
