"""Asynchronous SMILES lookup utilities."""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import random
from pathlib import Path
from urllib.parse import quote

import aiohttp
import nest_asyncio
from tqdm import tqdm

from uspto_revisit.chemspider import get_smiles_from_chemspider_async
from uspto_revisit.json_utils import fix_json_string, fix_name, is_valid_json

nest_asyncio.apply()

DEFAULT_KNOWN_SMILES = {
    "ice": "O",
    "DCM": "C(Cl)Cl",
    "DMA": "CC(=O)N",
    "ether": "CCOCC",
    "brine": "O.[Na+].[Cl-]",
    "Pd/C": "Pd",
    "DMSO": "CS(=O)C",
    "LiAlH4": "[Li+].[AlH4-]",
}

smiles_cache = {}


def load_cache(cache_path: str | Path = "smiles_cache.pkl") -> None:
    global smiles_cache
    path = Path(cache_path)
    if path.exists():
        with path.open("rb") as handle:
            smiles_cache = pickle.load(handle)


def save_cache(cache_path: str | Path = "smiles_cache.pkl") -> None:
    with Path(cache_path).open("wb") as handle:
        pickle.dump(smiles_cache, handle)


async def exponential_backoff(attempt: int, max_delay: int = 60) -> None:
    delay = min(max_delay, (2**attempt) + random.uniform(0, 1))
    await asyncio.sleep(delay)


async def fetch_smiles(session, url: str, semaphore, max_retries: int) -> str | None:
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
        f"{quote(compound_name)}/property/IsomericSMILES/JSON"
    )
    try:
        response_text = await fetch_smiles(session, url, semaphore, max_retries)
        if response_text:
            data = json.loads(response_text)
            properties = data.get("PropertyTable", {}).get("Properties", [])
            if properties:
                return properties[0]["IsomericSMILES"], "PubChem"
    except Exception as exc:
        logging.error("[PubChem] Failed for %s from URL: %s [Error] %s", compound_name, url, exc)
    return None, None


async def get_smiles_from_cir(session, compound_name, semaphore, max_retries):
    url = f"http://cactus.nci.nih.gov/chemical/structure/{quote(compound_name)}/smiles"
    try:
        return await fetch_smiles(session, url, semaphore, max_retries), "CIR"
    except Exception as exc:
        logging.error("[CIR] Failed for %s from URL: %s [Error] %s", compound_name, url, exc)
    return None, None


async def get_smiles_from_opsin(session, compound_name, semaphore, max_retries):
    url = f"https://opsin.ch.cam.ac.uk/opsin/{quote(compound_name)}.smi"
    try:
        return await fetch_smiles(session, url, semaphore, max_retries), "OPSIN"
    except Exception as exc:
        logging.error("[OPSIN] Failed for %s from URL: %s [Error] %s", compound_name, url, exc)
    return None, None


async def get_smiles(session, compound_name, fix_name_bool, semaphore):
    if not isinstance(compound_name, str):
        logging.warning(
            "compound_name is not a string: %s (type: %s)",
            compound_name,
            type(compound_name),
        )
        compound_name = str(compound_name)

    compound_name = compound_name.replace("′", "'")
    if fix_name_bool:
        if compound_name in DEFAULT_KNOWN_SMILES:
            return DEFAULT_KNOWN_SMILES[compound_name], "Cache"
        compound_name = fix_name(compound_name)

    if compound_name in smiles_cache:
        return smiles_cache[compound_name], "Cache"

    results = await asyncio.gather(
        get_smiles_from_opsin(session, compound_name, semaphore, max_retries=2)
    )
    for result, source in results:
        if result:
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
    async with aiohttp.ClientSession() as session:
        tasks = []
        task_positions = []
        results = [{} for _ in json_responses]
        for idx, json_response in enumerate(json_responses):
            if not is_valid_json(json_response):
                fixed_json = fix_json_string(json_response)
                if fixed_json and is_valid_json(fixed_json):
                    json_response = fixed_json
                else:
                    logging.error("Skipping invalid JSON response: %s", json_response)
                    continue

            task_positions.append(idx)
            tasks.append(get_smiles_dict(json.loads(json_response), session, fix_name_bool, semaphore))

        task_results = await asyncio.gather(*tasks)
        for idx, result in zip(task_positions, task_results):
            results[idx] = result
        return results


async def process_no_smi_entry(code, compound_name, session, semaphore, smiles_dict, idx):
    tasks = [
        get_smiles_from_cir(session, compound_name, semaphore, max_retries=5),
        get_smiles_from_chemspider_async(compound_name),
        get_smiles_from_pubchem(session, compound_name, semaphore, max_retries=5),
    ]

    for task in asyncio.as_completed(tasks):
        try:
            smiles, source = await task
            if smiles:
                logging.info("[%s] Found SMILES for %s from %s: %s", idx, compound_name, source, smiles)
                smiles_dict[code] = smiles
                return
        except Exception as exc:
            logging.error("[%s] Error processing %s: %s", idx, compound_name, exc)

    logging.info("[%s] No SMILES found for %s", idx, compound_name)
    smiles_dict[code] = f"[{compound_name} (NoSmi)]"


async def process_batch_final(smiles_dict_list, session, semaphore):
    tasks = []
    for idx, smiles_dict_item in enumerate(smiles_dict_list):
        no_smi_entries = {
            key: value for key, value in smiles_dict_item.items() if "(NoSmi)" in value
        }
        for code, compound_name_with_no_smi in no_smi_entries.items():
            compound_name = compound_name_with_no_smi.replace("(NoSmi)", "").strip("[] ")
            tasks.append(
                process_no_smi_entry(code, compound_name, session, semaphore, smiles_dict_item, idx)
            )
    await asyncio.gather(*tasks)


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
    with tqdm(total=total_batches, desc="Processing Batches", unit="batch") as pbar:
        for idx in range(0, len(smiles_dict_list), batch_size):
            batch = smiles_dict_list[idx : idx + batch_size]
            await process_batch_final(batch, session, semaphore)
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
