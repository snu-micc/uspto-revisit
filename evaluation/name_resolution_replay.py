"""Replay the 2026-10-16 resolver re-query offline ("fully automatic" names).

On 2026-10-16 every distinct extracted spelling of a name that a curated layer
resolved (the 46 overrides, the row-level ``use_smiles`` decisions and the
reviewed SMILES reused by name in other rows) was re-queried live against
PubChem, OPSIN, NCI CIR and ChEBI. Every HTTP exchange is stored in
``result/name_resolution/requery_2026-10-16_raw.jsonl`` (see the README in that
folder). This module answers the same questions from that recording, without
network access, using the released resolver code in ``uspto_revisit.smiles_fetch``.

Two resolution orders are available:

``two-pass`` (the released pipeline's order; default)
    First pass, as ``smiles_fetch.get_smiles_dict`` runs it with
    ``--fix-names``: the abbreviation table, then OPSIN, for the name and, if
    that fails, for ``fix_name(name)``. Names still unresolved go to the NoSmi
    retry below.
``retry``
    The NoSmi retry path alone (``resolve_no_smi_name``): abbreviation table,
    refusal of ambiguous or non-molecular names, identity-preserving alias,
    then PubChem, OPSIN, CIR and ChEBI concurrently; the first non-empty
    result in that priority order is kept.

Curated overrides, row decisions and the shared cache are disabled during the
replay. For each URL the last recorded attempt is used, as in the live run; a
URL without a recorded HTTP 200 response yields nothing. Requests that the
recording does not contain are listed in ``missing_urls``.

``lookup_tables=False`` also disables the two lookup tables of the resolver:
the table of 19 abbreviations and trivial names
(``smiles_fetch.DEFAULT_KNOWN_SMILES``) and the 13 spelling variants and
synonyms looked up under a standard name (``SAFE_NAME_ALIASES``). Names they
would answer or rewrite then go to the resolvers as written;
``requery_2026-10-23_no_tables_raw.jsonl`` records those names with both
tables disabled.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

DEFAULT_REQUERY_LOG = Path("result/name_resolution/requery_2026-10-16_raw.jsonl")
RESOLUTION_ORDERS = ("two-pass", "retry")
RETRY_RESOLVERS = ("PubChem", "OPSIN", "CIR", "ChEBI")


def capture_path(log_path: str | Path) -> Path:
    """The capture-metadata file that belongs to a recording."""
    log_path = Path(log_path)
    return log_path.with_name(log_path.name.replace("_raw.jsonl", "_capture.json"))


def load_requery_log(path: str | Path, *, verify: bool = True) -> dict[str, dict[str, Any]]:
    """Return the last recorded exchange for every URL.

    When the capture file exists next to the recording, the recording's
    SHA-256 must match the one stored there.
    """
    path = Path(path)
    if verify and capture_path(path).is_file():
        expected = json.loads(capture_path(path).read_text(encoding="utf-8")).get("raw_sha256")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected and expected != actual:
            raise ValueError(f"{path} does not match the SHA-256 recorded in {capture_path(path)}.")
    responses: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                responses[record["url"]] = record
    return responses


class RecordedResolver:
    """Resolve names with the released resolver code and recorded responses."""

    def __init__(self, log_path: str | Path = DEFAULT_REQUERY_LOG, *, lookup_tables: bool = True):
        self.responses = load_requery_log(log_path)
        self.missing_urls: set[str] = set()
        self.requested: list[str] = []
        self.lookup_tables = lookup_tables

    async def _fetch(self, session, url, semaphore, max_retries):
        self.requested.append(url)
        record = self.responses.get(url)
        if record is None:
            self.missing_urls.add(url)
            return None
        return record.get("body") if record.get("status") == 200 else None

    @contextlib.contextmanager
    def _patched_resolver(self):
        from uspto_revisit import smiles_fetch as sf

        saved = (
            sf.fetch_smiles,
            sf.resolution_overrides,
            sf.smiles_cache,
            sf.DEFAULT_KNOWN_SMILES,
            sf.SAFE_NAME_ALIASES,
        )
        sf.fetch_smiles = self._fetch
        sf.resolution_overrides = {}
        sf.smiles_cache = {}
        if not self.lookup_tables:
            sf.DEFAULT_KNOWN_SMILES = {}
            sf.SAFE_NAME_ALIASES = {}
        previous_disable = logging.root.manager.disable
        logging.disable(logging.WARNING)  # the resolvers log every miss
        try:
            yield sf
        finally:
            logging.disable(previous_disable)
            (
                sf.fetch_smiles,
                sf.resolution_overrides,
                sf.smiles_cache,
                sf.DEFAULT_KNOWN_SMILES,
                sf.SAFE_NAME_ALIASES,
            ) = saved

    @staticmethod
    async def _retry_path(sf, name: str) -> dict[str, Any]:
        """``resolve_no_smi_name`` without overrides or cache."""
        original = sf._clean_compound_name(name)
        known = sf._known_smiles(original)
        if known:
            return {"retry_smiles": known, "retry_source": "KnownAlias", "refused": 0,
                    "resolver_agreement": "known_alias", "resolver_candidates": ""}
        if sf.should_skip_automatic_resolution(original):
            return {"retry_smiles": "", "retry_source": "", "refused": 1,
                    "resolver_agreement": "refused", "resolver_candidates": ""}
        lookup = sf.resolve_name_alias(original)
        results = await asyncio.gather(
            sf.get_smiles_from_pubchem(None, lookup, None, max_retries=3),
            sf.get_smiles_from_opsin(None, lookup, None, max_retries=2),
            sf.get_smiles_from_cir(None, lookup, None, max_retries=2),
            sf.get_smiles_from_chebi(None, lookup, None, max_retries=2),
            return_exceptions=True,
        )
        candidates = []
        for resolver, result in zip(RETRY_RESOLVERS, results):
            if isinstance(result, Exception):
                continue
            smiles = sf.canonicalize_smiles(result[0])
            if smiles:
                candidates.append((resolver, smiles))
        comparison = sf.compare_resolver_results(candidates)
        selected = candidates[0] if candidates else ("", "")
        return {"retry_smiles": selected[1], "retry_source": selected[0], "refused": 0,
                "resolver_agreement": comparison["resolver_agreement"],
                "resolver_candidates": comparison["resolver_candidates"]}

    @staticmethod
    async def _first_pass(sf, name: str) -> tuple[str, str, str]:
        """``get_smiles`` with fix_name_bool=True, then once more on ``fix_name(name)``."""
        from uspto_revisit.json_utils import fix_name

        sf.smiles_cache.clear()
        smiles, source = await sf.get_smiles(None, name, True, None)
        stage = "first_pass"
        if not smiles:
            smiles, source = await sf.get_smiles(None, fix_name(name), True, None)
            stage = "first_pass_fixed_name"
        canonical = sf.canonicalize_smiles(smiles) if smiles else None
        return (canonical or "", source or "", stage)

    def resolve(self, names, order: str = "two-pass") -> dict[str, dict[str, Any]]:
        """Resolve each cleaned spelling; returns spelling -> resolution record."""
        if order not in RESOLUTION_ORDERS:
            raise ValueError(f"order must be one of {RESOLUTION_ORDERS}.")

        with self._patched_resolver() as sf:
            return asyncio.run(self._resolve_all(sf, list(names), order))

    async def _resolve_all(self, sf, names: list[str], order: str) -> dict[str, dict[str, Any]]:
        """Resolve names one at a time (subclasses may open a session here)."""
        return await self._resolve_names(sf, names, order)

    async def _resolve_names(self, sf, names: list[str], order: str) -> dict[str, dict[str, Any]]:
        out = {}
        for name in names:
            self.requested.clear()
            # The retry path is always evaluated so its resolver comparison is recorded.
            retry = await self._retry_path(sf, name)
            first_smiles, first_source, stage = ("", "", "")
            if order == "two-pass":
                first_smiles, first_source, stage = await self._first_pass(sf, name)
            if first_smiles:
                selected, source, how = first_smiles, first_source, stage
            elif retry["retry_smiles"]:
                selected, source, how = retry["retry_smiles"], retry["retry_source"], "retry_path"
            else:
                selected, source, how = "", "", "unresolved"
            out[name] = {
                "name": name,
                "first_pass_smiles": first_smiles,
                **retry,
                "selected_smiles": selected,
                "selected_source": source,
                "resolved_by": how,
            }
        return out


def clean_name(name: str) -> str:
    """Same as ``uspto_revisit.smiles_fetch._clean_compound_name``."""
    return " ".join(str(name).replace("′", "'").split())


def builtin_table_smiles(name: str) -> str | None:
    """The SMILES the first pass (``get_smiles_dict`` with ``--fix-names``) takes from the built-in table, if any."""
    from uspto_revisit import smiles_fetch as sf
    from uspto_revisit.json_utils import fix_name

    for start in (clean_name(name), clean_name(fix_name(name))):
        for candidate in (start, sf.resolve_name_alias(fix_name(sf.resolve_name_alias(start)))):
            smiles = sf._known_smiles(candidate)
            if smiles:
                return smiles
    return None


def alias_used(name: str) -> bool:
    """True when the first pass or the retry path looks the name up under a listed synonym (``SAFE_NAME_ALIASES``)."""
    from uspto_revisit import smiles_fetch as sf
    from uspto_revisit.json_utils import fix_name

    return any(sf._clean_compound_name(x).casefold() in sf.SAFE_NAME_ALIASES for x in (name, fix_name(name)))
