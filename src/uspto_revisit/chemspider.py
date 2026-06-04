"""ChemSpider lookup helpers."""

from __future__ import annotations

import asyncio
import logging
import os

from chemspipy import ChemSpider
from requests.exceptions import RequestException

REQUEST_LIMIT = 980

_current_key_index = 0
_request_count = 0


def _load_api_keys() -> list[str]:
    keys = os.getenv("CHEMSPIDER_API_KEY") or os.getenv("CHEMSPIDER_API_KEYS", "")
    return [key.strip() for key in keys.split(",") if key.strip()]


def get_current_api_key(api_keys: list[str] | None = None) -> str:
    """Return the active ChemSpider API key, rotating after REQUEST_LIMIT calls."""
    global _current_key_index, _request_count

    keys = api_keys or _load_api_keys()
    if not keys:
        raise RuntimeError("Set CHEMSPIDER_API_KEY before using ChemSpider lookups.")

    if _request_count >= REQUEST_LIMIT:
        _current_key_index = (_current_key_index + 1) % len(keys)
        _request_count = 0

    return keys[_current_key_index]


def get_smiles_from_chemspider(compound_name: str) -> tuple[str | None, str | None]:
    """Fetch a SMILES string for a compound name from ChemSpider."""
    global _request_count

    try:
        api_key = get_current_api_key()
    except RuntimeError:
        logging.info("[ChemSpider] CHEMSPIDER_API_KEY is not set; skipping %s", compound_name)
        return None, None

    chemspider = ChemSpider(api_key)

    try:
        results = chemspider.search(compound_name)
        _request_count += 1
        if results:
            return results[0].smiles, "ChemSpider"
        logging.error("[ChemSpider] No results found for %s", compound_name)
    except RequestException as exc:
        logging.error("[ChemSpider] Network error for %s: %s", compound_name, exc)
    except KeyError as exc:
        logging.error("[ChemSpider] KeyError for %s: %s", compound_name, exc)
    except Exception as exc:
        logging.error("[ChemSpider] Unexpected error for %s: %s", compound_name, exc)

    return None, None


async def get_smiles_from_chemspider_async(
    compound_name: str,
) -> tuple[str | None, str | None]:
    """Async wrapper for the blocking ChemSpider client."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_smiles_from_chemspider, compound_name)
