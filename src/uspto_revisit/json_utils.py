"""Helpers for validating and normalizing model JSON output."""

from __future__ import annotations

import json
import logging
import re


_INVALID_JSON_ESCAPE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')


def _repair_invalid_json_escapes(value: str) -> str:
    """Drop backslashes that do not begin a JSON escape sequence."""
    return _INVALID_JSON_ESCAPE.sub("", value)


def parse_json_object(value, max_depth: int = 3) -> dict | None:
    """Decode a JSON object, including JSON strings wrapped in JSON strings."""
    current = value
    for _ in range(max_depth):
        if isinstance(current, dict):
            return current
        if not isinstance(current, str):
            return None
        stripped = current.strip()
        try:
            current = json.loads(stripped)
        except (TypeError, ValueError):
            repaired = _repair_invalid_json_escapes(stripped)
            if repaired == stripped:
                return None
            try:
                current = json.loads(repaired)
            except (TypeError, ValueError):
                return None
    return current if isinstance(current, dict) else None


def is_valid_json(json_string: str) -> bool:
    parsed = parse_json_object(json_string)
    if parsed is not None:
        return True
    logging.error("Invalid JSON object: %s...", str(json_string)[:100])
    return False


def fix_json_string(json_string: str) -> str | None:
    try:
        fixed = json_string.strip()
        if not fixed.startswith("{"):
            fixed = "{" + fixed
        if not fixed.endswith("}"):
            fixed = fixed + "}"

        open_braces = fixed.count("{")
        close_braces = fixed.count("}")
        if open_braces > close_braces:
            fixed += "}" * (open_braces - close_braces)
        elif close_braces > open_braces:
            fixed = "{" * (close_braces - open_braces) + fixed
        return fixed
    except Exception as exc:
        logging.error("Failed to fix JSON string: %s [Error] %s", json_string, exc)
        return None


def fix_name(compound_name: str) -> str:
    remove_patterns = [
        r"\d+\s+normal",
        r"\d+(\.\d+)?\s*N-",
        r"\d+(\.\d+)?\s*N",
        r"\d+(\.\d+)?\s*M",
        r"\d+%",
        r"\s*\(\s*\)",
        r"\([^()]*\)$",
        r"·",
        r"\([IVXLCDM]+\)",
        "anhydrous",
        "concentrated",
        "catalyst",
        "-catalyst",
        "saturated",
        "ice",
        "ice-",
        "dried",
        "aqueous",
        "solution",
        "normal",
        "solid",
        "complex",
        "resin",
        "adduct",
        "corresponding",
        "atmosphere",
        "gas",
        "solvent",
        "crystal",
        "crystals",
        "buffer",
        ".conc",
        "fuming",
        "glacial",
    ]
    pattern = f"({'|'.join(remove_patterns)})"
    return re.sub(pattern, "", compound_name, flags=re.I).strip()
