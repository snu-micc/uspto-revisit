"""Helpers for validating and normalizing model JSON output."""

from __future__ import annotations

import json
import logging
import re


def is_valid_json(json_string: str) -> bool:
    try:
        json.loads(json_string)
        return True
    except ValueError as exc:
        logging.error("Invalid JSON string: %s... [Error] %s", json_string[:100], exc)
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
