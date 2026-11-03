"""Small file IO helpers."""

from __future__ import annotations

import json
from pathlib import Path


def ensure_directory(directory: str | Path) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_smiles_dict(smiles_dict: dict, filename: str | Path) -> None:
    with Path(filename).open("w", encoding="utf-8-sig") as handle:
        json.dump(smiles_dict, handle, ensure_ascii=False, indent=2)
