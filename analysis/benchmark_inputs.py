"""Read the unified benchmark input and validate record IDs.

A separate membership path is supported for importing older split inputs.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
INPUT = DATA / "input.csv"
STRATUM_ISSUE_LABELS = {
    "many_products": "many_product (> 1)",
    "many_reactants": "many_reactant (> 2)",
    "many_reagents": "many_reagent (> 5)",
    "no_reagent": "no_reagent",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_benchmark(input_path: Path = INPUT, membership_path: Path | None = None) -> list[dict[str, str]]:
    """Return input-ordered records, validating IDs and shared fields."""
    input_path = Path(input_path)
    inputs = _read(input_path)
    if membership_path is None:
        if any("idx" not in row for row in inputs):
            raise ValueError("The unified input requires an explicit idx column")
        membership = inputs
    else:
        membership = _read(Path(membership_path))
    by_id = {}
    for row in membership:
        idx = int(row["idx"])
        if idx in by_id:
            raise ValueError(f"Duplicate membership ID: {idx}")
        by_id[idx] = {**row, "idx": str(idx)}
    input_ids = [int(row.get("idx", str(i))) for i, row in enumerate(inputs)]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("Duplicate input IDs")
    if set(input_ids) != set(by_id):
        raise ValueError("Input and membership IDs differ")
    joined = []
    for idx, row in zip(input_ids, inputs):
        record = {**row, "idx": str(idx)}
        source = by_id[idx]
        for field in record.keys() & source.keys():
            if record[field] != source[field]:
                raise ValueError(f"Conflicting {field} for benchmark ID {idx}")
        joined.append({**record, **source})
    return joined


def load_benchmark_frame(input_path: Path = INPUT, membership_path: Path | None = None):
    """Pandas view indexed by integer benchmark ID, for the comparison scorers."""
    import pandas as pd

    frame = pd.DataFrame(load_benchmark(input_path, membership_path)).set_index("idx")
    frame.index = frame.index.astype(int)
    return frame


def input_hashes(input_path: Path = INPUT, membership_path: Path | None = None) -> dict[str, str]:
    """Record input-file hashes in manifests of new executions."""
    paths = [input_path] + ([membership_path] if membership_path is not None else [])
    return {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
