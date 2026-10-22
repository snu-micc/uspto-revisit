"""Build a shared audited name→SMILES cache from prior evaluated model runs.

This script produced the shared cache used by run_offline_final_pipeline.py and its audit,
result/name_resolution/shared_master_smiles_cache_audit.csv (1,890 normalized
names, 20 with more than one candidate structure). For each normalized name the
most frequent structure across the earlier model runs is selected; ties are
broken by the lexicographically smallest SMILES. The earlier runs and the base
pickle cache are not distributed; the resulting cache is released as
result/name_resolution/shared_master_smiles_cache.csv.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


# Release note: the research copy pointed at a sibling checkout; here the
# repository's own src/ is used.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from uspto_revisit import smiles_fetch as sf  # noqa: E402


def load_cache_file(path: str | Path) -> dict[str, str]:
    """Read a name->SMILES cache from a released CSV (key,smiles) or a pickle."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        import csv

        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {row["key"]: row["smiles"] for row in csv.DictReader(handle)}
    import pickle

    with path.open("rb") as handle:
        return pickle.load(handle)


DEFAULT_MODELS = (
    "gpt-4.1-mini",
    "gpt_4.1_mini_ft",
    "gpt-5.4",
    "gpt-5.6-sol",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash-lite",
)


def parse_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_prediction(value: Any) -> dict[str, Any]:
    parsed = sf.parse_json_object(str(value or ""))
    if parsed is not None:
        return parsed
    fixed = sf.fix_json_string(str(value or ""))
    parsed = sf.parse_json_object(fixed) if fixed else None
    return parsed if isinstance(parsed, dict) else {}


def entity_names(prediction: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in (
        "Reactants, Solvents, Catalysts",
        "Reactant, Solvents, Catalysts",
        "Reactants",
        "Products",
        "Product",
    ):
        section = prediction.get(key)
        if isinstance(section, dict):
            result.update({str(code): str(name) for code, name in section.items()})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    args = parser.parse_args()

    base: dict[str, str] = load_cache_file(args.base_cache)

    candidates: dict[str, Counter[str]] = defaultdict(Counter)
    observed_names: dict[str, set[str]] = defaultdict(set)
    for model in args.models:
        path = Path(args.results_dir) / f"{model}_reaction_smiles_final.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        smiles_column = f"{model}_smiles"
        for _, record in frame.iterrows():
            raw_prediction = record.get("prediction")
            if not str(raw_prediction or "").strip():
                raw_prediction = record.get("model_output")
            names = entity_names(parse_prediction(raw_prediction))
            smiles_by_code = parse_dict(record.get(smiles_column))
            for code, name in names.items():
                smiles = smiles_by_code.get(code)
                if "nosmi" in str(smiles).lower():
                    continue
                canonical = sf.canonicalize_smiles(smiles)
                if not canonical:
                    continue
                normalized = sf.normalize_name_key(sf._clean_compound_name(name))
                if normalized:
                    candidates[normalized][canonical] += 1
                    observed_names[normalized].add(name)

    merged = dict(base)
    audit_rows: list[dict[str, Any]] = []
    for normalized, counts in candidates.items():
        selected, selected_count = sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
        merged[normalized] = selected
        for name in observed_names[normalized]:
            merged[name] = selected
            merged[sf._clean_compound_name(name)] = selected
        audit_rows.append(
            {
                "normalized_name": normalized,
                "selected_smiles": selected,
                "selected_count": selected_count,
                "candidate_count": len(counts),
                "all_candidates": json.dumps(dict(counts), sort_keys=True),
                "observed_names": json.dumps(sorted(observed_names[normalized])),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        pd.DataFrame(
            sorted(merged.items()), columns=["key", "smiles"]
        ).to_csv(output, index=False, encoding="utf-8-sig")
    else:
        with output.open("wb") as handle:
            pickle.dump(merged, handle)
    audit_frame = pd.DataFrame(audit_rows)
    if not audit_frame.empty:
        audit_frame = audit_frame.sort_values("normalized_name")
    audit_frame.to_csv(args.audit, index=False, encoding="utf-8-sig")
    conflicts = sum(len(counts) > 1 for counts in candidates.values())
    print(
        f"Wrote {len(merged)} cache keys from {len(candidates)} normalized names; "
        f"{conflicts} names had multiple candidate structures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
