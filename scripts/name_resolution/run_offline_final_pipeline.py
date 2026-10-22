"""Resolve cached chemical names and atom-map reactions without web lookups.

This gives both providers the same deterministic post-processing condition:
validated shared cache + built-in aliases + curated overrides. Names absent
from those sources remain explicit ``NoSmi`` entries.

This is the post-processing used for the 13 API configurations in result/model_outputs (both gpt-4.1-mini
configurations, gpt-5.4, gpt-5.6-sol, gemini-2.5-flash, gemini-3.1-pro-preview
and gemini-3.6-flash). With result/name_resolution/shared_master_smiles_cache.csv,
config/nosmi_overrides.json and the alias table of the released commit it
regenerates every published ``<config>_smiles`` dictionary (400/400 rows for
each of the 13 configurations). Only the import bootstrap and CSV cache
support were changed for the release. Note that the alias fixes for "DMA" and
"Pd/C" (see README) change rows 136, 288, 296 and 386 of the Gemini 3.x and
gemini-2.5-flash configurations if the pipeline is rerun with current code.

Example::

    python scripts/name_resolution/run_offline_final_pipeline.py \
        --input <config>_output.csv --output-prefix <config> \
        --output result/<config>_reaction_smiles_final.csv \
        --with-smiles-output result/<config>_with_smiles.csv \
        --smiles-dict-output result/smiles_batches/<config>.json \
        --cache result/name_resolution/shared_master_smiles_cache.csv \
        --overrides config/nosmi_overrides.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# Release note: the research copy pointed at a sibling checkout; here the
# repository's own src/ is used.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from uspto_revisit.atom_mapping import add_atom_mapping_columns  # noqa: E402
from uspto_revisit.reaction_smiles import process_smiles_data  # noqa: E402
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-column", default="prediction")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--with-smiles-output", required=True)
    parser.add_argument("--smiles-dict-output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--overrides", required=True)
    parser.add_argument("--mapping-device", default="cpu")
    parser.add_argument("--mapping-model-version", default="202403")
    parser.add_argument("--mapping-batch-size", type=int, default=32)
    return parser


def cached_smiles(name: Any) -> str | None:
    original = sf._clean_compound_name(str(name))
    candidates: list[str] = []
    for candidate in (
        original,
        sf.resolve_name_alias(original),
        sf.fix_name(original),
        sf.resolve_name_alias(sf.fix_name(original)),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    override = sf.resolution_overrides.get(sf.normalize_name_key(original))
    if override and override.get("smiles"):
        value = sf.canonicalize_smiles(override["smiles"])
        if value:
            return value

    for candidate in candidates:
        known = sf._known_smiles(candidate)
        value = sf.canonicalize_smiles(known)
        if value:
            return value
        for cache_key in (candidate, sf.normalize_name_key(candidate)):
            value = sf.canonicalize_smiles(sf.smiles_cache.get(cache_key))
            if value:
                return value
    return None


def parse_prediction(value: Any) -> dict[str, Any] | None:
    text = str(value or "")
    parsed = sf.parse_json_object(text)
    if parsed is not None:
        return parsed
    fixed = sf.fix_json_string(text)
    return sf.parse_json_object(fixed) if fixed else None


def prediction_to_smiles_dict(value: Any) -> dict[str, str]:
    parsed = parse_prediction(value)
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, str] = {}
    for section_name in (
        "Reactants, Solvents, Catalysts",
        "Reactant, Solvents, Catalysts",
        "Reactants",
        "Products",
        "Product",
    ):
        section = parsed.get(section_name)
        if not isinstance(section, dict):
            continue
        for code, name in section.items():
            resolved = cached_smiles(name)
            result[str(code)] = (
                resolved if resolved else f"[{str(name).strip()} (NoSmi)]"
            )
    return result


def main() -> int:
    args = build_parser().parse_args()
    sf.smiles_cache = load_cache_file(args.cache)
    sf.load_resolution_overrides(args.overrides)

    frame = pd.read_csv(args.input)
    if args.model_column not in frame:
        raise ValueError(f"Missing column: {args.model_column}")
    predictions = frame[args.model_column].fillna("").astype(str).tolist()
    smiles_dicts = [prediction_to_smiles_dict(value) for value in predictions]

    smiles_dict_path = Path(args.smiles_dict_output)
    smiles_dict_path.parent.mkdir(parents=True, exist_ok=True)
    smiles_dict_path.write_text(
        json.dumps(smiles_dicts, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )

    prefix = args.output_prefix
    frame[f"{prefix}_smiles"] = [
        json.dumps(value, ensure_ascii=False) for value in smiles_dicts
    ]
    skeletons, reactions, _errors = process_smiles_data(
        predictions,
        smiles_dicts,
    )
    frame[f"{prefix}_skeleton"] = skeletons
    frame[f"{prefix}_rxn"] = reactions
    with_smiles_path = Path(args.with_smiles_output)
    with_smiles_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(with_smiles_path, index=False, encoding="utf-8-sig")

    output_frame = frame.drop(
        columns=[column for column in ("idx", "model") if column in frame],
    )
    output_frame = add_atom_mapping_columns(
        output_frame,
        f"{prefix}_rxn",
        prefix,
        device=args.mapping_device,
        model_version=args.mapping_model_version,
        batch_size=args.mapping_batch_size,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Pipeline completed: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
