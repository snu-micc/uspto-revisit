#!/usr/bin/env python3
"""Recover the union of NoSmi names across multiple model outputs once."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

import aiohttp
import pandas as pd

from uspto_revisit.atom_mapping import add_atom_mapping_columns, create_localmapper
from uspto_revisit.reaction_smiles import _coerce_mapping, process_smiles_data
from uspto_revisit.smiles_fetch import (
    load_cache,
    load_resolution_overrides,
    normalize_name_key,
    process_batch_final,
    save_cache,
    should_skip_automatic_resolution,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine NoSmi names from multiple model outputs, resolve every unique "
            "name once, rebuild reaction SMILES, and optionally rerun LocalMapper."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("PREFIX", "CSV", "SMILES_DICT"),
        required=True,
        help="Repeat for each model: output prefix, current CSV, and current JSON dictionary.",
    )
    parser.add_argument(
        "--overrides",
        default="config/nosmi_overrides.json",
        help="Auditable NoSmi override JSON. Default: config/nosmi_overrides.json",
    )
    parser.add_argument("--cache", default=None, help="Shared successful-resolution cache.")
    parser.add_argument(
        "--output-dir",
        default="result",
        help="Directory for web-recovered CSV and JSON outputs. Default: result",
    )
    parser.add_argument(
        "--audit",
        default="result/nosmi_recovery_audit.csv",
        help="Combined before/after audit CSV.",
    )
    parser.add_argument(
        "--summary",
        default="result/nosmi_recovery_summary.json",
        help="Machine-readable recovery summary JSON.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--map-atoms", action="store_true")
    parser.add_argument("--mapping-device", default="cpu")
    parser.add_argument("--mapping-model-version", default="202403")
    parser.add_argument("--mapping-batch-size", type=int, default=32)
    return parser.parse_args()


def output_paths(output_dir: Path, prefix: str) -> tuple[Path, Path]:
    return (
        output_dir / f"{prefix}_reaction_smiles_final.csv",
        output_dir / "smiles_batches" / prefix / "smiles_dict_final.json",
    )


def parse_reaction_list(value) -> list[str]:
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return []


def collect_reaction_no_smi(
    frame: pd.DataFrame,
    prefix: str,
) -> dict[str, dict]:
    records: dict[str, dict] = defaultdict(
        lambda: {
            "names": Counter(),
            "occurrences": 0,
            "sides": Counter(),
            "rows": set(),
            "models": Counter(),
        }
    )
    skeleton_column = f"{prefix}_skeleton"
    smiles_column = f"{prefix}_smiles"
    for row_index, row in frame.iterrows():
        try:
            smiles_dict = _coerce_mapping(row[smiles_column], row_index, smiles_column)
        except ValueError:
            smiles_dict = {}
        for reaction in parse_reaction_list(row.get(skeleton_column, "")):
            parts = reaction.split(">")
            for side_index, part in enumerate(parts):
                if side_index == 0:
                    side = "left"
                elif side_index == len(parts) - 1:
                    side = "product"
                else:
                    side = "middle"
                for code in part.split("."):
                    value = str(smiles_dict.get(code, ""))
                    suffix = " (NoSmi)]"
                    if not (value.startswith("[") and value.endswith(suffix)):
                        continue
                    name = " ".join(value[1 : -len(suffix)].split())
                    key = normalize_name_key(name)
                    record = records[key]
                    record["names"][name] += 1
                    record["occurrences"] += 1
                    record["sides"][side] += 1
                    record["rows"].add(row_index)
                    record["models"][prefix] += 1
    return records


def count_reaction_no_smi(frame: pd.DataFrame, prefix: str) -> int:
    """Count every unresolved code referenced by a reaction skeleton."""
    return sum(
        record["occurrences"]
        for record in collect_reaction_no_smi(frame, prefix).values()
    )


def merge_no_smi_records(target: dict[str, dict], source: dict[str, dict]) -> None:
    for key, source_record in source.items():
        target_record = target.setdefault(
            key,
            {
                "names": Counter(),
                "occurrences": 0,
                "sides": Counter(),
                "rows": set(),
                "models": Counter(),
            },
        )
        target_record["names"].update(source_record["names"])
        target_record["occurrences"] += source_record["occurrences"]
        target_record["sides"].update(source_record["sides"])
        target_record["rows"].update(source_record["rows"])
        target_record["models"].update(source_record["models"])


def resolved_smiles_by_name(
    frame: pd.DataFrame,
    smiles_dicts: list[dict],
) -> dict[str, Counter]:
    values: dict[str, Counter] = defaultdict(Counter)
    for index, (prediction, smiles_dict) in enumerate(
        zip(frame["prediction"].fillna(""), smiles_dicts)
    ):
        try:
            parsed = _coerce_mapping(prediction, index, "prediction")
        except ValueError:
            continue
        names = {}
        for section in ("Reactants, Solvents, Catalysts", "Product", "Products"):
            if isinstance(parsed.get(section), dict):
                names.update(parsed[section])
        for code, smiles in smiles_dict.items():
            name = names.get(code)
            if not name or "(NoSmi)" in str(smiles):
                continue
            values[normalize_name_key(name)][str(smiles)] += 1
    return values


def mapping_counts(frame: pd.DataFrame, prefix: str) -> dict[str, int]:
    mapped_column = f"{prefix}_mapped_rxn"
    if mapped_column not in frame.columns:
        return {"mapped_steps": 0, "total_steps": 0}
    mapped_steps = 0
    total_steps = 0
    for rxn_value, mapped_value in zip(
        frame[f"{prefix}_rxn"].fillna(""),
        frame[mapped_column].fillna(""),
    ):
        reactions = parse_reaction_list(rxn_value)
        mapped = parse_reaction_list(mapped_value)
        total_steps += len(reactions)
        mapped_steps += sum(value not in {"None", "", "nan"} for value in mapped)
    return {"mapped_steps": mapped_steps, "total_steps": total_steps}


async def run(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)

    specs = []
    all_dicts = []
    union_before: dict[str, dict] = {}
    for prefix, csv_path, dict_path in args.model:
        frame = pd.read_csv(csv_path)
        with Path(dict_path).open("r", encoding="utf-8-sig") as handle:
            smiles_dicts = json.load(handle)
        if len(frame) != len(smiles_dicts):
            raise ValueError(
                f"{prefix}: CSV rows ({len(frame)}) and dictionaries "
                f"({len(smiles_dicts)}) differ."
            )
        start = len(all_dicts)
        all_dicts.extend(smiles_dicts)
        stop = len(all_dicts)
        specs.append((prefix, Path(csv_path), Path(dict_path), frame, start, stop))
        merge_no_smi_records(union_before, collect_reaction_no_smi(frame, prefix))

    if args.cache:
        load_cache(args.cache)
    loaded_overrides = load_resolution_overrides(args.overrides)

    resolution_cache = {}
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=60)
    headers = {"User-Agent": "uspto-revisit/0.1 NoSmi recovery"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        await process_batch_final(
            all_dicts,
            session,
            semaphore,
            resolution_cache=resolution_cache,
        )
    if args.cache:
        save_cache(args.cache)

    mapper = None
    if args.map_atoms:
        mapper = create_localmapper(
            device=args.mapping_device,
            model_version=args.mapping_model_version,
        )

    union_after: dict[str, dict] = {}
    resolved_values: dict[str, Counter] = defaultdict(Counter)
    outputs = {}
    model_summaries = {}
    for prefix, _csv_path, _dict_path, frame, start, stop in specs:
        recovered_dicts = all_dicts[start:stop]
        csv_output, dict_output = output_paths(output_dir, prefix)
        dict_output.parent.mkdir(parents=True, exist_ok=True)
        with dict_output.open("w", encoding="utf-8-sig") as handle:
            json.dump(recovered_dicts, handle, ensure_ascii=False, indent=2)

        result = frame.copy()
        result[f"{prefix}_smiles"] = recovered_dicts
        skeleton, reactions, _errors = process_smiles_data(
            result["prediction"].fillna("").astype(str).tolist(),
            recovered_dicts,
        )
        result[f"{prefix}_skeleton"] = skeleton
        result[f"{prefix}_rxn"] = reactions
        if args.map_atoms:
            result = add_atom_mapping_columns(
                result,
                f"{prefix}_rxn",
                prefix,
                mapper=mapper,
                batch_size=args.mapping_batch_size,
            )
        result.to_csv(csv_output, index=False, encoding="utf-8-sig")

        merge_no_smi_records(union_after, collect_reaction_no_smi(result, prefix))
        for key, counter in resolved_smiles_by_name(result, recovered_dicts).items():
            resolved_values[key].update(counter)
        outputs[prefix] = {
            "csv": str(csv_output),
            "smiles_dict": str(dict_output),
        }
        model_summaries[prefix] = {
            "before_no_smi_occurrences": count_reaction_no_smi(frame, prefix),
            "after_no_smi_occurrences": count_reaction_no_smi(result, prefix),
            **mapping_counts(result, prefix),
        }

    audit_rows = []
    for key, before in sorted(
        union_before.items(),
        key=lambda item: (-item[1]["occurrences"], item[0]),
    ):
        after_count = union_after.get(key, {}).get("occurrences", 0)
        display_name = before["names"].most_common(1)[0][0]
        override = loaded_overrides.get(key, {})
        resolution = resolution_cache.get(key, (None, None))
        smiles = resolved_values.get(key, Counter())
        if after_count == 0:
            status = "resolved"
        elif after_count < before["occurrences"]:
            status = "partially_resolved"
        elif should_skip_automatic_resolution(display_name) and not override:
            status = "retained_ambiguous_or_nonmolecular"
        else:
            status = "not_found"
        audit_rows.append(
            {
                "name": display_name,
                "normalized_name": key,
                "before_occurrences": before["occurrences"],
                "after_occurrences": after_count,
                "status": status,
                "models": "; ".join(sorted(before["models"])),
                "model_occurrences": "; ".join(
                    f"{model}:{count}"
                    for model, count in sorted(before["models"].items())
                ),
                "sides": "; ".join(
                    f"{side}:{count}" for side, count in sorted(before["sides"].items())
                ),
                "rows_zero_based": ";".join(map(str, sorted(before["rows"]))),
                "resolved_smiles": " | ".join(smiles),
                "resolver_source": resolution[1] or "",
                "override_kind": override.get("kind", ""),
                "lookup_name": override.get("lookup_name", ""),
                "evidence_url": override.get("evidence_url", ""),
                "note": override.get("note", ""),
            }
        )
    pd.DataFrame(audit_rows).to_csv(args.audit, index=False, encoding="utf-8-sig")

    before_occurrences = sum(
        count_reaction_no_smi(frame, prefix)
        for prefix, _csv, _dictionary, frame, _start, _stop in specs
    )
    after_occurrences = sum(
        model_summary["after_no_smi_occurrences"]
        for model_summary in model_summaries.values()
    )
    resolved_unique = sum(row["status"] == "resolved" for row in audit_rows)
    summary = {
        "before": {
            "parsed_unique_names": len(union_before),
            "occurrences": before_occurrences,
        },
        "after": {
            "parsed_unique_names": len(union_after),
            "occurrences": after_occurrences,
        },
        "recovered": {
            "parsed_unique_names": resolved_unique,
            "occurrences": before_occurrences - after_occurrences,
        },
        "models": model_summaries,
        "outputs": outputs,
        "audit": str(args.audit),
        "audit_name_scope": (
            "Unique-name counts cover parseable placeholders; occurrence counts "
            "include every literal (NoSmi), including nested-bracket names."
        ),
        "override_count": len(loaded_overrides),
    }
    Path(args.summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
