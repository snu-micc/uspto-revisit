"""Evaluate each stage and the complete USPTO reaction-curation pipeline.

The script produces a row-level audit table and a compact summary table for a
single LLM configuration.  It intentionally reports both cumulative success
rates (relative to all 400 records) and conditional success rates (relative to
the records that reached the preceding stage), so errors introduced at each
downstream step can be distinguished from errors inherited from earlier steps.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_bootstrap_root = Path(__file__).resolve().parents[1]
if "--project-root" in sys.argv:
    try:
        _bootstrap_root = Path(sys.argv[sys.argv.index("--project-root") + 1]).resolve()
    except (IndexError, ValueError):
        pass
if str(_bootstrap_root) not in sys.path:
    sys.path.insert(0, str(_bootstrap_root))

try:
    from analysis.evaluate_table3 import (
        build_ground_truth_manifest,
        configure_matching,
        signature_text,
    )
except ModuleNotFoundError:
    from evaluate_table3 import (
        build_ground_truth_manifest,
        configure_matching,
        signature_text,
    )


NO_VALUE = {"", "none", "null", "nan", "nosmi", "not found", "not_found"}
ATOM_MAP = re.compile(r":\d+\]")


def text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def truthy(value: Any) -> bool:
    return text(value).lower() in {"1", "true", "yes"}


def parse_literal(value: Any, expected: type) -> Any:
    raw = text(value)
    if not raw:
        return expected()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(raw)
            return parsed if isinstance(parsed, expected) else expected()
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return expected()


def parse_list(value: Any) -> list[Any]:
    return parse_literal(value, list)


def parse_dict(value: Any) -> dict[str, Any]:
    return parse_literal(value, dict)


def nonblank(value: Any) -> bool:
    return text(value).lower() not in NO_VALUE


def list_complete(values: list[Any]) -> bool:
    return bool(values) and all(nonblank(value) for value in values)


def structured_output_success(prediction: Any) -> tuple[bool, int]:
    payload = parse_dict(prediction)
    if not payload:
        return False, 0
    lower_keys = {str(key).strip().lower(): key for key in payload}
    component_key = next(
        (
            original
            for key, original in lower_keys.items()
            if key in {"reactants, solvents, catalysts", "reactants"}
        ),
        None,
    )
    product_key = next(
        (original for key, original in lower_keys.items() if key in {"product", "products"}),
        None,
    )
    steps_key = next(
        (original for key, original in lower_keys.items() if key == "reaction steps"),
        None,
    )
    if component_key is None or product_key is None or steps_key is None:
        return False, 0
    if not isinstance(payload[component_key], dict) or not isinstance(payload[product_key], dict):
        return False, 0
    steps = payload[steps_key]
    if not isinstance(steps, dict) or not steps:
        return False, 0
    reaction_steps = [
        value
        for key, value in steps.items()
        if "reaction" in str(key).lower() and "work-up" not in str(key).lower()
    ]
    valid = bool(reaction_steps) and all("->" in text(value) for value in reaction_steps)
    return valid, len(reaction_steps)


def valid_reaction_string(value: Any, *, mapped: bool = False) -> bool:
    raw = text(value)
    separator = ">>" if mapped else ">"
    if not raw or separator not in raw:
        return False
    if any(token in raw.lower() for token in ("nosmi", "not found", "not_found")):
        return False
    left, right = raw.rsplit(separator, 1)
    if not left.strip() or not right.strip():
        return False
    return not mapped or (bool(ATOM_MAP.search(left)) and bool(ATOM_MAP.search(right)))


def valid_skeleton_string(value: Any) -> bool:
    raw = text(value)
    if not raw or ">" not in raw:
        return False
    parts = [part.strip() for part in raw.split(">")]
    return len(parts) >= 2 and bool(parts[0]) and bool(parts[-1])


def current_extraction_exact(
    table2_row: pd.Series,
    manifest_row: pd.Series,
) -> bool:
    if not manifest_row["ground_truth_valid"]:
        return False
    accepted = {signature_text(item) for item in manifest_row["_accepted_signatures"]}
    return (
        text(table2_row.get("predicted_signature")) in accepted
        and not truthy(table2_row.get("has_unresolved_nosmi"))
    )


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def build_row_flags(
    review: pd.DataFrame,
    pipeline: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    review = review.copy()
    review["idx"] = review["idx"].astype(int)
    review = review.sort_values("idx").reset_index(drop=True)
    if not (len(review) == len(pipeline) == 400):
        raise ValueError(
            f"Expected 400 rows; found review={len(review)}, pipeline={len(pipeline)}."
        )

    manifest = build_ground_truth_manifest(review).set_index("source_row")
    table2 = table2.loc[table2["method"].eq(model)].copy()
    table3 = table3.loc[table3["model"].eq(model)].copy()
    for name, frame in (("Table 2 flags", table2), ("Table 3 flags", table3)):
        if len(frame) != 400 or frame["source_row"].duplicated().any():
            raise ValueError(f"{name} must contain one {model} row for each of 400 records.")
    table2 = table2.set_index("source_row")
    table3 = table3.set_index("source_row")

    prefix = f"{model}_"
    required_pipeline_columns = {
        "prediction",
        "error",
        f"{prefix}smiles",
        f"{prefix}skeleton",
        f"{prefix}rxn",
        f"{prefix}localmapper_rxn",
        f"{prefix}mapped_rxn",
        f"{prefix}mapping_template",
        f"{prefix}mapping_error",
    }
    missing = required_pipeline_columns - set(pipeline.columns)
    if missing:
        raise ValueError("Pipeline CSV is missing columns: " + ", ".join(sorted(missing)))

    rows: list[dict[str, Any]] = []
    for source_row in range(400):
        review_row = review.iloc[source_row]
        pipeline_row = pipeline.iloc[source_row]
        table2_row = table2.loc[source_row]
        table3_row = table3.loc[source_row]
        manifest_row = manifest.loc[source_row]

        structured_ok, json_reaction_steps = structured_output_success(
            pipeline_row.get("prediction")
        )
        skeleton = parse_list(pipeline_row.get(f"{prefix}skeleton"))
        reactions = parse_list(pipeline_row.get(f"{prefix}rxn"))
        localmapper_reactions = parse_list(pipeline_row.get(f"{prefix}localmapper_rxn"))
        mapped_reactions = parse_list(pipeline_row.get(f"{prefix}mapped_rxn"))
        templates = parse_list(pipeline_row.get(f"{prefix}mapping_template"))
        mapping_errors = [text(value) for value in parse_list(pipeline_row.get(f"{prefix}mapping_error"))]
        smiles_dictionary = parse_dict(pipeline_row.get(f"{prefix}smiles"))

        generation_ok = nonblank(pipeline_row.get("prediction")) and not nonblank(
            pipeline_row.get("error")
        )
        processing_ok = bool(
            generation_ok
            and structured_ok
            and list_complete(skeleton)
            and all(valid_skeleton_string(item) for item in skeleton)
        )
        conversion_ok = bool(
            processing_ok
            and smiles_dictionary
            and all(nonblank(value) for value in smiles_dictionary.values())
            and list_complete(reactions)
            and len(skeleton) == len(reactions)
            and all(valid_reaction_string(item) for item in reactions)
        )
        mapping_ok = bool(
            conversion_ok
            and len(localmapper_reactions) == len(reactions)
            and len(mapped_reactions) == len(reactions)
            and list_complete(localmapper_reactions)
            and all(valid_reaction_string(item, mapped=True) for item in mapped_reactions)
            and not any(mapping_errors)
        )
        template_ok = bool(
            mapping_ok
            and len(templates) == len(reactions)
            and list_complete(templates)
        )

        noise_free = bool(manifest_row["ground_truth_valid"])
        heuristic_passed = truthy(table3_row.get("rare_template_passed"))
        heuristic_filtered = truthy(table3_row.get("rare_template_filtered"))
        if heuristic_passed == heuristic_filtered:
            raise ValueError(
                f"Row {source_row}: heuristic pass/filter flags are not complementary."
            )
        extraction_exact = current_extraction_exact(table2_row, manifest_row)
        denoising_correct = heuristic_passed if noise_free else heuristic_filtered
        strict_e2e = (
            extraction_exact and mapping_ok and template_ok and heuristic_passed
            if noise_free
            else heuristic_filtered
        )

        rows.append(
            {
                "source_row": source_row,
                "idx": int(review_row["idx"]),
                "title": text(review_row.get("title")),
                "ground_truth_valid": noise_free,
                "noise_label": "noise-free" if noise_free else "noisy",
                "generation_success": int(generation_ok),
                "structured_output_success": int(structured_ok),
                "json_reaction_step_count": json_reaction_steps,
                "reaction_processing_success": int(processing_ok),
                "name_to_smiles_success": int(conversion_ok),
                "reaction_step_count": len(reactions),
                "atom_mapping_success": int(mapping_ok),
                "template_processing_success": int(template_ok),
                "extraction_exact": int(extraction_exact),
                "template_anomaly_passed": int(heuristic_passed),
                "template_anomaly_filtered": int(heuristic_filtered),
                "denoising_correct": int(denoising_correct),
                "strict_end_to_end_success": int(strict_e2e),
            }
        )
    return pd.DataFrame(rows)


def build_summary(flags: pd.DataFrame) -> pd.DataFrame:
    total = len(flags)
    stages = [
        ("LLM generation", "generation_success", None),
        ("Structured output", "structured_output_success", "generation_success"),
        ("Reaction-step processing", "reaction_processing_success", "structured_output_success"),
        ("Name-to-SMILES conversion", "name_to_smiles_success", "reaction_processing_success"),
        ("Atom mapping", "atom_mapping_success", "name_to_smiles_success"),
        ("Template processing", "template_processing_success", "atom_mapping_success"),
        ("Exact extraction (noise-free only)", "extraction_exact", None),
        ("Correct denoising decision", "denoising_correct", None),
        ("Strict end-to-end", "strict_end_to_end_success", None),
    ]
    rows: list[dict[str, Any]] = []
    for stage, column, previous in stages:
        if column == "extraction_exact":
            denominator_mask = flags["noise_label"].eq("noise-free")
        elif previous is not None:
            denominator_mask = flags[previous].eq(1)
        else:
            denominator_mask = pd.Series(True, index=flags.index)
        denominator = int(denominator_mask.sum())
        successes = int((denominator_mask & flags[column].eq(1)).sum())
        cumulative_successes = int(flags[column].sum())
        lower, upper = wilson_interval(successes, denominator)
        rows.append(
            {
                "stage": stage,
                "success_n": successes,
                "evaluated_n": denominator,
                "conditional_success_rate": successes / denominator if denominator else math.nan,
                "conditional_failure_n": denominator - successes,
                "cumulative_success_n": cumulative_successes,
                "cumulative_success_rate_all_400": cumulative_successes / total,
                "wilson_ci_lower": lower,
                "wilson_ci_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def build_noise_free_step_table(flags: pd.DataFrame) -> pd.DataFrame:
    """Build the manuscript step table on the fixed noise-free denominator."""
    noise_free = flags.loc[flags["noise_label"].eq("noise-free")].copy()
    reference_n = len(noise_free)
    if reference_n == 0:
        raise ValueError("No noise-free rows were available for step evaluation.")

    stages = [
        ("LLM generation", "generation_success"),
        ("Structured output", "structured_output_success"),
        ("Reaction-step processing", "reaction_processing_success"),
        ("Name-to-SMILES conversion", "name_to_smiles_success"),
        ("Atom mapping", "atom_mapping_success"),
        ("Template processing", "template_processing_success"),
    ]
    active = pd.Series(True, index=noise_free.index)
    rows: list[dict[str, Any]] = []
    for stage, column in stages:
        conditional_n = int(active.sum())
        active = active & noise_free[column].eq(1)
        success_n = int(active.sum())
        rows.append(
            {
                "step": stage,
                "success_n": success_n,
                "reference_n": reference_n,
                "success_over_reference": f"{success_n}/{reference_n}",
                "overall_success_rate_percent": 100 * success_n / reference_n,
                "conditional_success_n": success_n,
                "conditional_evaluated_n": conditional_n,
                "conditional_success_rate_percent": (
                    100 * success_n / conditional_n if conditional_n else math.nan
                ),
            }
        )

    exact_n = int(noise_free["extraction_exact"].eq(1).sum())
    rows.append(
        {
            "step": "Exact-extraction accuracy",
            "success_n": exact_n,
            "reference_n": reference_n,
            "success_over_reference": f"{exact_n}/{reference_n}",
            "overall_success_rate_percent": 100 * exact_n / reference_n,
            "conditional_success_n": math.nan,
            "conditional_evaluated_n": math.nan,
            "conditional_success_rate_percent": math.nan,
        }
    )
    return pd.DataFrame(rows)

if __name__ == '__main__':
    raise SystemExit('Run python analysis/reproduce_manuscript.py for current manuscript and SI results.')
