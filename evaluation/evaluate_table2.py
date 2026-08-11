"""Compute Table 2 extraction metrics against the finalized ground truth."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger

try:
    from evaluation.evaluate_table3 import (
        COMPONENT_FIELDS,
        MODELS,
        build_ground_truth_manifest,
        comparison_smiles,
        components_from_row,
        parse_bool,
        signature_text,
    )
except ModuleNotFoundError:
    from evaluate_table3 import (  # type: ignore[no-redef]
        COMPONENT_FIELDS,
        MODELS,
        build_ground_truth_manifest,
        comparison_smiles,
        components_from_row,
        parse_bool,
        signature_text,
    )

RDLogger.DisableLog("rdApp.*")

ORIGINAL_METHOD = "original_data"
AI_METHOD = "ai_without_title"
METHOD_LABELS = {
    ORIGINAL_METHOD: "Original data",
    AI_METHOD: "Ai et al.",
    "qwen-3.5-9b-off": "qwen-3.5-9B (none)",
    "qwen-3.5-9b-low": "qwen-3.5-9B (low)",
    "qwen-3.5-9b-high": "qwen-3.5-9B (high)",
    "qwen-3.5-9b-ft-off": "fine-tuned qwen-3.5-9B (none)",
    "qwen-3.5-9b-ft-low": "fine-tuned qwen-3.5-9B (low)",
    "qwen-3.5-9b-ft-high": "fine-tuned qwen-3.5-9B (high)",
    "gpt-4.1-mini-nonreasoning-shared": "gpt-4.1-mini (none)",
    "gpt-4.1-mini-ft-nonreasoning-shared": "fine-tuned gpt-4.1-mini (none)",
    "gpt-5.4-none": "gpt-5.4 (none)",
    "gpt-5.4-low": "gpt-5.4 (low)",
    "gpt-5.4-high": "gpt-5.4 (high)",
    "gpt-5.6-sol-none": "gpt-5.6-sol (none)",
    "gpt-5.6-sol-low": "gpt-5.6-sol (low)",
    "gpt-5.6-sol-high": "gpt-5.6-sol (high)",
    "gemini-2.5-flash-none": "gemini-2.5-flash (none)",
    "gemini-3.1-pro-preview-low": "gemini-3.1-pro-preview (low)",
    "gemini-3.1-pro-preview-high": "gemini-3.1-pro-preview (high)",
    "gemini-3.6-flash-low": "gemini-3.6-flash (low)",
    "gemini-3.6-flash-high": "gemini-3.6-flash (high)",
}
ROLE_AS_SET = {
    "reactants": False,
    "reagents": True,
    "products": True,
}
Signature = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


def strip_lowe_metadata(value: Any) -> str:
    """Remove whitespace-separated CXSMILES metadata from a Lowe reaction."""
    return str(value or "").strip().split(maxsplit=1)[0]


def atom_maps(smiles: str) -> set[int] | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return {
        atom.GetAtomMapNum()
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }


def classify_lowe_components(value: Any) -> dict[str, list[str]]:
    """Classify Lowe compounds by atom-map overlap with product atoms."""
    reaction = strip_lowe_metadata(value)
    parts = reaction.split(">")
    if len(parts) != 3:
        return {field: [] for field in COMPONENT_FIELDS}

    left, middle, right = (
        [item.strip() for item in section.split(".") if item.strip()]
        for section in parts
    )
    product_maps: set[int] = set()
    for smiles in right:
        maps = atom_maps(smiles)
        if maps:
            product_maps.update(maps)

    reactants: list[str] = []
    reagents: list[str] = []
    for smiles in left + middle:
        maps = atom_maps(smiles)
        if maps and maps & product_maps:
            reactants.append(smiles)
        else:
            reagents.append(smiles)
    return {
        "reactants": reactants,
        "reagents": reagents,
        "products": right,
    }


def normalized_prediction_section(
    components: list[str],
    *,
    as_set: bool,
) -> tuple[str, ...]:
    """Normalize predictions while retaining invalid entries as false positives."""
    normalized: list[str] = []
    for smiles in components:
        canonical = comparison_smiles(smiles)
        normalized.append(
            canonical
            if canonical is not None
            else f"__invalid__:{smiles.strip()}"
        )
    return tuple(sorted(set(normalized) if as_set else normalized))


def prediction_signature(components: dict[str, list[str]]) -> Signature:
    return tuple(
        normalized_prediction_section(
            components[field],
            as_set=ROLE_AS_SET[field],
        )
        for field in COMPONENT_FIELDS
    )  # type: ignore[return-value]


def section_counts(
    predicted: tuple[str, ...],
    ground_truth: tuple[str, ...],
) -> dict[str, int]:
    predicted_counts = Counter(predicted)
    ground_truth_counts = Counter(ground_truth)
    true_positive = sum((predicted_counts & ground_truth_counts).values())
    false_positive = sum(predicted_counts.values()) - true_positive
    false_negative = sum(ground_truth_counts.values()) - true_positive
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "predicted_count": sum(predicted_counts.values()),
        "ground_truth_count": sum(ground_truth_counts.values()),
        "exact": int(predicted_counts == ground_truth_counts),
        "ground_truth_covered": int(
            not bool(ground_truth_counts - predicted_counts)
        ),
    }


def compare_signatures(
    predicted: Signature,
    ground_truth: Signature,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    total_tp = 0
    total_error = 0
    all_exact = True
    for field, predicted_section, ground_truth_section in zip(
        COMPONENT_FIELDS,
        predicted,
        ground_truth,
    ):
        counts = section_counts(predicted_section, ground_truth_section)
        for key, value in counts.items():
            result[f"{field}_{key}"] = value
        total_tp += counts["tp"]
        total_error += counts["fp"] + counts["fn"]
        all_exact &= bool(counts["exact"])
    result["component_exact"] = int(all_exact)
    result["_score"] = (int(all_exact), total_tp, -total_error)
    return result


def choose_ground_truth(
    predicted: Signature,
    alternatives: list[Signature],
    *,
    prefer_coverage: bool = False,
) -> tuple[int, Signature, dict[str, Any]]:
    """Choose one whole-reaction GT alternative without role-wise cherry-picking."""
    if not alternatives:
        raise ValueError("At least one ground-truth alternative is required.")
    candidates = []
    for slot, ground_truth in enumerate(alternatives, start=1):
        comparison = compare_signatures(predicted, ground_truth)
        all_ground_truth_covered = int(
            all(
                bool(comparison[f"{field}_ground_truth_covered"])
                for field in COMPONENT_FIELDS
            )
        )
        score = (
            (all_ground_truth_covered, *comparison["_score"])
            if prefer_coverage
            else comparison["_score"]
        )
        candidates.append((score, -slot, slot, ground_truth, comparison))
    _, _, slot, ground_truth, comparison = max(candidates)
    comparison = dict(comparison)
    comparison.pop("_score")
    return slot, ground_truth, comparison


def model_has_nosmi(row: pd.Series, model: str) -> bool:
    value = row.get(f"{model}_has_unresolved_nosmi", 0)
    try:
        return bool(int(value or 0))
    except (TypeError, ValueError):
        return str(value).strip().lower() in {"true", "yes"}


def build_flags(
    review: pd.DataFrame,
    input_frame: pd.DataFrame,
    models: tuple[str, ...],
) -> pd.DataFrame:
    if len(review) != len(input_frame):
        raise ValueError(
            f"Review CSV has {len(review)} rows but input CSV has "
            f"{len(input_frame)} rows."
        )
    if "Lowe_smiles" not in input_frame:
        raise ValueError("Input CSV must contain Lowe_smiles.")

    manifest = build_ground_truth_manifest(review).set_index("source_row")
    review_by_idx = review.set_index("idx", drop=False)
    expected_indices = set(range(len(input_frame)))
    if set(review_by_idx.index) != expected_indices:
        raise ValueError("Review idx values must match input row positions.")

    methods = (ORIGINAL_METHOD, *models)
    rows: list[dict[str, Any]] = []
    for source_row in range(len(input_frame)):
        review_row = review_by_idx.loc[source_row]
        ground_truth_row = manifest.loc[source_row]
        included = bool(ground_truth_row["ground_truth_valid"])
        alternatives: list[Signature] = ground_truth_row["_accepted_signatures"]

        for method in methods:
            if method == ORIGINAL_METHOD:
                components = classify_lowe_components(
                    input_frame.iloc[source_row]["Lowe_smiles"]
                )
                has_nosmi = False
                source_type = "Lowe"
            else:
                components = components_from_row(review_row, f"{method}_")
                has_nosmi = model_has_nosmi(review_row, method)
                source_type = "model"

            predicted = prediction_signature(components)
            row: dict[str, Any] = {
                "source_row": source_row,
                "method": method,
                "extraction_method": METHOD_LABELS.get(method, method),
                "source_type": source_type,
                "title": str(review_row.get("title", "") or ""),
                "ground_truth_valid": included,
                "ground_truth_source": ground_truth_row["ground_truth_source"],
                "evaluation_included": int(included),
                "exclusion_reason": (
                    str(ground_truth_row["review_note"] or "")
                    if not included
                    else ""
                ),
                "has_unresolved_nosmi": int(has_nosmi),
                "predicted_signature": signature_text(predicted),
                "matched_ground_truth": "",
                "ground_truth_signature": "",
                "component_exact": "",
                "reaction_correct": "",
            }
            if included:
                slot, selected_ground_truth, comparison = choose_ground_truth(
                    predicted,
                    alternatives,
                    prefer_coverage=method == AI_METHOD,
                )
                row.update(comparison)
                row["matched_ground_truth"] = slot
                row["ground_truth_signature"] = signature_text(selected_ground_truth)
                if method == AI_METHOD:
                    row["reaction_correct"] = int(
                        all(
                            bool(comparison[f"{field}_ground_truth_covered"])
                            for field in COMPONENT_FIELDS
                        )
                    )
                else:
                    row["reaction_correct"] = int(
                        bool(comparison["component_exact"]) and not has_nosmi
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def build_metrics(
    flags: pd.DataFrame,
    methods: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for method in methods:
        selected = flags.loc[
            flags["method"].eq(method) & flags["evaluation_included"].eq(1)
        ]
        if selected.empty:
            raise ValueError(f"No evaluable rows found for {method}.")
        record: dict[str, Any] = {
            "method": method,
            "extraction_method": METHOD_LABELS.get(method, method),
            "evaluated_reactions": len(selected),
            "excluded_reactions": int(
                (
                    flags["method"].eq(method)
                    & flags["evaluation_included"].eq(0)
                ).sum()
            ),
        }
        for field in COMPONENT_FIELDS:
            tp = int(selected[f"{field}_tp"].sum())
            fp = int(selected[f"{field}_fp"].sum())
            fn = int(selected[f"{field}_fn"].sum())
            record.update(
                {
                    f"{field}_tp": tp,
                    f"{field}_fp": fp,
                    f"{field}_fn": fn,
                    f"{field}_precision": safe_divide(tp, tp + fp),
                    f"{field}_recall": safe_divide(tp, tp + fn),
                }
            )
        reaction_correct = int(selected["reaction_correct"].sum())
        total = len(selected)
        lower, upper = wilson_interval(reaction_correct, total)
        record.update(
            {
                "reaction_correct": reaction_correct,
                "reaction_incorrect": total - reaction_correct,
                "reaction_accuracy": reaction_correct / total,
                "reaction_accuracy_ci_lower": lower,
                "reaction_accuracy_ci_upper": upper,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def publication_table(
    metrics: pd.DataFrame,
    methods: tuple[str, ...],
) -> pd.DataFrame:
    metrics_by_method = metrics.set_index("method")
    rows = []
    for method in methods:
        row = metrics_by_method.loc[method]
        rows.append(
            [
                row["extraction_method"],
                f"{row['reactants_precision'] * 100:.4f}",
                f"{row['reactants_recall'] * 100:.4f}",
                f"{row['reagents_precision'] * 100:.4f}",
                f"{row['reagents_recall'] * 100:.4f}",
                f"{row['products_precision'] * 100:.4f}",
                f"{row['products_recall'] * 100:.4f}",
                (
                    f"{row['reaction_accuracy'] * 100:.4f} "
                    f"({row['reaction_accuracy_ci_lower'] * 100:.4f}"
                    f"\u2013{row['reaction_accuracy_ci_upper'] * 100:.4f})"
                ),
            ]
        )
    columns = pd.MultiIndex.from_tuples(
        [
            ("Extraction method", ""),
            ("Reactants", "Precision (%)"),
            ("Reactants", "Recall (%)"),
            ("Reagents", "Precision (%)"),
            ("Reagents", "Recall (%)"),
            ("Product", "Precision (%)"),
            ("Product", "Recall (%)"),
            ("Reaction", "Accuracy (%, 95% Wilson CI)"),
        ]
    )
    return pd.DataFrame(rows, columns=columns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review",
        default="evaluation/benchmark_all_configurations.csv",
        help="Finalized ground-truth review CSV.",
    )
    parser.add_argument(
        "--input",
        default="examples/input.csv",
        help="Input CSV containing Lowe_smiles.",
    )
    parser.add_argument("--output-dir", default="evaluation/results/table2")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument(
        "--ai-components",
        default="benchmarks/ai_et_al/components.csv",
        help=(
            "Optional Ai baseline component CSV produced by "
            "prepare_ai_baseline.py."
        ),
    )
    return parser


def merge_ai_components(
    review: pd.DataFrame,
    component_path: str | Path,
) -> pd.DataFrame:
    """Attach externally prepared Ai component lists after row validation."""
    components = pd.read_csv(component_path, encoding="utf-8-sig").fillna("")
    required = {
        "idx",
        "title",
        "paragraph",
        f"{AI_METHOD}_reactants_smiles",
        f"{AI_METHOD}_reagents_smiles",
        f"{AI_METHOD}_products_smiles",
        f"{AI_METHOD}_has_unresolved_nosmi",
    }
    missing = required - set(components.columns)
    if missing:
        raise ValueError(
            "Ai component CSV is missing columns: " + ", ".join(sorted(missing))
        )
    components["idx"] = components["idx"].astype(int)
    if components["idx"].duplicated().any():
        raise ValueError("Ai component CSV contains duplicate idx values.")
    if set(components["idx"]) != set(review["idx"]):
        raise ValueError("Ai component idx values do not match ground truth.")

    metadata = review[["idx", "title", "paragraph"]].merge(
        components[["idx", "title", "paragraph"]],
        on="idx",
        suffixes=("_ground_truth", "_ai"),
        validate="one_to_one",
    )
    if not (
        metadata["title_ground_truth"].eq(metadata["title_ai"])
        & metadata["paragraph_ground_truth"].eq(metadata["paragraph_ai"])
    ).all():
        raise ValueError("Ai component title/paragraph values do not match ground truth.")

    ai_columns = [
        column
        for column in components.columns
        if column == "idx" or column.startswith(f"{AI_METHOD}_")
    ]
    return review.merge(
        components[ai_columns],
        on="idx",
        how="left",
        validate="one_to_one",
    )


def main() -> int:
    args = build_parser().parse_args()

    review = pd.read_csv(args.review, encoding="utf-8-sig").fillna("")
    review["idx"] = review["idx"].astype(int)
    models = list(args.models)
    ai_component_path = Path(args.ai_components)
    if ai_component_path.is_file():
        review = merge_ai_components(review, ai_component_path)
        if AI_METHOD not in models:
            models.append(AI_METHOD)
    models_tuple = tuple(models)
    methods = (ORIGINAL_METHOD, *models_tuple)

    input_frame = pd.read_csv(args.input, encoding="utf-8-sig").fillna("")
    flags = build_flags(review, input_frame, models_tuple)
    metrics = build_metrics(flags, methods)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flags.to_csv(
        output_dir / "table2_flags.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        output_dir / "table2_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    publication_path = output_dir / "table2.csv"
    try:
        publication_table(metrics, methods).to_csv(
            publication_path,
            index=False,
            encoding="utf-8-sig",
        )
    except PermissionError:
        publication_path = output_dir / "table2_with_ai.csv"
        publication_table(metrics, methods).to_csv(
            publication_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(
            "table2.csv is open in another program; wrote the current table "
            f"to {publication_path} instead."
        )
    evaluated = int(metrics.iloc[0]["evaluated_reactions"])
    excluded = int(metrics.iloc[0]["excluded_reactions"])
    print(
        f"Evaluated {len(methods)} extraction methods on {evaluated} reactions; "
        f"excluded {excluded} rows without valid ground truth."
    )
    print(f"Manuscript-format table: {publication_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
