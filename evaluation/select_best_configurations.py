"""Select one best-performing reasoning configuration per model family."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from evaluation.evaluate_table2 import publication_table
except ModuleNotFoundError:
    from evaluate_table2 import publication_table  # type: ignore[no-redef]


FAMILIES = {
    "qwen-3.5-9b-off": "qwen-3.5-9B",
    "qwen-3.5-9b-low": "qwen-3.5-9B",
    "qwen-3.5-9b-high": "qwen-3.5-9B",
    "qwen-3.5-9b-ft-off": "fine-tuned qwen-3.5-9B",
    "qwen-3.5-9b-ft-low": "fine-tuned qwen-3.5-9B",
    "qwen-3.5-9b-ft-high": "fine-tuned qwen-3.5-9B",
    "gpt-4.1-mini-nonreasoning-shared": "gpt-4.1-mini",
    "gpt-4.1-mini-ft-nonreasoning-shared": "fine-tuned gpt-4.1-mini",
    "gpt-5.4-none": "gpt-5.4",
    "gpt-5.4-low": "gpt-5.4",
    "gpt-5.4-high": "gpt-5.4",
    "gpt-5.6-sol-none": "gpt-5.6-sol",
    "gpt-5.6-sol-low": "gpt-5.6-sol",
    "gpt-5.6-sol-high": "gpt-5.6-sol",
    "gemini-2.5-flash-none": "gemini-2.5-flash",
    "gemini-3.1-pro-preview-low": "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-high": "gemini-3.1-pro-preview",
    "gemini-3.6-flash-low": "gemini-3.6-flash",
    "gemini-3.6-flash-high": "gemini-3.6-flash",
}
EFFORT_RANK = {"off": 0, "none": 0, "nonreasoning": 0, "low": 1, "high": 2}
FAMILY_ORDER = (
    "qwen-3.5-9B",
    "fine-tuned qwen-3.5-9B",
    "gpt-4.1-mini",
    "fine-tuned gpt-4.1-mini",
    "gpt-5.4",
    "gpt-5.6-sol",
    "gemini-2.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
)


def effort_rank(method: str) -> int:
    return next(
        (rank for label, rank in EFFORT_RANK.items() if label in method),
        99,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("evaluation/results/table2/table2_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/table2_best_models.csv"),
    )
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    original = metrics.loc[metrics["method"].eq("original_data")]
    models = metrics.loc[metrics["method"].isin(FAMILIES)].copy()
    models["model_family"] = models["method"].map(FAMILIES)
    models["effort_rank"] = models["method"].map(effort_rank)
    selected = (
        models.sort_values(
            ["model_family", "reaction_accuracy", "effort_rank"],
            ascending=[True, False, True],
        )
        .groupby("model_family", sort=False, as_index=False)
        .first()
    )
    selected["model_family"] = pd.Categorical(
        selected["model_family"], categories=FAMILY_ORDER, ordered=True
    )
    selected = selected.sort_values("model_family")
    selected["extraction_method"] = selected["model_family"].astype(str)
    selected = pd.concat([original, selected[metrics.columns]], ignore_index=True)
    methods = tuple(selected["method"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publication_table(selected, methods).to_csv(
        args.output, index=False, encoding="utf-8-sig"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
