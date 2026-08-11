"""Reproduce the heuristic-specific weighted extrapolation in Table S18."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COUNT_COLUMNS = ("tp", "fn", "fp", "tn")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/data/weighted_sampling_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/weighted_extrapolation.csv"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame["sampling_weight"] = frame["population_n"] / frame["sample_n"]
    for name in COUNT_COLUMNS:
        frame[f"weighted_{name}"] = frame[f"sample_{name}"] * frame["sampling_weight"]

    total = {
        "orderly_sampling_category": "TOTAL",
        "population_n": int(frame["population_n"].sum()),
        "sample_n": int(frame["sample_n"].sum()),
        "sampling_weight": float("nan"),
    }
    for name in COUNT_COLUMNS:
        total[f"sample_{name}"] = int(frame[f"sample_{name}"].sum())
        total[f"weighted_{name}"] = frame[f"weighted_{name}"].sum()

    output = pd.concat([frame, pd.DataFrame([total])], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, float_format="%.2f")
    print(f"Wrote {args.output}")
    print(
        "Weighted totals: "
        f"TP={total['weighted_tp']:.0f}, FN={total['weighted_fn']:.0f}, "
        f"FP={total['weighted_fp']:.0f}, TN={total['weighted_tn']:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
