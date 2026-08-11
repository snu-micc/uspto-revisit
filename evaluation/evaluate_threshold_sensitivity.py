"""Reproduce the template-anomaly frequency-threshold sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


THRESHOLDS = (("N=1", 1), ("N<5", 5), ("N<10", 10))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def is_frequency_flagged(frequency: int, label: str, cutoff: int) -> bool:
    return frequency == 1 if label == "N=1" else 0 < frequency < cutoff


def evaluate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for label, cutoff in THRESHOLDS:
        counts = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
        for row in rows:
            filtered = (
                is_frequency_flagged(int(row["template_frequency"]), label, cutoff)
                or as_bool(row["template_failure"])
                or as_bool(row["identical_structures"])
            )
            noise_free = as_bool(row["noise_free"])
            if noise_free and not filtered:
                counts["tp"] += 1
            elif noise_free:
                counts["fn"] += 1
            elif filtered:
                counts["tn"] += 1
            else:
                counts["fp"] += 1

        noise_free_total = counts["tp"] + counts["fn"]
        noisy_total = counts["fp"] + counts["tn"]
        total = noise_free_total + noisy_total
        results.append(
            {
                "template_anomaly_frequency_threshold": label,
                "noise_free_reactions": noise_free_total,
                "noisy_reactions": noisy_total,
                "total_reactions": total,
                "passed_tp": counts["tp"],
                "filtered_fn": counts["fn"],
                "passed_fp": counts["fp"],
                "filtered_tn": counts["tn"],
                "noise_free_data_saved_rate": counts["tp"] / noise_free_total,
                "noisy_data_filtered_rate": counts["tn"] / noisy_total,
                "denoising_accuracy": (counts["tp"] + counts["tn"]) / total,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evaluation/data/template_threshold_audit.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/template_threshold_sensitivity.csv"),
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 400:
        raise ValueError(f"Expected 400 audited rows, found {len(rows)}.")

    results = evaluate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
