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
                (row["template_frequency"] not in (None, "") and is_frequency_flagged(int(row["template_frequency"]), label, cutoff))
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

if __name__ == '__main__':
    raise SystemExit('Run python analysis/reproduce_manuscript.py for current manuscript and SI results.')
