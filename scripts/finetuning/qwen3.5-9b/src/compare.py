from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .project import configure_local_caches, inside_project


def read_predictions(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            record_id = str(record["id"])
            if record_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id!r}")
            records[record_id] = record
    return records


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


REQUIRED_KEYS = {"Reactants, Solvents, Catalysts", "Products", "Reaction Steps"}


def parse_json(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def schema_valid(value) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == REQUIRED_KEYS
        and all(isinstance(value[key], dict) for key in REQUIRED_KEYS)
    )


def rouge_l_f1(reference: str, prediction: str) -> float:
    reference_tokens = normalize(reference).split()
    prediction_tokens = normalize(prediction).split()
    if not reference_tokens or not prediction_tokens:
        return float(reference_tokens == prediction_tokens)
    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, prediction_token in enumerate(prediction_tokens, start=1):
            if reference_token == prediction_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    longest = previous[-1]
    precision = longest / len(prediction_tokens)
    recall = longest / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if longest else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and adapter predictions.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure_local_caches()
    baseline_path = inside_project(args.baseline, must_exist=True)
    finetuned_path = inside_project(args.finetuned, must_exist=True)
    output_path = inside_project(args.output)

    baseline = read_predictions(baseline_path)
    finetuned = read_predictions(finetuned_path)
    if baseline.keys() != finetuned.keys():
        raise ValueError("Baseline and fine-tuned prediction IDs do not match")

    rows = []
    for record_id in baseline:
        before, after = baseline[record_id], finetuned[record_id]
        if before["reference"] != after["reference"]:
            raise ValueError(f"Reference mismatch for {record_id}")
        reference = before["reference"]
        reference_json = parse_json(reference)
        baseline_json = parse_json(before["prediction"])
        finetuned_json = parse_json(after["prediction"])
        baseline_score = rouge_l_f1(reference, before["prediction"])
        finetuned_score = rouge_l_f1(reference, after["prediction"])
        rows.append(
            {
                "id": record_id,
                "baseline_exact": normalize(before["prediction"]) == normalize(reference),
                "finetuned_exact": normalize(after["prediction"]) == normalize(reference),
                "baseline_json_valid": baseline_json is not None,
                "finetuned_json_valid": finetuned_json is not None,
                "baseline_schema_valid": schema_valid(baseline_json),
                "finetuned_schema_valid": schema_valid(finetuned_json),
                "baseline_json_exact": baseline_json is not None and baseline_json == reference_json,
                "finetuned_json_exact": finetuned_json is not None and finetuned_json == reference_json,
                "baseline_rouge_l": baseline_score,
                "finetuned_rouge_l": finetuned_score,
                "delta": finetuned_score - baseline_score,
                "reference": reference,
                "baseline": before["prediction"],
                "finetuned": after["prediction"],
            }
        )

    count = len(rows)
    baseline_exact = sum(row["baseline_exact"] for row in rows) / count
    finetuned_exact = sum(row["finetuned_exact"] for row in rows) / count
    baseline_rouge = statistics.fmean(row["baseline_rouge_l"] for row in rows)
    finetuned_rouge = statistics.fmean(row["finetuned_rouge_l"] for row in rows)
    rate_metrics = [
        ("JSON valid", "json_valid"),
        ("Schema valid", "schema_valid"),
        ("JSON exact match", "json_exact"),
    ]
    report = [
        "# Qwen3.5-9B baseline vs QLoRA",
        "",
        f"Evaluated examples: {count}",
        "",
        "| Metric | Baseline | Fine-tuned | Delta |",
        "|---|---:|---:|---:|",
        f"| Exact match | {baseline_exact:.4f} | {finetuned_exact:.4f} | {finetuned_exact-baseline_exact:+.4f} |",
        f"| ROUGE-L F1 | {baseline_rouge:.4f} | {finetuned_rouge:.4f} | {finetuned_rouge-baseline_rouge:+.4f} |",
    ]
    for label, key in rate_metrics:
        before_rate = sum(row[f"baseline_{key}"] for row in rows) / count
        after_rate = sum(row[f"finetuned_{key}"] for row in rows) / count
        report.append(f"| {label} | {before_rate:.4f} | {after_rate:.4f} | {after_rate-before_rate:+.4f} |")
    report.extend(["", "## Per-example results", ""])
    for row in rows:
        report.extend(
            [
                f"### {row['id']} (ROUGE-L delta {row['delta']:+.4f})",
                "",
                f"Reference: {row['reference']}",
                "",
                f"Baseline: {row['baseline']}",
                "",
                f"Fine-tuned: {row['finetuned']}",
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved comparison report to {output_path}")


if __name__ == "__main__":
    main()
