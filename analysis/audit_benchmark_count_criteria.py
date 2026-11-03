"""Audit existing benchmark strata using only source-derived Table 1 counts.

Distinguishes meeting the assigned flag from meeting exactly that flag. This
does not import historical reagent/Issue annotations to determine molecular
roles, and does not claim to audit template rarity or chemical duplication.
"""
import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from analysis.rebuild_lowe_population import roles, criteria, LABELS
    from analysis.benchmark_inputs import load_benchmark
except ModuleNotFoundError:
    from rebuild_lowe_population import roles, criteria, LABELS
    from benchmark_inputs import load_benchmark


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--membership", type=Path,
                   help="Optional metadata for older split inputs; omit for the unified input.csv.")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv); a.output.mkdir(parents=True, exist_ok=True)
    records = load_benchmark(a.input, a.membership)
    rows = []
    for i, r in enumerate(records):
        assigned = r["stratum"]
        if assigned not in LABELS:
            raise ValueError(f"Unknown sampling stratum: {assigned}")
        rr, rg, rp, unmapped = roles(r["Lowe_smiles"])
        met = criteria(len(rr), len(rg), len(rp))
        rows.append({"idx": r.get("idx", i), "assigned_stratum": assigned,
            "n_reactants": len(rr), "n_reagents": len(rg), "n_products": len(rp),
            "criteria": "+".join(met), "assigned_criterion_met": assigned in met,
            "single_criterion_match": met == [assigned],
            "reason": "" if met == [assigned] else "multiple criteria" if assigned in met else "assigned criterion not met",
            "reaction_sha256": hashlib.sha256(r["Lowe_smiles"].encode()).hexdigest()})
    with (a.output / "benchmark_count_criteria.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
        "records": len(rows), "strata": dict(Counter(r["assigned_stratum"] for r in rows)),
        "assigned_criterion_met": sum(r["assigned_criterion_met"] for r in rows),
        "single_criterion_match": sum(r["single_criterion_match"] for r in rows),
        "count_criteria_replacement_ids": [int(r["idx"]) for r in rows if not r["single_criterion_match"]],
        "scope": "four molecular-count flags only; rarity and duplication are separate audits"}
    if a.membership:
        result["membership_sha256"] = hashlib.sha256(a.membership.read_bytes()).hexdigest()
    (a.output / "benchmark_count_summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
