"""Recount Table 1 from the newly generated grants_issue.csv, not legacy labels."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

try:
    from analysis import rebuild_lowe_population as B
except ModuleNotFoundError:
    import rebuild_lowe_population as B


def recount(population, manifest_path, summary_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or not summary["complete_source"]:
        raise ValueError("A pilot or incomplete population cannot produce the final Table 1")
    if manifest["raw_sha256"] != B.RAW_SHA256 or manifest["policy"] != B.POLICY:
        raise ValueError("Unrecognized source or population policy")
    if B.sha256(population) != manifest["output_sha256"]["grants_issue.csv"]:
        raise ValueError("Population CSV differs from its completed manifest")
    count = Counter(); n = 0
    with population.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["Policy"] != B.POLICY:
                raise ValueError("Mixed policies in population CSV")
            n += 1
            try:
                rr, rg, rp, u = B.roles(r["ReactionSmiles"])
            except ValueError:
                if r["ProcessingStatus"] != "reaction_syntax_error":
                    raise
                continue
            actual = (len(rr), len(rg), len(rp))
            stored = tuple(int(r[k]) for k in ["NReactants", "NReagents", "NProducts"])
            if actual != stored:
                raise ValueError("Molecular-count mismatch at raw row " + r["RawRow"])
            met = B.criteria(*actual)
            count.update(met); count["union"] += bool(met)
    if n != summary["flagged_rows"] or any(count[k] != summary["table1"].get(k, 0) for k in [*B.LABELS, "union"]):
        raise ValueError("Recount differs from full-source stage")
    return {"source_rows": summary["source_rows"], "csv_rows": n, "table1": dict(count),
            "csv_sha256": B.sha256(population), "full_source_counts_equal": True,
            "stored_molecular_counts_checked": True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population-dir", required=True, type=Path)
    a = p.parse_args(argv)
    result = recount(a.population_dir / "grants_issue.csv", a.population_dir / "manifest.json", a.population_dir / "summary.json")
    B.dump(a.population_dir / "table1_recount_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
