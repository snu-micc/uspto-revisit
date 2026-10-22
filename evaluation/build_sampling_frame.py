"""Rebuild the ORDerly-flag population counts and the benchmark sampling frame.

Reads the released ``grants_issue.csv`` (or ``grants_issue.zip``), whose
``Issue`` column holds a Python-style list of labels per reaction, and
reproduces:

* Table 1: reactions carrying each of the four sampled ORDerly flags
  (many-products 82,203; many-reactants 52,758; many-reagents 236,368;
  no-reagent 200,594) and their union (536,372);
* the 501,997 reactions flagged by exactly one of the four (any ``rare_temp``
  or ``duplicated`` labels allowed), the population used for the weighted
  extrapolation in Table S18/S21;
* the 454,475 reactions whose ``Issue`` is exactly one of the four labels and
  nothing else (no ``rare_temp``/``duplicated``): many-products 59,282;
  many-reactants 20,163; many-reagents 207,221; no-reagent 167,809. All 400
  benchmark reactions come from this single-label frame.

It also links each benchmark row to its population rows by exact (trimmed)
reaction SMILES and writes the 400-row membership list. The random seed used
to draw the original 100-per-stratum sample was not recorded, so the draw
itself cannot be re-run; this script documents the frame and the membership.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from evaluation.build_template_threshold_audit import (
        normalize_reaction,
        open_population_csv,
    )
except ModuleNotFoundError:
    from build_template_threshold_audit import (  # type: ignore[no-redef]
        normalize_reaction,
        open_population_csv,
    )

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

STRATA = {
    "many_product (> 1)": "many_products",
    "many_reactant (> 2)": "many_reactants",
    "many_reagent (> 5)": "many_reagents",
    "no_reagent": "no_reagent",
}
OTHER_LABELS = ("rare_temp", "duplicated")
EXPECTED = {
    ("table1_any_flag", "many_products"): 82203,
    ("table1_any_flag", "many_reactants"): 52758,
    ("table1_any_flag", "many_reagents"): 236368,
    ("table1_any_flag", "no_reagent"): 200594,
    ("table1_any_flag", "union"): 536372,
    ("exactly_one_of_four", "total"): 501997,
    ("single_label_frame", "many_products"): 59282,
    ("single_label_frame", "many_reactants"): 20163,
    ("single_label_frame", "many_reagents"): 207221,
    ("single_label_frame", "no_reagent"): 167809,
    ("single_label_frame", "total"): 454475,
    ("all_rows", "total"): 623290,
}
FRAME_DESCRIPTIONS = {
    "all_rows": "Every data row of grants_issue.csv.",
    "label_any": "Rows whose Issue list contains the label (any combination).",
    "table1_any_flag": "Rows carrying the sampled ORDerly flag, alone or with other labels (Table 1).",
    "exactly_one_of_four": (
        "Rows carrying exactly one of the four sampled flags; rare_temp and "
        "duplicated may also be present (weighted-extrapolation population)."
    ),
    "single_label_frame": (
        "Rows whose Issue is exactly one sampled flag and nothing else; the "
        "frame all 400 benchmark rows were drawn from."
    ),
}
MEMBERSHIP_FIELDS = (
    "idx",
    "stratum",
    "issue_label",
    "source_row",
    "source_row_unique",
    "candidate_count",
    "candidate_source_rows",
    "year",
    "patent_number",
    "paragraph_num",
    "candidate_patent_numbers",
    "template_frequency",
    "reaction_sha256",
)


def parse_issue(value: str, cache: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    if value not in cache:
        parsed = ast.literal_eval(value) if value.strip() else []
        if isinstance(parsed, str):
            parsed = [parsed]
        cache[value] = tuple(str(item) for item in parsed)
    return cache[value]


def scan(population: Path, member: str | None, targets: set[str]) -> dict[str, Any]:
    """Count label combinations and collect population rows matching ``targets``."""
    combinations: Counter[tuple[str, ...]] = Counter()
    matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issue_cache: dict[str, tuple[str, ...]] = {}
    with open_population_csv(population, member) as (handle, metadata):
        reader = csv.DictReader(handle)
        for source_row, row in enumerate(reader):
            labels = parse_issue(row.get("Issue", ""), issue_cache)
            combinations[labels] += 1
            reaction = normalize_reaction(row.get("ReactionSmiles", ""))
            if reaction in targets:
                matches[reaction].append(
                    {
                        "source_row": source_row,
                        "labels": labels,
                        "year": str(row.get("Year", "")).strip(),
                        "patent_number": str(row.get("PatentNumber", "")).strip(),
                        "paragraph_num": str(row.get("ParagraphNum", "")).strip(),
                        "template_frequency": str(row.get("TemplateFreq", "")).strip(),
                        "reaction_smiles": row.get("ReactionSmiles", ""),
                    }
                )
    return {"combinations": combinations, "matches": matches, "metadata": metadata}


def stratum_of_single_label(labels: tuple[str, ...]) -> str | None:
    return STRATA.get(labels[0]) if len(labels) == 1 else None


def frame_counts(combinations: Counter[tuple[str, ...]]) -> list[dict[str, Any]]:
    """Rows of (frame, group, n) for every population definition."""
    counts: Counter[tuple[str, str]] = Counter()
    for labels, n in combinations.items():
        counts[("all_rows", "total")] += n
        sampled = [STRATA[label] for label in labels if label in STRATA]
        for label in set(labels):
            counts[("label_any", label)] += n
        for stratum in set(sampled):
            counts[("table1_any_flag", stratum)] += n
        if sampled:
            counts[("table1_any_flag", "union")] += n
        if len(set(sampled)) == 1:
            counts[("exactly_one_of_four", sampled[0])] += n
            counts[("exactly_one_of_four", "total")] += n
        single = stratum_of_single_label(labels)
        if single:
            counts[("single_label_frame", single)] += n
            counts[("single_label_frame", "total")] += n
    rows = []
    for (frame, group), n in sorted(counts.items()):
        expected = EXPECTED.get((frame, group), "")
        rows.append(
            {
                "frame": frame,
                "group": group,
                "n": n,
                "manuscript_value": expected,
                "matches_manuscript": "" if expected == "" else int(n == expected),
                "definition": FRAME_DESCRIPTIONS[frame],
            }
        )
    return rows


def read_benchmark(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for column in ("Issue", "Lowe_smiles"):
        if rows and column not in rows[0]:
            raise ValueError(f"{path} has no {column} column.")
    return rows


def link_benchmark(
    benchmark: list[dict[str, str]],
    matches: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issue_cache: dict[str, tuple[str, ...]] = {}
    membership = []
    provenance = []
    for idx, record in enumerate(benchmark):
        labels = parse_issue(record["Issue"], issue_cache)
        stratum = stratum_of_single_label(labels)
        if stratum is None:
            raise ValueError(f"Benchmark row {idx} is not a single-label row: {labels}.")
        reaction = normalize_reaction(record["Lowe_smiles"])
        candidates = [item for item in matches.get(reaction, []) if item["labels"] == labels]
        if not candidates:
            raise ValueError(f"Benchmark row {idx} has no population row with label {labels}.")
        first = candidates[0]
        membership.append(
            {
                "idx": idx,
                "stratum": stratum,
                "issue_label": labels[0],
                "source_row": first["source_row"],
                "source_row_unique": int(len(candidates) == 1),
                "candidate_count": len(candidates),
                "candidate_source_rows": ";".join(str(item["source_row"]) for item in candidates),
                "year": first["year"],
                "patent_number": first["patent_number"],
                "paragraph_num": first["paragraph_num"],
                "candidate_patent_numbers": ";".join(
                    dict.fromkeys(item["patent_number"] for item in candidates)
                ),
                "template_frequency": first["template_frequency"],
                "reaction_sha256": hashlib.sha256(reaction.encode("utf-8")).hexdigest(),
            }
        )
        provenance.append(
            {
                "idx": idx,
                **record,
                "stratum": stratum,
                "source_row": first["source_row"],
                "source_row_unique": int(len(candidates) == 1),
                "candidate_source_rows": ";".join(str(item["source_row"]) for item in candidates),
                "year": first["year"],
                "patent_number": first["patent_number"],
                "paragraph_num": first["paragraph_num"],
                "original_lowe_smiles": first["reaction_smiles"],
                "original_equals_input": int(first["reaction_smiles"] == record["Lowe_smiles"]),
            }
        )
    return membership, provenance


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def single_label_weighting_input(
    summary_path: Path,
    counts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy the weighted-extrapolation input with single-label-frame populations."""
    population = {
        row["group"]: row["n"] for row in counts if row["frame"] == "single_label_frame"
    }
    with summary_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["population_n"] = population.get(row["orderly_sampling_category"], 0)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--population",
        type=Path,
        default=Path("evaluation/data/grants_issue.zip"),
        help="grants_issue.zip or grants_issue.csv. Default: %(default)s",
    )
    parser.add_argument("--population-member", default=None)
    parser.add_argument("--input", type=Path, default=Path("examples/input.csv"))
    parser.add_argument(
        "--weighting-input",
        type=Path,
        default=Path("evaluation/data/weighted_sampling_summary.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/data"))
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=Path("examples/input_with_provenance.csv"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any count differs from the manuscript value.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = read_benchmark(args.input)
    targets = {normalize_reaction(record["Lowe_smiles"]) for record in benchmark}
    scanned = scan(args.population, args.population_member, targets)

    counts = frame_counts(scanned["combinations"])
    membership, provenance = link_benchmark(benchmark, scanned["matches"])

    output_dir = args.output_dir
    write_csv(
        output_dir / "sampling_frame_counts.csv",
        counts,
        ("frame", "group", "n", "manuscript_value", "matches_manuscript", "definition"),
    )
    combination_rows = [
        {
            "issue_labels": json.dumps(list(labels)),
            "label_count": len(labels),
            "n": n,
            "in_single_label_frame": int(stratum_of_single_label(labels) is not None),
        }
        for labels, n in sorted(scanned["combinations"].items(), key=lambda item: (-item[1], item[0]))
    ]
    write_csv(
        output_dir / "sampling_frame_label_combinations.csv",
        combination_rows,
        ("issue_labels", "label_count", "n", "in_single_label_frame"),
    )
    write_csv(output_dir / "benchmark_sample_membership.csv", membership, MEMBERSHIP_FIELDS)
    if args.weighting_input.is_file():
        weighting = single_label_weighting_input(args.weighting_input, counts)
        write_csv(
            output_dir / "weighted_sampling_summary_single_label_frame.csv",
            weighting,
            list(weighting[0]),
        )
    provenance_fields = list(provenance[0])
    write_csv(args.provenance_output, provenance, provenance_fields)

    mismatches = [row for row in counts if row["matches_manuscript"] == 0]
    for row in counts:
        if row["manuscript_value"] != "":
            status = "ok" if row["matches_manuscript"] else "MISMATCH"
            print(f"{row['frame']:22s} {row['group']:16s} {row['n']:>8,d}  [{status}]")
    per_stratum = Counter(row["stratum"] for row in membership)
    unique = sum(row["source_row_unique"] for row in membership)
    print(f"Benchmark rows linked: {len(membership)} {dict(sorted(per_stratum.items()))}")
    print(
        f"Unique population row: {unique}/{len(membership)}; the rest match several "
        "rows with the same reaction SMILES and label (first listed as source_row)."
    )
    print(f"Wrote {output_dir / 'sampling_frame_counts.csv'} and related files.")
    return 1 if (args.strict and mismatches) else 0


if __name__ == "__main__":
    raise SystemExit(main())
