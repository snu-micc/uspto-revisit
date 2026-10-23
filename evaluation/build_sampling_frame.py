"""Rebuild the ORDerly-flag population counts and the benchmark sampling frame.

Reads the released ``grants_issue.csv`` (or ``grants_issue.zip``), whose
``Issue`` column holds a Python-style list of labels per reaction, and
reproduces:

* Table 1: reactions meeting each criterion, counted from the reaction SMILES
  and the Reagent field. A product is a product-side molecule with mapped
  atoms, a reactant is a molecule that shares an atom-map number with a
  product, and the reagents are the molecules in the Reagent field: many
  reactants (> 2) 100,777; many products (> 1) 82,203; many reagents (> 5)
  236,253; no reagent 201,761; union 550,070. Table 1 gives
  them as percentages of the 1,808,937 reactions of Lowe's grant files.
* the 430,428 reactions that meet exactly one Table 1 criterion and carry no
  ``rare_temp`` or ``duplicated`` label: many-products 44,157; many-reactants
  20,157; many-reagents 198,305; no-reagent 167,809. This single-criterion
  frame is the sampling frame of the revised benchmark and of its population
  estimates.
* The released ``Issue`` flags (82,203; 52,758; 236,368; 200,594; union
  536,372). Their reactant count included one extra molecule when no agents
  were listed, and their reagent count an empty entry, so they differ from
  Table 1. The first-release benchmark was drawn from these flags:
* the 501,997 reactions flagged by exactly one of the four (any ``rare_temp``
  or ``duplicated`` labels allowed), the population of the first-release
  weighted extrapolation;
* the 454,475 reactions whose ``Issue`` is exactly one of the four labels and
  nothing else (no ``rare_temp``/``duplicated``): many-products 59,282;
  many-reactants 20,163; many-reagents 207,221; no-reagent 167,809. All 400
  first-release benchmark reactions come from this single-label frame; 28 of
  them meet two Table 1 criteria and were redrawn from the single-criterion
  frame (``evaluation/draw_replacements.py``).

It also links each benchmark row to its population rows by exact (trimmed)
reaction SMILES, writes the 400-row membership list and reports how many rows
fall in the single-criterion frame (400 for the revised benchmark). The random
seed used to draw the original 100-per-stratum sample was not recorded, so that
draw cannot be re-run; this script documents the frames and the membership.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
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
ATOM_MAP = re.compile(r":(\d+)\]")
# Reactions in the patent-grant files of Lowe's release (figshare, version 1); Table 1 percentages use it.
LOWE_GRANT_REACTIONS = 1_808_937
EXPECTED = {
    ("table1", "many_products"): 82203,
    ("table1", "many_reactants"): 100777,
    ("table1", "many_reagents"): 236253,
    ("table1", "no_reagent"): 201761,
    ("table1", "union"): 550070,
    ("single_criterion_frame", "many_products"): 44157,
    ("single_criterion_frame", "many_reactants"): 20157,
    ("single_criterion_frame", "many_reagents"): 198305,
    ("single_criterion_frame", "no_reagent"): 167809,
    ("single_criterion_frame", "total"): 430428,
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
    "table1": (
        "Rows meeting the criterion counted from the reaction SMILES (reactants share an atom-map "
        "number with a mapped product) and the Reagent field (Table 1)."
    ),
    "single_criterion_frame": (
        "Rows meeting exactly one Table 1 criterion and carrying no rare_temp or duplicated label; the frame of "
        "the revised benchmark and its population estimates."
    ),
    "table1_any_flag": "Rows carrying the sampled ORDerly flag as released, alone or with other labels.",
    "exactly_one_of_four": (
        "Rows carrying exactly one of the four sampled flags; rare_temp and "
        "duplicated may also be present (weighted-extrapolation population)."
    ),
    "single_label_frame": (
        "Rows whose Issue is exactly one sampled flag and nothing else; the "
        "frame all 400 first-release benchmark rows were drawn from."
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


def table1_criteria(reaction_smiles: str, reagents: str) -> tuple[str, ...]:
    """Table 1 criteria met by one record, counted from its atom-mapped reaction SMILES and Reagent field."""
    parts = reaction_smiles.split(" |")[0].split(">")
    if len(parts) != 3:
        return ()
    left, agents, right = parts
    products = [m for m in right.split(".") if ATOM_MAP.search(m)]
    product_maps = set(ATOM_MAP.findall(".".join(products)))
    reactants = sum(1 for m in left.split(".") + agents.split(".") if m and set(ATOM_MAP.findall(m)) & product_maps)
    n_reagents = sum(1 for m in reagents.split(".") if m)
    met = []
    if len(products) > 1:
        met.append("many_products")
    if reactants > 2:
        met.append("many_reactants")
    if n_reagents > 5:
        met.append("many_reagents")
    if n_reagents == 0:
        met.append("no_reagent")
    return tuple(met)


def stratum_of_criteria(criteria: tuple[str, ...], labels: tuple[str, ...]) -> str | None:
    """The single-criterion-frame stratum: exactly one Table 1 criterion and no rare_temp or duplicated label."""
    return criteria[0] if len(criteria) == 1 and not set(labels) & set(OTHER_LABELS) else None


def scan(population: Path, member: str | None, targets: set[str]) -> dict[str, Any]:
    """Count label combinations and Table 1 criteria, and collect population rows matching ``targets``."""
    combinations: Counter[tuple[str, ...]] = Counter()
    table1: Counter[tuple[str, ...]] = Counter()
    single_criterion: Counter[str] = Counter()
    matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    issue_cache: dict[str, tuple[str, ...]] = {}
    with open_population_csv(population, member) as (handle, metadata):
        reader = csv.DictReader(handle)
        for source_row, row in enumerate(reader):
            labels = parse_issue(row.get("Issue", ""), issue_cache)
            combinations[labels] += 1
            criteria = table1_criteria(row.get("ReactionSmiles", ""), row.get("Reagent", ""))
            table1[criteria] += 1
            stratum = stratum_of_criteria(criteria, labels)
            if stratum:
                single_criterion[stratum] += 1
            reaction = normalize_reaction(row.get("ReactionSmiles", ""))
            if reaction in targets:
                matches[reaction].append(
                    {
                        "source_row": source_row,
                        "labels": labels,
                        "criteria_stratum": stratum,
                        "year": str(row.get("Year", "")).strip(),
                        "patent_number": str(row.get("PatentNumber", "")).strip(),
                        "paragraph_num": str(row.get("ParagraphNum", "")).strip(),
                        "template_frequency": str(row.get("TemplateFreq", "")).strip(),
                        "reaction_smiles": row.get("ReactionSmiles", ""),
                    }
                )
    return {"combinations": combinations, "table1": table1, "single_criterion": single_criterion,
            "matches": matches, "metadata": metadata}


def stratum_of_single_label(labels: tuple[str, ...]) -> str | None:
    return STRATA.get(labels[0]) if len(labels) == 1 else None


def frame_counts(
    combinations: Counter[tuple[str, ...]],
    table1: Counter[tuple[str, ...]] | None = None,
    single_criterion: Counter[str] | None = None,
) -> list[dict[str, Any]]:
    """Rows of (frame, group, n) for every population definition."""
    counts: Counter[tuple[str, str]] = Counter()
    for criteria, n in (table1 or {}).items():
        for criterion in criteria:
            counts[("table1", criterion)] += n
        if criteria:
            counts[("table1", "union")] += n
    for stratum, n in (single_criterion or {}).items():
        counts[("single_criterion_frame", stratum)] += n
        counts[("single_criterion_frame", "total")] += n
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

    counts = frame_counts(scanned["combinations"], scanned["table1"], scanned["single_criterion"])
    membership, provenance = link_benchmark(benchmark, scanned["matches"])
    criteria_stratum = {item["source_row"]: item["criteria_stratum"]
                        for items in scanned["matches"].values() for item in items}

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
            share = f"  {100 * row['n'] / LOWE_GRANT_REACTIONS:.2f}%" if row["frame"] == "table1" else ""
            print(f"{row['frame']:22s} {row['group']:16s} {row['n']:>8,d}  [{status}]{share}")
    per_stratum = Counter(row["stratum"] for row in membership)
    unique = sum(row["source_row_unique"] for row in membership)
    in_frame = sum(criteria_stratum[row["source_row"]] == row["stratum"] for row in membership)
    print(f"Benchmark rows linked: {len(membership)} {dict(sorted(per_stratum.items()))}")
    print(
        f"Unique population row: {unique}/{len(membership)}; the rest match several "
        "rows with the same reaction SMILES and label (first listed as source_row)."
    )
    print(f"In the single-criterion frame of their stratum: {in_frame}/{len(membership)}")
    print(f"Wrote {output_dir / 'sampling_frame_counts.csv'} and related files.")
    return 1 if (args.strict and mismatches) else 0


if __name__ == "__main__":
    raise SystemExit(main())
