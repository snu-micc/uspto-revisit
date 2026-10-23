"""Re-run the two replacement draws of the revised benchmark.

Draw 1 (seed 20261021) replaced records 15 and 315 of the first release, which repeated reactions already sampled as
records 8 and 309 (other patents of the same families). Draw 2 (seed 20261022) replaced the 28 records that meet two
Table 1 criteria when the criteria are counted explicitly (``build_sampling_frame.table1_criteria``). Both mirror
the original sampling:

  step 1  the patents of the stratum's frame in random order (each patent equally likely);
  step 2  one frame row of that patent, uniformly at random;

and accept the first candidates that pass rules fixed before the draw:

  R1  the patent is not already sampled in the stratum (any candidate population row of a benchmark record of that
      stratum) or drawn in this draw;
  R2  the reaction is not a benchmark reaction (reaction SMILES or candidate population row) or drawn in this draw;
  R3  the row is linked to a reaction in Lowe's text files;
  R4  the linked paragraph is not empty;
  R5  (draw 2) the paragraph text is not that of a benchmark record or of a record drawn in this draw.

Draw 1 uses the frame of the released flags (``Issue`` exactly the stratum label) and the first-release benchmark;
draw 2 uses the single-criterion frame (exactly one Table 1 criterion, no rare_temp or duplicated label) and the
benchmark after draw 1. The random stream is ``numpy.random.default_rng([seed, stream])``, one stream per stratum,
and accepted rows go to the replaced records of the stratum in ascending order.

R1 and R2 are checked against the population file. R3 to R5 need Lowe's CML files, which are not part of this
repository: ``evaluation/data/revised/replacement_draws.csv`` lists, for every candidate the draws examined, the Lowe
file and the SHA-1 and length of the whitespace-normalized paragraph it is linked to, and the script stops if a
draw examines a candidate that is not listed. It checks that the accepted rows are the population rows of the
replacement records in ``benchmark_sample_membership.csv``, that each listed paragraph is the one in ``input.csv``,
and that the replaced records are exactly those whose paragraph differs from the first release.

    python evaluation/draw_replacements.py --population evaluation/data/grants_issue.zip
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation.build_sampling_frame import STRATA, parse_issue, stratum_of_criteria, table1_criteria  # noqa: E402
from evaluation.build_template_threshold_audit import open_population_csv  # noqa: E402

DATA = ROOT / "evaluation" / "data"
REVISED = DATA / "revised"
FIRST_RELEASE = ROOT / "examples" / "input_with_provenance.csv"
LABEL = {stratum: label for label, stratum in STRATA.items()}
DUPLICATES = {15: 8, 315: 309}  # record replaced in draw 1 -> the earlier record with the same reaction
DRAWS = {
    1: {"seed": 20261021, "streams": {"many_products": 0, "no_reagent": 1}},
    2: {"seed": 20261022, "streams": {"many_products": 0, "many_reagents": 1}},
}
SPACE = re.compile(r"\s+")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def trimmed(smiles: str) -> str:
    """The reaction SMILES without the CXSMILES extension, as the draws compared reactions."""
    return str(smiles).strip().split()[0] if str(smiles).strip() else ""


def paragraph_sha1(text: str) -> str:
    return hashlib.sha1(SPACE.sub(" ", str(text)).strip().encode("utf-8")).hexdigest()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def read_population(path: Path, member: str | None) -> pd.DataFrame:
    """Patent, reaction, released Issue and single-criterion stratum of every population row."""
    rows = []
    with open_population_csv(path, member) as (handle, _):
        cache: dict[str, tuple[str, ...]] = {}
        for row in csv.DictReader(handle):
            reaction, issue = row.get("ReactionSmiles", ""), row.get("Issue", "")
            criteria = table1_criteria(reaction, row.get("Reagent", ""))
            rows.append((row.get("PatentNumber", ""), trimmed(reaction), issue,
                         stratum_of_criteria(criteria, parse_issue(issue, cache)) or ""))
    return pd.DataFrame(rows, columns=["patent_number", "reaction", "issue", "criteria_stratum"])


def candidate_rows(value: str) -> list[int]:
    return [int(r) for r in str(value).split(";") if r.isdigit()]


def draw(number: int, population: pd.DataFrame, benchmark: pd.DataFrame, replaced: dict[str, list[int]],
         links: pd.DataFrame) -> list[dict]:
    """Candidates examined in one draw, in order, with their rejection reasons and the record each one replaces."""
    spec = DRAWS[number]
    reactions = {trimmed(s) for s in benchmark.Lowe_smiles}
    rows_sampled = {r for value in benchmark.candidate_source_rows for r in candidate_rows(value)}
    texts = {paragraph_sha1(p) for p in benchmark.paragraph}
    listed = links[links["round"].eq(str(number))].set_index(["stratum", "rank"])
    out = []
    for stratum, stream in spec["streams"].items():
        if number == 1:
            pool = population.index[population.issue.eq(f"['{LABEL[stratum]}']")]
        else:
            pool = population.index[population.criteria_stratum.eq(stratum)]
        rows_by_patent = defaultdict(list)
        for row in pool:  # ascending, so each patent's rows are sorted
            rows_by_patent[population.patent_number.iat[row]].append(int(row))
        patents = sorted(rows_by_patent)
        stratum_rows = [r for value in benchmark.candidate_source_rows[benchmark.stratum.eq(stratum)]
                        for r in candidate_rows(value)]
        sampled_patents = set(population.patent_number.iloc[stratum_rows])
        rng = np.random.default_rng([spec["seed"], stream])
        order = rng.permutation(len(patents))
        wanted = replaced[stratum]
        drawn_patents, drawn_reactions, drawn_texts, accepted = set(), set(), set(), 0
        for rank, position in enumerate(order, start=1):
            if accepted == len(wanted):
                break
            patent = patents[position]
            rows = rows_by_patent[patent]
            row = rows[int(rng.integers(len(rows)))]
            reaction = population.reaction.iat[row]
            if (stratum, str(rank)) not in listed.index:
                raise SystemExit(f"draw {number}, {stratum}: candidate {rank} (population row {row}) is not in "
                                 "replacement_draws.csv")
            link = listed.loc[(stratum, str(rank))]
            if int(link.source_row) != row:
                raise SystemExit(f"draw {number}, {stratum}: candidate {rank} is row {row}, listed {link.source_row}")
            reasons = []
            if patent in sampled_patents or patent in drawn_patents:
                reasons.append("R1 patent already sampled")
            if reaction in reactions or row in rows_sampled or reaction in drawn_reactions:
                reasons.append("R2 reaction already sampled")
            if not link.text_file:
                reasons.append("R3 no Lowe text record")
            if int(link.paragraph_length or 0) == 0:
                reasons.append("R4 empty paragraph")
            if number == 2 and (link.paragraph_sha1 in texts or link.paragraph_sha1 in drawn_texts):
                reasons.append("R5 paragraph already sampled")
            record = {"draw": number, "seed": spec["seed"], "stratum": stratum, "rank": rank, "source_row": row,
                      "patent_number": patent, "patent_rows_in_frame": len(rows), "frame_rows": len(pool),
                      "frame_patents": len(patents), "rejection": "; ".join(reasons), "replaces_idx": ""}
            if not reasons:
                record["replaces_idx"] = wanted[accepted]
                drawn_patents.add(patent)
                drawn_reactions.add(reaction)
                drawn_texts.add(link.paragraph_sha1)
                accepted += 1
            out.append(record)
        if accepted < len(wanted):
            raise SystemExit(f"draw {number}, {stratum}: {accepted} of {len(wanted)} accepted")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--population", type=Path, default=DATA / "grants_issue.zip",
                        help="grants_issue.zip or grants_issue.csv (release data-v1.0.0). Default: %(default)s")
    parser.add_argument("--population-member", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "results" / "revised" /
                        "replacement_draw_candidates.csv")
    args = parser.parse_args(argv)

    population = read_population(args.population, args.population_member)
    first = read(FIRST_RELEASE)
    revised = read(REVISED / "input_with_provenance.csv")
    membership = read(REVISED / "benchmark_sample_membership.csv")
    links = read(REVISED / "replacement_draws.csv")
    first_membership = read(DATA / "benchmark_sample_membership.csv")

    # Draw 1: the second copy of each duplicated reaction.
    for idx, earlier in DUPLICATES.items():
        assert trimmed(first.Lowe_smiles[idx]) == trimmed(first.Lowe_smiles[earlier]), idx
    replaced = {1: {s: sorted(i for i in DUPLICATES if first.stratum[i] == s) for s in DRAWS[1]["streams"]}}
    candidates = draw(1, population, first, replaced[1], links)

    # Draw 2: the benchmark after draw 1; records whose population row meets more than its own criterion.
    after = first.copy()
    for idx in DUPLICATES:
        after.loc[idx] = revised.loc[idx, after.columns].values
    source_rows = first_membership.source_row.astype(int).tolist()
    for idx in DUPLICATES:
        source_rows[idx] = int(revised.source_row[idx])
    two = [i for i, row in enumerate(source_rows) if population.criteria_stratum.iat[row] != after.stratum[i]]
    replaced[2] = {s: sorted(i for i in two if after.stratum[i] == s) for s in DRAWS[2]["streams"]}
    assert sum(map(len, replaced[2].values())) == len(two), "a record outside the drawn strata meets two criteria"
    candidates += draw(2, population, after, replaced[2], links)

    frame = pd.DataFrame(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")

    accepted = frame[frame.replaces_idx.ne("")]
    problems = []
    for row in accepted.itertuples():
        idx = int(row.replaces_idx)
        if membership.source_row[idx] != str(row.source_row):
            problems.append(f"record {idx}: drawn row {row.source_row}, membership {membership.source_row[idx]}")
        link = links[links.source_row.eq(str(row.source_row))]
        if paragraph_sha1(revised.paragraph[idx]) != link.paragraph_sha1.iloc[0]:
            problems.append(f"record {idx}: the listed paragraph is not the one in input.csv")
    changed = sorted(int(i) for i in np.flatnonzero(first.paragraph.ne(revised.paragraph)))
    if sorted(int(i) for i in accepted.replaces_idx) != changed:
        problems.append("the replaced records differ from the records whose paragraph changed")
    for number in DRAWS:
        rows = frame[frame["draw"].eq(number)]
        print(f"draw {number} (seed {DRAWS[number]['seed']}): {len(rows)} candidates examined, "
              f"{int(rows.replaces_idx.ne('').sum())} accepted; replaced "
              f"{ {s: v for s, v in replaced[number].items()} }")
    print(f"wrote {args.output}")
    print("\n".join(problems) if problems else "the draws reproduce the replacement records of the revised benchmark")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
