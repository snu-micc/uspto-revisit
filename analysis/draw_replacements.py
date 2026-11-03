"""Re-run replacement draws 1 and 2 of the benchmark and check draw 3.

Draw 1 (seed 20261021) replaced records 15 and 315 of the first release, which repeated reactions already sampled as
records 8 and 309 (other patents of the same families). Draw 2 (seed 20261022) replaced the 28 records that meet two
Table 1 criteria when the criteria are counted explicitly (``table1_criteria``). Both mirror the original sampling:

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

R1 and R2 are checked against the population file (``grants_issue.zip``, see README.md). R3 to R5
need Lowe's CML files, which are not part of this repository: ``analysis/data/replacement_draws.csv`` lists, for
every replacement record, the Lowe file and the SHA-1 and length of the whitespace-normalized paragraph it is linked
to. Draw 2 also accepted a record for 62 that draw 3 later replaced; its link is ``SUPERSEDED`` below. The script
stops if a draw examines a candidate that is not listed.

The earlier benchmark states are in ``analysis/data/replacement_history.csv``, one row per record: the first
release (``first_*``), the paragraph SHA-1 before draw 3 and the final population and candidate rows. The final
reaction SMILES and paragraphs are those of ``input.csv``. The script checks that the accepted rows are the final
population rows of the replacement records (except record 62), that each listed paragraph is the one before draw 3,
and that the replaced records are exactly those whose paragraph differs from the first release.

Draw 3 (seed 20261026) replaced seven records that fail the single-criterion audit. It needs Lowe's CML files and is
made by ``prepare_lowe_replacements.py``; this script checks its listed records against the final benchmark.

    python analysis/draw_replacements.py --population analysis/data/grants_issue.zip
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import re
import sys
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "data"
HISTORY = DATA / "replacement_history.csv"
LABEL = {  # stratum -> its label in the released Issue field
    "many_products": "many_product (> 1)",
    "many_reactants": "many_reactant (> 2)",
    "many_reagents": "many_reagent (> 5)",
    "no_reagent": "no_reagent",
}
OTHER_LABELS = ("rare_temp", "duplicated")
ATOM_MAP = re.compile(r":(\d+)\]")
DUPLICATES = {15: 8, 315: 309}  # record replaced in draw 1 -> the earlier record with the same reaction
DRAWS = {
    1: {"seed": 20261021, "streams": {"many_products": 0, "no_reagent": 1}},
    2: {"seed": 20261022, "streams": {"many_products": 0, "many_reagents": 1}},
}
# Record 62's draw-2 replacement, replaced again in draw 3; replacement_draws.csv keeps only the final records.
SUPERSEDED = pd.DataFrame([{
    "round": "2", "seed": "20261022", "stratum": "many_products", "rank": "13", "source_row": "174680",
    "patent_number": "US06326386B1", "replaces_idx": "62", "text_file": "grants/2001/20011204.xml",
    "paragraph_sha1": "c4958825cbbf3f9f898b0789b236220a5c84725d", "paragraph_length": "535",
}])
SPACE = re.compile(r"\s+")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def trimmed(smiles: str) -> str:
    """The reaction SMILES without the CXSMILES extension, as the draws compared reactions."""
    return str(smiles).strip().split()[0] if str(smiles).strip() else ""


def paragraph_sha1(text: str) -> str:
    return hashlib.sha1(SPACE.sub(" ", str(text)).strip().encode("utf-8")).hexdigest()


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


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


@contextmanager
def open_population(path: Path, member: str | None) -> Iterator[TextIO]:
    """grants_issue.csv, or the release archive that contains it."""
    if path.suffix.lower() != ".zip":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield handle
        return
    with zipfile.ZipFile(path) as archive:
        if member is None:
            members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            named = [n for n in members if Path(n).name.lower() == "grants_issue.csv"]
            if len(named) != 1 and len(members) != 1:
                raise SystemExit(f"{path}: found CSV members {members}; use --population-member")
            member = named[0] if len(named) == 1 else members[0]
        with archive.open(member) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as handle:
            yield handle


def read_population(path: Path, member: str | None) -> pd.DataFrame:
    """Patent, reaction, released Issue and single-criterion stratum of every population row."""
    rows = []
    with open_population(path, member) as handle:
        cache: dict[str, tuple[str, ...]] = {}
        for row in csv.DictReader(handle):
            reaction, issue = row.get("ReactionSmiles", ""), row.get("Issue", "")
            criteria = table1_criteria(reaction, row.get("Reagent", ""))
            rows.append((row.get("PatentNumber", ""), trimmed(reaction), issue,
                         stratum_of_criteria(criteria, parse_issue(issue, cache)) or ""))
    return pd.DataFrame(rows, columns=["patent_number", "reaction", "issue", "criteria_stratum"])


def benchmarks() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """The first-release benchmark, the paragraph SHA-1s before draw 3 and the final benchmark, by record."""
    history, final = read(HISTORY), read(DATA / "input.csv")
    if not history.idx.equals(final.idx):
        raise SystemExit(f"{HISTORY.name} and input.csv list different records")
    first = pd.DataFrame({
        "stratum": final.stratum, "Lowe_smiles": history.first_lowe_smiles,
        "candidate_source_rows": history.first_candidate_source_rows,
        "paragraph_sha1": history.first_paragraph_sha1, "source_row": history.first_source_row,
    })
    final = final.assign(paragraph_sha1=final.paragraph.map(paragraph_sha1), source_row=history.source_row,
                         candidate_source_rows=history.candidate_source_rows)[first.columns]
    return first, history.before_draw3_paragraph_sha1, final


def candidate_rows(value: str) -> list[int]:
    return [int(r) for r in str(value).split(";") if r.isdigit()]


def draw(number: int, population: pd.DataFrame, benchmark: pd.DataFrame, replaced: dict[str, list[int]],
         links: pd.DataFrame) -> list[dict]:
    """Candidates examined in one draw, in order, with their rejection reasons and the record each one replaces."""
    spec = DRAWS[number]
    reactions = {trimmed(s) for s in benchmark.Lowe_smiles}
    rows_sampled = {r for value in benchmark.candidate_source_rows for r in candidate_rows(value)}
    texts = set(benchmark.paragraph_sha1)
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
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "results" /
                        "replacement_draw_candidates.csv")
    args = parser.parse_args(argv)

    population = read_population(args.population, args.population_member)
    first, before, final = benchmarks()
    final_links = read(DATA / "replacement_draws.csv")
    links = pd.concat([final_links, SUPERSEDED], ignore_index=True)
    third = final_links[final_links["round"].eq("3")]

    # Draw 1: the second copy of each duplicated reaction.
    for idx, earlier in DUPLICATES.items():
        assert trimmed(first.Lowe_smiles[idx]) == trimmed(first.Lowe_smiles[earlier]), idx
    replaced = {1: {s: sorted(i for i in DUPLICATES if first.stratum[i] == s) for s in DRAWS[1]["streams"]}}
    candidates = draw(1, population, first, replaced[1], links)

    # Draw 2: the benchmark after draw 1; records whose population row meets more than its own criterion.
    after = first.copy()
    for idx in DUPLICATES:
        after.loc[idx] = final.loc[idx, after.columns].values
    two = [i for i, row in enumerate(after.source_row.astype(int))
           if population.criteria_stratum.iat[row] != after.stratum[i]]
    replaced[2] = {s: sorted(i for i in two if after.stratum[i] == s) for s in DRAWS[2]["streams"]}
    assert sum(map(len, replaced[2].values())) == len(two), "a record outside the drawn strata meets two criteria"
    candidates += draw(2, population, after, replaced[2], links)

    frame = pd.DataFrame(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")

    accepted = frame[frame.replaces_idx.ne("")]
    redrawn = {int(i) for i in third.replaces_idx}
    problems = []
    for row in accepted.itertuples():
        idx = int(row.replaces_idx)
        if idx not in redrawn and final.source_row[idx] != str(row.source_row):
            problems.append(f"record {idx}: drawn row {row.source_row}, final row {final.source_row[idx]}")
        link = links[links.source_row.eq(str(row.source_row))]
        if before[idx] != link.paragraph_sha1.iloc[0]:
            problems.append(f"record {idx}: the listed paragraph is not the one before draw 3")
    changed = sorted(int(i) for i in np.flatnonzero(first.paragraph_sha1.ne(before)))
    if sorted(int(i) for i in accepted.replaces_idx) != changed:
        problems.append("the records replaced in draws 1 and 2 differ from the records whose paragraph changed")

    # Draw 3: the listed records against the final benchmark.
    if sorted(redrawn) != sorted(int(i) for i in np.flatnonzero(before.ne(final.paragraph_sha1))):
        problems.append("the records listed for draw 3 differ from the records whose paragraph changed in it")
    for link in third.itertuples():
        idx = int(link.replaces_idx)
        if final.source_row[idx] != link.source_row:
            problems.append(f"record {idx}: listed row {link.source_row}, final row {final.source_row[idx]}")
        if final.paragraph_sha1[idx] != link.paragraph_sha1:
            problems.append(f"record {idx}: the listed paragraph is not the one in input.csv")
    for number in DRAWS:
        rows = frame[frame["draw"].eq(number)]
        print(f"draw {number} (seed {DRAWS[number]['seed']}): {len(rows)} candidates examined, "
              f"{int(rows.replaces_idx.ne('').sum())} accepted; replaced "
              f"{ {s: v for s, v in replaced[number].items()} }")
    print(f"draw 3 (seed 20261026, not re-run): {len(third)} listed records; replaced {sorted(redrawn)}")
    print(f"wrote {args.output}")
    print("\n".join(problems) if problems else "the draws reproduce the replacement records of the benchmark")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
