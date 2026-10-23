import csv

from evaluation import build_sampling_frame


def _write(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_frames_and_membership(tmp_path):
    population = tmp_path / "grants_issue.csv"
    fields = ["Year", "PatentNumber", "ParagraphNum", "ReactionSmiles", "TemplateFreq", "Issue"]
    rows = [
        ("1990", "US1", "CC>>CCO", "3", "['no_reagent']"),
        ("1991", "US2", "CC>>CCO", "3", "['no_reagent']"),
        ("1992", "US3", "CC>>CCN", "1", "['rare_temp', 'no_reagent']"),
        ("1993", "US4", "C>>CO", "7", "['many_product (> 1)']"),
        ("1994", "US5", "C>>CN", "7", "['many_product (> 1)', 'no_reagent']"),
        ("1995", "US6", "C>>CC", "7", "['duplicated']"),
    ]
    _write(
        population,
        [
            dict(zip(fields, (year, patent, "", smiles, freq, issue)))
            for year, patent, smiles, freq, issue in rows
        ],
        fields,
    )
    benchmark = tmp_path / "input.csv"
    _write(
        benchmark,
        [
            {"Issue": "['no_reagent']", "title": "t0", "paragraph": "p0", "Lowe_smiles": "CC>>CCO "},
            {"Issue": "['many_product (> 1)']", "title": "t1", "paragraph": "p1", "Lowe_smiles": "C>>CO"},
        ],
        ["Issue", "title", "paragraph", "Lowe_smiles"],
    )
    weighting = tmp_path / "weights.csv"
    _write(
        weighting,
        [
            {"orderly_sampling_category": name, "population_n": 0, "sample_n": 1}
            for name in ("many_products", "many_reactants", "many_reagents", "no_reagent")
        ],
        ["orderly_sampling_category", "population_n", "sample_n"],
    )
    out = tmp_path / "out"
    status = build_sampling_frame.main(
        [
            "--population", str(population),
            "--input", str(benchmark),
            "--weighting-input", str(weighting),
            "--output-dir", str(out),
            "--provenance-output", str(tmp_path / "provenance.csv"),
        ]
    )
    assert status == 0

    with (out / "sampling_frame_counts.csv").open(encoding="utf-8-sig") as handle:
        counts = {(row["frame"], row["group"]): int(row["n"]) for row in csv.DictReader(handle)}
    assert counts[("table1_any_flag", "no_reagent")] == 4
    assert counts[("table1_any_flag", "union")] == 5
    assert counts[("exactly_one_of_four", "total")] == 4
    assert counts[("single_label_frame", "total")] == 3
    assert counts[("single_label_frame", "no_reagent")] == 2
    # Without atom maps or a Reagent field every row meets only "no reagent"; rare_temp and duplicated rows are out.
    assert counts[("single_criterion_frame", "no_reagent")] == 4
    assert counts[("single_criterion_frame", "total")] == 4

    with (out / "benchmark_sample_membership.csv").open(encoding="utf-8-sig") as handle:
        membership = list(csv.DictReader(handle))
    assert [row["stratum"] for row in membership] == ["no_reagent", "many_products"]
    assert membership[0]["source_row"] == "0"
    assert membership[0]["candidate_source_rows"] == "0;1"
    assert membership[0]["source_row_unique"] == "0"
    assert membership[1]["source_row"] == "3"

    with (out / "weighted_sampling_summary_single_label_frame.csv").open(encoding="utf-8-sig") as handle:
        weights = {row["orderly_sampling_category"]: row["population_n"] for row in csv.DictReader(handle)}
    assert weights["no_reagent"] == "2"


def test_table1_criteria_count_molecules_explicitly():
    criteria = build_sampling_frame.table1_criteria
    # Three molecules share atom maps with a product, and the product side holds two mapped molecules.
    three = "[CH3:1][OH:2].[Cl:3][CH2:4][CH3:5].[NH3:6]>>[CH3:1][O:2][CH2:4][CH2:5][NH2:6].[Na+:7]"
    assert criteria(three, "") == ("many_products", "many_reactants", "no_reagent")
    # An empty agent field adds no reactant, and an empty entry in the Reagent field is not a reagent.
    two = "[CH3:1][OH:2].[Cl:3][CH3:4]>>[CH3:1][O:2][CH3:4] |f:0.1|"
    assert criteria(two, ".CCO") == ()
    assert criteria(two, "O.CCO.C1CCOC1.CN(C)C=O.N.CCN(CC)CC") == ("many_reagents",)
