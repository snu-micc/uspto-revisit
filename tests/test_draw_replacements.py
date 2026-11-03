from pathlib import Path

import pandas as pd
import pytest

from analysis import draw_replacements as D

ROOT = Path(__file__).resolve().parents[1]
POPULATION = ROOT / "analysis" / "data" / "grants_issue.zip"


def test_listed_paragraphs_are_the_benchmark_paragraphs():
    links = D.read(D.DATA / "replacement_draws.csv")
    final = D.read(D.DATA / "input.csv")
    assert links.groupby("round").size().to_dict() == {"1": 2, "2": 27, "3": 7}
    for link in links.itertuples():
        assert D.paragraph_sha1(final.paragraph[int(link.replaces_idx)]) == link.paragraph_sha1
        assert int(link.paragraph_length) > 0 and link.text_file


def test_superseded_draw_2_link_is_the_paragraph_before_draw_3():
    first, before, final = D.benchmarks()
    for link in D.SUPERSEDED.itertuples():
        idx = int(link.replaces_idx)
        assert before[idx] == link.paragraph_sha1
        assert before[idx] != final.paragraph_sha1[idx]  # replaced again in draw 3


def test_history_changes_only_the_replaced_records():
    first, before, final = D.benchmarks()
    replaced = set(D.read(D.DATA / "replacement_draws.csv").replaces_idx.astype(int))
    assert len(replaced) == 36
    assert set(first.index[first.paragraph_sha1.ne(final.paragraph_sha1)]) == replaced
    assert set(first.index[first.Lowe_smiles.ne(final.Lowe_smiles)]) <= replaced


def test_trimmed_drops_the_cxsmiles_extension():
    assert D.trimmed(" CC>>CO |f:0.1| ") == "CC>>CO"
    assert D.trimmed("") == ""


def test_table1_criteria_counts_mapped_reactants_and_reagents():
    reaction = "[CH3:1][OH:2].[Na+].[Cl:3][CH3:4]>>[CH3:1][O:2][CH3:4]"
    assert D.table1_criteria(reaction, "") == ("no_reagent",)
    assert D.table1_criteria(reaction, ".".join("O" * 6)) == ("many_reagents",)
    assert D.stratum_of_criteria(("no_reagent",), ("duplicated",)) is None


@pytest.mark.skipif(not POPULATION.is_file(), reason="needs analysis/data/grants_issue.zip (release data-v1.0.0)")
def test_draws_reproduce_the_replacement_records(tmp_path):
    assert D.main(["--population", str(POPULATION), "--output", str(tmp_path / "candidates.csv")]) == 0
