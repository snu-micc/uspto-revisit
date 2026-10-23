from pathlib import Path

import pandas as pd
import pytest

from evaluation import draw_replacements as D

ROOT = Path(__file__).resolve().parents[1]
POPULATION = ROOT / "evaluation" / "data" / "grants_issue.zip"


def test_listed_paragraphs_are_the_benchmark_paragraphs():
    links = pd.read_csv(D.REVISED / "replacement_draws.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    revised = D.read(D.REVISED / "input.csv")
    assert links.groupby("round").size().to_dict() == {"1": 2, "2": 28}
    for link in links.itertuples():
        assert D.paragraph_sha1(revised.paragraph[int(link.replaces_idx)]) == link.paragraph_sha1
        assert int(link.paragraph_length) > 0 and link.text_file


def test_trimmed_drops_the_cxsmiles_extension():
    assert D.trimmed(" CC>>CO |f:0.1| ") == "CC>>CO"
    assert D.trimmed("") == ""


@pytest.mark.skipif(not POPULATION.is_file(), reason="needs evaluation/data/grants_issue.zip (release data-v1.0.0)")
def test_draws_reproduce_the_replacement_records(tmp_path):
    assert D.main(["--population", str(POPULATION), "--output", str(tmp_path / "candidates.csv")]) == 0
