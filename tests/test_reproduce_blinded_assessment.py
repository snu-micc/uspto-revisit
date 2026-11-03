"""The released independent blinded assessment (SI S1, Table S1) is regenerated from its inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import evaluate_table3 as T3
from analysis import reproduce_blinded_assessment as B

RELEASED = Path(__file__).resolve().parents[1] / "analysis" / "results" / "blinded_assessment"


def test_released_assessment_is_regenerated(tmp_path):
    if not (B.RESULTS / "table2_flags_lenient.csv").is_file():
        pytest.skip("run python analysis/reproduce_manuscript.py first (its row-level flags are not committed)")
    out = tmp_path / "assessment"
    try:
        B.run(B.DEFAULT, out)
    finally:
        T3.configure_matching()  # run() leaves strict matching configured
    for released in sorted(RELEASED.iterdir()):
        produced = (out / released.name).read_bytes().replace(b"\r\n", b"\n")
        assert produced == released.read_bytes().replace(b"\r\n", b"\n"), released.name
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert (summary["label_agree"], summary["noise_free"]) == (46, 33)
    assert round(summary["label_cohen_kappa"], 2) == 0.81
    assert summary["reference_agreement"]["lenient"]["both_sets_exact"] == 25
    assert summary["reference_agreement"]["strict"]["both_sets_exact"] == 24
