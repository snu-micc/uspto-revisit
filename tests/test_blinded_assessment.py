"""Keep the stored blinded evaluation aligned with the numbers reported in the SI."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "analysis/results/blinded_assessment"


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_blinded_input_covers_every_noise_free_reference():
    data = read(ROOT / "analysis/data/blinded_assessment/labels_and_structures.csv")
    assert set(data[0]) == {"benchmark_id", "label", "reactants", "reagents", "products"}
    expected = {r["benchmark_id"] for r in data if r["label"] == "noise-free"}
    assert len(data) == 50 and len(expected) == 33 and {"163", "174"} <= expected


def test_stored_comparison_has_the_reported_counts():
    rows = {(r["scoring"], r["reference"]): r for r in read(RESULTS / "model_comparison.csv")}
    assert len(rows) == 4 and {int(r["n"]) for r in rows.values()} == {33}
    reported = {  # gpt-5.4 correct, USPTO correct
        ("lenient", "evaluator"): (21, 13),
        ("lenient", "benchmark"): (22, 15),
        ("strict", "evaluator"): (20, 13),
        ("strict", "benchmark"): (21, 15),
    }
    for key, counts in reported.items():
        assert (int(rows[key]["gpt-5.4_correct"]), int(rows[key]["USPTO_correct"])) == counts
    lenient, strict = rows["lenient", "evaluator"], rows["strict", "evaluator"]
    assert (int(lenient["gpt-5.4_only_correct"]), int(lenient["USPTO_only_correct"])) == (9, 1)
    assert (int(strict["gpt-5.4_only_correct"]), int(strict["USPTO_only_correct"])) == (8, 1)
    assert round(float(lenient["p_exact"]), 3) == 0.021
    assert round(float(strict["p_exact"]), 3) == 0.039


def test_stored_summary_has_the_reported_agreement():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert (summary["labels"], summary["label_agree"], summary["noise_free"]) == (50, 46, 33)
    assert round(summary["label_cohen_kappa"], 2) == 0.81
    assert [round(100 * bound, 1) for bound in summary["label_agreement_wilson95"]] == [81.2, 96.8]
    for scoring, both in (("lenient", 25), ("strict", 24)):
        assert summary["reference_agreement"][scoring]["both_sets_exact"] == both
        assert summary["reference_agreement"][scoring]["product_set_exact"] == 33
