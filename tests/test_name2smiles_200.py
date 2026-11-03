"""Recompute both generations from predictions, without historical match flags."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/name2smiles_200"
spec = importlib.util.spec_from_file_location("name2smiles_score", DATA / "score.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def test_both_stored_generations_are_reproduced(tmp_path):
    assert score.main(["--output", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n"] == 200
    assert summary["original"]["correct"] == 82
    assert summary["rerun"]["correct"] == 90
    assert summary["rerun"]["failure_counts"]["invalid_smiles"] == 22
    assert summary["rerun"]["failure_counts"]["null_or_empty_prediction"] == 2
    for filename in ("comparison.csv", "summary.json"):
        assert (tmp_path / filename).read_bytes() == (DATA / filename).read_bytes()


def test_scoring_uses_structure_matches_and_rejects_failed_outputs():
    reference = score.canonical("CCO")
    for record, reason in [
        ({}, "missing_record"),
        ({"smiles": None}, "null_or_empty_prediction"),
        ({"smiles": "invalid"}, "invalid_smiles"),
        ({"smiles": "CCO", "error": "incomplete response"}, "generation_error"),
        ({"smiles": "CCN"}, "canonical_structure_mismatch"),
    ]:
        result = score.score_prediction(record, reference)
        assert not result["counted_as_correct"]
        assert result["scoring_reason"] == reason
    matched = score.score_prediction({"smiles": "C(C)O"}, reference)
    assert matched["counted_as_correct"]
    assert matched["scoring_reason"] == "exact_match"


def test_new_generation_accepts_reference_matches_for_previously_flagged_names(tmp_path):
    references = score.load_records(DATA / "references.json")
    records = tmp_path / "records"
    records.mkdir()
    compound_ids = {8, 59, 76, 92, 106}
    for i in compound_ids:
        (records / f"{i:03d}.json").write_text(
            json.dumps({"smiles": references[i]["reference_smiles"]}), encoding="utf-8")
    output = tmp_path / "scored"
    score.main(["--records-dir", str(records), "--output", str(output)])
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["rerun"]["correct"] == len(compound_ids)
    assert summary["rerun"]["failure_counts"]["missing_record"] == 200 - len(compound_ids)
    with (output / "comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = {int(r["compound_id"]): r for r in csv.DictReader(handle)}
    assert all(rows[i]["rerun_scoring_reason"] == "exact_match" for i in compound_ids)


def test_correction_table_matches_fixed_references():
    references = score.load_records(DATA / "references.json")
    with (DATA / "reference_corrections.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len({r["compound_id"] for r in rows}) == 15
    for row in rows:
        ref = references[int(row["compound_id"])]
        assert row["chemical_name"] == ref["name"]
        assert row["reference_smiles_after_correction"] == ref["reference_smiles"]
        assert row["error_in_previous_reference"] and row["reference_correction_explanation"]


def test_new_generation_does_not_fall_back_to_stored_rerun(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    output = tmp_path / "scored"
    score.main(["--records-dir", str(records), "--output", str(output)])
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["original"]["correct"] == 82
    assert summary["rerun"]["correct"] == 0
    assert summary["rerun"]["failure_counts"]["missing_record"] == 200
