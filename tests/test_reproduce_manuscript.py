import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import benchmark_statistics as BS
from analysis import evaluate_table2
from analysis import evaluate_table3
from analysis import reproduce_manuscript as R

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "data"
RESULTS = ROOT / "analysis" / "results"


def test_percentages_round_half_up():
    assert R.pct(285, 400) == "71.3"
    assert R.pct(257, 400) == "64.3"
    assert R.pct(299, 400) == "74.8"
    assert R.pct(245, 283) == "86.6"
    assert R.acc_text(222, 283) == "78.4 (73.3–82.8)"


def test_statistics_helpers():
    assert BS.wilson(245, 283) == evaluate_table2.wilson_interval(245, 283)
    assert BS.mcnemar_exact([1, 1, 1, 0], [0, 0, 0, 0]) == (3, 0, 0.25)
    assert BS.mcnemar_exact([1, 0], [1, 0]) == (0, 0, 1.0)
    assert np.allclose(BS.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])












REPLACED = [3, 4, 6, 8, 9, 10, 11, 13, 14, 15, 20, 28, 32, 34, 42, 50, 52, 62, 66, 68, 69, 71, 73, 76, 77, 82, 84, 94, 96,
            212, 218, 234, 235, 281, 296, 315]


def test_ground_truth():
    gt = R.load_ground_truth()
    assert "structure_evaluation_eligible" not in gt and "structure_exclusion_reason" not in gt
    assert "reference_source" not in gt and "component_reference_available" not in gt
    assert len(gt) == 400 and gt.index.tolist() == list(range(400))
    assert gt.label.value_counts().to_dict() == {"noise-free": 292, "noisy": 108}
    assert int(gt.eligible.sum()) == 291 and not (gt.eligible & ~gt.noise_free).any()
    assert gt.index[gt.noise_free & ~gt.eligible].tolist() == [215]
    assert gt.component_reference.equals(gt.noise_free)
    assert set(gt.noisy_reason[~gt.noise_free]) == set(R.NOISY_REASONS)
    references = gt[[c for c in gt if c.startswith("ground_truth_") and c.endswith("_reaction_smiles")]]
    assert references[gt.eligible].ne("").any(axis=1).all()
    assert references[~gt.noise_free].eq("").all(axis=None)
    assert gt.salt_formation.eq("True").sum() == 34


def test_partial_reference_counts_unrepresentable_reagents_as_missed():
    gt = R.load_ground_truth()
    row = gt.loc[215]
    assert row.ground_truth_reaction_smiles == "" and row.component_reference and not row.eligible
    reactants, reagents, products = evaluate_table3.partial_signature_from_row(row)
    missing = [r for r in reagents if r.startswith(evaluate_table3.UNREPRESENTABLE_PREFIX)]
    assert len(missing) == 2 and reactants and products
    molecular = (reactants, tuple(r for r in reagents if r not in missing), products)
    counts = evaluate_table2.compare_signatures(molecular, (reactants, reagents, products))
    assert (counts["reagents_fn"], counts["reagents_fp"], counts["component_exact"]) == (2, 0, 0)
    other = gt.loc[gt.noise_free & gt.eligible].iloc[0]
    with pytest.raises(ValueError, match="partial reference"):
        evaluate_table3.partial_signature_from_row(other)


def test_scoring_rejects_alternative_references():
    gt = R.load_ground_truth()
    assert not any(c.startswith("ground_truth_2_") for c in gt)
    gt.loc[259, "ground_truth_2_reaction_smiles"] = "CCO>>CC=O"
    with pytest.raises(ValueError, match="one reference"):
        R.validate_single_references(gt)


def test_reference_conventions():
    gt = R.load_ground_truth()
    for idx in (259, 298):
        reagents = json.loads(gt.loc[idx, "ground_truth_reagents_smiles"])
        assert "[AlH4-]" in reagents and "[Li+]" in reagents
        assert "[Al+3]" not in reagents
    assert "[Ar]" not in json.loads(gt.loc[268, "ground_truth_reagents_smiles"])
    assert "[OH][Pd][OH]" in json.loads(gt.loc[272, "ground_truth_reagents_smiles"])
    assert len(json.loads(gt.loc[297, "ground_truth_products_smiles"])) == 1


def test_scoring_rejects_missing_references():
    gt = R.load_ground_truth()
    gt.loc[259, "ground_truth_reaction_smiles"] = ""
    with pytest.raises(ValueError, match="complete reference"):
        R.validate_single_references(gt)


def test_input_and_membership():
    lowe = pd.read_csv(R.DATA / "input.csv", encoding="utf-8-sig").fillna("")
    assert R.replaced_records(lowe) == REPLACED
    inputs = pd.read_csv(R.DATA / "input.csv", encoding="utf-8-sig").set_index("idx")
    assert inputs.stratum.value_counts().tolist() == [100, 100, 100, 100]
    audit = pd.read_csv(RESULTS / "lowe_population" / "benchmark_membership.csv", encoding="utf-8-sig").set_index("idx")
    assert inputs.template_frequency.equals(audit.template_frequency)
    assert audit.eligible.all()
    assert inputs.template_frequency.isna().sum() == 39
    assert inputs.raw_source_rows.astype(str).equals(audit.raw_rows.astype(str))
    draws = pd.read_csv(R.DATA / "replacement_draws.csv", encoding="utf-8-sig")
    assert sorted(draws.replaces_idx) == REPLACED
    assert (inputs.loc[draws.replaces_idx, "patent_number"].to_numpy() == draws.patent_number.to_numpy()).all()
    assert (inputs.loc[draws.replaces_idx, "stratum"].to_numpy() == draws.stratum.to_numpy()).all()


def test_sampling_frame():
    summary = json.loads((RESULTS / "lowe_population" / "summary.json").read_text(encoding="utf-8"))
    assert summary["sampling_frame_total"] == 536030
    assert R.FRAME is None  # Retained historical sample cannot support new-frame extrapolation.


def test_results_headline():
    summary_path = RESULTS / "summary.json"
    if not summary_path.is_file():
        pytest.skip("run python analysis/reproduce_manuscript.py first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    primary = summary["fully_automatic"]["primary"]
    assert R.PRIMARY == "gpt-5.4-none"
    assert (primary["reaction_correct"], primary["evaluated_reactions"]) == (209, 292)
    # Model rankings differ between automatic and 46-override scenarios.
    assert R.SCENARIOS == ("fully_automatic", "curated")
    assert summary["fully_automatic"]["best_method"] == R.PRIMARY
    assert summary["curated"]["primary"]["reaction_correct"] == 217
    assert summary["salt_formation"]["denoising_correct"] == 283
    assert summary["salt_formation"]["denoising_correct_if_noisy"] == 314
    bounds = pd.read_csv(RESULTS / "fully_automatic" / "end_to_end_bounds.csv", encoding="utf-8-sig")
    row = bounds[bounds.model.eq(R.PRIMARY) & bounds.heuristic.eq(R.TEMPLATE) & bounds.scoring.eq("lenient")].iloc[0]
    assert (row.assessable_correct, row.assessable_n, row.lower_correct, row.full_n) == (240, 400, 240, 400)
    assert row.upper_correct == row.lower_correct and row.unresolved_retained == 0
    tests = pd.read_csv(RESULTS / "fully_automatic" / "extraction_paired_tests_primary_lenient.csv",
                        encoding="utf-8-sig")
    assert set(tests.best) == {R.PRIMARY}
    assert int((tests[tests.family.eq("T2")].p_holm >= 0.05).sum()) == 13
    assert not (RESULTS / "fully_automatic" / "population_bounds.csv").exists()


def test_ai_baseline_scores_every_noise_free_paragraph():
    gt = R.load_ground_truth()
    lowe = pd.read_csv(R.DATA / "input.csv", encoding="utf-8-sig").fillna("")
    metrics = R.ai_baseline(R.build_review("fully_automatic", gt), gt, lowe).set_index("scoring")
    assert metrics.evaluated_reactions.tolist() == [292, 292]
    assert (metrics.at["exact", "reaction_correct"], metrics.at["coverage", "reaction_correct"]) == (12, 16)
    released = pd.read_csv(RESULTS / "ai_metrics.csv", encoding="utf-8-sig").set_index("scoring")
    assert released.reaction_correct.to_dict() == metrics.reaction_correct.to_dict()


def test_incomplete_smiles_is_failure_with_partial_component_reference():
    gt = pd.DataFrame({"noise_free": [True, True, False], "eligible": [True, False, False],
                       "component_reference": [True, True, False]})
    flags = pd.DataFrame({"source_row": [0, 1, 2], "reaction_correct": [1, "", ""],
                          "evaluation_included": [1, 0, 0], "exclusion_reason": ["", "missing", "noisy"],
                          "reagents_tp": [2, 3, np.nan]})
    result = R.include_processing_failures(flags, gt)
    assert result.evaluation_included.tolist() == [1, 1, 0]
    assert result.component_evaluation_included.tolist() == [1, 1, 0]
    assert result.loc[1, "reaction_correct"] == 0
    assert result.loc[1, "processing_failure"] == 1
    assert result.loc[1, "reagents_tp"] == 3
    excluded = R.include_processing_failures(flags, gt, (1,))
    assert excluded.evaluation_included.tolist() == [1, 0, 0]
