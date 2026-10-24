import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import benchmark_statistics as BS
from evaluation import evaluate_table2
from evaluation import reproduce_revised as R

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation" / "data"
RESULTS = ROOT / "evaluation" / "results"


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


@pytest.fixture(scope="module")
def first_release_flags():
    flags2 = pd.read_csv(RESULTS / "table2" / "table2_flags.csv", encoding="utf-8-sig", low_memory=False)
    flags2 = flags2[flags2["method"].isin((evaluate_table2.ORIGINAL_METHOD, *R.MODELS))]
    flags3 = pd.read_csv(RESULTS / "table3" / "table3_flags.csv", encoding="utf-8-sig", low_memory=False)
    membership = pd.read_csv(DATA / "benchmark_sample_membership.csv", encoding="utf-8-sig")
    strata = membership.set_index("idx")["issue_label"].sort_index()
    return BS.RecordFlags(flags2, flags3, strata, R.MODELS), flags3


FIRST = BS.FIRST_RELEASE_PRIMARY


def test_record_flags_reproduce_first_release_headline(first_release_flags):
    flags, _ = first_release_flags
    assert int(flags.correct_nf()[FIRST].sum()) == 245
    assert int(flags.denoise(R.TEMPLATE)[FIRST].sum()) == 299
    assert int(flags.e2e(R.TEMPLATE)[FIRST].sum()) == 272
    tests = BS.table2_tests(flags)
    assert set(tests["best"]) == {FIRST}
    assert len(tests) == 19


def test_population_estimates_on_first_release_flags(first_release_flags):
    flags, _ = first_release_flags
    strata = flags.strata.to_numpy()
    e2e = BS.stratified_total(flags.e2e_tp(R.TEMPLATE)[FIRST], strata, BS.FRAME_SINGLE_LABEL)
    tp = BS.stratified_total(flags.denoise_tp(R.TEMPLATE)[FIRST], strata, BS.FRAME_SINGLE_LABEL)
    tp_paper = BS.stratified_total(flags.denoise_tp(R.TEMPLATE)[FIRST], strata, BS.FRAME_EXACTLY_ONE)
    assert round(e2e["total"]) == 276344
    assert round(tp["total"]) == 298706
    assert round(tp_paper["total"]) == 328986
    assert e2e["ci_low"] < e2e["total"] < e2e["ci_high"]


def test_threshold_table_reproduces_first_release(first_release_flags):
    _, flags3 = first_release_flags
    audit = pd.read_csv(DATA / "template_threshold_audit.csv", encoding="utf-8-sig")
    raw, formatted = R.threshold_tables(flags3, audit.set_index("source_row")["template_frequency"], FIRST)
    published = pd.read_csv(RESULTS / "template_threshold_sensitivity.csv", encoding="utf-8-sig")
    mine = raw[raw["template_anomaly_frequency_threshold"] != "none (primary criterion)"]
    columns = ["passed_tp", "filtered_fn", "passed_fp", "filtered_tn"]
    assert (mine[columns].to_numpy() == published[columns].to_numpy()).all()
    assert formatted.iloc[0]["Denoising accuracy (%, 95% CI)"].startswith("74.8")


REPLACED = [4, 8, 9, 10, 11, 14, 15, 20, 28, 32, 34, 42, 52, 62, 66, 68, 71, 73, 76, 77, 84, 94, 96,
            212, 218, 234, 235, 281, 296, 315]


def test_revised_ground_truth():
    gt = R.load_ground_truth()
    assert len(gt) == 400 and gt.index.tolist() == list(range(400))
    assert gt.label.value_counts().to_dict() == {"noise-free": 292, "noisy": 108}
    assert int(gt.eligible.sum()) == 291 and not (gt.eligible & ~gt.noise_free).any()
    assert gt.index[gt.noise_free & ~gt.eligible].tolist() == [215]
    assert set(gt.noisy_reason[~gt.noise_free]) == set(R.NOISY_REASONS)
    references = gt[[c for c in gt if c.startswith("ground_truth_") and c.endswith("_reaction_smiles")]]
    assert references[gt.eligible].ne("").any(axis=1).all()
    assert references[~gt.noise_free].eq("").all(axis=None)
    assert gt.salt_formation.eq("True").sum() == 34


def test_revised_input_and_membership():
    lowe = pd.read_csv(R.DATA / "input.csv", encoding="utf-8-sig").fillna("")
    assert R.replaced_records(lowe) == REPLACED
    revised = pd.read_csv(R.DATA / "benchmark_sample_membership.csv", encoding="utf-8-sig").set_index("idx")
    first = pd.read_csv(DATA / "benchmark_sample_membership.csv", encoding="utf-8-sig").set_index("idx")
    assert revised.issue_label.equals(first.issue_label)  # every replacement comes from the same stratum
    kept = revised.index.difference(REPLACED)
    audit = pd.read_csv(DATA / "template_threshold_audit.csv", encoding="utf-8-sig").set_index("source_row")
    assert revised.template_frequency.loc[kept].equals(audit.template_frequency.loc[kept])
    assert revised.loc[[15, 315], "template_frequency"].tolist() == [21688, 4265]
    draws = pd.read_csv(R.DATA / "replacement_draws.csv", encoding="utf-8-sig")
    assert sorted(draws.replaces_idx) == REPLACED
    assert (revised.loc[draws.replaces_idx, "source_row"].to_numpy() == draws.source_row.to_numpy()).all()


def test_revised_sampling_frame():
    assert sum(BS.FRAME_SINGLE_CRITERION.values()) == 430428
    assert R.FRAME is BS.FRAME_SINGLE_CRITERION


def test_revised_results_headline():
    summary_path = RESULTS / "revised" / "summary.json"
    if not summary_path.is_file():
        pytest.skip("run python evaluation/reproduce_revised.py first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    primary = summary["fully_automatic"]["primary"]
    assert R.PRIMARY == "gpt-5.4-none"
    assert (primary["reaction_correct"], primary["evaluated_reactions"]) == (211, 291)
    # The selected configuration has the highest extraction accuracy in the primary scenario; with the lookup tables
    # or the curated rules, gemini-3.6-flash (high) is highest.
    assert summary["fully_automatic"]["best_method"] == R.PRIMARY
    assert summary["fully_automatic_with_tables"]["best_method"] == BS.FIRST_RELEASE_PRIMARY
    assert summary["fully_automatic_with_tables"]["primary"]["reaction_correct"] == 213
    assert summary["curated"]["primary"]["reaction_correct"] == 227
    assert summary["salt_formation"]["denoising_correct"] == 285
    assert summary["salt_formation"]["denoising_correct_if_noisy"] == 316
    bounds = pd.read_csv(RESULTS / "revised" / "fully_automatic" / "end_to_end_bounds.csv", encoding="utf-8-sig")
    row = bounds[bounds.model.eq(R.PRIMARY) & bounds.heuristic.eq(R.TEMPLATE) & bounds.scoring.eq("lenient")].iloc[0]
    assert (row.assessable_correct, row.assessable_n, row.lower_correct, row.full_n) == (242, 399, 242, 400)
    tests = pd.read_csv(RESULTS / "revised" / "fully_automatic" / "extraction_paired_tests_primary_lenient.csv",
                        encoding="utf-8-sig")
    assert set(tests.best) == {R.PRIMARY}
    assert int((tests[tests.family.eq("T2")].p_holm >= 0.05).sum()) == 11
    population = pd.read_csv(RESULTS / "revised" / "fully_automatic" / "population_bounds.csv", encoding="utf-8-sig")
    exact = population[population.quantity.eq("exact_retained_lower") & population.scoring.eq("lenient")].iloc[0]
    assert (round(exact.total), exact.frame_n) == (223176, 430428)
