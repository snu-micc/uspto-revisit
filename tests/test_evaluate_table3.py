import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "evaluate_table3.py"
SPEC = importlib.util.spec_from_file_location("evaluate_table3", MODULE_PATH)
evaluate_table3 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate_table3)


def test_mapped_left_classifies_reagents_by_product_atom_maps():
    result = evaluate_table3.evaluate_mapped_step(
        "[CH3:1][Br:2].[OH2:3].[Na+:4]>>[CH3:1][OH:3]",
        template_missing=False,
        mapping_error="",
    )

    assert result["reactant_count"] == 2
    assert result["reagent_count"] == 1
    assert result["many_reactants"] == 0
    assert result["no_reagent"] == 0


def test_identical_structures_are_detected():
    result = evaluate_table3.evaluate_mapped_step(
        "[CH3:1][CH3:2]>>[CH3:1][CH3:2]",
        template_missing=False,
        mapping_error="",
    )

    assert result["identical_structures"] == 1


def test_missing_product_atom_maps_cause_template_failure():
    result = evaluate_table3.evaluate_mapped_step(
        "[CH3:1]>>[CH3:1][OH:2]",
        template_missing=False,
        mapping_error="",
    )

    assert result["missing_product_atom_maps"] == 1
    assert result["template_failure"] == 1


def test_singleton_issue_marks_template_anomaly():
    model = "model-a"
    row = pd.Series(
        {
            f"{model}_rxn": '["C>>C"]',
            f"{model}_mapped_rxn": '["[CH3:1]>>[CH3:1]"]',
            f"{model}_mapping_template": '["template"]',
            f"{model}_mapping_error": "[]",
        }
    )

    result = evaluate_table3.evaluate_model_heuristics(
        row,
        model=model,
        issue_rare=True,
        relevant_nosmi=False,
    )

    assert result["rare_template"] == 1


def test_ground_truth_boolean_defines_fixed_noise_label():
    assert evaluate_table3.is_noise_free_paragraph(True)
    assert evaluate_table3.is_noise_free_paragraph("True")
    assert not evaluate_table3.is_noise_free_paragraph(False)


def test_metrics_use_all_four_confusion_matrix_cells():
    flags = pd.DataFrame(
        [
            {"model": "model-a", "noise_free": 1, "rare_template_filtered": 0},
            {"model": "model-a", "noise_free": 1, "rare_template_filtered": 1},
            {"model": "model-a", "noise_free": 0, "rare_template_filtered": 0},
            {"model": "model-a", "noise_free": 0, "rare_template_filtered": 1},
        ]
    )
    for heuristic in ("many_reactants", "many_products", "many_reagents", "no_reagent"):
        flags[f"{heuristic}_filtered"] = flags["rare_template_filtered"]

    metrics = evaluate_table3.build_metrics(flags, ("model-a",))
    row = metrics.loc[metrics["heuristic"].eq("rare_template")].iloc[0]

    assert (
        row["passed_tp"],
        row["filtered_fn"],
        row["passed_fp"],
        row["filtered_tn"],
    ) == (1, 1, 1, 1)
    assert row["denoising_accuracy"] == 0.5


def test_publication_table_reports_percentages_to_four_decimal_places():
    rows = []
    for heuristic in evaluate_table3.HEURISTICS:
        rows.append(
            {
                "model": "model-a",
                "heuristic": heuristic,
                "noise_free_reactions": 3,
                "noisy_reactions": 1,
                "total_reactions": 4,
                "passed_tp": 2,
                "filtered_fn": 1,
                "passed_fp": 0,
                "filtered_tn": 1,
                "noise_free_data_saved_rate": 2 / 3,
                "noisy_data_filtered_rate": 1.0,
                "denoising_accuracy": 0.75,
            }
        )

    table = evaluate_table3.publication_table(pd.DataFrame(rows), "model-a")

    assert table.iloc[0, 5] == "66.6667"
    assert table.iloc[0, 6] == "100.0000"
    assert table.iloc[0, 7] == "75.0000"
    assert table.columns[7][1] == "Denoising accuracy (%)"
