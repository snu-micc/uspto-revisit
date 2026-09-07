"""Check manuscript configuration selection independently of input row order."""

import pandas as pd
import pytest

from evaluation.select_best_configurations import select_best_metrics


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_accuracy_selects_high_with_its_own_component_metrics(reverse):
    rows = [
        {"method": "gpt-5.4-low", "reaction_accuracy": 237 / 283,
         "extraction_method": "low", "reactants_precision": 0.970},
        {"method": "gpt-5.4-high", "reaction_accuracy": 237 / 283,
         "extraction_method": "high", "reactants_precision": 0.983},
    ]
    selected = select_best_metrics(pd.DataFrame(rows[::-1] if reverse else rows))
    assert selected.iloc[0]["method"] == "gpt-5.4-high"
    assert selected.iloc[0]["reactants_precision"] == 0.983


def test_higher_accuracy_takes_precedence_over_reasoning_effort():
    metrics = pd.DataFrame([
        {"method": "original_data", "reaction_accuracy": 0.5, "extraction_method": "Original data"},
        {"method": "gpt-5.4-low", "reaction_accuracy": 0.9, "extraction_method": "low"},
        {"method": "gpt-5.4-high", "reaction_accuracy": 0.8, "extraction_method": "high"},
    ])
    selected = select_best_metrics(metrics)
    assert selected["method"].tolist() == ["original_data", "gpt-5.4-low"]
