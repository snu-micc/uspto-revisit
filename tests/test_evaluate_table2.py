import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "evaluate_table2.py"
SPEC = importlib.util.spec_from_file_location("evaluate_table2", MODULE_PATH)
evaluate_table2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate_table2)


def test_publication_table_reports_percentages_to_four_decimal_places():
    metrics = pd.DataFrame(
        [
            {
                "method": "example",
                "extraction_method": "Example model",
                "reactants_precision": 2 / 3,
                "reactants_recall": 1.0,
                "reagents_precision": 0.5,
                "reagents_recall": 0.25,
                "products_precision": 0.75,
                "products_recall": 0.125,
                "reaction_accuracy": 0.75,
                "reaction_accuracy_ci_lower": 0.5,
                "reaction_accuracy_ci_upper": 0.9,
            }
        ]
    )

    table = evaluate_table2.publication_table(metrics, ("example",))

    assert table.iloc[0, 1] == "66.6667"
    assert table.iloc[0, 2] == "100.0000"
    assert table.iloc[0, 7] == "75.0000 (50.0000–90.0000)"
    assert table.columns[7][1] == "Accuracy (%, 95% Wilson CI)"
