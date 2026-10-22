"""Regression tests for the Ai ORD-JSON baseline conversion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.evaluate_table2 import AI_METHOD, merge_ai_components
from evaluation.prepare_ai_baseline import (
    build_parser,
    extract_ord_names,
    parse_ord_prediction,
    replay,
)

ROOT = Path(__file__).resolve().parents[1]


def test_trailing_marker_is_removed_without_repairing_json():
    payload, error = parse_ord_prediction('{"inputs": {}, "outcomes": []}###')
    assert error == ""
    assert payload == {"inputs": {}, "outcomes": []}

    malformed, malformed_error = parse_ord_prediction('{"inputs": {"broken": }###')
    assert malformed is None
    assert malformed_error


def test_ord_roles_are_converted_and_other_roles_are_ignored():
    def component(name, role):
        return {"identifiers": [{"type": "NAME", "value": name}], "reaction_role": role}

    payload = {
        "inputs": {
            "group": {
                "components": [
                    component("A", "REACTANT"),
                    component("B", "SOLVENT"),
                    component("C", "CATALYST"),
                    component("D", "REAGENT"),
                    component("E", "CONDITION"),
                    component("P", "PRODUCT"),
                ]
            }
        },
        "outcomes": [{"products": [{"identifiers": [{"type": "NAME", "value": "P"}]}]}],
    }
    names, ignored = extract_ord_names(payload)
    assert names["reactants"] == ["A"]
    assert names["reagents"] == ["B", "C", "D"]
    assert names["products"] == ["P"]
    assert ignored == ["CONDITION"]


def test_component_merge_validates_and_attaches_ai_columns(tmp_path):
    review = pd.DataFrame([{"idx": 0, "title": "title", "paragraph": "paragraph"}])
    path = tmp_path / "components.csv"
    pd.DataFrame(
        [
            {
                "idx": 0,
                "title": "title",
                "paragraph": "paragraph",
                f"{AI_METHOD}_reactants_smiles": "['CCO']",
                f"{AI_METHOD}_reagents_smiles": "['O']",
                f"{AI_METHOD}_products_smiles": "['CC=O']",
                f"{AI_METHOD}_has_unresolved_nosmi": 0,
            }
        ]
    ).to_csv(path, index=False)

    merged = merge_ai_components(review, path)

    assert merged.loc[0, f"{AI_METHOD}_products_smiles"] == "['CC=O']"


def test_replay_regenerates_released_components(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    args = build_parser().parse_args(
        [
            "--output", str(tmp_path / "components.csv"),
            "--resolution-output", str(tmp_path / "resolution.csv"),
        ]
    )

    # replay() itself returns 0 only if the component file matches the release.
    assert replay(args) == 0
    released = ROOT / "benchmarks/ai_et_al/preprocessing/ai_baseline_name_resolution.csv"
    produced = pd.read_csv(tmp_path / "resolution.csv", encoding="utf-8-sig", dtype=str)
    expected = pd.read_csv(released, encoding="utf-8-sig", dtype=str)
    assert produced.fillna("").equals(expected.fillna(""))
