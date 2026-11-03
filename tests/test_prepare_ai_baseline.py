"""Regression tests for the Ai ORD-JSON baseline conversion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.evaluate_table2 import AI_METHOD, merge_ai_components
from analysis.prepare_ai_baseline import (
    CURRENT_INPUT,
    ai_spellings,
    automatic,
    automatic_resolutions,
    build_parser,
    current_benchmark_records,
    extract_ord_names,
    parse_ord_prediction,
)

ROOT = Path(__file__).resolve().parents[1]
AI_OUTPUT = ROOT / "benchmarks" / "ai_et_al" / "output.json"
REQUERY_SCRIPT = ROOT / "scripts" / "name_resolution" / "requery_curated_names.py"


def _current_records():
    return current_benchmark_records(AI_OUTPUT, ROOT / CURRENT_INPUT)


def test_current_predictions_align_exactly_once():
    rows = _current_records()
    current = pd.read_csv(ROOT / CURRENT_INPUT, encoding='utf-8-sig').fillna('')
    assert [r['source_row'] for r in rows] == list(range(400))
    assert [r['paragraph'] for r in rows] == current.paragraph.tolist()


def test_current_predictions_reject_unmatched_or_duplicate_paragraphs(tmp_path):
    rows = json.loads(AI_OUTPUT.read_text(encoding='utf-8'))
    rows[1] = rows[0]
    path = tmp_path / 'duplicate.json'
    path.write_text(json.dumps(rows), encoding='utf-8')
    with pytest.raises(ValueError, match='could not be matched'):
        current_benchmark_records(path, ROOT / CURRENT_INPUT)


def _planned_urls(spellings):
    spec = importlib.util.spec_from_file_location("requery_curated_names", REQUERY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    planner = module.PlannedRequests()
    planner.resolve(spellings, order="two-pass")
    return sorted(planner.missing_urls)


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


def _args(tmp_path, *extra):
    return build_parser().parse_args(
        ["--output", str(tmp_path / "components.csv"),
         "--resolution-output", str(tmp_path / "resolution.csv"), *extra]
    )


def test_automatic_regenerates_released_files(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    # automatic() returns 0 only if both files match the released ones.
    assert automatic(_args(tmp_path)) == 0
    resolution = pd.read_csv(tmp_path / "resolution.csv", encoding="utf-8-sig").fillna("")
    assert not resolution.resolution_source.str.contains("Review|Cache|Conflict").any()
    assert int(resolution.is_nosmi.sum()) == 228








def test_automatic_resolution_answers_only_from_the_recording(tmp_path):
    spellings = ["ethanol", "unlisted compound xyz"]
    urls = _planned_urls(spellings)
    log = tmp_path / "requery_test_raw.jsonl"
    with log.open("w", encoding="utf-8") as handle:
        for url in urls:
            hit = "opsin" in url and "/ethanol." in url
            handle.write(json.dumps({"url": url, "status": 200 if hit else 404, "body": "CCO" if hit else ""}) + "\n")

    resolved = automatic_resolutions(spellings, log)

    assert resolved["ethanol"][0] == "CCO"
    assert resolved["ethanol"][1].startswith("OPSIN")
    assert resolved["unlisted compound xyz"] == (None, "not_found")

    # The retry path always asks PubChem; without that answer the replay must refuse.
    kept = [line for line in log.read_text(encoding="utf-8").splitlines()
            if not ("pubchem" in line and "unlisted" in line)]
    log.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    with pytest.raises(ValueError, match="not in"):
        automatic_resolutions(spellings, log)
