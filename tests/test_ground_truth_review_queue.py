import json

import pandas as pd
import pytest

from analysis import build_ground_truth_review as builder
from analysis import evaluate_table3 as scorer


def test_unanimous_candidates_stay_pending_and_saved_decisions_survive(tmp_path):
    models = tuple(f"model-{n}" for n in range(5))
    mapped = "[CH3:1][Br:2].[OH:3]>>[CH3:1][OH:3]"
    for model in models:
        row = {
            "title": "Methanol", "paragraph": "Test paragraph", "issue": "no_reagent",
            f"{model}_rxn": '["CBr.O>>CO"]',
            f"{model}_mapped_rxn": json.dumps([mapped]),
            f"{model}_mapping_error": "[]",
        }
        pd.DataFrame([row] * 3).to_csv(tmp_path / f"{model}_reaction_smiles_final.csv", index=False)
    saved = tmp_path / "saved.csv"
    pd.DataFrame([
        {"idx": 1, "ground_truth_valid": "False", "review_note": "Author excluded"},
        {"idx": 2, "ground_truth_valid": "True", "ground_truth_products_smiles": '["CCO"]'},
    ]).to_csv(saved, index=False)
    queue = builder.build_queue(
        tmp_path, tmp_path / "queue.csv", models, tmp_path / "no_input.csv", saved,
        include_review_metadata=True,
    ).set_index("idx")
    assert queue.model_agreement_count.eq(5).all()
    assert queue.high_agreement_candidate.eq(1).all()
    assert queue.loc[0, "manual_review_required"] == 1
    assert queue.loc[0, "ground_truth_valid"] == ""
    assert queue.loc[0, "ground_truth_reaction_smiles"] == ""
    assert queue.loc[1, "manual_review_required"] == 0
    assert str(queue.loc[1, "ground_truth_valid"]).lower() == "false"
    assert queue.loc[1, "review_note"] == "Author excluded"
    assert queue.loc[2, "ground_truth_products_smiles"] == '["CCO"]'
    assert queue.loc[2, "manual_review_required"] == 0
    compact = builder.build_queue(
        tmp_path, tmp_path / "compact.csv", models, tmp_path / "no_input.csv", saved,
    ).set_index("idx")
    assert not set(builder.REVIEW_METADATA_COLUMNS) & set(compact.columns)
    pd.testing.assert_frame_equal(
        compact, queue.drop(columns=list(builder.REVIEW_METADATA_COLUMNS)),
    )


def test_scorer_rejects_missing_reference_even_when_candidate_is_present():
    review = pd.DataFrame([{
        "idx": 0, "ground_truth_valid": True,
        "candidate_reactants_smiles": '["CBr", "O"]',
        "candidate_reagents_smiles": "[]", "candidate_products_smiles": '["CO"]',
        "model_agreement_count": 19,
    }])
    with pytest.raises(ValueError, match="no valid ground truth"):
        scorer.build_ground_truth_manifest(review)
