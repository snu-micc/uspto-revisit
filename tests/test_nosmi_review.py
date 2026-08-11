import json

import pandas as pd
import pytest

from uspto_revisit.nosmi_review import (
    ModelSpec,
    apply_review,
    build_review_queue,
    validate_review_rows,
)


def _prediction(name: str) -> str:
    return json.dumps(
        {
            "Reactants, Solvents, Catalysts": {"A": name},
            "Products": {"B": "product"},
            "Reaction Steps": {"1 (Reaction)": "A->B"},
        }
    )


def _write_model(tmp_path, prefix, names):
    dictionaries = [
        {"A": f"[{name} (NoSmi)]", "B": "CCO"} for name in names
    ]
    frame = pd.DataFrame(
        {
            "title": [f"title {index}" for index in range(len(names))],
            "paragraph": [f"paragraph {index}" for index in range(len(names))],
            "prediction": [_prediction(name) for name in names],
            f"{prefix}_smiles": dictionaries,
            f"{prefix}_skeleton": [["A>>B"] for _ in names],
            f"{prefix}_rxn": [
                [f"[{name} (NoSmi)]>>CCO"] for name in names
            ],
            f"{prefix}_mapped_rxn": [["old"] for _ in names],
            f"{prefix}_mapping_error": [[""] for _ in names],
        }
    )
    csv_path = tmp_path / f"{prefix}_reaction_smiles_final.csv"
    dictionary_path = (
        tmp_path / "smiles_batches" / prefix / "smiles_dict_final.json"
    )
    dictionary_path.parent.mkdir(parents=True)
    frame.to_csv(csv_path, index=False)
    dictionary_path.write_text(json.dumps(dictionaries), encoding="utf-8")
    return ModelSpec(prefix, csv_path, dictionary_path)


def test_review_queue_is_scoped_by_source_row(tmp_path):
    first = _write_model(tmp_path, "model-a", ["compound 3", "compound 3"])
    second = _write_model(tmp_path, "model-b", ["compound 3", "different name"])

    review = build_review_queue([first, second], audit_path=None)

    compound_rows = review[review["normalized_name"] == "compound3"]
    assert len(compound_rows) == 2
    assert set(compound_rows["row_zero_based"]) == {0, 1}
    assert review.iloc[0]["occurrences"] == 2
    row_zero = compound_rows[compound_rows["row_zero_based"] == 0].iloc[0]
    assert row_zero["models"] == "model-a:1; model-b:1"
    assert row_zero["occurrences"] == 2


def test_review_queue_preserves_previous_human_fields(tmp_path):
    spec = _write_model(tmp_path, "model-a", ["unknown material"])
    original = build_review_queue([spec], audit_path=None)
    original.loc[0, "review_decision"] = "keep_unresolved"
    original.loc[0, "review_note"] = "No unique molecular structure."
    existing = tmp_path / "review.csv"
    original.to_csv(existing, index=False)

    refreshed = build_review_queue(
        [spec],
        audit_path=None,
        existing_review_path=existing,
    )

    assert refreshed.loc[0, "review_decision"] == "keep_unresolved"
    assert "reviewer" not in refreshed.columns


def test_validate_review_rejects_invalid_smiles():
    review = pd.DataFrame(
        [
            {
                "row_zero_based": 0,
                "name": "compound",
                "normalized_name": "compound",
                "review_decision": "use_smiles",
                "reviewed_smiles": "not-smiles",
                "evidence": "patent example",
                "review_note": "",
            }
        ]
    )

    with pytest.raises(ValueError, match="not valid SMILES"):
        validate_review_rows(review)


def test_apply_review_changes_only_the_accepted_row(tmp_path):
    spec = _write_model(tmp_path, "model-a", ["compound 3", "compound 3"])
    review = build_review_queue([spec], audit_path=None)
    review.loc[review["row_zero_based"] == 0, "review_decision"] = "use_smiles"
    review.loc[review["row_zero_based"] == 0, "reviewed_smiles"] = "C(C)O"
    review.loc[review["row_zero_based"] == 0, "evidence"] = "patent row 0"
    review.loc[review["row_zero_based"] == 1, "review_decision"] = "keep_unresolved"
    review.loc[review["row_zero_based"] == 1, "review_note"] = "Structure unavailable."
    review_path = tmp_path / "review.csv"
    review.to_csv(review_path, index=False)

    output_dir = tmp_path / "reviewed"
    summary = apply_review(
        [spec],
        review_path,
        output_dir=output_dir,
        summary_path=output_dir / "summary.json",
    )

    dictionary_path = (
        output_dir / "smiles_batches" / "model-a" / "smiles_dict_final.json"
    )
    dictionaries = json.loads(dictionary_path.read_text(encoding="utf-8-sig"))
    assert dictionaries[0]["A"] == "CCO"
    assert dictionaries[1]["A"] == "[compound 3 (NoSmi)]"
    result = pd.read_csv(output_dir / "model-a_reaction_smiles_final.csv")
    assert "(NoSmi)" not in result.loc[0, "model-a_rxn"]
    assert "(NoSmi)" in result.loc[1, "model-a_rxn"]
    assert "rerun atom mapping" in result.loc[0, "model-a_mapping_error"]
    assert summary["models"]["model-a"]["changed_rows"] == 1


def test_apply_review_excludes_confirmed_nonmapping_component(tmp_path):
    spec = _write_model(tmp_path, "model-a", ["CDA"])
    review = build_review_queue([spec], audit_path=None)
    review.loc[0, "review_decision"] = "exclude_from_mapping"
    review.loc[0, "review_note"] = "Catalyst/support; excluded from mapping."
    review_path = tmp_path / "review.csv"
    review.to_csv(review_path, index=False)

    output_dir = tmp_path / "reviewed"
    summary = apply_review(
        [spec],
        review_path,
        output_dir=output_dir,
        summary_path=output_dir / "summary.json",
    )

    result = pd.read_csv(output_dir / "model-a_reaction_smiles_final.csv")
    dictionaries = json.loads(
        (
            output_dir / "smiles_batches" / "model-a" / "smiles_dict_final.json"
        ).read_text(encoding="utf-8-sig")
    )
    assert dictionaries[0]["A"] == "[CDA (NoSmi)]"
    assert "(NoSmi)" not in result.loc[0, "model-a_rxn"]
    assert "A" not in result.loc[0, "model-a_skeleton"]
    assert summary["models"]["model-a"]["excluded_components"] == 1
