import json

import pandas as pd
import pytest

from evaluation import build_ground_truth_review, curation_ablation, evaluate_table3
from uspto_revisit import reaction_smiles, smiles_fetch


def test_helpers_match_extraction_package():
    names = [
        "Palladium on Carbon",
        "7α-(5-bromopentyl)-11β-fluoro",
        "N,N′-carbonyldiimidazole",
        "<i>tert</i>-butyl &amp; ester",
        "  diphenyl  phosphoryl azide ",
    ]
    for name in names:
        assert curation_ablation.normalize_name_key(name) == smiles_fetch.normalize_name_key(name)
    for smiles in ("OCC", "C[C@H](N)C(=O)O", "[Na+].[Cl-]", "[H]", "Br", "[C]=O", "not-smiles", ""):
        assert curation_ablation.canonicalize_smiles(smiles) == smiles_fetch.canonicalize_smiles(smiles)
    mapping = {"A": "CBr", "B": "[x (NoSmi)]"}
    for code in ("A.B>C", "A>B>C", "A"):
        assert curation_ablation.replace_with_smiles(code, mapping) == reaction_smiles.replace_with_smiles(
            code, mapping
        )


STEP_0 = "[CH3:1]Br.[CH3:2][CH2:3][OH:4].[Pd]>>[CH3:1][O:4][CH2:3][CH3:2]"
STEP_1 = "[CH3:1][O:2][CH2:3][CH3:4]>>[CH3:1][O:2][CH2:3][CH2:4]Cl"


def _fixture(tmp_path):
    names = {
        "Reactants, Solvents, Catalysts": {
            "A": "methyl bromide",
            "B": "Palladium on carbon",
            "C": "ethanol",
            "E": "Raney nickel",
        },
        "Products": {"D": "ethyl methyl ether", "F": "chloroethyl methyl ether"},
    }
    smiles = {"A": "CBr", "B": "[Pd]", "C": "CCO", "E": "[Ni]", "D": "CCOC", "F": "COCCCl"}
    rows = []
    for skeleton, rxn, mapped in (
        (["A.C.B>D"], ["CBr.CCO.[Pd]>CCOC"], [STEP_0]),
        (["A.C>D", "D>F"], ["CBr.CCO>CCOC", "CCOC>COCCCl"], [STEP_0.replace(".[Pd]", ""), STEP_1]),
    ):
        rows.append(
            {
                "title": "t",
                "paragraph": "p",
                "prediction": json.dumps(names),
                "error": "",
                "m_smiles": json.dumps(smiles),
                "m_skeleton": repr(skeleton),
                "m_rxn": repr(rxn),
                "m_localmapper_rxn": repr([r.replace(">", ">>", 1).replace(">>>", ">>") for r in rxn]),
                "m_mapped_rxn": repr(mapped),
                "m_mapping_template": repr(["template"] * len(rxn)),
                "m_mapping_confident": repr([True] * len(rxn)),
                "m_mapping_error": repr([""] * len(rxn)),
            }
        )
    results = tmp_path / "outputs"
    results.mkdir()
    pd.DataFrame(rows).to_csv(results / "m_reaction_smiles_final.csv", index=False, encoding="utf-8-sig")
    frame = pd.read_csv(results / "m_reaction_smiles_final.csv", encoding="utf-8-sig")
    review = pd.DataFrame(
        [
            {"idx": idx, **build_ground_truth_review.model_columns("m", build_ground_truth_review.model_row(row, "m"))}
            for idx, row in frame.iterrows()
        ]
    )
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "overrides": [
                    {"name": "palladium on carbon", "lookup_name": "palladium", "kind": "active_component"},
                    {"name": "Raney nickel", "lookup_name": "nickel", "kind": "active_component"},
                    {"name": "ethanol", "lookup_name": "ethyl alcohol", "kind": "identity_alias"},
                ]
            }
        ),
        encoding="utf-8",
    )
    decisions = tmp_path / "review.csv"
    pd.DataFrame(
        [
            # Row 1 carries the reviewed structure; row 0 carries another one.
            {"row_zero_based": 1, "name": "methyl bromide", "normalized_name": "", "review_decision": "use_smiles", "reviewed_smiles": "BrC"},
            {"row_zero_based": 0, "name": "methyl bromide", "normalized_name": "methylbromide", "review_decision": "use_smiles", "reviewed_smiles": "[2H]CBr"},
            {"row_zero_based": 0, "name": "biaryl", "normalized_name": "biaryl", "review_decision": "keep_unresolved", "reviewed_smiles": ""},
        ]
    ).to_csv(decisions, index=False)
    return review, results, overrides, decisions


def test_override_ablation_rebuilds_affected_steps_as_unmapped(tmp_path):
    review, results, overrides, _decisions = _fixture(tmp_path)
    ablated, audit, outputs = curation_ablation.ablate(
        review, ["m"], override_path=overrides, results_dir=results
    )

    # Every resolved component with a curated name counts, inside the skeleton or not.
    assert sorted(zip(audit["idx"], audit["label"], audit["in_skeleton"])) == [
        (0, "B", 1), (0, "C", 1), (0, "E", 0), (1, "B", 0), (1, "C", 1), (1, "E", 0),
    ]
    assert ablated["m_has_unresolved_nosmi"].tolist() == [1, 1]
    assert review["m_has_unresolved_nosmi"].tolist() == [0, 0]

    out = outputs["m"]
    assert json.loads(out.at[0, "m_smiles"])["E"] == "[Raney nickel (NoSmi)]"
    assert eval(out.at[0, "m_rxn"]) == ["CBr.[ethanol (NoSmi)].[Palladium on carbon (NoSmi)]>CCOC"]
    assert eval(out.at[1, "m_mapped_rxn"]) == [None, STEP_1]
    assert eval(out.at[1, "m_mapping_error"]) == [curation_ablation.NOSMI_MAPPING_ERROR, ""]
    assert eval(out.at[1, "m_mapping_template"]) == [None, "template"]
    # Components of the unmapped step drop out of the rebuilt review row.
    assert json.loads(ablated.at[1, "m_reactants_smiles"]) == ["CCOC"]

    heuristics = evaluate_table3.evaluate_model_heuristics(out.fillna("").iloc[1], model="m", relevant_nosmi=True)
    assert heuristics["max_reactant_count"] == 1


def test_blocked_scope_keeps_only_names_the_resolver_rejects(tmp_path):
    review, results, overrides, _decisions = _fixture(tmp_path)
    ablated, audit = curation_ablation.apply_curation_ablation(
        review, ["m"], override_path=overrides, override_scope="blocked", results_dir=results
    )

    assert sorted(set(audit["name"])) == ["Palladium on carbon", "Raney nickel"]
    assert set(audit["automatic_resolution_blocked"]) == {1}
    assert ablated["m_has_unresolved_nosmi"].tolist() == [1, 0]


def test_row_decision_requires_the_reviewed_smiles(tmp_path):
    review, results, _overrides, decisions = _fixture(tmp_path)
    ablated, audit = curation_ablation.apply_curation_ablation(
        review, ["m"], row_decision_path=decisions, results_dir=results
    )

    assert audit[["idx", "label", "reason"]].values.tolist() == [[1, "A", "row_decision"]]
    assert ablated["m_has_unresolved_nosmi"].tolist() == [0, 1]


def test_variant_names():
    assert curation_ablation.variant_name() == ""
    assert curation_ablation.variant_name(no_overrides=True) == "no_overrides"
    assert curation_ablation.variant_name(no_overrides=True, override_scope="blocked") == "no_overrides_blocked"
    assert (
        curation_ablation.variant_name(strict_structures=True, first_reference_only=True)
        == "strict_matching"
    )


@pytest.fixture
def strict_matching():
    evaluate_table3.configure_matching(strict_structures=True, first_reference_only=True)
    yield
    evaluate_table3.configure_matching()


def test_default_matching_ignores_stereo_charge_and_isotopes():
    compare = evaluate_table3.comparison_smiles
    assert compare("C[C@H](N)C(=O)O") == compare("CC(N)C(=O)O")
    assert compare("[NH3+]CC(=O)[O-]") == compare("NCC(=O)O")
    assert compare("[13CH3]O") == compare("CO")


def test_strict_matching_keeps_stereo_charge_and_isotopes(strict_matching):
    compare = evaluate_table3.comparison_smiles
    assert compare("[CH3:1][C@H:2]([NH2:3])[C:4](=[O:5])[OH:6]") == compare("C[C@H](N)C(=O)O")
    assert compare("C[C@H](N)C(=O)O") != compare("CC(N)C(=O)O")
    assert compare("[NH3+]CC(=O)[O-]") != compare("NCC(=O)O")
    assert compare("[13CH3]O") != compare("CO")


def test_first_reference_only_drops_the_second_reference(strict_matching):
    review = pd.DataFrame(
        [
            {
                "idx": 0,
                "ground_truth_valid": True,
                "ground_truth_reactants_smiles": "['CCO']",
                "ground_truth_reagents_smiles": "[]",
                "ground_truth_products_smiles": "['CC=O']",
                "ground_truth_2_reactants_smiles": "['CCO', 'O']",
                "ground_truth_2_reagents_smiles": "[]",
                "ground_truth_2_products_smiles": "['CC=O']",
            }
        ]
    )
    manifest = evaluate_table3.build_ground_truth_manifest(review)
    assert manifest.loc[0, "ground_truth_count"] == 1


def test_identical_structure_heuristic_is_unchanged_by_strict_matching(strict_matching):
    step = evaluate_table3.evaluate_mapped_step(
        "[NH2:1][CH2:2][C:3](=[O:4])[OH:5]>>[NH3+:1][CH2:2][C:3](=[O:4])[O-:5]",
        template_missing=False,
        mapping_error="",
    )
    assert step["identical_structures"] == 1


def test_fully_automatic_rebuild_turns_element_atoms_into_molecules(tmp_path):
    names = {
        "Reactants, Solvents, Catalysts": {"A": "styrene", "B": "hydrogen", "C": "bromine"},
        "Products": {"D": "ethylbenzene"},
    }
    smiles = {"A": "C=Cc1ccccc1", "B": "[H]", "C": "[Br]", "D": "CCc1ccccc1"}
    row = {
        "title": "t",
        "paragraph": "p",
        "prediction": json.dumps(names),
        "error": "",
        "m_smiles": json.dumps(smiles),
        "m_skeleton": repr(["A.B>D"]),
        "m_rxn": repr(["C=Cc1ccccc1.[H]>CCc1ccccc1"]),
        "m_localmapper_rxn": repr(["C=Cc1ccccc1.[H]>>CCc1ccccc1"]),
        "m_mapped_rxn": repr(["OLD"]),
        "m_mapping_template": repr(["template"]),
        "m_mapping_confident": repr([True]),
        "m_mapping_error": repr([""]),
    }
    results = tmp_path / "outputs"
    results.mkdir()
    pd.DataFrame([row]).to_csv(results / "m_reaction_smiles_final.csv", index=False, encoding="utf-8-sig")
    frame = pd.read_csv(results / "m_reaction_smiles_final.csv", encoding="utf-8-sig")
    review = pd.DataFrame(
        [{"idx": 0, **build_ground_truth_review.model_columns("m", build_ground_truth_review.model_row(frame.iloc[0], "m"))}]
    )
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"overrides": []}), encoding="utf-8")
    decisions = tmp_path / "review.csv"
    pd.DataFrame(columns=["row_zero_based", "name", "review_decision", "reviewed_smiles"]).to_csv(decisions, index=False)
    recording = tmp_path / "empty_raw.jsonl"
    recording.write_text("", encoding="utf-8")

    def stub(reaction):
        return {"localmapper_rxn": reaction, "mapped_rxn": "MAPPED", "mapping_template": "t",
                "mapping_confident": True, "mapping_error": ""}

    _review, audit, outputs = curation_ablation.automatic_ablate(
        review, ["m"], override_path=overrides, row_decision_path=decisions,
        requery_log=recording, results_dir=results, remap=stub,
    )

    assert sorted(zip(audit["label"], audit["automatic_smiles"], audit["reason"])) == [
        ("B", "[H][H]", "molecular_form"), ("C", "BrBr", "molecular_form"),
    ]
    out = outputs["m"]
    assert eval(out.at[0, "m_rxn"]) == ["C=Cc1ccccc1.[H][H]>CCc1ccccc1"]
    assert eval(out.at[0, "m_mapped_rxn"]) == ["MAPPED"]
    assert json.loads(out.at[0, "m_smiles"])["C"] == "BrBr"
