from uspto_revisit.reaction_smiles import process_smiles_data, replace_with_smiles


def test_replace_with_smiles_keeps_unknown_codes():
    rxn_code = "A.B>C>D"
    smiles_dict = {"A": "CCO", "B": "O", "D": "CC=O"}

    assert replace_with_smiles(rxn_code, smiles_dict) == "CCO.O>C>CC=O"


def test_process_smiles_data_ignores_self_referencing_intermediate():
    response = {
        "Reactants, Solvents, Catalysts": {"A": "reactant"},
        "Products": {"B": "product"},
        "Reaction Steps": {
            "1 (Reaction, Add)": "A->mixture1",
            "2 (Reaction, Analyze)": "mixture1->mixture1",
            "3 (Work-up, Isolate)": "mixture1->B",
        },
    }
    skeletons, reactions, errors = process_smiles_data(
        [response],
        [{"A": "CC", "B": "CO"}],
    )

    assert skeletons == [["A>B"]]
    assert reactions == [["CC>CO"]]
    assert errors == []
