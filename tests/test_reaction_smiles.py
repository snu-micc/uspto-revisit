from uspto_revisit.reaction_smiles import replace_with_smiles


def test_replace_with_smiles_keeps_unknown_codes():
    rxn_code = "A.B>C>D"
    smiles_dict = {"A": "CCO", "B": "O", "D": "CC=O"}

    assert replace_with_smiles(rxn_code, smiles_dict) == "CCO.O>C>CC=O"
