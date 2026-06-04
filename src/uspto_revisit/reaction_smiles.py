"""Convert reaction skeletons into reaction SMILES."""

from __future__ import annotations

import ast
from collections.abc import Mapping

from uspto_revisit.reaction_steps import process_reaction_data


def replace_with_smiles(rxn_code: str, smiles_dict: Mapping[str, str]) -> str:
    replaced_parts = []
    for part in rxn_code.split(">"):
        codes = part.split(".")
        replaced_parts.append(".".join(smiles_dict.get(code, code) for code in codes))
    return ">".join(replaced_parts)


def _coerce_mapping(value, index: int, name: str) -> dict:
    if isinstance(value, str):
        value = ast.literal_eval(value.strip())
    if not isinstance(value, dict):
        raise ValueError(f"{name} at index {index} is not a valid JSON string or dictionary.")
    return value


def process_smiles_data(merged_json_responses, merged_smiles_dict):
    skeleton_smiles = []
    final_smiles = []
    skeleton_error_smiles = []

    for idx, (json_response, smiles_dict) in enumerate(
        zip(merged_json_responses, merged_smiles_dict)
    ):
        try:
            json_response = _coerce_mapping(json_response, idx, "json_response")
            smiles_dict = _coerce_mapping(smiles_dict, idx, "smiles_dict")

            rxn_codes = process_reaction_data(json_response)
            skeleton_smiles.append(rxn_codes)
            if rxn_codes:
                final_smiles.append(
                    [replace_with_smiles(rxn_code, smiles_dict) for rxn_code in rxn_codes]
                )
            else:
                final_smiles.append(["Error: empty rxn_codes"])
        except Exception as exc:
            print(f"Exception at index {idx}: {exc}")
            error_message = f"{idx} Error: empty rxn_codes"
            final_smiles.append(error_message)
            skeleton_smiles.append(error_message)
            skeleton_error_smiles.append(error_message)

    return skeleton_smiles, final_smiles, skeleton_error_smiles
