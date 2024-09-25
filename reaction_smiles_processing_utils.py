import ast
from reaction_step_processing_utils import process_reaction_data

def replace_with_smiles(rxn_code, smiles_dict):
    # Split the fixed reaction code into parts separated by '>'
    parts = rxn_code.split('>')
    # Replace each code in the parts with the corresponding SMILES string
    replaced_parts = []
    for part in parts:
        codes = part.split('.')
        replaced_codes = [smiles_dict.get(code, code) for code in codes]  
        replaced_parts.append('.'.join(replaced_codes))
    replaced_rxn_code = '>'.join(replaced_parts)
    return replaced_rxn_code

def process_smiles_data(merged_json_responses, merged_smiles_dict):
    skeleton_smiles = []
    final_smiles = []
    skeleton_error_smiles = []
    
    for i, (json_response, smiles_dict) in enumerate(zip(merged_json_responses, merged_smiles_dict)):
        try:
            # Evaluate the string representation to convert back to dictionary or list
            if isinstance(json_response, str):
                json_response = ast.literal_eval(json_response.strip())
            elif not isinstance(json_response, dict):
                raise ValueError(f"json_response at index {i} is not a valid JSON string or dictionary.")

            if isinstance(smiles_dict, str):
                smiles_dict = ast.literal_eval(smiles_dict.strip())
            elif not isinstance(smiles_dict, dict):
                raise ValueError(f"smiles_dict at index {i} is not a valid JSON string or dictionary.")
            
            rxn_codes = process_reaction_data(json_response)
            skeleton_smiles.append(rxn_codes)
            fixed_rxn_codes = []
            if rxn_codes:
                for rxn_code in rxn_codes:
                    reaction_equation = replace_with_smiles(rxn_code, smiles_dict)
                    fixed_rxn_codes.append(reaction_equation)
            else:
                fixed_rxn_codes.append("Error: empty rxn_codes")
            final_smiles.append(fixed_rxn_codes)
        except Exception as e:
            print(f"Exception at index {i}: {e}")
            final_smiles.append(f"{i} Error: empty rxn_codes")
            skeleton_smiles.append(f"{i} Error: empty rxn_codes")
            skeleton_error_smiles.append(f"{i} Error: empty rxn_codes")

    # Return a tuple of all collected lists
    return skeleton_smiles, final_smiles, skeleton_error_smiles
