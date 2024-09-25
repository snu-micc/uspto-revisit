import json
import re
import logging

# JSON validation and fixing functions
def is_valid_json(json_string):
    try:
        json.loads(json_string)
        return True
    except ValueError as e:
        logging.error(f"Invalid JSON string: {json_string[:100]}... [Error] {e}")
        return False

def fix_json_string(json_string):
    try:
        json_string = json_string.strip()
        # Ensure JSON string starts with "{" and ends with "}"
        if not json_string.startswith('{'):
            json_string = '{' + json_string
        if not json_string.endswith('}'):
            json_string = json_string + '}'

        # Count the number of opening and closing braces
        open_braces = json_string.count('{')
        close_braces = json_string.count('}')

        # If there are more opening braces, add closing braces at the end
        if open_braces > close_braces:
            json_string += '}' * (open_braces - close_braces)
            
        # If there are more closing braces, add opening braces at the beginning
        elif close_braces > open_braces:
            json_string = '{' * (close_braces - open_braces) + json_string
        return json_string

    except Exception as e:
        logging.error(f"Failed to fix JSON string: {json_string} [Error] {e}")
        return None

def fix_name(compound_name):
    remove_patterns = [
        r"\d+\s+normal",
        r"\d+(\.\d+)?\s*N-",
        r"\d+(\.\d+)?\s*N",
        r"\d+(\.\d+)?\s*M",
        r"\d+%", 
        r"\s*\(\s*\)",
        r"\([^()]*\)$", 
        r"·",
        r'\([IVXLCDM]+\)',
        "anhydrous", "concentrated", "catalyst", "-catalyst", "saturated",
        "ice", "ice-", "dried", "aqueous", "solution", "normal", "solid", "complex", 
        "resin", "adduct", "corresponding", "atmosphere", "gas", "solvent", "crystal", 
        "crystals", "buffer",".conc","fuming","glacial"
    ]
    pattern = f"({'|'.join(remove_patterns)})"
    fixed_compound = re.sub(pattern, "", compound_name, flags=re.I)
    return fixed_compound.strip()