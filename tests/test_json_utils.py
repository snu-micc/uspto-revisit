import json

from uspto_revisit.json_utils import is_valid_json, parse_json_object
from uspto_revisit.reaction_smiles import _coerce_mapping


def test_parse_json_object_accepts_double_encoded_object():
    payload = {"Reactants, Solvents, Catalysts": {"A": "water"}}
    double_encoded = json.dumps(json.dumps(payload))

    assert parse_json_object(double_encoded) == payload
    assert is_valid_json(double_encoded)


def test_coerce_mapping_accepts_double_encoded_object():
    payload = {"A": "O"}
    double_encoded = json.dumps(json.dumps(payload))

    assert _coerce_mapping(double_encoded, 0, "payload") == payload


def test_parse_json_object_repairs_invalid_model_escape():
    payload = r'{"Products": {"A": "(+)\-tartrate"}}'
    double_encoded = json.dumps(payload)

    expected = {"Products": {"A": "(+)-tartrate"}}
    assert parse_json_object(double_encoded) == expected
    assert _coerce_mapping(double_encoded, 0, "payload") == expected
