import json

import pandas as pd
import pytest

from evaluation import build_ground_truth_review as builder

STEP_0 = "[CH3:1]Br.[CH3:2][CH2:3][OH:4]>>[CH3:1][O:4][CH2:3][CH3:2]"
STEP_1 = "[CH3:1][O:2][CH2:3][CH3:4].[Cl:5][Cl:6]>>[CH3:1][O:2][CH2:3][CH2:4][Cl:5]"


def record(title, skeleton=("A.B>M", "M.C>N")):
    return pd.Series({
        "title": title,
        "prediction": json.dumps({"Products": {"M": "ethyl methyl ether", "N": "2-chloroethyl methyl ether"}}),
        "m_skeleton": str(list(skeleton)),
        "m_smiles": json.dumps({"A": "CBr", "B": "CCO", "C": "ClCl", "M": "COCC", "N": "COCCCl"}),
        "m_rxn": str(["CBr.CCO>>COCC", "COCC.ClCl>>COCCCl"][: len(skeleton)]),
        "m_mapped_rxn": str([STEP_0, STEP_1][: len(skeleton)]),
        "m_mapping_error": str(["", ""][: len(skeleton)]),
    })


@pytest.fixture(autouse=True)
def default_scope():
    yield
    builder.configure_step_scope()


def test_title_step_follows_the_product_named_like_the_title():
    assert builder.title_step_index(record("2-Chloroethyl methyl ether"), "m") == 1
    assert builder.title_step_index(record("Ethyl methyl ether"), "m") == 0


def test_unmatched_title_falls_back_to_last_step_and_single_step_is_untouched():
    assert builder.title_step_index(record("unrelated heading"), "m") == 1
    assert builder.title_step_index(record("Ethyl methyl ether", skeleton=("A.B>M",)), "m") is None


def test_default_scope_keeps_every_step():
    row = builder.model_row(record("Ethyl methyl ether"), "m")
    assert sorted(row["products"]) == ["CCOC", "COCCCl"]


def test_title_step_scope_compares_only_the_title_forming_step():
    builder.configure_step_scope(title_step_only=True)
    row = builder.model_row(record("Ethyl methyl ether"), "m")
    assert row["reactants"] == ["CBr", "CCO"]
    assert row["products"] == ["CCOC"]
    assert row["mapped_reactions"] == [STEP_0]
