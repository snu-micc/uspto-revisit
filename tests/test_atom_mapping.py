import pytest

from uspto_revisit.atom_mapping import (
    _install_localmapper_compatibility,
    _legacy_resource_filename,
    add_atom_mapping_columns,
    map_reaction_values,
    normalize_localmapper_reaction,
    parse_reaction_smiles,
)


class FakeMapper:
    def get_atom_map(self, reactions, return_dict=False):
        single = isinstance(reactions, str)
        values = [reactions] if single else reactions
        if any("INVALID" in reaction for reaction in values):
            raise ValueError("invalid molecule")
        results = [
            {
                "rxn": reaction,
                "mapped_rxn": f"mapped:{reaction}",
                "template": "[C:1]>>[C:1]",
                "confident": True,
            }
            for reaction in values
        ]
        return results[0] if single else results


def test_legacy_resource_filename_resolves_packaged_file():
    path = _legacy_resource_filename("uspto_revisit", "__init__.py")
    assert path.endswith("__init__.py")


def test_localmapper_compatibility_disables_distributed_rpc(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "dgl.distributed", raising=False)
    _install_localmapper_compatibility()

    distributed = sys.modules["dgl.distributed"]
    assert distributed.__name__ == "dgl.distributed"
    assert distributed.DistGraph.__module__ == "uspto_revisit.atom_mapping"


def test_parse_reaction_smiles_accepts_serialized_list():
    assert parse_reaction_smiles("['CC>CO', 'CN>>CN']") == ["CC>CO", "CN>>CN"]


@pytest.mark.parametrize(
    ("reaction", "expected"),
    [
        ("CC>CO", "CC>>CO"),
        ("CC>O>CO", "CC.O>>CO"),
        ("CC>>CO", "CC>>CO"),
        ("CC>O>CN>CO", "CC.O.CN>>CO"),
    ],
)
def test_normalize_localmapper_reaction(reaction, expected):
    assert normalize_localmapper_reaction(reaction) == expected


def test_normalize_localmapper_reaction_rejects_unresolved_smiles():
    with pytest.raises(ValueError, match="NoSmi"):
        normalize_localmapper_reaction("CC>[unknown (NoSmi)]")


def test_map_reaction_values_isolates_failed_reactions():
    result = map_reaction_values(
        ["['CC>CO', 'CC>INVALID']", "['CN>>CN']"],
        FakeMapper(),
        batch_size=8,
    )

    assert result["localmapper_rxn"] == [
        ["CC>>CO", "CC>>INVALID"],
        ["CN>>CN"],
    ]
    assert result["mapped_rxn"] == [
        ["mapped:CC>>CO", None],
        ["mapped:CN>>CN"],
    ]
    assert result["mapping_confident"] == [[True, None], [True]]
    assert "invalid molecule" in result["mapping_error"][0][1]


def test_add_atom_mapping_columns_preserves_input_frame():
    import pandas as pd

    source = pd.DataFrame({"model_rxn": [["CC>CO"]]})
    result = add_atom_mapping_columns(
        source,
        "model_rxn",
        "model",
        mapper=FakeMapper(),
    )

    assert list(source.columns) == ["model_rxn"]
    assert result.loc[0, "model_mapped_rxn"] == ["mapped:CC>>CO"]
    assert result.loc[0, "model_mapping_template"] == ["[C:1]>>[C:1]"]
