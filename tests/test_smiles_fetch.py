import asyncio
import json

from uspto_revisit import smiles_fetch


def test_extract_pubchem_smiles_accepts_current_field():
    response = json.dumps(
        {"PropertyTable": {"Properties": [{"CID": 6228, "SMILES": "CN(C)C=O"}]}}
    )

    assert smiles_fetch.extract_pubchem_smiles(response) == "CN(C)C=O"


def test_extract_pubchem_smiles_accepts_legacy_field():
    response = json.dumps(
        {
            "PropertyTable": {
                "Properties": [{"CID": 702, "IsomericSMILES": "C(C)O"}]
            }
        }
    )

    assert smiles_fetch.extract_pubchem_smiles(response) == "CCO"


def test_canonicalize_smiles_rejects_invalid_value():
    assert smiles_fetch.canonicalize_smiles("not-a-smiles") is None


def test_safe_aliases_and_ambiguous_names_are_distinguished():
    assert smiles_fetch.resolve_name_alias("N,N'-carbonyldiimidazole") == (
        "1,1'-carbonyldiimidazole"
    )
    assert smiles_fetch.resolve_name_alias("RuPhos") == (
        "2-dicyclohexylphosphino-2',6'-diisopropoxybiphenyl"
    )
    assert smiles_fetch.should_skip_automatic_resolution("compound 3")
    assert smiles_fetch.should_skip_automatic_resolution("palladium on carbon")
    assert smiles_fetch.should_skip_automatic_resolution("silica gel")
    assert smiles_fetch.should_skip_automatic_resolution("Sephadex LH-20")
    assert not smiles_fetch.should_skip_automatic_resolution("irbesartan")


def test_process_batch_final_deduplicates_lookup_names(monkeypatch):
    calls = []

    async def fake_resolve(name, session, semaphore):
        calls.append(name)
        return "CCO", "test"

    monkeypatch.setattr(smiles_fetch, "resolve_no_smi_name", fake_resolve)
    values = [
        {"A": "[absolute ethanol (NoSmi)]"},
        {"B": "[absolute ethanol (NoSmi)]"},
    ]

    asyncio.run(
        smiles_fetch.process_batch_final(
            values,
            session=object(),
            semaphore=object(),
        )
    )

    assert calls == ["absolute ethanol"]
    assert values == [{"A": "CCO"}, {"B": "CCO"}]


def test_normalize_name_key_handles_greek_symbols_and_punctuation():
    assert smiles_fetch.normalize_name_key("7α-(5-bromopentyl)") == (
        "7alpha5bromopentyl"
    )


def test_load_resolution_overrides_validates_direct_smiles(tmp_path):
    override_path = tmp_path / "overrides.json"
    override_path.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "name": "supported catalyst",
                        "smiles": "[Pd]",
                        "kind": "active_component",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    try:
        loaded = smiles_fetch.load_resolution_overrides(override_path)
        assert loaded["supportedcatalyst"]["smiles"] == "[Pd]"
        assert asyncio.run(
            smiles_fetch.resolve_no_smi_name(
                "supported catalyst",
                session=object(),
                semaphore=object(),
            )
        ) == ("[Pd]", "Curated:active_component")
    finally:
        smiles_fetch.load_resolution_overrides(None)


def test_chebi_primary_name_requires_exact_match(monkeypatch):
    exact = json.dumps(
        {
            "results": [
                {
                    "_id": "46195",
                    "_source": {
                        "name": "paracetamol",
                        "ascii_name": "paracetamol",
                        "smiles": "CC(=O)Nc1ccc(O)cc1",
                    },
                }
            ]
        }
    )

    async def fake_fetch(session, url, semaphore, max_retries):
        return exact

    monkeypatch.setattr(smiles_fetch, "fetch_smiles", fake_fetch)
    result = asyncio.run(
        smiles_fetch.get_smiles_from_chebi(
            session=object(),
            compound_name="paracetamol",
            semaphore=object(),
            max_retries=1,
        )
    )

    assert result == ("CC(=O)Nc1ccc(O)cc1", "ChEBI")


def test_chebi_does_not_accept_short_acronym_synonym(monkeypatch):
    search = json.dumps(
        {
            "results": [
                {
                    "_id": "134609",
                    "_source": {
                        "name": "nitrosobis(2-oxopropyl)amine",
                        "smiles": "CC(=O)CN(CC(C)=O)N=O",
                    },
                }
            ]
        }
    )

    async def fake_fetch(session, url, semaphore, max_retries):
        return search

    monkeypatch.setattr(smiles_fetch, "fetch_smiles", fake_fetch)
    result = asyncio.run(
        smiles_fetch.get_smiles_from_chebi(
            session=object(),
            compound_name="BOP",
            semaphore=object(),
            max_retries=1,
        )
    )

    assert result == (None, None)
