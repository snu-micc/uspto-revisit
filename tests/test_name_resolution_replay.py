import importlib.util
import json
from pathlib import Path

import pytest

from analysis import curation_ablation, name_resolution_replay
from analysis.name_resolution_replay import RecordedResolver

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "result/name_resolution/requery_2026-10-27_ai_baseline_raw.jsonl"
SCRIPT = ROOT / "scripts" / "name_resolution" / "requery_curated_names.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("requery_curated_names", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recording_hash_is_verified(tmp_path):
    copy = tmp_path / "requery_test_raw.jsonl"
    copy.write_bytes(LOG.read_bytes() + b"\n")
    capture = name_resolution_replay.capture_path(LOG)
    name_resolution_replay.capture_path(copy).write_bytes(capture.read_bytes())
    with pytest.raises(ValueError, match="SHA-256"):
        name_resolution_replay.load_requery_log(copy)


def test_dry_run_sends_nothing(capsys):
    assert _load_script().main(["--dry-run"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 62


def test_fully_automatic_rebuild_with_stub_mapper(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "curation_ablation_tests", ROOT / "tests" / "test_curation_ablation.py"
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    STEP_1 = fixtures.STEP_1

    review, results, overrides = fixtures._fixture(tmp_path)
    # Methyl bromide is curated here too. Record every request the replay makes:
    # OPSIN resolves methyl bromide to a different structure and ethanol to the
    # curated one, and everything else is 404.
    rules = json.loads(overrides.read_text(encoding="utf-8"))
    rules["overrides"].append({"name": "methyl bromide", "lookup_name": "bromomethane", "kind": "identity_alias"})
    overrides.write_text(json.dumps(rules), encoding="utf-8")
    probe = tmp_path / "probe_raw.jsonl"
    probe.write_text("", encoding="utf-8")
    resolver = RecordedResolver(probe)
    resolver.resolve(["methyl bromide", "ethanol", "Palladium on carbon", "Raney nickel"])
    answers = {"methyl%20bromide.smi": "CI", "ethanol.smi": "CCO"}
    with (tmp_path / "test_raw.jsonl").open("w", encoding="utf-8") as handle:
        for url in sorted(resolver.missing_urls):
            body = next((v for k, v in answers.items() if url.endswith("/opsin/" + k)), None)
            record = {"url": url, "attempt": 1, "status": 200 if body else 404, "body": body or ""}
            handle.write(json.dumps(record) + "\n")

    def stub(reaction):
        return {"localmapper_rxn": reaction, "mapped_rxn": "MAPPED", "mapping_template": "t",
                "mapping_confident": True, "mapping_error": ""}

    ablated, audit, outputs = curation_ablation.automatic_ablate(
        review, ["m"], override_path=overrides,
        requery_log=tmp_path / "test_raw.jsonl", results_dir=results, remap=stub,
    )

    outcomes = {(r.idx, r.label): r.outcome for r in audit.itertuples()}
    assert outcomes[(0, "C")] == "unchanged"
    assert outcomes[(0, "B")] == "unresolved"
    assert outcomes[(1, "A")] == "replaced"
    out = outputs["m"]
    assert eval(out.at[1, "m_rxn"])[0] == "CI.CCO>CCOC"
    assert eval(out.at[1, "m_mapped_rxn"]) == ["MAPPED", STEP_1]
    assert eval(out.at[0, "m_mapping_error"]) == [curation_ablation.NOSMI_MAPPING_ERROR]
    assert ablated["m_has_unresolved_nosmi"].tolist() == [1, 0]


def test_no_fixed_aliases_or_structures_are_injected(tmp_path):
    probe = tmp_path / "probe_raw.jsonl"
    probe.write_text("", encoding="utf-8")
    resolver = RecordedResolver(probe)
    result = resolver.resolve(["THF", "DIEA"])
    assert result["THF"]["selected_smiles"] == ""
    assert any("DIEA" in url for url in resolver.missing_urls)
    assert not any("DIPEA" in url for url in resolver.missing_urls)
