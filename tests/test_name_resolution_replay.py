import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from evaluation import curation_ablation, name_resolution_replay
from evaluation.name_resolution_replay import RecordedResolver

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / name_resolution_replay.DEFAULT_REQUERY_LOG
SCRIPT = ROOT / "scripts" / "name_resolution" / "requery_curated_names.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("requery_curated_names", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def spellings():
    script = _load_script()
    return script.curated_spellings(
        ROOT / curation_ablation.DEFAULT_RESULTS_DIR,
        ROOT / curation_ablation.DEFAULT_OVERRIDES,
        ROOT / curation_ablation.DEFAULT_ROW_DECISIONS,
    )


def test_replay_of_released_recording(spellings):
    assert len(spellings) == 127
    resolver = RecordedResolver(LOG)
    two_pass = resolver.resolve(spellings, order="two-pass")
    assert not resolver.missing_urls
    assert Counter(row["resolved_by"] for row in two_pass.values()) == {
        "unresolved": 97, "first_pass": 22, "retry_path": 8,
    }
    retry = RecordedResolver(LOG).resolve(spellings, order="retry")
    assert sum(row["refused"] for row in retry.values()) == 32
    assert Counter(row["resolver_agreement"] for row in retry.values())["conflict"] == 4
    # OPSIN answers first in the two-pass order (diphenylphosphinic azide).
    assert two_pass["diphenyl phosphoryl azide"]["selected_smiles"] == "[N-]=[N+]=NP(=O)(c1ccccc1)c1ccccc1"
    assert retry["diphenyl phosphoryl azide"]["selected_smiles"] == "[N-]=[N+]=NP(=O)(Oc1ccccc1)Oc1ccccc1"


def test_recording_hash_is_verified(tmp_path):
    copy = tmp_path / "requery_test_raw.jsonl"
    copy.write_bytes(LOG.read_bytes() + b"\n")
    capture = name_resolution_replay.capture_path(LOG)
    name_resolution_replay.capture_path(copy).write_bytes(capture.read_bytes())
    with pytest.raises(ValueError, match="SHA-256"):
        name_resolution_replay.load_requery_log(copy)


class _Response:
    def __init__(self, record):
        self.status = record.get("status")
        self._body = record.get("body", "")
        self._error = record.get("error")

    async def __aenter__(self):
        if self.status is None:
            raise OSError(self._error or "no answer")
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return self._body


class _RecordedSession:
    """Serves the released recording instead of the network."""

    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        return _Response(self.responses[url])


def test_live_script_sends_the_recorded_requests(spellings, monkeypatch):
    script = _load_script()
    recorded = name_resolution_replay.load_requery_log(LOG)
    session = _RecordedSession(recorded)
    recorder = script.LiveRecorder()

    async def no_wait(url):
        return None

    async def offline_resolve_all(sf, names, order):
        recorder._session = session
        return await recorder._resolve_names(sf, names, order)

    async def no_backoff(attempt, max_delay=60):
        return None

    from uspto_revisit import smiles_fetch

    monkeypatch.setattr(recorder, "_wait", no_wait)
    monkeypatch.setattr(recorder, "_resolve_all", offline_resolve_all)
    monkeypatch.setattr(smiles_fetch, "exponential_backoff", no_backoff)
    live = recorder.resolve(spellings, order="two-pass")
    replay = RecordedResolver(LOG).resolve(spellings, order="two-pass")

    assert set(session.urls) <= set(recorded)
    assert {k: v["selected_smiles"] for k, v in live.items()} == {
        k: v["selected_smiles"] for k, v in replay.items()
    }


def test_dry_run_sends_nothing(capsys):
    assert _load_script().main(["--dry-run"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 127


NO_TABLES_LOG = ROOT / "result" / "name_resolution" / "requery_2026-10-23_no_tables_raw.jsonl"
REVISED_OUTPUTS = ROOT / "result" / "revised" / "model_outputs"


def test_lookup_tables_can_be_disabled(tmp_path):
    from uspto_revisit import smiles_fetch

    probe = tmp_path / "probe_raw.jsonl"
    probe.write_text("", encoding="utf-8")
    assert RecordedResolver(probe).resolve(["THF"])["THF"]["selected_source"] == "KnownAlias"
    resolver = RecordedResolver(probe, lookup_tables=False)
    assert resolver.resolve(["THF", "DIEA"])["THF"]["selected_smiles"] == ""
    assert any("pubchem" in url for url in resolver.missing_urls)
    # Without the synonym list DIEA is looked up as written, not as DIPEA.
    assert any("DIEA" in url for url in resolver.missing_urls)
    assert not any("DIPEA" in url for url in resolver.missing_urls)
    assert smiles_fetch.DEFAULT_KNOWN_SMILES["THF"] == "C1CCOC1"  # restored after the replay
    assert smiles_fetch.SAFE_NAME_ALIASES["diea"] == "DIPEA"
    # The first pass also reaches the table through fix_name ("anhydrous ether" -> "ether").
    assert name_resolution_replay.builtin_table_smiles("anhydrous ether") == "CCOCC"
    assert name_resolution_replay.builtin_table_smiles("dimethylformamide") is None
    assert name_resolution_replay.alias_used("RuPHOS") and not name_resolution_replay.alias_used("RuPhos ligand")


def test_lookup_tables_dry_run_lists_the_recorded_names(capsys):
    assert _load_script().main(["--dry-run", "--names", "lookup-tables"]) == 0
    listed = capsys.readouterr().out.splitlines()
    capture = json.loads(name_resolution_replay.capture_path(NO_TABLES_LOG).read_text(encoding="utf-8"))
    assert listed == capture["spellings"] and len(listed) == 42


def test_replay_of_no_tables_recording():
    names = _load_script().lookup_table_spellings(REVISED_OUTPUTS / "fully_automatic_with_tables")
    resolver = RecordedResolver(NO_TABLES_LOG, lookup_tables=False)
    result = resolver.resolve(names, order="two-pass")
    assert not resolver.missing_urls
    # 28 resolve; the capture's 27 counts the live pass before 15 unanswered CIR URLs were re-sent.
    assert sum(bool(row["selected_smiles"]) for row in result.values()) == 28
    assert result["THF"]["selected_smiles"] == "C1CCOC1"
    assert result["brine"]["selected_smiles"] == "" and result["Pd/C"]["selected_smiles"] == ""
    # Without the tables, CIR reads DCM as deoxycytidine monophosphate.
    assert result["DCM"]["selected_source"] == "CIR" and "P(=O)(O)O" in result["DCM"]["selected_smiles"]


def test_no_tables_scenario_from_the_recording():
    """Step 5 of rebuild_revised_outputs.py, with a stub mapper: the released structures and reactions."""
    from evaluation import evaluate_table3, lookup_tables

    def stub(reaction):
        return {"localmapper_rxn": reaction, "mapped_rxn": "", "mapping_template": "",
                "mapping_confident": False, "mapping_error": ""}

    outputs = {m: curation_ablation.read_model_output(REVISED_OUTPUTS / "fully_automatic_with_tables", m)
               for m in evaluate_table3.MODELS}
    rebuilt, components, _ = lookup_tables.without_tables(outputs, remap=stub)
    assert components.outcome.value_counts().to_dict() == {"unchanged": 1196, "unresolved": 801, "replaced": 144}
    for model in evaluate_table3.MODELS:
        released = curation_ablation.read_model_output(REVISED_OUTPUTS / "fully_automatic", model)
        for column in (f"{model}_smiles", f"{model}_rxn"):
            assert rebuilt[model][column].fillna("").tolist() == released[column].fillna("").tolist(), column


def test_fully_automatic_rebuild_with_stub_mapper(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "curation_ablation_tests", ROOT / "tests" / "test_curation_ablation.py"
    )
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    STEP_1 = fixtures.STEP_1

    review, results, overrides, decisions = fixtures._fixture(tmp_path)
    # Record every request the replay makes: OPSIN resolves methyl bromide to a
    # different structure and ethanol to the curated one; everything else 404.
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
        review, ["m"], override_path=overrides, row_decision_path=decisions,
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
