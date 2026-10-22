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
