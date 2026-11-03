import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_nonreasoning_replacement_calls_set_limit_and_record_it(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1]/"analysis/run_replacement_apis.py"
    spec = importlib.util.spec_from_file_location("replacement_api_settings", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []

    class Client:
        def __init__(self, **_kwargs):
            self.responses = self

        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text='{"reaction": "test"}', model=kwargs["model"], usage=None)

        async def close(self):
            pass

    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = Client
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("google.genai")
    fake_genai.types = SimpleNamespace()
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test")
    source = tmp_path/"input.csv"
    source.write_text("idx,title,paragraph\n0,Title,Paragraph\n", encoding="utf-8")
    prompt = tmp_path/"prompt.txt"
    prompt.write_text("{title}\n{paragraph}", encoding="utf-8")
    configurations = ["gpt-4.1-mini-nonreasoning-shared", "gpt-4.1-mini-ft-nonreasoning-shared"]
    args = SimpleNamespace(dotenv=None, output=tmp_path/"outputs", input=source,
                           prompt=prompt, configurations=configurations, timeout=10,
                           dry_run=False, concurrency=1)
    asyncio.run(runner.run(args))
    assert {call["model"] for call in calls} == {runner.CONFIGS[c][1] for c in configurations}
    assert len(calls) == 2
    assert all(call["max_output_tokens"] == 16384 for call in calls)
    assert all("reasoning" not in call for call in calls)
    manifest_path = args.output/"run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["identity"]["max_output_tokens"] == 16384
    assert manifest["status"] == "complete"
    asyncio.run(runner.run(args))
    assert len(calls) == 2

    # A previous manifest that did not bind the non-reasoning token limit must
    # not be silently resumed under the newly explicit request settings.
    manifest["identity"].pop("max_output_tokens")
    manifest["identity"]["max_output_tokens_reasoning"] = 16384
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to mix"):
        asyncio.run(runner.run(args))
