import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from uspto_revisit import cli


def test_pipeline_parser_uses_repository_no_smi_overrides():
    args = cli.build_parser().parse_args([])

    assert args.overrides is not None
    assert Path(args.overrides).as_posix().endswith("config/nosmi_overrides.json")


def test_default_final_paths_are_model_scoped():
    assert cli.default_reaction_smiles_output_path("gpt-5.4").as_posix() == (
        "result/gpt-5.4_reaction_smiles_final.csv"
    )


@pytest.mark.parametrize(
    ("provider", "environment_variable", "fallback"),
    [
        ("openai", "OPENAI_MODEL", "gpt-4.1-mini"),
        ("gemini", "GEMINI_MODEL", "gemini-2.5-flash"),
    ],
)
def test_default_model_is_provider_scoped(
    monkeypatch,
    provider,
    environment_variable,
    fallback,
):
    monkeypatch.delenv(environment_variable, raising=False)
    assert cli.default_model_for_provider(provider) == fallback

    monkeypatch.setenv(environment_variable, "custom-model")
    assert cli.default_model_for_provider(provider) == "custom-model"


def test_gpt_extract_accepts_gemini_provider():
    args = cli.build_parser().parse_args(
        ["gpt-extract", "--provider", "gemini", "--model", "gemini-test"]
    )

    assert args.provider == "gemini"
    assert args.model == "gemini-test"


def test_nosmi_human_review_commands_have_repository_defaults():
    export_args = cli.build_parser().parse_args(["nosmi-review-export"])
    apply_args = cli.build_parser().parse_args(["nosmi-review-apply"])

    assert export_args.output == "result/nosmi_human_review.csv"
    assert apply_args.review == "result/nosmi_human_review.csv"
    assert apply_args.output_dir == "result"


def test_final_reprocess_loads_overrides(tmp_path, monkeypatch):
    initial_path = tmp_path / "smiles_dict_initial.json"
    initial_path.write_text(json.dumps([]), encoding="utf-8")
    override_path = tmp_path / "overrides.json"
    calls = {}

    def fake_load_overrides(path):
        calls["overrides"] = path

    async def fake_reprocess(
        input_file,
        output_file,
        session,
        semaphore,
        batch_size,
    ):
        calls["input"] = input_file
        calls["batch_size"] = batch_size
        output_file.write_text(json.dumps([]), encoding="utf-8")

    monkeypatch.setattr(cli, "load_resolution_overrides", fake_load_overrides)
    monkeypatch.setattr(cli, "reprocess_no_smi", fake_reprocess)
    monkeypatch.setattr(cli, "aiohttp_ssl_context", lambda: None)

    class FakeClientSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return False

    fake_aiohttp = SimpleNamespace(
        ClientSession=FakeClientSession,
        TCPConnector=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)

    final_path = asyncio.run(
        cli.maybe_reprocess_no_smi(
            batch_dir=tmp_path,
            batch_size=25,
            reprocess_concurrency=2,
            skip_reprocess=False,
            overrides=override_path,
        )
    )

    assert calls["overrides"] == override_path
    assert calls["input"] == initial_path
    assert calls["batch_size"] == 25
    assert final_path == tmp_path / "smiles_dict_final.json"
    assert final_path.exists()
