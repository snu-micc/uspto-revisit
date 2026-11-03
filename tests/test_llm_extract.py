import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from uspto_revisit.llm_extract import generate_model_output, parse_json_output


class FakeOpenAIResponses:
    def __init__(self):
        self.arguments = None

    async def create(self, **kwargs):
        self.arguments = kwargs
        return SimpleNamespace(output_text='{"provider": "openai"}')


class FakeGeminiModels:
    def __init__(self):
        self.arguments = None

    async def generate_content(self, **kwargs):
        self.arguments = kwargs
        return SimpleNamespace(text='{"provider": "gemini"}')


@pytest.mark.parametrize("model", [
    "gpt-4.1-mini",
    "ft:gpt-4.1-mini-2025-04-14:micc:uspto-revisit:DNfwY5a3",
])
def test_generate_model_output_uses_openai_responses_api(model):
    responses = FakeOpenAIResponses()
    client = SimpleNamespace(responses=responses)

    output = asyncio.run(
        generate_model_output(
            client,
            "openai",
            model,
            "system prompt",
            "user prompt",
        )
    )

    assert output == '{"provider": "openai"}'
    assert responses.arguments == {
        "model": model,
        "instructions": "system prompt",
        "input": "user prompt",
        "max_output_tokens": 16384,
    }


def test_generate_model_output_uses_gemini_json_mode(monkeypatch):
    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_genai = types.ModuleType("google.genai")
    fake_genai.types = SimpleNamespace(GenerateContentConfig=GenerateContentConfig)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    models = FakeGeminiModels()
    client = SimpleNamespace(models=models)

    output = asyncio.run(
        generate_model_output(
            client,
            "gemini",
            "gemini-test-model",
            "system prompt",
            "user prompt",
        )
    )

    assert output == '{"provider": "gemini"}'
    assert models.arguments["model"] == "gemini-test-model"
    assert models.arguments["contents"] == "user prompt"
    assert models.arguments["config"].system_instruction == "system prompt"
    assert models.arguments["config"].response_mime_type == "application/json"
    assert models.arguments["config"].max_output_tokens == 16384


def test_prediction_forwards_custom_output_limit_to_api():
    from uspto_revisit.llm_extract import get_prediction_prompt_json

    responses = FakeOpenAIResponses()
    result = asyncio.run(get_prediction_prompt_json(
        client=SimpleNamespace(responses=responses), provider="openai",
        gpt_prompt="{title}: {paragraph}", system_prompt="system", idx=0,
        title="title", paragraph="paragraph", semaphore=asyncio.Semaphore(1),
        model_name="gpt-4.1-mini", max_output_tokens=2048,
    ))
    assert result["prediction"] == {"provider": "openai"}
    assert result["error"] is None
    assert responses.arguments["max_output_tokens"] == 2048


def test_parse_json_output_keeps_first_json_value_when_extra_text_follows():
    assert parse_json_output('{"reaction": "first"}\n{"reaction": "extra"}') == {
        "reaction": "first"
    }
