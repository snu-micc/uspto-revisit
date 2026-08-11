import asyncio
import sys
import types
from types import SimpleNamespace

from uspto_revisit.gpt_extract import generate_model_output, parse_json_output


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


def test_generate_model_output_uses_openai_responses_api():
    responses = FakeOpenAIResponses()
    client = SimpleNamespace(responses=responses)

    output = asyncio.run(
        generate_model_output(
            client,
            "openai",
            "openai-test-model",
            "system prompt",
            "user prompt",
        )
    )

    assert output == '{"provider": "openai"}'
    assert responses.arguments == {
        "model": "openai-test-model",
        "instructions": "system prompt",
        "input": "user prompt",
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


def test_parse_json_output_keeps_first_json_value_when_extra_text_follows():
    assert parse_json_output('{"reaction": "first"}\n{"reaction": "extra"}') == {
        "reaction": "first"
    }
