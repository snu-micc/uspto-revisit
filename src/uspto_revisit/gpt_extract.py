"""LLM-based reaction extraction using the bundled prompt template."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_PROVIDERS = ("openai", "gemini")

DEFAULT_SYSTEM_PROMPT = """
You are a chemical reaction extraction assistant.
Return exactly one valid JSON object and nothing else.
Do not use markdown.
Do not use code fences.
Do not include any text before or after the JSON.

Follow the requested schema exactly.
Use only information supported by the input text.
Be careful to distinguish reaction steps from work-up steps.
""".strip()


def default_prompt_path() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts" / "prompt.txt"


def load_prompt(prompt_path: str | Path | None = None) -> str:
    path = Path(prompt_path) if prompt_path else default_prompt_path()
    return path.read_text(encoding="utf-8-sig")


def build_user_prompt(prompt_template: str, title: Any, paragraph: Any) -> str:
    return prompt_template.replace("{title}", str(title)).replace("{paragraph}", str(paragraph))


def parse_json_output(raw_text: str) -> dict | list:
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # Some providers occasionally append a second JSON value despite the
        # JSON-only instruction. Keep the first complete response object.
        value, _ = json.JSONDecoder().raw_decode(text)
        return value


def normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        choices = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unsupported LLM provider '{provider}'. Choose one of: {choices}.")
    return normalized


def api_key_environment_variable(provider: str) -> str:
    normalized = normalize_provider(provider)
    return "OPENAI_API_KEY" if normalized == "openai" else "GEMINI_API_KEY"


async def generate_model_output(
    client: Any,
    provider: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    if provider == "openai":
        response = await client.responses.create(
            model=model_name,
            instructions=system_prompt,
            input=user_prompt,
        )
        return response.output_text

    from google.genai import types

    response = await client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


async def get_prediction_prompt_json(
    client: Any,
    provider: str,
    gpt_prompt: str,
    system_prompt: str,
    idx: int,
    title: Any,
    paragraph: Any,
    semaphore: asyncio.Semaphore,
    model_name: str,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(gpt_prompt, title, paragraph)

    async with semaphore:
        try:
            raw = await asyncio.wait_for(
                generate_model_output(
                    client,
                    provider,
                    model_name,
                    system_prompt,
                    user_prompt,
                ),
                timeout=timeout_seconds,
            )
            prediction = parse_json_output(raw)
            error = None
        except asyncio.TimeoutError:
            prediction = None
            error = f"Timed out after {timeout_seconds} seconds"
        except Exception as exc:
            prediction = None
            error = str(exc)

    return {
        "idx": idx,
        "title": title,
        "paragraph": paragraph,
        "prediction": prediction,
        "error": error,
    }


async def run_all_prompt_json(
    input_df: pd.DataFrame,
    model_name: str,
    provider: str = "openai",
    gpt_prompt: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    title_column: str = "title",
    paragraph_column: str = "paragraph",
    semaphore_size: int = 10,
    timeout_seconds: int = 180,
    partial_output_path: str | Path | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    provider = normalize_provider(provider)
    api_key_env = api_key_environment_variable(provider)
    api_key = api_key or os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} before running {provider} extraction.")

    if title_column not in input_df.columns or paragraph_column not in input_df.columns:
        available = ", ".join(input_df.columns)
        raise ValueError(
            f"Input must contain '{title_column}' and '{paragraph_column}' columns. "
            f"Available columns: {available}"
        )

    prompt_template = gpt_prompt or load_prompt()
    if provider == "openai":
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install a compatible openai package before running OpenAI extraction."
            ) from exc
        client = AsyncOpenAI(api_key=api_key)
    else:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Install the google-genai package before running Gemini extraction."
            ) from exc
        client = genai.Client(api_key=api_key).aio

    semaphore = asyncio.Semaphore(semaphore_size)
    tasks = [
        get_prediction_prompt_json(
            client,
            provider,
            prompt_template,
            system_prompt,
            idx,
            row[title_column],
            row[paragraph_column],
            semaphore,
            model_name,
            timeout_seconds,
        )
        for idx, row in input_df.iterrows()
    ]

    results = []
    try:
        for completed in asyncio.as_completed(tasks):
            result = await completed
            results.append(result)
            print(f"[{len(results)}/{len(tasks)}] {str(result['title'])[:40]}")
            if partial_output_path:
                partial_results = sorted(results, key=lambda item: item["idx"])
                results_to_frame(partial_results).to_csv(
                    partial_output_path,
                    index=False,
                    encoding="utf-8-sig",
                )
    finally:
        if provider == "openai":
            await client.close()
        else:
            await client.aclose()

    return sorted(results, key=lambda item: item["idx"])


def results_to_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    frame["prediction"] = frame["prediction"].apply(
        lambda value: json.dumps(value, ensure_ascii=False) if value is not None else None
    )
    return frame
