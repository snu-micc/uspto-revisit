"""GPT-based reaction extraction using the bundled prompt template."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI

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
    return json.loads(raw_text.strip())


async def get_prediction_prompt_json(
    client: AsyncOpenAI,
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
            response = await asyncio.wait_for(
                client.responses.create(
                    model=model_name,
                    instructions=system_prompt,
                    input=user_prompt,
                ),
                timeout=timeout_seconds,
            )
            raw = response.output_text
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
    gpt_prompt: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    title_column: str = "title",
    paragraph_column: str = "paragraph",
    semaphore_size: int = 10,
    timeout_seconds: int = 180,
    partial_output_path: str | Path | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running GPT extraction.")

    if title_column not in input_df.columns or paragraph_column not in input_df.columns:
        available = ", ".join(input_df.columns)
        raise ValueError(
            f"Input must contain '{title_column}' and '{paragraph_column}' columns. "
            f"Available columns: {available}"
        )

    prompt_template = gpt_prompt or load_prompt()
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(semaphore_size)
    tasks = [
        get_prediction_prompt_json(
            client,
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

    return sorted(results, key=lambda item: item["idx"])


def results_to_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    frame["prediction"] = frame["prediction"].apply(
        lambda value: json.dumps(value, ensure_ascii=False) if value is not None else None
    )
    return frame
