"""Run a resumable, reasoning-controlled extraction benchmark.

The two providers are configured as comparably as their APIs permit:

* OpenAI: reasoning.effort="low" and JSON-object response mode
* Gemini: thinking_level="low" (or thinking_budget=0 for "none") and
  application/json response MIME type

Each successful response is checkpointed with token and latency metadata.

Provenance: archived research copy (file dated 2026-08-01 17:25 KST), used for
the direct (synchronous) API calls of gpt-5.4, gpt-5.6-sol, gemini-2.5-flash
and the Gemini 3.x direct retries. The exact command lines were not saved; see
result/configuration_settings.csv for what is and is not known about the
settings of each configuration. Keys are read from the environment or from
the file given with --dotenv; never commit that file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


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

OUTPUT_COLUMNS = (
    "idx",
    "title",
    "paragraph",
    "prediction",
    "error",
    "provider",
    "model",
    "reasoning_level",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "latency_seconds",
    "attempt",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("openai", "gemini"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dotenv", default=None)
    parser.add_argument("--reasoning-level", default="low")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--paragraph-column", default="paragraph")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--semaphore-size", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--inter-chunk-delay-seconds", type=float, default=0.0)
    parser.add_argument("--minimum-chunk-period-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def load_dotenv(path: str | None) -> None:
    if not path:
        return
    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


def build_user_prompt(template: str, title: Any, paragraph: Any) -> str:
    return template.replace("{title}", str(title)).replace("{paragraph}", str(paragraph))


def parse_json_output(raw_text: str) -> dict[str, Any] | list[Any]:
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        value, _ = json.JSONDecoder().raw_decode(text)
        return value


def nested_value(obj: Any, *path: str) -> Any:
    current = obj
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


async def generate_openai(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_level: str,
    max_output_tokens: int,
) -> tuple[str, dict[str, Any]]:
    response = await client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        reasoning={"effort": reasoning_level},
        text={"format": {"type": "json_object"}},
        max_output_tokens=max_output_tokens,
        store=False,
    )
    usage = response.usage
    return response.output_text, {
        "input_tokens": nested_value(usage, "input_tokens"),
        "output_tokens": nested_value(usage, "output_tokens"),
        "reasoning_tokens": nested_value(
            usage, "output_tokens_details", "reasoning_tokens"
        ),
    }


async def generate_gemini(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_level: str,
    max_output_tokens: int,
) -> tuple[str, dict[str, Any]]:
    from google.genai import types

    if reasoning_level.lower() in {"none", "off", "disabled"}:
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
            include_thoughts=False,
        )
    else:
        thinking_config = types.ThinkingConfig(
            thinking_level=reasoning_level,
            include_thoughts=False,
        )

    response = await client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            thinking_config=thinking_config,
            max_output_tokens=max_output_tokens,
        ),
    )
    usage = response.usage_metadata
    return response.text or "", {
        "input_tokens": nested_value(usage, "prompt_token_count"),
        "output_tokens": nested_value(usage, "candidates_token_count"),
        "reasoning_tokens": nested_value(usage, "thoughts_token_count"),
    }


async def extract_one(
    *,
    client: Any,
    provider: str,
    model: str,
    system_prompt: str,
    prompt_template: str,
    reasoning_level: str,
    max_output_tokens: int,
    timeout_seconds: int,
    semaphore: asyncio.Semaphore,
    idx: int,
    title: Any,
    paragraph: Any,
    attempt: int,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(prompt_template, title, paragraph)
    started = time.perf_counter()
    prediction: dict[str, Any] | list[Any] | None = None
    usage: dict[str, Any] = {}
    error: str | None = None
    async with semaphore:
        try:
            generator = generate_openai if provider == "openai" else generate_gemini
            raw_text, usage = await asyncio.wait_for(
                generator(
                    client,
                    model,
                    system_prompt,
                    user_prompt,
                    reasoning_level,
                    max_output_tokens,
                ),
                timeout=timeout_seconds,
            )
            prediction = parse_json_output(raw_text)
        except asyncio.TimeoutError:
            error = f"Timed out after {timeout_seconds} seconds"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    return {
        "idx": idx,
        "title": title,
        "paragraph": paragraph,
        "prediction": (
            json.dumps(prediction, ensure_ascii=False)
            if prediction is not None
            else None
        ),
        "error": error,
        "provider": provider,
        "model": model,
        "reasoning_level": reasoning_level,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "latency_seconds": round(time.perf_counter() - started, 3),
        "attempt": attempt,
    }


def blank(value: Any) -> bool:
    return value is None or pd.isna(value) or not str(value).strip()


def needs_retry(record: dict[str, Any] | None) -> bool:
    return record is None or not blank(record.get("error")) or blank(
        record.get("prediction")
    )


def load_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    records: dict[int, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        record["idx"] = int(record["idx"])
        records[record["idx"]] = record
    return records


def save_records(records: dict[int, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([records[idx] for idx in sorted(records)])
    for column in OUTPUT_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame.loc[:, list(OUTPUT_COLUMNS)].to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def pending_indices(
    input_frame: pd.DataFrame,
    records: dict[int, dict[str, Any]],
    title_column: str,
    paragraph_column: str,
) -> list[int]:
    pending: list[int] = []
    for idx in input_frame.index:
        record = records.get(int(idx))
        if needs_retry(record):
            pending.append(int(idx))
        elif (
            str(record.get("title")) != str(input_frame.at[idx, title_column])
            or str(record.get("paragraph"))
            != str(input_frame.at[idx, paragraph_column])
        ):
            pending.append(int(idx))
    return pending


async def run_chunk(
    *,
    provider: str,
    model: str,
    api_key: str,
    frame: pd.DataFrame,
    indices: list[int],
    title_column: str,
    paragraph_column: str,
    system_prompt: str,
    prompt_template: str,
    reasoning_level: str,
    max_output_tokens: int,
    timeout_seconds: int,
    semaphore_size: int,
    attempt: int,
) -> list[dict[str, Any]]:
    if provider == "openai":
        from openai import AsyncOpenAI

        client: Any = AsyncOpenAI(api_key=api_key)
    else:
        from google import genai

        client = genai.Client(api_key=api_key).aio

    semaphore = asyncio.Semaphore(semaphore_size)
    tasks = [
        extract_one(
            client=client,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            reasoning_level=reasoning_level,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            semaphore=semaphore,
            idx=idx,
            title=frame.at[idx, title_column],
            paragraph=frame.at[idx, paragraph_column],
            attempt=attempt,
        )
        for idx in indices
    ]
    results = await asyncio.gather(*tasks)
    close = getattr(client, "aclose", None)
    if close is not None:
        await close()
    return results


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(args.dotenv)
    api_key_variable = (
        "OPENAI_API_KEY" if args.provider == "openai" else "GEMINI_API_KEY"
    )
    api_key = os.getenv(api_key_variable)
    if not api_key:
        raise RuntimeError(f"{api_key_variable} is not set")
    if min(
        args.chunk_size,
        args.semaphore_size,
        args.timeout_seconds,
        args.max_rounds,
        args.max_output_tokens,
    ) < 1:
        raise ValueError("All numeric run settings must be positive.")
    if args.inter_chunk_delay_seconds < 0:
        raise ValueError("inter-chunk-delay-seconds cannot be negative.")
    if args.minimum_chunk_period_seconds < 0:
        raise ValueError("minimum-chunk-period-seconds cannot be negative.")

    frame = pd.read_csv(args.input)
    if args.limit is not None:
        frame = frame.head(args.limit).copy()
    frame.index = range(len(frame))
    prompt_template = Path(args.prompt).read_text(encoding="utf-8-sig")
    output_path = Path(args.output)
    records = load_records(output_path)

    for round_number in range(1, args.max_rounds + 1):
        pending = pending_indices(
            frame, records, args.title_column, args.paragraph_column
        )
        if not pending:
            break
        print(
            f"[ROUND {round_number}/{args.max_rounds}] "
            f"{args.provider} {args.model}: {len(pending)} rows pending",
            flush=True,
        )
        for start in range(0, len(pending), args.chunk_size):
            chunk_started = time.perf_counter()
            indices = pending[start : start + args.chunk_size]
            results = asyncio.run(
                run_chunk(
                    provider=args.provider,
                    model=args.model,
                    api_key=api_key,
                    frame=frame,
                    indices=indices,
                    title_column=args.title_column,
                    paragraph_column=args.paragraph_column,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    prompt_template=prompt_template,
                    reasoning_level=args.reasoning_level,
                    max_output_tokens=args.max_output_tokens,
                    timeout_seconds=args.timeout_seconds,
                    semaphore_size=args.semaphore_size,
                    attempt=round_number,
                )
            )
            for result in results:
                records[int(result["idx"])] = result
            save_records(records, output_path)
            succeeded = sum(not needs_retry(record) for record in records.values())
            print(
                f"[CHECKPOINT] {succeeded}/{len(frame)} successful; "
                f"saved {output_path}",
                flush=True,
            )
            if start + args.chunk_size < len(pending):
                elapsed = time.perf_counter() - chunk_started
                delay = max(
                    args.inter_chunk_delay_seconds,
                    args.minimum_chunk_period_seconds - elapsed,
                    0.0,
                )
                if delay:
                    time.sleep(delay)

    unresolved = pending_indices(
        frame, records, args.title_column, args.paragraph_column
    )
    save_records(records, output_path)
    if unresolved:
        print(f"[INCOMPLETE] {len(unresolved)} rows unresolved", flush=True)
        return 2
    print(f"[COMPLETE] {len(frame)} rows: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
