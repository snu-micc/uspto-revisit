"""Submit, inspect, and collect Gemini low-thinking batch extraction jobs.

Provenance: archived research copy (file dated 2026-07-31 13:36 KST), used for
the Gemini Batch API jobs of gemini-3.1-pro-preview and gemini-3.6-flash at
both thinking levels (pass --reasoning-level high for the high runs; the
display name keeps the hard-coded "low" prefix). Run it from this directory or
with scripts/extraction on the import path, because it imports
run_reasoning_controlled_extraction. GEMINI_API_KEY is read from --dotenv.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from google import genai
from google.genai import types

from run_reasoning_controlled_extraction import (
    DEFAULT_SYSTEM_PROMPT,
    OUTPUT_COLUMNS,
    build_user_prompt,
    load_dotenv,
    parse_json_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("submit", "status", "collect"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-file", required=True)
    parser.add_argument("--dotenv", required=True)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--reasoning-level", default="low")
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    return parser


def load_records(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        int(record["idx"]): record
        for record in pd.read_csv(path).to_dict(orient="records")
    }


def blank(value: Any) -> bool:
    return value is None or pd.isna(value) or not str(value).strip()


def pending_indices(frame: pd.DataFrame, records: dict[int, dict[str, Any]]) -> list[int]:
    return [
        int(idx)
        for idx in frame.index
        if idx not in records
        or not blank(records[idx].get("error"))
        or blank(records[idx].get("prediction"))
    ]


def save_records(records: dict[int, dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame([records[idx] for idx in sorted(records)])
    for column in OUTPUT_COLUMNS:
        if column not in frame:
            frame[column] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(OUTPUT_COLUMNS)].to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def usage_value(usage: Any, field: str) -> Any:
    return getattr(usage, field, None) if usage is not None else None


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(args.dotenv)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    job_path = Path(args.job_file)

    if args.action == "submit":
        frame = pd.read_csv(args.input)
        frame.index = range(len(frame))
        records = load_records(Path(args.output))
        indices = pending_indices(frame, records)
        prompt_template = Path(args.prompt).read_text(encoding="utf-8-sig")
        requests = []
        for idx in indices:
            requests.append(
                types.InlinedRequest(
                    contents=build_user_prompt(
                        prompt_template,
                        frame.at[idx, "title"],
                        frame.at[idx, "paragraph"],
                    ),
                    metadata={"idx": str(idx)},
                    config=types.GenerateContentConfig(
                        system_instruction=DEFAULT_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(
                            thinking_level=args.reasoning_level,
                            include_thoughts=False,
                        ),
                        max_output_tokens=args.max_output_tokens,
                    ),
                )
            )
        job = client.batches.create(
            model=args.model,
            src=requests,
            config=types.CreateBatchJobConfig(
                display_name=(
                    "uspto-reaction-extraction-low-"
                    + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                )
            ),
        )
        job_path.parent.mkdir(parents=True, exist_ok=True)
        job_path.write_text(
            json.dumps(
                {
                    "name": job.name,
                    "state": str(job.state),
                    "model": args.model,
                    "reasoning_level": args.reasoning_level,
                    "indices": indices,
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Submitted {len(indices)} requests: {job.name} ({job.state})")
        return 0

    metadata = json.loads(job_path.read_text(encoding="utf-8"))
    job = client.batches.get(name=metadata["name"])
    print(
        json.dumps(
            {
                "name": job.name,
                "state": str(job.state),
                "error": str(job.error or ""),
                "completion_stats": str(job.completion_stats or ""),
                "update_time": str(job.update_time or ""),
            },
            indent=2,
        )
    )
    if args.action == "status":
        return 0
    if "SUCCEEDED" not in str(job.state).upper():
        return 2

    indices = [int(idx) for idx in metadata["indices"]]
    responses = (job.dest.inlined_responses if job.dest else None) or []
    if len(responses) != len(indices):
        raise RuntimeError(
            f"Expected {len(indices)} responses, received {len(responses)}"
        )
    frame = pd.read_csv(args.input)
    records = load_records(Path(args.output))
    for idx, item in zip(indices, responses):
        if item.response is None:
            records[idx] = {
                "idx": idx,
                "title": frame.at[idx, "title"],
                "paragraph": frame.at[idx, "paragraph"],
                "prediction": None,
                "error": f"BatchError: {item.error}",
                "provider": "gemini",
                "model": args.model,
                "reasoning_level": args.reasoning_level,
                "input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "latency_seconds": None,
                "attempt": "batch",
            }
            continue
        response = item.response
        try:
            prediction = json.dumps(
                parse_json_output(response.text or ""),
                ensure_ascii=False,
            )
            error = None
        except Exception as exc:
            prediction = None
            error = f"{type(exc).__name__}: {exc}"
        usage = response.usage_metadata
        records[idx] = {
            "idx": idx,
            "title": frame.at[idx, "title"],
            "paragraph": frame.at[idx, "paragraph"],
            "prediction": prediction,
            "error": error,
            "provider": "gemini",
            "model": args.model,
            "reasoning_level": args.reasoning_level,
            "input_tokens": usage_value(usage, "prompt_token_count"),
            "output_tokens": usage_value(usage, "candidates_token_count"),
            "reasoning_tokens": usage_value(usage, "thoughts_token_count"),
            "latency_seconds": None,
            "attempt": "batch",
        }
    save_records(records, Path(args.output))
    remaining = pending_indices(frame, records)
    print(f"Collected {len(responses)} responses; {len(remaining)} unresolved")
    return 0 if not remaining else 3


if __name__ == "__main__":
    raise SystemExit(main())
