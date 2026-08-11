"""Resume LLM extraction without repeating rows that already succeeded."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPOSITORY_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from uspto_revisit.cli import (  # noqa: E402
    default_gpt_output_path,
    default_model_for_provider,
    load_env_file,
    read_table,
)
from uspto_revisit.gpt_extract import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    load_prompt,
    results_to_frame,
    run_all_prompt_json,
)

OUTPUT_COLUMNS = ("idx", "title", "paragraph", "prediction", "error")


def build_parser() -> argparse.ArgumentParser:
    load_env_file(REPOSITORY_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Resume OpenAI or Gemini extraction and retry only incomplete rows."
    )
    parser.add_argument("--input", default="examples/input.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--provider",
        choices=("openai", "gemini"),
        default=os.getenv("LLM_PROVIDER", "openai").lower(),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--paragraph-column", default="paragraph")
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--semaphore-size", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--max-rounds", type=int, default=4)
    return parser


def is_blank(value: Any) -> bool:
    return value is None or pd.isna(value) or not str(value).strip()


def record_needs_retry(record: dict[str, Any] | None) -> bool:
    if record is None:
        return True
    return not is_blank(record.get("error")) or is_blank(record.get("prediction"))


def load_existing_records(output_path: Path) -> dict[int, dict[str, Any]]:
    if not output_path.exists():
        return {}

    frame = pd.read_csv(output_path)
    if "idx" not in frame.columns:
        raise ValueError(f"Existing output has no 'idx' column: {output_path}")

    records: dict[int, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        idx = int(record["idx"])
        record["idx"] = idx
        records[idx] = record
    return records


def save_records(records: dict[int, dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [records[idx] for idx in sorted(records)]
    frame = pd.DataFrame(ordered)
    for column in OUTPUT_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame.loc[:, list(OUTPUT_COLUMNS)].to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def pending_indices(
    input_frame: pd.DataFrame,
    records: dict[int, dict[str, Any]],
    title_column: str,
    paragraph_column: str,
) -> list[int]:
    pending = []
    for idx in input_frame.index:
        source_index = int(idx)
        record = records.get(source_index)
        if record_needs_retry(record):
            pending.append(source_index)
            continue
        if (
            str(record.get("title", "")) != str(input_frame.at[idx, title_column])
            or str(record.get("paragraph", ""))
            != str(input_frame.at[idx, paragraph_column])
        ):
            pending.append(source_index)
    return pending


def merge_results(
    records: dict[int, dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    frame = results_to_frame(results)
    for record in frame.to_dict(orient="records"):
        idx = int(record["idx"])
        record["idx"] = idx
        records[idx] = record


def credits_are_depleted(results: list[dict[str, Any]]) -> bool:
    errors = [
        str(result.get("error") or "").lower()
        for result in results
        if result.get("error")
    ]
    return bool(errors) and all(
        "prepayment credits are depleted" in error for error in errors
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.chunk_size < 1 or args.semaphore_size < 1 or args.max_rounds < 1:
        raise ValueError("chunk-size, semaphore-size, and max-rounds must be positive.")

    model_name = args.model or default_model_for_provider(args.provider)
    output_path = (
        Path(args.output)
        if args.output
        else default_gpt_output_path(model_name)
    )
    input_frame = read_table(args.input)
    prompt = load_prompt(args.prompt)
    records = load_existing_records(output_path)

    for round_number in range(1, args.max_rounds + 1):
        pending = pending_indices(
            input_frame,
            records,
            args.title_column,
            args.paragraph_column,
        )
        if not pending:
            break

        print(
            f"[ROUND {round_number}/{args.max_rounds}] "
            f"{len(pending)} rows pending"
        )
        for start in range(0, len(pending), args.chunk_size):
            chunk_indices = pending[start : start + args.chunk_size]
            chunk_frame = input_frame.loc[chunk_indices]
            results = asyncio.run(
                run_all_prompt_json(
                    input_df=chunk_frame,
                    model_name=model_name,
                    provider=args.provider,
                    gpt_prompt=prompt,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    title_column=args.title_column,
                    paragraph_column=args.paragraph_column,
                    semaphore_size=args.semaphore_size,
                    timeout_seconds=args.timeout_seconds,
                )
            )
            for source_index, result in zip(chunk_indices, results):
                result["idx"] = source_index
            merge_results(records, results)
            save_records(records, output_path)
            if credits_are_depleted(results):
                print(
                    "Gemini prepayment credits are depleted. "
                    "Add credits and run this command again to resume."
                )
                return 3
            remaining = len(
                pending_indices(
                    input_frame,
                    records,
                    args.title_column,
                    args.paragraph_column,
                )
            )
            print(
                f"[CHECKPOINT] {len(records)}/{len(input_frame)} rows saved; "
                f"{remaining} pending"
            )

    remaining = pending_indices(
        input_frame,
        records,
        args.title_column,
        args.paragraph_column,
    )
    save_records(records, output_path)
    if remaining:
        print(
            f"Extraction finished with {len(remaining)} unresolved rows: "
            f"{output_path}"
        )
        return 2

    print(f"Extraction completed: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
