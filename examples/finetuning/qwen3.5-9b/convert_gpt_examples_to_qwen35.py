"""Convert the existing GPT chat fine-tuning records to Qwen3.5 SFT JSONL.

The output uses the standard conversational ``messages`` schema consumed by
Hugging Face datasets/TRL. Assistant targets are normalized to a JSON object
string rather than the double-encoded JSON strings present in the source file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_ROLES = ("system", "user", "assistant")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            records.append(record)
    return records


def normalize_assistant_target(content: str, line_number: int) -> str:
    """Return one compact, valid JSON object string for the assistant target."""
    value: Any = content.strip()
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            break
    if not isinstance(value, dict):
        raise ValueError(
            f"Assistant target on line {line_number} is not a JSON object after decoding."
        )
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def convert_record(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError(f"Line {line_number} must contain exactly three messages.")
    roles = tuple(message.get("role") for message in messages)
    if roles != EXPECTED_ROLES:
        raise ValueError(
            f"Line {line_number} has roles {roles}; expected {EXPECTED_ROLES}."
        )

    converted_messages = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Line {line_number} contains an empty or non-string message."
            )
        if message["role"] == "assistant":
            content = normalize_assistant_target(content, line_number)
        converted_messages.append({"role": message["role"], "content": content})
    return {"messages": converted_messages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    source_records = load_jsonl(args.input)
    converted = [
        convert_record(record, line_number)
        for line_number, record in enumerate(source_records, start=1)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in converted:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    role_counts = Counter(
        message["role"]
        for record in converted
        for message in record["messages"]
    )
    summary = {
        "source_file": str(args.input),
        "output_file": str(args.output),
        "source_records": len(source_records),
        "converted_records": len(converted),
        "role_counts": dict(role_counts),
        "assistant_targets_valid_json_objects": len(converted),
        "assistant_targets_double_encoded": 0,
        "recommended_model": "Qwen/Qwen3.5-9B",
        "recommended_training_mode": "non-thinking",
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
