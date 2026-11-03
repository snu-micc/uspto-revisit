from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ALLOWED_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class Example:
    id: str
    messages: list[dict[str, str]]

    @property
    def reference(self) -> str:
        return self.messages[-1]["content"]

    @property
    def prompt_messages(self) -> list[dict[str, str]]:
        return self.messages[:-1]


def _validate_record(record: Any, line_number: int, source: Path) -> Example:
    prefix = f"{source}:{line_number}"
    if not isinstance(record, dict):
        raise ValueError(f"{prefix}: each line must be a JSON object")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{prefix}: messages must contain at least user and assistant")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{prefix}: messages[{index}] must be an object")
        role, content = message.get("role"), message.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"{prefix}: invalid role {role!r} at messages[{index}]")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{prefix}: empty/non-string content at messages[{index}]")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "assistant":
        raise ValueError(f"{prefix}: final message must be the assistant reference")
    if not any(message["role"] == "user" for message in normalized[:-1]):
        raise ValueError(f"{prefix}: prompt must contain a user message")
    example_id = str(record.get("id", f"line-{line_number}"))
    return Example(example_id, normalized)


def load_jsonl(path: Path) -> list[Example]:
    examples: list[Example] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            example = _validate_record(record, line_number, path)
            if example.id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {example.id!r}")
            seen.add(example.id)
            examples.append(example)
    if not examples:
        raise ValueError(f"{path}: no examples found")
    return examples


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
