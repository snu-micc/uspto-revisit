from __future__ import annotations

import argparse
import statistics

from .data import load_jsonl
from .project import configure_local_caches, inside_project


def summarize(label: str, path: str) -> None:
    resolved = inside_project(path, must_exist=True)
    examples = load_jsonl(resolved)
    turns = [len(example.messages) for example in examples]
    characters = [sum(len(message["content"]) for message in example.messages) for example in examples]
    print(
        f"{label}: {len(examples)} examples, "
        f"turns median={statistics.median(turns):g}, "
        f"characters median={statistics.median(characters):g}, max={max(characters)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate train/evaluation JSONL files.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    args = parser.parse_args()
    configure_local_caches()
    summarize("train", args.train)
    summarize("eval", args.eval)


if __name__ == "__main__":
    main()

