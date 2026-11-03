from __future__ import annotations

import argparse
import json
import random

from .data import load_jsonl, write_jsonl
from .project import configure_local_caches, inside_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic train/eval split.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-output", default="data/train.jsonl")
    parser.add_argument("--eval-output", default="data/eval.jsonl")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.eval_fraction < 1:
        raise ValueError("--eval-fraction must be between 0 and 1")

    configure_local_caches()
    source = inside_project(args.source, must_exist=True)
    train_output = inside_project(args.train_output)
    eval_output = inside_project(args.eval_output)
    examples = load_jsonl(source)
    indexed = list(enumerate(examples, start=1))
    random.Random(args.seed).shuffle(indexed)
    eval_count = max(1, round(len(indexed) * args.eval_fraction))
    eval_indices = {index for index, _ in indexed[:eval_count]}

    def records(use_eval: bool):
        for index, example in enumerate(examples, start=1):
            if (index in eval_indices) == use_eval:
                yield {
                    "id": f"reaction-{index:03d}",
                    "messages": example.messages,
                }

    write_jsonl(train_output, records(False))
    write_jsonl(eval_output, records(True))
    print(
        f"Split {len(examples)} records into {len(examples)-eval_count} train and "
        f"{eval_count} eval examples (seed={args.seed})"
    )


if __name__ == "__main__":
    main()

