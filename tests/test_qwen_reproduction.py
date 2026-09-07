"""Check the exact released split and comparison with the original CPU-only code."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


QWEN = Path(__file__).resolve().parents[1] / "examples/finetuning/qwen3.5-9b"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_module(module, root, args):
    code = (
        "import sys; from pathlib import Path; "
        "sys.path.insert(0, sys.argv[1]); "
        "from src import project; project.ROOT=Path(sys.argv[2]); "
        f"from src import {module}; "
        f"sys.argv=[sys.argv[0]]+sys.argv[3:]; {module}.main()"
    )
    subprocess.run([sys.executable, "-B", "-c", code, str(QWEN), str(root), *args],
                   check=True, capture_output=True)


def test_original_split_and_saved_predictions_reproduce(tmp_path):
    shutil.copyfile(QWEN / "qwen3_5_9b_sft_messages.jsonl", tmp_path / "source.jsonl")
    run_module("prepare_split", tmp_path, ["--source", "source.jsonl", "--seed", "42"])
    for name, count in [("train", 56), ("eval", 14)]:
        generated = load_jsonl(tmp_path / f"data/{name}.jsonl")
        assert len(generated) == count
        assert generated == load_jsonl(QWEN / f"data/{name}.jsonl")
    references = {r["id"]: r["messages"][-1]["content"] for r in load_jsonl(QWEN / "data/eval.jsonl")}
    for name in ["baseline", "finetuned"]:
        records = load_jsonl(QWEN / f"outputs/{name}.jsonl")
        assert {r["id"]: r["reference"] for r in records} == references
        shutil.copyfile(QWEN / f"outputs/{name}.jsonl", tmp_path / f"{name}.jsonl")
    run_module("compare", tmp_path, ["--baseline", "baseline.jsonl", "--finetuned", "finetuned.jsonl",
                                     "--output", "comparison.md"])
    assert (tmp_path / "comparison.md").read_text(encoding="utf-8") == (QWEN / "outputs/comparison.md").read_text(encoding="utf-8")


def test_released_hashes_and_checkpoint_selection():
    manifest = json.loads((QWEN / "source_manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((QWEN / name).read_bytes()).hexdigest() == digest
    history = json.loads((QWEN / "evaluation_loss_history.json").read_text())
    best = min(history, key=lambda row: row["eval_loss"])
    checkpoint = json.loads((QWEN / "checkpoint-42/trainer_state.json").read_text())
    assert best["step"] == checkpoint["best_global_step"] == 42
    assert best["eval_loss"] == checkpoint["best_metric"]
