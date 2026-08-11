import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_llm_resume.py"
SPEC = importlib.util.spec_from_file_location("run_llm_resume", MODULE_PATH)
run_llm_resume = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_llm_resume)


def test_pending_indices_retries_mismatched_checkpoint_rows():
    input_frame = pd.DataFrame(
        {
            "title": ["correct title", "second title"],
            "paragraph": ["correct paragraph", "second paragraph"],
        }
    )
    records = {
        0: {
            "idx": 0,
            "title": "correct title",
            "paragraph": "correct paragraph",
            "prediction": "{}",
            "error": None,
        },
        1: {
            "idx": 1,
            "title": "wrong title",
            "paragraph": "wrong paragraph",
            "prediction": "{}",
            "error": None,
        },
    }

    assert run_llm_resume.pending_indices(
        input_frame,
        records,
        "title",
        "paragraph",
    ) == [1]
