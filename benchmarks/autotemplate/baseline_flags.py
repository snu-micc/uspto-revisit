"""Per-record flags of the original USPTO record and gpt-5.4-none that the AutoTemplate scorers compare against.

``analysis/reproduce_manuscript.py`` writes them, for every configuration, to
``analysis/results/fully_automatic/table2_flags_{lenient,strict}.csv`` and ``table3_flags.csv``. Those
files are not distributed (about 11 MB). ``rebuild`` recomputes the rows the scorers read (``original_data`` and
``gpt-5.4-none``) offline, with the functions and settings of ``reproduce_manuscript.py`` (fully automatic scenario,
title-forming step, first reference only). ``read`` loads the files written by ``reproduce_manuscript.py`` instead.
Both return each table as ``pd.read_csv(path, keep_default_na=False)`` returns the written file.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from analysis import build_ground_truth_review as B  # noqa: E402
from analysis import evaluate_table2 as T2  # noqa: E402
from analysis import evaluate_table3 as T3  # noqa: E402
from analysis import reproduce_manuscript as R  # noqa: E402

SCENARIO = "fully_automatic"
MODEL = "gpt-5.4-none"
NAMES = ("table2_flags_lenient", "table2_flags_strict", "table3_flags")
STORED = ROOT / "analysis" / "results" / SCENARIO


def as_read(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame as the scorers read it back from the CSV that reproduce_manuscript.py writes."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    return pd.read_csv(io.StringIO(buffer.getvalue()), keep_default_na=False)


def review_for(gt: pd.DataFrame, models: tuple[str, ...] = (MODEL,)) -> pd.DataFrame:
    """reproduce_manuscript.build_review restricted to ``models``; every model column is computed per model."""
    B.configure_step_scope(title_step_only=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            B.build_queue(R.OUTPUTS / SCENARIO, path, models, R.DATA / "input.csv", Path(tmp) / "none.csv")
            review = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    finally:
        B.configure_step_scope()
    for column in [c for c in gt if c.startswith("ground_truth_")]:
        review[column] = review.idx.map(gt[column])
    review["ground_truth_valid"] = review.idx.map(gt.eligible)
    review["component_reference_valid"] = review.idx.map(gt.component_reference)
    review["review_note"] = review.idx.map(gt.noisy_reason)
    return review


def rebuild() -> dict[str, pd.DataFrame]:
    """The Table 2 (lenient, strict) and Table 3 flags of original_data and gpt-5.4-none, as in reproduce_manuscript."""
    gt = R.load_ground_truth()
    lowe = pd.read_csv(R.DATA / "input.csv", encoding="utf-8-sig").fillna("")
    review = review_for(gt)
    flags = {}
    try:
        for scoring in R.SCORINGS:
            T3.configure_matching(strict_structures=scoring == "strict", first_reference_only=True)
            f2 = T2.build_flags(review, lowe, (MODEL,))
            f2["paragraph_noise_free"] = f2.source_row.map(gt.noise_free)
            f2["structure_evaluation_eligible"] = f2.source_row.map(gt.eligible)
            flags[f"table2_flags_{scoring}"] = as_read(R.include_processing_failures(f2, gt))
    finally:
        T3.configure_matching()
    outputs = {MODEL: pd.read_csv(R.OUTPUTS / SCENARIO / f"{MODEL}_reaction_smiles_final.csv",
                                  encoding="utf-8-sig").fillna("")}
    f3 = T3.build_flags(review, T3.build_ground_truth_manifest(review), R.OUTPUTS / SCENARIO, (MODEL,),
                        result_frames=outputs)
    unavailable = gt.index[gt.noise_free & ~gt.eligible]
    f3["noise_free"] = f3.source_row.map(gt.noise_free).astype(int)
    f3["noise_label"] = np.where(f3.noise_free.eq(1), "noise-free", "noisy")
    f3["paragraph_noise_free"] = f3.noise_free
    f3["structure_evaluation_eligible"] = f3.source_row.map(gt.eligible)
    f3.loc[f3.source_row.isin(unavailable), "noise_reason"] = ""
    f3.loc[f3.source_row.isin(unavailable), "model_output_issue"] = "incomplete_molecular_smiles_processing_failure"
    flags["table3_flags"] = as_read(f3)
    return flags


def read(flags_dir: Path) -> dict[str, pd.DataFrame]:
    """The flags written by ``python analysis/reproduce_manuscript.py`` (all configurations)."""
    return {name: pd.read_csv(Path(flags_dir) / f"{name}.csv", keep_default_na=False) for name in NAMES}


def load(flags_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    return rebuild() if flags_dir is None else read(flags_dir)
