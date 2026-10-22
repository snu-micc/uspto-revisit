"""Rebuild the revised model outputs from the first-release outputs and the replacement runs.

  result/model_outputs/               outputs of the 19 configurations on the first-release input
  result/revised/replacement_runs/    the same configurations on the two redrawn records (15 and 315)

become ``curated/`` (the two records replaced) and ``fully_automatic/`` (every curated component replaced by
what the automatic resolvers return, replayed from the recorded re-query in ``result/name_resolution/`` in the
released two-pass order). In both, an element named as a reagent that the resolver returned as a single atom
([H], [N], [Cl], ...) becomes the diatomic molecule and the radical [C]=O becomes carbon monoxide ([C-]#[O+]);
steps whose components change are re-mapped with LocalMapper on the CPU.

The released copies are in ``result/revised/model_outputs/``. This script writes to a separate folder and reports
whether each rebuilt file equals the released one, ignoring line endings (a Windows checkout writes CRLF, also
inside the multi-line reasoning cells). Offline; needs ``python -m pip install -e ".[mapping]"``.

    python evaluation/rebuild_revised_outputs.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd  # noqa: E402

from evaluation import build_ground_truth_review as B  # noqa: E402
from evaluation import curation_ablation as CA  # noqa: E402
from evaluation import evaluate_table3 as T3  # noqa: E402

FIRST_RELEASE = ROOT / "result" / "model_outputs"
REPLACEMENT_RUNS = ROOT / "result" / "revised" / "replacement_runs"
RELEASED = ROOT / "result" / "revised" / "model_outputs"
INPUT = ROOT / "evaluation" / "data" / "revised" / "input.csv"
DEFAULT_OUTPUT_DIR = ROOT / "result" / "revised" / "rebuilt"
MODELS = tuple(T3.MODELS)


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def replace_records(model: str, paragraphs: list[str]) -> pd.DataFrame:
    """First-release outputs with the redrawn records taken from the replacement runs."""
    name = f"{model}_reaction_smiles_final.csv"
    frame, runs = read(FIRST_RELEASE / name), read(REPLACEMENT_RUNS / name)
    # The first-release gpt-4.1-mini files also keep an earlier run under legacy column names that no scorer
    # reads; the replacement runs have no such run, so those cells stay blank.
    runs = runs.assign(**{c: "" for c in frame.columns if c not in runs.columns})
    for idx, paragraph in enumerate(paragraphs):
        if frame.at[idx, "paragraph"] != paragraph:
            match = runs[runs.paragraph.eq(paragraph)]
            assert len(match) == 1, (model, idx)
            frame.loc[idx] = match.iloc[0][frame.columns].values
    assert frame.paragraph.tolist() == paragraphs, model
    return frame


def write_outputs(outputs: dict[str, pd.DataFrame], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for model, frame in outputs.items():
        frame.to_csv(folder / f"{model}_reaction_smiles_final.csv", index=False, encoding="utf-8-sig")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    out = args.output_dir
    paragraphs = pd.read_csv(INPUT, encoding="utf-8-sig").fillna("").paragraph.tolist()

    curated = out / "curated"
    write_outputs({model: replace_records(model, paragraphs) for model in MODELS}, curated)
    outputs = {model: CA.read_model_output(curated, model) for model in MODELS}
    forms = CA.molecular_form_components(outputs)
    edits: dict[tuple[str, int], dict[str, str]] = {}
    for row in forms:
        edits.setdefault((row["model"], int(row["idx"])), {})[row["label"]] = row["automatic_smiles"]
    write_outputs(CA.counterfactual_outputs(outputs, edits, remap=CA.localmapper_remap), curated)
    pd.DataFrame(forms).to_csv(out / "curated_molecular_forms.csv", index=False, encoding="utf-8-sig")
    print(f"curated: {len(forms)} element or carbon monoxide components rewritten", flush=True)

    # The automatic replacement reads every step, not only the title-forming one.
    with tempfile.TemporaryDirectory() as tmp:
        B.build_queue(curated, Path(tmp) / "review.csv", MODELS, INPUT, Path(tmp) / "none.csv")
        review = pd.read_csv(Path(tmp) / "review.csv", encoding="utf-8-sig").fillna("")
    _, audit, automatic = CA.automatic_ablate(
        review, MODELS, override_path=ROOT / CA.DEFAULT_OVERRIDES,
        row_decision_path=ROOT / CA.DEFAULT_ROW_DECISIONS, order="two-pass", results_dir=curated)
    write_outputs(automatic, out / "fully_automatic")
    audit.to_csv(out / "fully_automatic_audit.csv", index=False, encoding="utf-8-sig")
    print("fully automatic:", audit.reason.value_counts().to_dict(), flush=True)

    def content(path: Path) -> bytes:
        return path.read_bytes().replace(b"\r\n", b"\n")

    different = [f"{scenario}/{path.name}" for scenario in ("curated", "fully_automatic")
                 for path in sorted((out / scenario).glob("*.csv"))
                 if content(path) != content(RELEASED / scenario / path.name)]
    print("all rebuilt outputs equal the released copies" if not different else f"different: {different}")
    return int(bool(different))


if __name__ == "__main__":
    raise SystemExit(main())
