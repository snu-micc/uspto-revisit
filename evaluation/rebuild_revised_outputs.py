"""Rebuild the revised model outputs from the first-release outputs and the replacement runs.

  result/model_outputs/                       outputs of the 19 configurations on the first-release input
  result/revised/replacement_runs/20261021/   the same configurations on the two records redrawn first (15, 315)
  result/revised/replacement_runs/20261022/   and on the 28 records redrawn second (they met two Table 1 criteria)

The steps follow the order in which the released outputs were built:

1. The records redrawn first replace their first-release rows. An element named as a reagent that the resolver
   returned as a single atom ([H], [N], [Cl], ...) becomes the diatomic molecule and the radical [C]=O becomes carbon
   monoxide ([C-]#[O+]).
2. Every curated component (overrides and row decisions) gets what the automatic resolvers return, replayed from the
   2026-10-16 re-query in ``result/name_resolution/`` in the released two-pass order.
3. The 28 records redrawn second replace their rows in both (their runs used automatic name resolution and already
   write elements and carbon monoxide as molecules).
4. Components that still hold the June 2026 lookup-table value ("DMA" as acetamide, record 136) get the July 2026
   entry. This gives ``curated`` and ``fully_automatic_with_tables``.
5. Every component that either lookup table answered or rewrote (the tables the resolver had then,
   ``evaluation/historical_lookup_tables.py``) gets what the resolvers return without them, replayed from the
   2026-10-23 re-query (``evaluation/lookup_tables.py``). This gives the primary ``fully_automatic`` scenario.

Steps whose components change are re-mapped with LocalMapper on the CPU. The released copies are in
``result/revised/model_outputs/`` and the audits of steps 4 and 5 in ``result/revised/audits/``. This script writes to
a separate folder and reports whether each rebuilt file equals the released one, ignoring line endings (a Windows
checkout writes CRLF, also inside the multi-line reasoning cells). Offline; needs
``python -m pip install -e ".[mapping]"``.

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
from evaluation import lookup_tables as LT  # noqa: E402

FIRST_RELEASE = ROOT / "result" / "model_outputs"
REPLACEMENT_RUNS = ROOT / "result" / "revised" / "replacement_runs"
FIRST_DRAW, SECOND_DRAW = REPLACEMENT_RUNS / "20261021", REPLACEMENT_RUNS / "20261022"
RELEASED = ROOT / "result" / "revised" / "model_outputs"
RELEASED_AUDITS = ROOT / "result" / "revised" / "audits"
INPUT = ROOT / "evaluation" / "data" / "revised" / "input.csv"
FIRST_RELEASE_INPUT = ROOT / "examples" / "input.csv"
DEFAULT_OUTPUT_DIR = ROOT / "result" / "revised" / "rebuilt"
MODELS = tuple(T3.MODELS)
SCENARIOS = ("fully_automatic", "curated", "fully_automatic_with_tables")
AUDITS = ("lookup_table_components.csv", "lookup_table_spellings.csv", "stale_dma_fix.csv")


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def replace_records(frame: pd.DataFrame, runs: Path, model: str, paragraphs: list[str]) -> pd.DataFrame:
    """The rows of the records redrawn in one draw, taken from the runs on those records."""
    replacement = read(runs / f"{model}_reaction_smiles_final.csv")
    # The first-release gpt-4.1-mini files also keep an earlier run under legacy column names that no scorer
    # reads; the replacement runs have no such run, so those cells stay blank.
    assert set(replacement.columns) <= set(frame.columns), model
    replacement = replacement.assign(**{c: "" for c in frame.columns if c not in replacement.columns})
    position = {paragraph: idx for idx, paragraph in enumerate(paragraphs)}
    frame = frame.copy()
    for _, row in replacement.iterrows():
        frame.loc[position[row.paragraph]] = row[frame.columns].values
    return frame


def write_outputs(outputs: dict[str, pd.DataFrame], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for model, frame in outputs.items():
        frame.to_csv(folder / f"{model}_reaction_smiles_final.csv", index=False, encoding="utf-8-sig")


def read_outputs(folder: Path) -> dict[str, pd.DataFrame]:
    return {model: CA.read_model_output(folder, model) for model in MODELS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    revised = read(INPUT)
    paragraphs = revised.paragraph.tolist()
    # The input after the first draw: the first release with the records redrawn first.
    redrawn_first = set(read(FIRST_DRAW / f"{MODELS[0]}_reaction_smiles_final.csv").paragraph)
    input_first = read(FIRST_RELEASE_INPUT)
    for idx in revised.index[revised.paragraph.isin(redrawn_first)]:
        input_first.loc[idx] = revised.loc[idx, input_first.columns].values

    with tempfile.TemporaryDirectory() as tmp:
        first = {"curated": Path(tmp) / "curated", "fully_automatic_with_tables": Path(tmp) / "automatic"}
        input_first.to_csv(Path(tmp) / "input.csv", index=False, encoding="utf-8-sig")

        # 1. Records redrawn first; elements and carbon monoxide as molecules
        write_outputs({model: replace_records(read(FIRST_RELEASE / f"{model}_reaction_smiles_final.csv"),
                                              FIRST_DRAW, model, input_first.paragraph.tolist())
                       for model in MODELS}, first["curated"])
        outputs = read_outputs(first["curated"])
        assert all(frame.paragraph.tolist() == input_first.paragraph.tolist() for frame in outputs.values())
        forms = CA.molecular_form_components(outputs)
        edits: dict[tuple[str, int], dict[str, str]] = {}
        for row in forms:
            edits.setdefault((row["model"], int(row["idx"])), {})[row["label"]] = row["automatic_smiles"]
        write_outputs(CA.counterfactual_outputs(outputs, edits, remap=CA.localmapper_remap), first["curated"])
        pd.DataFrame(forms).to_csv(out / "curated_molecular_forms.csv", index=False, encoding="utf-8-sig")
        print(f"step 1: {len(forms)} element or carbon monoxide components rewritten", flush=True)

        # 2. Curated components resolved automatically (every step is read, not only the title-forming one)
        B.build_queue(first["curated"], Path(tmp) / "review.csv", MODELS, Path(tmp) / "input.csv", Path(tmp) / "none.csv")
        review = pd.read_csv(Path(tmp) / "review.csv", encoding="utf-8-sig").fillna("")
        _, audit, automatic = CA.automatic_ablate(
            review, MODELS, override_path=ROOT / CA.DEFAULT_OVERRIDES,
            row_decision_path=ROOT / CA.DEFAULT_ROW_DECISIONS, order="two-pass", results_dir=first["curated"])
        write_outputs(automatic, first["fully_automatic_with_tables"])
        audit.to_csv(out / "curated_components_resolved_automatically.csv", index=False, encoding="utf-8-sig")
        print("step 2:", audit.reason.value_counts().to_dict(), flush=True)

        # 3. Records redrawn second; 4. the stale "DMA" value
        dma = []
        for scenario, folder in first.items():
            target = out / scenario
            write_outputs({model: replace_records(read(folder / f"{model}_reaction_smiles_final.csv"),
                                                  SECOND_DRAW, model, paragraphs) for model in MODELS}, target)
            outputs = read_outputs(target)
            assert all(frame.paragraph.tolist() == paragraphs for frame in outputs.values()), scenario
            assert not CA.molecular_form_components(outputs), scenario
            edits, rows = LT.stale_dma_edits(outputs)
            write_outputs(CA.counterfactual_outputs(outputs, edits, remap=CA.localmapper_remap), target)
            dma.append(rows.assign(scenario=scenario))
            print(f"steps 3-4, {scenario}: {sum(len(v) for v in edits.values())} stale DMA components", flush=True)
    dma = pd.concat(dma)
    dma[["scenario", *[c for c in dma.columns if c != "scenario"]]].to_csv(
        out / "stale_dma_fix.csv", index=False, encoding="utf-8-sig")

    # 5. Both lookup tables off
    rebuilt, components, spellings = LT.without_tables(read_outputs(out / "fully_automatic_with_tables"))
    assert not CA.molecular_form_components(rebuilt)
    write_outputs(rebuilt, out / "fully_automatic")
    components.to_csv(out / "lookup_table_components.csv", index=False, encoding="utf-8-sig")
    spellings.to_csv(out / "lookup_table_spellings.csv", index=False, encoding="utf-8-sig")
    print("step 5:", components.outcome.value_counts().to_dict(), flush=True)

    def content(path: Path) -> bytes:
        return path.read_bytes().replace(b"\r\n", b"\n")

    different = [f"{scenario}/{path.name}" for scenario in SCENARIOS
                 for path in sorted((out / scenario).glob("*.csv"))
                 if content(path) != content(RELEASED / scenario / path.name)]
    different += [name for name in AUDITS if content(out / name) != content(RELEASED_AUDITS / name)]
    print("all rebuilt outputs and audits equal the released copies" if not different else f"different: {different}")
    return int(bool(different))


if __name__ == "__main__":
    raise SystemExit(main())
