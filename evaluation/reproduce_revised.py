"""Reproduce the results of the revised manuscript and SI from the released data.

Reads
  evaluation/data/revised/ground_truth.csv                  400 paragraph labels and the structure references
  evaluation/data/revised/input.csv                         benchmark input with the original USPTO reaction SMILES
  evaluation/data/revised/benchmark_sample_membership.csv   sampling stratum and template frequency of each record
  result/revised/model_outputs/<scenario>/                   stored outputs of the 19 configurations

and writes every table, test and figure to ``evaluation/results/revised/``. Three name-resolution scenarios are
scored. ``fully_automatic`` is the primary analysis: names are resolved only by OPSIN, PubChem, NCI CIR and ChEBI,
as the released resolver does, without the two lookup tables the resolver had when the outputs were produced
(``historical_lookup_tables.py``). ``fully_automatic_with_tables`` keeps those tables (sensitivity analysis), and
``curated`` adds the released overrides and row decisions. Each prediction is compared on its step that forms the title compound. Paired
extraction tests use the selected configuration (gemini-3.6-flash, high) as reference and, separately, the
configuration with the highest accuracy. Population totals use the single-criterion sampling frame
(``benchmark_statistics.FRAME_SINGLE_CRITERION``). Percentages are rounded half up to one decimal and intervals are
95% Wilson score intervals unless stated.

Offline and deterministic: no model, name-resolver or network call. The paired tests, selection analyses and
population estimates take a few minutes on a CPU.

    python evaluation/reproduce_revised.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation import benchmark_statistics as BS  # noqa: E402
from evaluation import build_ground_truth_review as B  # noqa: E402
from evaluation import evaluate_step_by_step as SBS  # noqa: E402
from evaluation import evaluate_table2 as T2  # noqa: E402
from evaluation import evaluate_table3 as T3  # noqa: E402
from evaluation.evaluate_threshold_sensitivity import evaluate as threshold_rows  # noqa: E402
from evaluation.select_best_configurations import FAMILIES, select_best_metrics  # noqa: E402

DATA = ROOT / "evaluation" / "data" / "revised"
OUTPUTS = ROOT / "result" / "revised" / "model_outputs"
DEFAULT_OUTPUT_DIR = ROOT / "evaluation" / "results" / "revised"
MODELS = tuple(T3.MODELS)
PRIMARY = BS.PRIMARY
TEMPLATE = BS.TEMPLATE
SCENARIOS = ("fully_automatic", "curated", "fully_automatic_with_tables")
FRAME = BS.FRAME_SINGLE_CRITERION
SCORINGS = ("lenient", "strict")
UNAVAILABLE = "noise-free; outside complete molecular-SMILES reference scope"
NOISY_REASONS = ("Untraceable cross-references", "Erroneous chemical names", "no reaction")


# ------------------------------------------------------------------ formatting
def _half_up(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def pct(k: int, n: int) -> str:
    """Exact k/n as a percentage, rounded half up to one decimal."""
    if not n:
        return ""
    return _half_up(Decimal(int(k)) * 100 / Decimal(int(n)))


def pct_float(x: float) -> str:
    return _half_up(Decimal(repr(float(x) * 100)))


def acc_text(k: int, n: int) -> str:
    low, high = BS.wilson(int(k), int(n))
    return f"{pct(k, n)} ({pct_float(low)}–{pct_float(high)})"


def _plain(value):
    """NumPy scalars as plain JSON values."""
    return value.item() if hasattr(value, "item") else str(value)


def write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------- inputs
def load_ground_truth() -> pd.DataFrame:
    gt = pd.read_csv(DATA / "ground_truth.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    gt = gt.set_index(gt.idx.astype(int)).sort_index()
    gt["noise_free"] = gt.label.eq("noise-free")
    gt["eligible"] = gt.structure_evaluation_eligible.eq("True")
    return gt


def build_review(scenario: str, gt: pd.DataFrame) -> pd.DataFrame:
    """Predictions of one scenario on the title-forming step, joined with the references."""
    B.configure_step_scope(title_step_only=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            B.build_queue(OUTPUTS / scenario, path, MODELS, DATA / "input.csv", Path(tmp) / "none.csv")
            review = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    finally:
        B.configure_step_scope()
    for column in [c for c in gt if c.startswith("ground_truth_")]:
        review[column] = review.idx.map(gt[column])
    review["ground_truth_valid"] = review.idx.map(gt.eligible)
    review["review_note"] = review.idx.map(gt.noisy_reason.where(~gt.noise_free, gt.structure_exclusion_reason))
    return review


def replaced_records(lowe: pd.DataFrame) -> list[int]:
    """Records redrawn after the first release (their paragraphs differ from examples/input.csv)."""
    first = pd.read_csv(ROOT / "examples" / "input.csv", encoding="utf-8-sig").fillna("")
    return [int(i) for i in np.flatnonzero(first.paragraph.ne(lowe.paragraph))]


# ---------------------------------------------------------------------- tables
def table2_rows(metrics: pd.DataFrame, strict: pd.DataFrame, methods) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_method, strict_by_method = metrics.set_index("method"), strict.set_index("method")
    raw, formatted = [], []
    for method in methods:
        row = by_method.loc[method]
        record = {"method": method, "extraction_method": T2.METHOD_LABELS.get(method, method),
                  "model_family": FAMILIES.get(method, "Original data")}
        shown = {"Extraction method": record["extraction_method"]}
        for field in ("reactants", "reagents", "products"):
            tp, fp, fn = (int(row[f"{field}_{k}"]) for k in ("tp", "fp", "fn"))
            record.update({f"{field}_tp": tp, f"{field}_fp": fp, f"{field}_fn": fn,
                           f"{field}_precision": tp / (tp + fp) if tp + fp else float("nan"),
                           f"{field}_recall": tp / (tp + fn) if tp + fn else float("nan")})
            shown[f"{field.capitalize()} precision (%)"] = pct(tp, tp + fp)
            shown[f"{field.capitalize()} recall (%)"] = pct(tp, tp + fn)
        k, n = int(row["reaction_correct"]), int(row["evaluated_reactions"])
        sk = int(strict_by_method.loc[method, "reaction_correct"])
        low, high = BS.wilson(k, n)
        slow, shigh = BS.wilson(sk, n)
        record.update(reaction_correct=k, evaluated=n, reaction_accuracy=k / n, ci_low=low, ci_high=high,
                      strict_correct=sk, strict_accuracy=sk / n, strict_ci_low=slow, strict_ci_high=shigh)
        shown.update({"Correct / evaluated": f"{k}/{n}", "Reaction accuracy (%, 95% CI)": acc_text(k, n),
                      "Strict correct / evaluated": f"{sk}/{n}",
                      "Strict reaction accuracy (%, 95% CI)": acc_text(sk, n)})
        raw.append(record)
        formatted.append(shown)
    return pd.DataFrame(raw), pd.DataFrame(formatted)


def threshold_tables(flags3: pd.DataFrame, frequency: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Template-anomaly filtering with an added template-frequency cutoff (selected configuration)."""
    selected = flags3[flags3["model"].eq(PRIMARY)].set_index("source_row").sort_index()
    rows = [{"template_frequency": int(frequency.loc[idx]), "template_failure": int(row["template_failure"]),
             "identical_structures": int(row["identical_structures"]), "noise_free": int(row["noise_free"])}
            for idx, row in selected.iterrows()]
    filtered = [bool(r["template_failure"] or r["identical_structures"]) for r in rows]
    tp = sum(r["noise_free"] and not f for r, f in zip(rows, filtered))
    fn = sum(r["noise_free"] for r in rows) - tp
    tn = sum(not r["noise_free"] and f for r, f in zip(rows, filtered))
    fp = sum(not r["noise_free"] for r in rows) - tn
    primary = {"template_anomaly_frequency_threshold": "none (primary criterion)", "noise_free_reactions": tp + fn,
               "noisy_reactions": fp + tn, "total_reactions": tp + fn + fp + tn, "passed_tp": tp, "filtered_fn": fn,
               "passed_fp": fp, "filtered_tn": tn, "noise_free_data_saved_rate": tp / (tp + fn),
               "noisy_data_filtered_rate": tn / (tn + fp), "denoising_accuracy": (tp + tn) / (tp + fn + fp + tn)}
    raw = pd.concat([pd.DataFrame([primary]), pd.DataFrame(threshold_rows(rows))], ignore_index=True)
    raw.insert(0, "model", PRIMARY)
    formatted = pd.DataFrame([{
        "Frequency cutoff": r.template_anomaly_frequency_threshold,
        "Passed (TP)": r.passed_tp, "Filtered (FN)": r.filtered_fn, "Passed (FP)": r.passed_fp,
        "Filtered (TN)": r.filtered_tn,
        "Noise-free data saved rate (%)": pct(r.passed_tp, r.passed_tp + r.filtered_fn),
        "Noisy data filtered rate (%)": pct(r.filtered_tn, r.filtered_tn + r.passed_fp),
        "Denoising accuracy (%, 95% CI)": acc_text(r.passed_tp + r.filtered_tn, r.total_reactions),
    } for r in raw.itertuples()])
    return raw, formatted


def end_to_end_bounds(flags2: pd.DataFrame, flags3: pd.DataFrame, gt: pd.DataFrame, strata: pd.Series,
                      scoring: str) -> tuple[list[dict], list[dict]]:
    """Whole-reaction success on all 400 records, bounding the references that cannot be written as SMILES.

    A noise-free record counts when it passes the filter and its extraction is correct, a noisy record when it is
    removed. Population totals of the selected configuration and filter are stratified estimates.
    """
    labels, eligible = gt.noise_free.to_numpy(), gt.eligible.to_numpy()
    correct = flags2.pivot(index="source_row", columns="method", values="reaction_correct")
    correct = correct.apply(pd.to_numeric, errors="coerce").fillna(0).sort_index()
    assessed = ~labels | eligible
    bounds, population = [], []
    for model in MODELS:
        model_flags = flags3[flags3.model.eq(model)].set_index("source_row").sort_index()
        right = correct[model].eq(1).to_numpy()
        for heuristic in T3.HEURISTICS:
            passed = model_flags[heuristic + "_passed"].astype(bool).to_numpy()
            lower = np.where(labels, right & passed, ~passed).astype(int)
            unknown = labels & ~eligible & passed
            upper = np.where(unknown, 1, lower)
            k, n = int(lower[assessed].sum()), int(assessed.sum())
            low, high = BS.wilson(k, n)
            bounds.append({"model": model, "heuristic": heuristic, "scoring": scoring, "assessable_correct": k,
                           "assessable_n": n, "assessable_accuracy": k / n, "ci_low": low, "ci_high": high,
                           "full_n": len(labels), "lower_correct": int(lower.sum()), "upper_correct": int(upper.sum()),
                           "lower_accuracy": lower.mean(), "upper_accuracy": upper.mean(),
                           "unresolved_retained": int(unknown.sum())})
            if model == PRIMARY and heuristic == TEMPLATE:
                for name, y in (("exact_retained_lower", labels & right & passed),
                                ("exact_retained_upper", labels & (right | ~eligible) & passed),
                                ("noise_free_retained", labels & passed),
                                ("noisy_removed", ~labels & ~passed)):
                    estimate = BS.stratified_total(y.astype(int), strata.loc[gt.index].to_numpy(), FRAME)
                    population.append({"quantity": name, "scoring": scoring, **estimate})
    return bounds, population


def denoising_tests(flags3: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Template anomaly against each ORDerly heuristic on all 400 paragraph labels (exact McNemar, Holm)."""
    selected = flags3[flags3.model.eq(PRIMARY)].set_index("source_row").sort_index()
    correct = {h: selected[h + "_passed"].eq(gt.noise_free.astype(int)).astype(int) for h in T3.HEURISTICS}
    rows = []
    for heuristic in T3.HEURISTICS:
        if heuristic == TEMPLATE:
            continue
        template_only, comparator_only, p = BS.mcnemar_exact(correct[TEMPLATE], correct[heuristic])
        rows.append({"heuristic": heuristic, "n": len(gt), "template_correct": int(correct[TEMPLATE].sum()),
                     "comparator_correct": int(correct[heuristic].sum()), "template_only": template_only,
                     "comparator_only": comparator_only, "p_exact": p})
    tests = pd.DataFrame(rows)
    tests["p_holm"] = BS.holm(tests.p_exact)
    return tests


def weighted_extrapolation(flags3: pd.DataFrame, strata: pd.Series) -> pd.DataFrame:
    """Template-anomaly outcomes of the selected configuration per stratum, weighted to the sampling frame."""
    selected = flags3[flags3.model.eq(PRIMARY)].set_index("source_row").sort_index()
    noise_free, passed = selected.noise_free.eq(1), selected[TEMPLATE + "_passed"].astype(bool)
    outcomes = {"tp": noise_free & passed, "fn": noise_free & ~passed, "fp": ~noise_free & passed,
                "tn": ~noise_free & ~passed}
    rows = []
    for stratum, population in FRAME.items():
        members = strata.index[strata.eq(stratum)]
        counts = {k: int(v.loc[members].sum()) for k, v in outcomes.items()}
        weight = population / len(members)
        rows.append({"stratum": BS.STRATUM_LABELS[stratum], "population": population, "sample": len(members),
                     "weight": weight, **counts, **{f"weighted_{k}": n * weight for k, n in counts.items()}})
    frame = pd.DataFrame(rows)
    total = {c: frame[c].sum() for c in frame.columns if c not in ("stratum", "weight")}
    return pd.concat([frame, pd.DataFrame([{"stratum": "Total", **total}])], ignore_index=True)


# -------------------------------------------------------------------- scenario
def score_scenario(scenario: str, gt: pd.DataFrame, lowe: pd.DataFrame, membership: pd.DataFrame,
                   out: Path) -> dict:
    labels, eligible = gt.noise_free, gt.eligible
    unavailable = gt.index[labels & ~eligible]
    strata = membership.issue_label
    review = build_review(scenario, gt)
    outputs = {m: pd.read_csv(OUTPUTS / scenario / f"{m}_reaction_smiles_final.csv", encoding="utf-8-sig").fillna("")
               for m in MODELS}

    flags2, metrics = {}, {}
    try:
        for scoring in SCORINGS:
            T3.configure_matching(strict_structures=scoring == "strict", first_reference_only=scoring == "strict")
            f2 = T2.build_flags(review, lowe, MODELS)
            f2["paragraph_noise_free"] = f2.source_row.map(labels)
            f2["structure_evaluation_eligible"] = f2.source_row.map(eligible)
            f2.loc[f2.source_row.isin(unavailable), "exclusion_reason"] = UNAVAILABLE
            flags2[scoring], metrics[scoring] = f2, T2.build_metrics(f2, (T2.ORIGINAL_METHOD, *MODELS))
            write(f2, out / f"table2_flags_{scoring}.csv")
            write(metrics[scoring], out / f"table2_metrics_{scoring}.csv")
    finally:
        T3.configure_matching()

    f3 = T3.build_flags(review, T3.build_ground_truth_manifest(review), OUTPUTS / scenario, MODELS,
                        result_frames=outputs)
    f3["noise_free"] = f3.source_row.map(labels).astype(int)
    f3["noise_label"] = np.where(f3.noise_free.eq(1), "noise-free", "noisy")
    f3["paragraph_noise_free"] = f3.noise_free
    f3["structure_evaluation_eligible"] = f3.source_row.map(eligible)
    f3.loc[f3.source_row.isin(unavailable), "noise_reason"] = ""
    f3.loc[f3.source_row.isin(unavailable), "model_output_issue"] = "not_scored_reference_representation_unavailable"
    table3 = T3.build_metrics(f3, MODELS)
    write(f3, out / "table3_flags.csv")
    write(table3, out / "table3_metrics.csv")

    best = tuple(select_best_metrics(metrics["lenient"]).method)
    for name, methods in (("table2", best), ("table_s18", (T2.ORIGINAL_METHOD, *MODELS))):
        raw, formatted = table2_rows(metrics["lenient"], metrics["strict"], methods)
        write(raw, out / f"{name}_raw.csv")
        write(formatted, out / f"{name}.csv")

    # Paired tests and selection analyses use the records whose references are complete or not needed.
    bounds, population, e2e_tests, selection = [], [], [], []
    for scoring, f2 in flags2.items():
        f = BS.RecordFlags(f2[~f2.source_row.isin(unavailable)], f3[~f3.source_row.isin(unavailable)], strata, MODELS)
        write(BS.table2_tests(f).assign(scenario=scenario, scoring=scoring), out / f"extraction_paired_tests_{scoring}.csv")
        write(BS.table2_tests(f, reference=PRIMARY).assign(scenario=scenario, scoring=scoring),
              out / f"extraction_paired_tests_primary_{scoring}.csv")
        tests = BS.table3_tests(f)
        e2e_tests.append(tests[tests.outcome.eq("end_to_end")].assign(scoring=scoring, scope="assessable_subset"))
        for kind in ("extraction", "two_stage", "joint"):
            record, _ = BS.optimism(f, kind)
            selection.append({**record, "scoring": scoring})
        b, p = end_to_end_bounds(f2, f3, gt, strata, scoring)
        bounds += b
        population += p
    write(denoising_tests(f3, gt), out / "denoising_paired_tests.csv")
    write(pd.concat(e2e_tests), out / "end_to_end_paired_tests.csv")
    write(pd.DataFrame(bounds), out / "end_to_end_bounds.csv")
    write(pd.DataFrame(population), out / "population_bounds.csv")
    write(pd.DataFrame(selection), out / "selection_adjustment.csv")
    write(weighted_extrapolation(f3, strata), out / "weighted_extrapolation.csv")
    raw, formatted = threshold_tables(f3, membership.template_frequency)
    write(raw, out / "threshold_sensitivity_raw.csv")
    write(formatted, out / "threshold_sensitivity.csv")

    steps = SBS.build_row_flags(review, outputs[PRIMARY], flags2["lenient"], f3, PRIMARY)
    steps["paragraph_noise_free"] = steps.source_row.map(labels)
    write(steps, out / "step_flags_structure_endpoint.csv")
    write(SBS.build_noise_free_step_table(steps), out / "step_table_structure_eligible.csv")
    lenient = metrics["lenient"]
    return {"review": review, "flags2": flags2, "flags3": f3, "table3": table3,
            "summary": {"primary": lenient.set_index("method").loc[PRIMARY].to_dict(),
                        "best_method": lenient.loc[lenient.reaction_accuracy.idxmax(), "method"]}}


# --------------------------------------------------------------- supplementary
def ai_baseline(review: pd.DataFrame, gt: pd.DataFrame, lowe: pd.DataFrame, replaced: list[int]) -> pd.DataFrame:
    """Ai et al. baseline (exact and coverage scoring); it was not run on the replaced records."""
    review = review.copy()
    review["ground_truth_valid"] = review.idx.map(gt.eligible) & ~review.idx.isin(replaced)
    components = pd.read_csv(ROOT / "benchmarks" / "ai_et_al" / "preprocessing" / "ai_baseline_components.csv",
                             encoding="utf-8-sig").fillna("")
    for idx in replaced:
        row = components.index[components.idx.eq(idx)][0]
        components.loc[row, ["title", "paragraph"]] = gt.loc[idx, ["title", "paragraph"]].values
        for column in [c for c in components if c.startswith(T2.AI_METHOD)]:
            components.at[row, column] = 1 if column.endswith("has_unresolved_nosmi") else ""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ai_baseline_components.csv"
        components.to_csv(path, index=False, encoding="utf-8-sig")
        review = T2.merge_ai_components(review, path)
    metrics = []
    for scoring in ("exact", "coverage"):
        flags = T2.build_flags(review, lowe, (T2.AI_METHOD,), ai_scoring=scoring)
        metrics.append(T2.build_metrics(flags, (T2.AI_METHOD,)).assign(scoring=scoring))
    return pd.concat(metrics)


def provenance_accuracy(flags2: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    flags = flags2[flags2.evaluation_included.eq(1)].assign(source=lambda f: f.source_row.map(gt.reference_source))
    rows = []
    for (method, source), group in flags.groupby(["method", "source"]):
        k, n = int(pd.to_numeric(group.reaction_correct).sum()), len(group)
        rows.append({"method": method, "source": source, "correct": k, "n": n, "accuracy": k / n,
                     "formatted": acc_text(k, n)})
    return pd.DataFrame(rows)


def filtering_by_reason(flags3: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    selected = flags3[flags3.model.eq(PRIMARY) & flags3.noise_free.eq(0)]
    selected = selected.assign(category=selected.source_row.map(gt.noisy_reason))
    rows = []
    for category, group in selected.groupby("category"):
        for heuristic in T3.HEURISTICS:
            rows.append({"category": category, "heuristic": heuristic, "n": len(group),
                         "filtered": int(group[heuristic + "_filtered"].sum()),
                         "rate": group[heuristic + "_filtered"].mean()})
    return pd.DataFrame(rows)


def label_changes(gt: pd.DataFrame, replaced: list[int]) -> pd.DataFrame:
    """Paragraph labels that differ from the first release (replaced records are different paragraphs)."""
    first = pd.read_csv(ROOT / "evaluation" / "ground_truth_review.csv", encoding="utf-8-sig").set_index("idx")
    published = gt.index.map(first.ground_truth_valid).astype(bool)
    changed = (published != gt.noise_free) & ~gt.index.isin(replaced)
    frame = pd.DataFrame({"idx": gt.index, "title": gt.title,
                          "first_release": np.where(published, "noise-free", "noisy"), "revised": gt.label})
    return frame[changed]


def salt_sensitivity(flags3: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Template-anomaly denoising if the noise-free salt-formation paragraphs were labeled noisy instead."""
    selected = flags3[flags3.model.eq(PRIMARY)].set_index("source_row").sort_index()
    passed = selected[TEMPLATE + "_passed"].astype(bool)
    correct = int(passed.eq(gt.noise_free).sum())
    salts = gt.index[gt.salt_formation.eq("True") & gt.noise_free]
    filtered = int((~passed.loc[salts]).sum())
    return {"salt_formation_noise_free": len(salts), "salt_formation_noisy": int((gt.salt_formation.eq("True")
            & ~gt.noise_free).sum()), "template_anomaly_filtered": filtered, "denoising_correct": correct,
            "denoising_correct_if_noisy": correct + filtered - (len(salts) - filtered), "records": len(gt)}


# --------------------------------------------------------------------- figures
def figure1(table3: pd.DataFrame, path: Path) -> None:
    """Stage 2 panel of Figure 1: retained and filtered noise-free/noisy reactions mirrored on the left, the
    heuristics in the middle and denoising accuracy on the right, sorted by accuracy (selected configuration)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    blue, red, purple = "#36A3D9", "#D9574F", "#7657B5"
    rows = table3[table3.model.eq(PRIMARY)]
    counts = pd.DataFrame({
        "heuristic": rows.heuristic,
        "saved": list(zip(rows.passed_tp, rows.passed_tp + rows.filtered_fn)),
        "filtered": list(zip(rows.filtered_tn, rows.filtered_tn + rows.passed_fp)),
        "accuracy": list(zip(rows.passed_tp + rows.filtered_tn,
                             rows.passed_tp + rows.filtered_fn + rows.passed_fp + rows.filtered_tn)),
    })
    counts = counts.assign(order=[k / n for k, n in counts.accuracy]).sort_values("order", ascending=False)
    spacing, pair = 1.18, 0.34
    centers = [i * spacing for i in range(len(counts))]
    upper, lower = [c - pair / 2 for c in centers], [c + pair / 2 for c in centers]

    def style(ax, reverse=False):
        ax.set_xlim((100, 0) if reverse else (0, 100))
        ax.set_xticks([100, 50, 0] if reverse else [0, 50, 100])
        ax.tick_params(axis="x", labelsize=36, length=7, width=1.4, color="#595959", labelcolor="#262626", pad=8)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.grid(axis="x", color="#D0D4DC", linewidth=1.0, zorder=0)
        for side, spine in ax.spines.items():
            spine.set_visible(side == "bottom")
        ax.spines["bottom"].set_linewidth(1.4)
        ax.spines["bottom"].set_color("#595959")

    def bars(ax, values, positions, height, color, size, reverse=False):
        """Bars with the value in white inside, or just beyond the end of a short bar."""
        ax.barh(positions, [100 * k / n for k, n in values], height, color=color, edgecolor="none", zorder=3)
        for (k, n), y in zip(values, positions):
            v, inside = 100 * k / n, 100 * k / n >= 10
            ax.text(v - 1.6 if inside else v + 1.2, y, pct(k, n), va="center", fontsize=size, fontweight="bold",
                    ha=("left" if reverse else "right") if inside else ("right" if reverse else "left"),
                    color="white" if inside else "#404040", zorder=5)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(31, 13.5), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 0.44, 1.0), wspace=0.08)
    left = fig.add_subplot(grid[0, 0])
    middle = fig.add_subplot(grid[0, 1], sharey=left)
    right = fig.add_subplot(grid[0, 2], sharey=left)
    style(left, reverse=True)
    style(right)
    bars(left, counts.saved, upper, pair, blue, 28, reverse=True)
    bars(left, counts.filtered, lower, pair, red, 28, reverse=True)
    bars(right, counts.accuracy, centers, 0.68, purple, 34)
    left.set_ylim(-0.72, centers[-1] + 0.72)
    left.invert_yaxis()
    middle.set_xlim(0, 1)
    middle.axis("off")
    for y, heuristic in zip(centers, counts.heuristic):
        middle.text(0.5, y, T3.HEURISTIC_LABELS[heuristic].replace("-", " ").capitalize(), ha="center", va="center",
                    fontsize=40, fontweight="bold")
    for ax, title in ((left, "Saved and filtered reactions"), (middle, "Heuristic"), (right, "Denoising accuracy")):
        ax.set_title(title, fontsize=46, fontweight="bold", pad=22)
    fig.legend(handles=[Patch(facecolor=blue, label="Clean saved"), Patch(facecolor=red, label="Noisy filtered")],
               loc="lower left", bbox_to_anchor=(0.08, 0.025), ncol=2, frameon=False, fontsize=36, handlelength=1.1,
               handletextpad=0.5, columnspacing=1.2)
    fig.supxlabel("Percentage (%)", fontsize=40, y=0.105)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.20)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_s2(by_reason: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = dict(zip(NOISY_REASONS, ("#3b8d9c", "#da9750", "#8474b4")))
    counts = by_reason.set_index(["category", "heuristic"])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(10.8, 5.8), facecolor="white")
    for i, heuristic in enumerate(T3.HEURISTICS):
        for offset, filtered in ((-.19, True), (.19, False)):
            bottom = 0
            for category, color in colors.items():
                row = counts.loc[(category, heuristic)]
                value = int(row.filtered) if filtered else int(row.n) - int(row.filtered)
                ax.bar(i + offset, value, .34, bottom=bottom, color=color, linewidth=.5,
                       edgecolor="white" if filtered else "#555555", hatch=None if filtered else "//")
                if value >= 4:
                    ax.text(i + offset, bottom + value / 2, str(value), ha="center", va="center", color="white", fontsize=9)
                bottom += value
            ax.text(i + offset, bottom + 1, str(bottom), ha="center", fontsize=9)
    total = int(by_reason[by_reason.heuristic.eq(TEMPLATE)].n.sum())
    ax.set_xticks(range(len(T3.HEURISTICS)), [T3.HEURISTIC_LABELS[h] for h in T3.HEURISTICS])
    ax.set_ylabel(f"Noisy paragraphs (n = {total})")
    ax.set_ylim(0, total + 8)
    ax.spines[["top", "right"]].set_visible(False)
    legend = [Patch(facecolor=color, label=category) for category, color in colors.items()]
    legend += [Patch(facecolor="white", edgecolor="#555555", label="Filtered (TN)"),
               Patch(facecolor="white", edgecolor="#555555", hatch="//", label="Retained (FP)")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.08, right=.99, top=.97, bottom=.29)
    fig.savefig(path, dpi=240)
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth()
    lowe = pd.read_csv(DATA / "input.csv", encoding="utf-8-sig").fillna("")
    membership = pd.read_csv(DATA / "benchmark_sample_membership.csv", encoding="utf-8-sig").set_index("idx").sort_index()
    assert gt.index.equals(membership.index) and gt.paragraph.tolist() == lowe.paragraph.tolist()
    replaced = replaced_records(lowe)

    results = {}
    for scenario in SCENARIOS:
        print(f"scoring {scenario} ...", flush=True)
        results[scenario] = score_scenario(scenario, gt, lowe, membership, out / scenario)
    primary = results["fully_automatic"]
    flags3 = primary["flags3"]

    write(ai_baseline(primary["review"], gt, lowe, replaced), out / "ai_metrics.csv")
    write(provenance_accuracy(primary["flags2"]["lenient"], gt), out / "provenance_accuracy.csv")
    reasons = gt.noisy_reason[~gt.noise_free].value_counts().rename_axis("category").reset_index(name="count")
    write(reasons, out / "noisy_reason_counts.csv")
    by_reason = filtering_by_reason(flags3, gt)
    write(by_reason, out / "filtering_by_reason.csv")
    write(label_changes(gt, replaced), out / "label_changes.csv")
    figure1(primary["table3"], out / "Figure1.png")
    figure_s2(by_reason, out / "FigureS2.png")

    unavailable = gt.index[gt.noise_free & ~gt.eligible]
    summary = {"records": len(gt), "noise_free": int(gt.noise_free.sum()), "noisy": int((~gt.noise_free).sum()),
               "structure_eligible": int(gt.eligible.sum()), "structure_unavailable": [int(i) for i in unavailable],
               "replaced_records": replaced, "salt_formation": salt_sensitivity(flags3, gt),
               **{scenario: result["summary"] for scenario, result in results.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_plain), encoding="utf-8")
    inputs = [*sorted(DATA.glob("*.csv")), *sorted(OUTPUTS.rglob("*.csv"))]
    # Line endings are normalized, so the hashes do not depend on how Git checked the files out.
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for p in inputs}
    (out / "input_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_plain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
