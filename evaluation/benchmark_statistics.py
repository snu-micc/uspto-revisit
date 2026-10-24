"""Paired tests, selection optimism and stratified population estimates.

Used by ``reproduce_revised.py``. All functions work on per-record 0/1
matrices (``RecordFlags``) built from the Table 2 and Table 3 flag frames.

* Exact (binomial) McNemar tests with a Holm correction inside each family:
  the best (or the selected) configuration against each of the other 18 and
  against the original USPTO record (Table 2), the template anomaly against each ORDerly
  heuristic for the selected configuration (denoising decision and
  end-to-end outcome), and every configuration against the selected one
  (end-to-end).
* Selection optimism: the 400 records both chose and scored the
  configuration. Nested 5-fold cross-validation stratified by sampling
  stratum (200 repeats), leave-one-stratum-out, and the out-of-bag stratified
  bootstrap (2,000 resamples; OOB mean, .632 estimate and Harrell's
  optimism-corrected estimate) estimate the accuracy of the selection
  procedure. Ties in the inner selection are averaged.
* Stratified population totals with finite-population correction and a 95%
  Wald interval, and the end-to-end total when the configuration is chosen
  out of fold.

The random streams (seed 20261016, stratum order, fold assignment) are fixed,
so the results are reproducible to the last digit.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

ORIGINAL = "original_data"
# The selected configuration: the highest extraction accuracy in the primary (fully automatic) scenario of the
# revised benchmark. The first release selected gemini-3.6-flash at high reasoning.
PRIMARY = "gpt-5.4-none"
FIRST_RELEASE_PRIMARY = "gemini-3.6-flash-high"
TEMPLATE = "rare_template"
HEURISTICS = ("many_products", "many_reactants", "many_reagents", "no_reagent", "rare_template")
STRATA = ("many_product (> 1)", "many_reactant (> 2)", "many_reagent (> 5)", "no_reagent")
STRATUM_LABELS = {
    "many_product (> 1)": "Many-products",
    "many_reactant (> 2)": "Many-reactants",
    "many_reagent (> 5)": "Many-reagents",
    "no_reagent": "No-reagent",
}
# Frames of the first release, by the released Issue flags: exactly one flag and no other label, and exactly one of
# the four flags with rare_temp or duplicated allowed.
FRAME_SINGLE_LABEL = {
    "many_product (> 1)": 59282,
    "many_reactant (> 2)": 20163,
    "many_reagent (> 5)": 207221,
    "no_reagent": 167809,
}
FRAME_EXACTLY_ONE = {
    "many_product (> 1)": 67030,
    "many_reactant (> 2)": 30709,
    "many_reagent (> 5)": 224249,
    "no_reagent": 180009,
}
# Frame of the revised benchmark: exactly one Table 1 criterion counted from the reaction SMILES and the Reagent
# field, and no rare_temp or duplicated label (build_sampling_frame.py).
FRAME_SINGLE_CRITERION = {
    "many_product (> 1)": 44157,
    "many_reactant (> 2)": 20157,
    "many_reagent (> 5)": 198305,
    "no_reagent": 167809,
}
Z95 = 1.959963984540054
SEED = 20261016
REPEATS = 200
BOOT = 2000


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval, as in ``evaluate_table2.wilson_interval``."""
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_exact(a, b) -> tuple[int, int, float]:
    """Exact two-sided McNemar (binomial) test on paired 0/1 vectors."""
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    return only_a, only_b, float(min(1.0, 2 * stats.binom.cdf(min(only_a, only_b), n, 0.5)))


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


class RecordFlags:
    """Per-record matrices for one scenario and scoring."""

    def __init__(
        self,
        flags2: pd.DataFrame,
        flags3: pd.DataFrame,
        strata: pd.Series,
        models: tuple[str, ...],
    ):
        frame = flags2[["method", "source_row", "evaluation_included", "reaction_correct"]].copy()
        frame["reaction_correct"] = pd.to_numeric(frame["reaction_correct"], errors="coerce").fillna(0).astype(int)
        rc = frame.pivot(index="source_row", columns="method", values="reaction_correct").sort_index()
        self.models = tuple(models)
        self.rows = rc.index.to_numpy()
        self.rc = rc[[ORIGINAL, *self.models]].astype(int)
        self.nf = frame.groupby("source_row")["evaluation_included"].first().loc[self.rows].astype(int)
        self.passed = {
            h: flags3.pivot(index="source_row", columns="model", values=f"{h}_passed")
            .loc[self.rows, list(self.models)]
            .astype(int)
            for h in HEURISTICS
        }
        self.strata = strata.loc[self.rows]

    def correct_nf(self) -> pd.DataFrame:
        """Extraction correctness on noise-free records (Table 2)."""
        return self.rc[self.nf == 1]

    def denoise(self, h: str) -> pd.DataFrame:
        """1 when heuristic h makes the right keep/remove decision."""
        p = self.passed[h]
        nf = self.nf.to_numpy()[:, None]
        return pd.DataFrame(np.where(nf == 1, p, 1 - p), index=p.index, columns=p.columns)

    def e2e(self, h: str = TEMPLATE) -> pd.DataFrame:
        """1 when a noise-free record is kept with an exact reaction, or a noisy one removed."""
        p = self.passed[h]
        rc = self.rc[list(self.models)]
        nf = self.nf.to_numpy()[:, None]
        return pd.DataFrame(np.where(nf == 1, rc * p, 1 - p), index=p.index, columns=p.columns)

    def e2e_tp(self, h: str = TEMPLATE) -> pd.DataFrame:
        p = self.passed[h]
        rc = self.rc[list(self.models)]
        nf = self.nf.to_numpy()[:, None]
        return pd.DataFrame(np.where(nf == 1, rc * p, 0), index=p.index, columns=p.columns)

    def denoise_tp(self, h: str = TEMPLATE) -> pd.DataFrame:
        p = self.passed[h]
        nf = self.nf.to_numpy()[:, None]
        return pd.DataFrame(np.where(nf == 1, p, 0), index=p.index, columns=p.columns)

    def denoise_tn(self, h: str = TEMPLATE) -> pd.DataFrame:
        p = self.passed[h]
        nf = self.nf.to_numpy()[:, None]
        return pd.DataFrame(np.where(nf == 0, 1 - p, 0), index=p.index, columns=p.columns)


def pick_best(scores: pd.Series) -> str:
    """Highest score; ties resolved toward the selected configuration."""
    top = scores[scores == scores.max()].index
    return PRIMARY if PRIMARY in top else top[0]


def table2_tests(f: RecordFlags, reference: str | None = None) -> pd.DataFrame:
    """The best configuration (or ``reference``) against the other 18 and against the original USPTO record."""
    c = f.correct_nf()
    n = len(c)
    configs = list(f.models)
    totals = c[configs].sum()
    best = reference or pick_best(totals)
    rows = []
    for other in [m for m in configs if m != best] + [ORIGINAL]:
        a, b, p = mcnemar_exact(c[best], c[other])
        lo, hi = wilson(int(c[other].sum()), n)
        rows.append(
            {
                "family": "USPTO" if other == ORIGINAL else "T2",
                "best": best,
                "comparator": other,
                "comparator_correct": int(c[other].sum()),
                "n": n,
                "comparator_accuracy": c[other].sum() / n,
                "comparator_ci_low": lo,
                "comparator_ci_high": hi,
                "best_only": a,
                "comparator_only": b,
                "p_exact": p,
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for family in ("T2", "USPTO"):
        mask = out.family == family
        out.loc[mask, "p_holm"] = holm(out.loc[mask, "p_exact"])
    lo, hi = wilson(int(totals[best]), n)
    out["best_correct"] = int(totals[best])
    out["best_accuracy"] = totals[best] / n
    out["best_ci_low"] = lo
    out["best_ci_high"] = hi
    return out


def table3_tests(f: RecordFlags) -> pd.DataFrame:
    rows = []
    for outcome, getter in (("denoising", f.denoise), ("end_to_end", f.e2e)):
        template = getter(TEMPLATE)[PRIMARY]
        n = len(template)
        for h in HEURISTICS:
            if h == TEMPLATE:
                continue
            other = getter(h)[PRIMARY]
            a, b, p = mcnemar_exact(template, other)
            rows.append(
                {
                    "outcome": outcome,
                    "model": PRIMARY,
                    "heuristic": h,
                    "n": n,
                    "template_correct": int(template.sum()),
                    "template_ci_low": wilson(int(template.sum()), n)[0],
                    "template_ci_high": wilson(int(template.sum()), n)[1],
                    "heuristic_correct": int(other.sum()),
                    "heuristic_ci_low": wilson(int(other.sum()), n)[0],
                    "heuristic_ci_high": wilson(int(other.sum()), n)[1],
                    "template_only": a,
                    "heuristic_only": b,
                    "p_exact": p,
                }
            )
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for outcome in out.outcome.unique():
        mask = out.outcome == outcome
        out.loc[mask, "p_holm"] = holm(out.loc[mask, "p_exact"])
    return out


def e2e_configs(f: RecordFlags) -> pd.DataFrame:
    e = f.e2e(TEMPLATE)
    tp = f.e2e_tp(TEMPLATE)
    n = len(e)
    rows = []
    for m in f.models:
        a, b, p = (0, 0, 1.0) if m == PRIMARY else mcnemar_exact(e[PRIMARY], e[m])
        lo, hi = wilson(int(e[m].sum()), n)
        rows.append(
            {
                "config": m,
                "e2e_correct": int(e[m].sum()),
                "n": n,
                "e2e_accuracy": e[m].sum() / n,
                "ci_low": lo,
                "ci_high": hi,
                "e2e_tp": int(tp[m].sum()),
                "template_denoising_correct": int(f.denoise(TEMPLATE)[m].sum()),
                "primary_only": a,
                "config_only": b,
                "p_exact_vs_primary": p,
            }
        )
    out = pd.DataFrame(rows)
    mask = out.config != PRIMARY
    out["p_holm_vs_primary"] = np.nan
    out.loc[mask, "p_holm_vs_primary"] = holm(out.loc[mask, "p_exact_vs_primary"])
    return out


class Procedure:
    """A selection rule and the outcome matrix it is scored on."""

    def __init__(self, f: RecordFlags, kind: str):
        configs = list(f.models)
        self.configs = configs
        self.kind = kind
        self.nf = f.nf.to_numpy()
        self.strata = f.strata.to_numpy()
        self.rc = f.rc[configs].to_numpy().astype(float)
        self.den = np.stack([f.denoise(h)[configs].to_numpy() for h in HEURISTICS], axis=2).astype(float)
        self.e2e = np.stack([f.e2e(h)[configs].to_numpy() for h in HEURISTICS], axis=2).astype(float)
        self.eligible = np.where(self.nf == 1)[0] if kind == "extraction" else np.arange(len(self.nf))

    def choose(self, train: np.ndarray):
        """Tied (configuration, heuristic) choices made on ``train``."""
        if self.kind == "extraction":
            s = self.rc[train].mean(0)
            return [(j, None) for j in np.flatnonzero(s >= s.max() - 1e-12)]
        if self.kind == "two_stage":
            nf_train = train[self.nf[train] == 1]
            s = self.rc[nf_train].mean(0)
            picks = []
            for j in np.flatnonzero(s >= s.max() - 1e-12):
                d = self.den[train, j, :].mean(0)
                picks += [(j, h) for h in np.flatnonzero(d >= d.max() - 1e-12)]
            return picks
        s = self.e2e[train].mean(0)
        return [tuple(ix) for ix in np.argwhere(s >= s.max() - 1e-12)]

    def score(self, picks, rows: np.ndarray) -> float:
        values = []
        for j, h in picks:
            if self.kind == "extraction":
                values.append(self.rc[rows, j].sum())
            else:
                values.append(self.e2e[rows, j, h].sum())
        return float(np.mean(values))

    def label(self, picks) -> str:
        j, h = picks[0]
        return self.configs[j] + ("" if h is None else f" | {HEURISTICS[h]}")


def stratified_folds(rows: np.ndarray, strata: np.ndarray, k: int, rng) -> list:
    folds = [[] for _ in range(k)]
    for s in STRATA:
        members = rng.permutation(rows[strata[rows] == s])
        for i, x in enumerate(members):
            folds[i % k].append(x)
    return [np.array(sorted(fold)) for fold in folds]


def optimism(f: RecordFlags, kind: str) -> tuple[dict, pd.Series]:
    p = Procedure(f, kind)
    rows = p.eligible
    n = len(rows)
    apparent_picks = p.choose(rows)
    apparent = p.score(apparent_picks, rows) / n
    picks_log = []

    cv = []
    for repeat in range(REPEATS):
        rng = np.random.default_rng(SEED + repeat)
        total = 0.0
        for fold in stratified_folds(rows, p.strata, 5, rng):
            train = np.setdiff1d(rows, fold)
            choice = p.choose(train)
            picks_log.append(p.label(choice))
            total += p.score(choice, fold)
        cv.append(total / n)
    cv = np.array(cv)

    loso_total = 0.0
    loso_picks = {}
    for s in STRATA:
        fold = rows[p.strata[rows] == s]
        choice = p.choose(np.setdiff1d(rows, fold))
        loso_picks[s] = p.label(choice)
        loso_total += p.score(choice, fold)
    loso = loso_total / n

    rng = np.random.default_rng(SEED)
    oob, harrell = [], []
    for _ in range(BOOT):
        sample = np.concatenate(
            [rng.choice(rows[p.strata[rows] == s], (p.strata[rows] == s).sum()) for s in STRATA]
        )
        out = np.setdiff1d(rows, sample)
        choice = p.choose(sample)
        oob.append(p.score(choice, out) / len(out))
        harrell.append(p.score(choice, sample) / n - p.score(choice, rows) / n)
    oob = np.array(oob)
    harrell = np.array(harrell)
    result = {
        "procedure": kind,
        "records": n,
        "apparent_selected": p.label(apparent_picks),
        "apparent_accuracy": apparent,
        "cv5_mean": cv.mean(),
        "cv5_p2_5": np.percentile(cv, 2.5),
        "cv5_p97_5": np.percentile(cv, 97.5),
        "loso": loso,
        "loso_picks": "; ".join(f"{k}: {v}" for k, v in loso_picks.items()),
        "oob_mean": oob.mean(),
        "boot_632": 0.368 * apparent + 0.632 * oob.mean(),
        "harrell_optimism": harrell.mean(),
        "harrell_corrected": apparent - harrell.mean(),
    }
    return result, pd.Series(picks_log).value_counts()


def stratified_total(y, strata, frame: dict) -> dict:
    """Stratified estimator of a population total with FPC and a Wald 95% CI."""
    y = np.asarray(y, dtype=float)
    strata = np.asarray(strata)
    total = 0.0
    variance = 0.0
    counts = {}
    for stratum, population in frame.items():
        values = y[strata == stratum]
        n = len(values)
        counts[stratum] = int(values.sum())
        total += population * values.mean()
        variance += population**2 * (1 - n / population) * values.var(ddof=1) / n
    se = math.sqrt(variance)
    return {
        "counts": counts,
        "total": total,
        "se": se,
        "ci_low": total - Z95 * se,
        "ci_high": total + Z95 * se,
        "frame_n": sum(frame.values()),
    }


def cv_selected_total(f: RecordFlags, frame: dict, repeats: int = REPEATS) -> dict:
    """Weighted end-to-end TP when the configuration is chosen out of fold."""
    rows = np.arange(len(f.rows))
    strata = f.strata.to_numpy()
    nf = f.nf.to_numpy()
    rc = f.rc[list(f.models)].to_numpy()
    y = f.e2e_tp(TEMPLATE)[list(f.models)].to_numpy()
    n_h = {s: (strata == s).sum() for s in STRATA}
    w = np.array([frame[s] / n_h[s] for s in strata])
    totals = []
    for repeat in range(repeats):
        rng = np.random.default_rng(SEED + repeat)
        folds = [[] for _ in range(5)]
        for s in STRATA:
            for i, x in enumerate(rng.permutation(rows[strata == s])):
                folds[i % 5].append(x)
        total = 0.0
        for fold in folds:
            fold = np.array(fold)
            train = np.setdiff1d(rows, fold)
            accuracy = rc[train[nf[train] == 1]].mean(0)
            tied = np.flatnonzero(accuracy >= accuracy.max() - 1e-12)
            total += np.mean([(w[fold] * y[fold, j]).sum() for j in tied])
        totals.append(total)
    totals = np.array(totals)
    return {"mean": totals.mean(), "p2_5": np.percentile(totals, 2.5), "p97_5": np.percentile(totals, 97.5)}
