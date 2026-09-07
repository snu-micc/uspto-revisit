# Evaluation protocol

This directory contains the finalized 400-paragraph ground truth and the code
used to calculate extraction, denoising, step-specific, end-to-end,
threshold-sensitivity, and weighted-extrapolation results.

## Ground truth

`ground_truth_review.csv` contains the final human-reviewed annotations. Every
record was inspected before the ground truth was finalized. The
`ground_truth_valid` column is Boolean:

- `True`: a valid ground-truth reaction was established (283 noise-free
  paragraphs);
- `False`: no valid ground-truth reaction could be established (117 noisy
  paragraphs).

All `True` rows contain the explicit final reactant, reagent, product, and
reaction fields. The public evaluation does not use separate approval and
correction status codes.

Running `build_ground_truth_review.py` creates
`benchmark_all_configurations.csv`, a local working file that combines these
fixed annotations with the processed outputs of all 19 configurations. It is
fully reproducible and therefore is not versioned. Rebuilding it preserves the
finalized human decisions and never replaces them with a model consensus.

## 1. Extraction evaluation (Table 2)

Run:

```bash
python evaluation/build_ground_truth_review.py
python evaluation/evaluate_table2.py
python evaluation/select_best_configurations.py
```

For each reactant, reagent, and product role:

- TP: predicted component present in the selected ground truth;
- FP: predicted component absent from the selected ground truth;
- FN: ground-truth component absent from the prediction;
- precision: TP / (TP + FP);
- recall: TP / (TP + FN).

Reactants are compared as multisets; reagents and products are compared as
sets. Atom maps, stereochemistry, formal charge, and protonation state are
ignored during structure comparison. A reaction-level success requires exact
agreement for all three roles. The 95% confidence interval is the Wilson score
interval.

Table 2 uses only the 283 noise-free paragraphs. All 19 configuration-specific
results are written to `results/table2/table2.csv`.
`results/table2_best_models.csv` retains the highest-accuracy configuration for
each model family; exact reaction-accuracy ties are resolved in favor of the
higher reasoning effort (`high`, then `low`, then `off`/`none`). Thus, gpt-5.4
uses `high` when `low` and `high` both achieve 237/283 (83.7456%). All component
metrics in the selected row come from that same configuration. This selection
rule does not change the reasoning setting used when generating model outputs.
The manuscript-format table reports percentages to four decimal places;
`table2_metrics.csv` retains the underlying full-precision ratios.

## 2. Denoising evaluation (Table 3)

Run:

```bash
python evaluation/evaluate_table3.py
```

All 400 paragraphs are included. For a pass/fail heuristic:

- noise-free and passed: TP;
- noise-free and filtered: FN;
- noisy and passed: FP;
- noisy and filtered: TN.

Reported metrics are:

- noise-free reactions saved: TP / (TP + FN);
- noisy reactions filtered: TN / (TN + FP);
- denoising accuracy: (TP + TN) / 400.

The manuscript-format tables report these rates as percentages to four decimal
places; `table3_metrics.csv` retains the underlying full-precision ratios.
`table3_flags.csv` retains row-level decisions for all configurations, while
`table3.csv` contains the selected manuscript table. Redundant per-model table
copies are not written because they can be reconstructed from
`table3_metrics.csv`.

The evaluated heuristics are many-products, many-reactants, many-reagents,
no-reagent, and template anomaly. Under the primary criterion, a template
anomaly is flagged when a reliable reaction center cannot be assigned because
of unresolved structures, mapping/template failure, or no detectable
reactant-product structural change. A low template frequency, including N=1,
is not part of this primary Table 3 criterion. Frequency cutoffs are assessed
only in the exploratory sensitivity analysis described in Section 4.

## 3. Step-specific and end-to-end evaluation

Run:

```bash
python evaluation/evaluate_step_by_step.py
```

The step-specific summary reports:

1. valid structured LLM output;
2. successful chemical-name-to-SMILES conversion;
3. successful reaction-step processing;
4. successful atom mapping;
5. successful template processing;
6. exact extraction among noise-free paragraphs;
7. the correct denoising decision;
8. strict end-to-end success.

A noise-free paragraph is an end-to-end success only if its extracted reaction
exactly matches the ground truth, all required downstream processing succeeds,
and the paragraph passes the evaluated heuristic. A noisy paragraph is a
success when it is filtered. The selected gemini-3.6-flash configuration
achieves 272/400 (68.0%). Row-level outcomes are retained in
`results/step_by_step/`.

The step-by-step script also writes
`table_s14_noise_free_steps_gemini-3.6-flash-high.csv`. This manuscript-facing
table uses the fixed set of 283 noise-free paragraphs as the denominator for
all overall step rates. Conditional rates for steps 1–6 use only paragraphs
that passed the preceding step; exact-extraction accuracy is evaluated
independently against the ground truth and therefore has no conditional rate.

## 4. Template-frequency sensitivity

The repository versions the downstream 400-row input
`data/template_threshold_audit.csv`, so the reported exploratory N=1, N<5, and
N<10 comparison can be reproduced directly:

```bash
python evaluation/evaluate_threshold_sensitivity.py
```

The script changes only the frequency cutoff while leaving template/mapping
failure and no-transformation conditions unchanged. N=1 is therefore a
sensitivity setting, not the primary template-anomaly definition.

### Rebuild the 400-row frequency audit

The versioned population artifact
[`grants_issue.zip`](https://github.com/snu-micc/uspto-revisit/releases/download/data-v1.0.0/grants_issue.zip)
is available from the `data-v1.0.0` release. It contains 623,290 data rows from
the ORDerly-flagged subset. Place it at `evaluation/data/grants_issue.zip`,
verify its SHA-256, and run:

```bash
python evaluation/build_template_threshold_audit.py
python evaluation/evaluate_threshold_sensitivity.py
```

Expected SHA-256:

```text
C5F6A5D1FEF673F247FCE5A481438E33594C32FFDB09807F7A2478D8E1D0EEBF
```

`build_template_threshold_audit.py` links the stored `TemplateFreq` values in
the released population file to the 400 benchmark reactions, combines them
with the selected model's mapping/template-failure and no-transformation flags,
validates the linkage, and rewrites `data/template_threshold_audit.csv`.
`evaluate_threshold_sensitivity.py` then produces
`results/template_threshold_sensitivity.csv`.

The released population schema is:

```text
Year, PatentNumber, ParagraphNum, ReactionSmiles, Template, Reagent,
NAtomChanges, TemplateFreq, NMolecules, Issue
```

This workflow has a deliberate provenance boundary. `grants_issue.zip` is the
623,290-row flagged subset, not the raw/full approximately 1.9-million-reaction
USPTO corpus. `Template` and `TemplateFreq` are precomputed upstream fields;
the repository does not include the historical extraction workflow required to
regenerate those two fields from the full corpus. In particular,
`TemplateFreq` must not be interpreted as a frequency newly calculated within
the 623,290 released rows. The code reproduces the benchmark linkage and Table
S22-style exploratory aggregation from the released values, not full-corpus
template extraction. Detailed field definitions and provenance are in
[`data/README.md`](data/README.md).

## 5. Weighted extrapolation

Run:

```bash
python evaluation/evaluate_weighted_extrapolation.py
```

For each mutually exclusive ORDerly sampling category, the script multiplies
the observed confusion-matrix counts by `population_n / sample_n` and sums the
weighted counts. The supported population contains 501,997 reactions. The
34,375 reactions flagged by overlapping categories are outside the sampled
domain and are not assigned point estimates.

Expected weighted totals are approximately:

- TP: 328,986;
- FN: 36,666;
- FP: 63,505;
- TN: 72,840.

## Contextual benchmarks

The prior-study outputs in `benchmarks/` are retained for contextual analysis
only. Because their tasks and schemas differ, they are not included as direct
comparators in the main Table 2 evaluation.
