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
each model family; ties are resolved in favor of the lower reasoning effort.
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
no-reagent, and template anomaly. A template anomaly is flagged when a
singleton template is present or when a reliable transformation template
cannot be obtained because of unresolved structures, mapping/template failure,
or no detectable reactant-product structural change.

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

Run:

```bash
python evaluation/evaluate_threshold_sensitivity.py
```

The script compares N=1, N<5, and N<10 while leaving template/mapping failure
and no-transformation conditions unchanged. It reads the released 400-row audit
file in `data/template_threshold_audit.csv`.

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
