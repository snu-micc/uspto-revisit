# Evaluation protocol

This directory contains the code used to calculate extraction, denoising,
step-specific, end-to-end, threshold-sensitivity and weighted-extrapolation
results. The results of the manuscript and SI use the revised benchmark in
`data/revised/` and are produced by one command (Section 8):

```bash
python evaluation/reproduce_revised.py
```

Sections 1 to 7 define the metrics. Their stand-alone scripts score the
first-release benchmark (`ground_truth_review.csv`) by default.

## Ground truth (first release)

`ground_truth_review.csv` contains the first-release annotations. Every
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

Expected totals with the single-label sampling-frame populations
(`--input evaluation/data/weighted_sampling_summary_single_label_frame.csv`,
see Section 6) are approximately TP 298,706, FN 32,990, FP 56,495 and
TN 66,284.

## 6. Population counts and sampling frame

Run:

```bash
python evaluation/build_sampling_frame.py --strict
```

The script reads `data/grants_issue.zip` and writes:

- `data/sampling_frame_counts.csv`: the Table 1 counts from the reaction
  SMILES and Reagent field (82,203 / 100,777 / 236,253 / 201,761; union
  550,070), the 430,428-reaction single-criterion frame of the revised
  benchmark (exactly one Table 1 criterion and no `rare_temp` or `duplicated`
  label; 44,157 / 20,157 / 198,305 / 167,809), the released flags (82,203 /
  52,758 / 236,368 / 200,594; union 536,372), the 501,997 reactions flagged by
  exactly one of the four sampled flags, and the 454,475-reaction single-label
  frame of the first release (59,282 / 20,163 / 207,221 / 167,809), which has
  no other `Issue` label. Each row is compared with the manuscript value;
  `--strict` fails on any difference.
- `data/sampling_frame_label_combinations.csv`: the count of every `Issue`
  label combination.
- `data/benchmark_sample_membership.csv` and
  `../examples/input_with_provenance.csv`: stratum and population row of each
  of the 400 benchmark records.
- `data/weighted_sampling_summary_single_label_frame.csv`: the weighting input
  with single-label populations.

All 400 first-release records lie in the single-label frame, 100 per stratum;
371 of them lie in the single-criterion frame of their stratum (the 28 records
redrawn for the revised benchmark and record 15, which repeats record 8). With
`--input data/revised/input.csv` the script reports 400 of 400. The random
seed used for the original draw was not recorded, so that draw cannot be
re-run; the two replacement draws of the revised benchmark can
(`draw_replacements.py`, Section 8). In this file `source_row` is the 0-based data row of
`grants_issue.csv`. Elsewhere in `results/`, `source_row` is the benchmark
index, `idx`. When a benchmark reaction occurs in several population rows
(247 of 400 records), `source_row` is the first such row and
`candidate_source_rows` lists all of them.

## 7. Sensitivity options

`evaluate_table2.py`, `evaluate_table3.py` and `evaluate_step_by_step.py`
share the following switches. All are off by default; with none of them the
published results are regenerated byte for byte. When a switch is used and
`--output-dir` is not, results are written to
`results/sensitivity/<variant>/{table2,table3,step_by_step}`.
`evaluate_step_by_step.py` reads Table 2 and Table 3 flags from the same
variant folder.

- `--no-overrides [--override-scope all|blocked] [--overrides PATH]`:
  rebuilds each configuration's output as the pipeline would have written it
  without `config/nosmi_overrides.json`.
  - Every resolved component whose normalized name is an override key
    becomes `[name (NoSmi)]` in the configuration's SMILES dictionary.
  - Each reaction step that then contains a NoSmi token gets the record
    `atom_mapping.py` writes for such steps: no mapping, template or
    confidence, and the error "Reaction contains an unresolved (NoSmi)
    compound." Other steps keep their published mapping.
  - The affected benchmark rows are rebuilt with
    `build_ground_truth_review.model_row`. Table 3 and the step-by-step
    script read the rebuilt outputs.
  - The record then fails exact extraction and name-to-SMILES conversion,
    the template-anomaly heuristic filters it, and the ORDerly role-count
    heuristics see the unmapped step.
  - The released outputs do not record the resolver behind each component, so
    matching is by name. `all` is an upper bound on the overrides' effect
    (a lower bound on accuracy without overrides).
  - `blocked` counts only names that the automatic resolver rejects as
    ambiguous or non-molecular, and that therefore cannot resolve without an
    override. It is a lower bound on the overrides' effect (an upper bound on
    accuracy without overrides) and needs the `uspto_revisit` package.
- `--no-row-decisions [--row-decisions PATH]`: the same, for the row-level
  `use_smiles` decisions in `result/nosmi_human_review.csv`. A component
  counts only if its (row, normalized name) has a decision and its SMILES
  equals the canonical `reviewed_smiles`. A component that carries another
  structure was not produced by the decision. One example is the non-stereo
  omeprazole in row 80 of 11 configurations; such components are left alone.
- `--name-resolution fully-automatic [--resolution-order two-pass|retry]
  [--requery-log PATH]`: replaces every curated component with what the
  automatic resolvers return.
  - Curated components are the override names, the row decisions, and
    reviewed SMILES reused by name in other rows.
  - The resolver answers are replayed offline from
    `result/name_resolution/requery_2026-10-16_raw.jsonl` with the released
    resolver code and, as when the recording was made, the lookup tables the
    resolver had then (`historical_lookup_tables.py`).
  - The default `two-pass` order is the pipeline's: lookup table and OPSIN
    first (also on `fix_name(name)`), then the NoSmi retry (refusal, synonym,
    PubChem/OPSIN/CIR/ChEBI, first non-empty). `retry` uses the NoSmi retry
    path alone.
  - A component whose structure changes gets the new SMILES; one that no
    resolver returns becomes NoSmi.
  - Steps with a NoSmi token are unmapped as above. Other changed steps are
    re-mapped with LocalMapper 202403 on the CPU, one reaction at a time
    (`mapping` extra).
  - The audit lists, per component, the automatic structure, its source and
    stage, and the outcome (replaced, unchanged or unresolved).
- `--strict-matching`: keep stereochemistry, formal charge, protonation and
  isotopes (only atom maps are removed), and accept only the first reference
  reaction (`ground_truth_*`, not `ground_truth_2_*`). The parts are available
  separately as `--strict-structures` and `--first-reference-only`. The
  template-anomaly heuristic's identical-structure test keeps the relaxed
  comparison, so only the comparison with the reference changes.
  `build_ground_truth_review.py` still de-duplicates a model's products
  without regard to stereochemistry.

Each variant folder contains `sensitivity_options.json`. The curated-resolution
switches also write `curation_ablation_audit.csv`, which lists every
switched-off component with its configuration, row, label, name, SMILES,
whether it is in the reaction skeleton, the reason and the override kind.

For gemini-3.6-flash-high the switches give the following results. The
published values are 245/283, 299/400 and 272/400.

| Variant | Exact extraction | Denoising | Strict end-to-end |
| --- | --- | --- | --- |
| `--no-overrides --override-scope blocked` (smallest effect) | 238/283 | 291/400 | 265/400 |
| `--no-overrides` (largest effect) | 220/283 | 278/400 | 253/400 |
| `--no-row-decisions` | 238/283 | 296/400 | 270/400 |
| `--no-overrides --no-row-decisions` | 215/283 | 278/400 | 253/400 |
| `--no-overrides --override-scope blocked --no-row-decisions` | 232/283 | 290/400 | 264/400 |
| `--strict-structures` | 243/283 | 299/400 | 272/400 |
| `--first-reference-only` | 242/283 | 299/400 | 270/400 |
| `--strict-matching` | 241/283 | 299/400 | 270/400 |
| `--no-overrides --strict-matching` | 216/283 | 278/400 | 251/400 |
| `--name-resolution fully-automatic` (two-pass) | 222/283 | 285/400 | 257/400 |
| `--name-resolution fully-automatic --strict-matching` | 218/283 | 285/400 | 255/400 |
| `--name-resolution fully-automatic --resolution-order retry` | 223/283 | 285/400 | 258/400 |
| `--name-resolution fully-automatic --resolution-order retry --strict-matching` | 219/283 | 285/400 | 256/400 |

## 8. Revised benchmark (manuscript and SI)

`reproduce_revised.py` reads `data/revised/` and the stored outputs in
`result/revised/model_outputs/<scenario>/`, and writes `results/revised/`. It
makes no model, resolver or network call.

- **Scenarios.** `fully_automatic` is primary: names are resolved only by
  OPSIN, PubChem, NCI CIR and ChEBI, as the released resolver does, without
  the curated layers and without the two lookup tables the resolver had when
  the outputs were produced (`historical_lookup_tables.py`, `lookup_tables.py`).
  `fully_automatic_with_tables` keeps those tables (sensitivity analysis), and
  `curated` adds the overrides and record-level decisions (secondary).
- **Labels.** All 400 paragraph labels (292 noise-free, 108 noisy) enter the
  denoising metrics. Extraction uses the 291 noise-free records with complete
  molecular references; record 215 is left out of the structure endpoints
  only.
- **Step scope.** Each prediction is compared on its step that forms the
  title compound (`build_ground_truth_review.configure_step_scope`).
- **End-to-end:** a noise-free record counts when its extraction is exact and
  the heuristic keeps it; a noisy record counts when the heuristic removes it.
  `end_to_end_bounds.csv` gives the 399 assessable records and lower and upper
  bounds over all 400, which treat the unavailable reference as wrong or right.
- **Population totals:** stratified totals with finite-population correction,
  `T = sum_h N_h ybar_h` and `Var = sum_h N_h^2 (1 - n_h/N_h) s_h^2 / n_h`,
  with a 95% Wald interval, on the 430,428-record single-criterion frame
  (`benchmark_statistics.FRAME_SINGLE_CRITERION`).
- **Statistics:** exact two-sided McNemar tests with a Holm correction within
  each family. Extraction tests compare the selected configuration
  (gemini-3.6-flash, high) with the other 18 and with the original USPTO
  record (`extraction_paired_tests_primary_*.csv`), and, separately, the
  configuration of highest accuracy (`extraction_paired_tests_*.csv`; without
  the lookup tables this is gpt-5.4 with no reasoning, one record higher).
  Selection-adjusted accuracy by nested 5-fold cross-validation (200 repeats),
  leave-one-stratum-out and the out-of-bag stratified bootstrap (2,000
  resamples), all with seed 20261016. Three selection procedures are reported:
  extraction only, extraction then filter (`two_stage`), and configuration and
  filter chosen jointly by end-to-end accuracy (`joint`). The implementation is
  in `benchmark_statistics.py`.
- **Template-frequency cutoffs:** the frequencies come from
  `data/revised/benchmark_sample_membership.csv`.
- **Reference provenance:** `reference_source` in `ground_truth.csv` separates
  references retained from the consensus candidate (169) from references saved
  as reviewed decisions (122).

`rebuild_revised_outputs.py` rebuilds the stored outputs of the three
scenarios from the first-release outputs, the runs on the replacement records
and the two recorded re-queries, in the order the released outputs were built
(the docstring lists the five steps), and checks that they and the audits in
`result/revised/audits/` equal the released copies. It needs the `mapping`
extra.

`draw_replacements.py` re-runs the two replacement draws from the population
file (`data/grants_issue.zip`): records 15 and 315, which repeated records 8
and 309 (seed 20261021, frame of the released flags), and the 28 records that
meet two Table 1 criteria (seed 20261022, single-criterion frame). The rules on
Lowe's paragraphs are checked against `data/revised/replacement_draws.csv`,
because Lowe's CML files are not part of the repository. It writes every
examined candidate to `results/revised/replacement_draw_candidates.csv` and
checks that the accepted rows are the population rows of the replacement
records.

## Contextual benchmarks

The prior-study outputs in `benchmarks/` are retained for contextual analysis
only. Because their tasks and schemas differ, they are not included as direct
comparators in the main Table 2 evaluation.

The Ai et al. row of SI Table S17 is reproduced by:

```bash
python evaluation/prepare_ai_baseline.py
python evaluation/evaluate_table2.py \
  --ai-components benchmarks/ai_et_al/preprocessing/ai_baseline_components.csv \
  --ai-scoring exact --output-dir evaluation/results/table2_with_ai
```

`prepare_ai_baseline.py` first confirms that the released component file is
regenerated exactly from `benchmarks/ai_et_al/output.json` and the released
name-resolution table. `--ai-scoring exact` applies the exact-match rule used
for every other method: 3.9% (11/283). The default `coverage` rule for this
optional row gives 5.7% (16/283).
