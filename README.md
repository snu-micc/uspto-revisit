# LLM-based curation of USPTO reaction data

Code, benchmark, model outputs and evaluation scripts for a two-stage curation
of USPTO reaction data:

1. **Reaction extraction.** An LLM reads a patent title and experimental
   paragraph and returns the reactants, reagents and products by name. The
   names are resolved to SMILES and the reaction is atom-mapped with
   LocalMapper.
2. **Denoising.** The template-anomaly filter removes reactions for which no
   reaction center can be assigned. It is compared with the four ORDerly
   heuristics (many-products, many-reactants, many-reagents and no-reagent).

The benchmark has 400 patent paragraphs, 100 from each ORDerly-flagged
stratum. The outputs of all 19 evaluated model configurations are included, so
every reported number can be reproduced offline and without API keys.

## Repository structure

```text
evaluation/reproduce_revised.py        All results of the manuscript and SI (one command)
evaluation/rebuild_revised_outputs.py  Rebuilds the revised model outputs (optional)
evaluation/data/revised/               Benchmark labels, references, input and sample membership
result/revised/model_outputs/          Outputs of the 19 configurations (fully automatic and curated)
result/revised/replacement_runs/       The 19 configurations on the two redrawn records
result/model_outputs/                  Outputs on the first-release input
result/name_resolution/                Shared name cache and the recorded resolver re-query
result/configuration_settings.csv      Run settings and query dates of the 19 configurations
config/nosmi_overrides.json            Curated name-resolution overrides
result/nosmi_human_review.csv          Record-level name-resolution decisions
src/uspto_revisit/                     Extraction, name resolution and atom mapping
prompts/prompt.txt                     Extraction prompt
scripts/                               Extraction runners and name-resolution pipeline
examples/finetuning/                   Fine-tuning examples (disjoint from the benchmark)
benchmarks/                            Outputs of Ai et al. and Zhang et al.
evaluation/                            Scoring, statistics and first-release evaluation
tests/                                 Unit tests
```

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/snu-micc/uspto-revisit.git
cd uspto-revisit
python -m pip install -e ".[dev]"
```

The `mapping` extra (`python -m pip install -e ".[dev,mapping]"`) installs
LocalMapper and its PyTorch/DGL stack. It is needed only to rebuild model
outputs or to process new extractions. LocalMapper is licensed separately
under CC BY-NC-SA 4.0. The project loads DGL for local, single-process
inference only; see [SECURITY.md](SECURITY.md).

## Reproduce the results

```bash
python evaluation/reproduce_revised.py
```

The script scores the stored model outputs against the released labels and
references. It writes every table, statistical test and figure to
`evaluation/results/revised/` in about two minutes on a CPU, and it makes no
model, name-resolver or network call. Percentages are rounded half up to one
decimal, and intervals are 95% Wilson score intervals unless stated.

Expected values for the selected configuration, gemini-3.6-flash with high
reasoning and fully automatic name resolution:

| Quantity | Value |
| --- | --- |
| Reaction-level exact-match extraction | 216/289 (74.7%; 95% CI 69.4–79.4%) |
| Strict matching | 210/289 |
| Original USPTO record, same references | 105/289 (36.3%) |
| Extraction with curated name resolution | 235/289 (81.3%) |
| Template-anomaly denoising, all 400 labels | 285/400 (71.3%) |
| Noise-free retained / noisy removed | 230/292 / 55/108 |
| End-to-end, all 400 records | 246/400 (61.5%) |
| Correct retained reactions in the 454,475-record sampling frame | about 250,000 (95% CI 224,000–276,000) |

Files are in `evaluation/results/revised/fully_automatic/` unless another
folder is given:

| Manuscript or SI | File |
| --- | --- |
| Table 2 | `table2.csv` (`table2_raw.csv` unrounded) |
| Table 3 | `table3_metrics.csv` |
| Figure 1 | `../Figure1.png` (Stage 2 panel; the Stage 1 accuracies are in `table2.csv`) |
| Figure S2 | `../filtering_by_reason.csv`, `../FigureS2.png` |
| Table S17 (Ai et al.) | `../ai_metrics.csv` |
| Table S18 | `table_s18.csv` |
| Table S19 | `step_table_structure_eligible.csv` |
| Table S20 | `end_to_end_bounds.csv`, `table2_metrics_lenient.csv`, `table3_metrics.csv` |
| Table S21 | `weighted_extrapolation.csv`, `population_bounds.csv` |
| Table S22 | `threshold_sensitivity.csv` |
| Tables S23 and S24 | the same files in `../curated/` |
| Table S25 | `table2_metrics_strict.csv` |
| Table S26 | `extraction_paired_tests_lenient.csv` |
| Table S27 | `denoising_paired_tests.csv`, `end_to_end_paired_tests.csv` |
| Table S28 | `selection_adjustment.csv` |
| Table S30 | `../provenance_accuracy.csv` |
| Table S32 | `../noisy_reason_counts.csv` |
| Section 2.5 | `../summary.json`, `../label_changes.csv` |

The row-level flags (`table2_flags_*.csv`, `table3_flags.csv` and
`step_flags_structure_endpoint.csv`) give the outcome behind every count.
`input_hashes.json` records the SHA-256 of every input file.

### Statistical analyses

- Exact McNemar tests with a Holm correction within each family: the selected
  configuration against the other 18 and against the original USPTO record,
  and the template anomaly against each ORDerly heuristic.
- Selection-adjusted accuracy: the 400 records both select and score the
  configuration, so nested 5-fold cross-validation stratified by sampling
  stratum (200 repeats), leave-one-stratum-out and an out-of-bag stratified
  bootstrap (2,000 resamples) estimate the accuracy of the selection
  procedure.
- Population totals: stratified estimates with a finite-population
  correction on the 454,475-record single-label sampling frame.

The random seeds are fixed (`evaluation/benchmark_statistics.py`), so the
results are reproduced to the last digit.

### Rebuild the model outputs (optional)

```bash
python evaluation/rebuild_revised_outputs.py
```

This builds `result/revised/model_outputs/` from the first-release outputs
and the replacement runs. It writes to `result/revised/rebuilt/` and reports
whether every rebuilt file equals the released copy (line endings aside). It needs the `mapping`
extra, because steps whose components change are re-mapped with LocalMapper.

## Benchmark

`evaluation/data/revised/` contains:

- `ground_truth.csv`: one row per paragraph with `label` (noise-free or
  noisy), `noisy_reason`, `salt_formation`, `structure_evaluation_eligible`,
  `structure_exclusion_reason`, `reference_source` and up to two reference
  reactions (`ground_truth_*` and `ground_truth_2_*`, as reactant, reagent,
  product and reaction SMILES);
- `input.csv`: the 400 titles and paragraphs with the original USPTO reaction
  SMILES (`Lowe_smiles`);
- `benchmark_sample_membership.csv` and `input_with_provenance.csv`: the
  stratum, population row, patent and template frequency of every record.

There are 292 noise-free and 108 noisy paragraphs. The noisy reasons are
untraceable cross-references (75), erroneous chemical names (26) and no
reaction (7). Of the noise-free paragraphs, 289 have complete molecular
references. Records 14, 215 and 235 (a dendrimer, a whole-cell catalyst with
an adsorption resin, and a resin-bound complex) cannot be written as complete
molecular SMILES. They keep their noise-free label and are left out of the
structure endpoints only.

Structures are compared after RDKit normalization that removes
stereochemistry, formal charges, isotope labels and atom-map numbers; the
primary comparison accepts either reference reaction. Strict matching keeps
these features and accepts only the first reference.

The labels and references follow these conventions:

- A paragraph that describes several steps is compared on the step that forms
  the title compound. Each prediction is compared on its step that forms the
  extracted product whose name best matches the title, or on its last step.
- Salt formation, salt exchange and diastereomeric salt resolution are
  noise-free. `salt_formation` marks these 28 paragraphs; one of them is noisy
  because of an untraceable cross-reference. `summary.json` gives the
  denoising accuracy if the other 27 were labeled noisy (312/400).
- An element named as a reagent is the diatomic molecule (H₂, N₂, Cl₂, Br₂ or
  I₂), and carbon monoxide is `[C-]#[O+]`, in the references and in the
  outputs of every configuration.
- Records 15 and 315 of the first release repeated reactions already in the
  sample (records 8 and 309, from other patents of the same families). Each
  was replaced by the first eligible record of the same stratum in a
  reproducible random order (seed 20261021), and all 19 configurations were
  run on the replacements with the same settings.

`label_changes.csv` in the results lists the paragraphs whose label differs
from the first release.

## Model outputs

`result/revised/model_outputs/fully_automatic/` (primary) and `curated/` hold
the reaction SMILES and atom maps of the 19 configurations:

- base and fine-tuned qwen-3.5-9B with reasoning off, low and high;
- base and fine-tuned gpt-4.1-mini;
- gpt-5.4 and gpt-5.6-sol with reasoning none, low and high;
- gemini-2.5-flash with reasoning off;
- gemini-3.1-pro-preview and gemini-3.6-flash with reasoning low and high.

`result/configuration_settings.csv` gives the model identifier, reasoning
parameter, output-token limit, API mode and query dates of each
configuration. Fields that were not recorded are marked `UNKNOWN`.

## Name-to-SMILES resolution

Names are resolved in two passes:

1. the alias table, the local cache and OPSIN;
2. for names still unresolved: PubChem, OPSIN, NCI CIR and ChEBI, queried
   concurrently. The first non-empty result in that order is kept, and names
   that look ambiguous or non-molecular are skipped.

Every structure is validated and canonicalized with RDKit. Disagreements
between resolvers (by InChIKey) are written to `resolver_audit.csv`;
`--resolver-conflict-policy reject` leaves such names unresolved. An element
name returned as a single atom (`[H]`, `[N]`, `[Cl]`, `[Br]`, `[I]`) becomes the
diatomic molecule, and the radical `[C]=O` becomes `[C-]#[O+]`.

Two name-resolution scenarios are evaluated:

- **Fully automatic (primary).** Only the automatic resolvers are used. The
  127 distinct spellings of the names covered by curated rules were re-queried
  on 2026-10-16 against PubChem, OPSIN, NCI CIR and ChEBI. The responses are in
  `result/name_resolution/` and are replayed offline
  (`evaluation/name_resolution_replay.py`).
- **Curated (secondary).** In addition, 46 name-resolution overrides
  (`config/nosmi_overrides.json`) and 44 record-level decisions
  (`result/nosmi_human_review.csv`) are applied. These rules were written with
  knowledge of the benchmark.

The API configurations were resolved offline with
`scripts/name_resolution/run_offline_final_pipeline.py` and the shared name
cache (`result/name_resolution/shared_master_smiles_cache.csv`); the Qwen
configurations were resolved with the live pipeline.

## Population counts and sampling frame (Table 1)

The ORDerly-flagged population (`grants_issue.csv`, 623,290 rows) is
distributed as the release asset
[`grants_issue.zip`](https://github.com/snu-micc/uspto-revisit/releases/download/data-v1.0.0/grants_issue.zip)
(SHA-256 `C5F6A5D1FEF673F247FCE5A481438E33594C32FFDB09807F7A2478D8E1D0EEBF`).
Download it to `evaluation/data/` and run:

```bash
python evaluation/build_sampling_frame.py --strict
python evaluation/build_template_threshold_audit.py
```

| Population | Count |
| --- | --- |
| Many-products / many-reactants / many-reagents / no-reagent, any combination | 82,203 / 52,758 / 236,368 / 200,594 |
| Union of the four flags | 536,372 |
| Exactly one flag and no other label (sampling frame) | 454,475 (59,282 / 20,163 / 207,221 / 167,809) |

The seed of the original 100-per-stratum draw was not recorded; the membership
files give the population row of every record instead. `TemplateFreq` is the
precomputed upstream template frequency. See
[evaluation/data/README.md](evaluation/data/README.md) for the schema.

## First-release evaluation

The first-release benchmark (283 noise-free and 117 noisy labels in
`evaluation/ground_truth_review.csv`, outputs in `result/model_outputs/`) is
scored by:

```bash
python evaluation/build_ground_truth_review.py
python evaluation/evaluate_table2.py
python evaluation/select_best_configurations.py
python evaluation/evaluate_table3.py
python evaluation/evaluate_step_by_step.py
python evaluation/evaluate_threshold_sensitivity.py
python evaluation/evaluate_weighted_extrapolation.py
```

These reproduce `evaluation/results/` (gemini-3.6-flash-high: 245/283
extraction, 299/400 denoising). Their options for name-resolution and
matching variants are described in [evaluation/README.md](evaluation/README.md).

## Run a new extraction

```bash
cp .env.example .env          # add the key of the provider you use
python main.py gpt-extract --provider gemini --model gemini-3.6-flash --input examples/input.csv
python main.py --input result/my_model_output.csv --model-column prediction \
  --output-prefix my_model --fix-names --map-atoms
```

The API extractions were run with `scripts/extraction/`:
`run_reasoning_controlled_extraction.py` (direct calls with an explicit
reasoning level) and `run_gemini_low_batch.py` (Gemini Batch API).

## Fine-tuning data

`examples/finetuning/` holds the 70 fine-tuning examples, in chat format for
gpt-4.1-mini (`openai/`) and in the conversational format for qwen-3.5-9B
(`qwen3.5-9b/`, with the conversion script). Qwen3.5-9B was adapted with 4-bit
QLoRA on 56 training and 14 evaluation examples; the selected adapter is
`checkpoint-42`, released as a
[GitHub Release](https://github.com/snu-micc/uspto-revisit/releases/tag/qwen3.5-9b-qlora-checkpoint-42).
The training settings are in
[examples/finetuning/qwen3.5-9b/README.md](examples/finetuning/qwen3.5-9b/README.md).

## Prior approaches

`benchmarks/` holds the outputs of Zhang et al. and Ai et al. The Ai et al.
components scored in Table S17 are in `benchmarks/ai_et_al/preprocessing/`
and are regenerated by `evaluation/prepare_ai_baseline.py`. The Ai et al.
model was not run on the two replacement records, so its denominator is 288.

## Tests

```bash
pytest
```

## Environment

`pyproject.toml` defines the dependencies; `requirements.txt` mirrors it. The
results were verified with the versions in `requirements-lock.txt` (Python
3.10, pandas 2.3, numpy 2.2, RDKit 2025.09).

## License

No license has been chosen yet; see [LICENSE-PENDING.md](LICENSE-PENDING.md).
The population data artifact is CC BY-SA 4.0
([evaluation/data/DATA_LICENSE.md](evaluation/data/DATA_LICENSE.md)).

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff).
