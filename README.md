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
evaluation/draw_replacements.py        Re-runs the two replacement draws (optional)
evaluation/data/revised/               Benchmark labels, references, input, sample membership and draws
result/revised/model_outputs/          Outputs of the 19 configurations (three name-resolution scenarios)
result/revised/replacement_runs/       The 19 configurations on the 30 redrawn records
result/model_outputs/                  Outputs on the first-release input
result/name_resolution/                Shared name cache and the recorded resolver re-queries
result/configuration_settings.csv      Run settings and query dates of the 19 configurations
config/nosmi_overrides.json            Curated name-resolution overrides (data; not used by the pipeline)
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
`evaluation/results/revised/` in about three minutes on a CPU, and it makes no
model, name-resolver or network call. Percentages are rounded half up to one
decimal, and intervals are 95% Wilson score intervals unless stated.

Expected values for the selected configuration, the one with the highest
extraction accuracy: gpt-5.4 without reasoning, with fully automatic name
resolution (without the curated rules and without the lookup tables; see
[Name-to-SMILES resolution](#name-to-smiles-resolution)):

| Quantity | Value |
| --- | --- |
| Reaction-level exact-match extraction | 211/291 (72.5%; 95% CI 67.1–77.3%) |
| Strict matching | 204/291 |
| Original USPTO record, same references | 102/291 (35.1%) |
| gemini-3.6-flash (high), selected in the first release | 210/291 (72.2%; paired exact McNemar p = 1.00) |
| Extraction with the lookup tables on | 213/291 (73.2%) |
| Extraction with curated name resolution | 227/291 (78.0%) |
| Template-anomaly denoising, all 400 labels | 285/400 (71.3%) |
| Noise-free retained / noisy removed | 223/292 / 62/108 |
| End-to-end, 399 assessable / all 400 records | 242/399 (60.7%) / 242/400 (60.5%) |
| Correct retained reactions in the 430,428-record sampling frame | about 223,000 (95% CI 198,000–248,000) |

The same model, gpt-5.4, was one of the six configurations whose consensus
seeded the references (a separate earlier run). Table S30
(`provenance_accuracy.csv`) separates references retained from consensus
candidates from those set by the reviewer.

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
| Supplementary Information 17, lookup tables on | the same files in `../fully_automatic_with_tables/` |
| Tables S23 and S24 | the same files in `../curated/` |
| Table S25 | `table2_metrics_strict.csv` |
| Table S26 | `extraction_paired_tests_primary_lenient.csv` |
| Table S27 | `denoising_paired_tests.csv`, `end_to_end_paired_tests.csv` |
| Table S28 | `selection_adjustment.csv` |
| Table S30 | `../provenance_accuracy.csv` |
| Table S32 | `../noisy_reason_counts.csv` |
| Section 2.5 and Supplementary Information 9 | `../summary.json`, `../label_changes.csv` |

The row-level flags (`table2_flags_*.csv`, `table3_flags.csv` and
`step_flags_structure_endpoint.csv`) give the outcome behind every count.
`input_hashes.json` records the SHA-256 of every input file.

### Statistical analyses

- Exact McNemar tests with a Holm correction within each family: the selected
  configuration against the other 18 and against the original USPTO record
  (`extraction_paired_tests_primary_*.csv`; `extraction_paired_tests_*.csv`
  uses the configuration of highest accuracy instead), and the template
  anomaly against each ORDerly heuristic.
- Selection-adjusted accuracy: the 400 records both select and score the
  configuration, so nested 5-fold cross-validation stratified by sampling
  stratum (200 repeats), leave-one-stratum-out and an out-of-bag stratified
  bootstrap (2,000 resamples) estimate the accuracy of the selection
  procedure.
- Population totals: stratified estimates with a finite-population
  correction on the 430,428-record single-criterion sampling frame.

The random seeds are fixed (`evaluation/benchmark_statistics.py`), so the
results are reproduced to the last digit.

### Rebuild the model outputs (optional)

```bash
python evaluation/rebuild_revised_outputs.py
```

This builds the three scenarios in `result/revised/model_outputs/` from the
first-release outputs, the runs on the replacement records and the two
recorded resolver re-queries, in the order in which they were built. It writes
to `result/revised/rebuilt/` and reports whether every rebuilt file, and the
audits in `result/revised/audits/`, equals the released copy (line endings
aside). It needs the `mapping` extra, because steps whose components change are
re-mapped with LocalMapper.

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
  stratum, population row, patent and template frequency of every record;
- `replacement_draws.csv`: the candidates examined by the two replacement
  draws (below).

There are 292 noise-free and 108 noisy paragraphs. The noisy reasons are
untraceable cross-references (78), erroneous chemical names (23) and no
reaction (7). Of the noise-free paragraphs, 291 have complete molecular
references. Record 215 (a whole-cell catalyst with an adsorption resin) cannot
be written as complete molecular SMILES; it keeps its noise-free label and is
left out of the structure endpoints only.

Structures are compared after RDKit normalization that removes
stereochemistry, formal charges, isotope labels and atom-map numbers; the
primary comparison accepts either reference reaction. Strict matching keeps
these features and accepts only the first reference.

The labels and references follow these conventions:

- A paragraph that describes several steps is compared on the step that forms
  the title compound. Each prediction is compared on its step that forms the
  extracted product whose name best matches the title, or on its last step.
- Salt formation, salt exchange and diastereomeric salt resolution are
  noise-free. `salt_formation` marks these 34 paragraphs; one of them is noisy
  because of an untraceable cross-reference. `summary.json` gives the
  denoising accuracy if the other 33 were labeled noisy (311/400).
- An element named as a reagent is the diatomic molecule (H₂, N₂, Cl₂, Br₂ or
  I₂), and carbon monoxide is `[C-]#[O+]`, in the references and in the
  outputs of every configuration.

Thirty records of the first release were replaced by random draws from the
same stratum, and all 19 configurations were run on the replacements with the
same settings and fully automatic name resolution:

- Records 15 and 315 repeated reactions already in the sample (records 8 and
  309, from other patents of the same families). Each was replaced by the
  first eligible record of its stratum in a reproducible random order (seed
  20261021).
- When the Table 1 criteria are counted explicitly, 28 sampled records met two
  criteria (21 many-products and many-reactants, 6 many-reactants and
  many-reagents, 1 many-products and no-reagent). Each was replaced by a draw
  from its own stratum of the single-criterion frame (seed 20261022).

Both draws place the patents of the frame in random order, take one row per
patent and accept the first rows whose patent and reaction are not yet in the
sample and that link to a non-empty paragraph of Lowe's text files (in the
second draw, also not a paragraph already in the sample). With the population
file (below), `python evaluation/draw_replacements.py` re-runs them and checks
that they give the released replacement records; the paragraph rules are
checked against `replacement_draws.csv`, because Lowe's CML files are not part
of this repository.

`label_changes.csv` in the results lists the paragraphs whose label differs
from the first release.

## Model outputs

`result/revised/model_outputs/fully_automatic/` (primary),
`fully_automatic_with_tables/` and `curated/` hold the reaction SMILES and atom
maps of the 19 configurations:

- base and fine-tuned qwen-3.5-9B with reasoning off, low and high;
- base and fine-tuned gpt-4.1-mini;
- gpt-5.4 and gpt-5.6-sol with reasoning none, low and high;
- gemini-2.5-flash with reasoning off;
- gemini-3.1-pro-preview and gemini-3.6-flash with reasoning low and high.

`result/configuration_settings.csv` gives the model identifier, reasoning
parameter, output-token limit, API mode and query dates of each
configuration. Fields that were not recorded are marked `UNKNOWN`.

## Name-to-SMILES resolution

Names are resolved fully automatically, in two passes:

1. the local cache and OPSIN;
2. for names still unresolved: PubChem, OPSIN, NCI CIR and ChEBI, queried
   concurrently. The first non-empty result in that order is kept, and names
   that look ambiguous or non-molecular are skipped.

The resolver has no lookup tables and applies no curated overrides. Until
2026-10-23 the resolver also
answered names from two tables: 19 abbreviations and trivial names (DCM, THF,
brine, ...) and 13 spelling variants and synonyms of catalyst and reagent
names looked up under a standard name. Most entries had been added after the
benchmark paragraphs were inspected, so the tables were removed. The stored
outputs were produced with them; `evaluation/historical_lookup_tables.py`
keeps them to replay those runs and for the sensitivity analysis below.

Every structure is validated and canonicalized with RDKit. Disagreements
between resolvers (by InChIKey) are written to `resolver_audit.csv`;
`--resolver-conflict-policy reject` leaves such names unresolved. An element
name returned as a single atom (`[H]`, `[N]`, `[Cl]`, `[Br]`, `[I]`) becomes the
diatomic molecule, and the radical `[C]=O` becomes `[C-]#[O+]`.

Three name-resolution scenarios are evaluated:

- **Fully automatic (primary).** Only OPSIN, PubChem, NCI CIR and ChEBI are
  used: no curated rules and no lookup tables, as the released resolver runs.
  Eleven of the 19 table entries and all 13 synonyms had been added in July
  2026, after the benchmark paragraphs were inspected. The 127
  distinct spellings of the names covered by curated rules were re-queried on
  2026-10-16, and the 42 spellings that either lookup table answers or
  rewrites were re-queried on 2026-10-23 with both tables off. The responses
  are in `result/name_resolution/` and are replayed offline
  (`evaluation/name_resolution_replay.py`, `evaluation/lookup_tables.py`).
  Without the tables some common names resolve to another compound (DCM as
  deoxycytidine monophosphate, DMA as a bis-benzimidazole dye) or not at all
  (brine, Pd/C); these errors are kept.
- **Fully automatic with the lookup tables (sensitivity).** As above, with the
  two historical tables. Every configuration extracts 1 to 7 more reactions
  correctly.
- **Curated (secondary).** In addition, 46 name-resolution overrides
  (`config/nosmi_overrides.json`) and 44 record-level decisions
  (`result/nosmi_human_review.csv`) are applied. These rules were written with
  knowledge of the benchmark; they are released as data for this analysis, and
  the pipeline does not apply them.

The June 2026 version of the abbreviation table read "DMA" as acetamide; six
stored outputs of record 136 kept that value, and the two scenarios that use
the table carry the July 2026 entry (N,N-dimethylacetamide) instead
(`result/revised/audits/stale_dma_fix.csv`).

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
python evaluation/draw_replacements.py
```

| Population | Count |
| --- | --- |
| Table 1: many products / many reactants / many reagents / no reagent | 82,203 / 100,777 / 236,253 / 201,761 |
| Table 1: total unique reactions | 550,070 |
| Exactly one Table 1 criterion and no other label (sampling frame) | 430,428 (44,157 / 20,157 / 198,305 / 167,809) |
| Released flags, any combination | 82,203 / 52,758 / 236,368 / 200,594 (union 536,372) |
| Exactly one released flag and no other label (first-release frame) | 454,475 (59,282 / 20,163 / 207,221 / 167,809) |

Table 1 counts each criterion from the atom-mapped reaction SMILES and the
Reagent field: a reactant is a molecule that shares an atom-map number with a
product. Its percentages are relative to the 1,808,937 reactions of Lowe's
grant files. The released flags counted one extra reactant when a reaction
lists no agents. The first-release benchmark was drawn from the released flags;
the 28 records that meet two criteria under the explicit count were redrawn
(see [Benchmark](#benchmark)), so every record of the revised benchmark lies in
the single-criterion frame, on which the population totals are based.

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

Names are resolved fully automatically, with no lookup tables and no curated
overrides.

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
model was not run on the 30 replacement records, so its denominator is 267.

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
