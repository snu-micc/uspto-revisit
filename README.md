# LLM-based curation of USPTO reaction data

This repository contains the code, 400-paragraph benchmark, final model
outputs, fine-tuning examples, and evaluation scripts used in our two-stage
reaction-data curation study.

The pipeline has two stages:

1. **Reaction extraction:** an LLM converts a patent title and experimental
   paragraph into structured reaction information, which is resolved to SMILES
   and atom-mapped with LocalMapper.
2. **Denoising:** a template-anomaly filter flags singleton transformation
   templates or reactions for which an assignable reaction center cannot be
   obtained.

The repository can reproduce the reported metrics without calling a vendor API;
all 19 evaluated model-configuration outputs are included.

## Repository structure

```text
src/uspto_revisit/              Extraction, SMILES conversion, and atom mapping
prompts/prompt.txt              Prompt used for structured extraction
examples/input.csv              400-paragraph evaluation input
examples/finetuning/            Non-overlapping fine-tuning examples
result/model_outputs/           Final outputs for all 19 configurations
evaluation/ground_truth_review.csv
                                Final expert annotations (400 paragraphs)
evaluation/*.py                 Scoring and sensitivity-analysis scripts
evaluation/results/             Reproduced tables and row-level outcomes
benchmarks/                     Contextual outputs from prior studies
tests/                          Unit tests
```

Temporary logs, API batch metadata, caches, intermediate review files, and
manuscript-editing artifacts are intentionally excluded.

## Installation

Python 3.9 or later is required. A clean environment is recommended.

```bash
git clone https://github.com/won-24/LLM_Denoise_final.git
cd LLM_Denoise_final
python -m pip install -e ".[dev]"
```

The environment includes RDKit and LocalMapper. LocalMapper installs its
PyTorch/DGL stack and is licensed separately under CC BY-NC-SA 4.0.

## Reproduce the reported evaluation

No API key is required for these commands because the final model outputs are
included.

```bash
python evaluation/build_ground_truth_review.py
python evaluation/evaluate_table2.py
python evaluation/select_best_configurations.py
python evaluation/evaluate_table3.py
python evaluation/evaluate_step_by_step.py
python evaluation/evaluate_threshold_sensitivity.py
python evaluation/evaluate_weighted_extrapolation.py
```

The principal outputs are:

```text
evaluation/results/table2/table2.csv
evaluation/results/table2/table2_metrics.csv
evaluation/results/table2_best_models.csv
evaluation/results/table3/table3.csv
evaluation/results/table3/table3_metrics.csv
evaluation/results/step_by_step/step_by_step_summary_gemini-3.6-flash-high.csv
evaluation/results/template_threshold_sensitivity.csv
evaluation/results/weighted_extrapolation.csv
```

Expected headline values for the selected `gemini-3.6-flash-high`
configuration are:

- reaction-level exact-match accuracy: **86.6%** (245/283);
- template-anomaly denoising accuracy: **74.8%** (299/400);
- strict end-to-end accuracy: **68.0%** (272/400);
- noise-free reactions saved: **86.2%**;
- noisy reactions filtered: **47.0%**.

See [evaluation/README.md](evaluation/README.md) for definitions and the exact
denominators used by every metric.

## Benchmark and annotations

The evaluation set contains 400 patent paragraphs sampled equally from four
ORDerly flag categories. Every ground-truth record was manually reviewed by
a human evaluator. The public `ground_truth_valid` column uses a single
Boolean label: `True` means that a valid ground-truth reaction was established
and `False` means that no valid reaction could be established. The final set
contains 283 `True` (noise-free) and 117 `False` (noisy) paragraphs. The
400-paragraph benchmark was used only for evaluation. The 70 fine-tuning
examples are stored separately and do not overlap with the benchmark.

`evaluation/ground_truth_review.csv` is the authoritative, human-validated
annotation file. For every `ground_truth_valid=True` row, the final reactant,
reagent, product, and reaction fields are explicitly materialized; no automatic
approval label is used.
Running `evaluation/build_ground_truth_review.py` creates the local working file
`evaluation/benchmark_all_configurations.csv` from the released annotations and
model outputs. Because it is fully reproducible, this generated file is not
versioned.

## Model outputs

`result/model_outputs/` contains the final reaction-SMILES and LocalMapper
outputs for:

- base and fine-tuned qwen-3.5-9B at none, low, and high reasoning;
- base and fine-tuned gpt-4.1-mini;
- gpt-5.4 and gpt-5.6-sol at none, low, and high reasoning;
- gemini-2.5-flash with reasoning disabled;
- gemini-3.1-pro-preview and gemini-3.6-flash at low and high reasoning.

Reasoning-specific results are retained for auditability, while
`evaluation/results/table2_best_models.csv` reports one best-performing
configuration per model family.

## Run a new extraction

Copy the environment template and add only the provider key you intend to use:

```bash
cp .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

Run extraction on the included input:

```bash
python main.py gpt-extract --input examples/input.csv
```

Select Gemini explicitly:

```bash
python main.py gpt-extract \
  --provider gemini \
  --model gemini-3.6-flash \
  --input examples/input.csv
```

Generate reaction SMILES and atom mappings from an extraction output:

```bash
python main.py \
  --input result/my_model_output.csv \
  --model-column prediction \
  --output-prefix my_model \
  --fix-names \
  --map-atoms
```

Chemical names are resolved through PubChem, OPSIN, NCI CIR, ChEBI, and
optionally ChemSpider. Recovered SMILES are validated and canonicalized with
RDKit before reaction construction and atom mapping.

## Fine-tuning data

The exact fine-tuning examples are provided under `examples/finetuning/`:

- `openai/samples_for_finetuning_final.jsonl`: chat-format examples used for
  gpt-4.1-mini fine-tuning;
- `qwen3.5-9b/qwen3_5_9b_sft_messages.jsonl`: the same targets converted to the
  conversational format used for qwen-3.5-9B;
- `qwen3.5-9b/convert_gpt_examples_to_qwen35.py`: deterministic conversion and
  validation script.

The fine-tuning examples and the 400 evaluation paragraphs are disjoint. See
[examples/finetuning/README.md](examples/finetuning/README.md) for details.

## Contextual benchmarks

Outputs from Zhang et al. and Ai et al. are included under `benchmarks/` for
contextual analysis. They are not treated as direct Table 2 comparators because
their inputs, output schemas, and training objectives differ from the task used
in this repository.

## Data and security policy

- `.env`, API keys, logs, caches, and batch-job metadata are excluded.
- Publication outputs and the final annotation files are versioned.
- No train/test split was applied to the 400-paragraph benchmark because it was
  used exclusively for evaluation.
- The source patent data and third-party tools remain subject to their original
  licenses and terms.

## Tests

```bash
pytest
```

Tests cover JSON parsing, reaction-step processing, SMILES resolution,
atom-mapping interfaces, CLI behavior, and annotation/evaluation utilities.
