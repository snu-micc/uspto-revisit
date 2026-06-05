# USPTO Revisit

Utilities for extracting structured reaction JSON from patent text, resolving chemical names to SMILES, and generating reaction SMILES.

## Features

- Generate reaction JSON from `title` and `paragraph` columns with OpenAI models.
- Parse GPT-style reaction JSON into reaction skeletons.
- Resolve chemical names to SMILES with OPSIN and optional PubChem/CIR/ChemSpider fallback.
- Create final reaction SMILES files with model-specific output names.

## Repository Layout

```text
src/uspto_revisit/      Python package
prompts/prompt.txt      GPT extraction prompt template
examples/examples_used_for_finetuning/    Examples used for finetuning gpt-4.1-mini
examples/input.csv               Default GPT input file
result/                 Local outputs, ignored by git
result/smiles_batches/  Local cache/intermediate SMILES files, ignored by git
```

## Installation

```bash
git clone https://github.com/snu-micc/uspto-revisit.git
cd uspto-revisit
python -m pip install -r requirements.txt
```

For editable development:

```bash
python -m pip install -e ".[dev]"
```

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell, use `Copy-Item .env.example .env`.

Fill in your OpenAI key and model:

```env
# Optional: used only for ChemSpider fallback during NoSmi reprocessing.
CHEMSPIDER_API_KEY=your_chemspider_key
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
# OPENAI_MODEL=gpt-5.4
```

`CHEMSPIDER_API_KEY` is optional. If it is empty, ChemSpider fallback is skipped.

Do not commit `.env` or real API keys.

## Input Format

By default, GPT extraction reads:

```text
input.csv
```

The file must contain at least:

```csv
title,paragraph
```

You can also pass another file with `--input`.

## Run GPT Extraction

```bash
python main.py gpt-extract
```

Recommended for longer runs:

```bash
python main.py gpt-extract --input input.csv --semaphore-size 5 --timeout-seconds 180
```

Default settings:

```text
input: input.csv
output: result/{OPENAI_MODEL}_output.csv
concurrent OpenAI requests: 10
request timeout: 20 seconds
```

The GPT output CSV is updated as rows finish, so partial progress is preserved during long runs.

To test GPT extraction with a small public example:

```bash
python main.py gpt-extract --input examples/sample_patent_text.csv
```

GPT output columns:

```text
idx
title
paragraph
prediction
error
```

## Generate Reaction SMILES

After GPT extraction, run:

```bash
python main.py --input result/gpt-4.1-mini_output.csv --model-column prediction --fix-names
```

For a faster test without PubChem/CIR/ChemSpider reprocessing:

```bash
python main.py --input result/gpt-4.1-mini_output.csv --model-column prediction --fix-names --skip-reprocess
```

Default final output:

```text
result/{OPENAI_MODEL}_reaction_smiles.csv
```

For example, `OPENAI_MODEL=gpt-4.1-mini` creates:

```text
result/gpt-4.1-mini_output.csv
result/gpt-4.1-mini_reaction_smiles.csv
```

Final reaction SMILES columns include:

```text
{OPENAI_MODEL}_smiles
{OPENAI_MODEL}_skeleton
{OPENAI_MODEL}_rxn
```

## Outputs

Generated files are written under `result/`:

```text
result/{OPENAI_MODEL}_output.csv
result/{OPENAI_MODEL}_reaction_smiles.csv
result/smiles_fetch.log
result/smiles_batches/
```

`result/`, `data/`, and `.env` are ignored by git.
