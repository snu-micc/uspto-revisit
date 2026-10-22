# Contextual benchmark outputs

This directory contains outputs from two prior approaches discussed in the
Supplementary Information:

- `zhang_et_al/output.csv`
- `ai_et_al/output.json`

These files are supplied for transparency and contextual analysis. They are not
direct Table 2 comparators because the original models use different inputs,
output schemas, and training objectives from the title-plus-paragraph reaction
extraction task evaluated in this repository.

`ai_et_al/output.json` is the original
`ai_results_accessory_max_seq_2048_reppen.json`. The Git blob is
byte-identical, with SHA-256
`e0b52aaff720f39c57a4b5630b29a274f1b6dc879f674fb2e9d7bc31aa2b23f0`. Windows
checkouts with `core.autocrlf=true` convert it to CRLF line endings; the
CRLF copy has SHA-256
`1fe465511e7a95d515e98dd1bbc51ed8a3eae7586bd55f13e69a579ff6f4d783`.

`ai_et_al/preprocessing/` holds the files behind SI Table S17:

- `ai_baseline_components.csv`: reactant, reagent and product SMILES per
  benchmark row;
- `ai_baseline_name_resolution.csv`: per-name SMILES and resolution source;
- `ai_nosmi_human_review.csv`: review decisions for unresolved names.

`evaluation/prepare_ai_baseline.py` regenerates the component file exactly.
See the top-level README for the Table S17 command.
