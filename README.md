# LLM-based curation of USPTO reaction data

Patent-text extraction, name-to-SMILES conversion, atom mapping and
template-anomaly filtering. Includes 400 author-reviewed paragraphs and frozen
predictions from 19 model configurations (settings: SI S3, Table S4).

## Reproduce results

Requires Python 3.10+. Run from the repository root:

```bash
python -m pip install -e ".[dev]" "rdkit==2025.9.3" "numpy==2.2.6" "pandas==2.3.3" "scipy==1.15.2"
python analysis/reproduce_manuscript.py
python benchmarks/name2smiles_200/score.py --output analysis/results/name2smiles_200
python benchmarks/zhang_et_al/summarize.py --figure
python -m pytest -q
```

Scoring is offline and needs no API keys. Outputs go to
[analysis/results/](analysis/results/); dependencies are recorded in
[requirements-lock.txt](requirements-lock.txt).

Additional resources: [comparators](benchmarks/),
[blinded assessment](analysis/reproduce_blinded_assessment.py),
[population and replacement draws](analysis/LOWE_POPULATION.md), and
[Llama 3 adapters and checksums](https://github.com/snu-micc/uspto-revisit/releases/tag/zhang-llama3-8b-adapters-v1.0.0).

## Data and interpretation

- [analysis/data/](analysis/data/): inputs, author-finalized references,
  replacement history and blinded assessment; model agreement supplied candidates.
- [result/model_outputs/](result/model_outputs/): `fully_automatic` is primary;
  `curated` adds 46 benchmark-informed name overrides as a secondary analysis.
- [config/](config/) and [scripts/finetuning/](scripts/finetuning/): prompts,
  overrides, training examples and run records.

Extraction uses all 292 noise-free paragraphs; denoising and end-to-end scoring
use all 400. Multi-step records use the title-forming step. Primary matching
ignores stereochemistry, charge, protonation, isotopes and atom maps; strict
matching preserves the first four. Selection and evaluation reuse this benchmark.

The [200-name comparison](benchmarks/name2smiles_200/) reproduces 82/200 original
and 90/200 rerun matches against the same corrected references. Its CSVs document
predictions and 15 reference corrections; [protocol.json](benchmarks/name2smiles_200/protocol.json)
records provenance and the unavailable original prompt/model snapshot.

Blinded assessment is a post hoc, single-evaluator check (46/50 label agreement).
Population projections are conditional on the retained benchmark.
AutoTemplate's corpus contains benchmark source reactions, so it is not
independent validation. Zhang results (SI S13, Figure S2, Table S18) measure
role presence, not accuracy; reactants use the selected primary-product query.

## License and citation

[LICENSE-PENDING.md](LICENSE-PENDING.md) · [CITATION.cff](CITATION.cff) ·
[SECURITY.md](SECURITY.md). Third-party licenses apply; LocalMapper is CC BY-NC-SA 4.0.
Built with Meta Llama 3; [license](benchmarks/zhang_et_al/LLAMA3_LICENSE.txt).
