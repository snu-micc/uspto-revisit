# Rebuild Table 1 from Lowe's grant SMILES

Rebuilds the flagged population and sampling frame from Lowe's figshare v1
patent-grant release (1,808,937 reactions; policy `lowe-grants-raw-v3`).

## Run

Verified with RDKit 2025.09.6 and LocalMapper 0.1.5. Only the RDKit template
extractor is used; DGL, neural mapping and API keys are unnecessary.
Run from the repository root:

```powershell
python analysis/rebuild_lowe_population.py --output analysis/data/lowe_raw_v3 --benchmark analysis/data/input.csv --workers 12
python analysis/table1_from_lowe_population.py --population-dir analysis/data/lowe_raw_v3
python analysis/audit_benchmark_count_criteria.py --input analysis/data/input.csv --output analysis/results/lowe_count_audit
```

These commands rebuild the population, independently recount Table 1 and audit
benchmark eligibility. The first downloads and verifies the source archive;
extraction requires `tar` with 7z support. Use `--archive PATH` or `--raw PATH`
for local inputs. Completed chunks resume only with matching inputs and settings.
`--limit N` is for pilots, not final frequencies or eligibility.

| Source | SHA-256 |
| --- | --- |
| `1976_Sep2016_USPTOgrants_smiles.rsmi` | `5e91117efdb228642647ff26fc5f41b8babf544ca60b2967d76adf243352b626` |
| `1976_Sep2016_USPTOgrants_smiles.7z` | `d10f8ecb013d3f7bc85abdf6a8c5da8a7dd992931c1fda5db14fbd764343bc4e` |

## Rules and results

- **Roles:** mapped product-side molecules are products. Input molecules sharing
  a positive atom-map number with a product are reactants; others are reagents.
  Count whole dot-separated molecules, including repeats. Unmapped products
  are recorded separately.
- **Flags:** reactants >2, products >1, reagents >5, or reagents =0, regardless
  of template-extraction success.
- **Templates:** extract reaction centers with LocalMapper
  (`include_neighbors=False`). Frequencies include duplicate source rows;
  failed or missing templates have no frequency, not frequency one.
  Mapping enumeration is capped at 4,096 permutations.
- **Duplicates:** compare canonical molecule lists by role, retaining
  multiplicity, stereochemistry, charge and isotopes but removing atom maps.
  Patent and paragraph identifiers are excluded from this comparison.
- **Sampling frame:** exactly one count flag and no assigned singleton template.
  Duplicate flags and template failures alone do not exclude records.

| Result | Reactions |
| --- | ---: |
| Many products / many reactants / many reagents / no reagent | 142,390 / 172,536 / 249,291 / 222,957 |
| Table 1 union | 677,316 (37.44%) |
| Sampling frame | 536,030 |
| Output CSV, including diagnostic flags | 1,188,575 |

## Output files

Generated files are in `analysis/data/lowe_raw_v3/` (not distributed):

| File | Contents |
| --- | --- |
| `grants_issue.csv` | Count and diagnostic flags; `RawRow` is the zero-based source row |
| `table1.csv` | Counts, union and percentages |
| `sampling_frame.csv.gz` | Eligible rows and sampling groups |
| `benchmark_membership.csv` | Source matches, counts, template frequencies and eligibility reasons |
| `summary.json`, `manifest.json` | Totals, completeness, versions and hashes |
| `table1_recount_validation.json` | Independent recount checks |

The released summaries, manifests, membership records and final audit are in
`analysis/results/lowe_population/`. The CSV row count includes diagnostic flags
and is therefore larger than the Table 1 union.

## Replacement sampling

Draw history and seeds are in `analysis/data/replacement_draws.csv`.
`draw_replacements.py` reruns the first two draws (20261021 and 20261022)
and checks all three rounds.

For round 3, `prepare_lowe_replacements.py` (seed 20261026) drew seven
replacements within the same sampling groups. It requires the generated frame,
Lowe's CML files and an index (`--cml-root`, `--cml-index`); the latter inputs
are not distributed. Its default `--eligibility-scope count-criteria` addresses
count-criterion violations.

The author reviewed all seven replacements, and all 19 configurations were
rerun. Outputs are in `result/replacement_outputs/source_audit/`; raw responses are
not distributed. The final benchmark has 100 records per group, but is not a
fresh probability sample of the rebuilt frame. Population projections remain
conditional.
