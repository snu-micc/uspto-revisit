# Evaluation data and provenance

This directory contains small, versioned inputs used by the evaluation scripts.
The large population artifact described below is distributed separately from
the Git repository.

## `revised/`

The revised benchmark used for the manuscript and SI
(`evaluation/reproduce_revised.py`).

| File | Contents |
| --- | --- |
| `ground_truth.csv` | One row per paragraph (`idx` 0–399): `title`, `paragraph`, `label` (`noise-free` or `noisy`), `noisy_reason` (untraceable cross-references, erroneous chemical names or no reaction), `salt_formation`, `structure_evaluation_eligible`, `structure_exclusion_reason`, `reference_source` (`retained_consensus_candidate` or `saved_gt`), and the reference reactions `ground_truth_{reactants,reagents,products,reaction}_smiles` with an optional alternative `ground_truth_2_*`. References are given only for the 291 eligible records. |
| `input.csv` | The extraction input: `Issue`, `title`, `paragraph` and the original USPTO reaction SMILES `Lowe_smiles`. Thirty rows hold redrawn records: 15 and 315 (first draw) and the 28 records of the second draw (`replacement_draws.csv`). |
| `benchmark_sample_membership.csv` | As the first-release file below, with `template_frequency` for every record. `source_row` is the candidate record whose paragraph in Lowe's CML files is the benchmark paragraph, where the paragraph text can be matched (otherwise the first candidate); the 30 redrawn records keep the row actually drawn. Every record lies in the single-criterion frame of its stratum. |
| `input_with_provenance.csv` | `input.csv` with stratum, population row and patent metadata. |
| `replacement_draws.csv` | The candidates examined by the two replacement draws (all were accepted): `round` (1: seed 20261021; 2: seed 20261022), `stratum`, `rank` in the random order, population `source_row`, `patent_number`, the record it replaces (`replaces_idx`), and the paragraph of Lowe's text files it is linked to (`text_file`, and SHA-1 and length of the whitespace-normalized text). `evaluation/draw_replacements.py` re-runs the draws and checks the paragraph rules against this file. |

## `grants_issue.zip`

Download the versioned artifact from the
[`data-v1.0.0` release](https://github.com/snu-micc/uspto-revisit/releases/download/data-v1.0.0/grants_issue.zip)
and save it as:

```text
evaluation/data/grants_issue.zip
```

Artifact metadata:

- archive member: `grants_issue.csv`;
- data rows: 623,290 (excluding the header);
- scope: reactions in the released ORDerly-flagged subset;
- SHA-256:
  `C5F6A5D1FEF673F247FCE5A481438E33594C32FFDB09807F7A2478D8E1D0EEBF`.

Verify the archive before using it:

```bash
sha256sum evaluation/data/grants_issue.zip
```

PowerShell:

```powershell
Get-FileHash evaluation/data/grants_issue.zip -Algorithm SHA256
```

The CSV schema is:

| Column | Description |
| --- | --- |
| `Year` | Patent year stored with the source reaction record. |
| `PatentNumber` | Patent identifier. |
| `ParagraphNum` | Paragraph identifier stored with the reaction. |
| `ReactionSmiles` | Reaction SMILES used for benchmark linkage. |
| `Template` | Precomputed reaction-template representation. |
| `Reagent` | Reagent field stored with the source reaction. |
| `NAtomChanges` | Precomputed number of atom-level changes. |
| `TemplateFreq` | Precomputed corpus-level frequency associated with `Template`. |
| `NMolecules` | Stored molecule count. |
| `Issue` | ORDerly issue/flag label associated with the record. |

### Provenance boundary

The archive is not the raw/full approximately 1.9-million-reaction USPTO
corpus. It contains 623,290 flagged records selected for the study. The
`Template` and `TemplateFreq` columns were computed upstream and are published
here as analysis inputs. The historical full-corpus extraction code and full
source corpus required to independently regenerate these two columns are not
part of this release. Consequently, `TemplateFreq` must not be recomputed by
simply counting `Template` within this subset and must not be described as a
frequency derived from only these 623,290 rows.

This repository supports the following reproducible path:

```text
released grants_issue.zip with precomputed TemplateFreq
    -> build_template_threshold_audit.py
    -> template_threshold_audit.csv
    -> evaluate_threshold_sensitivity.py
    -> template_threshold_sensitivity.csv
```

It does not claim the following path:

```text
raw/full USPTO corpus
    -> historical template extraction
    -> corpus-level TemplateFreq
```

### Source attribution and reuse

This archive is an adaptation of the Open Reaction Database `uspto-grants`
records selected with ORDerly criteria. The data artifact is released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), consistent
with the license of the upstream [`ord-data` datasets](https://github.com/open-reaction-database/ord-data/blob/main/LICENSE).
The ORDerly software is separately distributed under the
[MIT License](https://github.com/sustainable-processes/ORDerly/blob/main/LICENSE),
and its method is described by Wigh et al.
([DOI: 10.1021/acs.jcim.4c00292](https://doi.org/10.1021/acs.jcim.4c00292)).
See [`DATA_LICENSE.md`](DATA_LICENSE.md) for attribution and a description of
the changes represented by this artifact.

## `template_threshold_audit.csv`

This versioned, 400-row file contains the benchmark-level inputs for the
exploratory frequency-threshold comparison. It can be regenerated from the
released population artifact with:

```bash
python evaluation/build_template_threshold_audit.py
```

The resulting file records the selected model, ground-truth validity,
precomputed template frequency, mapping/template-failure and identical-
structure flags, and the N=1, N<5, and N<10 decisions. These frequency cutoffs
are exploratory sensitivity settings. The primary Table 3 template-anomaly
criterion is the inability to assign a reliable reaction center and does not
use a frequency cutoff.

## `reference_provenance.csv`

How each benchmark reference was established, one row per record (`idx`):

- `consensus_accepted` (219 records): the reviewer accepted the reaction on
  which the consensus models agreed.
- `hand_set` (64 records): the reviewer set the reference by hand.
- `no_valid_reference` (117 records): noisy; no valid reaction could be
  established.

`review_status` gives the corresponding review outcome: approved, corrected
or excluded. The file describes the first-release references; the revised
references carry `reference_source` in `revised/ground_truth.csv`.

## `weighted_sampling_summary.csv`

This file contains the sampling-frame counts used by
`evaluation/evaluate_weighted_extrapolation.py`. See the parent
[`README.md`](../README.md) for the weighting definitions. Its `population_n`
values are the 501,997 reactions flagged by exactly one of the four sampled
flags (other labels allowed).

## Sampling-frame files

The following files are written by `evaluation/build_sampling_frame.py` from
`grants_issue.zip` and `examples/input.csv`.

| File | Contents |
| --- | --- |
| `sampling_frame_counts.csv` | One row per population definition and group: `frame`, `group`, `n`, the manuscript value, whether they match, and the definition. `table1` gives the Table 1 counts and union (criteria counted from the reaction SMILES and the Reagent field), `single_criterion_frame` the frame of the revised benchmark (exactly one Table 1 criterion, no `rare_temp` or `duplicated` label), `table1_any_flag` the same groups from the released flags, `exactly_one_of_four` the first-release weighting population, and `single_label_frame` the frame the first-release benchmark was drawn from. `label_any` and `all_rows` are for reference. |
| `sampling_frame_label_combinations.csv` | Count of each `Issue` label combination, and whether it belongs to the single-label frame. |
| `benchmark_sample_membership.csv` | One row per benchmark record (`idx`), with columns: `stratum`, the ORDerly label, `source_row` (0-based data row of `grants_issue.csv`; the first match when several rows share the reaction SMILES and label), `source_row_unique`, `candidate_count`, `candidate_source_rows`, year, patent and paragraph number of `source_row`, all candidate patent numbers, `TemplateFreq` and the SHA-256 of the trimmed reaction SMILES. |
| `weighted_sampling_summary_single_label_frame.csv` | `weighted_sampling_summary.csv` with `population_n` replaced by the single-label-frame counts, for use with `evaluate_weighted_extrapolation.py --input`. |

The random seed of the original draw was not recorded. The two replacement
draws of the revised benchmark have recorded seeds and are re-run by
`evaluation/draw_replacements.py`.
