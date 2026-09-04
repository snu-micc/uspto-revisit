# Evaluation data and provenance

This directory contains small, versioned inputs used by the evaluation scripts.
The large population artifact described below is distributed separately from
the Git repository.

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

## `weighted_sampling_summary.csv`

This file contains the sampling-frame counts used by
`evaluation/evaluate_weighted_extrapolation.py`. See the parent
[`README.md`](../README.md) for the weighting definitions.
