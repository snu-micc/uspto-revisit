# Matched AutoTemplate comparison

AutoTemplate core at pinned upstream commit, using source atom mapping. Corpus-informed comparison on the existing flagged benchmark; not independent validation.

| Endpoint | Method | Correct / denominator | Accuracy (95% Wilson CI) |
|---|---|---:|---:|
| primary_structure | AutoTemplate_source_mapping | 45/292 | 15.4% (11.7–20.0) |
| primary_pooled_set | AutoTemplate_source_mapping | 52/292 | 17.8% (13.8–22.6) |
| primary_end_to_end | AutoTemplate_source_mapping | 64/400 | 16.0% (12.7–19.9) |
| denoising | AutoTemplate_source_mapping | 218/400 | 54.5% (49.6–59.3) |
| primary_structure | Original_USPTO | 99/292 | 33.9% (28.7–39.5) |
| primary_pooled_set | Original_USPTO | 119/292 | 40.8% (35.3–46.5) |
| primary_end_to_end | Original_USPTO | 99/400 | 24.8% (20.8–29.2) |
| denoising | Original_USPTO | 292/400 | 73.0% (68.4–77.1) |
| primary_structure | GPT_5.4_none | 209/292 | 71.6% (66.1–76.4) |
| primary_pooled_set | GPT_5.4_none | 217/292 | 74.3% (69.0–79.0) |
| primary_end_to_end | GPT_5.4_none | 240/400 | 60.0% (55.1–64.7) |
| denoising | GPT_5.4_none | 283/400 | 70.8% (66.1–75.0) |
| strict_structure | AutoTemplate_source_mapping | 43/292 | 14.7% (11.1–19.2) |
| strict_pooled_set | AutoTemplate_source_mapping | 47/292 | 16.1% (12.3–20.7) |
| strict_end_to_end | AutoTemplate_source_mapping | 62/400 | 15.5% (12.3–19.4) |
| strict_structure | Original_USPTO | 95/292 | 32.5% (27.4–38.1) |
| strict_pooled_set | Original_USPTO | 109/292 | 37.3% (32.0–43.0) |
| strict_end_to_end | Original_USPTO | 95/400 | 23.8% (19.8–28.2) |
| strict_structure | GPT_5.4_none | 205/292 | 70.2% (64.7–75.2) |
| strict_pooled_set | GPT_5.4_none | 209/292 | 71.6% (66.1–76.4) |
| strict_end_to_end | GPT_5.4_none | 239/400 | 59.8% (54.9–64.4) |

Execution statuses: {"exception": 27, "retained": 261, "no_template_match": 61, "timeout": 51}.

All 292 noise-free paragraphs remain in extraction denominators. The one incomplete structural reference and all filtered/failed outputs count as non-exact. Component precision/recall uses all 292 noise-free paragraphs; the two reagents of record 215 without a molecular structure count as missed. The pooled reactant/reagent set endpoint is a separate sensitivity analysis.

Exact paired McNemar tests and Holm correction within each endpoint are in `paired_tests.csv`. GPT-5.4-none was selected on this same benchmark, so these are internal comparisons.

Both primary and strict original-source/GPT per-record flags were reproduced before scoring new outputs. No labels, references or prior model outputs were modified.
