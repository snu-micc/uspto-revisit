# AutoTemplate with RXNMapper remapping

Pinned AutoTemplate core on official RXNMapper 0.3.0 maps, following the upstream workflow. Corpus-informed comparison on the existing flagged benchmark; not independent validation.

| Endpoint | Method | Correct / denominator | Accuracy (95% Wilson CI) |
|---|---|---:|---:|
| primary_structure | AutoTemplate_RXNMapper | 50/292 | 17.1% (13.2–21.9) |
| primary_pooled_set | AutoTemplate_RXNMapper | 53/292 | 18.2% (14.2–23.0) |
| primary_end_to_end | AutoTemplate_RXNMapper | 94/400 | 23.5% (19.6–27.9) |
| denoising | AutoTemplate_RXNMapper | 257/400 | 64.3% (59.4–68.8) |
| primary_structure | AutoTemplate_RXNMapper_skip_top3 | 50/292 | 17.1% (13.2–21.9) |
| primary_pooled_set | AutoTemplate_RXNMapper_skip_top3 | 53/292 | 18.2% (14.2–23.0) |
| primary_end_to_end | AutoTemplate_RXNMapper_skip_top3 | 94/400 | 23.5% (19.6–27.9) |
| denoising | AutoTemplate_RXNMapper_skip_top3 | 257/400 | 64.3% (59.4–68.8) |
| primary_structure | AutoTemplate_source_mapping | 45/292 | 15.4% (11.7–20.0) |
| primary_pooled_set | AutoTemplate_source_mapping | 52/292 | 17.8% (13.8–22.6) |
| primary_end_to_end | AutoTemplate_source_mapping | 64/400 | 16.0% (12.7–19.9) |
| denoising | AutoTemplate_source_mapping | 218/400 | 54.5% (49.6–59.3) |
| primary_structure | AutoTemplate_RXNMapper_no_change_as_failure | 50/292 | 17.1% (13.2–21.9) |
| primary_pooled_set | AutoTemplate_RXNMapper_no_change_as_failure | 53/292 | 18.2% (14.2–23.0) |
| primary_end_to_end | AutoTemplate_RXNMapper_no_change_as_failure | 89/400 | 22.3% (18.4–26.6) |
| denoising | AutoTemplate_RXNMapper_no_change_as_failure | 252/400 | 63.0% (58.2–67.6) |
| primary_structure | Original_USPTO | 99/292 | 33.9% (28.7–39.5) |
| primary_pooled_set | Original_USPTO | 119/292 | 40.8% (35.3–46.5) |
| primary_end_to_end | Original_USPTO | 99/400 | 24.8% (20.8–29.2) |
| denoising | Original_USPTO | 292/400 | 73.0% (68.4–77.1) |
| primary_structure | GPT_5.4_none | 209/292 | 71.6% (66.1–76.4) |
| primary_pooled_set | GPT_5.4_none | 217/292 | 74.3% (69.0–79.0) |
| primary_end_to_end | GPT_5.4_none | 240/400 | 60.0% (55.1–64.7) |
| denoising | GPT_5.4_none | 283/400 | 70.8% (66.1–75.0) |
| strict_structure | AutoTemplate_RXNMapper | 46/292 | 15.8% (12.0–20.4) |
| strict_pooled_set | AutoTemplate_RXNMapper | 47/292 | 16.1% (12.3–20.7) |
| strict_end_to_end | AutoTemplate_RXNMapper | 90/400 | 22.5% (18.7–26.8) |
| strict_structure | AutoTemplate_RXNMapper_skip_top3 | 46/292 | 15.8% (12.0–20.4) |
| strict_pooled_set | AutoTemplate_RXNMapper_skip_top3 | 47/292 | 16.1% (12.3–20.7) |
| strict_end_to_end | AutoTemplate_RXNMapper_skip_top3 | 90/400 | 22.5% (18.7–26.8) |
| strict_structure | AutoTemplate_source_mapping | 43/292 | 14.7% (11.1–19.2) |
| strict_pooled_set | AutoTemplate_source_mapping | 47/292 | 16.1% (12.3–20.7) |
| strict_end_to_end | AutoTemplate_source_mapping | 62/400 | 15.5% (12.3–19.4) |
| strict_structure | AutoTemplate_RXNMapper_no_change_as_failure | 46/292 | 15.8% (12.0–20.4) |
| strict_pooled_set | AutoTemplate_RXNMapper_no_change_as_failure | 47/292 | 16.1% (12.3–20.7) |
| strict_end_to_end | AutoTemplate_RXNMapper_no_change_as_failure | 85/400 | 21.3% (17.5–25.5) |
| strict_structure | Original_USPTO | 95/292 | 32.5% (27.4–38.1) |
| strict_pooled_set | Original_USPTO | 109/292 | 37.3% (32.0–43.0) |
| strict_end_to_end | Original_USPTO | 95/400 | 23.8% (19.8–28.2) |
| strict_structure | GPT_5.4_none | 205/292 | 70.2% (64.7–75.2) |
| strict_pooled_set | GPT_5.4_none | 209/292 | 71.6% (66.1–76.4) |
| strict_end_to_end | GPT_5.4_none | 239/400 | 59.8% (54.9–64.4) |

Execution statuses: {"AutoTemplate_RXNMapper": {"no_template_match": 123, "retained": 277}, "AutoTemplate_RXNMapper_skip_top3": {"no_template_match": 123, "retained": 277}, "AutoTemplate_source_mapping": {"exception": 27, "retained": 261, "no_template_match": 61, "timeout": 51}, "AutoTemplate_RXNMapper_no_change_as_failure": {"no_template_match": 95, "retained": 277, "exception": 28}}.

`AutoTemplate_RXNMapper` is primary. `_skip_top3` applies the upstream branch the pinned CLI never reaches; `_no_change_as_failure` counts the no-change return as the source-mapped run did. All 292 noise-free paragraphs remain in extraction denominators; failed outputs count as non-exact. Component precision/recall uses all 292 noise-free paragraphs; the two reagents of record 215 without a molecular structure count as missed.

Exact paired McNemar tests (primary versus original USPTO, GPT-5.4-none and source-mapped AutoTemplate) with Holm correction within each endpoint are in `paired_tests.csv`. GPT-5.4-none was selected on this same benchmark, so these are internal comparisons.

Original-source, GPT and earlier source-mapped AutoTemplate per-record flags were reproduced before scoring new outputs. No labels, references or prior model outputs were modified.
