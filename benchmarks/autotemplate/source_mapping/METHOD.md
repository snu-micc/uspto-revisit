# Source-mapping run

AutoTemplate on the original Lowe atom maps. Settings and scoring are in
[protocol.json](protocol.json); results are in [RESULTS.md](RESULTS.md).

| File | Contents |
| --- | --- |
| `run_comparison.py` | Windows wrapper around the upstream functions. |
| `verify_provenance.py`, `provenance_verified.json` | Vendor check against upstream Git blob SHA-1; SHA-256 and runtime versions. |
| `validate_scorer.py`, `scorer_validation.json` | Check that the scorer reproduces the baseline per-record flags. |
| `score_comparison.py` | Scorer. |
| `application_freeze.json` | Integrity check and hashes, frozen before application. |
| `completion.json` | Hashes of the final outputs. |

## Implementation

- Upstream https://github.com/Lung-Yi/AutoTemplate, commit
  `6b17ea08cc3dded90a9eee8e25b74e8a60e4cb94`. The 24 files in `../vendor` are
  byte-identical to the Git blobs in `../vendor/upstream_manifest.json`;
  line-ending changes from transfer were restored only where the result
  matched the upstream object exactly. No vendored chemistry function was
  patched.
- The wrapper imports the upstream extractor and execution utilities, loads
  the pure graph module without its optional CGRtools package initializer, and
  compiles the four top-level application and reconstruction functions from
  the pinned CLI's AST. Product balancing, rendering, Reaxys preprocessing and
  RXNMapper are not invoked. This avoids the CLI's Windows worker-pickling and
  failed-record-writing problems; it does not reproduce the CLI as a process.
- Library: Lowe's left and middle SMILES sections are folded together;
  upstream isotope clearing and reagent removal run unchanged. The merge skips
  only isomorphism comparisons that a necessary graph invariant rules out,
  keeping upstream insertion order and replacement decisions. It had to equal
  the original function on the label-blind pilot (first 1,000 source
  reactions) and the first 500 templates before full merging. Frequency ties
  keep source order.
- Execution: extraction started with 12 workers and resumed with 20 workers
  after 567,000 committed rows to increase CPU utilization. The 60-second
  timeout and chemistry settings were unchanged; no benchmark scores were
  inspected for this decision.
- Checkpoints: one SQLite row per source reaction. Exceptions, crashes and
  60 s timeouts are recorded, not dropped. After the restart, committed rows
  were kept and uncommitted or in-flight rows recomputed. Integrity, coverage
  and endpoint indices were verified before application.
- Application: upstream preprocessing, first successful product. Every target
  traverses the frequency-ordered library, as in the pinned CLI, whose
  per-record `template` field stays `None`, so its skip-top-3 branch never
  runs. No title or ground truth is used to build the library or to choose a
  template or product.
- Roles: removed spectators are restored as a separate reagent stream. Their
  map numbers can collide with renumbered product maps, so the parser
  classifies the curated core by positive product-map overlap and appends the
  removed-reagent stream as reagents. A naive combined-SMILES parser is kept
  as a diagnostic.
- Statistics: Holm correction covers the two comparator tests within each
  endpoint. Component precision and recall use all 292 noise-free paragraphs;
  the two reagents of record 215 without a molecular structure count as missed.
  Pooled reactant/reagent sets are a separate sensitivity endpoint.

## Reproduction

Use the versions in `provenance_verified.json` and set `PYTHONHASHSEED=0`.
Obtain Lowe's grant source using the
[population reconstruction instructions](../../../analysis/LOWE_POPULATION.md#run).
The wrapper reads `analysis/data/lowe_raw_v3/raw/1976_Sep2016_USPTOgrants_smiles.rsmi`
by default; set `AUTOTEMPLATE_LOWE_SOURCE` to use another location.
Run from the repository root:

```bash
python benchmarks/autotemplate/source_mapping/verify_provenance.py
python benchmarks/autotemplate/source_mapping/run_comparison.py extract --workers 20 --timeout 60
python benchmarks/autotemplate/source_mapping/run_comparison.py merge
python benchmarks/autotemplate/source_mapping/validate_scorer.py
python benchmarks/autotemplate/source_mapping/run_comparison.py apply --workers 12 --timeout 60
python benchmarks/autotemplate/source_mapping/score_comparison.py
```

Extraction of the last 1.24 million reactions took 25 min with 20 workers, and
application 10 min with 12 workers. On other hardware, timeouts can differ
despite fixed seeds; report execution coverage with accuracy. The 60 s cap is
a condition of this comparison, not a chemical failure criterion of
AutoTemplate.

## Related RXNMapper execution

Corpus mapping resumed from committed SQLite rows after two interruptions.
The first suppressed excessive Transformers warnings without changing mapping
settings. The second reduced workers from four to three, capped GPU memory at
0.23 per worker and added a 1,200-second stall watchdog. Batch size 8, chunk
size 256, input preparation and failure handling were unchanged; failed batches
were retried one reaction at a time. Under the cap, all 400 saved benchmark
maps were reproduced exactly. Uncommitted rows were recomputed, and benchmark
scores were not inspected to decide these changes.
