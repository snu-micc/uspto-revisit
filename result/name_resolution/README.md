# Name-resolution records

This folder documents how chemical names in the released model outputs were
turned into structures. It also holds the recording that the fully automatic
sensitivity analysis replays offline.

| File | Contents |
| --- | --- |
| `shared_master_smiles_cache.csv` | The shared name-to-SMILES cache (4,378 keys) used to resolve the 13 API configurations offline. |
| `shared_master_smiles_cache_audit.csv` | How each cached name was chosen: majority vote over earlier runs, with all candidates. |
| `qwen_prior_cache_audit.csv` | The same audit for the prior cache that seeded the live resolution of the six Qwen configurations. |
| `model_name_smiles_conflicts.csv` | Cross-model conflicts in the earlier runs and the structure chosen for each. |
| `nosmi_recovery_audit.csv`, `nosmi_recovery_summary.json` | The earlier NoSmi recovery, including which names the curated overrides resolved. |
| `requery_2026-10-16_raw.jsonl` | Every HTTP exchange of the 2026-10-16 re-query of the curated names (see below). |
| `requery_2026-10-16_capture.json` | Query times, endpoints, rate limits, response counts and the SHA-256 of the recording. |

## The 2026-10-16 re-query

The released outputs carry structures that came from curated choices: the 46
entries of `config/nosmi_overrides.json`, the row-level `use_smiles` decisions
in `result/nosmi_human_review.csv`, and reviewed SMILES that the shared cache
reused for the same name in other rows (alcohol and hexanes). To report a
fully automatic result, every distinct extracted spelling of these names (127
spellings) was sent to the public resolvers once, with the authors' approval,
and every exchange was recorded.

- **Date and time:** 2026-10-16. Main run 08:19–08:22 UTC. Re-sending of
  unanswered requests 08:23–08:30 UTC. The first-pass OPSIN requests that the
  main run had not made were sent 10:42–10:43 UTC.
- **Resolvers and endpoints:**
  - PubChem PUG REST: `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/SMILES/JSON`
  - OPSIN: `https://opsin.ch.cam.ac.uk/opsin/{name}.smi`
  - NCI CIR: `http://cactus.nci.nih.gov/chemical/structure/{name}/smiles`
  - ChEBI search and record:
    `https://www.ebi.ac.uk/chebi/backend/api/public/es_search?term={name}&page=1&size=3`
    and `https://www.ebi.ac.uk/chebi/backend/api/public/compound/{id}/`
- **Request rules:**
  - The released resolver code (`src/uspto_revisit/smiles_fetch.py`) was used,
    with its own retry rules (back-off on HTTP 429 or busy, no retry on other
    4xx).
  - Overrides, row decisions and the shared cache were disabled.
  - The SHA-256 of `smiles_fetch.py` used for the run is recorded in the
    capture file and equals the released file.
  - Requests were rate-limited per host: PubChem at most one per 0.25 s, CIR
    one per second, OPSIN and ChEBI one per 0.2 s. At most four ran at once,
    each with a 120 s timeout.
- **CIR retry handling.** CIR dropped connections intermittently, so some
  requests got no HTTP answer at all. A dropped connection is not a resolver
  answer, so the 58 URLs without an HTTP answer were re-sent one at a time,
  5 s apart, with up to 6 attempts each. This took 2 rounds and 66 requests,
  after which every recorded URL had an HTTP answer.
- **Totals:** 568 requests. The recording keeps every attempt, including
  connection errors. Replay uses the last attempt for each URL, as the live
  run did.
- **Integrity:** `evaluation/name_resolution_replay.py` refuses a recording
  whose SHA-256 differs from `raw_sha256` in the capture file. The recording
  is stored without line-ending conversion (`.gitattributes`).

Each record holds `utc`, `url`, `attempt`, and either `status` with `body` or
`error`. Retries carry `retry_of_unanswered`; the later OPSIN requests carry
`first_pass_fill`. The bodies are the services' verbatim responses. Error
messages from the Windows network stack are in the operating system's
language.

### Using it

Replay is the default and needs no network:

```bash
python evaluation/evaluate_table2.py --name-resolution fully-automatic
python evaluation/evaluate_table3.py --name-resolution fully-automatic
python evaluation/evaluate_step_by_step.py --name-resolution fully-automatic
```

Changed reaction steps are re-mapped with LocalMapper, which needs
`python -m pip install -e ".[mapping]"`. `--resolution-order retry` replays
the NoSmi retry path alone instead of the released two-pass order.

To redo the queries live (optional; not needed for any reported number):

```bash
python scripts/name_resolution/requery_curated_names.py --dry-run
python scripts/name_resolution/requery_curated_names.py --live --output result/name_resolution/requery_<date>_raw.jsonl
```

Pass the new file to the evaluation scripts with `--requery-log`.
