"""Resolve and map replacement outputs together with a fresh automatic cache.

No ground truth, historical lookup tables or curated override file is read.
Unresolved names and extraction/mapping failures remain explicit outputs.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from analysis.run_replacement_apis import CONFIGS
except ModuleNotFoundError:
    from run_replacement_apis import CONFIGS

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(2**31-1)


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--wait", action="store_true")
    a = p.parse_args(argv); a.output.mkdir(parents=True, exist_ok=True)
    while True:
        mp = a.api_dir / "run_manifest.json"
        m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        if m.get("status") in ("complete", "incomplete"):
            break
        if not a.wait:
            raise ValueError("API run has not finished")
        time.sleep(5)
    combined, raw_hashes = [], {}
    for config in CONFIGS:
        path = a.api_dir / (config + "_output.csv")
        rows = read(path); raw_hashes[config] = sha(path)
        if len(rows) != 7 or {int(r["benchmark_idx"]) for r in rows} != {3,6,13,50,62,69,82}:
            raise ValueError("Expected only the seven approved replacement IDs: " + config)
        combined.extend({"configuration": config, **r} for r in rows)
    merged = a.output / "combined_extractions.csv"
    write(merged, combined)
    processed = a.output / "combined_processed.csv"
    command = [sys.executable, str(ROOT / "main.py"), "--input", str(merged.resolve()),
               "--model-column", "prediction", "--output-prefix", "replacement",
               "--output", str(processed.resolve()), "--with-smiles-output", str((a.output / "with_smiles.csv").resolve()),
               "--batch-dir", str((a.output / "automatic_cache").resolve()),
               "--log-file", str((a.output / "resolver.log").resolve()),
               "--lookup-concurrency", "16", "--reprocess-concurrency", "8", "--map-atoms", "--mapping-device", "cpu"]
    subprocess.run(command, check=True)
    rows = read(processed)
    if len(rows) != len(combined):
        raise ValueError("Postprocessing dropped records")
    files = {}
    for config in CONFIGS:
        subset = [{(config+k[len("replacement"):] if k.startswith("replacement_") else k):v for k,v in r.items()}
                  for r in rows if r["configuration"] == config]
        if len(subset) != 7:
            raise ValueError("Incomplete processed configuration: " + config)
        path = a.output / (config + "_reaction_smiles_final.csv")
        write(path, subset); files[config] = sha(path)
    manifest = {"status": "processed_not_scored", "records_per_configuration": 7, "configurations": len(CONFIGS),
                "raw_sha256": raw_hashes, "output_sha256": files, "command": command,
                "ground_truth_read": False, "sensitivity_lookup_tables_loaded": False,
                "curated_overrides_loaded": False, "cache": "fresh automatic cache shared across the 13 configurations",
                "author_ground_truth_status": "not supplied to this step"}
    (a.output / "processing_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("Processed 13 x 7 API outputs; no labels scored", flush=True)


if __name__ == "__main__":
    main()
