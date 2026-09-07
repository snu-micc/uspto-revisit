"""Download the exact author-reported Qwen snapshot for a new run."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.project import DEFAULT_MODEL, configure_local_caches


def main():
    from huggingface_hub import snapshot_download

    record = json.loads((ROOT / "reported_run_config.json").read_text(encoding="utf-8"))
    configure_local_caches()
    snapshot_download(
        repo_id=record["base_model"]["name_or_path"],
        revision=record["base_model"]["revision"],
        local_dir=DEFAULT_MODEL,
        max_workers=4,
    )
    print(DEFAULT_MODEL)


if __name__ == "__main__":
    main()
