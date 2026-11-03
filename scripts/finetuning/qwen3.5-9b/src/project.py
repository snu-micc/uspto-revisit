from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "Qwen3.5-9B"


def inside_project(path: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a path and reject all reads/writes outside this project."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if candidate != ROOT and ROOT not in candidate.parents:
        raise ValueError(f"Path must remain inside {ROOT}: {candidate}")
    if must_exist and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def configure_local_caches() -> None:
    """Keep caches and temporary artifacts in the user-approved project tree."""
    cache = ROOT / "hf-cache"
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "transformers"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("TMPDIR", str(ROOT / "tmp"))
    for path in (cache, cache / "hub", cache / "transformers", ROOT / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
