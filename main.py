"""Command-line entry point for running the USPTO Revisit pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    src_path = Path(__file__).resolve().parent / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))


def main() -> int:
    _ensure_src_on_path()
    from uspto_revisit.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
