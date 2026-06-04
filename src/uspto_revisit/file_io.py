"""Small file IO helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event

import pandas as pd


def ensure_directory(directory: str | Path) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_smiles_dict(smiles_dict: dict, filename: str | Path) -> None:
    with Path(filename).open("w", encoding="utf-8-sig") as handle:
        json.dump(smiles_dict, handle, ensure_ascii=False, indent=2)


def monitor_log(file_path: str | Path, stop_event: Event, lines_per_batch: int = 1) -> None:
    """Print selected lines from a log until processing completes."""
    with Path(file_path).open("r", encoding="utf-8") as handle:
        handle.seek(0, 2)
        batch_counter = 0
        line_counter = 0

        while not stop_event.is_set():
            line = handle.readline()
            if not line:
                time.sleep(0.1)
                continue

            if "Starting batch" in line:
                line_counter = 0
                batch_counter += 1
                print(f"--- Batch {batch_counter} ---")

            if line_counter < lines_per_batch:
                print(line, end="")
                line_counter += 1

            if "Processing completed" in line:
                print("All batches processed. Stopping log monitoring.")
                stop_event.set()


def add_smiles_dict(
    column_name: str,
    existing_file_path: str | Path,
    smiles_dict_file_path: str | Path,
    output_path: str | Path,
) -> None:
    """Add a column of SMILES dictionaries to a CSV file."""
    response_frame = pd.read_csv(existing_file_path)
    with Path(smiles_dict_file_path).open("r", encoding="utf-8-sig") as handle:
        smiles_dict = json.load(handle)

    response_frame[column_name] = None
    for idx, _row in response_frame.iterrows():
        response_frame.at[idx, column_name] = smiles_dict[idx]

    response_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
