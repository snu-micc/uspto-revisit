"""Protect record alignment when loading text and provenance from separate CSVs."""
import csv

import pytest

from analysis.benchmark_inputs import load_benchmark


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def fixture(tmp_path):
    text = [{"idx": "0", "title": "first", "Lowe_smiles": "C>>C"},
            {"idx": "1", "title": "second", "Lowe_smiles": "O>>O"}]
    meta = [{"idx": str(i), "patent_number": f"US{i}"} for i, r in enumerate(text)]
    inputs = tmp_path / "input.csv"; membership = tmp_path / "benchmark_sample_membership.csv"
    write(inputs, text); write(membership, list(reversed(meta)))
    return inputs, membership, text, meta


def test_join_uses_ids_instead_of_membership_row_order(tmp_path):
    inputs, membership, _, _ = fixture(tmp_path)
    joined = load_benchmark(inputs, membership)
    assert [(r["idx"], r["title"], r["patent_number"]) for r in joined] == [
        ("0", "first", "US0"), ("1", "second", "US1")]


def test_unified_input_preserves_the_split_input_records(tmp_path):
    inputs, membership, _, _ = fixture(tmp_path)
    expected = load_benchmark(inputs, membership)
    unified = tmp_path / "unified.csv"
    write(unified, expected)
    assert load_benchmark(unified) == expected


@pytest.mark.parametrize("problem", ["duplicate", "missing", "duplicate_input", "conflict"])
def test_join_rejects_ambiguous_or_misaligned_data(tmp_path, problem):
    inputs, membership, text, meta = fixture(tmp_path)
    if problem == "duplicate": write(membership, [meta[0], meta[0]])
    elif problem == "missing": write(membership, meta[:1])
    elif problem == "duplicate_input": write(inputs, [text[0], text[0]])
    else: write(membership, [{**r, "title": "inconsistent"} for r in meta])
    with pytest.raises(ValueError):
        load_benchmark(inputs, membership)
