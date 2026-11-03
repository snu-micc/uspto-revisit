"""The AutoTemplate comparison (SI Section S16, Tables S31-S32) re-scores exactly from the released files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks" / "autotemplate"
SOURCE_MAPPING = BASE / "source_mapping"
RXNMAPPER = BASE / "rxnmapper"
RESCORED = ("comparison_metrics.csv", "component_metrics.csv", "paired_tests.csv", "retention_metrics.csv",
            "RESULTS.md")
# The released files were scored with the RDKit of requirements-lock.txt. Other versions can write the
# predicted SMILES in comparison_flags.csv differently without changing any decision.
PINNED_RDKIT = "2025.09.3"
# Exact McNemar p-values can differ in the last digit with other SciPy versions or builds than the conda build
# of SciPy 1.15.2 that wrote the released files (e.g. SciPy 1.18.1, or the Windows PyPI wheel of 1.15.2);
# counts must still match exactly.
BUILD_DEPENDENT = ("paired_tests.csv",)
# Hashes of repository files depend on the line endings of the checkout, and the source-mapped run was
# scored with an earlier evaluate_table2.py; every other summary field must be reproduced.
CHECKOUT_DEPENDENT = ("source_gt_sha256", "scorer_sha256")

for _path in (SOURCE_MAPPING, BASE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import baseline_flags  # noqa: E402


def _module(path: Path):
    spec = importlib.util.spec_from_file_location(f"autotemplate_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def baseline():
    return baseline_flags.rebuild()


def _counts(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    metrics = pd.read_csv(path)
    return {(r.endpoint, r.method): (int(r.correct), int(r.denominator)) for r in metrics.itertuples()}


def _assert_reproduced(released: Path, rescored: Path) -> None:
    for name in RESCORED:
        if name in BUILD_DEPENDENT:
            pd.testing.assert_frame_equal(pd.read_csv(rescored / name), pd.read_csv(released / name),
                                          check_exact=False, rtol=1e-12, atol=0)
            continue
        assert (rescored / name).read_bytes() == (released / name).read_bytes(), name
    if rdBase.rdkitVersion == PINNED_RDKIT:
        assert (rescored / "comparison_flags.csv").read_bytes() == (released / "comparison_flags.csv").read_bytes()
    else:
        new_flags, old_flags = (pd.read_csv(d / "comparison_flags.csv", dtype=str, keep_default_na=False)
                                .drop(columns="predicted_signature") for d in (rescored, released))
        pd.testing.assert_frame_equal(new_flags, old_flags)
    new = json.loads((rescored / "comparison_summary.json").read_text(encoding="utf-8"))
    old = json.loads((released / "comparison_summary.json").read_text(encoding="utf-8"))
    assert set(new["scorer_sha256"]) == set(old["scorer_sha256"]) == {"analysis/evaluate_table2.py",
                                                                      "analysis/evaluate_table3.py"}
    for key in CHECKOUT_DEPENDENT:
        new.pop(key), old.pop(key)
    assert new == old


def test_vendored_core_is_byte_identical_to_upstream():
    manifest = json.loads((BASE / "vendor" / "upstream_manifest.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == "6b17ea08cc3dded90a9eee8e25b74e8a60e4cb94"
    assert len(manifest["files"]) == 24
    assert "README.md" not in {entry["path"] for entry in manifest["files"]}
    for entry in manifest["files"]:
        data = (BASE / "vendor" / entry["path"]).read_bytes()
        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == entry["git_blob_sha1"], entry["path"]


def test_rxnmapper_run_rescores_exactly(tmp_path, baseline):
    scorer = _module(RXNMAPPER / "score_rxnmapper.py")
    assert scorer.main(["--output-dir", str(tmp_path)], baseline=baseline) == 0
    _assert_reproduced(RXNMAPPER, tmp_path)
    counts = _counts(tmp_path / "comparison_metrics.csv")
    at = "AutoTemplate_RXNMapper"
    assert counts["primary_structure", at] == (50, 292)
    assert counts["strict_structure", at] == (46, 292)
    assert counts["primary_end_to_end", at] == (94, 400)
    assert counts["denoising", at] == (257, 400)
    assert counts["primary_end_to_end", at + "_no_change_as_failure"] == (89, 400)
    assert counts["denoising", at + "_no_change_as_failure"] == (252, 400)
    assert counts["primary_structure", "GPT_5.4_none"] == (209, 292)
    assert counts["primary_structure", "Original_USPTO"] == (99, 292)
    tests = pd.read_csv(tmp_path / "paired_tests.csv").set_index(["endpoint", "comparator"])
    assert tuple(tests.loc[("primary_structure", "Original_USPTO"),
                           ["autotemplate_rxnmapper_only_correct", "comparator_only_correct"]]) == (6, 55)
    assert tuple(tests.loc[("primary_end_to_end", "GPT_5.4_none"),
                           ["autotemplate_rxnmapper_only_correct", "comparator_only_correct"]]) == (34, 180)


def test_source_mapping_run_rescores_exactly(tmp_path, baseline):
    scorer = _module(SOURCE_MAPPING / "score_comparison.py")
    assert scorer.main(["--output-dir", str(tmp_path)], baseline=baseline) == 0
    _assert_reproduced(SOURCE_MAPPING, tmp_path)
    counts = _counts(tmp_path / "comparison_metrics.csv")
    at = "AutoTemplate_source_mapping"
    assert counts["primary_structure", at] == (45, 292)
    assert counts["strict_structure", at] == (43, 292)
    assert counts["primary_end_to_end", at] == (64, 400)
    assert counts["denoising", at] == (218, 400)
    assert counts["primary_structure", "GPT_5.4_none"] == (209, 292)
    assert counts["primary_structure", "Original_USPTO"] == (99, 292)
