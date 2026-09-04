import csv
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "build_template_threshold_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_template_threshold_audit",
    MODULE_PATH,
)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    sample_path = tmp_path / "input.csv"
    write_csv(
        sample_path,
        ["Issue", "Lowe_smiles"],
        [
            {"Issue": "rare", "Lowe_smiles": " R1>>P1 "},
            {"Issue": "other", "Lowe_smiles": "R2>>P2"},
        ],
    )

    population_csv = tmp_path / "nested" / "grants_issue.csv"
    write_csv(
        population_csv,
        [
            "PatentNumber",
            "ParagraphNum",
            "ReactionSmiles",
            "Template",
            "TemplateFreq",
        ],
        [
            {
                "PatentNumber": "US-1",
                "ParagraphNum": "10",
                "ReactionSmiles": "R1>>P1",
                "Template": "T1",
                "TemplateFreq": "1",
            },
            {
                "PatentNumber": "US-1",
                "ParagraphNum": "11",
                "ReactionSmiles": " R1>>P1 ",
                "Template": "T1",
                "TemplateFreq": "1.0",
            },
            {
                "PatentNumber": "US-2",
                "ParagraphNum": "20",
                "ReactionSmiles": "R2>>P2",
                "Template": "T2",
                "TemplateFreq": "9",
            },
            {
                "PatentNumber": "US-3",
                "ParagraphNum": "30",
                "ReactionSmiles": "unrelated",
                "Template": "",
                "TemplateFreq": "",
            },
        ],
    )
    population_zip = tmp_path / "grants_issue.zip"
    with zipfile.ZipFile(population_zip, "w") as archive:
        archive.write(population_csv, "nested/grants_issue.csv")
        archive.writestr("README.txt", "test archive")

    flags_path = tmp_path / "table3_flags.csv"
    write_csv(
        flags_path,
        [
            "source_row",
            "model",
            "ground_truth_valid",
            "noise_free",
            "template_failure",
            "identical_structures",
        ],
        [
            {
                "source_row": 1,
                "model": audit.MODEL_DEFAULT,
                "ground_truth_valid": "False",
                "noise_free": "0",
                "template_failure": "1",
                "identical_structures": "0",
            },
            {
                "source_row": 0,
                "model": audit.MODEL_DEFAULT,
                "ground_truth_valid": "True",
                "noise_free": "1",
                "template_failure": "0",
                "identical_structures": "0",
            },
            {
                "source_row": 0,
                "model": "another-model",
                "ground_truth_valid": "True",
                "noise_free": "1",
                "template_failure": "0",
                "identical_structures": "0",
            },
        ],
    )
    return sample_path, population_zip, flags_path


def test_run_builds_current_audit_schema_and_provenance(tmp_path):
    sample_path, population_path, flags_path = build_inputs(tmp_path)
    output_path = tmp_path / "audit.csv"
    linkage_path = tmp_path / "linkage.csv"
    summary_path = tmp_path / "summary.json"

    summary = audit.run(
        sample_path=sample_path,
        population_path=population_path,
        table3_flags_path=flags_path,
        output_path=output_path,
        linkage_output=linkage_path,
        summary_output=summary_path,
        expected_rows=2,
    )

    assert output_path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = read_csv(output_path)
    assert list(rows[0]) == list(audit.AUDIT_FIELDS)
    assert rows == [
        {
            "source_row": "0",
            "model": audit.MODEL_DEFAULT,
            "ground_truth_valid": "True",
            "noise_free": "1",
            "template_frequency": "1",
            "template_failure": "0",
            "identical_structures": "0",
            "rare_N=1": "1",
            "filtered_N=1": "1",
            "rare_N<5": "1",
            "filtered_N<5": "1",
            "rare_N<10": "1",
            "filtered_N<10": "1",
        },
        {
            "source_row": "1",
            "model": audit.MODEL_DEFAULT,
            "ground_truth_valid": "False",
            "noise_free": "0",
            "template_frequency": "9",
            "template_failure": "1",
            "identical_structures": "0",
            "rare_N=1": "0",
            "filtered_N=1": "1",
            "rare_N<5": "0",
            "filtered_N<5": "1",
            "rare_N<10": "1",
            "filtered_N<10": "1",
        },
    ]

    linkage = read_csv(linkage_path)
    assert linkage[0]["population_match_count"] == "2"
    assert linkage[0]["patent_number"] == "US-1"
    assert linkage[0]["reaction_sha256"] == audit.sha256_text("R1>>P1")
    assert summary["inputs"]["population"]["archive_member"] == (
        "nested/grants_issue.csv"
    )
    assert summary["inputs"]["population"]["matching_population_records"] == 3
    assert summary["thresholds"]["N=1"]["benchmark_frequency_flags"] == 1
    assert "\\" not in summary["inputs"]["population"]["path"]
    assert "\\" not in summary["outputs"]["audit"]["path"]
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_cli_defaults_use_released_population_and_emit_provenance():
    args = audit.build_parser().parse_args([])

    assert args.population == Path("evaluation/data/grants_issue.zip")
    assert args.linkage_output == Path(
        "evaluation/results/template_threshold_linkage.csv"
    )
    assert args.summary_output == Path(
        "evaluation/results/template_threshold_audit_summary.json"
    )


def test_conflicting_population_frequencies_are_rejected(tmp_path):
    sample = [
        audit.SampleReaction(
            source_row=0,
            reaction="R>>P",
            reaction_sha256=audit.sha256_text("R>>P"),
        )
    ]
    population_path = tmp_path / "grants_issue.csv"
    write_csv(
        population_path,
        ["ReactionSmiles", "Template", "TemplateFreq"],
        [
            {"ReactionSmiles": "R>>P", "Template": "T", "TemplateFreq": 2},
            {"ReactionSmiles": "R>>P", "Template": "T", "TemplateFreq": 3},
        ],
    )

    with pytest.raises(ValueError, match="conflicting TemplateFreq"):
        audit.scan_population(population_path, sample)


def test_missing_population_reaction_is_rejected(tmp_path):
    sample = [
        audit.SampleReaction(
            source_row=7,
            reaction="missing",
            reaction_sha256=audit.sha256_text("missing"),
        )
    ]
    population_path = tmp_path / "grants_issue.csv"
    write_csv(
        population_path,
        ["ReactionSmiles", "TemplateFreq"],
        [{"ReactionSmiles": "other", "TemplateFreq": 10}],
    )

    with pytest.raises(ValueError, match=r"source_row values: \[7\]"):
        audit.scan_population(population_path, sample)


def test_model_flag_coverage_and_boolean_consistency_are_strict(tmp_path):
    flags_path = tmp_path / "table3_flags.csv"
    write_csv(
        flags_path,
        [
            "source_row",
            "model",
            "ground_truth_valid",
            "noise_free",
            "template_failure",
            "identical_structures",
        ],
        [
            {
                "source_row": 0,
                "model": audit.MODEL_DEFAULT,
                "ground_truth_valid": "True",
                "noise_free": "0",
                "template_failure": "0",
                "identical_structures": "0",
            }
        ],
    )

    with pytest.raises(ValueError, match="ground_truth_valid and noise_free disagree"):
        audit.load_model_flags(
            flags_path,
            model=audit.MODEL_DEFAULT,
            expected_source_rows={0},
        )
