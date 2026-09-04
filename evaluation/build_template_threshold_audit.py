"""Build the 400-row template-frequency threshold audit from grants_issue.

This script supplies the missing provenance step between the released
``grants_issue.csv`` (or an archive containing it), the 400-row benchmark, and
the row-level Table 3 decisions.  It deliberately uses the stored
``TemplateFreq`` values; it does not claim to regenerate those values from the
full USPTO corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator, TextIO


MODEL_DEFAULT = "gemini-3.6-flash-high"
THRESHOLDS = (("N=1", 1), ("N<5", 5), ("N<10", 10))
AUDIT_FIELDS = (
    "source_row",
    "model",
    "ground_truth_valid",
    "noise_free",
    "template_frequency",
    "template_failure",
    "identical_structures",
    "rare_N=1",
    "filtered_N=1",
    "rare_N<5",
    "filtered_N<5",
    "rare_N<10",
    "filtered_N<10",
)
LINKAGE_FIELDS = (
    "source_row",
    "match_method",
    "reaction_sha256",
    "template_frequency",
    "population_match_count",
    "first_population_record",
    "patent_number",
    "paragraph_number",
    "template_sha256",
)


@dataclass(frozen=True)
class SampleReaction:
    source_row: int
    reaction: str
    reaction_sha256: str


@dataclass
class PopulationMatch:
    frequency: int
    count: int
    first_record: int
    patent_number: str
    paragraph_number: str
    template_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_reaction(value: object) -> str:
    """Normalize only field-edge whitespace; do not change chemical syntax."""
    return str(value).strip()


def parse_bool(value: object, *, field: str, context: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "1.0", "true", "yes"}:
        return True
    if normalized in {"0", "0.0", "false", "no"}:
        return False
    raise ValueError(f"{context}: {field} must be Boolean, found {value!r}.")


def parse_integer(
    value: object,
    *,
    field: str,
    context: str,
    minimum: int | None = None,
) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{context}: {field} is blank.")
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(
            f"{context}: {field} must be an integer, found {value!r}."
        ) from error
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(
            f"{context}: {field} must be an integer, found {value!r}."
        )
    integer = int(number)
    if minimum is not None and integer < minimum:
        raise ValueError(
            f"{context}: {field} must be >= {minimum}, found {integer}."
        )
    return integer


def require_columns(
    fieldnames: list[str] | None,
    required: set[str],
    *,
    context: str,
) -> None:
    available = set(fieldnames or [])
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}.")


def load_sample(path: Path, expected_rows: int) -> list[SampleReaction]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, {"Lowe_smiles"}, context=str(path))
        rows = list(reader)

    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} benchmark rows in {path}, found {len(rows)}."
        )

    sample: list[SampleReaction] = []
    for source_row, row in enumerate(rows):
        reaction = normalize_reaction(row["Lowe_smiles"])
        if not reaction:
            raise ValueError(f"{path}: Lowe_smiles is blank at source_row {source_row}.")
        sample.append(
            SampleReaction(
                source_row=source_row,
                reaction=reaction,
                reaction_sha256=sha256_text(reaction),
            )
        )
    return sample


def _select_zip_member(
    archive: zipfile.ZipFile,
    requested_member: str | None,
) -> zipfile.ZipInfo:
    files = [item for item in archive.infolist() if not item.is_dir()]
    if requested_member is not None:
        matches = [item for item in files if item.filename == requested_member]
        if len(matches) != 1:
            raise ValueError(
                f"ZIP member {requested_member!r} was not found exactly once."
            )
        selected = matches[0]
        if Path(selected.filename).suffix.lower() != ".csv":
            raise ValueError(f"ZIP member {requested_member!r} is not a CSV file.")
        return selected

    csv_members = [item for item in files if Path(item.filename).suffix.lower() == ".csv"]
    named = [
        item
        for item in csv_members
        if Path(item.filename).name.lower() == "grants_issue.csv"
    ]
    if len(named) == 1:
        return named[0]
    if len(csv_members) == 1:
        return csv_members[0]
    choices = [item.filename for item in csv_members]
    raise ValueError(
        "The population ZIP must contain one CSV or one uniquely named "
        f"grants_issue.csv; found {choices}. Use --population-member."
    )


@contextmanager
def open_population_csv(
    path: Path,
    member: str | None = None,
) -> Iterator[tuple[TextIO, dict[str, object]]]:
    if path.suffix.lower() != ".zip":
        if member is not None:
            raise ValueError("--population-member can only be used with a ZIP input.")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            yield handle, {"format": "csv", "archive_member": None}
        return

    with zipfile.ZipFile(path) as archive:
        selected = _select_zip_member(archive, member)
        with archive.open(selected) as binary:
            import io

            with io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as handle:
                yield handle, {
                    "format": "zip",
                    "archive_member": selected.filename,
                    "archive_member_crc32": f"{selected.CRC:08x}",
                    "archive_member_size": selected.file_size,
                }


def frequency_is_rare(frequency: int, label: str, cutoff: int) -> bool:
    return frequency == 1 if label == "N=1" else 0 < frequency < cutoff


def scan_population(
    path: Path,
    sample: list[SampleReaction],
    *,
    member: str | None = None,
) -> tuple[dict[str, PopulationMatch], dict[str, object]]:
    targets = {item.reaction for item in sample}
    matches: dict[str, PopulationMatch] = {}
    population_rows = 0
    blank_frequency_rows = 0
    threshold_counts = {label: 0 for label, _cutoff in THRESHOLDS}

    with open_population_csv(path, member) as (handle, input_metadata):
        reader = csv.DictReader(handle)
        require_columns(
            reader.fieldnames,
            {"ReactionSmiles", "TemplateFreq"},
            context=str(path),
        )
        for record_index, row in enumerate(reader, start=1):
            population_rows += 1
            frequency_text = str(row.get("TemplateFreq", "")).strip()
            frequency: int | None
            if frequency_text:
                frequency = parse_integer(
                    frequency_text,
                    field="TemplateFreq",
                    context=f"{path}, population record {record_index}",
                    minimum=1,
                )
                for label, cutoff in THRESHOLDS:
                    if frequency_is_rare(frequency, label, cutoff):
                        threshold_counts[label] += 1
            else:
                frequency = None
                blank_frequency_rows += 1

            reaction = normalize_reaction(row.get("ReactionSmiles", ""))
            if reaction not in targets:
                continue
            if frequency is None:
                raise ValueError(
                    f"{path}, population record {record_index}: a benchmark "
                    "reaction has blank TemplateFreq."
                )

            template = str(row.get("Template", "")).strip()
            template_hash = sha256_text(template) if template else ""
            existing = matches.get(reaction)
            if existing is None:
                matches[reaction] = PopulationMatch(
                    frequency=frequency,
                    count=1,
                    first_record=record_index,
                    patent_number=str(row.get("PatentNumber", "")).strip(),
                    paragraph_number=str(row.get("ParagraphNum", "")).strip(),
                    template_sha256=template_hash,
                )
                continue

            if existing.frequency != frequency:
                raise ValueError(
                    "A benchmark ReactionSmiles has conflicting TemplateFreq "
                    f"values {existing.frequency} and {frequency} at population "
                    f"record {record_index}; reaction SHA-256={sha256_text(reaction)}."
                )
            if (
                existing.template_sha256
                and template_hash
                and existing.template_sha256 != template_hash
            ):
                raise ValueError(
                    "A benchmark ReactionSmiles has conflicting Template values at "
                    f"population record {record_index}; reaction "
                    f"SHA-256={sha256_text(reaction)}."
                )
            existing.count += 1

    missing_rows = [item.source_row for item in sample if item.reaction not in matches]
    if missing_rows:
        raise ValueError(
            "No population match was found for benchmark source_row values: "
            f"{missing_rows}."
        )

    metadata = {
        **input_metadata,
        "rows_scanned": population_rows,
        "rows_with_blank_template_frequency": blank_frequency_rows,
        "unique_benchmark_reactions": len(targets),
        "matched_unique_reactions": len(matches),
        "matching_population_records": sum(match.count for match in matches.values()),
        "frequency_flag_counts": threshold_counts,
    }
    return matches, metadata


def load_model_flags(
    path: Path,
    *,
    model: str,
    expected_source_rows: set[int],
) -> dict[int, dict[str, object]]:
    required = {
        "source_row",
        "model",
        "ground_truth_valid",
        "noise_free",
        "template_failure",
        "identical_structures",
    }
    selected: dict[int, dict[str, object]] = {}
    available_models: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, required, context=str(path))
        for record_index, row in enumerate(reader, start=1):
            row_model = str(row["model"]).strip()
            if row_model:
                available_models.add(row_model)
            if row_model != model:
                continue
            source_row = parse_integer(
                row["source_row"],
                field="source_row",
                context=f"{path}, record {record_index}",
                minimum=0,
            )
            if source_row in selected:
                raise ValueError(
                    f"{path}: duplicate {model!r} flag row for source_row {source_row}."
                )
            ground_truth_valid = parse_bool(
                row["ground_truth_valid"],
                field="ground_truth_valid",
                context=f"{path}, source_row {source_row}",
            )
            noise_free = parse_bool(
                row["noise_free"],
                field="noise_free",
                context=f"{path}, source_row {source_row}",
            )
            if ground_truth_valid != noise_free:
                raise ValueError(
                    f"{path}, source_row {source_row}: ground_truth_valid and "
                    "noise_free disagree."
                )
            selected[source_row] = {
                "ground_truth_valid": ground_truth_valid,
                "noise_free": noise_free,
                "template_failure": parse_bool(
                    row["template_failure"],
                    field="template_failure",
                    context=f"{path}, source_row {source_row}",
                ),
                "identical_structures": parse_bool(
                    row["identical_structures"],
                    field="identical_structures",
                    context=f"{path}, source_row {source_row}",
                ),
            }

    if not selected:
        raise ValueError(
            f"Model {model!r} was not found in {path}; available models: "
            f"{sorted(available_models)}."
        )
    found_source_rows = set(selected)
    if found_source_rows != expected_source_rows:
        missing = sorted(expected_source_rows - found_source_rows)
        unexpected = sorted(found_source_rows - expected_source_rows)
        raise ValueError(
            f"{path}: {model!r} source_row coverage is incomplete; "
            f"missing={missing}, unexpected={unexpected}."
        )
    return selected


def build_audit_rows(
    sample: list[SampleReaction],
    matches: dict[str, PopulationMatch],
    flags: dict[int, dict[str, object]],
    *,
    model: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    audit_rows: list[dict[str, object]] = []
    linkage_rows: list[dict[str, object]] = []
    for item in sample:
        match = matches[item.reaction]
        flag = flags[item.source_row]
        template_failure = bool(flag["template_failure"])
        identical = bool(flag["identical_structures"])
        audit: dict[str, object] = {
            "source_row": item.source_row,
            "model": model,
            "ground_truth_valid": bool(flag["ground_truth_valid"]),
            "noise_free": int(bool(flag["noise_free"])),
            "template_frequency": match.frequency,
            "template_failure": int(template_failure),
            "identical_structures": int(identical),
        }
        for label, cutoff in THRESHOLDS:
            rare = frequency_is_rare(match.frequency, label, cutoff)
            audit[f"rare_{label}"] = int(rare)
            audit[f"filtered_{label}"] = int(
                rare or template_failure or identical
            )
        audit_rows.append(audit)
        linkage_rows.append(
            {
                "source_row": item.source_row,
                "match_method": "trimmed_exact_ReactionSmiles",
                "reaction_sha256": item.reaction_sha256,
                "template_frequency": match.frequency,
                "population_match_count": match.count,
                "first_population_record": match.first_record,
                "patent_number": match.patent_number,
                "paragraph_number": match.paragraph_number,
                "template_sha256": match.template_sha256,
            }
        )
    return audit_rows, linkage_rows


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    sample_path: Path,
    population_path: Path,
    table3_flags_path: Path,
    output_path: Path,
    linkage_output: Path | None,
    sample: list[SampleReaction],
    audit_rows: list[dict[str, object]],
    population_metadata: dict[str, object],
    model: str,
) -> dict[str, object]:
    threshold_summary = {}
    for label, _cutoff in THRESHOLDS:
        threshold_summary[label] = {
            "benchmark_frequency_flags": sum(
                int(row[f"rare_{label}"]) for row in audit_rows
            ),
            "benchmark_union_filtered": sum(
                int(row[f"filtered_{label}"]) for row in audit_rows
            ),
        }
    summary: dict[str, object] = {
        "schema_version": 1,
        "scope": (
            "Links stored grants_issue TemplateFreq values to the benchmark; "
            "does not regenerate frequencies from the full USPTO corpus."
        ),
        "model": model,
        "inputs": {
            "sample": {
                "path": sample_path.as_posix(),
                "sha256": sha256_file(sample_path),
                "rows": len(sample),
            },
            "population": {
                "path": population_path.as_posix(),
                "sha256": sha256_file(population_path),
                **population_metadata,
            },
            "table3_flags": {
                "path": table3_flags_path.as_posix(),
                "sha256": sha256_file(table3_flags_path),
                "selected_model_rows": len(audit_rows),
            },
        },
        "outputs": {
            "audit": {
                "path": output_path.as_posix(),
                "sha256": sha256_file(output_path),
                "rows": len(audit_rows),
            }
        },
        "thresholds": threshold_summary,
    }
    if linkage_output is not None:
        outputs = summary["outputs"]
        assert isinstance(outputs, dict)
        outputs["linkage"] = {
            "path": linkage_output.as_posix(),
            "sha256": sha256_file(linkage_output),
            "rows": len(audit_rows),
        }
    return summary


def run(
    *,
    sample_path: Path,
    population_path: Path,
    table3_flags_path: Path,
    output_path: Path,
    model: str = MODEL_DEFAULT,
    population_member: str | None = None,
    linkage_output: Path | None = None,
    summary_output: Path | None = None,
    expected_rows: int = 400,
) -> dict[str, object]:
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive.")
    sample = load_sample(sample_path, expected_rows)
    matches, population_metadata = scan_population(
        population_path,
        sample,
        member=population_member,
    )
    flags = load_model_flags(
        table3_flags_path,
        model=model,
        expected_source_rows={item.source_row for item in sample},
    )
    audit_rows, linkage_rows = build_audit_rows(
        sample,
        matches,
        flags,
        model=model,
    )
    write_csv(output_path, audit_rows, AUDIT_FIELDS)
    if linkage_output is not None:
        write_csv(linkage_output, linkage_rows, LINKAGE_FIELDS)
    summary = build_summary(
        sample_path=sample_path,
        population_path=population_path,
        table3_flags_path=table3_flags_path,
        output_path=output_path,
        linkage_output=linkage_output,
        sample=sample,
        audit_rows=audit_rows,
        population_metadata=population_metadata,
        model=model,
    )
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-input", type=Path, default=Path("examples/input.csv"))
    parser.add_argument(
        "--population",
        type=Path,
        default=Path("evaluation/data/grants_issue.zip"),
        help="grants_issue.csv or a ZIP archive containing it.",
    )
    parser.add_argument(
        "--population-member",
        help="Exact CSV member name when a ZIP contains multiple CSV files.",
    )
    parser.add_argument(
        "--table3-flags",
        type=Path,
        default=Path("evaluation/results/table3/table3_flags.csv"),
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/data/template_threshold_audit.csv"),
    )
    parser.add_argument(
        "--linkage-output",
        type=Path,
        default=Path("evaluation/results/template_threshold_linkage.csv"),
        help="Row-level provenance CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("evaluation/results/template_threshold_audit_summary.json"),
        help="JSON containing input/output hashes and validation counts.",
    )
    parser.add_argument("--expected-rows", type=int, default=400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(
        sample_path=args.sample_input,
        population_path=args.population,
        table3_flags_path=args.table3_flags,
        output_path=args.output,
        model=args.model,
        population_member=args.population_member,
        linkage_output=args.linkage_output,
        summary_output=args.summary_output,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
