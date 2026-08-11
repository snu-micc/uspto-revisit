"""Human review workflow for unresolved reaction-SMILES compounds."""

from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from uspto_revisit.atom_mapping import add_atom_mapping_columns
from uspto_revisit.reaction_smiles import _coerce_mapping, replace_with_smiles
from uspto_revisit.smiles_fetch import canonicalize_smiles, normalize_name_key

REVIEW_DECISIONS = {
    "pending",
    "use_smiles",
    "keep_unresolved",
    "exclude_from_mapping",
}
EDITABLE_REVIEW_COLUMNS = (
    "review_decision",
    "reviewed_smiles",
    "evidence",
    "review_note",
)


@dataclass(frozen=True)
class ModelSpec:
    """Paths and output prefix for one model result."""

    prefix: str
    csv_path: Path
    smiles_dict_path: Path


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(table_path)
    return pd.read_csv(table_path)


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if table_path.suffix.lower() in {".xlsx", ".xls"}:
        frame.to_excel(table_path, index=False)
    else:
        frame.to_csv(table_path, index=False, encoding="utf-8-sig")


def discover_model_specs(result_dir: str | Path = "result") -> list[ModelSpec]:
    """Discover complete final model outputs and their SMILES dictionaries."""
    result_path = Path(result_dir)
    suffix = "_reaction_smiles_final.csv"
    specs = []
    for csv_path in sorted(result_path.glob(f"*{suffix}")):
        prefix = csv_path.name[: -len(suffix)]
        dictionary = (
            result_path
            / "smiles_batches"
            / prefix
            / "smiles_dict_final.json"
        )
        if dictionary.is_file():
            specs.append(ModelSpec(prefix, csv_path, dictionary))
    if not specs:
        raise FileNotFoundError(
            f"No complete *_reaction_smiles_final.csv outputs with matching "
            f"SMILES dictionaries were found under {result_path}."
        )
    return specs


def model_specs_from_values(
    values: list[list[str]] | None,
    result_dir: str | Path = "result",
) -> list[ModelSpec]:
    if not values:
        return discover_model_specs(result_dir)
    return [
        ModelSpec(prefix, Path(csv_path), Path(dictionary))
        for prefix, csv_path, dictionary in values
    ]


def parse_reaction_list(value) -> list[str]:
    if value is None:
        return []
    parsed = value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return []


def remove_codes_from_skeleton(reaction_code: str, excluded_codes: set[str]) -> str:
    """Remove confirmed non-mapping components while preserving reaction roles."""
    return ">".join(
        ".".join(
            code for code in part.split(".") if code and code not in excluded_codes
        )
        for part in reaction_code.split(">")
    )


def restore_left_codes_in_skeleton(reaction_code: str, approved_codes: set[str]) -> str:
    """Restore approved left-side labels omitted when they were originally NoSmi."""
    if not approved_codes:
        return reaction_code
    parts = reaction_code.split(">")
    present = {code for part in parts for code in part.split(".") if code}
    missing = sorted(approved_codes - present)
    if not missing:
        return reaction_code
    parts[0] = ".".join([code for code in [parts[0], *missing] if code])
    return ">".join(parts)


def approved_left_codes_for_spec(review: pd.DataFrame, prefix: str) -> dict[int, set[str]]:
    """Find approved codes that the review queue identified as left-side components."""
    codes_by_row: dict[int, set[str]] = defaultdict(set)
    for _, row in review.fillna("").iterrows():
        if _clean_cell(row.get("review_decision", "")).lower() != "use_smiles":
            continue
        if "left" not in _clean_cell(row.get("sides", "")).lower():
            continue
        for item in _clean_cell(row.get("model_codes", "")).split(";"):
            model, separator, codes = item.strip().partition(":")
            if separator and model == prefix:
                codes_by_row[int(row["row_zero_based"])].update(
                    code for code in codes.split("|") if code
                )
    return codes_by_row


def extract_no_smi_name(value) -> str | None:
    text = str(value).strip()
    suffix = " (NoSmi)]"
    if not (text.startswith("[") and text.endswith(suffix)):
        return None
    return " ".join(text[1 : -len(suffix)].split())


def _load_model(spec: ModelSpec) -> tuple[pd.DataFrame, list[dict]]:
    frame = read_table(spec.csv_path)
    with spec.smiles_dict_path.open("r", encoding="utf-8-sig") as handle:
        dictionaries = json.load(handle)
    if len(frame) != len(dictionaries):
        raise ValueError(
            f"{spec.prefix}: CSV rows ({len(frame)}) and SMILES dictionaries "
            f"({len(dictionaries)}) differ."
        )
    for column in (f"{spec.prefix}_skeleton", "prediction"):
        if column not in frame.columns:
            raise ValueError(f"{spec.csv_path} does not contain required column {column!r}.")
    return frame, dictionaries


def _audit_by_name(audit_path: str | Path | None) -> dict[str, dict]:
    if not audit_path or not Path(audit_path).is_file():
        return {}
    audit = read_table(audit_path).fillna("")
    return {
        str(row["normalized_name"]): row.to_dict()
        for _, row in audit.iterrows()
        if str(row.get("normalized_name", "")).strip()
    }


def _join_counter(counter: Counter) -> str:
    return "; ".join(f"{key}:{value}" for key, value in sorted(counter.items()))


def build_review_queue(
    specs: list[ModelSpec],
    audit_path: str | Path | None = "result/nosmi_recovery_audit.csv",
    existing_review_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build one review row per source-row and normalized compound name."""
    audit = _audit_by_name(audit_path)
    records: dict[tuple[int, str], dict] = {}

    for spec in specs:
        frame, dictionaries = _load_model(spec)
        skeleton_column = f"{spec.prefix}_skeleton"
        for row_index, row in frame.iterrows():
            smiles_dict = _coerce_mapping(
                dictionaries[row_index],
                row_index,
                f"{spec.prefix}_smiles",
            )
            for reaction in parse_reaction_list(row[skeleton_column]):
                parts = reaction.split(">")
                for side_index, part in enumerate(parts):
                    side = (
                        "left"
                        if side_index == 0
                        else "product"
                        if side_index == len(parts) - 1
                        else "middle"
                    )
                    for code in part.split("."):
                        name = extract_no_smi_name(smiles_dict.get(code, ""))
                        if not name:
                            continue
                        normalized = normalize_name_key(name)
                        key = (int(row_index), normalized)
                        record = records.setdefault(
                            key,
                            {
                                "names": Counter(),
                                "models": Counter(),
                                "model_codes": defaultdict(set),
                                "sides": Counter(),
                                "title": str(row.get("title", "")),
                                "paragraph": str(row.get("paragraph", "")),
                            },
                        )
                        record["names"][name] += 1
                        record["models"][spec.prefix] += 1
                        record["model_codes"][spec.prefix].add(code)
                        record["sides"][side] += 1

    rows = []
    for (row_index, normalized), record in records.items():
        audit_row = audit.get(normalized, {})
        display_name = record["names"].most_common(1)[0][0]
        rows.append(
            {
                "review_id": f"row-{row_index}__{normalized}",
                "row_zero_based": row_index,
                "name": display_name,
                "normalized_name": normalized,
                "currently_unresolved": True,
                "automatic_status": audit_row.get("status", ""),
                "occurrences": sum(record["models"].values()),
                "models": _join_counter(record["models"]),
                "model_codes": "; ".join(
                    f"{model}:{'|'.join(sorted(codes))}"
                    for model, codes in sorted(record["model_codes"].items())
                ),
                "sides": _join_counter(record["sides"]),
                "candidate_smiles": audit_row.get("resolved_smiles", ""),
                "resolver_source": audit_row.get("resolver_source", ""),
                "title": record["title"],
                "paragraph": record["paragraph"],
                "review_decision": "pending",
                "reviewed_smiles": "",
                "evidence": "",
                "review_note": "",
            }
        )

    current = {row["review_id"]: row for row in rows}
    if existing_review_path and Path(existing_review_path).is_file():
        existing = read_table(existing_review_path).fillna("")
        if "review_id" not in existing.columns:
            raise ValueError("Existing review file does not contain a review_id column.")
        for _, old_row in existing.iterrows():
            review_id = str(old_row["review_id"])
            if review_id in current:
                for column in EDITABLE_REVIEW_COLUMNS:
                    if column in existing.columns:
                        current[review_id][column] = old_row[column]
            elif str(old_row.get("review_decision", "")).strip().lower() != "pending":
                preserved = old_row.to_dict()
                preserved["currently_unresolved"] = False
                preserved["occurrences"] = 0
                current[review_id] = preserved

    columns = [
        "review_id",
        "row_zero_based",
        "name",
        "normalized_name",
        "currently_unresolved",
        "automatic_status",
        "occurrences",
        "models",
        "model_codes",
        "sides",
        "candidate_smiles",
        "resolver_source",
        "title",
        "paragraph",
        *EDITABLE_REVIEW_COLUMNS,
    ]
    review = pd.DataFrame(current.values(), columns=columns)
    if review.empty:
        return review
    return review.sort_values(
        ["currently_unresolved", "occurrences", "row_zero_based", "name"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def _clean_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def validate_review_rows(
    review: pd.DataFrame,
) -> tuple[dict[tuple[int, str], str], set[tuple[int, str]], dict]:
    """Validate decisions and return structures and mapping exclusions."""
    required = {"row_zero_based", "name", "normalized_name", "review_decision"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"Review file is missing columns: {', '.join(sorted(missing))}")

    approvals: dict[tuple[int, str], str] = {}
    exclusions: set[tuple[int, str]] = set()
    counts = Counter()
    errors = []
    for index, row in review.fillna("").iterrows():
        decision = _clean_cell(row.get("review_decision", "")).lower() or "pending"
        if decision not in REVIEW_DECISIONS:
            errors.append(
                f"review row {index}: decision {decision!r} must be one of "
                f"{', '.join(sorted(REVIEW_DECISIONS))}"
            )
            continue
        counts[decision] += 1
        if decision == "pending":
            continue

        key = (int(row["row_zero_based"]), _clean_cell(row["normalized_name"]))
        if decision in {"keep_unresolved", "exclude_from_mapping"}:
            if decision == "exclude_from_mapping":
                exclusions.add(key)
            continue

        smiles = canonicalize_smiles(_clean_cell(row.get("reviewed_smiles", "")))
        if not smiles:
            errors.append(f"review row {index}: reviewed_smiles is not valid SMILES")
            continue
        previous = approvals.get(key)
        if previous and previous != smiles:
            errors.append(
                f"review row {index}: conflicting approved SMILES for row/name {key}"
            )
        approvals[key] = smiles

    if errors:
        raise ValueError("Human review validation failed:\n- " + "\n- ".join(errors))
    return approvals, exclusions, dict(counts)


def _clear_stale_mapping(frame: pd.DataFrame, prefix: str, changed_rows: set[int]) -> None:
    if not changed_rows:
        return
    mapping_columns = [
        f"{prefix}_localmapper_rxn",
        f"{prefix}_mapped_rxn",
        f"{prefix}_mapping_template",
        f"{prefix}_mapping_confident",
        f"{prefix}_mapping_error",
    ]
    for column in mapping_columns:
        if column not in frame.columns:
            continue
        frame.loc[list(changed_rows), column] = None
    error_column = f"{prefix}_mapping_error"
    if error_column in frame.columns:
        frame.loc[list(changed_rows), error_column] = (
            "Human-reviewed SMILES applied; rerun atom mapping."
        )


def _write_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )


def apply_review(
    specs: list[ModelSpec],
    review_path: str | Path,
    output_dir: str | Path = "result",
    summary_path: str | Path = "result/nosmi_human_review_summary.json",
    map_atoms: bool = False,
    mapping_device: str = "cpu",
    mapping_model_version: str = "202403",
    mapping_batch_size: int = 32,
) -> dict:
    """Apply row-scoped human approvals and rebuild every model output."""
    review = read_table(review_path)
    approvals, exclusions, decision_counts = validate_review_rows(review)
    output_path = Path(output_dir)
    model_summaries = {}
    matched_decisions = set()

    for spec in specs:
        frame, dictionaries = _load_model(spec)
        approved_left_codes = approved_left_codes_for_spec(review, spec.prefix)
        changed_rows = set()
        replacements = 0
        excluded_codes_by_row: dict[int, set[str]] = defaultdict(set)
        for row_index, smiles_dict in enumerate(dictionaries):
            for code, value in list(smiles_dict.items()):
                name = extract_no_smi_name(value)
                if not name:
                    continue
                key = (row_index, normalize_name_key(name))
                if key in approvals:
                    smiles_dict[code] = approvals[key]
                    changed_rows.add(row_index)
                    replacements += 1
                    matched_decisions.add(key)
                if key in exclusions:
                    excluded_codes_by_row[row_index].add(code)
                    changed_rows.add(row_index)
                    matched_decisions.add(key)

        result = frame.copy()
        result[f"{spec.prefix}_smiles"] = dictionaries
        if changed_rows:
            for row_index in sorted(changed_rows):
                row_skeleton = [
                    restore_left_codes_in_skeleton(
                        remove_codes_from_skeleton(reaction, excluded_codes_by_row[row_index]),
                        approved_left_codes.get(row_index, set()),
                    )
                    for reaction in parse_reaction_list(
                        result.at[row_index, f"{spec.prefix}_skeleton"]
                    )
                ]
                row_reactions = [
                    replace_with_smiles(reaction, dictionaries[row_index])
                    for reaction in row_skeleton
                ]
                result.at[row_index, f"{spec.prefix}_skeleton"] = row_skeleton
                result.at[row_index, f"{spec.prefix}_rxn"] = row_reactions
        if map_atoms and changed_rows:
            result = add_atom_mapping_columns(
                result,
                f"{spec.prefix}_rxn",
                spec.prefix,
                device=mapping_device,
                model_version=mapping_model_version,
                batch_size=mapping_batch_size,
            )
        else:
            _clear_stale_mapping(result, spec.prefix, changed_rows)

        csv_output = output_path / f"{spec.prefix}_reaction_smiles_final.csv"
        dictionary_output = (
            output_path
            / "smiles_batches"
            / spec.prefix
            / "smiles_dict_final.json"
        )
        write_table(result, csv_output)
        _write_json(dictionaries, dictionary_output)
        model_summaries[spec.prefix] = {
            "replacements": replacements,
            "excluded_components": sum(
                len(codes) for codes in excluded_codes_by_row.values()
            ),
            "changed_rows": len(changed_rows),
            "csv": str(csv_output),
            "smiles_dict": str(dictionary_output),
        }

    unmatched = sorted(
        f"row-{row_index}__{normalized}"
        for row_index, normalized in approvals.keys() | exclusions
        if (row_index, normalized) not in matched_decisions
    )
    summary = {
        "review": str(review_path),
        "decisions": decision_counts,
        "approved_row_names": len(approvals),
        "excluded_row_names": len(exclusions),
        "matched_reviewed_row_names": len(matched_decisions),
        "unmatched_reviewed_row_names": unmatched,
        "atom_mapping_rerun": map_atoms,
        "models": model_summaries,
    }
    _write_json(summary, Path(summary_path))
    return summary
