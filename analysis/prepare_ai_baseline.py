"""Rebuild the current Ai et al. component baseline offline.

The 400 stored predictions are aligned to the current benchmark. ORD roles are
parsed strictly; names are resolved using the recorded automatic, table-free
two-pass resolver. Malformed predictions and unresolved names remain failures.
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
from rdkit import RDLogger
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / 'src', PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from uspto_revisit.smiles_fetch import canonicalize_smiles, normalize_name_key
RDLogger.DisableLog('rdApp.*')
AI_METHOD = 'ai_without_title'
RELEASE_DIR = Path('benchmarks/ai_et_al/preprocessing')
DEFAULT_OUTPUT_DIR = Path('analysis/results/ai_baseline')
DEFAULT_REQUERY_LOG = Path('result/name_resolution/requery_2026-10-27_ai_baseline_raw.jsonl')
CURRENT_INPUT = Path('analysis/data/input.csv')
COMPONENT_FIELDS = ('reactants', 'reagents', 'products')
ROLE_TO_FIELD = {'REACTANT': 'reactants', 'SOLVENT': 'reagents', 'CATALYST': 'reagents', 'REAGENT': 'reagents', 'PRODUCT': 'products'}
TRAILING_MARKER = re.compile(r'\s*#{3,}\s*$')


def clean_prediction(value: Any) -> str:
    """Remove transport wrappers without repairing model-generated JSON."""
    text = str(value or "").strip()
    text = TRAILING_MARKER.sub("", text).strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_ord_prediction(value: Any) -> tuple[dict[str, Any] | None, str]:
    """Parse only valid JSON objects; malformed generations remain failures."""
    current: Any = clean_prediction(value)
    for _ in range(3):
        if isinstance(current, dict):
            return current, ""
        if not isinstance(current, str):
            break
        try:
            current = json.loads(current)
        except (TypeError, ValueError) as exc:
            return None, f"{type(exc).__name__}: {exc}"
    if isinstance(current, dict):
        return current, ""
    return None, "Prediction did not decode to a JSON object."


def component_names(component: Any) -> list[str]:
    if not isinstance(component, dict):
        return []
    identifiers = component.get("identifiers", [])
    if isinstance(identifiers, dict):
        identifiers = [identifiers]
    names = []
    for identifier in identifiers if isinstance(identifiers, list) else []:
        if not isinstance(identifier, dict):
            continue
        if str(identifier.get("type", "")).upper() != "NAME":
            continue
        value = str(identifier.get("value", "") or "").strip()
        if value:
            names.append(value)
    return names


def iter_input_components(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    inputs = payload.get("inputs", {})
    groups: Iterable[Any]
    if isinstance(inputs, dict):
        groups = inputs.values()
    elif isinstance(inputs, list):
        groups = inputs
    else:
        groups = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        components = group.get("components", [])
        if isinstance(components, dict):
            components = [components]
        for component in components if isinstance(components, list) else []:
            if isinstance(component, dict):
                yield component


def iter_outcome_products(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    outcomes = payload.get("outcomes", [])
    if isinstance(outcomes, dict):
        outcomes = [outcomes]
    for outcome in outcomes if isinstance(outcomes, list) else []:
        if not isinstance(outcome, dict):
            continue
        products = outcome.get("products", [])
        if isinstance(products, dict):
            products = [products]
        for product in products if isinstance(products, list) else []:
            if isinstance(product, dict):
                yield product


def append_unique(target: list[str], values: Iterable[str]) -> None:
    known = {normalize_name_key(value) for value in target}
    for value in values:
        normalized = normalize_name_key(value)
        if normalized and normalized not in known:
            target.append(value)
            known.add(normalized)


def extract_ord_names(
    payload: dict[str, Any],
) -> tuple[dict[str, list[str]], list[str]]:
    """Convert explicit ORD roles into reactant, reagent, and product names."""
    result = {field: [] for field in COMPONENT_FIELDS}
    ignored_roles: list[str] = []
    for component in iter_input_components(payload):
        role = str(component.get("reaction_role", "") or "").upper()
        field = ROLE_TO_FIELD.get(role)
        if field is None:
            if role:
                ignored_roles.append(role)
            continue
        append_unique(result[field], component_names(component))

    for product in iter_outcome_products(payload):
        append_unique(result["products"], component_names(product))
    return result, sorted(set(ignored_roles))


def match_source_rows(
    records: list[dict[str, Any]],
    review: pd.DataFrame,
) -> list[int]:
    queues: dict[tuple[str, str], deque[int]] = defaultdict(deque)
    for _, row in review.sort_values("idx").iterrows():
        key = (
            str(row.get("title", "") or ""),
            str(row.get("paragraph", "") or ""),
        )
        queues[key].append(int(row["idx"]))

    indices = []
    for record in records:
        key = (
            str(record.get("title", "") or ""),
            str(record.get("paragraph", "") or ""),
        )
        if not queues[key]:
            raise ValueError(
                "Ai result could not be matched to ground truth by title and "
                f"paragraph: {key[0]!r}"
            )
        indices.append(queues[key].popleft())
    if any(queue for queue in queues.values()):
        raise ValueError("Ground-truth rows remain unmatched after Ai result import.")
    return indices


def parse_record(record: dict[str, Any], source_row: int) -> dict[str, Any]:
    payload, parse_error = parse_ord_prediction(record.get("prediction", ""))
    if payload is None:
        names = {field: [] for field in COMPONENT_FIELDS}
        ignored_roles: list[str] = []
    else:
        names, ignored_roles = extract_ord_names(payload)
    return {
        "source_row": source_row,
        "title": str(record.get("title", "") or ""),
        "paragraph": str(record.get("paragraph", "") or ""),
        "raw_prediction": str(record.get("prediction", "") or ""),
        "parse_error": parse_error,
        "ignored_roles": ignored_roles,
        "names": names,
    }


def read_benchmark_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return frame.assign(idx=range(len(frame)))


def load_records(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"{path}: Ai result JSON must contain a list of records.")
    return records


def ai_spellings(parsed_rows: list[dict[str, Any]]) -> list[str]:
    """Every distinct cleaned spelling the resolver is asked about."""
    from analysis.name_resolution_replay import clean_name

    return sorted({
        clean_name(name)
        for row in parsed_rows
        for field in COMPONENT_FIELDS
        for name in row["names"][field]
    })


def automatic_resolutions(spellings: list[str], requery_log: Path) -> dict[str, tuple[str | None, str]]:
    """spelling -> (SMILES or None, source), replayed from a recording without lookup tables."""
    from analysis.name_resolution_replay import RecordedResolver

    resolver = RecordedResolver(requery_log)
    resolved = resolver.resolve(spellings, order="two-pass")
    if resolver.missing_urls:
        raise ValueError(
            f"{len(resolver.missing_urls)} requests are not in {requery_log}, e.g. "
            f"{sorted(resolver.missing_urls)[0]}"
        )
    result = {}
    for spelling, record in resolved.items():
        smiles = canonicalize_smiles(record["selected_smiles"]) if record["selected_smiles"] else None
        if smiles:
            result[spelling] = (smiles, f"{record['selected_source']} ({record['resolved_by']})")
        else:
            result[spelling] = (None, "refused" if record["refused"] else "not_found")
    return result


def build_outputs(
    parsed_rows: list[dict[str, Any]],
    resolve: Any,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Assemble component rows; ``resolve(row, field, name)`` gives (SMILES, source)."""
    component_rows = []
    resolution_rows: list[dict[str, Any]] = []
    for row in parsed_rows:
        output_components = {field: [] for field in COMPONENT_FIELDS}
        unresolved_names = []
        for field in COMPONENT_FIELDS:
            for name in row["names"][field]:
                normalized = normalize_name_key(name)
                smiles, source = resolve(row, field, name)
                if smiles:
                    output_value = smiles
                    is_nosmi = False
                else:
                    output_value = f"[{name} (NoSmi)]"
                    unresolved_names.append(name)
                    is_nosmi = True
                output_components[field].append(output_value)
                resolution_rows.append(
                    {
                        "source_row": row["source_row"],
                        "title": row["title"],
                        "paragraph": row["paragraph"],
                        "role": field,
                        "name": name,
                        "normalized_name": normalized,
                        "smiles": smiles or "",
                        "resolution_source": source or "not_found",
                        "is_nosmi": int(is_nosmi),
                    }
                )

        parse_failed = bool(row["parse_error"])
        component_rows.append(
            {
                "idx": row["source_row"],
                "title": row["title"],
                "paragraph": row["paragraph"],
                f"{AI_METHOD}_reactants_smiles": output_components["reactants"],
                f"{AI_METHOD}_reagents_smiles": output_components["reagents"],
                f"{AI_METHOD}_products_smiles": output_components["products"],
                f"{AI_METHOD}_has_unresolved_nosmi": int(
                    parse_failed or bool(unresolved_names)
                ),
                f"{AI_METHOD}_unresolved_nosmi": unresolved_names,
                f"{AI_METHOD}_parse_error": row["parse_error"],
                f"{AI_METHOD}_ignored_roles": row["ignored_roles"],
                f"{AI_METHOD}_raw_output": row["raw_prediction"],
            }
        )
    output = pd.DataFrame(component_rows).sort_values("idx")
    return output, resolution_rows


def write_outputs(
    args: argparse.Namespace,
    output: pd.DataFrame,
    resolution_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    resolution_frame = pd.DataFrame(resolution_rows)
    resolution_path = Path(args.resolution_output)
    resolution_path.parent.mkdir(parents=True, exist_ok=True)
    resolution_frame.to_csv(
        resolution_path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    return resolution_frame


def matches_reference(args: argparse.Namespace, reference_dir: Path) -> int:
    """0 when the produced files equal the reference files that exist, else 1."""
    identical = True
    for produced, name in (
        (args.output, "ai_baseline_components.csv"),
        (args.resolution_output, "ai_baseline_name_resolution.csv"),
    ):
        reference_path = reference_dir / name
        if not reference_path.is_file():
            continue
        produced_frame = pd.read_csv(produced, encoding="utf-8-sig", dtype=str).fillna("")
        reference = pd.read_csv(reference_path, encoding="utf-8-sig", dtype=str).fillna("")
        same = produced_frame.equals(reference)
        identical &= same
        print(f"Matches released file: {'yes' if same else 'NO'} ({reference_path})")
    return 0 if identical else 1


def print_summary(output: pd.DataFrame, resolution_frame: pd.DataFrame, args) -> None:
    parse_failures = int(output[f"{AI_METHOD}_parse_error"].astype(bool).sum())
    unresolved_occurrences = int(resolution_frame["is_nosmi"].sum())
    unresolved_names = (
        int(resolution_frame.loc[resolution_frame["is_nosmi"].eq(1), "normalized_name"].nunique())
        if not resolution_frame.empty
        else 0
    )
    print(f"Prepared {len(output)} Ai baseline rows.")
    print(f"Strict JSON parse failures: {parse_failures}")
    print(
        "Unresolved name occurrences: "
        f"{unresolved_occurrences} ({unresolved_names} unique names)"
    )
    print(f"Components: {args.output}")


def current_benchmark_records(input_json: Path, current_input: Path = CURRENT_INPUT) -> list[dict[str, Any]]:
    """Match all predictions to the current title/paragraph pairs, exactly once."""
    records = load_records(input_json)
    source_rows = match_source_rows(records, read_benchmark_input(current_input))
    parsed = [parse_record(record, idx) for record, idx in zip(records, source_rows)]
    return sorted(parsed, key=lambda row: row['source_row'])

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-json', default='benchmarks/ai_et_al/output.json')
    parser.add_argument('--current-input', default=str(CURRENT_INPUT))
    parser.add_argument('--requery-log', default=str(DEFAULT_REQUERY_LOG))
    parser.add_argument('--reference-dir', default=str(RELEASE_DIR))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_DIR / 'ai_baseline_components.csv'))
    parser.add_argument('--resolution-output', default=str(DEFAULT_OUTPUT_DIR / 'ai_baseline_name_resolution.csv'))
    return parser

def automatic(args: argparse.Namespace) -> int:
    from analysis.name_resolution_replay import clean_name
    rows = current_benchmark_records(Path(args.input_json), Path(args.current_input))
    resolutions = automatic_resolutions(ai_spellings(rows), Path(args.requery_log))
    output, resolution_rows = build_outputs(rows, lambda row, field, name: resolutions[clean_name(name)])
    resolution_frame = write_outputs(args, output, resolution_rows)
    print_summary(output, resolution_frame, args)
    return matches_reference(args, Path(args.reference_dir))

def main() -> int:
    return automatic(build_parser().parse_args())

if __name__ == '__main__':
    raise SystemExit(main())
