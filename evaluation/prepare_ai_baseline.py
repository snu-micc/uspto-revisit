"""Prepare the Ai et al. ORD-JSON baseline for Table 2 evaluation.

The source model was evaluated on a different output schema. This script
converts its explicit ORD roles into the three component lists used by the
current ground truth and leaves unresolved names as NoSmi markers.

Two modes are available:

``replay`` (default)
    Parses ``benchmarks/ai_et_al/output.json`` and takes every name's SMILES
    from the released name-resolution table
    (``benchmarks/ai_et_al/preprocessing/ai_baseline_name_resolution.csv``).
    This needs no cache or network access and regenerates the released
    component file exactly; the script reports whether it did.

``rebuild``
    The original procedure: names are resolved from the project's earlier
    model outputs and pickle caches (majority vote, with the conflict choices
    in ``model_name_smiles_conflicts.csv``), then from row-level human review,
    and optionally from external services (``--lookup-external``). It needs
    the per-model SMILES caches of the original runs, which are not distributed. A rerun in
    2026-10 with RDKit 2025.09 differed from the released file in 16 of 400
    rows, mainly in how RDKit writes some metal-containing SMILES. The
    original external lookups also applied the curated overrides
    (``config/nosmi_overrides.json``); the resolver no longer applies them.

Outputs go to ``evaluation/results/ai_baseline/`` by default so that the
released files are never overwritten.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import pickle
import re
import ssl
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rdkit import RDLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uspto_revisit.json_utils import fix_name, parse_json_object
from uspto_revisit import smiles_fetch
from uspto_revisit.smiles_fetch import (
    canonicalize_smiles,
    normalize_name_key,
    resolve_no_smi_name,
)

RDLogger.DisableLog("rdApp.*")

AI_METHOD = "ai_without_title"
RELEASE_DIR = Path("benchmarks/ai_et_al/preprocessing")
DEFAULT_OUTPUT_DIR = Path("evaluation/results/ai_baseline")
COMPONENT_FIELDS = ("reactants", "reagents", "products")
ROLE_TO_FIELD = {
    "REACTANT": "reactants",
    "SOLVENT": "reagents",
    "CATALYST": "reagents",
    "REAGENT": "reagents",
    "PRODUCT": "products",
}
TRAILING_MARKER = re.compile(r"\s*#{3,}\s*$")
NO_SMI_DECISIONS = {"keep_unresolved", "exclude_from_mapping"}


def is_blank(value: Any) -> bool:
    return value is None or (
        isinstance(value, float) and pd.isna(value)
    ) or not str(value).strip()


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


def parse_literal_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_blank(value):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def standard_prediction_names(value: Any) -> dict[str, str]:
    payload = parse_json_object(value)
    if payload is None:
        return {}
    names: dict[str, str] = {}
    for key in ("Reactants, Solvents, Catalysts", "Product", "Products"):
        section = payload.get(key, {})
        if isinstance(section, dict):
            names.update(
                {
                    str(code): str(name)
                    for code, name in section.items()
                    if not is_blank(name)
                }
            )
    return names


def add_cache_candidate(
    candidates: dict[str, Counter[str]],
    examples: dict[str, str],
    name: Any,
    smiles: Any,
) -> None:
    if is_blank(name) or is_blank(smiles) or "(NoSmi)" in str(smiles):
        return
    normalized = normalize_name_key(str(name))
    canonical = canonicalize_smiles(str(smiles))
    if not normalized or canonical is None:
        return
    candidates[normalized][canonical] += 1
    examples.setdefault(normalized, str(name).strip())


def load_pickle_candidates(
    results_dir: Path,
    candidates: dict[str, Counter[str]],
    examples: dict[str, str],
) -> None:
    for path in sorted((results_dir / "smiles_batches").glob("*/smiles_cache.pkl")):
        try:
            with path.open("rb") as handle:
                cache = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError):
            continue
        if not isinstance(cache, dict):
            continue
        for name, smiles in cache.items():
            add_cache_candidate(candidates, examples, name, smiles)


def load_model_output_candidates(
    results_dir: Path,
    candidates: dict[str, Counter[str]],
    examples: dict[str, str],
) -> None:
    for path in sorted(results_dir.glob("*_reaction_smiles_final.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig").fillna("")
        smiles_columns = [
            column
            for column in frame.columns
            if column.endswith("_smiles") and column != "Lowe_smiles"
        ]
        if "prediction" not in frame or not smiles_columns:
            continue
        smiles_column = smiles_columns[0]
        for _, row in frame.iterrows():
            names = standard_prediction_names(row["prediction"])
            structures = parse_literal_dict(row[smiles_column])
            for code, name in names.items():
                if code in structures:
                    add_cache_candidate(
                        candidates,
                        examples,
                        name,
                        structures[code],
                    )


def load_conflict_selections(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    selected: dict[str, str] = {}
    for _, row in frame.iterrows():
        normalized = str(row.get("normalized_name", "") or "").strip()
        canonical = canonicalize_smiles(str(row.get("selected_smiles", "") or ""))
        if normalized and canonical:
            selected[normalized] = canonical
    return selected


def select_existing_cache(
    results_dir: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Select a single cached structure by curated choice or strict majority."""
    candidates: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, str] = {}
    load_pickle_candidates(results_dir, candidates, examples)
    load_model_output_candidates(results_dir, candidates, examples)
    conflict_selections = load_conflict_selections(
        results_dir / "model_name_smiles_conflicts.csv"
    )

    selected: dict[str, str] = {}
    sources: dict[str, str] = {}
    for normalized, counts in candidates.items():
        if normalized in conflict_selections:
            selected[normalized] = conflict_selections[normalized]
            sources[normalized] = "ExistingConflictAudit"
            continue
        ranked = counts.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            selected[normalized] = ranked[0][0]
            sources[normalized] = "ExistingModelCache"
    return selected, sources, examples


def load_row_review(path: Path) -> dict[tuple[int, str], dict[str, str]]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    result: dict[tuple[int, str], dict[str, str]] = {}
    for _, row in frame.iterrows():
        try:
            source_row = int(row.get("row_zero_based", ""))
        except (TypeError, ValueError):
            continue
        normalized = str(row.get("normalized_name", "") or "").strip()
        if not normalized:
            normalized = normalize_name_key(str(row.get("name", "") or ""))
        if normalized:
            result[(source_row, normalized)] = {
                key: str(row.get(key, "") or "")
                for key in (
                    "review_decision",
                    "reviewed_smiles",
                    "evidence",
                    "review_note",
                )
            }
    return result


def load_ai_cache(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not path.is_file():
        return {}, {}
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
        return {}, {}
    if not isinstance(payload, dict):
        return {}, {}
    structures = payload.get("structures", payload)
    sources = payload.get("sources", {}) if "structures" in payload else {}
    if not isinstance(structures, dict):
        return {}, {}
    selected: dict[str, str] = {}
    for normalized, smiles in structures.items():
        canonical = canonicalize_smiles(str(smiles))
        if canonical:
            selected[str(normalized)] = canonical
    return selected, {
        str(key): str(value) for key, value in sources.items()
    } if isinstance(sources, dict) else {}


def save_ai_cache(
    path: Path,
    structures: dict[str, str],
    sources: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"structures": structures, "sources": sources}, handle)


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


async def resolve_names_online(
    targets: dict[str, str],
    *,
    concurrency: int,
) -> tuple[dict[str, str], dict[str, str]]:
    if not targets:
        return {}, {}
    # Imported here so that replay and offline rebuilds work without them.
    import aiohttp
    import certifi

    semaphore = asyncio.Semaphore(concurrency)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_context),
        timeout=timeout,
    ) as session:
        async def resolve_one(normalized: str, name: str) -> tuple[str, str | None, str]:
            smiles, source = await resolve_no_smi_name(name, session, semaphore)
            if smiles is None:
                fixed = fix_name(name)
                if fixed and fixed != name:
                    smiles, source = await resolve_no_smi_name(
                        fixed,
                        session,
                        semaphore,
                    )
                    if smiles:
                        source = f"{source}+FixedName"
            return normalized, canonicalize_smiles(smiles), str(source or "")

        results = await asyncio.gather(
            *(resolve_one(normalized, name) for normalized, name in targets.items())
        )
    structures = {
        normalized: smiles
        for normalized, smiles, _ in results
        if smiles is not None
    }
    sources = {
        normalized: source
        for normalized, smiles, source in results
        if smiles is not None
    }
    return structures, sources


def reviewed_structure(
    decision: dict[str, str] | None,
) -> tuple[str | None, str]:
    if not decision:
        return None, ""
    action = decision.get("review_decision", "").strip()
    if action == "use_smiles":
        canonical = canonicalize_smiles(decision.get("reviewed_smiles", ""))
        return canonical, "HumanReview" if canonical else ""
    if action in NO_SMI_DECISIONS:
        return None, f"HumanReview:{action}"
    return None, ""


def build_review_queue(
    resolution_rows: list[dict[str, Any]],
    previous: dict[tuple[int, str], dict[str, str]],
) -> pd.DataFrame:
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    global_counts = Counter(
        row["normalized_name"] for row in resolution_rows if row["normalized_name"]
    )
    for row in resolution_rows:
        if not row["is_nosmi"]:
            continue
        key = (int(row["source_row"]), row["normalized_name"])
        current = grouped.setdefault(
            key,
            {
                "review_id": f"{key[0]}:{key[1]}",
                "row_zero_based": key[0],
                "name": row["name"],
                "normalized_name": key[1],
                "currently_unresolved": True,
                "automatic_status": "not_found",
                "occurrences": global_counts[key[1]],
                "roles": set(),
                "candidate_smiles": "",
                "resolver_source": row["resolution_source"],
                "title": row["title"],
                "paragraph": row["paragraph"],
                "review_decision": "",
                "reviewed_smiles": "",
                "evidence": "",
                "review_note": "",
            },
        )
        current["roles"].add(row["role"])

    for key, decision in previous.items():
        if key not in grouped:
            continue
        grouped[key].update(decision)

    columns = [
        "review_id",
        "row_zero_based",
        "name",
        "normalized_name",
        "currently_unresolved",
        "automatic_status",
        "occurrences",
        "roles",
        "candidate_smiles",
        "resolver_source",
        "title",
        "paragraph",
        "review_decision",
        "reviewed_smiles",
        "evidence",
        "review_note",
    ]
    rows = []
    for row in grouped.values():
        row = dict(row)
        row["roles"] = " | ".join(sorted(row["roles"]))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["occurrences", "row_zero_based", "normalized_name"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=("replay", "rebuild"), default="replay")
    parser.add_argument(
        "--input-json",
        default="benchmarks/ai_et_al/output.json",
        help=(
            "Ai et al. predictions. The released file is byte-identical (in Git) to "
            "the original ai_results_accessory_max_seq_2048_reppen.json."
        ),
    )
    parser.add_argument(
        "--ground-truth",
        default="evaluation/ground_truth_review.csv",
    )
    parser.add_argument(
        "--resolution-table",
        default=str(RELEASE_DIR / "ai_baseline_name_resolution.csv"),
        help="replay: released per-row name-resolution table.",
    )
    parser.add_argument(
        "--reference-components",
        default=str(RELEASE_DIR / "ai_baseline_components.csv"),
        help="replay: released component file used for the equality check.",
    )
    parser.add_argument(
        "--results-dir",
        default="result",
        help=(
            "rebuild: result folder of the original runs with *_reaction_smiles_final.csv, "
            "smiles_batches/*/smiles_cache.pkl and model_name_smiles_conflicts.csv."
        ),
    )
    parser.add_argument(
        "--existing-review",
        default="result/nosmi_human_review.csv",
        help="rebuild: row-specific decisions shared by the five earlier models.",
    )
    parser.add_argument(
        "--ai-review",
        default=str(RELEASE_DIR / "ai_nosmi_human_review.csv"),
        help="rebuild: row-specific decisions for Ai et al. names.",
    )
    parser.add_argument(
        "--review-output",
        default=str(DEFAULT_OUTPUT_DIR / "ai_nosmi_human_review.csv"),
        help="rebuild: refreshed review queue (decisions from --ai-review are kept).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "ai_baseline_components.csv"),
    )
    parser.add_argument(
        "--resolution-output",
        default=str(DEFAULT_OUTPUT_DIR / "ai_baseline_name_resolution.csv"),
    )
    parser.add_argument("--lookup-concurrency", type=int, default=8)
    parser.add_argument(
        "--lookup-external",
        action="store_true",
        help=(
            "Query external chemistry services for names absent from local "
            "caches. By default, unresolved names remain NoSmi."
        ),
    )
    return parser


def parse_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Parse the Ai predictions and attach benchmark row numbers."""
    with Path(args.input_json).open("r", encoding="utf-8-sig") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Ai result JSON must contain a list of records.")
    review = pd.read_csv(args.ground_truth, encoding="utf-8-sig").fillna("")
    review["idx"] = review["idx"].astype(int)
    source_rows = match_source_rows(records, review)

    parsed_rows: list[dict[str, Any]] = []
    for record, source_row in zip(records, source_rows):
        payload, parse_error = parse_ord_prediction(record.get("prediction", ""))
        if payload is None:
            names = {field: [] for field in COMPONENT_FIELDS}
            ignored_roles: list[str] = []
        else:
            names, ignored_roles = extract_ord_names(payload)
        parsed_rows.append(
            {
                "source_row": source_row,
                "title": str(record.get("title", "") or ""),
                "paragraph": str(record.get("paragraph", "") or ""),
                "raw_prediction": str(record.get("prediction", "") or ""),
                "parse_error": parse_error,
                "ignored_roles": ignored_roles,
                "names": names,
            }
        )
    return parsed_rows


def load_resolution_table(path: Path) -> dict[tuple[int, str, str], tuple[str, str]]:
    """Read the released (row, role, name) -> (SMILES, source) table."""
    frame = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    table: dict[tuple[int, str, str], tuple[str, str]] = {}
    for _, row in frame.iterrows():
        key = (int(row["source_row"]), str(row["role"]), str(row["normalized_name"]))
        if key in table:
            raise ValueError(f"{path}: duplicate resolution row {key}.")
        smiles = "" if int(row["is_nosmi"]) else str(row["smiles"])
        table[key] = (smiles, str(row["resolution_source"]))
    return table


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
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    resolution_frame = pd.DataFrame(resolution_rows)
    resolution_path = Path(args.resolution_output)
    resolution_path.parent.mkdir(parents=True, exist_ok=True)
    resolution_frame.to_csv(
        resolution_path,
        index=False,
        encoding="utf-8-sig",
    )
    return resolution_frame


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


def replay(args: argparse.Namespace) -> int:
    """Regenerate the released component file from the released name table."""
    parsed_rows = parse_records(args)
    table = load_resolution_table(Path(args.resolution_table))
    used: set[tuple[int, str, str]] = set()

    def resolve(row: dict[str, Any], field: str, name: str) -> tuple[str, str]:
        key = (int(row["source_row"]), field, normalize_name_key(name))
        if key not in table:
            raise ValueError(f"No released resolution for row/role/name {key}.")
        used.add(key)
        return table[key]

    output, resolution_rows = build_outputs(parsed_rows, resolve)
    unused = set(table) - used
    if unused:
        raise ValueError(f"{len(unused)} released resolution rows were not used.")
    resolution_frame = write_outputs(args, output, resolution_rows)
    print_summary(output, resolution_frame, args)

    reference_path = Path(args.reference_components)
    if reference_path.is_file():
        produced = pd.read_csv(args.output, encoding="utf-8-sig", dtype=str).fillna("")
        reference = pd.read_csv(reference_path, encoding="utf-8-sig", dtype=str).fillna("")
        identical = produced.equals(reference)
        print(
            "Matches released component file: "
            f"{'yes' if identical else 'NO'} ({reference_path})"
        )
        return 0 if identical else 1
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "replay":
        return replay(args)

    results_dir = Path(args.results_dir)
    cache_path = results_dir / "smiles_batches" / AI_METHOD / "smiles_cache.pkl"
    parsed_rows = parse_records(args)

    selected, selected_sources, cache_examples = select_existing_cache(results_dir)
    ai_cache, ai_sources = load_ai_cache(cache_path)
    selected.update(ai_cache)
    selected_sources.update(ai_sources)

    existing_review = load_row_review(Path(args.existing_review))
    ai_review = load_row_review(Path(args.ai_review))

    target_examples: dict[str, str] = dict(cache_examples)
    for row in parsed_rows:
        for field in COMPONENT_FIELDS:
            for name in row["names"][field]:
                normalized = normalize_name_key(name)
                if normalized:
                    target_examples.setdefault(normalized, name)

    unresolved_targets = {
        normalized: name
        for normalized, name in target_examples.items()
        if normalized not in selected
    }
    if args.lookup_external:
        smiles_fetch.smiles_cache.clear()
        for normalized, smiles in selected.items():
            example = target_examples.get(normalized, normalized)
            smiles_fetch.smiles_cache[example] = smiles
        online, online_sources = asyncio.run(
            resolve_names_online(
                unresolved_targets,
                concurrency=max(1, args.lookup_concurrency),
            )
        )
        selected.update(online)
        selected_sources.update(online_sources)
    save_ai_cache(cache_path, selected, selected_sources)

    def resolve(row: dict[str, Any], field: str, name: str) -> tuple[str | None, str]:
        normalized = normalize_name_key(name)
        key = (row["source_row"], normalized)
        manual = ai_review.get(key) or existing_review.get(key)
        manual_smiles, manual_source = reviewed_structure(manual)
        if manual_source.startswith("HumanReview:"):
            return None, manual_source
        if manual_smiles:
            return manual_smiles, manual_source
        return selected.get(normalized), selected_sources.get(normalized, "")

    output, resolution_rows = build_outputs(parsed_rows, resolve)
    resolution_frame = write_outputs(args, output, resolution_rows)

    review_queue = build_review_queue(resolution_rows, ai_review)
    review_path = Path(args.review_output)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_queue.to_csv(review_path, index=False, encoding="utf-8-sig")

    print_summary(output, resolution_frame, args)
    print(f"NoSmi review: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
