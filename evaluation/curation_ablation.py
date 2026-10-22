"""Switch off curated NoSmi resolutions when scoring the released outputs.

The released model outputs were resolved with two kinds of manual input:

* ``config/nosmi_overrides.json``: 46 dataset-specific name aliases and
  active-component choices applied during NoSmi retry;
* ``result/nosmi_human_review.csv``: row-scoped ``use_smiles`` decisions made
  during the NoSmi human review.

``--no-overrides`` and ``--no-row-decisions`` rebuild each configuration's
output as the pipeline would have written it without the switched-off choices.
In practice the 19 configurations received these choices through the
name-keyed shared cache, so a component counts as curated as follows:

* overrides: the component is resolved and its normalized name is an override
  key;
* row decisions: the component is resolved, its (row, normalized name) has a
  ``use_smiles`` decision, and its SMILES equals the canonical
  ``reviewed_smiles``. A component carrying another structure was not produced
  by the decision and is left alone.

Each curated component becomes ``[name (NoSmi)]`` in the configuration's
SMILES dictionary. Every reaction step that then contains a NoSmi token gets
the record ``atom_mapping.py`` writes for such steps: no LocalMapper input,
mapped reaction, template or confidence, and the error "Reaction contains an
unresolved (NoSmi) compound." Other steps keep their published mapping. The
affected benchmark rows are rebuilt with ``build_ground_truth_review.model_row``,
and Table 3 reads the rebuilt outputs. So the extraction is scored as
unresolved, the template-anomaly heuristic filters the record, and the
ORDerly role-count heuristics see the unmapped step.

The released outputs do not record which resolver produced each component.
With the default ``--override-scope all`` every override name is switched off,
which gives the largest effect: a name that an automatic resolver might also
have resolved (for example the two ``canonical_structure`` entries, which
correct a resolver result rather than unblock a name) still counts as
unresolved. ``--override-scope blocked`` switches off only names that the
automatic resolver rejects as ambiguous or non-molecular
(``should_skip_automatic_resolution``); these cannot be resolved without an
override, so this is the smallest effect. Both switches are off by default, and
the published tables use neither.

``--name-resolution fully-automatic`` replaces the curated layers with the
automatic resolvers instead of only switching them off. Every curated
component (override names, row decisions, and reviewed SMILES reused by name
in other rows) gets the SMILES that the released resolver chain returns for
its spelling in the recorded 2026-10-16 re-query (see
``name_resolution_replay``), or NoSmi when the chain returns nothing. Every
other stored component that is a lone element atom ([H], [N], [O], [F],
[Cl], [Br], [I], which OPSIN and the shared cache gave for "hydrogen",
"nitrogen", "bromine" and so on) or the radical [C]=O (carbon monoxide)
becomes the molecule that the corrected resolver returns (audit reason
``molecular_form``). Steps
whose reaction text changes are rebuilt as the pipeline would: steps with a
NoSmi token are unmapped, and the other changed steps are re-mapped with
LocalMapper 202403 on the CPU, one reaction at a time (needs the ``mapping``
extra). This is offline.
"""

from __future__ import annotations

import ast
import html
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OVERRIDES = Path("config/nosmi_overrides.json")
DEFAULT_ROW_DECISIONS = Path("result/nosmi_human_review.csv")
DEFAULT_RESULTS_DIR = Path("result/model_outputs")
NAME_RESOLUTIONS = ("published", "fully-automatic")
COMPONENT_SECTIONS = ("Reactants, Solvents, Catalysts", "Products", "Product")
OVERRIDE_SCOPES = ("all", "blocked")
NOSMI_MAPPING_ERROR = "Reaction contains an unresolved (NoSmi) compound."
MAPPING_COLUMNS = (
    "localmapper_rxn",
    "mapped_rxn",
    "mapping_template",
    "mapping_confident",
    "mapping_error",
)
AUDIT_COLUMNS = (
    "model",
    "idx",
    "label",
    "name",
    "normalized_name",
    "smiles",
    "in_skeleton",
    "reason",
    "override_kind",
    "automatic_resolution_blocked",
)
AUTOMATIC_AUDIT_COLUMNS = AUDIT_COLUMNS + (
    "spelling",
    "automatic_smiles",
    "automatic_source",
    "resolved_by",
    "resolver_agreement",
    "outcome",
)


def normalize_name_key(compound_name: str) -> str:
    """Same key as ``uspto_revisit.smiles_fetch.normalize_name_key``.

    Duplicated so the evaluation scripts do not need the extraction package;
    tests/test_curation_ablation.py checks that the two stay identical.
    """
    cleaned = " ".join(str(compound_name).replace("′", "'").split())
    cleaned = html.unescape(re.sub(r"<[^>]+>", "", cleaned))
    cleaned = (
        cleaned.replace("α", "alpha")
        .replace("β", "beta")
        .replace("γ", "gamma")
        .replace("δ", "delta")
    )
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", "", cleaned)


# Same as ``uspto_revisit.smiles_fetch.MOLECULAR_FORMS``: an element name
# resolved to one atom ("hydrogen" -> [H]) is the diatomic molecule, and
# carbon monoxide resolved to the radical [C]=O is [C-]#[O+].
MOLECULAR_FORMS = {
    "[H]": "[H][H]",
    "[N]": "N#N",
    "[O]": "O=O",
    "[F]": "FF",
    "[Cl]": "ClCl",
    "[Br]": "BrBr",
    "[I]": "II",
    "[C]=O": "[C-]#[O+]",
}


def canonicalize_smiles(smiles: Any) -> str | None:
    """Same as ``uspto_revisit.smiles_fetch.canonicalize_smiles``."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        molecule = Chem.MolFromSmiles(smiles.strip())
        if molecule is None:
            return None
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        return MOLECULAR_FORMS.get(canonical, canonical)
    except Exception:
        return None


def molecular_form_components(outputs: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Audit rows for stored components that are a lone element atom or [C]=O.

    The released outputs keep the resolver's form ([H], [Br], [C]=O, ...); the
    corrected resolver returns the molecule, so the fully automatic rebuild
    replaces them.
    """
    rows = []
    for model, frame in outputs.items():
        frame = frame.fillna("")
        for idx in range(len(frame)):
            smiles = loads_any(frame.at[idx, f"{model}_smiles"])
            if not isinstance(smiles, dict):
                continue
            names = prediction_names(frame.at[idx, "prediction"])
            in_skeleton = skeleton_labels(frame.at[idx, f"{model}_skeleton"])
            for label, value in smiles.items():
                molecule = Chem.MolFromSmiles(str(value)) if "(NoSmi)" not in str(value) else None
                atom = Chem.MolToSmiles(molecule) if molecule is not None else None
                if atom not in MOLECULAR_FORMS:
                    continue
                name = names.get(label, "")
                rows.append(
                    {
                        "model": model,
                        "idx": idx,
                        "label": label,
                        "name": name,
                        "normalized_name": normalize_name_key(name),
                        "smiles": value,
                        "in_skeleton": int(label in in_skeleton),
                        "reason": "molecular_form",
                        "override_kind": "",
                        "automatic_resolution_blocked": "",
                        "spelling": name,
                        "automatic_smiles": MOLECULAR_FORMS[atom],
                        "automatic_source": "MolecularForm",
                        "resolved_by": "",
                        "resolver_agreement": "",
                        "outcome": "replaced",
                    }
                )
    return rows


def replace_with_smiles(rxn_code: str, smiles_dict: dict[str, str]) -> str:
    """Same as ``uspto_revisit.reaction_smiles.replace_with_smiles``."""
    return ">".join(
        ".".join(smiles_dict.get(code, code) for code in part.split("."))
        for part in rxn_code.split(">")
    )


def is_blank(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or not str(value).strip()
    )


def repository_path(path: str | Path) -> Path:
    """Use ``path`` as given, or relative to the repository when it does not exist."""
    path = Path(path)
    if path.exists() or path.is_absolute():
        return path
    candidate = REPOSITORY_ROOT / path
    return candidate if candidate.exists() else path


def loads_any(value: Any) -> Any:
    """Parse a JSON or Python-literal value as stored in the model outputs."""
    if not isinstance(value, str) or not value.strip():
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            parsed = parse(value)
        except Exception:
            continue
        if isinstance(parsed, str):
            for inner in (json.loads, ast.literal_eval):
                try:
                    parsed = inner(parsed)
                    break
                except Exception:
                    continue
        return parsed
    return None


def parse_pylist(value: Any) -> list:
    """Parse the Python-repr lists stored in the *_rxn and mapping columns."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]


def prediction_names(raw_output: Any) -> dict[str, str]:
    """Map entity labels to the names in the model's structured output."""
    parsed = loads_any(raw_output)
    names: dict[str, str] = {}
    if isinstance(parsed, dict):
        for section in COMPONENT_SECTIONS:
            values = parsed.get(section)
            if isinstance(values, dict):
                names.update({str(label): str(name) for label, name in values.items()})
    return names


def skeleton_labels(skeleton: Any) -> set[str]:
    """Entity labels that take part in a reaction skeleton."""
    labels: set[str] = set()
    for reaction in parse_pylist(skeleton):
        for section in str(reaction).split(">"):
            for token in section.split("."):
                label = token.split("(", 1)[0].strip()
                if label:
                    labels.add(label)
    return labels


def load_override_keys(path: str | Path) -> dict[str, dict[str, Any]]:
    with repository_path(path).open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    entries = payload.get("overrides", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list of overrides.")
    keys = {}
    for entry in entries:
        key = normalize_name_key(entry.get("name", ""))
        if key:
            keys[key] = entry
    return keys


def load_row_decisions(path: str | Path) -> dict[tuple[int, str], str | None]:
    """Return (row, normalized name) -> canonical reviewed SMILES for use_smiles."""
    frame = pd.read_csv(repository_path(path), encoding="utf-8-sig").fillna("")
    required = {"row_zero_based", "review_decision", "reviewed_smiles"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    decisions = {}
    for _, row in frame.iterrows():
        if str(row["review_decision"]).strip().lower() != "use_smiles":
            continue
        key = str(row.get("normalized_name", "") or "").strip()
        key = key or normalize_name_key(str(row.get("name", "")))
        if key:
            decisions[(int(row["row_zero_based"]), key)] = canonicalize_smiles(
                str(row["reviewed_smiles"])
            )
    return decisions


def automatic_resolution_blocker():
    """Return the extraction package's ambiguity filter.

    Imported lazily so that evaluations without ``--override-scope blocked``
    do not need the ``uspto_revisit`` package.
    """
    try:
        from uspto_revisit.smiles_fetch import should_skip_automatic_resolution
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "--override-scope blocked needs the uspto_revisit package; run "
            "'python -m pip install -e .' or add src/ to PYTHONPATH."
        ) from exc
    return should_skip_automatic_resolution


def read_model_output(results_dir: str | Path, model: str) -> pd.DataFrame:
    """Read one released output as ``build_ground_truth_review`` does."""
    path = repository_path(results_dir) / f"{model}_reaction_smiles_final.csv"
    return pd.read_csv(path, encoding="utf-8-sig")


def find_curated_components(
    outputs: dict[str, pd.DataFrame],
    *,
    override_path: str | Path | None = None,
    row_decision_path: str | Path | None = None,
    override_scope: str = "all",
    reused_review_smiles: bool = False,
) -> pd.DataFrame:
    """List every resolved component produced by a switched-off curated choice.

    ``reused_review_smiles`` also counts components in rows without a decision
    for their name that carry a SMILES reviewed for that name in another row
    (the shared cache spread these, for example to alcohol and hexanes).
    """
    if override_scope not in OVERRIDE_SCOPES:
        raise ValueError(f"override_scope must be one of {OVERRIDE_SCOPES}.")
    override_keys = load_override_keys(override_path) if override_path else {}
    decisions = load_row_decisions(row_decision_path) if row_decision_path else {}
    reviewed_by_name: dict[str, set] = {}
    for (_row, key), reviewed in decisions.items():
        reviewed_by_name.setdefault(key, set()).add(reviewed)
    is_blocked = None
    if override_keys:
        try:
            is_blocked = automatic_resolution_blocker()
        except ImportError:
            # Only the smallest-effect scope needs the filter; otherwise the
            # audit column is left blank.
            if override_scope == "blocked":
                raise

    rows = []
    for model, frame in outputs.items():
        frame = frame.fillna("")
        for idx in range(len(frame)):
            smiles = loads_any(frame.at[idx, f"{model}_smiles"])
            if not isinstance(smiles, dict):
                continue
            names = prediction_names(frame.at[idx, "prediction"])
            in_skeleton = skeleton_labels(frame.at[idx, f"{model}_skeleton"])
            for label, name in names.items():
                value = str(smiles.get(label, ""))
                canonical = None if "(NoSmi)" in value else canonicalize_smiles(value)
                if canonical is None:
                    continue
                key = normalize_name_key(name)
                blocked = is_blocked(name) if is_blocked is not None else None
                reasons = []
                if key in override_keys and (override_scope == "all" or blocked):
                    reasons.append("override")
                if (idx, key) in decisions and decisions[(idx, key)] == canonical:
                    reasons.append("row_decision")
                elif (
                    reused_review_smiles
                    and (idx, key) not in decisions
                    and canonical in reviewed_by_name.get(key, set())
                ):
                    reasons.append("review_smiles_reused")
                if not reasons:
                    continue
                rows.append(
                    {
                        "model": model,
                        "idx": idx,
                        "label": label,
                        "name": name,
                        "normalized_name": key,
                        "smiles": value,
                        "in_skeleton": int(label in in_skeleton),
                        "reason": "+".join(reasons),
                        "override_kind": override_keys.get(key, {}).get("kind", ""),
                        "automatic_resolution_blocked": "" if blocked is None else int(blocked),
                    }
                )
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def rebuild_row(
    frame: pd.DataFrame,
    idx: int,
    model: str,
    smiles_dict: dict,
    remap=None,
) -> None:
    """Rewrite one output row after its SMILES dictionary changed.

    Steps whose reaction now contains a NoSmi token get the record that
    ``atom_mapping.py`` writes for them; unchanged steps keep their mapping;
    other changed steps are re-mapped with ``remap(reaction) -> record``.
    """
    skeleton = parse_pylist(frame.at[idx, f"{model}_skeleton"])
    old_rxn = parse_pylist(frame.at[idx, f"{model}_rxn"])
    old = {c: parse_pylist(frame.at[idx, f"{model}_{c}"]) for c in MAPPING_COLUMNS}
    frame.at[idx, f"{model}_smiles"] = json.dumps(smiles_dict, ensure_ascii=False)
    if not skeleton:
        return
    new_rxn = [replace_with_smiles(code, smiles_dict) for code in skeleton]
    if len(new_rxn) != len(old_rxn):
        raise ValueError(f"{model} row {idx}: skeleton and reaction lists differ in length.")
    new: dict[str, list] = {c: [] for c in MAPPING_COLUMNS}
    for step, reaction in enumerate(new_rxn):
        if reaction == old_rxn[step]:
            for column in MAPPING_COLUMNS:
                new[column].append(old[column][step] if step < len(old[column]) else None)
        elif "(NoSmi)" in reaction:
            for column in MAPPING_COLUMNS[:-1]:
                new[column].append(None)
            new["mapping_error"].append(NOSMI_MAPPING_ERROR)
        elif remap is not None:
            record = remap(reaction)
            for column in MAPPING_COLUMNS:
                new[column].append(record[column])
        else:
            raise ValueError(f"{model} row {idx}: step {step} changed but no mapper was given.")
    frame.at[idx, f"{model}_rxn"] = repr(new_rxn)
    for column in MAPPING_COLUMNS:
        frame.at[idx, f"{model}_{column}"] = repr(new[column])


def counterfactual_outputs(
    outputs: dict[str, pd.DataFrame],
    edits: dict[tuple[str, int], dict[str, str | None]],
    remap=None,
) -> dict[str, pd.DataFrame]:
    """Copies of the outputs with SMILES edits applied and steps rebuilt.

    ``edits`` maps (configuration, row) to {label: new SMILES}, where None
    means unresolved (the label becomes ``[name (NoSmi)]``).
    """
    rebuilt = {}
    for model, frame in outputs.items():
        frame = frame.copy()
        for column in [f"{model}_smiles", f"{model}_rxn", *(f"{model}_{c}" for c in MAPPING_COLUMNS)]:
            frame[column] = frame[column].astype(object)
        for (edit_model, idx), changes in sorted(edits.items()):
            if edit_model != model:
                continue
            smiles = loads_any(frame.at[idx, f"{model}_smiles"])
            names = prediction_names(frame.at[idx, "prediction"])
            for label, value in changes.items():
                smiles[label] = f"[{names[label]} (NoSmi)]" if value is None else value
            rebuild_row(frame, idx, model, smiles, remap=remap)
        rebuilt[model] = frame
    return rebuilt


def switched_off_edits(audit: pd.DataFrame) -> dict[tuple[str, int], dict[str, None]]:
    edits: dict[tuple[str, int], dict[str, None]] = {}
    for row in audit.itertuples():
        edits.setdefault((row.model, int(row.idx)), {})[row.label] = None
    return edits


_MAPPER = None


def localmapper_remap(reaction: str) -> dict[str, Any]:
    """Map one reaction as the pipeline does (LocalMapper 202403, CPU, batch 1)."""
    global _MAPPER
    try:
        from uspto_revisit.atom_mapping import create_localmapper, map_reaction_values
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "--name-resolution fully-automatic re-maps changed steps and needs the "
            "uspto_revisit package with the mapping extra: "
            "python -m pip install -e \".[mapping]\""
        ) from exc
    if _MAPPER is None:
        _MAPPER = create_localmapper(device="cpu", model_version="202403")
    out = map_reaction_values([reaction], _MAPPER, batch_size=1)
    return {column: out[column][0][0] for column in MAPPING_COLUMNS}


def rebuild_review_rows(
    review: pd.DataFrame,
    rebuilt: dict[str, pd.DataFrame],
    rows: set[tuple[str, int]],
) -> pd.DataFrame:
    """Recompute the per-configuration review columns of the given rows."""
    builder = _review_builder()
    review = review.copy()
    positions = {int(idx): position for position, idx in zip(review.index, review["idx"])}
    for model, idx in sorted(rows):
        columns = builder.model_columns(model, builder.model_row(rebuilt[model].iloc[idx], model))
        for column, value in columns.items():
            if column in review.columns and review[column].dtype != object:
                review[column] = review[column].astype(object)
            review.at[positions[idx], column] = value
    return review


def _review_builder():
    try:
        from evaluation import build_ground_truth_review
    except ModuleNotFoundError:
        import build_ground_truth_review  # type: ignore[no-redef]
    return build_ground_truth_review


def ablate(
    review: pd.DataFrame,
    models: tuple[str, ...] | list[str],
    *,
    override_path: str | Path | None = None,
    row_decision_path: str | Path | None = None,
    override_scope: str = "all",
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (rebuilt review, audit, rebuilt model outputs).

    ``review`` is ``evaluation/benchmark_all_configurations.csv``. Only the
    per-configuration columns of affected rows are rebuilt; the cross-model
    consensus columns, which no scorer reads, are left as they are.
    """
    outputs = {model: read_model_output(results_dir, model) for model in models}
    audit = find_curated_components(
        outputs,
        override_path=override_path,
        row_decision_path=row_decision_path,
        override_scope=override_scope,
    )
    edits = switched_off_edits(audit)
    rebuilt = counterfactual_outputs(outputs, edits)
    review = rebuild_review_rows(review, rebuilt, set(edits))
    return review, audit, rebuilt


def automatic_ablate(
    review: pd.DataFrame,
    models: tuple[str, ...] | list[str],
    *,
    override_path: str | Path = DEFAULT_OVERRIDES,
    row_decision_path: str | Path = DEFAULT_ROW_DECISIONS,
    requery_log: str | Path | None = None,
    order: str = "two-pass",
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    remap=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Replace every curated component with the recorded automatic resolution.

    Returns (rebuilt review, audit with one row per curated component and its
    outcome, rebuilt model outputs). ``remap`` defaults to LocalMapper.
    """
    try:
        from evaluation.name_resolution_replay import (
            DEFAULT_REQUERY_LOG,
            RecordedResolver,
            clean_name,
        )
    except ModuleNotFoundError:
        from name_resolution_replay import (  # type: ignore[no-redef]
            DEFAULT_REQUERY_LOG,
            RecordedResolver,
            clean_name,
        )

    outputs = {model: read_model_output(results_dir, model) for model in models}
    audit = find_curated_components(
        outputs,
        override_path=override_path,
        row_decision_path=row_decision_path,
        override_scope="all",
        reused_review_smiles=True,
    )
    resolver = RecordedResolver(repository_path(requery_log or DEFAULT_REQUERY_LOG))
    spellings = sorted({clean_name(name) for name in audit["name"]})
    resolved = resolver.resolve(spellings, order=order)
    if resolver.missing_urls:
        raise ValueError(
            f"The re-query recording lacks {len(resolver.missing_urls)} requested URLs, "
            f"for example {sorted(resolver.missing_urls)[0]}."
        )

    records = []
    edits: dict[tuple[str, int], dict[str, str | None]] = {}
    for row in audit.to_dict("records"):
        spelling = clean_name(row["name"])
        result = resolved[spelling]
        new = canonicalize_smiles(result["selected_smiles"]) if result["selected_smiles"] else None
        old = canonicalize_smiles(row["smiles"])
        outcome = "unresolved" if new is None else ("unchanged" if new == old else "replaced")
        records.append(
            {
                **row,
                "spelling": spelling,
                "automatic_smiles": new or "",
                "automatic_source": result["selected_source"],
                "resolved_by": result["resolved_by"],
                "resolver_agreement": result["resolver_agreement"],
                "outcome": outcome,
            }
        )
        if outcome != "unchanged":
            edits.setdefault((row["model"], int(row["idx"])), {})[row["label"]] = new
    # A curated replacement wins; every other stored element atom or [C]=O becomes the molecule.
    for row in molecular_form_components(outputs):
        changes = edits.setdefault((row["model"], int(row["idx"])), {})
        if row["label"] not in changes:
            changes[row["label"]] = row["automatic_smiles"]
            records.append(row)
    audit = pd.DataFrame(records, columns=AUTOMATIC_AUDIT_COLUMNS)
    rebuilt = counterfactual_outputs(outputs, edits, remap=remap or localmapper_remap)
    review = rebuild_review_rows(review, rebuilt, set(edits))
    return review, audit, rebuilt


def apply_curation_ablation(
    review: pd.DataFrame,
    models: tuple[str, ...] | list[str],
    *,
    override_path: str | Path | None = None,
    row_decision_path: str | Path | None = None,
    override_scope: str = "all",
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rebuilt review, audit); see ``ablate`` for the rebuilt outputs."""
    review, audit, _outputs = ablate(
        review,
        models,
        override_path=override_path,
        row_decision_path=row_decision_path,
        override_scope=override_scope,
        results_dir=results_dir,
    )
    return review, audit


def variant_name(
    *,
    no_overrides: bool = False,
    override_scope: str = "all",
    no_row_decisions: bool = False,
    strict_structures: bool = False,
    first_reference_only: bool = False,
    name_resolution: str = "published",
    resolution_order: str = "two-pass",
) -> str:
    """Name of the sensitivity variant; empty for the published configuration."""
    parts = []
    if name_resolution == "fully-automatic":
        parts.append("fully_automatic" if resolution_order == "two-pass" else "fully_automatic_retry")
    if no_overrides:
        parts.append(
            "no_overrides" if override_scope == "all" else f"no_overrides_{override_scope}"
        )
    if no_row_decisions:
        parts.append("no_row_decisions")
    if strict_structures and first_reference_only:
        parts.append("strict_matching")
    elif strict_structures:
        parts.append("strict_structures")
    elif first_reference_only:
        parts.append("first_reference_only")
    return "+".join(parts)


def add_sensitivity_arguments(parser) -> None:
    """Add the shared, default-off sensitivity switches to an evaluation parser."""
    group = parser.add_argument_group(
        "sensitivity options (all off by default; the published tables use none)"
    )
    group.add_argument(
        "--no-overrides",
        action="store_true",
        help=(
            "Rebuild the outputs without the curated choices in "
            "config/nosmi_overrides.json (components become NoSmi)."
        ),
    )
    group.add_argument(
        "--overrides",
        default=DEFAULT_OVERRIDES.as_posix(),
        help="Override file used by --no-overrides. Default: %(default)s",
    )
    group.add_argument(
        "--override-scope",
        choices=OVERRIDE_SCOPES,
        default="all",
        help=(
            "With --no-overrides: 'all' (largest effect) switches off every name in "
            "the override file; 'blocked' (smallest effect) only names the automatic "
            "resolver rejects as ambiguous or non-molecular. Default: %(default)s"
        ),
    )
    group.add_argument(
        "--no-row-decisions",
        action="store_true",
        help=(
            "Rebuild the outputs without the row-level use_smiles decisions in "
            "result/nosmi_human_review.csv (only components that carry the "
            "reviewed SMILES)."
        ),
    )
    group.add_argument(
        "--row-decisions",
        default=DEFAULT_ROW_DECISIONS.as_posix(),
        help="Human-review file used by --no-row-decisions. Default: %(default)s",
    )
    group.add_argument(
        "--name-resolution",
        choices=NAME_RESOLUTIONS,
        default="published",
        help=(
            "'fully-automatic' replaces every curated component with the result of "
            "the automatic resolvers, replayed offline from the 2026-10-16 re-query, "
            "and re-maps changed steps with LocalMapper (mapping extra). "
            "Default: %(default)s"
        ),
    )
    group.add_argument(
        "--resolution-order",
        choices=("two-pass", "retry"),
        default="two-pass",
        help=(
            "With --name-resolution fully-automatic: the released two-pass order "
            "(abbreviations and OPSIN first, then the NoSmi retry) or the NoSmi "
            "retry path alone. Default: %(default)s"
        ),
    )
    group.add_argument(
        "--requery-log",
        default="result/name_resolution/requery_2026-10-16_raw.jsonl",
        help="Recorded re-query used by --name-resolution fully-automatic. Default: %(default)s",
    )
    group.add_argument(
        "--strict-matching",
        action="store_true",
        help=(
            "Keep stereochemistry, formal charge, protonation and isotopes when "
            "comparing structures, and accept only the first reference reaction. "
            "Equivalent to --strict-structures --first-reference-only."
        ),
    )
    group.add_argument(
        "--strict-structures",
        action="store_true",
        help="Keep stereochemistry, charge, protonation and isotopes; atom maps are still ignored.",
    )
    group.add_argument(
        "--first-reference-only",
        action="store_true",
        help="Ignore the optional second accepted reference (ground_truth_2_* columns).",
    )


def resolve_sensitivity(args) -> dict[str, Any]:
    """Normalize parsed sensitivity switches into one dictionary."""
    strict_structures = bool(args.strict_matching or args.strict_structures)
    first_reference_only = bool(args.strict_matching or args.first_reference_only)
    options = {
        "no_overrides": bool(args.no_overrides),
        "override_scope": args.override_scope,
        "no_row_decisions": bool(args.no_row_decisions),
        "strict_structures": strict_structures,
        "first_reference_only": first_reference_only,
        "name_resolution": args.name_resolution,
        "resolution_order": args.resolution_order,
    }
    if args.name_resolution != "published" and (args.no_overrides or args.no_row_decisions):
        raise SystemExit(
            "--name-resolution fully-automatic already replaces the curated layers; "
            "do not combine it with --no-overrides or --no-row-decisions."
        )
    options["variant"] = variant_name(**options)
    options["override_path"] = args.overrides if args.no_overrides else None
    options["row_decision_path"] = args.row_decisions if args.no_row_decisions else None
    options["requery_log"] = (
        args.requery_log if args.name_resolution == "fully-automatic" else None
    )
    if args.name_resolution == "fully-automatic":
        options["override_path"] = args.overrides
        options["row_decision_path"] = args.row_decisions
    return options


def sensitivity_root(variant: str) -> Path:
    """Default output folder for a non-default variant."""
    return Path("evaluation/results/sensitivity") / variant
