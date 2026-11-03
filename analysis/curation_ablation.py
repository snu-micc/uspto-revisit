"""Shared helpers for curated output construction and programmatic ablation.

The final manuscript compares frozen fully_automatic outputs with the curated
outputs built by build_curated_outputs.py using the 46 released overrides.
The optional ablation helpers operate on supplied data; automatic_ablate
requires an explicit resolver recording. No historical ablation recording is bundled.
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
DEFAULT_RESULTS_DIR = Path("result/model_outputs/curated")
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
    override_scope: str = "all",
) -> pd.DataFrame:
    """List every resolved component produced by a switched-off override."""
    if override_scope not in OVERRIDE_SCOPES:
        raise ValueError(f"override_scope must be one of {OVERRIDE_SCOPES}.")
    override_keys = load_override_keys(override_path) if override_path else {}
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
        from analysis import build_ground_truth_review
    except ModuleNotFoundError:
        import build_ground_truth_review  # type: ignore[no-redef]
    return build_ground_truth_review


def ablate(
    review: pd.DataFrame,
    models: tuple[str, ...] | list[str],
    *,
    override_path: str | Path | None = None,
    override_scope: str = "all",
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (rebuilt review, audit, rebuilt model outputs).

    ``review`` is ``analysis/benchmark_all_configurations.csv``. Only the
    per-configuration columns of affected rows are rebuilt. Optional review
    metadata is left unchanged and is not used by the scorer.
    """
    outputs = {model: read_model_output(results_dir, model) for model in models}
    audit = find_curated_components(
        outputs,
        override_path=override_path,
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
        from analysis.name_resolution_replay import (
            RecordedResolver,
            clean_name,
        )
    except ModuleNotFoundError:
        from name_resolution_replay import (  # type: ignore[no-redef]
            RecordedResolver,
            clean_name,
        )

    if requery_log is None:
        raise ValueError("Automatic sensitivity analysis requires an explicit requery_log recording.")
    outputs = {model: read_model_output(results_dir, model) for model in models}
    audit = find_curated_components(outputs, override_path=override_path, override_scope="all")
    resolver = RecordedResolver(repository_path(requery_log))
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
    override_scope: str = "all",
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rebuilt review, audit); see ``ablate`` for the rebuilt outputs."""
    review, audit, _outputs = ablate(
        review,
        models,
        override_path=override_path,
        override_scope=override_scope,
        results_dir=results_dir,
    )
    return review, audit
