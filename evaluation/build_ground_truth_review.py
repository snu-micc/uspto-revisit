"""Build a human-review queue for reaction-SMILES ground truth annotation."""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


DEFAULT_MODELS = (
    "qwen-3.5-9b-off",
    "qwen-3.5-9b-low",
    "qwen-3.5-9b-high",
    "qwen-3.5-9b-ft-off",
    "qwen-3.5-9b-ft-low",
    "qwen-3.5-9b-ft-high",
    "gpt-4.1-mini-nonreasoning-shared",
    "gpt-4.1-mini-ft-nonreasoning-shared",
    "gpt-5.4-none",
    "gpt-5.4-low",
    "gpt-5.4-high",
    "gpt-5.6-sol-none",
    "gpt-5.6-sol-low",
    "gpt-5.6-sol-high",
    "gemini-2.5-flash-none",
    "gemini-3.1-pro-preview-low",
    "gemini-3.1-pro-preview-high",
    "gemini-3.6-flash-low",
    "gemini-3.6-flash-high",
)
DEFAULT_INPUT_PATH = Path("examples/input.csv")
REVIEW_COLUMNS = (
    "ground_truth_valid",
    "ground_truth_reactants_smiles",
    "ground_truth_reagents_smiles",
    "ground_truth_products_smiles",
    "ground_truth_reaction_smiles",
    "ground_truth_2_reactants_smiles",
    "ground_truth_2_reagents_smiles",
    "ground_truth_2_products_smiles",
    "ground_truth_2_reaction_smiles",
    "review_note",
)
FINALIZED_CANDIDATE_COLUMNS = (
    "candidate_reactants_smiles",
    "candidate_reagents_smiles",
    "candidate_products_smiles",
    "candidate_reaction_smiles",
    "product_mapping_complete",
    "rare_template_input",
)


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip()


def parse_list(value: Any) -> list[str]:
    if is_blank(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return [str(value)]
    return [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]


def unresolved_nosmi_compounds(value: Any) -> list[str]:
    """Extract unresolved named compounds retained in the model's SMILES dictionary."""
    if is_blank(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = value
    values = parsed.values() if isinstance(parsed, dict) else parse_list(parsed)
    return [str(item) for item in values if "nosmi" in str(item).lower()]


def unresolved_nosmi_labels(value: Any) -> dict[str, str]:
    """Return entity labels for unresolved compounds, e.g. ``G(NoSmi)``."""
    if is_blank(value):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(label): str(name) for label, name in parsed.items() if "nosmi" in str(name).lower()}


def product_labels(raw_output: str) -> set[str]:
    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()
    products = parsed.get("Products", {}) if isinstance(parsed, dict) else {}
    return {str(label) for label in products} if isinstance(products, dict) else set()


def display_skeleton(skeleton: Any, raw_output: str, nosmi_labels: dict[str, str]) -> str:
    """Keep unresolved labels visible in skeletons instead of silently dropping them."""
    unresolved_products = product_labels(raw_output) & set(nosmi_labels)
    annotated = []
    for reaction in parse_list(skeleton):
        sections = reaction.split(">")
        for index, section in enumerate(sections):
            if not section and index == len(sections) - 1 and unresolved_products:
                sections[index] = ".".join(f"{label}(NoSmi)" for label in sorted(unresolved_products))
                continue
            sections[index] = ".".join(
                f"{label}(NoSmi)" if label in nosmi_labels else label
                for label in section.split(".") if label
            )
        annotated.append(">".join(sections))
    return str(annotated)


def skeleton_compound_labels(skeleton: Any) -> set[str]:
    """Return only entity labels that participate in the reaction skeleton."""
    labels: set[str] = set()
    for reaction in parse_list(skeleton):
        for section in reaction.split(">"):
            for token in section.split("."):
                label = token.split("(", 1)[0].strip()
                if label:
                    labels.add(label)
    return labels


def split_reaction(reaction: str) -> tuple[list[str], list[str], list[str]]:
    parts = reaction.split(">")
    if len(parts) == 2:
        left, products = parts
        middle = ""
    elif len(parts) == 3:
        left, middle, products = parts
    else:
        return [], [], []
    split = lambda section: [item for item in section.split(".") if item]
    return split(left), split(middle), split(products)


def unmapped_smiles(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    molecule = Chem.Mol(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def mapped_components(
    mapped_reactions: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str], bool]:
    """Classify mapped left-side components by atom-map overlap with products."""
    reactants: list[str] = []
    reagents: list[str] = []
    products: list[str] = []
    mapped_reactants: list[str] = []
    mapped_reagents: list[str] = []
    mapped_products: list[str] = []
    complete = bool(mapped_reactions)
    for reaction in mapped_reactions:
        left, _, step_products = split_reaction(reaction)
        if not left or not step_products:
            complete = False
            continue
        product_maps: set[int] = set()
        normalized_products: list[str] = []
        for item in step_products:
            molecule = Chem.MolFromSmiles(item)
            normalized = unmapped_smiles(item)
            if molecule is None or normalized is None:
                complete = False
                continue
            product_maps.update(atom.GetAtomMapNum() for atom in molecule.GetAtoms() if atom.GetAtomMapNum())
            normalized_products.append(normalized)
            mapped_products.append(item)
        if not product_maps:
            complete = False
        products.extend(normalized_products)
        for item in left:
            molecule = Chem.MolFromSmiles(item)
            normalized = unmapped_smiles(item)
            if molecule is None or normalized is None:
                complete = False
                continue
            maps = {atom.GetAtomMapNum() for atom in molecule.GetAtoms() if atom.GetAtomMapNum()}
            if maps & product_maps:
                reactants.append(normalized)
                mapped_reactants.append(item)
            else:
                reagents.append(normalized)
                mapped_reagents.append(item)
    return (
        reactants,
        reagents,
        products,
        mapped_reactants,
        mapped_reagents,
        mapped_products,
        complete,
    )


def reaction_signature(reactions: list[str]) -> str | None:
    """Map-independent canonical signature used only to group model candidates."""
    normalized_steps = []
    for reaction in reactions:
        left, middle, products = split_reaction(reaction)
        if not left or not products:
            return None
        sections = []
        for section in (left, middle, products):
            normalized = [unmapped_smiles(item) for item in section]
            if any(item is None for item in normalized):
                return None
            sections.append(".".join(sorted(normalized)))
        normalized_steps.append(">".join(sections))
    return " || ".join(sorted(normalized_steps)) if normalized_steps else None


def structure_for_comparison(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles
    molecule = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(molecule)
    for atom in molecule.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    molecule.UpdatePropertyCache(False)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def unique_product_components(products: list[str]) -> list[str]:
    """Keep one product representative per stereo-agnostic structure."""
    unique = []
    seen = set()
    for smiles in products:
        identity = structure_for_comparison(smiles)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(smiles)
    return unique


def component_signature(
    reactants: list[str], reagents: list[str], products: list[str]
) -> str | None:
    """Compare the mapped-and-classified components shown to the reviewer.

    Some extracted raw reaction strings contain extra '>' delimiters, even when
    the local mapper recovered valid components.  The review agreement should
    reflect those recoverable components rather than reject the raw string.
    """
    if not products:
        return None

    normalized_reactants = sorted(structure_for_comparison(smiles) for smiles in reactants)
    normalized_reagents = sorted({structure_for_comparison(smiles) for smiles in reagents})
    normalized_products = sorted({structure_for_comparison(smiles) for smiles in products})
    return ">".join(
        ".".join(section)
        for section in (normalized_reactants, normalized_reagents, normalized_products)
    )


def reaction_from_parts(reactants: list[str], reagents: list[str], products: list[str]) -> str:
    return f"{'.'.join(reactants)}>{'.'.join(reagents)}>{'.'.join(products)}"


def lowe_signature(value: Any) -> str | None:
    """Build the same comparison signature for a mapped Lowe reaction."""
    reaction = str(value).strip().split(maxsplit=1)[0]
    parts = reaction.split(">")
    if len(parts) != 3:
        return None
    left, middle, right = ([item for item in part.split(".") if item] for part in parts)
    product_maps: set[int] = set()
    products = []
    for item in right:
        molecule = Chem.MolFromSmiles(item)
        normalized = unmapped_smiles(item)
        if molecule is None or normalized is None:
            return None
        product_maps.update(atom.GetAtomMapNum() for atom in molecule.GetAtoms() if atom.GetAtomMapNum())
        products.append(normalized)
    if not product_maps:
        return None
    reactants, reagents = [], []
    for item in left + middle:
        molecule = Chem.MolFromSmiles(item)
        normalized = unmapped_smiles(item)
        if molecule is None or normalized is None:
            return None
        atom_maps = {atom.GetAtomMapNum() for atom in molecule.GetAtoms() if atom.GetAtomMapNum()}
        (reactants if atom_maps & product_maps else reagents).append(normalized)
    return component_signature(reactants, reagents, products)


def model_row(record: pd.Series, prefix: str) -> dict[str, Any]:
    raw_output = str(record.get("prediction", "") or "")
    nosmi_labels = unresolved_nosmi_labels(record.get(f"{prefix}_smiles", ""))
    skeleton = display_skeleton(
        record.get(f"{prefix}_skeleton", ""),
        raw_output,
        nosmi_labels,
    )
    unresolved_skeleton_labels = set(nosmi_labels) & skeleton_compound_labels(skeleton)
    has_unresolved_nosmi = bool(unresolved_skeleton_labels)
    reactions = parse_list(record.get(f"{prefix}_rxn"))
    mapped = parse_list(record.get(f"{prefix}_mapped_rxn"))
    mapping_errors = parse_list(record.get(f"{prefix}_mapping_error"))
    (
        reactants,
        reagents,
        products,
        mapped_reactants,
        mapped_reagents,
        mapped_products,
        mapped_complete,
    ) = mapped_components(mapped)
    products = unique_product_components(products)
    product_mapping_complete = bool(
        mapped_complete
        and len(mapped) == len(reactions)
        and products
        and not has_unresolved_nosmi
        and not any(error.strip() for error in mapping_errors)
    )
    return {
        "raw_output": raw_output,
        "generation_error": str(record.get("error", "") or ""),
        "skeleton": skeleton,
        "unresolved_nosmi": unresolved_nosmi_compounds(record.get(f"{prefix}_smiles", "")),
        "unresolved_nosmi_in_skeleton": sorted(unresolved_skeleton_labels),
        "reactions": reactions,
        "mapped_reactions": mapped,
        "signature": (
            None
            if has_unresolved_nosmi
            else component_signature(reactants, reagents, products)
        ),
        "has_unresolved_nosmi": has_unresolved_nosmi,
        "reactants": reactants,
        "reagents": reagents,
        "products": products,
        "mapped_reactants": mapped_reactants,
        "mapped_reagents": mapped_reagents,
        "mapped_products": mapped_products,
        "product_mapping_complete": product_mapping_complete,
        "mapping_error": " | ".join(error for error in mapping_errors if error.strip()),
    }


def issue_is_rare(value: Any) -> bool:
    return "rare-template" in str(value).lower().replace("_", "-")


def load_existing(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "idx" not in frame:
        return {}
    return {int(row.idx): row._asdict() for row in frame.itertuples(index=False)}


def build_queue(
    result_dir: Path,
    output: Path,
    models: tuple[str, ...],
    input_path: Path = DEFAULT_INPUT_PATH,
    ground_truth_path: Path = Path("evaluation/ground_truth_review.csv"),
) -> pd.DataFrame:
    frames: dict[str, pd.DataFrame] = {}
    for model in models:
        path = result_dir / f"{model}_reaction_smiles_final.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frames[model] = pd.read_csv(path, encoding="utf-8-sig")

    row_count = len(next(iter(frames.values())))
    if any(len(frame) != row_count for frame in frames.values()):
        raise ValueError("All model output files must have the same number of rows.")
    lowe_values = (
        pd.read_csv(input_path, encoding="utf-8-sig").fillna("")["Lowe_smiles"].tolist()
        if input_path.is_file()
        else []
    )
    if lowe_values and len(lowe_values) != row_count:
        raise ValueError("Lowe input rows do not match model output rows.")
    existing = load_existing(ground_truth_path)
    rows = []
    for index in range(row_count):
        base = next(iter(frames.values())).iloc[index]
        candidates = {model: model_row(frame.iloc[index], model) for model, frame in frames.items()}
        groups: dict[str, list[str]] = {}
        for model, candidate in candidates.items():
            if candidate["signature"]:
                groups.setdefault(candidate["signature"], []).append(model)
        signature, agreeing_models = max(groups.items(), key=lambda item: len(item[1]), default=(None, []))
        agreement_groups = sorted(groups.values(), key=lambda members: (-len(members), members))
        disagreeing_models = [model for model in models if model not in agreeing_models]
        comparison_groups = [members.copy() for members in agreement_groups]
        source_signature = lowe_signature(lowe_values[index]) if lowe_values else None
        if source_signature:
            for group_signature, members in groups.items():
                if group_signature == source_signature:
                    comparison_groups[agreement_groups.index(members)].append("Lowe_smiles")
                    break
            else:
                comparison_groups.append(["Lowe_smiles"])
        comparison_groups.sort(key=lambda members: (-len(members), members))
        largest_comparison_group = comparison_groups[0] if comparison_groups else []
        comparison_disagreeing = [
            item
            for group in comparison_groups[1:]
            for item in group
            if item not in largest_comparison_group
        ]
        representative = candidates[agreeing_models[0]] if agreeing_models else {"reactants": [], "reagents": [], "products": []}
        all_products_mapped = bool(agreeing_models) and all(
            candidates[model]["product_mapping_complete"] for model in agreeing_models
        )
        rare_template_input = issue_is_rare(base.get("issue", ""))
        manual_required = int(
            len(agreeing_models) < 4 or not all_products_mapped or rare_template_input
        )
        row = {
            "idx": index,
            "title": base.get("title", ""),
            "paragraph": base.get("paragraph", ""),
            "candidate_models": json.dumps(agreeing_models),
            "model_agreement_count": len(agreeing_models),
            "model_agreement_groups": json.dumps(agreement_groups),
            "model_disagreeing_models": json.dumps(disagreeing_models),
            "reaction_comparison_groups": json.dumps(comparison_groups),
            "reaction_comparison_disagreeing": json.dumps(comparison_disagreeing),
            "candidate_reactants_smiles": json.dumps(representative["reactants"]),
            "candidate_reagents_smiles": json.dumps(representative["reagents"]),
            "candidate_products_smiles": json.dumps(representative["products"]),
            "candidate_reaction_smiles": reaction_from_parts(
                representative["reactants"], representative["reagents"], representative["products"]
            ) if agreeing_models else "",
            "product_mapping_complete": int(all_products_mapped),
            "rare_template_input": int(rare_template_input),
            "manual_review_required": manual_required,
            "automatic_consensus_candidate": int(not manual_required),
        }
        for model, candidate in candidates.items():
            row[f"{model}_raw_output"] = candidate["raw_output"]
            row[f"{model}_generation_error"] = candidate["generation_error"]
            row[f"{model}_skeleton"] = candidate["skeleton"]
            row[f"{model}_unresolved_nosmi"] = json.dumps(candidate["unresolved_nosmi"])
            row[f"{model}_unresolved_nosmi_in_skeleton"] = json.dumps(
                candidate["unresolved_nosmi_in_skeleton"]
            )
            row[f"{model}_has_unresolved_nosmi"] = int(candidate["has_unresolved_nosmi"])
            row[f"{model}_mapped_reactions"] = json.dumps(candidate["mapped_reactions"])
            row[f"{model}_reactants_smiles"] = json.dumps(candidate["reactants"])
            row[f"{model}_reagents_smiles"] = json.dumps(candidate["reagents"])
            row[f"{model}_products_smiles"] = json.dumps(candidate["products"])
            row[f"{model}_mapped_reactants_smiles"] = json.dumps(candidate["mapped_reactants"])
            row[f"{model}_mapped_reagents_smiles"] = json.dumps(candidate["mapped_reagents"])
            row[f"{model}_mapped_products_smiles"] = json.dumps(candidate["mapped_products"])
            row[f"{model}_product_mapping_complete"] = int(candidate["product_mapping_complete"])
            row[f"{model}_mapping_error"] = candidate["mapping_error"]
        prior = existing.get(index, {})
        if str(prior.get("ground_truth_valid", "")).strip().lower() in {
            "true",
            "false",
            "1",
            "0",
        }:
            for column in FINALIZED_CANDIDATE_COLUMNS:
                if column in prior:
                    row[column] = prior[column]
        for column in REVIEW_COLUMNS:
            row[column] = prior.get(column, "")
        rows.append(row)
    queue = pd.DataFrame(rows)
    queue = queue.sort_values(
        ["manual_review_required", "product_mapping_complete", "model_agreement_count", "idx"],
        ascending=[False, True, True, True],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output, index=False, encoding="utf-8-sig")
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path, default=Path("result/model_outputs")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/benchmark_all_configurations.csv"),
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("evaluation/ground_truth_review.csv"),
        help="Human-validated Boolean ground-truth CSV.",
    )
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    args = parser.parse_args()
    queue = build_queue(
        args.result_dir,
        args.output,
        tuple(args.models),
        args.input,
        args.ground_truth,
    )
    print(f"Wrote {len(queue)} review rows to {args.output}")


if __name__ == "__main__":
    main()
