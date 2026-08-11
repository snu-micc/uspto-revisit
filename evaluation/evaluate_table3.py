"""Evaluate denoising heuristics against the finalized reaction ground truth."""

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

MODELS = (
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
HEURISTICS = (
    "many_products",
    "many_reactants",
    "many_reagents",
    "no_reagent",
    "rare_template",
)
HEURISTIC_LABELS = {
    "many_products": "Many-products",
    "many_reactants": "Many-reactants",
    "many_reagents": "Many-reagents",
    "no_reagent": "No-reagent",
    "rare_template": "Template-anomaly",
}
COMPONENT_FIELDS = ("reactants", "reagents", "products")


def is_blank(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or not str(value).strip()
    )


def parse_bool(value: Any) -> bool:
    """Parse the public ground-truth Boolean without truthy-string ambiguity."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(int(value))
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected a Boolean value, found: {value!r}")


def parse_list(value: Any) -> list[Any]:
    if is_blank(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return [value]
    return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]


def parse_components(value: Any) -> list[str]:
    return [str(item).strip() for item in parse_list(value) if str(item).strip()]


def split_reaction(reaction: str) -> tuple[list[str], list[str], list[str]]:
    parts = str(reaction).strip().split(">")
    if len(parts) == 2:
        left, products = parts
        middle = ""
    elif len(parts) == 3:
        left, middle, products = parts
    else:
        raise ValueError("Reaction SMILES must contain two or three sections.")

    def split(section: str) -> list[str]:
        return [item.strip() for item in section.split(".") if item.strip()]

    return split(left), split(middle), split(products)


def comparison_smiles(smiles: str) -> str | None:
    """Canonicalize while ignoring atom maps, stereo, charge, and protonation."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    molecule = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    molecule.UpdatePropertyCache(False)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def normalized_section(
    components: list[str],
    *,
    as_set: bool,
) -> tuple[str, ...] | None:
    normalized = [comparison_smiles(smiles) for smiles in components]
    if any(smiles is None for smiles in normalized):
        return None
    values = [str(smiles) for smiles in normalized]
    return tuple(sorted(set(values) if as_set else values))


def component_signature(
    reactants: list[str],
    reagents: list[str],
    products: list[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    if not products:
        return None
    normalized = (
        normalized_section(reactants, as_set=False),
        normalized_section(reagents, as_set=True),
        normalized_section(products, as_set=True),
    )
    if any(section is None for section in normalized):
        return None
    return normalized  # type: ignore[return-value]


def signature_text(
    signature: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None,
) -> str:
    if signature is None:
        return ""
    return json.dumps(
        {
            field: list(section)
            for field, section in zip(COMPONENT_FIELDS, signature)
        },
        sort_keys=True,
    )


def component_columns(prefix: str) -> dict[str, str]:
    return {
        field: f"{prefix}{field}_smiles"
        for field in COMPONENT_FIELDS
    }


def components_from_row(row: pd.Series, prefix: str) -> dict[str, list[str]]:
    return {
        field: parse_components(row.get(column, ""))
        for field, column in component_columns(prefix).items()
    }


def signature_from_row(
    row: pd.Series,
    prefix: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    components = components_from_row(row, prefix)
    return component_signature(**components)


def build_ground_truth_manifest(review: pd.DataFrame) -> pd.DataFrame:
    if "idx" not in review or "ground_truth_valid" not in review:
        raise ValueError(
            "Ground-truth CSV must contain idx and ground_truth_valid columns."
        )
    if review["idx"].duplicated().any():
        raise ValueError("Review CSV contains duplicate idx values.")

    rows = []
    for _, row in review.sort_values("idx").iterrows():
        valid = parse_bool(row["ground_truth_valid"])
        if valid:
            first_signature = signature_from_row(row, "ground_truth_")
            second_signature = signature_from_row(row, "ground_truth_2_")
            source = "human_validated"
        else:
            first_signature = None
            second_signature = None
            source = "no_valid_ground_truth"

        if valid and first_signature is None:
            raise ValueError(
                f"Record {row['idx']} is True but has no valid ground truth."
            )
        accepted = [
            signature
            for signature in (first_signature, second_signature)
            if signature is not None
        ]
        rows.append(
            {
                "source_row": int(row["idx"]),
                "ground_truth_valid": valid,
                "ground_truth_source": source,
                "ground_truth_count": len(accepted),
                "ground_truth_1_signature": signature_text(first_signature),
                "ground_truth_2_signature": signature_text(second_signature),
                "review_note": str(row.get("review_note", "") or ""),
                "_accepted_signatures": accepted,
            }
        )
    return pd.DataFrame(rows)


def is_noise_free_paragraph(ground_truth_valid: Any) -> bool:
    """Return the fixed paragraph-level label used by Table 3."""
    return parse_bool(ground_truth_valid)


def atom_maps(molecule: Chem.Mol) -> set[int]:
    return {
        atom.GetAtomMapNum()
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }


def mapped_component_info(smiles: str) -> tuple[set[int], str | None, bool]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return set(), None, False
    maps = atom_maps(molecule)
    all_heavy_atoms_mapped = all(
        atom.GetAtomicNum() == 1 or atom.GetAtomMapNum() > 0
        for atom in molecule.GetAtoms()
    )
    return maps, comparison_smiles(smiles), all_heavy_atoms_mapped


def evaluate_mapped_step(
    reaction: str,
    *,
    template_missing: bool,
    mapping_error: str,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        left, middle, products = split_reaction(reaction)
    except ValueError as exc:
        return {
            "reactant_count": 0,
            "reagent_count": 0,
            "product_count": 0,
            "many_reactants": 0,
            "many_products": 0,
            "many_reagents": 0,
            "no_reagent": 0,
            "template_failure": 1,
            "missing_product_atom_maps": 1,
            "identical_structures": 0,
            "error": str(exc),
        }

    if not left:
        errors.append("missing left-side components")
    if not products:
        errors.append("missing products")

    product_maps: set[int] = set()
    product_structures: list[str] = []
    product_atoms_complete = bool(products)
    structures_valid = True
    for smiles in products:
        maps, canonical, atoms_complete = mapped_component_info(smiles)
        product_maps.update(maps)
        product_atoms_complete &= atoms_complete
        if canonical is None:
            structures_valid = False
        else:
            product_structures.append(canonical)

    reactants: list[str] = []
    reagents = list(middle)
    left_maps: set[int] = set()
    reactant_structures: list[str] = []
    for smiles in left:
        maps, canonical, _atoms_complete = mapped_component_info(smiles)
        left_maps.update(maps)
        if canonical is None:
            structures_valid = False
            continue
        if maps & product_maps:
            reactants.append(smiles)
            reactant_structures.append(canonical)
        else:
            reagents.append(smiles)

    missing_product_maps = bool(
        products
        and (
            not product_atoms_complete
            or not product_maps
            or not product_maps.issubset(left_maps)
        )
    )
    if missing_product_maps:
        errors.append("product atom mapping is incomplete")
    if template_missing:
        errors.append("template extraction failed")
    if mapping_error.strip():
        errors.append(mapping_error.strip())
    if not structures_valid:
        errors.append("invalid SMILES")
    if not reactants:
        errors.append("mapped reactants unavailable")

    identical = bool(
        reactant_structures
        and Counter(reactant_structures) == Counter(product_structures)
    )
    template_failure = bool(errors)
    return {
        "reactant_count": len(reactants),
        "reagent_count": len(reagents),
        "product_count": len(products),
        "many_reactants": int(len(reactants) > 2),
        "many_products": int(len(products) > 1),
        "many_reagents": int(len(reagents) > 5),
        "no_reagent": int(bool(reactants and products) and len(reagents) == 0),
        "template_failure": int(template_failure),
        "missing_product_atom_maps": int(missing_product_maps),
        "identical_structures": int(identical),
        "error": " | ".join(dict.fromkeys(errors)),
    }


def evaluate_unmapped_step(reaction: Any, error: str) -> dict[str, Any]:
    product_count = 0
    try:
        _left, _middle, products = split_reaction(str(reaction))
        product_count = len(products)
    except ValueError:
        pass
    return {
        "reactant_count": 0,
        "reagent_count": 0,
        "product_count": product_count,
        "many_reactants": 0,
        "many_products": int(product_count > 1),
        "many_reagents": 0,
        "no_reagent": 0,
        "template_failure": 1,
        "missing_product_atom_maps": 1,
        "identical_structures": 0,
        "error": error or "mapped reaction is unavailable",
    }


def value_at(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def evaluate_model_heuristics(
    result_row: pd.Series,
    *,
    model: str,
    issue_rare: bool,
    relevant_nosmi: bool,
) -> dict[str, Any]:
    raw_reactions = parse_list(result_row.get(f"{model}_rxn", ""))
    mapped_reactions = parse_list(result_row.get(f"{model}_mapped_rxn", ""))
    templates = parse_list(result_row.get(f"{model}_mapping_template", ""))
    mapping_errors = parse_list(result_row.get(f"{model}_mapping_error", ""))
    step_count = max(
        len(raw_reactions),
        len(mapped_reactions),
        len(templates),
        len(mapping_errors),
        1,
    )

    steps = []
    for index in range(step_count):
        mapped = value_at(mapped_reactions, index)
        mapping_error = value_at(mapping_errors, index)
        error_text = "" if is_blank(mapping_error) else str(mapping_error)
        if not is_blank(mapped):
            steps.append(
                evaluate_mapped_step(
                    str(mapped),
                    template_missing=is_blank(value_at(templates, index)),
                    mapping_error=error_text,
                )
            )
        else:
            steps.append(
                evaluate_unmapped_step(
                    value_at(raw_reactions, index),
                    error_text,
                )
            )

    result: dict[str, Any] = {
        "reaction_count": len(steps),
        "max_reactant_count": max(step["reactant_count"] for step in steps),
        "max_reagent_count": max(step["reagent_count"] for step in steps),
        "max_product_count": max(step["product_count"] for step in steps),
        "template_failure": int(
            relevant_nosmi or any(step["template_failure"] for step in steps)
        ),
        "missing_product_atom_maps": int(
            any(step["missing_product_atom_maps"] for step in steps)
        ),
        "identical_structures": int(
            any(step["identical_structures"] for step in steps)
        ),
        "evaluation_error": " | ".join(
            dict.fromkeys(step["error"] for step in steps if step["error"])
        ),
    }
    result["many_reactants"] = int(
        any(step["many_reactants"] for step in steps)
    )
    result["many_products"] = int(any(step["many_products"] for step in steps))
    result["many_reagents"] = int(any(step["many_reagents"] for step in steps))
    result["no_reagent"] = int(any(step["no_reagent"] for step in steps))
    result["rare_template"] = int(
        issue_rare
        or result["template_failure"]
        or result["identical_structures"]
    )
    return result


def infer_model(path: Path, frame: pd.DataFrame) -> str:
    suffix = "_reaction_smiles_final.csv"
    model = path.name.removesuffix(suffix)
    required = (
        f"{model}_rxn",
        f"{model}_mapped_rxn",
        f"{model}_mapping_template",
        f"{model}_mapping_error",
    )
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{path.name} is missing: {', '.join(missing)}")
    return model


def build_flags(
    review: pd.DataFrame,
    manifest: pd.DataFrame,
    results_dir: Path,
    models: tuple[str, ...],
) -> pd.DataFrame:
    review_by_idx = review.set_index("idx", drop=False)
    manifest_by_idx = manifest.set_index("source_row", drop=False)
    rows = []

    for model in models:
        path = results_dir / f"{model}_reaction_smiles_final.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        result = pd.read_csv(path, encoding="utf-8-sig").fillna("")
        inferred = infer_model(path, result)
        if inferred != model:
            raise ValueError(f"Model mismatch for {path.name}: {inferred}")
        if len(result) != len(review):
            raise ValueError(
                f"{path.name} has {len(result)} rows; expected {len(review)}."
            )

        for source_row, result_row in result.iterrows():
            if source_row not in review_by_idx.index:
                raise ValueError(f"Review CSV has no idx {source_row}.")
            review_row = review_by_idx.loc[source_row]
            ground_truth = manifest_by_idx.loc[source_row]
            model_components = components_from_row(review_row, f"{model}_")
            model_signature = component_signature(**model_components)
            relevant_nosmi = bool(
                int(review_row.get(f"{model}_has_unresolved_nosmi", 0) or 0)
            )
            evaluated = evaluate_model_heuristics(
                result_row,
                model=model,
                issue_rare=bool(int(review_row.get("rare_template_input", 0) or 0)),
                relevant_nosmi=relevant_nosmi,
            )

            accepted = ground_truth["_accepted_signatures"]
            matched_slot = ""
            if (
                ground_truth["ground_truth_valid"]
                and model_signature is not None
                and not relevant_nosmi
                and not evaluated["template_failure"]
            ):
                for slot, signature in enumerate(accepted, start=1):
                    if model_signature == signature:
                        matched_slot = str(slot)
                        break
            # Table 3 evaluates paragraph-level denoising. The human-validated
            # Boolean label is fixed and shared by every model.
            noise_free = is_noise_free_paragraph(
                ground_truth["ground_truth_valid"]
            )
            noise_reason = "" if noise_free else "no_valid_ground_truth"

            if not ground_truth["ground_truth_valid"]:
                model_output_issue = "not_applicable_no_valid_ground_truth"
            elif relevant_nosmi:
                model_output_issue = "NoSmi_in_skeleton"
            elif evaluated["template_failure"]:
                model_output_issue = "mapping_or_template_failure"
            elif model_signature is None:
                model_output_issue = "invalid_model_components"
            elif not matched_slot:
                model_output_issue = "ground_truth_mismatch"
            else:
                model_output_issue = ""

            row = {
                "source_row": source_row,
                "model": model,
                "title": str(review_row.get("title", "") or ""),
                "ground_truth_valid": bool(ground_truth["ground_truth_valid"]),
                "ground_truth_source": ground_truth["ground_truth_source"],
                "ground_truth_count": ground_truth["ground_truth_count"],
                "matched_ground_truth": matched_slot,
                "noise_label": "noise-free" if noise_free else "noisy",
                "noise_free": int(noise_free),
                "noise_reason": noise_reason,
                "model_output_matches_ground_truth": int(bool(matched_slot)),
                "model_output_issue": model_output_issue,
                "model_signature": signature_text(model_signature),
                "relevant_nosmi": int(relevant_nosmi),
                **evaluated,
            }
            for heuristic in HEURISTICS:
                row[f"{heuristic}_filtered"] = int(evaluated[heuristic])
                row[f"{heuristic}_passed"] = int(not evaluated[heuristic])
            rows.append(row)
    return pd.DataFrame(rows)


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_metrics(flags: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    records = []
    for model in models:
        model_rows = flags.loc[flags["model"].eq(model)]
        noise_free = model_rows["noise_free"].astype(bool)
        noisy = ~noise_free
        for heuristic in HEURISTICS:
            filtered = model_rows[f"{heuristic}_filtered"].astype(bool)
            passed = ~filtered
            tp = int((noise_free & passed).sum())
            fn = int((noise_free & filtered).sum())
            fp = int((noisy & passed).sum())
            tn = int((noisy & filtered).sum())
            total = tp + fn + fp + tn
            if total != len(model_rows):
                raise AssertionError(f"{model}/{heuristic}: confusion total mismatch")
            records.append(
                {
                    "model": model,
                    "heuristic": heuristic,
                    "heuristic_type": HEURISTIC_LABELS[heuristic],
                    "noise_free_reactions": tp + fn,
                    "noisy_reactions": fp + tn,
                    "total_reactions": total,
                    "passed_tp": tp,
                    "filtered_fn": fn,
                    "passed_fp": fp,
                    "filtered_tn": tn,
                    "noise_free_data_saved_rate": safe_divide(tp, tp + fn),
                    "noisy_data_filtered_rate": safe_divide(tn, tn + fp),
                    "denoising_accuracy": safe_divide(tp + tn, total),
                }
            )
    return pd.DataFrame(records)


def publication_table(metrics: pd.DataFrame, model: str) -> pd.DataFrame:
    selected = metrics.loc[metrics["model"].eq(model)].set_index("heuristic")
    if len(selected) != len(HEURISTICS):
        raise ValueError(f"Metrics are incomplete for {model}.")
    noise_free = int(selected.iloc[0]["noise_free_reactions"])
    noisy = int(selected.iloc[0]["noisy_reactions"])
    total = int(selected.iloc[0]["total_reactions"])
    rows = []
    for heuristic in HEURISTICS:
        row = selected.loc[heuristic]
        rows.append(
            [
                HEURISTIC_LABELS[heuristic],
                int(row["passed_tp"]),
                int(row["filtered_fn"]),
                int(row["passed_fp"]),
                int(row["filtered_tn"]),
                f"{row['noise_free_data_saved_rate'] * 100:.4f}",
                f"{row['noisy_data_filtered_rate'] * 100:.4f}",
                f"{row['denoising_accuracy'] * 100:.4f}",
            ]
        )
    columns = pd.MultiIndex.from_tuples(
        [
            ("Heuristic type", ""),
            (f"Noise-free reactions ({noise_free})", "Passed (TP)"),
            (f"Noise-free reactions ({noise_free})", "Filtered (FN)"),
            (f"Noisy reactions ({noisy})", "Passed (FP)"),
            (f"Noisy reactions ({noisy})", "Filtered (TN)"),
            (f"All reactions ({total})", "Noise-free data saved rate (%)"),
            (f"All reactions ({total})", "Noisy data filtered rate (%)"),
            (f"All reactions ({total})", "Denoising accuracy (%)"),
        ]
    )
    return pd.DataFrame(rows, columns=columns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review",
        default="evaluation/benchmark_all_configurations.csv",
        help="Finalized ground-truth review CSV.",
    )
    parser.add_argument("--results-dir", default="result/model_outputs")
    parser.add_argument("--output-dir", default="evaluation/results/table3")
    parser.add_argument("--primary-model", default="gemini-3.6-flash-high")
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    models = tuple(args.models)
    if args.primary_model not in models:
        raise ValueError("The primary model must be included in --models.")

    review = pd.read_csv(args.review, encoding="utf-8-sig").fillna("")
    review["idx"] = review["idx"].astype(int)
    manifest = build_ground_truth_manifest(review)
    flags = build_flags(review, manifest, Path(args.results_dir), models)
    metrics = build_metrics(flags, models)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flags.to_csv(
        output_dir / "table3_flags.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        output_dir / "table3_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    publication_table(metrics, args.primary_model).to_csv(
        output_dir / "table3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Evaluated {len(models)} models across {len(review)} finalized rows.")
    print(f"Primary manuscript-format table: {output_dir / 'table3.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
