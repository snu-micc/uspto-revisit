"""Audit finalized reaction ground truths with structure and atom-map checks."""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")

MODELS = (
    "gpt-4.1-mini",
    "gpt_4.1_mini_ft",
    "gpt-5.4",
    "gpt-5.6-sol",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash-lite",
)

CHEMIST_FINDINGS: dict[tuple[int, int], tuple[str, str, str]] = {
    (4, 1): (
        "incorrect_product_structure",
        "retain ground_truth_valid=True and replace the product SMILES",
        "The stated sulfonylation should retain the benzhydryl ester, but the current product is split into a carboxylate and benzhydrol; one product oxygen is consequently unmapped.",
    ),
    (10, 1): (
        "incorrect_atom_source_and_roles",
        "revise the validated ground truth",
        "Ethanolic HCl forms the product hydrochloride, but HCl is absent and the mapping instead sources chloride from dichloromethane; the amino alcohol is also duplicated across roles.",
    ),
    (14, 1): (
        "unrepresentable_product",
        "set ground_truth_valid=False",
        "The reported product is a thiourea-terminated PAMAM dendrimer. The current small-molecule thiourea omits the dendrimer and incorrectly maps a buffer amine into the product.",
    ),
    (33, 1): (
        "flattened_multistep",
        "split into two stepwise reactions or exclude from single-reaction mapping",
        "O-alkylation and subsequent oxazoline hydrolysis/HCl-salt formation are flattened, with the intermediate appearing on both sides.",
    ),
    (42, 1): (
        "unrepresentable_polymer_product",
        "set ground_truth_valid=False",
        "The paragraph reports copolymerization, while the current product is merely the unchanged styrene and methylstyrene monomers; no polymeric transformation is encoded.",
    ),
    (73, 1): (
        "missing_counterion_source",
        "retain ground_truth_valid=True and include HCl in the mapped inputs",
        "The hydrochloride product contains chloride, but ethanolic HCl from the paragraph is omitted from the reaction inputs, leaving product chloride unmapped.",
    ),
    (96, 1): (
        "hydrolysis_atom_source_missing",
        "retain ground_truth_valid=True and make the record atom-map complete",
        "Nitrile hydrolysis is chemically plausible, but the aqueous HCl/water source of both carboxyl oxygens is absent, so the product cannot be completely atom-mapped.",
    ),
    (108, 1): (
        "flattened_multistep",
        "split into acetate formation and pyrolysis reactions",
        "The acetate intermediate is simultaneously an input and a product in one flattened record.",
    ),
    (116, 1): (
        "flattened_multistep",
        "split coupling and TFA deprotection",
        "The Boc-protected coupling product and the deprotected product are combined as products of one reaction.",
    ),
    (177, 1): (
        "flattened_multistep",
        "retain only the oxidation step or store both steps separately",
        "Precursor formation and periodate oxidation are combined, with the sulfone intermediate on both sides.",
    ),
    (180, 1): (
        "flattened_multistep",
        "store the four transformations stepwise or exclude from single-reaction mapping",
        "Methylenedioxy formation, lithiation/SO2 trapping, chlorination, and ammonolysis are not represented as separate reactions.",
    ),
    (220, 1): (
        "flattened_multistep",
        "split reductive amination and hydrochloride-salt formation",
        "The free-base intermediate is present on both sides and the subsequent HCl treatment is merged into the same reaction.",
    ),
    (242, 1): (
        "flattened_multistep",
        "split mesyl-azide preparation and diazo transfer",
        "Methanesulfonyl azide is encoded as both an input and a product because two reactions were flattened.",
    ),
    (273, 2): (
        "invalid_alternative_flattened_multistep",
        "remove ground-truth alternative 2; retain or refine the net ground truth 1",
        "Diazotization, azide substitution, and cyclization intermediates are all placed on both sides of one reaction. Ground truth 1 already provides a map-compatible net transformation.",
    ),
    (285, 1): (
        "mapping_artifact_only",
        "retain ground_truth_valid=True and review the atom map",
        "The reaction is a chemically valid carboxylic-acid reduction. LocalMapper incorrectly assigns the product alcohol oxygen to isobutyl chloroformate instead of the substrate carboxyl group.",
    ),
    (296, 1): (
        "flattened_multistep",
        "split enaminone formation and quinazoline cyclization",
        "The DMF-DMA intermediate is both an input and product in the flattened reaction.",
    ),
    (297, 1): (
        "incorrect_reagent_set",
        "rewrite ground truth 1 using only the ketone and LiAlH4 inputs",
        "A chlorophenyl alcohol from the cited analogy example is incorrectly included as a reagent for the difluorophenyl substrate.",
    ),
    (297, 2): (
        "duplicated_alternative_products",
        "store the S and R alcohols as separate ground-truth alternatives",
        "One substrate molecule is mapped to two complete stereoisomeric product molecules in a single reaction, violating atom conservation.",
    ),
    (300, 1): (
        "flattened_multistep",
        "split acid-chloride formation and esterification",
        "The acid chloride intermediate is encoded on both the input and product sides.",
    ),
    (356, 1): (
        "flattened_multistep",
        "split demethylation and allylation",
        "The hydroxybenzothiazole intermediate appears on both sides of one flattened reaction.",
    ),
    (386, 1): (
        "flattened_multistep",
        "split chlorination, amination, and debenzylation",
        "Three distinct transformations and their intermediates are combined into one mapped reaction.",
    ),
}


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip()


def parse_list(value: Any) -> list[str]:
    if is_blank(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = value
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def split_reaction(value: str) -> tuple[list[str], list[str], list[str]]:
    parts = str(value).strip().split(">")
    if len(parts) == 2:
        left, products = parts
        middle = ""
    elif len(parts) == 3:
        left, middle, products = parts
    else:
        raise ValueError("reaction must contain two or three sections")
    split = lambda section: [item.strip() for item in section.split(".") if item.strip()]
    return split(left), split(middle), split(products)


def canonical_smiles(
    smiles: str,
    *,
    ignore_stereo: bool = False,
    relaxed: bool = False,
) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    molecule = Chem.Mol(molecule)
    if ignore_stereo or relaxed:
        Chem.RemoveStereochemistry(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
        if relaxed:
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)
    molecule.UpdatePropertyCache(False)
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=not (ignore_stereo or relaxed),
    )


def signature(
    reactants: list[str],
    reagents: list[str],
    products: list[str],
    *,
    relaxed: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    if not reactants or not products:
        return None
    sections: list[tuple[str, ...]] = []
    for index, components in enumerate((reactants, reagents, products)):
        normalized = [canonical_smiles(item, relaxed=relaxed) for item in components]
        if any(item is None for item in normalized):
            return None
        values = [str(item) for item in normalized]
        if index in (1, 2):
            values = list(set(values))
        sections.append(tuple(sorted(values)))
    return tuple(sections)  # type: ignore[return-value]


def model_components(row: pd.Series, model: str) -> tuple[list[str], list[str], list[str]]:
    return tuple(
        parse_list(row.get(f"{model}_{field}_smiles", ""))
        for field in ("reactants", "reagents", "products")
    )  # type: ignore[return-value]


def accepted_reactions(row: pd.Series) -> list[tuple[int, str]]:
    if str(row.get("ground_truth_valid", "")).strip().lower() not in {"true", "1"}:
        return []
    reactions = []
    for slot, column in (
        (1, "ground_truth_reaction_smiles"),
        (2, "ground_truth_2_reaction_smiles"),
    ):
        value = str(row.get(column, "") or "").strip()
        if value:
            reactions.append((slot, value))
    return reactions


def atom_counter(smiles_values: list[str]) -> Counter[int] | None:
    counts: Counter[int] = Counter()
    for smiles in smiles_values:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        counts.update(atom.GetAtomicNum() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1)
    return counts


def map_counts(molecules: list[Chem.Mol]) -> Counter[int]:
    return Counter(
        atom.GetAtomMapNum()
        for molecule in molecules
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetAtomMapNum() > 0
    )


def mapped_molecules(section: str) -> tuple[list[Chem.Mol], list[str]]:
    molecules: list[Chem.Mol] = []
    invalid: list[str] = []
    for smiles in [item for item in section.split(".") if item]:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            invalid.append(smiles)
        else:
            molecules.append(molecule)
    return molecules, invalid


def bond_inventory(molecules: list[Chem.Mol]) -> dict[tuple[int, int], float]:
    inventory: dict[tuple[int, int], float] = {}
    for molecule in molecules:
        for bond in molecule.GetBonds():
            first = bond.GetBeginAtom().GetAtomMapNum()
            second = bond.GetEndAtom().GetAtomMapNum()
            if first <= 0 or second <= 0:
                continue
            key = tuple(sorted((first, second)))
            inventory[key] = float(bond.GetBondTypeAsDouble())
    return inventory


def atom_inventory(molecules: list[Chem.Mol]) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for molecule in molecules:
        for atom in molecule.GetAtoms():
            atom_map = atom.GetAtomMapNum()
            if atom_map > 0:
                result[atom_map] = (atom.GetAtomicNum(), atom.GetFormalCharge())
    return result


def assign_left_roles(
    mapped_left: list[Chem.Mol],
    reactants: list[str],
    reagents: list[str],
) -> tuple[list[str], int]:
    reactant_counts = Counter(canonical_smiles(item) for item in reactants)
    reagent_counts = Counter(canonical_smiles(item) for item in reagents)
    roles: list[str] = []
    unmatched = 0
    for molecule in mapped_left:
        value = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        identity = canonical_smiles(value)
        if reactant_counts[identity] > 0:
            reactant_counts[identity] -= 1
            roles.append("reactant")
        elif reagent_counts[identity] > 0:
            reagent_counts[identity] -= 1
            roles.append("reagent")
        else:
            roles.append("unknown")
            unmatched += 1
    unmatched += sum(reactant_counts.values()) + sum(reagent_counts.values())
    return roles, unmatched


def mapping_audit(
    mapped_reaction: str,
    reactants: list[str],
    reagents: list[str],
    products: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mapped_reaction": mapped_reaction,
        "product_mapping_complete": 0,
        "duplicate_atom_maps": 0,
        "atom_element_mismatch": 0,
        "reagent_contributes_to_product": 0,
        "noncontributing_reactant": 0,
        "role_assignment_unmatched_components": 0,
        "formed_bonds": 0,
        "broken_bonds": 0,
        "bond_order_changes": 0,
        "leaving_group_bonds": 0,
        "atom_charge_changes": 0,
        "no_detectable_transformation": 0,
        "large_reaction_center": 0,
        "mapping_detail": "",
    }
    parts = mapped_reaction.split(">")
    if len(parts) == 2:
        left_text, product_text = parts
    elif len(parts) == 3:
        left_text = ".".join(parts[:2])
        product_text = parts[2]
    else:
        result["mapping_detail"] = "invalid mapped reaction format"
        return result

    left_molecules, invalid_left = mapped_molecules(left_text)
    product_molecules, invalid_products = mapped_molecules(product_text)
    if invalid_left or invalid_products or not left_molecules or not product_molecules:
        result["mapping_detail"] = "invalid or empty mapped structure"
        return result

    left_map_counts = map_counts(left_molecules)
    product_map_counts = map_counts(product_molecules)
    product_heavy_atoms = sum(
        atom.GetAtomicNum() > 1 for molecule in product_molecules for atom in molecule.GetAtoms()
    )
    complete = (
        len(product_map_counts) == product_heavy_atoms
        and all(count == 1 for count in product_map_counts.values())
        and set(product_map_counts).issubset(left_map_counts)
    )
    result["product_mapping_complete"] = int(complete)
    result["duplicate_atom_maps"] = int(
        any(count > 1 for count in left_map_counts.values())
        or any(count > 1 for count in product_map_counts.values())
    )

    left_atoms = atom_inventory(left_molecules)
    product_atoms = atom_inventory(product_molecules)
    shared_maps = set(left_atoms) & set(product_atoms)
    result["atom_element_mismatch"] = int(
        any(left_atoms[value][0] != product_atoms[value][0] for value in shared_maps)
    )
    result["atom_charge_changes"] = sum(
        left_atoms[value][1] != product_atoms[value][1] for value in shared_maps
    )

    product_maps = set(product_map_counts)
    roles, unmatched = assign_left_roles(left_molecules, reactants, reagents)
    result["role_assignment_unmatched_components"] = unmatched
    for molecule, role in zip(left_molecules, roles):
        component_maps = {
            atom.GetAtomMapNum()
            for atom in molecule.GetAtoms()
            if atom.GetAtomicNum() > 1 and atom.GetAtomMapNum() > 0
        }
        contributes = bool(component_maps & product_maps)
        if role == "reagent" and contributes:
            result["reagent_contributes_to_product"] += 1
        elif role == "reactant" and not contributes:
            result["noncontributing_reactant"] += 1

    left_bonds_all = bond_inventory(left_molecules)
    product_bonds = bond_inventory(product_molecules)
    left_bonds = {
        key: value
        for key, value in left_bonds_all.items()
        if key[0] in product_maps and key[1] in product_maps
    }
    leaving_group_bonds = 0
    for molecule in left_molecules:
        for bond in molecule.GetBonds():
            first = bond.GetBeginAtom().GetAtomMapNum()
            second = bond.GetEndAtom().GetAtomMapNum()
            if (first in product_maps) != (second in product_maps):
                leaving_group_bonds += 1
    result["leaving_group_bonds"] = leaving_group_bonds
    shared_bonds = set(left_bonds) & set(product_bonds)
    result["bond_order_changes"] = sum(
        left_bonds[key] != product_bonds[key] for key in shared_bonds
    )
    result["formed_bonds"] = len(set(product_bonds) - set(left_bonds))
    result["broken_bonds"] = len(set(left_bonds) - set(product_bonds)) + leaving_group_bonds
    center_size = (
        result["bond_order_changes"]
        + result["formed_bonds"]
        + result["broken_bonds"]
        + result["atom_charge_changes"]
    )
    result["no_detectable_transformation"] = int(center_size == 0)
    result["large_reaction_center"] = int(center_size > 8)

    details = []
    if not complete:
        details.append("incomplete product atom mapping")
    if result["duplicate_atom_maps"]:
        details.append("duplicate atom-map numbers")
    if result["atom_element_mismatch"]:
        details.append("mapped atom changes element")
    if result["reagent_contributes_to_product"]:
        details.append("a labeled reagent contributes product atoms")
    if result["noncontributing_reactant"]:
        details.append("a labeled reactant contributes no product atoms")
    if unmatched:
        details.append("mapped components could not be matched to labeled roles")
    if result["no_detectable_transformation"]:
        details.append("no mapped bond/charge change")
    if result["large_reaction_center"]:
        details.append("unusually large reaction center")
    result["mapping_detail"] = "; ".join(details)
    return result


def build_candidates(review: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, source in review.iterrows():
        if str(source.get("ground_truth_valid", "")).strip().lower() not in {"true", "1"}:
            continue
        for slot, reaction in accepted_reactions(source):
            record: dict[str, Any] = {
                "idx": int(source["idx"]),
                "ground_truth_slot": slot,
                "ground_truth_valid": True,
                "title": source.get("title", ""),
                "paragraph": source.get("paragraph", ""),
                "review_note": source.get("review_note", ""),
                "reaction_smiles": reaction,
                "model_agreement_count_at_labeling": int(source.get("model_agreement_count", 0)),
            }
            try:
                reactants, reagents, products = split_reaction(reaction)
                record["reactants_json"] = json.dumps(reactants)
                record["reagents_json"] = json.dumps(reagents)
                record["products_json"] = json.dumps(products)
                record["reactant_count"] = len(reactants)
                record["reagent_count"] = len(reagents)
                record["product_count"] = len(products)
                invalid = [
                    item
                    for item in reactants + reagents + products
                    if Chem.MolFromSmiles(item) is None
                ]
                record["invalid_smiles_count"] = len(invalid)
                record["invalid_smiles"] = " | ".join(invalid)
                left_atoms = atom_counter(reactants + reagents)
                product_atoms = atom_counter(products)
                deficits = (
                    {
                        Chem.GetPeriodicTable().GetElementSymbol(element): count - left_atoms[element]
                        for element, count in product_atoms.items()
                        if count > left_atoms[element]
                    }
                    if left_atoms is not None and product_atoms is not None
                    else {}
                )
                record["element_deficit"] = json.dumps(deficits, sort_keys=True)
                record["element_deficit_flag"] = int(bool(deficits))
                strict_gt = signature(reactants, reagents, products, relaxed=False)
                relaxed_gt = signature(reactants, reagents, products, relaxed=True)
                strict_support = []
                relaxed_support = []
                for model in MODELS:
                    components = model_components(source, model)
                    if strict_gt is not None and signature(*components, relaxed=False) == strict_gt:
                        strict_support.append(model)
                    if relaxed_gt is not None and signature(*components, relaxed=True) == relaxed_gt:
                        relaxed_support.append(model)
                record["strict_model_support_count"] = len(strict_support)
                record["strict_model_support"] = json.dumps(strict_support)
                record["relaxed_model_support_count"] = len(relaxed_support)
                record["relaxed_model_support"] = json.dumps(relaxed_support)
                multistep_models = [
                    model
                    for model in MODELS
                    if len(parse_list(source.get(f"{model}_mapped_reactions", ""))) > 1
                ]
                record["multistep_model_count"] = len(multistep_models)
                record["multistep_models"] = json.dumps(multistep_models)
                reactant_ids = Counter(canonical_smiles(item, ignore_stereo=True) for item in reactants)
                reagent_ids = Counter(canonical_smiles(item, ignore_stereo=True) for item in reagents)
                product_ids = Counter(canonical_smiles(item, ignore_stereo=True) for item in products)
                record["identical_reactant_product"] = int(reactant_ids == product_ids)
                record["role_overlap_count"] = sum(
                    min(reactant_ids[key], reagent_ids[key]) for key in reactant_ids.keys() & reagent_ids.keys()
                )
                record["product_reactant_overlap_count"] = sum(
                    min(product_ids[key], reactant_ids[key])
                    for key in product_ids.keys() & reactant_ids.keys()
                )
                record["product_reagent_overlap_count"] = sum(
                    min(product_ids[key], reagent_ids[key])
                    for key in product_ids.keys() & reagent_ids.keys()
                )
                localmapper_input = f"{'.'.join(reactants + reagents)}>>{'.'.join(products)}"
                record["localmapper_input"] = localmapper_input
                record["pre_mapping_error"] = ""
            except Exception as exc:
                record.update(
                    {
                        "reactants_json": "[]",
                        "reagents_json": "[]",
                        "products_json": "[]",
                        "reactant_count": 0,
                        "reagent_count": 0,
                        "product_count": 0,
                        "invalid_smiles_count": 1,
                        "invalid_smiles": "",
                        "element_deficit": "{}",
                        "element_deficit_flag": 0,
                        "strict_model_support_count": 0,
                        "strict_model_support": "[]",
                        "relaxed_model_support_count": 0,
                        "relaxed_model_support": "[]",
                        "multistep_model_count": 0,
                        "multistep_models": "[]",
                        "identical_reactant_product": 0,
                        "role_overlap_count": 0,
                        "product_reactant_overlap_count": 0,
                        "product_reagent_overlap_count": 0,
                        "localmapper_input": "",
                        "pre_mapping_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            rows.append(record)
    return pd.DataFrame(rows)


def map_candidates(frame: pd.DataFrame, device: str, batch_size: int) -> pd.DataFrame:
    os.environ.setdefault("DGLBACKEND", "pytorch")
    from localmapper import localmapper

    mapper = localmapper(device=device, model_version="202403")
    mapped_records: list[dict[str, Any]] = [
        {"mapped_rxn": "", "template": "", "confident": False, "error": ""}
        for _ in range(len(frame))
    ]
    valid_indices = [
        index
        for index, row in frame.iterrows()
        if row["localmapper_input"] and not row["pre_mapping_error"]
    ]
    for start in range(0, len(valid_indices), batch_size):
        indices = valid_indices[start : start + batch_size]
        reactions = [frame.at[index, "localmapper_input"] for index in indices]
        try:
            results = mapper.get_atom_map(reactions, return_dict=True)
            if isinstance(results, dict):
                results = [results]
            if len(results) != len(indices):
                raise ValueError("LocalMapper returned the wrong batch size")
            for index, result in zip(indices, results):
                mapped_records[index] = {
                    "mapped_rxn": str(result.get("mapped_rxn", "") or ""),
                    "template": str(result.get("template", "") or ""),
                    "confident": bool(result.get("confident", False)),
                    "error": "",
                }
        except Exception:
            for index, reaction in zip(indices, reactions):
                try:
                    result = mapper.get_atom_map(reaction, return_dict=True)
                    mapped_records[index] = {
                        "mapped_rxn": str(result.get("mapped_rxn", "") or ""),
                        "template": str(result.get("template", "") or ""),
                        "confident": bool(result.get("confident", False)),
                        "error": "",
                    }
                except Exception as exc:
                    mapped_records[index]["error"] = f"{type(exc).__name__}: {exc}"

    audits = []
    for index, row in frame.iterrows():
        mapping = mapped_records[index]
        reactants = json.loads(row["reactants_json"])
        reagents = json.loads(row["reagents_json"])
        products = json.loads(row["products_json"])
        if mapping["mapped_rxn"]:
            audit = mapping_audit(mapping["mapped_rxn"], reactants, reagents, products)
        else:
            audit = mapping_audit("", reactants, reagents, products)
            audit["mapping_detail"] = mapping["error"] or row["pre_mapping_error"] or "mapping unavailable"
        audit["mapping_template"] = mapping["template"]
        audit["mapping_confident"] = int(mapping["confident"])
        audit["mapping_error"] = mapping["error"]
        audits.append(audit)
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(audits)], axis=1)


def classify_risk(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    reasons = []
    levels = []
    for _, row in result.iterrows():
        critical = []
        review = []
        if row["invalid_smiles_count"]:
            critical.append("invalid SMILES")
        if row["element_deficit_flag"]:
            critical.append("product contains an element absent from all inputs")
        if row["mapping_error"] or not row["mapped_reaction"]:
            critical.append("atom mapping failed")
        if not row["product_mapping_complete"]:
            critical.append("product atom mapping incomplete")
        if row["duplicate_atom_maps"]:
            critical.append("duplicate atom-map numbers")
        if row["atom_element_mismatch"]:
            critical.append("atom-map element mismatch")
        if row["reagent_contributes_to_product"]:
            review.append("reagent contributes product atoms")
        if row["noncontributing_reactant"]:
            review.append("reactant contributes no product atoms")
        if row["role_assignment_unmatched_components"]:
            review.append("mapped and labeled components do not align")
        if row["no_detectable_transformation"]:
            review.append("no mapped covalent/charge change")
        if row["large_reaction_center"]:
            review.append("unusually large reaction center")
        if not row["mapping_confident"]:
            review.append("LocalMapper marked mapping unconfident")
        if not row["mapping_template"]:
            review.append("no mapping template")
        if row["role_overlap_count"]:
            review.append("same structure labeled as reactant and reagent")
        if row["product_reactant_overlap_count"] or row["product_reagent_overlap_count"]:
            review.append("a product is also present on the input side")
        if row["multistep_model_count"] >= 2 and row["product_count"] > 1:
            review.append("possible flattened multistep sequence")
        if row["relaxed_model_support_count"] == 0:
            review.append("validated ground truth has no model support")
        if row["identical_reactant_product"]:
            review.append("reactant and product structure sets are identical")
        if critical:
            levels.append("critical")
        elif review:
            levels.append("manual_review")
        else:
            levels.append("pass")
        reasons.append("; ".join(critical + review))
    result["audit_level"] = levels
    result["audit_reasons"] = reasons
    priority = {"critical": 0, "manual_review": 1, "pass": 2}
    result["_priority"] = result["audit_level"].map(priority)
    return result.sort_values(
        ["_priority", "relaxed_model_support_count", "idx", "ground_truth_slot"]
    ).drop(columns="_priority")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--findings-output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    review = pd.read_csv(args.review, encoding="utf-8-sig").fillna("")
    candidates = build_candidates(review)
    audited = classify_risk(map_candidates(candidates, args.device, args.batch_size))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audited.to_csv(args.output, index=False, encoding="utf-8-sig")
    if args.findings_output is not None:
        findings = []
        indexed = audited.set_index(["idx", "ground_truth_slot"], drop=False)
        for key, (finding, action, rationale) in CHEMIST_FINDINGS.items():
            row = indexed.loc[key]
            findings.append(
                {
                    "idx": key[0],
                    "ground_truth_slot": key[1],
                    "ground_truth_valid": True,
                    "title": row["title"],
                    "chemist_finding": finding,
                    "recommended_action": action,
                    "chemistry_rationale": rationale,
                    "product_mapping_complete": row["product_mapping_complete"],
                    "mapping_confident": row["mapping_confident"],
                    "mapping_detail": row["mapping_detail"],
                    "reaction_smiles": row["reaction_smiles"],
                    "paragraph": row["paragraph"],
                }
            )
        args.findings_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(findings).to_csv(
            args.findings_output,
            index=False,
            encoding="utf-8-sig",
        )
    summary = {
        "review_rows": int(
            review["ground_truth_valid"].astype(str).str.lower().isin(["true", "1"]).sum()
        ),
        "accepted_reactions": len(audited),
        "ground_truth_valid_counts": review["ground_truth_valid"].value_counts().to_dict(),
        "audit_level_counts": audited["audit_level"].value_counts().to_dict(),
        "mapping_complete": int(audited["product_mapping_complete"].sum()),
        "mapping_confident": int(audited["mapping_confident"].sum()),
        "no_detectable_transformation": int(audited["no_detectable_transformation"].sum()),
        "role_mismatch_candidates": int(
            ((audited["reagent_contributes_to_product"] > 0) | (audited["noncontributing_reactant"] > 0)).sum()
        ),
        "zero_model_support_validated": int(
            (audited["relaxed_model_support_count"] == 0).sum()
        ),
        "chemist_findings": len(CHEMIST_FINDINGS),
        "unique_paragraphs_with_findings": len({key[0] for key in CHEMIST_FINDINGS}),
        "unique_paragraphs_requiring_ground_truth_change": len(
            {
                key[0]
                for key, value in CHEMIST_FINDINGS.items()
                if value[0] != "mapping_artifact_only"
            }
        ),
    }
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
