"""Local Streamlit UI for reviewing reaction-SMILES ground-truth candidates."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor


DEFAULT_PATH = Path("evaluation/ground_truth_review.csv")
DEFAULT_INPUT_PATH = Path("examples/input.csv")
VALIDITY_OPTIONS = ("unreviewed", "True", "False")
MODEL_DISPLAY_ORDER = (
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash-lite",
    "gpt-5.6-sol",
    "gpt-5.4",
    "gpt_4.1_mini_ft",
    "gpt-4.1-mini",
)


def parse_components(value: object) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def canonical_without_atom_maps(
    smiles: str,
    *,
    ignore_stereochemistry: bool = False,
) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return smiles
    molecule = Chem.Mol(molecule)
    if ignore_stereochemistry:
        Chem.RemoveStereochemistry(molecule)
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=not ignore_stereochemistry,
    )


def unique_components(
    components: list[str],
    *,
    ignore_stereochemistry: bool = False,
) -> list[str]:
    """Return one representative per selected chemical-structure identity."""
    unique = []
    seen = set()
    for smiles in components:
        identity = canonical_without_atom_maps(
            smiles,
            ignore_stereochemistry=ignore_stereochemistry,
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(smiles)
    return unique


def component_text(value: object, *, product_set: bool = False) -> str:
    components = parse_components(value)
    if product_set:
        components = unique_components(
            components,
            ignore_stereochemistry=True,
        )
    return "\n".join(components)


def model_group_text(value: object) -> str:
    try:
        groups = json.loads(str(value))
    except (TypeError, ValueError):
        return ""
    return " | ".join(" = ".join(group) for group in groups if group)


def save_components(text: str) -> str:
    return json.dumps([line.strip() for line in text.splitlines() if line.strip()])


def save_product_components(text: str) -> str:
    products = [line.strip() for line in text.splitlines() if line.strip()]
    return json.dumps(
        unique_components(products, ignore_stereochemistry=True)
    )


def reaction_from_values(reactants: str, reagents: str, products: str) -> str:
    sections = tuple(map(parse_components, (reactants, reagents, products)))
    if not any(sections):
        return ""
    return ">".join(".".join(section) for section in sections)


def draft_key(slot: int, field: str) -> str:
    return f"ground_truth_{field}" if slot == 1 else f"ground_truth_2_{field}"


def saved_column(slot: int, field: str) -> str:
    return (
        f"ground_truth_{field}_smiles"
        if slot == 1
        else f"ground_truth_2_{field}_smiles"
    )


def load_ground_truth_draft(row: pd.Series) -> None:
    """Load the saved value, or the consensus candidate, into editable widgets."""
    value = str(row.get("ground_truth_valid", "")).strip().lower()
    st.session_state["ground_truth_status"] = {
        "true": "True",
        "1": "True",
        "false": "False",
        "0": "False",
    }.get(value, "unreviewed")
    for slot in (1, 2):
        for field in ("reactants", "reagents", "products"):
            saved = row.get(saved_column(slot, field), "")
            fallback = row[f"candidate_{field}_smiles"] if slot == 1 else ""
            st.session_state[draft_key(slot, field)] = component_text(
                saved or fallback,
                product_set=field == "products",
            )
    st.session_state["ground_truth_note"] = row["review_note"]


def use_model_as_draft(row: pd.Series, model: str, slot: int) -> None:
    """Copy one model's un-mapped components into the editable ground truth."""
    for field in ("reactants", "reagents", "products"):
        st.session_state[draft_key(slot, field)] = component_text(
            row[f"{model}_{field}_smiles"],
            product_set=field == "products",
        )
    st.session_state["ground_truth_status"] = "True"


@st.cache_data(show_spinner=False)
def structure_image(components: tuple[str, ...]):
    """Render high-resolution 2D depictions with atom-map numbers visible."""
    molecules = []
    invalid = []
    for smiles in components:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            invalid.append(smiles)
            continue
        display_molecule = Chem.Mol(molecule)
        rdDepictor.Compute2DCoords(display_molecule)
        molecules.append(display_molecule)
    if not molecules:
        return None, invalid
    return Draw.MolsToGridImage(
        molecules,
        molsPerRow=min(3, len(molecules)),
        subImgSize=(260, 190),
    ), invalid


def show_structures(
    heading: str,
    value: object,
    *,
    as_set: bool = False,
    ignore_stereochemistry: bool = False,
) -> None:
    show_component_structures(
        heading,
        parse_components(value),
        as_set=as_set,
        ignore_stereochemistry=ignore_stereochemistry,
    )


def show_component_structures(
    heading: str,
    components: list[str],
    *,
    as_set: bool = False,
    ignore_stereochemistry: bool = False,
) -> None:
    st.caption(heading)
    if as_set:
        components = unique_components(
            components,
            ignore_stereochemistry=ignore_stereochemistry,
        )
    if not components:
        st.caption("(none)")
        return
    image, invalid = structure_image(tuple(components))
    if image is not None:
        st.image(image, width=min(780, 260 * len(components)))
    if invalid:
        st.warning("Invalid SMILES: " + " | ".join(invalid))


def ground_truth_editor(slot: int) -> tuple[str, str, str]:
    st.markdown(f"**Ground truth {slot}**")
    left, middle, right = st.columns(3)
    reactants = left.text_area(
        f"Ground-truth {slot} reactants",
        height=180,
        key=draft_key(slot, "reactants"),
    )
    reagents = middle.text_area(
        f"Ground-truth {slot} reagents",
        height=180,
        key=draft_key(slot, "reagents"),
    )
    products = right.text_area(
        f"Ground-truth {slot} products",
        height=180,
        key=draft_key(slot, "products"),
    )
    preview = st.columns(3)
    with preview[0]:
        show_structures(
            f"Ground-truth {slot} reactant preview",
            save_components(reactants),
        )
    with preview[1]:
        show_structures(
            f"Ground-truth {slot} reagent preview",
            save_components(reagents),
            as_set=True,
        )
    with preview[2]:
        show_structures(
            f"Ground-truth {slot} product preview",
            save_components(products),
            as_set=True,
            ignore_stereochemistry=True,
        )
    return reactants, reagents, products


def lowe_reaction_smiles(value: object) -> str:
    """Discard whitespace-separated Lowe metadata such as atom-map format fields."""
    return str(value).strip().split(maxsplit=1)[0]


def classify_lowe_components(reaction: str) -> tuple[list[str], list[str], list[str]]:
    """Classify Lowe left-side compounds by atom-map overlap with its products."""
    parts = reaction.split(">")
    if len(parts) != 3:
        return [], [], []
    left, middle, right = ([smiles for smiles in part.split(".") if smiles] for part in parts)
    product_maps: set[int] = set()
    for smiles in right:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            product_maps.update(atom.GetAtomMapNum() for atom in molecule.GetAtoms() if atom.GetAtomMapNum())
    reactants: list[str] = []
    reagents: list[str] = []
    for smiles in left + middle:
        molecule = Chem.MolFromSmiles(smiles)
        atom_maps = (
            {atom.GetAtomMapNum() for atom in molecule.GetAtoms() if atom.GetAtomMapNum()}
            if molecule is not None
            else set()
        )
        if atom_maps & product_maps:
            reactants.append(smiles)
        else:
            reagents.append(smiles)
    return reactants, reagents, right


def use_lowe_as_draft(value: object, slot: int) -> None:
    """Load the Lowe reaction into the editable ground-truth component fields."""
    reactants, reagents, products = classify_lowe_components(lowe_reaction_smiles(value))
    for field, components in zip(
        ("reactants", "reagents", "products"), (reactants, reagents, products)
    ):
        if field == "products":
            components = unique_components(
                components,
                ignore_stereochemistry=True,
            )
        st.session_state[draft_key(slot, field)] = "\n".join(
            canonical_without_atom_maps(smiles) for smiles in components
        )
    st.session_state["ground_truth_status"] = "True"


def navigate_record(record_ids: list[int], direction: int) -> None:
    current = st.session_state.get("selected_record", record_ids[0])
    try:
        current_position = record_ids.index(current)
    except ValueError:
        current_position = 0
    next_position = max(0, min(len(record_ids) - 1, current_position + direction))
    st.session_state["selected_record"] = record_ids[next_position]


def show_navigation(
    record_ids: list[int],
    current_position: int,
    *,
    key_prefix: str,
) -> None:
    navigation = st.columns([0.45, 0.45, 5])
    navigation[0].button(
        "←",
        key=f"{key_prefix}_previous",
        help="Previous record",
        disabled=current_position == 0,
        on_click=navigate_record,
        args=(record_ids, -1),
    )
    navigation[1].button(
        "→",
        key=f"{key_prefix}_next",
        help="Next record",
        disabled=current_position == len(record_ids) - 1,
        on_click=navigate_record,
        args=(record_ids, 1),
    )
    navigation[2].markdown(
        f"**Record {current_position + 1} / {len(record_ids)}**"
    )


def show_lowe_reaction(value: object) -> None:
    reaction = lowe_reaction_smiles(value)
    reactants, reagents, products = classify_lowe_components(reaction)
    if not any((reactants, reagents, products)):
        st.caption("(invalid or unavailable Lowe reaction SMILES)")
        return
    labels = ("Lowe reactants", "Lowe reagents", "Lowe products")
    for index, (column, label, components) in enumerate(
        zip(st.columns(3), labels, (reactants, reagents, products))
    ):
        with column:
            show_component_structures(
                label,
                components,
                as_set=index in (1, 2),
                ignore_stereochemistry=index == 2,
            )


def classify_mapped_step(reaction: str) -> tuple[list[str], list[str], list[str]]:
    """Split one mapped step into reactants, reagents, and products by map overlap."""
    parts = reaction.split(">")
    if len(parts) == 2:
        left, right = parts
        middle = ""
    elif len(parts) == 3:
        left, middle, right = parts
    else:
        return [], [], []
    left_components = [smiles for smiles in (left + "." + middle).split(".") if smiles]
    products = [smiles for smiles in right.split(".") if smiles]
    product_maps = {
        atom.GetAtomMapNum()
        for smiles in products
        if (molecule := Chem.MolFromSmiles(smiles)) is not None
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum()
    }
    reactants, reagents = [], []
    for smiles in left_components:
        molecule = Chem.MolFromSmiles(smiles)
        atom_maps = (
            {atom.GetAtomMapNum() for atom in molecule.GetAtoms() if atom.GetAtomMapNum()}
            if molecule is not None
            else set()
        )
        (reactants if atom_maps & product_maps else reagents).append(smiles)
    return reactants, reagents, products


def show_mapped_step(step_number: int, reaction: str) -> None:
    reactants, reagents, products = classify_mapped_step(reaction)
    st.markdown(f"**Step {step_number}**")
    for index, (column, heading, components) in enumerate(
        zip(
            st.columns(3),
            ("Mapped reactants", "Mapped reagents", "Mapped products"),
            (reactants, reagents, products),
        )
    ):
        with column:
            displayed_components = (
                unique_components(
                    components,
                    ignore_stereochemistry=index == 2,
                )
                if index in (1, 2)
                else components
            )
            show_component_structures(heading, displayed_components)
            st.code(
                "\n".join(
                    canonical_without_atom_maps(smiles)
                    for smiles in displayed_components
                )
                or "(none)",
                wrap_lines=True,
            )


@st.cache_data(show_spinner=False)
def load_queue(path_text: str, modified: float) -> pd.DataFrame:
    del modified
    frame = pd.read_csv(path_text, encoding="utf-8-sig").fillna("")
    for column in (
        "ground_truth_2_reactants_smiles",
        "ground_truth_2_reagents_smiles",
        "ground_truth_2_products_smiles",
        "ground_truth_2_reaction_smiles",
    ):
        if column not in frame:
            frame[column] = ""
    if "ground_truth_valid" not in frame:
        frame["ground_truth_valid"] = ""
    frame["ground_truth_valid"] = frame["ground_truth_valid"].map(
        lambda value: {
            "true": "True",
            "1": "True",
            "false": "False",
            "0": "False",
        }.get(str(value).strip().lower(), "unreviewed")
    )
    return frame


@st.cache_data(show_spinner=False)
def load_lowe_smiles(path_text: str, modified: float) -> dict[int, str]:
    """Read the original Lowe reactions by row index without altering the review CSV."""
    del modified
    source = pd.read_csv(path_text, encoding="utf-8-sig").fillna("")
    if "Lowe_smiles" not in source:
        return {}
    return {index: value for index, value in source["Lowe_smiles"].items() if value}


def main() -> None:
    st.set_page_config(page_title="Ground Truth Labeler", layout="wide")
    st.markdown(
        """
        <style>
        textarea[disabled] {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            opacity: 1 !important;
            font-weight: 500 !important;
        }
        .paragraph-box {
            color: #000000;
            background: #f8fafc;
            border: 1px solid #d1d5db;
            border-radius: 4px;
            padding: 0.75rem 0.9rem;
            white-space: pre-wrap;
            line-height: 1.5;
            font-weight: 500;
        }
        .title-box {
            color: #000000;
            font-size: 1.1rem;
            font-weight: 700;
            margin: 0.15rem 0 0.4rem;
        }
        .agreement-summary {
            color: #000000;
            font-size: 0.875rem;
            font-weight: 500;
            margin: 0.15rem 0;
        }
        [data-testid="stCode"] pre { font-size: 0.52rem !important; line-height: 1.05 !important; }
        [data-testid="stWidgetLabel"] p {
            color: #111827 !important;
            font-weight: 700 !important;
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p {
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Ground Truth Labeler")
    path = Path(st.sidebar.text_input("Review CSV", str(DEFAULT_PATH)))
    if not path.exists():
        st.error(f"Review CSV not found: {path}")
        st.stop()
    frame = load_queue(str(path), path.stat().st_mtime)
    lowe_smiles = (
        load_lowe_smiles(str(DEFAULT_INPUT_PATH), DEFAULT_INPUT_PATH.stat().st_mtime)
        if DEFAULT_INPUT_PATH.exists()
        else {}
    )
    status_filter = st.sidebar.multiselect(
        "Ground truth valid", VALIDITY_OPTIONS, default=["unreviewed"]
    )
    manual_only = st.sidebar.checkbox("Manual review required only", value=True)
    filtered = (
        frame[frame["ground_truth_valid"].isin(status_filter)]
        if status_filter
        else frame
    )
    if manual_only:
        filtered = filtered[filtered["manual_review_required"].eq(1)]
        filtered = filtered.sort_values(
            [
                "manual_review_required",
                "product_mapping_complete",
                "model_agreement_count",
                "idx",
            ],
            ascending=[False, True, True, True],
        )
    else:
        filtered = filtered.sort_values("idx")
    st.sidebar.caption(f"{len(filtered)} / {len(frame)} rows shown")
    if filtered.empty:
        st.info("No rows match the current filters.")
        st.stop()
    record_ids = filtered["idx"].tolist()
    if st.session_state.get("selected_record") not in record_ids:
        st.session_state["selected_record"] = record_ids[0]
    selected_idx = st.sidebar.selectbox(
        "Record",
        record_ids,
        format_func=lambda value: f"#{value}",
        key="selected_record",
    )
    row = frame.loc[frame["idx"].eq(selected_idx)].iloc[0]
    if st.session_state.get("ground_truth_record") != row["idx"]:
        load_ground_truth_draft(row)
        st.session_state["ground_truth_record"] = row["idx"]

    current_position = record_ids.index(selected_idx)
    show_navigation(record_ids, current_position, key_prefix="top")
    lowe_reaction = lowe_smiles.get(int(row["idx"]), "")
    st.markdown("**Lowe_smiles**")
    st.code(lowe_reaction_smiles(lowe_reaction) or "(not available)", wrap_lines=True)
    show_lowe_reaction(lowe_reaction)
    lowe_buttons = st.columns(2)
    for slot, column in enumerate(lowe_buttons, start=1):
        if column.button(
            f"Use Lowe_smiles as ground truth {slot}",
            key=f"use_lowe_{slot}_{row['idx']}",
        ):
            use_lowe_as_draft(lowe_reaction, slot)
            st.rerun()
    st.markdown("**Title**")
    st.markdown(
        f'<div class="title-box">#{row["idx"]} &nbsp; {escape(str(row["title"]))}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Paragraph**")
    st.markdown(f'<div class="paragraph-box">{escape(str(row["paragraph"]))}</div>', unsafe_allow_html=True)
    metrics = st.columns(4)
    metrics[0].metric("Model agreement", f"{row['model_agreement_count']} / 5")
    metrics[1].metric("Product mapping complete", "Yes" if row["product_mapping_complete"] else "No")
    metrics[2].metric("Manual review", "Required" if row["manual_review_required"] else "Candidate")
    metrics[3].metric("Rare template input", "Yes" if row["rare_template_input"] else "No")

    st.subheader("Model Structures")
    comparison_groups = row.get("reaction_comparison_groups", row.get("model_agreement_groups", ""))
    st.markdown(
        f'<div class="agreement-summary">Reaction comparison groups (including Lowe_smiles): '
        f'{model_group_text(comparison_groups) or "(none)"}</div>',
        unsafe_allow_html=True,
    )
    comparison_disagreeing = ", ".join(
        parse_components(
            row.get("reaction_comparison_disagreeing", row.get("model_disagreeing_models", ""))
        )
    )
    if comparison_disagreeing:
        st.markdown(
            f'<div class="agreement-summary">Different from the largest comparison group: '
            f'{comparison_disagreeing}</div>',
            unsafe_allow_html=True,
        )
    show_raw_output = st.toggle("Show raw model outputs", value=False)
    available_models = {
        column.removesuffix("_product_mapping_complete")
        for column in frame.columns
        if column.endswith("_product_mapping_complete")
    }
    for model in [model for model in MODEL_DISPLAY_ORDER if model in available_models]:
        status = "Product mapping OK" if row[f"{model}_product_mapping_complete"] else "Product mapping CHECK"
        st.markdown(f"**{model}**  |  {status}")
        model_buttons = st.columns(2)
        for slot, column in enumerate(model_buttons, start=1):
            if column.button(
                f"Use as ground truth {slot}",
                key=f"use_{slot}_{row['idx']}_{model}",
            ):
                use_model_as_draft(row, model, slot)
                st.rerun()
        mapped_steps = parse_components(row.get(f"{model}_mapped_reactions", ""))
        if len(mapped_steps) > 1:
            for step_number, reaction in enumerate(mapped_steps, start=1):
                show_mapped_step(step_number, reaction)
        else:
            columns = st.columns(3)
            with columns[0]:
                show_structures("Mapped reactants", row[f"{model}_mapped_reactants_smiles"])
                st.code(component_text(row[f"{model}_reactants_smiles"]) or "(none)", wrap_lines=True)
            with columns[1]:
                show_structures(
                    "Mapped reagents",
                    row[f"{model}_mapped_reagents_smiles"],
                    as_set=True,
                )
                st.code(component_text(row[f"{model}_reagents_smiles"]) or "(none)", wrap_lines=True)
            with columns[2]:
                show_structures(
                    "Mapped products",
                    row[f"{model}_mapped_products_smiles"],
                    as_set=True,
                    ignore_stereochemistry=True,
                )
                st.code(component_text(row[f"{model}_products_smiles"]) or "(none)", wrap_lines=True)
        if show_raw_output:
            st.caption("Raw model output")
            st.code(row.get(f"{model}_raw_output", "(not available)") or "(empty)", wrap_lines=True)
            st.caption("Skeleton")
            st.code(row.get(f"{model}_skeleton", "(not available)") or "(empty)", wrap_lines=True)
            unresolved_nosmi = component_text(row.get(f"{model}_unresolved_nosmi", ""))
            if unresolved_nosmi:
                st.caption("Unresolved NoSmi compounds")
                st.code(unresolved_nosmi, wrap_lines=True)
            if row.get(f"{model}_generation_error", ""):
                st.warning("Generation error: " + row[f"{model}_generation_error"])
        if row[f"{model}_mapping_error"]:
            st.warning(row[f"{model}_mapping_error"])

    validity = st.selectbox(
        "Ground truth valid", VALIDITY_OPTIONS, key="ground_truth_status"
    )
    st.caption(
        "True = 사람이 검토하여 유효한 ground-truth reaction을 확정 | "
        "False = 사람이 검토했으나 유효한 reaction을 확정할 수 없음 | "
        "unreviewed = 아직 검토하지 않음"
    )
    st.caption("One SMILES per line. The combined ground-truth reaction is generated automatically.")
    ground_truth_1 = ground_truth_editor(1)
    ground_truth_2 = ground_truth_editor(2)
    if validity == "False":
        exclusion_buttons = st.columns(2)
        if exclusion_buttons[0].button("Erroneous chemical names", key=f"exclude_names_{row['idx']}"):
            st.session_state["ground_truth_note"] = "Erroneous chemical names"
            st.rerun()
        if exclusion_buttons[1].button("Untraceable cross-references", key=f"exclude_refs_{row['idx']}"):
            st.session_state["ground_truth_note"] = "Untraceable cross-references"
            st.rerun()
    note = st.text_area("Review note", height=90, key="ground_truth_note")
    if st.button("Save decision", type="primary"):
        target = frame["idx"].eq(row["idx"])
        frame.loc[target, "ground_truth_valid"] = (
            "" if validity == "unreviewed" else validity == "True"
        )
        for slot, values in ((1, ground_truth_1), (2, ground_truth_2)):
            reactants_json = save_components(values[0])
            reagents_json = save_components(values[1])
            products_json = save_product_components(values[2])
            frame.loc[target, saved_column(slot, "reactants")] = reactants_json
            frame.loc[target, saved_column(slot, "reagents")] = reagents_json
            frame.loc[target, saved_column(slot, "products")] = products_json
            reaction_column = (
                "ground_truth_reaction_smiles"
                if slot == 1
                else "ground_truth_2_reaction_smiles"
            )
            frame.loc[target, reaction_column] = reaction_from_values(
                reactants_json,
                reagents_json,
                products_json,
            )
        frame.loc[target, "review_note"] = note
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        load_queue.clear()
        st.success(f"Saved #{row['idx']}")

    show_navigation(record_ids, current_position, key_prefix="bottom")


if __name__ == "__main__":
    main()
