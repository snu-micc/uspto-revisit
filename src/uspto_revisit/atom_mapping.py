"""Atom-map generated reaction SMILES with LocalMapper."""

from __future__ import annotations

import ast
import importlib.resources
import math
import os
import sys
import types
from collections.abc import Iterable
from typing import Any

import pandas as pd


def _legacy_resource_filename(package: str, resource: str) -> str:
    """Resolve package data for LocalMapper without setuptools.pkg_resources."""
    return str(importlib.resources.files(package).joinpath(resource))


def _install_localmapper_compatibility() -> None:
    """Install narrow compatibility shims required by LocalMapper and DGL.

    LocalMapper 0.1.4 imports the removed ``pkg_resources.resource_filename``.
    DGL 2.2.1 imports its distributed subsystem eagerly; that subsystem is not
    needed for local atom mapping and currently has an unpatched RPC advisory.
    Supplying a local-only module avoids loading the distributed RPC code.
    """
    if "pkg_resources" not in sys.modules:
        pkg_resources = types.ModuleType("pkg_resources")
        pkg_resources.resource_filename = _legacy_resource_filename
        sys.modules["pkg_resources"] = pkg_resources

    if "dgl.distributed" not in sys.modules:
        distributed = types.ModuleType("dgl.distributed")
        distributed.__path__ = []
        distributed.DistGraph = type("DistGraph", (), {})
        distributed.DistDataLoader = type("DistDataLoader", (), {})
        sys.modules["dgl.distributed"] = distributed


def parse_reaction_smiles(value: Any) -> list[str]:
    """Return reaction SMILES from a list, a serialized list, or one string."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []

    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = stripped

    if isinstance(parsed, str):
        return [parsed.strip()] if parsed.strip() else []
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    raise ValueError(
        "Reaction SMILES must be a string, list, tuple, or a serialized list."
    )


def normalize_localmapper_reaction(reaction_smiles: str) -> str:
    """Convert reaction SMILES to LocalMapper's reactants>>products format."""
    reaction = reaction_smiles.strip()
    if not reaction:
        raise ValueError("Reaction SMILES is empty.")
    if "(NoSmi)" in reaction:
        raise ValueError("Reaction contains an unresolved (NoSmi) compound.")
    if "Error:" in reaction:
        raise ValueError(reaction)

    parts = reaction.split(">")
    if len(parts) < 2:
        raise ValueError(
            "Reaction must contain at least one '>' separator between reactants "
            "and products."
        )

    *reactant_sections, products = parts
    reactants = ".".join(part for part in reactant_sections if part)

    if not reactants or not products:
        raise ValueError("Reaction must contain both reactants and products.")
    return f"{reactants}>>{products}"


def create_localmapper(device: str = "cpu", model_version: str = "202403"):
    """Create LocalMapper lazily so non-mapping commands stay lightweight."""
    os.environ.setdefault("DGLBACKEND", "pytorch")
    _install_localmapper_compatibility()
    try:
        from localmapper import localmapper
    except ImportError as exc:
        raise RuntimeError(
            "LocalMapper could not be imported. Install the mapping dependencies "
            "with `python -m pip install -e \".[mapping]\"`. "
            f"Original import error: {exc}"
        ) from exc
    return localmapper(device=device, model_version=model_version)


def _empty_mapping_record(
    localmapper_rxn: str | None = None,
    error: str = "",
) -> dict[str, Any]:
    return {
        "localmapper_rxn": localmapper_rxn,
        "mapped_rxn": None,
        "template": None,
        "confident": None,
        "error": error,
    }


def _store_mapping_result(record: dict[str, Any], result: Any) -> None:
    if not isinstance(result, dict) or not result.get("mapped_rxn"):
        raise ValueError("LocalMapper returned an invalid result.")
    record["mapped_rxn"] = result["mapped_rxn"]
    record["template"] = result.get("template")
    record["confident"] = result.get("confident")


def _map_one(mapper, record: dict[str, Any]) -> None:
    try:
        result = mapper.get_atom_map(record["localmapper_rxn"], return_dict=True)
        _store_mapping_result(record, result)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"


def _map_batch(mapper, records: list[dict[str, Any]]) -> None:
    """Map a batch, falling back to one-by-one mapping if one item is invalid."""
    if not records:
        return
    reactions = [record["localmapper_rxn"] for record in records]
    try:
        results = mapper.get_atom_map(reactions, return_dict=True)
        if isinstance(results, dict):
            results = [results]
        if not isinstance(results, Iterable):
            raise ValueError("LocalMapper did not return an iterable batch result.")
        results = list(results)
        if len(results) != len(records):
            raise ValueError("LocalMapper returned a different number of results.")
        for record, result in zip(records, results):
            _store_mapping_result(record, result)
    except Exception:
        for record in records:
            _map_one(mapper, record)


def map_reaction_values(
    values: Iterable[Any],
    mapper,
    batch_size: int = 32,
) -> dict[str, list[list[Any]]]:
    """Atom-map a reaction column while retaining per-reaction failures."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    rows: list[list[dict[str, Any]]] = []
    valid_records: list[dict[str, Any]] = []
    for value in values:
        try:
            reactions = parse_reaction_smiles(value)
        except ValueError as exc:
            rows.append([_empty_mapping_record(error=str(exc))])
            continue
        if not reactions:
            rows.append([_empty_mapping_record(error="Reaction SMILES is empty.")])
            continue

        row_records = []
        for reaction in reactions:
            try:
                normalized = normalize_localmapper_reaction(reaction)
                record = _empty_mapping_record(localmapper_rxn=normalized)
                valid_records.append(record)
            except ValueError as exc:
                record = _empty_mapping_record(error=str(exc))
            row_records.append(record)
        rows.append(row_records)

    for start in range(0, len(valid_records), batch_size):
        _map_batch(mapper, valid_records[start : start + batch_size])

    return {
        "localmapper_rxn": [
            [record["localmapper_rxn"] for record in row] for row in rows
        ],
        "mapped_rxn": [[record["mapped_rxn"] for record in row] for row in rows],
        "mapping_template": [
            [record["template"] for record in row] for row in rows
        ],
        "mapping_confident": [
            [record["confident"] for record in row] for row in rows
        ],
        "mapping_error": [[record["error"] for record in row] for row in rows],
    }


def add_atom_mapping_columns(
    frame: pd.DataFrame,
    reaction_column: str,
    output_prefix: str,
    *,
    mapper=None,
    device: str = "cpu",
    model_version: str = "202403",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Return a copy of a DataFrame with LocalMapper result columns added."""
    if reaction_column not in frame.columns:
        available = ", ".join(frame.columns)
        raise ValueError(
            f"Column '{reaction_column}' was not found. Available columns: {available}"
        )
    if mapper is None:
        mapper = create_localmapper(device=device, model_version=model_version)

    mapped = map_reaction_values(frame[reaction_column].tolist(), mapper, batch_size)
    result = frame.copy()
    for suffix, values in mapped.items():
        result[f"{output_prefix}_{suffix}"] = values
    return result
