"""The primary (fully automatic) scenario without the lookup tables the resolver had.

The stored outputs were resolved while ``smiles_fetch`` answered names from two fixed tables before or while asking
the public resolvers (``historical_lookup_tables.py``): 19 abbreviations and trivial names (e.g. DCM, THF and brine)
and 13 spelling variants and synonyms of catalysts and reagents looked up under a standard name (e.g. DIEA and
RuPhos). Eleven of the 19 entries and all 13 synonyms were added in July 2026, after the benchmark paragraphs had
been inspected, so they were removed from the resolver and the primary scenario does without them: a name is
resolved only by OPSIN, PubChem, NCI CIR and ChEBI, in the released two-pass order. Results with the tables
(``fully_automatic_with_tables``) are a sensitivity analysis. Writing element names and carbon monoxide as molecules,
and refusing mixtures and patent-local labels, are rules rather than lookup tables and stay.

* ``table_components``: every resolved component of the stored outputs whose name either table answers or
  rewrites.
* ``without_tables``: those components get what the resolvers return with both tables off, replayed from
  ``result/name_resolution/requery_2026-10-23_no_tables_raw.jsonl`` (NoSmi when none answers); changed steps are
  re-mapped.
* ``stale_dma_edits``: the June 2026 table read "DMA" as acetamide; the stored outputs of record 136 kept that value
  in six configurations, and it is replaced by the July 2026 entry (N,N-dimethylacetamide) in the scenarios that use
  the table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from evaluation import curation_ablation as CA
from evaluation.name_resolution_replay import RecordedResolver, alias_used, builtin_table_smiles, clean_name

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NO_TABLES_LOG = REPOSITORY_ROOT / "result" / "name_resolution" / "requery_2026-10-23_no_tables_raw.jsonl"
ACETAMIDE = "CC(=O)N"


def _resolved_components(outputs: dict[str, pd.DataFrame]):
    """(model, row, label, name, stored SMILES, row of the output) for every component with a structure."""
    for model, frame in outputs.items():
        frame = frame.fillna("")
        for idx in range(len(frame)):
            smiles = CA.loads_any(frame.at[idx, f"{model}_smiles"])
            if not isinstance(smiles, dict):
                continue
            for label, name in CA.prediction_names(frame.at[idx, "prediction"]).items():
                yield model, idx, label, name, str(smiles.get(label, "")), frame.iloc[idx]


def table_components(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every resolved component whose name the built-in table answers or the synonym list rewrites."""
    rows = []
    for model, idx, label, name, stored, row in _resolved_components(outputs):
        if not stored or "(NoSmi)" in stored:
            continue
        table, synonym = bool(builtin_table_smiles(name)), alias_used(name)
        if table or synonym:
            rows.append({"model": model, "idx": idx, "label": label, "name": name, "spelling": clean_name(name),
                         "table": "+".join(t for t, used in (("built-in", table), ("synonym", synonym)) if used),
                         "stored": stored, "smiles": CA.canonicalize_smiles(stored) or "",
                         "in_skeleton": int(label in CA.skeleton_labels(row[f"{model}_skeleton"]))})
    return pd.DataFrame(rows)


def without_tables(
    outputs: dict[str, pd.DataFrame],
    requery_log: str | Path = NO_TABLES_LOG,
    remap: Callable[[str], dict] = CA.localmapper_remap,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """The outputs with every table-answered component resolved without the tables, the per-component audit and
    the per-spelling resolutions."""
    audit = table_components(outputs)
    resolver = RecordedResolver(requery_log, lookup_tables=False)
    resolved = resolver.resolve(sorted(set(audit.spelling)), order="two-pass")
    if resolver.missing_urls:
        raise ValueError(f"{len(resolver.missing_urls)} requests are not in {requery_log}, e.g. "
                         f"{sorted(resolver.missing_urls)[0]}")
    automatic = {s: CA.canonicalize_smiles(r["selected_smiles"]) if r["selected_smiles"] else None
                 for s, r in resolved.items()}
    audit["automatic_smiles"] = audit.spelling.map(lambda s: automatic[s] or "")
    audit["automatic_source"] = audit.spelling.map(lambda s: resolved[s]["selected_source"])
    audit["outcome"] = ["unchanged" if new == old else "unresolved" if not new else "replaced"
                        for new, old in zip(audit.automatic_smiles, audit.smiles)]
    first = audit.drop_duplicates("spelling").set_index("spelling")
    spellings = pd.DataFrame([
        {"spelling": s, "table": first.at[s, "table"], "with_tables": first.at[s, "smiles"],
         "without_tables": automatic[s] or "",
         **{k: resolved[s][k] for k in ("selected_source", "resolved_by", "resolver_agreement", "resolver_candidates")},
         "components": int(audit.spelling.eq(s).sum())}
        for s in sorted(resolved)])
    edits: dict[tuple[str, int], dict[str, str | None]] = {}
    for row in audit[audit.outcome != "unchanged"].itertuples():
        edits.setdefault((row.model, int(row.idx)), {})[row.label] = row.automatic_smiles or None
    return CA.counterfactual_outputs(outputs, edits, remap=remap), audit, spellings


def stale_dma_edits(outputs: dict[str, pd.DataFrame]) -> tuple[dict[tuple[str, int], dict[str, str]], pd.DataFrame]:
    """Components that still hold the June 2026 table value (acetamide for "DMA"), with the July 2026 entry."""
    acetamide = CA.canonicalize_smiles(ACETAMIDE)
    edits: dict[tuple[str, int], dict[str, str]] = {}
    rows = []
    for model, idx, label, name, stored, _ in _resolved_components(outputs):
        current = builtin_table_smiles(name)
        if current and CA.canonicalize_smiles(stored) == acetamide and CA.canonicalize_smiles(current) != acetamide:
            edits.setdefault((model, idx), {})[label] = CA.canonicalize_smiles(current)
            rows.append({"model": model, "idx": idx, "label": label, "name": name, "stored": stored,
                         "table_now": CA.canonicalize_smiles(current)})
    return edits, pd.DataFrame(rows)
