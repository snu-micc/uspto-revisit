"""The two lookup tables the resolver held while the stored outputs were produced (removed on 2026-10-23).

``uspto_revisit.smiles_fetch`` resolves names only with OPSIN, PubChem, NCI CIR and ChEBI; its lookup tables
(``DEFAULT_KNOWN_SMILES`` and ``SAFE_NAME_ALIASES``) are empty. Until 2026-10-23 they held the entries below: 19
abbreviations and trivial names answered before OPSIN, and 13 spelling variants and synonyms of catalysts and reagents
looked up under a standard name. Eleven of the 19 entries and all 13 synonyms were added in July 2026, after the
benchmark paragraphs had been inspected, so the primary results do not use them.

The stored outputs in ``result/`` and the 2026-10-16 re-query were produced with these tables. The evaluation code
installs them (``installed``) only to replay those runs, to rebuild the sensitivity scenario with the tables on and to
identify the components they answered (``lookup_tables.py``).
"""

from __future__ import annotations

import contextlib

# Every value parses with RDKit (tests/test_smiles_fetch.py). "DMA" is N,N-dimethylacetamide (the solvent sense used
# in the patents); the June 2026 version read it as acetamide (CC(=O)N). "Pd/C" follows the active-component
# convention of config/nosmi_overrides.json; the June value "Pd" was not valid SMILES and was ignored. "brine" is a
# water + NaCl representation of the aqueous mixture.
KNOWN_SMILES = {
    "ice": "O",
    "DCM": "C(Cl)Cl",
    "DMA": "CN(C)C(C)=O",
    "DMF": "CN(C)C=O",
    "THF": "C1CCOC1",
    "ether": "CCOCC",
    "absolute ethanol": "CCO",
    "dry acetone": "CC(=O)C",
    "dry ether": "CCOCC",
    "dry tetrahydrofuran": "C1CCOC1",
    "dry toluene": "Cc1ccccc1",
    "ultrapure water": "O",
    "sodium methylate": "C[O-].[Na+]",
    "lithiumaluminium hydride": "[Li+].[AlH4-]",
    "racemic phenylalanine": "NC(Cc1ccccc1)C(=O)O",
    "brine": "O.[Na+].[Cl-]",
    "Pd/C": "[Pd]",
    "DMSO": "CS(=O)C",
    "LiAlH4": "[Li+].[AlH4-]",
}

NAME_ALIASES = {
    "1,1'-carbonylbis(imidazole)": "1,1'-carbonyldiimidazole",
    "n,n'-carbonyldiimidazole": "1,1'-carbonyldiimidazole",
    "bis-(triphenylphosphine)palladium(ii) dichloride": (
        "bis(triphenylphosphine)palladium(II) dichloride"
    ),
    "bis(triphenylphosphine)palladium(ii) chloride": (
        "bis(triphenylphosphine)palladium(II) dichloride"
    ),
    "diea": "DIPEA",
    "mo(co)6": "molybdenum hexacarbonyl",
    "palladium tetrakis(triphenylphosphine)": (
        "tetrakis(triphenylphosphine)palladium(0)"
    ),
    "phosphorous pentachloride": "phosphorus pentachloride",
    "ruphos": "2-dicyclohexylphosphino-2',6'-diisopropoxybiphenyl",
    "tetrakis(triphenylphosphine)palladium": (
        "tetrakis(triphenylphosphine)palladium(0)"
    ),
    "tri(dibenzylidenacetone)dipalladium(0)": (
        "tris(dibenzylideneacetone)dipalladium(0)"
    ),
    "tri(dibenzylideneacetone)dipalladium(0)": (
        "tris(dibenzylideneacetone)dipalladium(0)"
    ),
    "tris(dibenzylideneacetone)dipalladium": (
        "tris(dibenzylideneacetone)dipalladium(0)"
    ),
}


@contextlib.contextmanager
def installed(tables: bool = True):
    """The resolver with these tables (``tables=True``) or with none while the block runs."""
    from uspto_revisit import smiles_fetch as sf

    saved = (sf.DEFAULT_KNOWN_SMILES, sf.SAFE_NAME_ALIASES)
    sf.DEFAULT_KNOWN_SMILES = dict(KNOWN_SMILES) if tables else {}
    sf.SAFE_NAME_ALIASES = dict(NAME_ALIASES) if tables else {}
    try:
        yield sf
    finally:
        sf.DEFAULT_KNOWN_SMILES, sf.SAFE_NAME_ALIASES = saved
