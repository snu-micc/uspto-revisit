"""Shared task definitions, kept identical to upstream (zw-SIMM/SFTLLMs_for_ChemText_Mining)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TASKS = {
    "role": {"data_dir": ROOT / "data/role", "instruction": "annotate the reaction roles in the paragraph. "},
    "prod": {"data_dir": ROOT / "data/prod", "instruction": "annotate the products in the paragraph. "},
}

SOURCE, TARGET = "input", "output"


def train_text(instruction, src, tgt):
    # Upstream format: Mistral-style [INST] template for every model, "!!!" as stop marker.
    return f"<s>[INST] {instruction}{src} [/INST] {tgt}!!! </s>"


def prompt_text(instruction, src):
    return f"<s>[INST] {instruction}{src} [/INST]"
