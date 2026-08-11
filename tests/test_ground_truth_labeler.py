import importlib.util
import json
import sys
import types
from pathlib import Path


try:
    import streamlit  # noqa: F401
except ModuleNotFoundError:
    streamlit_stub = types.ModuleType("streamlit")

    def cache_data(*_args, **_kwargs):
        def decorate(function):
            function.clear = lambda: None
            return function

        return decorate

    streamlit_stub.cache_data = cache_data
    sys.modules["streamlit"] = streamlit_stub


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "ground_truth_labeler.py"
SPEC = importlib.util.spec_from_file_location("ground_truth_labeler", MODULE_PATH)
ground_truth_labeler = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ground_truth_labeler)


def test_unique_components_uses_structure_identity_without_atom_maps():
    assert ground_truth_labeler.unique_components(
        ["CCO", "OCC", "[CH3:1][CH2:2][OH:3]", "O"]
    ) == ["CCO", "O"]


def test_unique_products_ignore_stereochemistry():
    first = "ClC[C@@H]1CN2CCC[C@H]2CO1"
    second = "ClC[C@H]1CN2CCC[C@H]2CO1"
    assert ground_truth_labeler.unique_components(
        [first, second],
        ignore_stereochemistry=True,
    ) == [first]
    assert json.loads(
        ground_truth_labeler.save_product_components(f"{first}\n{second}")
    ) == [first]


def test_set_structure_preview_draws_each_structure_once(monkeypatch):
    rendered = []

    def capture_components(components):
        rendered.append(components)
        return None, []

    monkeypatch.setattr(ground_truth_labeler, "structure_image", capture_components)
    monkeypatch.setattr(
        ground_truth_labeler,
        "st",
        types.SimpleNamespace(caption=lambda *_args, **_kwargs: None),
    )

    ground_truth_labeler.show_component_structures(
        "Reagents",
        ["CCO", "OCC", "O"],
        as_set=True,
    )

    assert rendered == [("CCO", "O")]


def test_product_preview_draws_stereoisomers_once(monkeypatch):
    first = "ClC[C@@H]1CN2CCC[C@H]2CO1"
    second = "ClC[C@H]1CN2CCC[C@H]2CO1"
    rendered = []

    def capture_components(components):
        rendered.append(components)
        return None, []

    monkeypatch.setattr(ground_truth_labeler, "structure_image", capture_components)
    monkeypatch.setattr(
        ground_truth_labeler,
        "st",
        types.SimpleNamespace(caption=lambda *_args, **_kwargs: None),
    )

    ground_truth_labeler.show_component_structures(
        "Products",
        [first, second],
        as_set=True,
        ignore_stereochemistry=True,
    )

    assert rendered == [(first,)]
