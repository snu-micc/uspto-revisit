import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _toml_list(text: str, header: str) -> list[str]:
    match = re.search(rf"^{re.escape(header)}\s*=\s*\[(.*?)^\]", text, re.M | re.S)
    assert match, header
    return re.findall(r'"([^"]+)"', match.group(1))


def test_requirements_txt_mirrors_pyproject():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    expected = _toml_list(pyproject, "dependencies") + _toml_list(pyproject, "mapping")
    requirements = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements == expected


def test_lock_file_has_exact_pins_only():
    lines = [
        line
        for line in (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert lines
    assert all(re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[^\s@]+", line) for line in lines)
    assert not any(line.lower().startswith("uspto-revisit==") for line in lines)


def test_citation_file_has_required_fields():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for key in ("cff-version:", "message:", "title:", "authors:"):
        assert key in citation
