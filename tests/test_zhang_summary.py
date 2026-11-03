import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'benchmarks/zhang_et_al'
spec = importlib.util.spec_from_file_location('zhang_summary', BENCH / 'summarize.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_primary_product_presence_and_all_query_distinction():
    summary, rows = module.summarize(BENCH / 'llama3_8b_predictions.csv', ROOT / 'analysis/data/input.csv')
    assert len(rows) == 400
    assert summary['counts'] == {'both': 172, 'products_only': 15, 'reactants_only': 0, 'neither': 213}
    assert summary['any_product_query_with_reactants'] == 173


def test_rejects_misaligned_input_and_changed_primary_choice(tmp_path):
    rows = module.read_rows(BENCH / 'llama3_8b_predictions.csv')
    path = tmp_path / 'predictions.csv'
    def save():
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    original = rows[0]['paragraph']
    rows[0]['paragraph'] = 'wrong input'
    save()
    with pytest.raises(ValueError, match='Input mismatch'):
        module.summarize(path, ROOT / 'analysis/data/input.csv')
    rows[0]['paragraph'] = original
    selected = next(r for r in rows if r['primary_product'])
    selected['primary_product'] = 'different product'
    save()
    with pytest.raises(ValueError, match='Primary-product selection mismatch'):
        module.summarize(path, ROOT / 'analysis/data/input.csv')
