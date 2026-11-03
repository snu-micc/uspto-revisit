"""Reproduce SI S13 role-presence counts and examples; no chemical scoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXAMPLE_IDS = (328, 327, 26, 15, 375, 200)
ROLE_TAGS = ('Reactants', 'Catalyst_Reagents', 'Solvent', 'Temperature', 'Time', 'Yield', 'Reaction')


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def summarize(predictions, inputs):
    rows, source = read_rows(predictions), read_rows(inputs)
    ids = [r['idx'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != {r['idx'] for r in source}:
        raise ValueError('Prediction IDs must uniquely cover the input benchmark.')
    if len(source) != 400 or len({r['idx'] for r in source}) != 400:
        raise ValueError('Expected 400 unique benchmark records.')
    reference = {r['idx']: r for r in source}
    counts = Counter()
    any_query = 0
    for row in rows:
        if any(row[k] != reference[row['idx']][k] for k in ('title', 'paragraph')):
            raise ValueError(f"Input mismatch at idx {row['idx']}")
        reactants = json.loads(row['Reactants'] or '[]')
        products = json.loads(row['all_products'])
        queries = json.loads(row['all_roles_json'])
        if not all(isinstance(x, list) for x in (reactants, products, queries)):
            raise ValueError('Parsed fields must be JSON lists.')
        if len(products) != len(queries) or len(products) != int(row['n_products']):
            raise ValueError(f"Product count mismatch at idx {row['idx']}")
        if (row['product_found'].lower() == 'true') != bool(products):
            raise ValueError('product_found disagrees with all_products.')
        if queries:
            selected = max(enumerate(queries), key=lambda item: (
                sum(len(item[1]['roles'].get(tag, [])) for tag in ROLE_TAGS), item[0]))[1]
            if selected['product'] != row['primary_product'] or selected['roles'].get('Reactants', []) != reactants:
                raise ValueError(f"Primary-product selection mismatch at idx {row['idx']}")
        any_query += any(bool(q['roles'].get('Reactants', [])) for q in queries)
        key = ('both' if reactants else 'products_only') if products else ('reactants_only' if reactants else 'neither')
        counts[key] += 1
    result = {
        'model': 'llama3-8b', 'records': len(rows),
        'definition': 'Reactants from the selected primary-product query; products from all_products. Presence is not correctness.',
        'counts': {key: counts[key] for key in ('both', 'products_only', 'reactants_only', 'neither')},
        'any_product_query_with_reactants': any_query,
        'prediction_sha256': hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
        'input_sha256': hashlib.sha256(Path(inputs).read_bytes()).hexdigest(),
        'example_ids': list(EXAMPLE_IDS),
    }
    return result, rows


def plot_summary(summary, output):
    # Pillow is optional; summary and example extraction use only the standard library.
    from PIL import Image, ImageDraw, ImageFont
    font_paths = [Path('C:/Windows/Fonts/arial.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    font_path = next((p for p in font_paths if p.exists()), None)
    def font(size):
        return ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size=size)
    image = Image.new('RGB', (1800, 710), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((900, 35), summary['model'], font=font(40), fill='black', anchor='mt')
    for value in (0, 20, 40, 60):
        y = 530 - 405 * value / 60
        draw.line((140, y, 1750, y), fill='#dddddd', width=2)
        draw.text((120, y), str(value), font=font(28), fill='black', anchor='rm')
    draw.text((30, 95), '%', font=font(30), fill='black')
    labels = ('Reactants + products', 'Products only', 'Reactants only', 'Neither')
    for i, ((key, count), label) in enumerate(zip(summary['counts'].items(), labels)):
        x, percent = 140 + (i + .5) * 1610 / 4, 100 * count / summary['records']
        y = 530 - 405 * percent / 60
        if count:
            draw.rectangle((x - 110, y, x + 110, 530), fill='#4472A0')
        draw.text((x, y - 14), f"{count}/{summary['records']} ({percent:.2f}%)", font=font(29), fill='black', anchor='mb')
        draw.text((x, 558), label, font=font(28), fill='black', anchor='mt')
    draw.text((900, 655), 'Reactants: primary-product query; presence is not accuracy', font=font(27), fill='#444444', anchor='mm')
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=HERE / 'llama3_8b_predictions.csv')
    parser.add_argument('--input', type=Path, default=ROOT / 'analysis/data/input.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis/results/zhang_et_al')
    parser.add_argument('--figure', action='store_true', help='Also generate Figure S2 (requires Pillow).')
    args = parser.parse_args()
    summary, rows = summarize(args.predictions, args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    by_id = {int(r['idx']): r for r in rows}
    with (args.output / 'table_s18_examples.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ('idx', 'title', 'paragraph', 'Reactants', 'primary_product', 'all_products')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: by_id[i][k] for k in fields} for i in EXAMPLE_IDS)
    if args.figure:
        plot_summary(summary, args.output / 'figure_s2.png')
    print(json.dumps(summary['counts']))


if __name__ == '__main__':
    main()
