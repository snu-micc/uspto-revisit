"""Recalculate independent binary-label agreement with Python's standard library."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / 'data/independent_assessment/paired_labels.csv'

def summarize(path):
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('No paired judgments')
    ids = [int(row['benchmark_id']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate benchmark IDs')
    labels = ('noise-free', 'noisy')
    pairs = [(row['evaluator_A'], row['evaluator_B']) for row in rows]
    if any(a not in labels or b not in labels for a, b in pairs):
        raise ValueError('Every row must contain two binary judgments')
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    observed = agree / n
    pa = sum(a == 'noisy' for a, b in pairs) / n
    pb = sum(b == 'noisy' for a, b in pairs) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    kappa = (observed - chance) / (1 - chance) if chance != 1 else None
    z = 1.959963984540054
    denominator = 1 + z * z / n
    centre = (observed + z * z / (2 * n)) / denominator
    half = z * math.sqrt(observed * (1 - observed) / n + z * z / (4 * n * n)) / denominator
    counts = Counter(pairs)
    return {
        'paired_binary_n': n,
        'agree': agree,
        'disagree': n - agree,
        'agreement': observed,
        'agreement_percent': 100 * observed,
        'agreement_wilson95': [centre - half, centre + half],
        'cohen_kappa': kappa,
        'contingency': {f'{a} / {b}': counts[a, b] for a in labels for b in labels},
        'paired_labels_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.input), indent=2) + '\n'
    if args.output:
        args.output.write_text(result, encoding='utf-8')
    else:
        print(result, end='')

if __name__ == '__main__':
    main()
