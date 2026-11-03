"""Build curated outputs from the frozen primary outputs plus exactly 46 overrides.

No labels, reference structures, shared name cache, historical lookup tables or
row-level review decisions are read. The explicit resolved_smiles values in the
override file are the curated structures. Only matching component labels change.
Changed reaction steps use the released LocalMapper-202403 mapping cache; an
uncached step requires --allow-map and the mapping extra. No resolver is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import pandas as pd
from analysis import curation_ablation as CA

RULES = ROOT / 'config/nosmi_overrides.json'
PRIMARY = ROOT / 'result/model_outputs/fully_automatic'
CURATED = ROOT / 'result/model_outputs/curated'
MAPPING_CACHE = ROOT / 'result/name_resolution/curated46_mapping.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rules(path=RULES):
    entries = json.loads(Path(path).read_text(encoding='utf-8-sig'))['overrides']
    rules = {}
    for entry in entries:
        key = CA.normalize_name_key(entry['name'])
        if not key or key in rules:
            raise ValueError('Override names must be nonempty and unique')
        value = CA.canonicalize_smiles(entry.get('resolved_smiles', ''))
        if not value:
            raise ValueError('Missing or invalid resolved_smiles: ' + entry['name'])
        rules[key] = {**entry, 'resolved_smiles': value}
    if len(rules) != 46:
        raise ValueError('This sensitivity analysis requires exactly 46 overrides')
    return rules


def override_edits(outputs, rules):
    edits, audit = {}, []
    for model, frame in outputs.items():
        for idx, row in frame.iterrows():
            names = CA.prediction_names(row['prediction'])
            smiles = CA.loads_any(row[f'{model}_smiles'])
            if not isinstance(smiles, dict):
                continue
            for label, name in names.items():
                rule = rules.get(CA.normalize_name_key(name))
                if rule is None or label not in smiles:
                    continue
                before, after = str(smiles[label]), rule['resolved_smiles']
                if CA.canonicalize_smiles(before) == after:
                    continue
                edits.setdefault((model, int(idx)), {})[label] = after
                audit.append(dict(configuration=model, idx=int(idx), label=label,
                                  name=name, override_name=rule['name'], kind=rule['kind'],
                                  before=before, after=after))
    return edits, audit


def build(primary=PRIMARY, output=CURATED, rules_path=RULES,
          mapping_cache=MAPPING_CACHE, allow_map=False):
    rules = load_rules(rules_path)
    paths = sorted(Path(primary).glob('*_reaction_smiles_final.csv'))
    if len(paths) != 19:
        raise ValueError('Expected the 19 frozen primary configurations')
    outputs = {p.name.removesuffix('_reaction_smiles_final.csv'):
               pd.read_csv(p, encoding='utf-8-sig', dtype=str, keep_default_na=False) for p in paths}
    if any(len(f) != 400 for f in outputs.values()):
        raise ValueError('Each configuration must contain all 400 records')
    edits, audit = override_edits(outputs, rules)
    cache = json.loads(Path(mapping_cache).read_text(encoding='utf-8')) if Path(mapping_cache).exists() else {}
    records = cache.get('reactions', {})
    used = {}

    def remap(reaction):
        if reaction not in records:
            if not allow_map:
                raise ValueError('Uncached reaction: use --allow-map with LocalMapper 202403')
            records[reaction] = CA.localmapper_remap(reaction)
        used[reaction] = records[reaction]
        return records[reaction]

    rebuilt = CA.counterfactual_outputs(outputs, edits, remap=remap)
    # Guard the scientific comparison: every component difference is one of the
    # 46 rules, and every non-derived column is byte-for-byte equal as text.
    derived = ('smiles', 'rxn', *CA.MAPPING_COLUMNS)
    for model, frame in rebuilt.items():
        original = outputs[model]
        fixed = [c for c in original if c not in {f'{model}_{x}' for x in derived}]
        if not frame[fixed].equals(original[fixed]):
            raise AssertionError('Non-derived data changed: ' + model)
        for idx in frame.index:
            old, new = (CA.loads_any(f.at[idx, f'{model}_smiles']) for f in (original, frame))
            if old != new:
                if set(old) != set(new):
                    raise AssertionError('Component labels changed')
                expected = edits.get((model, idx), {})
                if {k: v for k, v in new.items() if old[k] != v} != expected:
                    raise AssertionError('Unapproved structure edit')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for model, frame in rebuilt.items():
        frame.to_csv(output / f'{model}_reaction_smiles_final.csv', index=False, encoding='utf-8-sig')
    report = dict(definition='fully_automatic plus exactly 46 name overrides',
                  override_count=46, historical_lookup_tables_used=False,
                  row_decisions_used=False, shared_name_cache_used=False,
                  ground_truth_read=False, model_calls=0, resolver_calls=0,
                  changed_components=len(audit), changed_configuration_rows=len(edits),
                  used_override_names=sorted({x['override_name'] for x in audit}),
                  primary_sha256={p.name:sha(p) for p in paths}, rules_sha256=sha(rules_path),
                  output_sha256={p.name:sha(p) for p in sorted(output.glob('*.csv'))},
                  changes=audit)
    Path(mapping_cache).parent.mkdir(parents=True, exist_ok=True)
    Path(mapping_cache).write_text(json.dumps(dict(mapper='LocalMapper', model_version='202403',
        reactions=used), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report['mapping_cache_sha256'] = sha(mapping_cache)
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--primary', type=Path, default=PRIMARY)
    parser.add_argument('--output', type=Path, default=CURATED)
    parser.add_argument('--rules', type=Path, default=RULES)
    parser.add_argument('--mapping-cache', type=Path, default=MAPPING_CACHE)
    parser.add_argument('--allow-map', action='store_true')
    args = parser.parse_args()
    report = build(args.primary, args.output, args.rules, args.mapping_cache, args.allow_map)
    print(json.dumps({k:report[k] for k in ['definition','override_count','changed_components','changed_configuration_rows']}))


if __name__ == '__main__':
    main()
