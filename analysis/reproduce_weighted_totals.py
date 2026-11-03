"""Conditional stratum-weighted projections from final per-record benchmark flags.

Run after reproduce_manuscript.py, or on its stored results. Population counts use
the source-derived mutually exclusive frame, not the overlapping Table 1 rows.
Intervals condition on within-stratum representativeness and the fixed model.
They exclude frame-transfer and model-selection uncertainty.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'gpt-5.4-none'
HEURISTICS = ('many_products', 'many_reactants', 'many_reagents', 'no_reagent', 'rare_template')
Z = 1.959963984540054

def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def reproduce(results, population, membership, output):
    results, population, membership, output = map(Path, (results, population, membership, output))
    source = json.loads(population.read_text(encoding='utf-8'))
    assert source['complete_source'] and source['source_rows'] == 1808937
    frame = source['sampling_frame']
    strata = {int(r['idx']): r['stratum'] for r in read(membership)}
    assert len(strata) == 400
    flags = {int(r['source_row']): r for r in read(results/'fully_automatic/table3_flags.csv') if r['model'] == MODEL}
    assert flags.keys() == strata.keys()
    exact = {}
    for scoring in ('lenient', 'strict'):
        exact[scoring] = {int(r['source_row']): int(r['reaction_correct'] or 0)
                          for r in read(results/f'fully_automatic/table2_flags_{scoring}.csv') if r['method'] == MODEL}
        assert exact[scoring].keys() == strata.keys()
    detail, totals = [], []
    for h in HEURISTICS:
        outcomes = {}
        for i, r in flags.items():
            nf, passed = int(r['noise_free']), int(r[h+'_passed'])
            outcomes[i] = {'noise_free_retained': nf*passed, 'noise_free_removed': nf*(1-passed),
                           'noisy_retained': (1-nf)*passed, 'noisy_removed': (1-nf)*(1-passed),
                           'exact_retained_primary': nf*passed*exact['lenient'][i],
                           'exact_retained_strict': nf*passed*exact['strict'][i]}
        for s, N in frame.items():
            ids = [i for i in strata if strata[i] == s]; n = len(ids)
            assert n == 100
            row = {'heuristic': h, 'stratum': s, 'population_n': N, 'sample_n': n, 'weight': N/n}
            for q in outcomes[ids[0]]:
                k = sum(outcomes[i][q] for i in ids)
                row['sample_'+q] = k; row['weighted_'+q] = N*k/n
            detail.append(row)
        for q in next(iter(outcomes.values())):
            value = variance = 0.0
            for s, N in frame.items():
                y = [outcomes[i][q] for i in strata if strata[i] == s]
                n = len(y); p = sum(y)/n
                value += N*p
                variance += N*N*(1-n/N)*p*(1-p)/(n-1)
            se = math.sqrt(variance)
            totals.append({'heuristic': h, 'quantity': q, 'estimate': value, 'se': se,
                           'conditional_ci_low': value-Z*se, 'conditional_ci_high': value+Z*se,
                           'frame_n': sum(frame.values()), 'percent_all_source': 100*value/source['source_rows']})
        selected = [r for r in detail if r['heuristic']==h]
        assert abs(sum(r['weighted_'+q] for r in selected for q in
                       ('noise_free_retained','noise_free_removed','noisy_retained','noisy_removed'))-sum(frame.values()))<1e-6
    output.mkdir(parents=True, exist_ok=True)
    write(output/'weighted_by_stratum.csv', detail); write(output/'weighted_totals.csv', totals)
    summary = {'model':MODEL, 'scenario':'fully_automatic', 'source_rows':source['source_rows'],
               'table1_union':source['table1']['union'], 'frame':frame, 'frame_n':sum(frame.values()),
               'frame_definition':'Exactly one of the four source-count criteria and source template frequency not equal to one. Duplicate and unassigned-template flags are diagnostic.',
               'interpretation':'Conditional projections assuming that benchmark outcome rates transfer within each stratum. Not an independently validated population estimate.',
               'interval_method':'Normal approximation with within-stratum Bernoulli sample variance and finite-population correction. Conditional on representative within-stratum rates and a fixed model. Excludes transfer bias and model-selection uncertainty.',
               'primary_template_anomaly':[r for r in totals if r['heuristic']=='rare_template'],
               'input_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in (
                   ('population_summary',population),
                   ('benchmark_membership',membership),
                   ('table3_flags',results/'fully_automatic/table3_flags.csv'),
                   ('table2_flags_lenient',results/'fully_automatic/table2_flags_lenient.csv'),
                   ('table2_flags_strict',results/'fully_automatic/table2_flags_strict.csv'))}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'analysis/results')
    p.add_argument('--population',type=Path,default=ROOT/'analysis/results/lowe_population/summary.json')
    p.add_argument('--membership',type=Path,default=ROOT/'analysis/data/input.csv',help='Unified benchmark input with idx and stratum.')
    p.add_argument('--output',type=Path)
    a=p.parse_args();print(json.dumps(reproduce(a.results,a.population,a.membership,a.output or a.results/'weighted_estimates'),indent=2))

if __name__=='__main__': main()
