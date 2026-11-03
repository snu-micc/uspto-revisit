"""Offline single-evaluator label and pooled input/product-set assessment.

Run after reproduce_manuscript.py, using the recorded RDKit environment. This
supplementary endpoint leaves the main benchmark's role-specific scores intact.
All 33 evaluator-labeled noise-free records have supplied reference structures
and are evaluated, including records 163 and 174. Labels and structures define
the assessment; no availability flags or model-dependent exclusions are used.

summary.json holds the label agreement and the agreement between the evaluator's
and the benchmark's references. model_comparison.csv scores gpt-5.4-none and the
original USPTO record against each reference (evaluator or benchmark).
"""
from pathlib import Path
import argparse,csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis import evaluate_table3 as T3,benchmark_statistics as BS
DEFAULT=ROOT/'analysis/data/blinded_assessment/labels_and_structures.csv'
RESULTS=ROOT/'analysis/results/fully_automatic'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def pooled(s):return set(s['reactants'])|set(s['reagents']),set(s['products'])
def decode(s):return json.loads(s)
def correct(rows,refs):
    return [int(pooled(decode(r['predicted_signature']))==pooled(ref) and not int(r['has_unresolved_nosmi'])) for r,ref in zip(rows,refs)]
def kappa(pairs):
    n=len(pairs);observed=sum(a==b for a,b in pairs)/n
    chance=sum(sum(a==l for a,_ in pairs)/n*sum(b==l for _,b in pairs)/n for l in ('noise-free','noisy'))
    return (observed-chance)/(1-chance)
def run(data,out):
    if out.exists():raise FileExistsError('Use a new output directory.')
    rows=read(data);gt={int(r['idx']):r for r in read(ROOT/'analysis/data/ground_truth.csv')}
    assert len(rows)==50 and len({r['benchmark_id'] for r in rows})==50
    complete={int(r['benchmark_id']):r for r in rows if r['label']=='noise-free'}
    assert len(complete)==33
    if any(not r['reactants'].strip() or not r['products'].strip() for r in complete.values()):
        raise ValueError('Every confirmed noise-free reference requires reactant and product structures.')
    n_agree=sum(r['label']==gt[int(r['benchmark_id'])]['label'] for r in rows)
    label_kappa=kappa([(r['label'],gt[int(r['benchmark_id'])]['label']) for r in rows])
    comparison=[];agreement={};ids=sorted(complete)
    for scoring in ['lenient','strict']:
        T3.configure_matching(strict_structures=scoring=='strict',first_reference_only=True)
        evaluator=[decode(T3.signature_text(T3.component_signature(**{k:[x for x in complete[i][k].split('.') if x] for k in T3.COMPONENT_FIELDS}))) for i in ids]
        source=read(RESULTS/f'table2_flags_{scoring}.csv')
        model,uspto=(sorted((r for r in source if r['method']==m and int(r['source_row']) in complete),key=lambda r:int(r['source_row'])) for m in (BS.PRIMARY,'original_data'))
        assert [int(r['source_row']) for r in model]==[int(r['source_row']) for r in uspto]==ids
        benchmark=[decode(r['ground_truth_signature']) for r in uspto]
        sets=[(pooled(a),pooled(b)) for a,b in zip(evaluator,benchmark)]
        agreement[scoring]={'input_set_exact':sum(a[0]==b[0] for a,b in sets),'product_set_exact':sum(a[1]==b[1] for a,b in sets),'both_sets_exact':sum(a==b for a,b in sets)}
        for reference,refs in [('evaluator',evaluator),('benchmark',benchmark)]:
            model_correct,uspto_correct=correct(model,refs),correct(uspto,refs)
            model_only,uspto_only,p=BS.mcnemar_exact(model_correct,uspto_correct)
            comparison.append({'scoring':scoring,'reference':reference,'n':len(ids),'gpt-5.4_correct':sum(model_correct),'USPTO_correct':sum(uspto_correct),
                'gpt-5.4_only_correct':model_only,'USPTO_only_correct':uspto_only,'p_exact':p})
    out.mkdir(parents=True)
    write(out/'model_comparison.csv',comparison)
    summary={'labels':50,'noise_free':33,'noisy':17,'label_agree':n_agree,
        'label_agreement':n_agree/50,'label_agreement_wilson95':BS.wilson(n_agree,50),'label_cohen_kappa':label_kappa,
        'reference_agreement':agreement}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=DEFAULT)
    p.add_argument('--out',type=Path,default=ROOT/'analysis/results/blinded_assessment')
    a=p.parse_args();run(a.input.resolve(),a.out.resolve())
