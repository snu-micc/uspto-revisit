"""Matched frozen-reference scoring. No model or resolver calls.

Reads benchmark_outputs.jsonl and protocol.json next to this file, the benchmark in
analysis/data/ and the per-record original-USPTO and gpt-5.4-none flags of the fully automatic
scenario (rebuilt offline by ../baseline_flags.py, or read with --baseline-flags-dir from the files
`python analysis/reproduce_manuscript.py` writes to analysis/results/fully_automatic/). Writes
comparison_flags.csv, comparison_metrics.csv, component_metrics.csv, paired_tests.csv, retention_metrics.csv,
comparison_summary.json and RESULTS.md to --output-dir (default: this directory, i.e. the released files).

    python benchmarks/autotemplate/source_mapping/score_comparison.py --output-dir <dir>
"""
from pathlib import Path
import argparse,collections,json,sys
import numpy as np
import pandas as pd
from run_comparison import HERE,ROOT,dump,sha
for p in [ROOT/'src',ROOT,HERE.parent]:
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from analysis import evaluate_table2 as T2,evaluate_table3 as T3,benchmark_statistics as BS
from analysis.reproduce_manuscript import pct,pct_float,load_ground_truth
import baseline_flags as BF
from analysis.benchmark_inputs import load_benchmark_frame

DATA=ROOT/'analysis/data'
PREDICTIONS=HERE/'benchmark_outputs.jsonl'
METHOD='AutoTemplate_source_mapping'

def load_inputs():
    gt=load_ground_truth()
    inputs=load_benchmark_frame()
    assert len(gt)==400 and gt.label.eq('noise-free').sum()==292
    assert gt.eligible.sum()==291
    assert gt.component_reference.sum()==292
    assert gt[[c for c in gt if c.startswith('ground_truth_2_')]].eq('').all(axis=None)
    pred={r['idx']:r for r in (json.loads(s) for s in PREDICTIONS.read_text(encoding='utf-8').splitlines())}
    assert set(pred)==set(range(400))
    return gt,inputs,pred

def pooled(sig):return tuple(sorted(set(sig[0]+sig[1]))),tuple(sorted(set(sig[2])))
def from_text(s):
    o=json.loads(s);return tuple(tuple(o[k]) for k in T3.COMPONENT_FIELDS)
def metric(endpoint,method,y,ids):
    values=np.asarray(y,dtype=int);n=len(values);k=int(values.sum());lo,hi=BS.wilson(k,n)
    return {'endpoint':endpoint,'method':method,'correct':k,'denominator':n,'accuracy':k/n,'ci95_low':lo,'ci95_high':hi}
def write_csv(rows,path):
    # CRLF on every platform, as in the released files (written on Windows).
    pd.DataFrame(rows).to_csv(path,index=False,lineterminator='\r\n')

def score(out,baseline):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    GT,INPUT,PRED=load_inputs()
    allflags=[];metrics=[];paired=[];micro=[];vectors={};parity={}
    nf=GT.index[GT.label.eq('noise-free')].tolist();allids=list(GT.index)
    stored3=baseline['table3_flags']
    gpt3=stored3[stored3.model.eq('gpt-5.4-none')].set_index('source_row').sort_index()
    assert len(gpt3)==400
    gpt_pass=gpt3.rare_template_passed.astype(int).to_dict()
    for strict,name,expected_source,expected_gpt in [(False,'primary',99,209),(True,'strict',95,205)]:
        T3.configure_matching(strict_structures=strict,first_reference_only=True)
        baseline2=baseline[f'table2_flags_{"strict" if strict else "lenient"}']
        gpt=baseline2[baseline2.method.eq('gpt-5.4-none')].set_index('source_row').sort_index()
        orig=baseline2[baseline2.method.eq('original_data')].set_index('source_row').sort_index()
        assert int(gpt.loc[nf,'reaction_correct'].astype(int).sum())==expected_gpt
        assert int(orig.loc[nf,'reaction_correct'].astype(int).sum())==expected_source
        flag_by_method={m:{} for m in [METHOD,'Original_USPTO','GPT_5.4_none']}
        for idx,row in GT.iterrows():
            eligible=row.eligible;noise_free=row.label=='noise-free'
            truth=T3.signature_from_row(row,'ground_truth_') if eligible else None
            # Record 215 has a partial reference: components only, never exact.
            component_truth=truth if eligible else (T3.partial_signature_from_row(row) if row.component_reference else None)
            result=PRED[idx];retained=result['status']=='retained';completed=result['status'] in ['retained','no_template_match']
            comps=T2.classify_lowe_components(result['curated_core']) if retained else {r:[] for r in T3.COMPONENT_FIELDS}
            if retained:comps['reagents'] += [s for s in result.get('removed_reagents','').split('.') if s]
            sigs={METHOD:T2.prediction_signature(comps),'Original_USPTO':T2.prediction_signature(T2.classify_lowe_components(INPUT.loc[idx,'Lowe_smiles']))}
            # GPT's persisted normalized signature is frozen and checked against the
            # fixed current reference, rather than regenerating any resolver output.
            txt=gpt.loc[idx,'predicted_signature'];sigs['GPT_5.4_none']=from_text(txt) if txt else ((),(),())
            naive=T2.prediction_signature(T2.classify_lowe_components(result['output'])) if retained else ((),(),())
            for method,sig in sigs.items():
                counts=T2.compare_signatures(sig,component_truth) if component_truth is not None else None
                exact=int(bool(eligible and counts['component_exact']))
                if method==METHOD and not retained:exact=0
                if method=='GPT_5.4_none':
                    persisted=int(gpt.loc[idx,'reaction_correct'] or 0)
                    # Unresolved molecular placeholders cannot count as exact.
                    if int(gpt.loc[idx,'has_unresolved_nosmi'] or 0):exact=0
                    assert exact==persisted,(name,idx,exact,persisted)
                if method=='Original_USPTO':assert exact==int(orig.loc[idx,'reaction_correct'] or 0),(name,idx)
                pool=int(bool(eligible and pooled(sig)==pooled(truth)))
                if method==METHOD:
                    passed=retained;denoise=int(completed and (passed==noise_free));status=result['status']
                elif method=='GPT_5.4_none':passed=bool(gpt_pass[idx]);denoise=int(passed==noise_free);status='persisted_pipeline_decision'
                else:passed=True;denoise=int(noise_free);status='unchanged_source'
                end=int((noise_free and exact and passed) or (not noise_free and not passed and (completed if method==METHOD else True)))
                out_row={'idx':idx,'scoring':name,'method':method,'label':row.label,'structure_eligible':int(eligible),'status':status,'retained':int(passed),'exact':exact,'pooled_set_exact':pool,'denoising_correct':denoise,'end_to_end_correct':end,'predicted_signature':T3.signature_text(sig)}
                if method==METHOD:
                    out_row['naive_merged_smiles_exact']=int(bool(eligible and T2.compare_signatures(naive,truth)['component_exact']))
                    out_row['reagent_map_collision_affects_roles']=int(retained and sig!=naive)
                if counts:
                    for k,v in counts.items():
                        if k!='_score':out_row[k]=v
                allflags.append(out_row);flag_by_method[method][idx]=out_row
        for method,rr in flag_by_method.items():
            for suffix,field,ids in [('structure','exact',nf),('pooled_set','pooled_set_exact',nf),('end_to_end','end_to_end_correct',allids)]:
                endpoint=f'{name}_{suffix}';vec=[rr[i][field] for i in ids];vectors[endpoint,method]=vec
                if endpoint=='primary_end_to_end' and method=='GPT_5.4_none':assert sum(vec)==240
                metrics.append(metric(endpoint,method,vec,ids))
            if not strict:
                vec=[rr[i]['denoising_correct'] for i in allids];vectors['denoising',method]=vec
                if method=='GPT_5.4_none':assert sum(vec)==283
                metrics.append(metric('denoising',method,vec,allids))
            for role in T3.COMPONENT_FIELDS:
                totals={key:sum(int(rr[i].get(f'{role}_{key}',0)) for i in nf if GT.loc[i,'component_reference']) for key in ['tp','fp','fn']}
                tp,fp,fn=[totals[k] for k in ['tp','fp','fn']]
                micro.append({'scoring':name,'method':method,'role':role,'reference_denominator':292,**totals,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None})
        parity[name]={'original_exact_reproduced':expected_source,'gpt_exact_reproduced':expected_gpt,'all_original_and_gpt_per_record_flags_equal':True}
    T3.configure_matching()

    for endpoint in sorted({k[0] for k in vectors}):
        tests=[]
        for comparator in ['Original_USPTO','GPT_5.4_none']:
            a=vectors[endpoint,METHOD];b=vectors[endpoint,comparator]
            only_a,only_b,p=BS.mcnemar_exact(a,b)
            tests.append({'endpoint':endpoint,'comparator':comparator,'n':len(a),'autotemplate_only_correct':only_a,'comparator_only_correct':only_b,'p_exact':p})
        for row,adj in zip(tests,BS.holm([r['p_exact'] for r in tests])):row['p_holm_within_endpoint']=float(adj)
        paired.extend(tests)
    write_csv(allflags,out/'comparison_flags.csv')
    write_csv(metrics,out/'comparison_metrics.csv')
    write_csv(micro,out/'component_metrics.csv')
    write_csv(paired,out/'paired_tests.csv')
    status=dict(collections.Counter(r['status'] for r in PRED.values()))
    retention=[]
    for method in [METHOD,'Original_USPTO','GPT_5.4_none']:
        rr=[r for r in allflags if r['scoring']=='primary' and r['method']==method]
        unresolved=lambda r:method==METHOD and r['status'] not in ['retained','no_template_match']
        retention.append({'method':method,'noise_free_retained':sum(r['label']=='noise-free' and r['retained'] for r in rr),'noise_free_denominator':292,'noisy_rejected_completed':sum(r['label']=='noisy' and not r['retained'] and not unresolved(r) for r in rr),'noisy_denominator':108,'execution_unresolved':sum(unresolved(r) for r in rr),'retained_total':sum(r['retained'] for r in rr)})
    write_csv(retention,out/'retention_metrics.csv')
    summary={'method':'AutoTemplate core with source atom mapping','statuses':status,'benchmark_rows':400,'noise_free_denominator':292,'complete_reference_denominator':291,'component_reference_denominator':292,'metrics':metrics,'retention':retention,'scorer_parity':parity,'protocol_sha256':sha(HERE/'protocol.json'),'outputs_sha256':sha(PREDICTIONS),'source_gt_sha256':sha(DATA/'ground_truth.csv'),'scorer_sha256':{Path(p).resolve().relative_to(ROOT).as_posix():sha(Path(p)) for p in [T2.__file__,T3.__file__]},'limitations':['Transductive source-mapped comparator; no RXNMapper remapping','Same flagged 400-record development/selection benchmark, not an independent test','Different method inputs; operational retention does not validate paragraph interpretability','Unflagged sample and targeted human review have no completed labels or model results','Upstream isotope clearing and first-successful-product behavior preserved','Timeouts/crashes/exceptions are incorrect, not successful chemical rejections']}
    dump(out/'comparison_summary.json',summary)
    lines=['# Matched AutoTemplate comparison','','AutoTemplate core at pinned upstream commit, using source atom mapping. Corpus-informed comparison on the existing flagged benchmark; not independent validation.','','| Endpoint | Method | Correct / denominator | Accuracy (95% Wilson CI) |','|---|---|---:|---:|']
    for m in metrics:lines.append(f"| {m['endpoint']} | {m['method']} | {m['correct']}/{m['denominator']} | {pct(m['correct'],m['denominator'])}% ({pct_float(m['ci95_low'])}–{pct_float(m['ci95_high'])}) |")
    lines+=['','Execution statuses: '+json.dumps(status)+'.','','All 292 noise-free paragraphs remain in extraction denominators. The one incomplete structural reference and all filtered/failed outputs count as non-exact. Component precision/recall uses all 292 noise-free paragraphs; the two reagents of record 215 without a molecular structure count as missed. The pooled reactant/reagent set endpoint is a separate sensitivity analysis.','','Exact paired McNemar tests and Holm correction within each endpoint are in `paired_tests.csv`. GPT-5.4-none was selected on this same benchmark, so these are internal comparisons.','','Both primary and strict original-source/GPT per-record flags were reproduced before scoring new outputs. No labels, references or prior model outputs were modified.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\r\n')
    return {'statuses':status,'metrics':metrics}

def main(argv=None,baseline=None):
    p=argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--output-dir',type=Path,default=HERE,help='default: this directory (overwrites the released files)')
    p.add_argument('--baseline-flags-dir',type=Path,help='read the flags written by analysis/reproduce_manuscript.py instead of rebuilding them')
    a=p.parse_args(argv)
    result=score(a.output_dir,baseline if baseline is not None else BF.load(a.baseline_flags_dir))
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
