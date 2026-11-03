"""Check per-record parity against released frozen flags before new scoring.

The original-USPTO and gpt-5.4-none flags of the fully automatic scenario are rebuilt offline by
../baseline_flags.py, or read with --baseline-flags-dir from the files `python analysis/reproduce_manuscript.py`
writes to analysis/results/fully_automatic/. Writes scorer_validation.json to --output-dir
(default: this directory).
"""
from pathlib import Path
import argparse,sys,json
import pandas as pd
from run_comparison import HERE,ROOT,dump,sha
for p in [ROOT/'src',ROOT,HERE.parent]:
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from analysis import evaluate_table2 as T2,evaluate_table3 as T3
from analysis.reproduce_manuscript import load_ground_truth
from analysis.benchmark_inputs import load_benchmark_frame
import baseline_flags as BF

def validate(out,baseline):
    data=ROOT/'analysis/data'
    gt=load_ground_truth()
    inputs=load_benchmark_frame()
    report={}
    for strict,name in [(False,'lenient'),(True,'strict')]:
        T3.configure_matching(strict_structures=strict,first_reference_only=True)
        flags=baseline[f'table2_flags_{name}']
        result={}
        for method in ['original_data','gpt-5.4-none']:
            rr=flags[flags.method.eq(method)].set_index('source_row');correct=0;checked=0
            for idx,r in gt.iterrows():
                if r.label!='noise-free':continue
                eligible=r.eligible
                truth=T3.signature_from_row(r,'ground_truth_') if eligible else None
                if method=='original_data':pred=T2.prediction_signature(T2.classify_lowe_components(inputs.loc[idx,'Lowe_smiles']))
                else:
                    txt=rr.loc[idx,'predicted_signature'];o=json.loads(txt) if txt else {k:[] for k in T3.COMPONENT_FIELDS}
                    pred=tuple(tuple(o[k]) for k in T3.COMPONENT_FIELDS)
                exact=int(bool(eligible and T2.compare_signatures(pred,truth)['component_exact']))
                if int(rr.loc[idx,'has_unresolved_nosmi'] or 0):exact=0
                assert exact==int(rr.loc[idx,'reaction_correct'] or 0),(name,method,idx)
                correct+=exact;checked+=1
            result[method]={'per_record_flags_checked':checked,'correct':correct}
        report[name]=result
    T3.configure_matching()
    assert report['lenient']['original_data']['correct']==99
    assert report['strict']['original_data']['correct']==95
    assert report['lenient']['gpt-5.4-none']['correct']==209
    assert report['strict']['gpt-5.4-none']['correct']==205
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    dump(out/'scorer_validation.json',{'all_per_record_flags_reproduced':True,'results':report,'ground_truth_sha256':sha(data/'ground_truth.csv'),'scorer_sha256':{Path(p).resolve().relative_to(ROOT).as_posix():sha(Path(p)) for p in [T2.__file__,T3.__file__]}})
    return report

def main(argv=None,baseline=None):
    p=argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--output-dir',type=Path,default=HERE)
    p.add_argument('--baseline-flags-dir',type=Path,help='read the flags written by analysis/reproduce_manuscript.py instead of rebuilding them')
    a=p.parse_args(argv)
    report=validate(a.output_dir,baseline if baseline is not None else BF.load(a.baseline_flags_dir))
    print(json.dumps(report,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
