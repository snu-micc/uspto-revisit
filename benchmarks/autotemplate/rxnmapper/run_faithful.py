"""Upstream-faithful AutoTemplate run on RXNMapper-remapped reactions; run under localmapper_env.

Reuses ../source_mapping/run_comparison.py (pinned core, supervised workers, verified
graph merge). Differences from that source-mapped run all move toward the upstream
workflow: official RXNMapper 0.3.0 maps, long per-record limits in place of 60 s, and
the --retain_reagents "no change" early return handled as the CLI's no-retain path
handles it (the reaction passes through with nothing removed) rather than as a failure.
The CLI never reaches its skip-top-3 branch (its per-record template stays None), so
that branch is reported only as a separate variant.

The corpus read by `extract` is rxnmapper_corpus.tsv, written next to this file by
`map_rxnmapper.py export` (not distributed). Paths are relative to the repository.
"""
from pathlib import Path
import argparse, ast, json, os, sqlite3, sys, time

HERE = Path(__file__).resolve().parent
# The runner reads the remapped corpus, never the original Lowe file named by this variable elsewhere.
os.environ['AUTOTEMPLATE_LOWE_SOURCE'] = str(HERE/'rxnmapper_corpus.tsv')
sys.path.insert(0, str(HERE.parent/'source_mapping'))
import run_comparison as rc

ORIGINAL_WORKER, ORIGINAL_PROCESS = rc.worker, rc.process_record
NO_CHANGE = False

def preprocess(smiles):
    global NO_CHANGE
    c=rc.load_core()
    parts=smiles.split()[0].split('>')
    if len(parts)!=3: raise ValueError('Expected three reaction SMILES sections')
    folded='.'.join(s for s in parts[:2] if s)+'>>'+parts[2]
    result=c['RemoveReagent'](c['clearIsotope'](folded), retain_reagents=True)
    if isinstance(result,str):
        NO_CHANGE=True
        return result,''
    return result

def process_record(task, mode, templates):
    global NO_CHANGE
    NO_CHANGE=False
    out=ORIGINAL_PROCESS(task,mode,templates)
    out['no_change_return']=NO_CHANGE
    return out

def worker(conn, mode, templates):
    # Spawned children import run_comparison afresh; patch its module globals there.
    rc.preprocess=preprocess;rc.process_record=process_record
    ORIGINAL_WORKER(conn,mode,templates)

def extract(args):
    rc.worker=worker
    rc.extract(argparse.Namespace(database=str(HERE/args.database),limit=args.limit,workers=args.workers,timeout=args.timeout))

def merge(args):
    """rc.merge, additionally keeping the raw-to-merged template map for the skip-top-3 variant."""
    db=sqlite3.connect(HERE/args.database)
    counts=dict(db.execute('SELECT template,count(*) FROM results WHERE template IS NOT NULL GROUP BY template ORDER BY min(idx)'))
    test=dict(list(counts.items())[:500]);g=rc.load_core()['graph']
    assert g.find_unique_templates_dict(test)==rc.merge_fast(test),'Graph-bucket optimization diverged from upstream'
    start=time.monotonic();merged,changes=rc.merge_fast(counts)
    kept=sorted(((t,n) for t,n in merged.items() if n>5),key=lambda x:x[1],reverse=True)
    rc.dump(HERE/args.library,{'templates':kept,'raw_unique':len(counts),'merged_unique':len(merged),'cutoff':'count > 5','parity_templates':len(test),'merge_seconds':time.monotonic()-start,'database_sha256':rc.sha(HERE/args.database)})
    rc.dump(HERE/args.changes,changes)
    print(json.dumps({'raw_unique':len(counts),'merged_unique':len(merged),'retained':len(kept)}),flush=True)

def give_mapnum():
    """Upstream give_mapnum_for_unmapped_reactant_atoms, without its module's CGRtools import."""
    path=rc.VENDOR/'autotemplate/graph_utils/reaction_balancing.py'
    fn=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='give_mapnum_for_unmapped_reactant_atoms']
    ns={'Chem':rc.load_core()['Chem']};exec(compile(ast.Module(body=fn,type_ignores=[]),str(path),'exec'),ns)
    return ns['give_mapnum_for_unmapped_reactant_atoms']

def representative(changes, template):
    seen=set()
    while template in changes and changes[template]!=template and template not in seen:
        seen.add(template);template=changes[template]
    return template

def write(path, rows):
    with path.open('w',encoding='utf-8') as f:
        for r in sorted(rows,key=lambda r:r['idx']): f.write(json.dumps(r,ensure_ascii=False)+'\n')

def apply(args):
    rc.worker=worker
    templates=[t for t,_ in json.loads((HERE/args.library).read_text(encoding='utf-8'))['templates']]
    mapped=[json.loads(s) for s in (HERE/args.benchmark).read_text(encoding='utf-8').splitlines()]
    tasks=[(r['idx'],r['mapped_rxn']) for r in mapped if r['status']=='mapped']
    failed=[{'idx':r['idx'],'status':r['status'],'template':None,'output':None,'seconds':0.0} for r in mapped if r['status']!='mapped']
    start=time.monotonic();own={}
    for r in rc.supervised(tasks,'extract',None,args.workers,args.extract_timeout): own[r['idx']]=r
    primary=list(failed)
    with (HERE/(args.output+'.partial')).open('w',encoding='utf-8') as f:
        for r in rc.supervised(tasks,'apply',templates,args.workers,args.timeout):
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();primary.append(r)
            if len(primary)%10==0:print(json.dumps({'applied':len(primary),'seconds':time.monotonic()-start}),flush=True)
    write(HERE/args.output,primary);(HERE/(args.output+'.partial')).unlink()
    # Variant: upstream's intended skip of records whose own template is among the top three.
    changes=json.loads((HERE/args.changes).read_text(encoding='utf-8'));top=set(templates[:3]);mapnum=give_mapnum()
    variant=[]
    for r in primary:
        o=own.get(r['idx'],{});t=o.get('template')
        if t and representative(changes,t) in top:
            rxn,reagents=preprocess(dict(tasks)[r['idx']]);curated=mapnum(rxn)
            variant.append({'idx':r['idx'],'status':'retained','template':representative(changes,t),'output':(reagents+'.' if reagents else '')+curated,
                            'curated_core':curated,'removed_reagents':reagents,'kept_by_top3':True,'seconds':o.get('seconds')})
        else: variant.append({**r,'kept_by_top3':False})
    write(HERE/args.variant_output,variant)
    status=lambda rows:{s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}
    rc.dump(HERE/(args.output+'.summary.json'),{'rows':len(primary),'statuses':status(primary),'no_change_return':sum(bool(r.get('no_change_return')) for r in primary),
        'own_template_statuses':status(list(own.values())),'skip_top3_kept':sum(r['kept_by_top3'] for r in variant),'skip_top3_statuses':status(variant),
        'seconds':time.monotonic()-start,'timeout':args.timeout,'extract_timeout':args.extract_timeout,'workers':args.workers,
        'library_sha256':rc.sha(HERE/args.library),'benchmark_maps_sha256':rc.sha(HERE/args.benchmark)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['extract','merge','apply']);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--database',default='rxnmapper_templates.sqlite');p.add_argument('--library',default='rxnmapper_template_library.json')
    p.add_argument('--changes',default='rxnmapper_template_changes.json');p.add_argument('--benchmark',default='rxnmapper_benchmark.jsonl')
    p.add_argument('--output',default='rxnmapper_benchmark_outputs.jsonl');p.add_argument('--variant_output',default='rxnmapper_skip_top3_outputs.jsonl')
    p.add_argument('--workers',type=int,default=20);p.add_argument('--timeout',type=float,default=3600);p.add_argument('--extract_timeout',type=float,default=600)
    a=p.parse_args();globals()[a.command](a)
