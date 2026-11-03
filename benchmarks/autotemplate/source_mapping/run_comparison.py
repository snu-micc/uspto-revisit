"""Pinned AutoTemplate core, Windows runner; no GT is read during induction/application.

Run under localmapper_env (versions in provenance_verified.json). The vendor tree
(../vendor) is never modified. Source records are checkpointed in SQLite;
timeout/crash/exception records remain in the denominator.

Paths are relative to the repository. The Lowe grant source (not distributed) is
read from analysis/data/lowe_raw_v3/raw/, where
`analysis/rebuild_lowe_population.py --output analysis/data/lowe_raw_v3` puts
it, or from the path in AUTOTEMPLATE_LOWE_SOURCE.
"""
from pathlib import Path
import argparse, ast, collections, csv, hashlib, importlib.util, json
import multiprocessing as mp
from multiprocessing.connection import wait
import os, sqlite3, sys, time, traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = Path(os.environ.get('AUTOTEMPLATE_LOWE_SOURCE',str(ROOT/'analysis/data/lowe_raw_v3/raw/1976_Sep2016_USPTOgrants_smiles.rsmi')))
VENDOR = HERE.parent/'vendor'
sys.path.insert(0, str(ROOT))
from analysis.benchmark_inputs import load_benchmark, input_hashes
CORE = None

def dump(path, value):
    # CRLF on every platform, as in the released files (written on Windows).
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\r\n')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def load_core():
    global CORE
    if CORE is not None: return CORE
    sys.path.insert(0, str(VENDOR))
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    import numpy as np
    from autotemplate.extract_utils import extract_from_rxn_smiles, canon_remap
    from autotemplate.run_utils import ReassignMapping, rdchiralRunText_modified, RemoveReagent, clearIsotope
    spec=importlib.util.spec_from_file_location('pinned_autotemplate_graph', VENDOR/'autotemplate/graph_utils/networkx_GraphMatch.py')
    graph=importlib.util.module_from_spec(spec);spec.loader.exec_module(graph)
    ns=dict(Chem=Chem, extract_from_rxn_smiles=extract_from_rxn_smiles, canon_remap=canon_remap,
            ReassignMapping=ReassignMapping, rdchiralRunText_modified=rdchiralRunText_modified,
            mapping_for_gold_multiple_smiles=graph.mapping_for_gold_multiple_smiles)
    tree=ast.parse((VENDOR/'auto_process.py').read_text(encoding='utf-8'))
    names={'IsInvalidSmiles','remap_all_templates','remap_one_template','check_reconstruct'}
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(definitions)==4
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(VENDOR/'auto_process.py'),'exec'),ns)
    ns.update(graph=graph,np=np,RemoveReagent=RemoveReagent,clearIsotope=clearIsotope)
    CORE=ns
    return ns

def preprocess(smiles):
    c=load_core()
    parts=smiles.split()[0].split('>')
    if len(parts)!=3: raise ValueError('Expected three reaction SMILES sections')
    folded='.'.join(s for s in parts[:2] if s)+'>>'+parts[2]
    result=c['RemoveReagent'](c['clearIsotope'](folded), retain_reagents=True)
    if not isinstance(result,tuple) or len(result)!=2:
        raise ValueError('Upstream RemoveReagent returned non-tuple for no-change reaction')
    return result

def process_record(task, mode, templates):
    idx,smiles=task;c=load_core();c['np'].random.seed(20261027+idx)
    t=time.monotonic()
    out={'idx':idx,'status':None,'template':None,'output':None}
    try:
        rxn,reagents=preprocess(smiles)
        if mode=='extract':
            template=c['extract_from_rxn_smiles'](rxn,radius=0)
            if template and c['check_reconstruct'](template,rxn,retro=True):
                out.update(status='valid_template',template=template)
            else: out['status']='no_reconstructable_template'
        else:
            reactants,products=rxn.split('>>');curated=used=None
            for product in products.split('.'):
                curated,used=c['remap_all_templates'](reactants,product,templates,retro=True,return_ChiralStereo=True)
                if curated: break
            if curated:
                out.update(status='retained',template=used,output=(reagents+'.' if reagents else '')+curated,
                           curated_core=curated,removed_reagents=reagents)
            else: out['status']='no_template_match'
    except Exception as e:
        out.update(status='exception',error=f'{type(e).__name__}: {e}')
    out['seconds']=time.monotonic()-t
    return out

def worker(conn, mode, templates):
    # Vendor debug prints are suppressed, errors are captured structurally.
    sys.stdout=open(os.devnull,'w');sys.stderr=open(os.devnull,'w')
    load_core();conn.send('ready')
    while True:
        task=conn.recv()
        if task is None: return
        conn.send(process_record(task,mode,templates))

def supervised(tasks, mode, templates, workers, timeout):
    """One in-flight record per process; terminate only our timed-out workers."""
    ctx=mp.get_context('spawn');slots={};it=iter(tasks);exhausted=False
    def spawn():
        parent,child=ctx.Pipe();p=ctx.Process(target=worker,args=(child,mode,templates));p.start();child.close()
        slots[parent]={'p':p,'task':None,'start':time.monotonic(),'ready':False}
    for _ in range(workers): spawn()
    try:
        while slots:
            for conn in wait(list(slots),timeout=.1):
                slot=slots[conn]
                try: result=conn.recv()
                except (EOFError,OSError): result=None
                if result=='ready': slot['ready']=True
                elif result is not None:
                    yield result;slot['task']=None
                elif slot['task'] is not None:
                    yield {'idx':slot['task'][0],'status':'worker_crash','template':None,'output':None,'seconds':time.monotonic()-slot['start']}
                    slot['task']=None;slot['ready']=False
                if result is None:
                    slot['p'].join(timeout=1);conn.close();del slots[conn]
                    if not exhausted: spawn()
                    continue
                if slot['ready'] and slot['task'] is None:
                    try: task=next(it)
                    except StopIteration:
                        exhausted=True;conn.send(None);slot['p'].join(timeout=2);conn.close();del slots[conn];continue
                    conn.send(task);slot.update(task=task,start=time.monotonic())
            now=time.monotonic()
            for conn,slot in list(slots.items()):
                limit=timeout if slot['ready'] else 120
                if now-slot['start']>limit:
                    slot['p'].terminate();slot['p'].join(timeout=3)
                    if slot['task'] is not None:
                        yield {'idx':slot['task'][0],'status':'timeout','template':None,'output':None,'seconds':now-slot['start']}
                    else: raise RuntimeError('Worker initialization timed out')
                    conn.close();del slots[conn]
                    if not exhausted: spawn()
    finally:
        for conn,slot in slots.items():
            if slot['p'].is_alive(): slot['p'].terminate()
            slot['p'].join(timeout=3);conn.close()

def source_tasks(limit, completed):
    with SOURCE.open(encoding='utf-8',newline='') as f:
        for idx,row in enumerate(csv.DictReader(f,delimiter='\t')):
            if limit and idx>=limit: break
            if idx not in completed: yield idx,row['ReactionSmiles']

def extract(args):
    db=sqlite3.connect(HERE/args.database)
    db.execute('CREATE TABLE IF NOT EXISTS results(idx INTEGER PRIMARY KEY, status TEXT, template TEXT, seconds REAL, error TEXT)')
    completed={x[0] for x in db.execute('SELECT idx FROM results')}
    count=len(completed);start=time.monotonic();last=start
    for r in supervised(source_tasks(args.limit,completed),'extract',None,args.workers,args.timeout):
        db.execute('INSERT INTO results VALUES(?,?,?,?,?)',(r['idx'],r['status'],r.get('template'),r['seconds'],r.get('error')));count+=1
        if count%1000==0: db.commit()
        if time.monotonic()-last>30:
            db.commit();print(json.dumps({'completed':count,'new_per_second':(count-len(completed))/(time.monotonic()-start)}),flush=True);last=time.monotonic()
    db.commit()
    summary={'rows':count,'limit':args.limit,'seconds_this_run':time.monotonic()-start,'statuses':dict(db.execute('SELECT status,count(*) FROM results GROUP BY status'))}
    dump(HERE/(args.database+'.summary.json'),summary);print(json.dumps(summary),flush=True)

def invariant(graphs):
    def side(g):
        return (g.number_of_nodes(),g.number_of_edges(),tuple(sorted((d['Symbol'],d['Charge'],g.degree(n)) for n,d in g.nodes(data=True))),tuple(sorted(d['weight'] for _,_,d in g.edges(data=True))))
    return side(graphs['reactant']),side(graphs['product'])

def merge_fast(counts):
    """Same upstream loop/order, restricting comparisons by necessary graph invariants."""
    c=load_core();g=c['graph'];unique={};changed={};buckets=collections.defaultdict(dict)
    for template,count in sorted(counts.items(),key=lambda x:x[1],reverse=True):
        p,r=template.split('>>')
        if c['Chem'].MolFromSmarts(p).GetNumAtoms()>c['Chem'].MolFromSmarts(r).GetNumAtoms():continue
        info=g.convert_template2graphs(template);key=invariant(info)
        if unique and info['BCN']==c['np'].inf:continue
        for old in list(buckets[key]):
            prev=unique[old]
            if g.compare_two_graphs(prev,info):
                if prev['BCN']<=info['BCN']:
                    prev['count']+=count;changed[template]=old;break
                elif prev['count']<count:
                    info['count']=count+prev['count'];unique[template]=info;del unique[old]
                    buckets[key][template]=None;del buckets[key][old];changed[old]=template;break
        else:
            info['count']=count;unique[template]=info;buckets[key][template]=None;changed[template]=template
    return {t:v['count'] for t,v in unique.items()},changed

def merge(args):
    db=sqlite3.connect(HERE/args.database)
    counts=dict(db.execute('SELECT template,count(*) FROM results WHERE template IS NOT NULL GROUP BY template ORDER BY min(idx)'))
    # A label-blind parity check, including count ties, precedes accelerated merging.
    test=dict(list(counts.items())[:500]);g=load_core()['graph']
    expected=g.find_unique_templates_dict(test);actual=merge_fast(test)
    assert expected==actual,'Graph-bucket optimization diverged from upstream'
    start=time.monotonic();merged,changes=merge_fast(counts)
    kept=sorted(((t,n) for t,n in merged.items() if n>5),key=lambda x:x[1],reverse=True)
    dump(HERE/args.library,{'templates':kept,'raw_unique':len(counts),'merged_unique':len(merged),'cutoff':'count > 5','parity_templates':len(test),'merge_seconds':time.monotonic()-start,'database_sha256':sha(HERE/args.database)})
    print(json.dumps({'raw_unique':len(counts),'merged_unique':len(merged),'retained':len(kept)}),flush=True)

def apply(args):
    templates=[x[0] for x in json.loads((HERE/args.library).read_text(encoding='utf-8'))['templates']]
    tasks=[(int(r['idx']),r['Lowe_smiles']) for r in load_benchmark()]
    results=[];start=time.monotonic()
    with (HERE/args.output).open('w',encoding='utf-8') as f:
        for r in supervised(tasks,'apply',templates,args.workers,args.timeout):
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();results.append(r)
            if len(results)%10==0:print(json.dumps({'applied':len(results),'seconds':time.monotonic()-start}),flush=True)
    dump(HERE/(args.output+'.summary.json'),{'rows':len(results),'statuses':dict(collections.Counter(r['status'] for r in results)),'seconds':time.monotonic()-start,'library_sha256':sha(HERE/args.library),'input_sha256':input_hashes()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['extract','merge','apply']);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--database',default='source_templates.sqlite');p.add_argument('--library',default='template_library.json');p.add_argument('--output',default='benchmark_outputs.jsonl')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--timeout',type=float,default=60)
    a=p.parse_args();globals()[a.command](a)
