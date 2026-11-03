"""Official RXNMapper 0.3.0 remapping step of the upstream AutoTemplate workflow.

Run with rxnmapper_v2_env's interpreter. The official PyPI wheel unpacked in
_deps/site shadows that environment's RXNMapper_v2 fork. Lowe reactant and agent
sections are folded as in the source-mapped run, original Indigo maps are removed
(Reaxys inputs are unmapped), and upstream ValidRxnSmiles is applied unchanged.
Upstream silently drops mapping failures; here they are recorded and stay failures.

Paths are relative to the repository. _deps/ (the wheel in _deps/wheels and its
unpacked contents in _deps/site) sits next to this file and is not distributed. The
Lowe grant source (not distributed) is read from analysis/data/lowe_raw_v3/raw/,
where `analysis/rebuild_lowe_population.py --output analysis/data/lowe_raw_v3`
puts it, or from the path in AUTOTEMPLATE_LOWE_SOURCE.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, multiprocessing as mp, os, sqlite3, sys, time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'_deps/site'))
ROOT = HERE.parents[2]
VENDOR = HERE.parent/'vendor'
SOURCE = Path(os.environ.get('AUTOTEMPLATE_LOWE_SOURCE', str(ROOT/'analysis/data/lowe_raw_v3/raw/1976_Sep2016_USPTOgrants_smiles.rsmi')))
sys.path.insert(0, str(ROOT))
from analysis.benchmark_inputs import load_benchmark, input_hashes
csv.field_size_limit(2**31-1)
MAPPER = VALID = None
BATCH = 8

def dump(path, value):
    # CRLF on every platform, as in the released files (written on Windows).
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\r\n')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def init(batch=None, gpu_fraction=None):
    global MAPPER, VALID, BATCH
    if batch: BATCH=batch
    if MAPPER is not None: return
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    if gpu_fraction:
        # The Windows driver pages an overfull GPU into system RAM, which stalled all workers;
        # a hard cap turns that into an OOM, handled in map_chunk by mapping one reaction at a time.
        import torch;torch.cuda.set_per_process_memory_fraction(gpu_fraction)
    # Transformers repeats its SDPA-fallback notice on every batch; mapping is unaffected.
    import transformers
    transformers.logging.set_verbosity_error()
    import rxnmapper
    assert Path(rxnmapper.__file__).is_relative_to(HERE/'_deps/site'), rxnmapper.__file__
    spec=importlib.util.spec_from_file_location('upstream_rxnmapper_mapping', VENDOR/'rxnmapper_mapping.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    VALID=module.ValidRxnSmiles;MAPPER=rxnmapper.RXNMapper()

def prepare(lowe):
    """Lowe reactants>agents>products -> unmapped reactants>>products, then upstream ValidRxnSmiles."""
    from rdkit import Chem
    parts=lowe.split()[0].split('>') if lowe.strip() else []
    if len(parts)!=3: return None
    def unmapped(side):
        mol=Chem.MolFromSmiles(side)
        if mol is None: return None
        for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol)
    left,right=unmapped('.'.join(s for s in parts[:2] if s)),unmapped(parts[2])
    if not left or not right: return None
    return VALID(left+'>>'+right) or None

def map_one(rxn):
    try:
        r=MAPPER.get_attention_guided_atom_maps([rxn])[0]
        return 'mapped',r['mapped_rxn'],r['confidence'],None
    except Exception as e:
        return 'mapping_failed',None,None,f'{type(e).__name__}: {e}'[:300]

def map_chunk(chunk):
    """Batched for throughput; a failing batch is redone one reaction at a time, as upstream calls it."""
    init();out={};ready=[]
    for idx,lowe in chunk:
        try: rxn=prepare(lowe)
        except Exception as e: rxn=None
        if rxn: ready.append((idx,rxn))
        else: out[idx]=('invalid_input',None,None,None)
    ready.sort(key=lambda x:len(x[1]))
    for i in range(0,len(ready),BATCH):
        part=ready[i:i+BATCH]
        try:
            res=MAPPER.get_attention_guided_atom_maps([r for _,r in part])
            for (idx,_),r in zip(part,res): out[idx]=('mapped',r['mapped_rxn'],r['confidence'],None)
        except Exception:
            import torch;torch.cuda.empty_cache()
            for idx,rxn in part: out[idx]=map_one(rxn)
    return [(idx,*out[idx]) for idx,_ in chunk]

def source_rows(skip):
    with SOURCE.open(encoding='utf-8',newline='') as f:
        for idx,row in enumerate(csv.DictReader(f,delimiter='\t')):
            if idx not in skip: yield idx,row['ReactionSmiles']

def chunks(rows, size):
    block=[]
    for row in rows:
        block.append(row)
        if len(block)==size: yield block;block=[]
    if block: yield block

def selftest(args):
    """The official package's own expected maps (tests/test_mapper.py)."""
    init()
    sys.path.insert(0,str(HERE/'_deps/site'))
    import torch, transformers, rdkit
    src=(HERE/'_deps/site/tests/test_mapper.py').read_text(encoding='utf-8')
    import ast
    tree=ast.parse(src);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_example_maps')
    rxns=ast.literal_eval(fn.body[0].value);expected=ast.literal_eval(fn.body[1].value)
    got=MAPPER.get_attention_guided_atom_maps(rxns)
    checks=[{'map_equal':g['mapped_rxn']==e['mapped_rxn'],'confidence':g['confidence'],'expected_confidence':e['confidence']} for g,e in zip(got,expected)]
    result={'checks':checks,'all_maps_equal':all(c['map_equal'] for c in checks),'device':str(MAPPER.device),
            'torch':torch.__version__,'transformers':transformers.__version__,'rdkit':rdkit.__version__,
            'wheel_sha256':sha(next((HERE/'_deps/wheels').glob('rxnmapper-0.3.0-*.whl')))}
    dump(HERE/'mapper_selftest.json',result);print(json.dumps(result,indent=1))

def corpus(args):
    db=sqlite3.connect(HERE/args.database)
    db.execute('CREATE TABLE IF NOT EXISTS mapped(idx INTEGER PRIMARY KEY, status TEXT, mapped_rxn TEXT, confidence REAL, error TEXT)')
    done={x[0] for x in db.execute('SELECT idx FROM mapped')}
    rows=source_rows(done)
    if args.limit: rows=(r for r in rows if r[0]<args.limit)
    start=time.monotonic();last=start;new=0
    with mp.get_context('spawn').Pool(args.workers,initializer=init,initargs=(args.batch,args.gpu_fraction)) as pool:
        results=pool.imap_unordered(map_chunk,chunks(rows,args.chunk))
        while True:
            # Fail loudly instead of hanging silently; committed rows resume on rerun.
            try: result=results.next(timeout=args.stall_timeout)
            except StopIteration: break
            except mp.TimeoutError: db.commit();raise SystemExit(f'No chunk finished in {args.stall_timeout} s')
            db.executemany('INSERT INTO mapped VALUES(?,?,?,?,?)',result);new+=len(result)
            if time.monotonic()-last>30:
                db.commit();last=time.monotonic()
                print(json.dumps({'completed':len(done)+new,'new_per_second':new/(last-start)}),flush=True)
    db.commit()
    summary={'rows':len(done)+new,'seconds_this_run':time.monotonic()-start,'workers':args.workers,
             'statuses':dict(db.execute('SELECT status,count(*) FROM mapped GROUP BY status'))}
    dump(HERE/(args.database+'.summary.json'),summary);print(json.dumps(summary),flush=True)

def export(args):
    """Row-aligned source file for the AutoTemplate runner.

    Unmapped rows get a placeholder that the runner records as an exception; an empty
    line would be skipped by csv.DictReader and shift every later source index.
    """
    db=sqlite3.connect(HERE/args.database)
    n=db.execute('SELECT count(*),max(idx) FROM mapped').fetchone()
    assert n[0]==n[1]+1==1808937,n
    with (HERE/args.export).open('w',encoding='utf-8',newline='') as f:
        f.write('ReactionSmiles\n')
        for (smiles,) in db.execute("SELECT coalesce(nullif(mapped_rxn,''),'MAPPING_FAILED') FROM mapped ORDER BY idx"):
            assert '\t' not in smiles and '\n' not in smiles;f.write(smiles+'\n')
    dump(HERE/(args.export+'.summary.json'),{'rows':n[0],'sha256':sha(HERE/args.export),'database_sha256':sha(HERE/args.database)})

def benchmark(args):
    """One reaction per call, exactly as upstream; batched corpus maps are compared for drift."""
    init()
    rows=[(int(r['idx']),r['Lowe_smiles']) for r in load_benchmark()]
    out=[]
    for idx,lowe in rows:
        rxn=prepare(lowe)
        status,mapped,conf,err=map_one(rxn) if rxn else ('invalid_input',None,None,None)
        out.append({'idx':idx,'status':status,'mapped_rxn':mapped,'confidence':conf,'error':err,'prepared':rxn})
    batched={r[0]:r for r in map_chunk(rows)}
    for r in out: r['batched_map_equal']=batched[r['idx']][2]==r['mapped_rxn']
    with (HERE/args.output).open('w',encoding='utf-8') as f:
        for r in out: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary={'rows':len(out),'statuses':{s:sum(r['status']==s for r in out) for s in {r['status'] for r in out}},
             'batched_map_equal':sum(r['batched_map_equal'] for r in out),'input_sha256':input_hashes()}
    dump(HERE/(args.output+'.summary.json'),summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['selftest','corpus','export','benchmark'])
    p.add_argument('--database',default='rxnmapper_corpus.sqlite');p.add_argument('--export',default='rxnmapper_corpus.tsv')
    p.add_argument('--output',default='rxnmapper_benchmark.jsonl')
    p.add_argument('--workers',type=int,default=6);p.add_argument('--chunk',type=int,default=256);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--gpu_fraction',type=float,default=0)
    p.add_argument('--stall_timeout',type=float,default=1200)
    a=p.parse_args();globals()[a.command](a)
