"""Run the frozen 200-name experiment; successful responses are never regenerated."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,datetime,hashlib,json,os,time,urllib.request,urllib.error,shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(path,value):
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 tmp.replace(path)
def main():
 global HERE
 data=HERE
 parser=argparse.ArgumentParser(description='Generate a new 200-name experiment using the documented prompt. API charges apply without --dry-run.')
 parser.add_argument('--limit',type=int,default=200)
 parser.add_argument('--output',type=Path,required=True,help='Separate directory for a new generation and its raw responses.')
 parser.add_argument('--dry-run',action='store_true')
 args=parser.parse_args()
 HERE=args.output.resolve()
 if HERE==data.resolve():raise RuntimeError('Choose a separate output directory')
 if not 1<=args.limit<=200:raise RuntimeError('--limit must be between 1 and 200')
 HERE.mkdir(parents=True,exist_ok=True)
 for name in ['protocol.json','prompt.txt','names_only.json','references.json']:
  target=HERE/name
  if target.exists():assert target.read_bytes()==(data/name).read_bytes(),name+' changed'
  else:shutil.copy2(data/name,target)
 if args.dry_run:
  print('Dry run: frozen prompt and 200 name-only inputs copied; no API calls made.');return

 protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
 prompt=(HERE/'prompt.txt').read_text(encoding='utf-8')
 assert hashlib.sha256(prompt.encode()).hexdigest()==protocol['prompt_sha256']
 names=json.loads((HERE/'names_only.json').read_text(encoding='utf-8'))
 assert len(names)==200 and all(set(r)=={'row_id','name'} for r in names)
 env={}
 for line in ((ROOT/'.env').read_text(encoding='utf-8-sig').splitlines() if (ROOT/'.env').exists() else []):
  if line.strip() and not line.lstrip().startswith('#') and '=' in line:
   k,v=line.split('=',1);env[k.strip()]=v.strip().strip('\"\'')
 key=os.environ.get('OPENAI_API_KEY') or env.get('OPENAI_API_KEY')
 if not key: raise RuntimeError('OPENAI_API_KEY is unavailable; key not printed')
 for folder in ['requests','raw_responses','records','attempt_errors']:(HERE/folder).mkdir(exist_ok=True)
 identity={k:digest(HERE/k) for k in ['prompt.txt','protocol.json','names_only.json','references.json']}
 manifest_path=HERE/'run_manifest.json'
 manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'identity':identity,'started_utc':now(),'status':'prepared'}
 assert manifest['identity']==identity,'Frozen inputs or protocol changed'
 manifest.update(status='running',last_started_utc=now());atomic(manifest_path,manifest)
 def run(item):
  i=item['row_id']; stem=f'{i:03d}';record=HERE/'records'/f'{stem}.json'
  if record.exists():return i,'existing'
  request={'model':protocol['model_requested'],'instructions':prompt,'input':protocol['user_template'].replace('{name}',item['name']),
   'reasoning':{'effort':protocol['reasoning_effort']},'text':{'format':{'type':'json_object'}},
   'max_output_tokens':protocol['max_output_tokens'],'store':False}
  atomic(HERE/'requests'/f'{stem}.json',request)
  raw_path=HERE/'raw_responses'/f'{stem}.json'
  result=None
  if raw_path.exists():result=json.loads(raw_path.read_text(encoding='utf-8'))
  for attempt in range(1,4):
   if result is not None:break
   started=now()
   try:
    req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(request).encode(),
     headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as response:
     result=json.load(response)
    atomic(raw_path,result)
   except Exception as e:
    code=e.code if isinstance(e,urllib.error.HTTPError) else None
    detail=e.read().decode('utf-8','replace') if isinstance(e,urllib.error.HTTPError) else ''
    message=(str(e)+' '+detail).replace(key,'[REDACTED]')
    atomic(HERE/'attempt_errors'/f'{stem}-{attempt}.json',{'utc':started,'code':code,'type':type(e).__name__,'message':message})
    if code in (400,401,403,404):raise RuntimeError(f'API rejected row {i}: HTTP {code}; {message[:1500]}')
    if attempt==3:raise RuntimeError(f'Transport/API failure for row {i}: {type(e).__name__}; resumable')
    time.sleep(2**attempt)
  texts=[c['text'] for out in result.get('output',[]) for c in out.get('content',[]) if c.get('type')=='output_text']
  text=''.join(texts);error=None;smiles=None
  try:
   parsed=json.loads(text)
   assert isinstance(parsed,dict) and set(parsed)=={'smiles'}
   assert parsed['smiles'] is None or isinstance(parsed['smiles'],str)
   smiles=parsed['smiles']
  except Exception:error='invalid_output_schema'
  if result.get('status')!='completed':error='response_'+str(result.get('status'))
  if error:smiles=None
  atomic(record,{'row_id':i,'name':item['name'],'smiles':smiles,'error':error,'response_model':result.get('model'),
   'response_id':result.get('id'),'created_at':result.get('created_at'),'usage':result.get('usage'),
   'response_status':result.get('status'),'request_sha256':digest(HERE/'requests'/f'{stem}.json'),
   'raw_response_sha256':digest(raw_path),'saved_utc':now()})
  return i,'saved'
 selected=names[:args.limit]
 try:
  with ThreadPoolExecutor(max_workers=protocol['concurrency']) as pool:
   futures=[pool.submit(run,item) for item in selected]
   for f in as_completed(futures):
    i,status=f.result()
    count=len(list((HERE/'records').glob('*.json')))
    if count%10==0 or args.limit==1:print(f'Completed {count}/200; row {i}: {status}',flush=True)
 except Exception:
  manifest.update(status='interrupted',records_completed=len(list((HERE/'records').glob('*.json'))),updated_utc=now())
  atomic(manifest_path,manifest);raise
 records=[json.loads(p.read_text(encoding='utf-8')) for p in (HERE/'records').glob('*.json')]
 manifest.update(status='complete' if len(records)==200 else 'pilot_complete',records_completed=len(records),
  response_models=sorted({r['response_model'] for r in records}),updated_utc=now(),
  total_input_tokens=sum((r.get('usage') or {}).get('input_tokens',0) for r in records),
  total_output_tokens=sum((r.get('usage') or {}).get('output_tokens',0) for r in records))
 atomic(manifest_path,manifest)
 print(json.dumps({k:v for k,v in manifest.items() if k!='identity'},indent=2),flush=True)
if __name__=='__main__':main()
