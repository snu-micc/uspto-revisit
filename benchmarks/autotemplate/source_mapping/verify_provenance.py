from pathlib import Path
import hashlib,json,sys
from run_comparison import HERE,SOURCE,VENDOR,sha,dump
m=json.loads((VENDOR/'upstream_manifest.json').read_text(encoding='utf-8'))
checked=[]
for x in m['files']:
    path=VENDOR/x['path'];b=path.read_bytes()
    # Transport normalizes CRLF and appends a terminal LF. Restore bytes only
    # when a candidate exactly matches the independently fetched Git object.
    candidates=[b,b[:-1],b.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'),b[:-1].replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')]
    for candidate in candidates:
        if hashlib.sha1(f'blob {len(candidate)}\0'.encode()+candidate).hexdigest()==x['git_blob_sha1']:
            if b!=candidate:path.write_bytes(candidate)
            b=candidate;break
    h=hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
    assert h==x['git_blob_sha1'],(x['path'],h,x['git_blob_sha1'])
    checked.append({'path':x['path'],'git_blob_sha1':h,'sha256':hashlib.sha256(b).hexdigest()})
source_hash=sha(SOURCE)
assert source_hash==json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))['source_sha256']
import rdkit,numpy,networkx,rdchiral,importlib.metadata
dump(HERE/'provenance_verified.json',{'vendor_files':checked,'source_sha256':source_hash,'protocol_sha256':sha(HERE/'protocol.json'),'runner_sha256':sha(HERE/'run_comparison.py'),'python':sys.version,'executable':sys.executable,'packages':{x:importlib.metadata.version(x) for x in ['rdkit','numpy','networkx','rdchiral']}})
print('Verified upstream Git blob hashes, source SHA256 and runtime versions.')
