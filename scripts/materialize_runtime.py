"""Materialize a named source variant; never runs an experiment."""
from pathlib import Path
import argparse,shutil,json,hashlib
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['original','projection'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists(),'Output must be new'
shutil.copytree(r/'package',a.output,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
if a.variant=='original':shutil.copyfile(r/'verification/reference_adapter.py',a.output/'prc_nc/adapters/vllm_scheduler.py')
s={str(x.relative_to(a.output)):hashlib.sha256(x.read_bytes()).hexdigest()for x in a.output.rglob('*.py')}
(a.output/'SOURCE_VARIANT.json').write_text(json.dumps({'variant':a.variant,'GPU_runs':0,'python_files_sha256':s},indent=2)+'\n')
print(json.dumps({'variant':a.variant,'python_files':len(s),'GPU_runs':0}))
