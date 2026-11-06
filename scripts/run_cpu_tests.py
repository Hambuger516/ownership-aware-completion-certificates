"""CPU only; no CUDA/model/network operations."""
from pathlib import Path
import os,sys,subprocess
r=Path(__file__).resolve().parents[1]
e={**os.environ,'PYTHONPATH':str(r/'package'),'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''}
for folder in ['package/tests','verification']:
 p=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',folder,'-p','test_*.py','-v'],cwd=r,env=e)
 if p.returncode:raise SystemExit(p.returncode)
