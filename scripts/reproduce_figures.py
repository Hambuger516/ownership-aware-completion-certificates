"""Rebuild current figure sources without executing any serving algorithm."""
from pathlib import Path
import os,subprocess,sys,json
R=Path(__file__).resolve().parents[1];env={**os.environ,'MPLCONFIGDIR':str(R/'results/matplotlib'),'CUDA_VISIBLE_DEVICES':''};(R/'results/matplotlib').mkdir(parents=True,exist_ok=True)
for n in ['redesign_main_v22.py','redesign_legacy_v22.py','redesign_modern_si_v22.py','redesign_main_matrices_v23.py','redesign_main_matrices_v24.py']:
 cmd=[sys.executable,str(R/'figure_source'/n)]
 if n in ['redesign_main_v22.py','redesign_main_matrices_v23.py','redesign_main_matrices_v24.py']:cmd+=['--manuscript-root',str(R)]
 subprocess.run(cmd,cwd=R,env=env,check=True)
current=json.loads((R/'provenance/PAPER_FIGURE_REGISTRY.json').read_text())['figures']
for row in current:
 for ext in ['pdf','svg','png']:assert(R/'figs'/(row['new_stem']+'.'+ext)).is_file()
print(json.dumps({'figures_reproduced':len(current),'GPU_runs':0,'scope':'Public plotting/derived-number layer, not original raw-file qualification.'}))
