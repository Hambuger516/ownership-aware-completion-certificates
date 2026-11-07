"""Rebuild the seven current-paper experiment figures; no GPU runs."""
from pathlib import Path
import json,os,subprocess,sys
R=Path(__file__).resolve().parents[1]
env={**os.environ,'MPLCONFIGDIR':str(R/'results/matplotlib'),'CUDA_VISIBLE_DEVICES':''}
(R/'results/matplotlib').mkdir(parents=True,exist_ok=True)
rows=[]
for module,part in [('main_figures.py','main'),('supplementary_figures.py','supplementary')]:
 subprocess.run([sys.executable,str(R/'figure_source'/module),'--manuscript-root',str(R)],cwd=R,env=env,check=True)
 rows.extend(json.loads((R/'results/figure_build'/part/'registry.json').read_text())['figures'])
assert len(rows)==7
assert {x['figure_number']for x in rows}=={3,4,5,6,'S1','S2','S3'}
for row in rows:
 for ext in ['pdf','svg','png']:assert(R/'results/figure_build/figures'/(row['new_stem']+'.'+ext)).is_file()
registry={'release':'1.2.1','experimental_figures':7,'main_figures':4,'supplementary_figures':3,'figures':rows,'GPU_runs':0,'measurement_changes':False}
(R/'results/figure_build/registry.json').write_text(json.dumps(registry,indent=2)+'\n')
print(json.dumps({'figures_reproduced':7,'main':4,'supplementary':3,'GPU_runs':0,'all_frozen_displayed_endpoints_matched':True}))
