"""New portable configurations only: no model load, network, GPU or resource qualification."""
from pathlib import Path
import argparse,json,sys,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'package'))
from prc_nc.configuration import validate_config
profiles=json.loads((R/'benchmark/PROFILES.json').read_text());env=json.loads((R/'benchmark/launch_environment.json').read_text())
p=argparse.ArgumentParser();p.add_argument('--study',choices=list(profiles),required=True);p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists(),'Use a new output directory'
pr=profiles[a.study];a.output.mkdir(parents=True);plans=[]
for phase,reps in [('audit',[0]),('timing',range(pr['repetitions']))]:
 for ci,(case,q)in enumerate((c,q)for c in pr['cases']for q in pr['factors']):
  inp=R/'benchmark/studies'/a.study/'inputs';t=json.loads((inp/(case+'.template.json')).read_text());usable=t['blocks']-1
  for rep in reps:
   unit=f'{phase}_{case}_q{q}_r{rep}';d=a.output/unit;d.mkdir();arms=pr['arms'];shift=(ci+rep)%len(arms);order=arms[shift:]+arms[:shift];runs=[]
   for arm in order:
    c=dict(t);c.update(run_id=f'public_{unit}_{arm}',case=case,replicate=rep,output_root=str((d/'runs').resolve()),trace_path=str((inp/(case+'.json')).resolve()),model_path=str(Path(a.model).resolve()),data_path='',blocks=1+usable*q,audit=phase=='audit',drain_after_steps=256,private_refusal_cache=False,peer_resident_limit=4 if arm in('fixed','adaptive','reuse')else 0,peer_queue_bypass=arm in('fixed','adaptive','reuse'),peer_capacity_adaptive=arm in('adaptive','reuse'),timeout_seconds=1200,method='shared_incremental',certificate_variant='shared_incremental',drain_policy='timer',drain_scope='cohort',drain_batching='cohort_certified',ordinary_witness_priority=True,admission_policy='c0_first_shorter_cap',enable_prc_pipeline=False,active_drain_admission=False,rank_order=False,zero_growth_fast=False,deny_cache=False)
    if arm in('full_c0','full_c1'):c['certificate_variant']='common_'+arm
    if arm=='banker_c0':c.update(certificate_variant='exclusive_only',ordinary_witness_priority=False,admission_policy='unrestricted')
    if arm in('hard_reserve','native'):c.update(method='hard_reserve'if arm=='hard_reserve'else'native_plain',certificate_variant='shared_incremental',drain_policy='none',drain_batching='serial',ordinary_witness_priority=False,admission_policy='unrestricted')
    c=validate_config(c,require_paths=True);f=d/(arm+'.json');f.write_text(json.dumps(c,indent=2)+'\n');runs.append({'arm':arm,'config':str(f.relative_to(a.output)),'configuration_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'environment':{**env['common'],**env[phase],'PRC_ZERO_PROJECTION_REUSE':'1'if arm=='reuse'else'0'}})
   plans.append({'unit':unit,'case':case,'phase':phase,'repeat':rep,'quota_factor':q,'runs':runs})
manifest={'status':'NEW_PORTABLE_PLAN_NOT_EXECUTED_NOT_ORIGINAL_RESOURCE_QUALIFICATION','study':a.study,'runtime_variant':pr['runtime_variant'],'GPU_runs':0,'plans':plans,'services':sum(len(x['runs'])for x in plans)}
(a.output/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'status':manifest['status'],'services':manifest['services'],'GPU_runs':0}))
