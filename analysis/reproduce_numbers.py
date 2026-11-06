"""Recompute arithmetic from analysis-ready process rows, not absent physical raw streams."""
from pathlib import Path
import csv,json,math,statistics,argparse
R=Path(__file__).resolve().parents[1]
def read(n):
 with(R/n).open()as f:return list(csv.DictReader(f))
def gm(v):return math.exp(statistics.mean(math.log(float(x))for x in v))
owner=read('source_data/v18_verified/ownership/all_pairs.csv');assert len(owner)==36
base=read('source_data/v18_verified/baselines/all_pairs.csv');b={}
for q in [1,2,4]:
 b[str(q)]={}
 for method in ['banker_c0','hard_reserve','native']:
  rs=[r for r in base if int(r['quota_factor'])==q and r['target']=='adaptive'and r['baseline']==method];assert len(rs)==8
  b[str(q)][method]={'M_ratio':gm(r['M_ratio']for r in rs),'brief_ratio':gm(r['brief_ratio']for r in rs),'paired_processes':len(rs)}
pr={}
for stage in ['development','confirmation']:
 rs=[r for r in read(f'source_data/v19_projection/{stage}/all_services.csv')if r['phase']=='timing'];lookup={(r['case'],r['quota_factor'],r['repeat'],r['arm']):r for r in rs};pr[stage]={}
 for q in sorted({r['quota_factor']for r in rs}):
  cells={}
  for case in sorted({r['case']for r in rs}):
   vals=[];nat=[];off=[]
   for rep in sorted({r['repeat']for r in rs if r['case']==case and r['quota_factor']==q}):
    x={arm:lookup[case,q,rep,arm]for arm in ['reuse','adaptive','native']};vals.append(float(x['reuse']['M_s'])/float(x['adaptive']['M_s']));nat.append(float(x['reuse']['M_s'])/float(x['native']['M_s']));off.append(float(x['adaptive']['M_s'])/float(x['native']['M_s']))
   assert len(vals)==6;cells[case]={'on_off_M':gm(vals),'on_Native_M':gm(nat),'off_Native_M':gm(off),'paired_processes':6}
  pr[stage][q]={'cells':cells,'equal_family':{k:gm(v[k]for v in cells.values())for k in ['on_off_M','on_Native_M','off_Native_M']}}
r=[x for x in read('experiments/v20_review/runtime_statistics/output/runtime_factor4_per_service.csv')if x['stage']=='confirmation'];key={(x['case'],x['repeat'],x['arm']):x for x in r};ledger={}
for case in sorted({x['case']for x in r}):
 vals=[]
 for rep in sorted({x['repeat']for x in r if x['case']==case}):
  n=key[case,rep,'native'];a=key[case,rep,'reuse'];vals.append((float(a['M_s'])-float(a['schedule_wall_s'])+float(n['schedule_wall_s']))/float(n['M_s']))
 ledger[case]=gm(vals)
r=[x for x in read('source_data/v17_verified/confirmation_all_primary_pairs.csv')if int(x['quota_factor'])==1 and x['target']=='adaptive'and x['baseline']=='inc'];families={}
for fam,is_alpaca in [('prefix',False),('instruction',True)]:families[fam]=gm(x['brief_ratio']for x in r if x['case'].startswith('alpaca')==is_alpaca)
out={'scope':'Arithmetic on public derived tables; not a new physical/raw audit or GPU run','ownership_fullC1_fullC0_M':gm(x['M_ratio']for x in owner),'baseline_equal_family':b,'projection':pr,'eight_scenario_base_Adaptive_INC_brief':{'families':families,'equal_family_ratio':gm(families.values())},'scheduler_replacement_accounting_scenario':{'families':ledger,'equal_family_ratio':gm(ledger.values()),'interpretation':'Arithmetic substitution, not causal intervention, achievable optimization prediction, engineering floor or GPU timer'}}
assert math.isclose(out['ownership_fullC1_fullC0_M'],.9439059546735752,rel_tol=1e-12)
assert math.isclose(pr['confirmation']['4']['equal_family']['on_Native_M'],1.1016925384,rel_tol=1e-9)
assert math.isclose(gm(ledger.values()),1.014908418692134,rel_tol=1e-12)
assert math.isclose(gm(families.values()),.5716218504249138,rel_tol=1e-12)
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=R/'results/reproduced_numbers.json');a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
