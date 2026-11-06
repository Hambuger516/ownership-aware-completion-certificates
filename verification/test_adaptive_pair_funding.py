import ast, copy, itertools, json, sys, time, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from prc_nc.peer_admission import peer_admission_decision, pair_funding_exemption
from prc_nc.admission import AdmissionContext
from prc_nc.configuration import validate_config
sys.path.insert(0,str(Path(__file__).resolve().parent))
from peer_gate_audit import audit_peer_gate
W=Path(__file__).resolve().parents[1]

class AdaptivePairTests(unittest.TestCase):
 def base(self):
  return peer_admission_decision(4,AdmissionContext('new',12,512,tuple((f'r{i}',512)for i in range(4))))
 def test_two_largest_equals_all_pair_funding_without_release(self):
  g=self.base();count=0
  for values in itertools.product(range(4),repeat=4):
   for candidate in range(1,5):
    claims=[candidate]+list(values)
    bound=max(a+b for a,b in itertools.combinations(claims,2))
    for free in range(9):
     z=pair_funding_exemption(g,free,tuple((f'r{i}',v)for i,v in enumerate(values)),candidate)
     self.assertEqual(z['pair_funding']['critical_pair_pages'],bound)
     self.assertIs(z['allows'],all(a+b<=free for a,b in itertools.combinations(claims,2)))
     count+=1
  self.assertEqual(count,9216)
 def test_capacity_monotonic_and_malformed_context_rejected(self):
  g=self.base();needs=tuple((f'r{i}',32)for i in range(4))
  self.assertFalse(pair_funding_exemption(g,65,needs,34)['allows'])
  self.assertTrue(pair_funding_exemption(g,66,needs,34)['allows'])
  for free,n,u in [(True,needs,34),(100,needs,True),(100,needs[:-1],34),(100,tuple(reversed(needs)),34),(100,(('r0',-1),)+needs[1:],34)]:
   with self.assertRaises(ValueError):pair_funding_exemption(g,free,n,u)
 def hook(self):
  tree=ast.parse((W/'package/prc_nc/adapters/vllm_scheduler.py').read_text())
  cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='CertificateScheduler')
  fun=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='_prc_allow')
  env=dict(time=time,AdmissionContext=AdmissionContext,peer_admission_decision=peer_admission_decision,pair_funding_exemption=pair_funding_exemption)
  exec(compile(ast.Module(body=[fun],type_ignores=[]),'actual_adaptive_hook','exec'),env)
  return env['_prc_allow']
 def fake(self,free):
  events=[];state=NS(p=1,n=4,free=(free,),need=((32,),)*4)
  ids=tuple(f'r{i}'for i in range(4));a=NS(ids=ids,version=12,pending=None,virtual=state,confirmed=state)
  o=NS(_drain_enabled=False,_active_drain_admission=False,_mode='shared_incremental',_drain=NS(active=False,allows=lambda r:True),_authority=a,_peer_resident_limit=4,_peer_capacity_adaptive=True,_peer_gate_checks=0,_peer_gate_deferrals=0,_peer_pair_checks=0,_peer_pair_exemptions=0,requests={r:NS(max_tokens=512)for r in ids},_pm=NS(block_size=16),max_model_len=1024,_log=lambda k,**v:events.append((k,v)))
  return o,events
 def test_real_hook_relaxes_only_utility_and_keeps_original_guard(self):
  o,ev=self.fake(100)
  with self.assertRaisesRegex(RuntimeError,'External/speculative'):
   self.hook()(o,NS(request_id='new',num_prompt_tokens=32,max_tokens=512),1,num_external_computed_tokens=1)
  self.assertTrue(ev[0][1]['capacity_exemption']);self.assertEqual(o._peer_pair_exemptions,1)
 def test_real_hook_tight_and_unsettled_fail_closed(self):
  o,ev=self.fake(50);self.assertFalse(self.hook()(o,NS(request_id='new',num_prompt_tokens=32,max_tokens=512),1))
  self.assertFalse(ev[0][1]['allows']);self.assertEqual(o._peer_last_deferred_id,'new')
  o,ev=self.fake(100);o._authority.pending=object()
  with self.assertRaisesRegex(RuntimeError,'settled'):self.hook()(o,NS(request_id='new',num_prompt_tokens=32,max_tokens=512),1)
  self.assertEqual(ev,[])
 def test_configuration_supports_only_reviewed_composition(self):
  c=json.loads((W/'inputs/confirm02_back.template.json').read_text())
  c.update(peer_resident_limit=4,peer_queue_bypass=True,peer_capacity_adaptive=True)
  self.assertTrue(validate_config(c)['peer_capacity_adaptive'])
  for change in [dict(peer_resident_limit=0),dict(peer_queue_bypass=False),dict(peer_capacity_adaptive=1),dict(certificate_variant='force_full'),dict(method='native_plain')]:
   with self.assertRaises(ValueError):validate_config(c|change)
 def fixture(self,capacity):
  trace=[dict(max_tokens=512,prompt_token_ids=[1]*16)for _ in range(5)]
  cfg=dict(peer_resident_limit=4,peer_capacity_adaptive=True,blocks=capacity+1,block_size=16,max_model_len=1024)
  events=[dict(event='adapter',peer_resident_limit=4,peer_capacity_adaptive=True)]
  version=0;free=capacity;res={};checks=deferrals=exempt=0
  for i in range(5):
   rid=f'w0-r{i}';base=i<4;allows=base or free>=65;checks+=1
   gate=dict(event='peer_admission_gate',proof_serial=checks,limit=4,context=dict(candidate_id=rid,authority_version=version,candidate_cap=512,incumbent_caps=[list(x)for x in sorted(res.items())]),pending_grant=False,resource_states_equal=True,actual_cap_source='native Request.max_tokens',max_incumbent_cap=512 if res else 0,shorter_cap=False,peers_at_least_candidate_cap=len(res),adaptive_enabled=True,allows=allows)
   if not base:
    gate.update(base_allows=False,capacity_exemption=allows,pair_funding=dict(free_pages=free,incumbent_need_pages=[[r,32]for r in sorted(res)],candidate_upper_pages=33,critical_pair_pages=65,pair_funded=allows,semantics='two_largest_remaining_claims_no_release_or_sharing_credit'));exempt+=allows
   events.append(gate)
   if not allows:deferrals+=1;continue
   gid='g'+str(i);version+=1;events.append(dict(event='causal_authorize',request_id=rid,grant_id=gid,state_version=version,confirmed_free=[free]))
   events.append(dict(event='proposal',request_id=rid,allow=True,new_pages=1,reused_blocks=0))
   events.append(dict(event='causal_execute',request_id=rid,grant_id=gid,state_version=version));version+=1;free-=1
   events.append(dict(event='causal_acknowledge',request_id=rid,grant_id=gid,state_version=version,confirmed_free=[free]));res[rid]=512
  for rid in list(res):
   version+=1;free+=1;events.append(dict(event='causal_finish',request_id=rid,state_version=version,outcome='completed',confirmed_free=[free]))
  stats=dict(peer_admission_stats=dict(limit=4,checks=checks,deferrals=deferrals,adaptive_enabled=True,pair_checks=1,pair_exemptions=exempt),authority_version=version)
  return events,trace,cfg,stats
 def test_independent_receipt_replay_funded_and_pressure_cases(self):
  for capacity in [64,100]:self.assertTrue(audit_peer_gate(*self.fixture(capacity))['pass_'])
 def test_independent_replay_rejects_forged_free_needs_envelope_and_decision(self):
  for key in ['free_pages','candidate_upper_pages','critical_pair_pages','incumbent_need_pages','pair_funded']:
   ev,tr,c,st=self.fixture(100);g=next(x for x in ev if x.get('pair_funding'))
   if key=='incumbent_need_pages':g['pair_funding'][key][0][1]+=1
   elif key=='pair_funded':g['pair_funding'][key]=False
   else:g['pair_funding'][key]+=1
   with self.subTest(key=key),self.assertRaises(AssertionError):audit_peer_gate(ev,tr,c,st)

if __name__=='__main__':unittest.main()
