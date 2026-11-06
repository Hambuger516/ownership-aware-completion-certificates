"""Independent truth-table and real adapter-hook tests; no CUDA/vLLM import."""
from pathlib import Path
from types import SimpleNamespace as NS
import itertools,ast,unittest,time
from prc_nc.peer_admission import peer_admission_decision
from prc_nc.admission import AdmissionContext
from prc_nc.configuration import validate_config
W=Path(__file__).resolve().parents[1]
class PeerAdmissionTests(unittest.TestCase):
 def test_exhaustive_cap_truth_table_and_shorter_exemption(self):
  count=0
  for n in range(6):
   for caps in itertools.product((8,64,512),repeat=n):
    for cap in (8,64,512,1024):
     context=AdmissionContext('new',12,cap,tuple((f'r{i}',v)for i,v in enumerate(caps)))
     for limit in (0,2,4):
      g=peer_admission_decision(limit,context);expected=limit==0 or any(x>cap for x in caps)or sum(x>=cap for x in caps)<limit
      self.assertIs(g['allows'],expected);self.assertEqual(g['context']['candidate_cap'],cap);count+=1
  self.assertEqual(count,4368)
 def test_no_peek_at_outputs_and_does_not_certify(self):
  ctx=AdmissionContext('n',0,512,(('a',512),('b',512)))
  self.assertFalse(peer_admission_decision(2,ctx)['allows']);self.assertTrue(peer_admission_decision(4,ctx)['allows'])
  self.assertTrue(peer_admission_decision(2,AdmissionContext('n',0,64,ctx.incumbent_caps))['allows'])
  self.assertTrue(peer_admission_decision(2,AdmissionContext('n',0,512,()))['allows'])
  for value in [-1,1,3,True,'2']:
   with self.assertRaises(ValueError):peer_admission_decision(value,ctx)
 def test_configuration_is_explicit_and_isolated(self):
  template=__import__('json').loads((W/'inputs/confirm02_back.template.json').read_text());template['private_refusal_cache']=False
  for limit in (0,2,4):self.assertEqual(validate_config(template|dict(peer_resident_limit=limit))['peer_resident_limit'],limit)
  for x in [True,1,3,-1]:
   with self.assertRaises(ValueError):validate_config(template|dict(peer_resident_limit=x))
  for changes in [dict(private_refusal_cache=True),dict(active_drain_admission=True),dict(certificate_variant='force_full'),dict(ordinary_witness_priority=False),dict(drain_policy='none'),dict(method='native_plain')]:
   with self.assertRaises(ValueError):validate_config(template|dict(peer_resident_limit=2)|changes)
 def hook(self):
  tree=ast.parse((W/'package/prc_nc/adapters/vllm_scheduler.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef)and x.name=='CertificateScheduler');f=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='_prc_allow')
  env=dict(time=time,AdmissionContext=AdmissionContext,peer_admission_decision=peer_admission_decision);exec(compile(ast.Module(body=[f],type_ignores=[]),'actual_adapter_hook','exec'),env);return env['_prc_allow']
 def fake(self,limit=2):
  events=[];a=NS(ids=('a','b'),version=20,pending=None,virtual='same',confirmed='same')
  obj=NS(_active_drain_admission=False,_drain_enabled=False,_mode='shared_incremental',_drain=NS(active=False,allows=lambda rid:True),_peer_resident_limit=limit,_authority=a,requests={'a':NS(max_tokens=512),'b':NS(max_tokens=512)},_peer_gate_checks=0,_peer_gate_deferrals=0,_log=lambda kind,**v:events.append((kind,v)))
  return obj,events
 def test_actual_adapter_denial_precedes_allocation_and_full_search(self):
  obj,events=self.fake();result=self.hook()(obj,NS(request_id='n',max_tokens=512),1)
  self.assertIs(result,False);self.assertEqual(obj._peer_gate_deferrals,1);self.assertEqual(len(events),1);self.assertEqual(events[0][0],'peer_admission_gate');self.assertFalse(events[0][1]['allows']);self.assertEqual(events[0][1]['context']['authority_version'],20)
 def test_actual_adapter_allow_does_not_bypass_original_guards(self):
  obj,events=self.fake()
  with self.assertRaisesRegex(RuntimeError,'External/speculative'):self.hook()(obj,NS(request_id='n',max_tokens=64),1,num_external_computed_tokens=1)
  self.assertTrue(events[0][1]['allows']);self.assertEqual(obj._peer_gate_deferrals,0)
 def test_existing_resident_and_OFF_have_original_control_path(self):
  for limit,rid in [(2,'a'),(0,'n')]:
   obj,events=self.fake(limit)
   with self.assertRaisesRegex(RuntimeError,'External/speculative'):self.hook()(obj,NS(request_id=rid,max_tokens=512),1,num_external_computed_tokens=1)
   self.assertEqual(events,[])
 def test_unsettled_rights_fail_closed(self):
  obj,events=self.fake();obj._authority.pending=object()
  with self.assertRaisesRegex(RuntimeError,'settled'):self.hook()(obj,NS(request_id='n',max_tokens=512),1)
  self.assertEqual(events,[])
if __name__=='__main__':unittest.main()
