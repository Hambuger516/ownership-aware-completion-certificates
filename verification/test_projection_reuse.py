"""CPU transaction/equivalence checks of the actual adapter AST; no vLLM/GPU import.

Runs old-v18 and candidate off/on with the same physical allocator fixture. It
checks complete semantic events and authoritative states, not expected copies of
the implementation. Runtime/performance claims require separately matched GPUs.
"""
from pathlib import Path
import ast,copy,hashlib,inspect,json,os,random,re,time,unittest,sys
from collections import defaultdict
from types import SimpleNamespace as NS
R=Path(__file__).resolve().parent;P=R.parent/'package';sys.path.insert(0,str(P))
from prc_nc.algorithms import State,Group,indexed_closure,LinearBuildMarginIndex,linear_margins,interval_trial
from prc_nc.controller import Refused,StaleState
from prc_nc.runtime import AllocationAuthority
from prc_nc.physical import PageOwnershipIndex
from prc_nc.admission import AdmissionContext,AdmissionDeferred
from prc_nc.peer_admission import peer_admission_decision,pair_funding_exemption
from prc_nc.drain import DrainPolicy
from prc_nc.drain_batching import fixed_witness_pair
from prc_nc.model import canonical,finish_request
OLD=R/'reference_adapter.py'
# R=.../65/candidate, R.parents[1]=SMC_2026
NEW=P/'prc_nc/adapters/vllm_scheduler.py'

def adapter(path):
 tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='CertificateScheduler')
 wanted={'_snapshot','_prc_allow','_capture_request','_study_candidate_counterfactual'}
 methods=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in wanted]
 init=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='__init__')
 alloc=next(n for n in init.body if isinstance(n,ast.FunctionDef)and n.name=='traced_allocate')
 factory=ast.parse('def wrap(self,original_allocate,allocation_signature):\n pass\n').body[0];factory.body=[alloc,ast.Return(value=ast.Name(id='traced_allocate',ctx=ast.Load()))]
 ns=dict(globals());module=ast.Module(body=[ast.ClassDef(name='Adapter',bases=[],keywords=[],body=methods,decorator_list=[]),factory],type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
 return ns['Adapter'],ns['wrap']

A_OLD,W_OLD=adapter(OLD);A_NEW,W_NEW=adapter(NEW)

def fixture(on=False,audit=False,seed=0,old=False):
 rng=random.Random(seed);a=(A_OLD if old else A_NEW)();n=rng.randint(1,6);bs=16
 # Private and shared physical pages; every complete initial demand fits free.
 owner_sets=[{i}for i in range(n)for _ in range(rng.randint(1,3))]
 if n>1:owner_sets+=[set(range(n))]
 free=24;capacity=len(owner_sets)+free
 blocks=[NS(block_id=i,is_null=i==0,ref_cnt=0)for i in range(capacity+1)]
 mapping={chr(97+i):[]for i in range(n)}
 for j,who in enumerate(owner_sets,1):
  for i in who:mapping[chr(97+i)].append(blocks[j]);blocks[j].ref_cnt+=1
 a._pm=NS(req_to_blocks=mapping,block_size=bs)
 a.kv_cache_manager=NS(block_pool=NS(blocks=blocks,num_gpu_blocks=capacity+1,get_num_free_blocks=lambda:sum(not b.is_null and b.ref_cnt==0 for b in blocks)))
 a.requests={rid:NS(request_id=rid,num_prompt_tokens=len(row)*bs,max_tokens=16*rng.randint(1,3),num_computed_tokens=len(row)*bs-1,num_preemptions=0,output_token_ids=[])for rid,row in mapping.items()}
 a.max_model_len=1024;a._mode='shared_incremental';a._audit=audit;a._step=1
 a._drain=DrainPolicy(256,scope='cohort',batching='cohort_certified');a._drain_enabled=True
 a._index=a._index_state=a._index_ids=None;a._pending_commit={};a._pending_projection={};a._pending_index={}
 a._zero_growth_fast=False;a._zero_growth_pass=set();a._deny_cache_enabled=False;a._proposals={};a._pipeline_enabled=False;a._online=None
 a._serial=0;a._zero_projection_reuse=on;a._zero_projection_checks=a._zero_projection_hits=a._zero_projection_audit_checks=0
 a._peer_resident_limit=4;a._peer_queue_bypass=True;a._peer_capacity_adaptive=True;a._ordinary_witness_priority=True
 a._peer_gate_checks=a._peer_gate_deferrals=a._peer_pair_checks=a._peer_pair_exemptions=0
 a.events=[];a._log=lambda event,**kw:a.events.append(dict(event=event,**kw))
 a._ownership=PageOwnershipIndex(capacity)
 for rid,row in mapping.items():a._ownership.observe_request(rid,[b.block_id for b in row])
 s,ids,_=a._snapshot();a._authority=AllocationAuthority(s,ids,audit=audit,admission_policy='c0_first_shorter_cap',on_event=lambda e:a.events.append(dict(event='causal_'+e['kind'],**{k:v for k,v in e.items()if k!='kind'})))
 a.calls=0
 def allocate(request,num_new_tokens,num_new_computed_tokens=0,new_computed_blocks=None,num_external_computed_tokens=0,num_lookahead_tokens=0):
  a.calls+=1
  row=mapping.setdefault(request.request_id,[])
  cached=list(new_computed_blocks.blocks[0])if new_computed_blocks is not None else []
  for block in cached:row.append(block);block.ref_cnt+=1
  required=(min(request.num_computed_tokens+num_new_computed_tokens+num_new_tokens,a.max_model_len)+bs-1)//bs
  need=max(0,required-len(row));available=[b for b in blocks if not b.is_null and b.ref_cnt==0]
  if need>len(available):return None
  for b in available[:need]:row.append(b);b.ref_cnt+=1
  return tuple(b.block_id for b in row)
 a.original_allocate=allocate
 a.traced_allocate=(W_OLD if old else W_NEW)(a,allocate,inspect.signature(allocate))
 return a

def stable_events(a):
 def clean(e):return {k:v for k,v in e.items()if not k.endswith('_ns')}
 return [clean(e)for e in a.events if e['event']!='zero_projection_reuse']

def snapshot(a):
 q=a._authority
 return dict(state=canonical(q.confirmed),virtual=canonical(q.virtual),ids=q.ids,version=q.version,controller_version=q.controller.version,order=q.controller.order,stats=q.controller.stats,faulted=q.faulted,pending=q.pending)

class ProjectionReuseTests(unittest.TestCase):
 def test_randomized_old_off_on_decision_projection_and_full_authority_trace_equivalence(self):
  for audit in (False,True):
   for seed in range(100):
    aa=[fixture(False,audit,seed,True),fixture(False,audit,seed),fixture(True,audit,seed)]
    for rid in list(aa[0].requests):
     for a in aa:
      before=a._authority.confirmed;r=a.requests[rid]
      original_snapshot=a._snapshot;snapshots=[]
      def record_snapshot(extra=None):
       value=original_snapshot(extra);snapshots.append(value);return value
      a._snapshot=record_snapshot
      self.assertTrue(a._prc_allow(r,1));post=a._pending_projection[rid][0]
      if a._zero_projection_reuse:self.assertIs(post,snapshots[0][0])
      a._snapshot=original_snapshot
      self.assertIsNotNone(a._authority.pending)
      self.assertEqual(a._pending_commit[rid].pages,(0,))
      self.assertNotIn(rid,a._zero_growth_pass)
      a.traced_allocate(r,1);r.num_computed_tokens+=1
      self.assertEqual(a._pending_projection,{})
      self.assertEqual(a._pending_commit,{})
     self.assertEqual(snapshot(aa[0]),snapshot(aa[1]));self.assertEqual(snapshot(aa[1]),snapshot(aa[2]))
     self.assertEqual(stable_events(aa[0]),stable_events(aa[1]));self.assertEqual(stable_events(aa[1]),stable_events(aa[2]))
    self.assertEqual(aa[2]._zero_projection_hits,len(aa[2].requests))
    self.assertEqual(aa[2]._zero_projection_audit_checks,len(aa[2].requests)if audit else 0)
    # Crossing the next block boundary must use the old growth transaction.
    for a in aa:
     r=next(iter(a.requests.values()));self.assertTrue(a._prc_allow(r,1));a.traced_allocate(r,1)
    self.assertEqual(snapshot(aa[0]),snapshot(aa[2]));self.assertEqual(stable_events(aa[0]),stable_events(aa[2]))
    self.assertEqual(aa[2]._zero_projection_hits,len(aa[2].requests))

 def test_pending_authority_never_reuses_zero_projection(self):
  for on in(False,True):
   a=fixture(on);r=next(iter(a.requests.values()))
   self.assertTrue(a._prc_allow(r,1));h=a._zero_projection_hits
   self.assertFalse(a._prc_allow(r,1));self.assertEqual(a._zero_projection_hits,h)
   self.assertEqual(a.calls,0)

 def test_prefix_append_external_and_lookahead_retain_fail_closed_behavior(self):
  for on in(False,True):
   for args in [dict(num_new_computed_tokens=16,new_computed_blocks=NS(blocks=[[NS(block_id=900,is_null=False)]])),dict(num_external_computed_tokens=1),dict(num_lookahead_tokens=1)]:
    a=fixture(on);r=next(iter(a.requests.values()))
    with self.assertRaises(RuntimeError):a._prc_allow(r,1,**args)
    self.assertEqual(a._zero_projection_hits,0);self.assertIsNone(a._authority.pending)

 def test_version_change_during_snapshot_does_not_reuse_stale_projection(self):
  a=fixture(True);original=a._snapshot
  def changed(extra=None):a._authority.version+=1;return original(extra)
  a._snapshot=changed;r=next(iter(a.requests.values()))
  self.assertTrue(a._prc_allow(r,1));self.assertEqual(a._zero_projection_hits,0)
  self.assertEqual(a._pending_commit[r.request_id].state_version,a._authority.version)

 def test_inconsistent_physical_cohort_is_not_reused_or_dispatched(self):
  a=fixture(True);r=next(iter(a.requests.values()));original=a._snapshot
  def changed(extra=None):
   s,ids,owners=original(extra);return s,list(reversed(ids)),owners
  if len(a.requests)<2:a=fixture(True,seed=5);r=next(iter(a.requests.values()));original=a._snapshot
  a._snapshot=changed
  try:a._prc_allow(r,1)
  except(ValueError,RuntimeError,AssertionError):pass
  self.assertEqual(a._zero_projection_hits,0);self.assertEqual(a.calls,0)

 def test_allocator_mismatch_still_faults_even_for_zero_projection(self):
  for on in(False,True):
   a=fixture(on);r=next(iter(a.requests.values()));self.assertTrue(a._prc_allow(r,1))
   def illegal_allocate():
    block=next(b for b in a.kv_cache_manager.block_pool.blocks if not b.is_null and b.ref_cnt==0)
    a._pm.req_to_blocks[r.request_id].append(block);block.ref_cnt=1;return (block.block_id,)
   def observe():a._capture_request(r.request_id);s,ids,_=a._snapshot();return s,ids
   with self.assertRaises(RuntimeError):a._authority.dispatch(a._pending_commit[r.request_id],illegal_allocate,observe)
   self.assertTrue(a._authority.faulted)

 def test_zero_allocator_refund_and_release_keep_exact_protocol(self):
  for failure in(False,True):
   aa=[fixture(False,False,3,True),fixture(True,False,3)]
   for a in aa:
    r=next(iter(a.requests.values()));self.assertTrue(a._prc_allow(r,1));g=a._pending_commit[r.request_id]
    def observe():a._capture_request(r.request_id);s,ids,_=a._snapshot();return s,ids
    a._authority.dispatch(g,(lambda:None)if failure else(lambda:tuple(b.block_id for b in a._pm.req_to_blocks[r.request_id])),observe)
    if failure:self.assertEqual(a._authority._outcomes[g.grant_id],None)
    def release():
     for b in a._pm.req_to_blocks.pop(r.request_id):b.ref_cnt-=1
     a._ownership.observe_request(r.request_id,[])
    a._authority.release(r.request_id,release,lambda:(lambda x:(x[0],x[1]))(a._snapshot()))
   self.assertEqual(snapshot(aa[0]),snapshot(aa[1]));self.assertEqual(stable_events(aa[0]),stable_events(aa[1]))

 def test_frozen_head_precondition_is_not_bypassed_by_reuse(self):
  for on in(False,True):
   a=fixture(on,seed=5);ids=a._authority.ids
   if len(ids)<2:raise AssertionError('fixture requires peers')
   a._drain.order=ids;a._drain.active=True
   follower=a.requests[ids[1]]
   self.assertFalse(a._prc_allow(follower,1,scheduled_tokens={}))
   self.assertEqual(a._zero_projection_hits,0);self.assertIsNone(a._authority.pending)


 def test_structural_release_rebuilds_physical_cohort_before_next_zero_projection(self):
  aa=[fixture(False,False,5,True),fixture(False,False,5),fixture(True,False,5)]
  for a in aa:
   first,second=list(a.requests)[:2];r=a.requests[first]
   self.assertTrue(a._prc_allow(r,1));a.traced_allocate(r,1)
   old_state=a._authority.confirmed
   def release():
    for b in a._pm.req_to_blocks.pop(first):b.ref_cnt-=1
    a._ownership.observe_request(first,[])
   a._authority.release(first,release,lambda:(lambda x:(x[0],x[1]))(a._snapshot()))
   self.assertNotEqual(a._authority.confirmed,old_state)
   r=a.requests[second];self.assertTrue(a._prc_allow(r,1))
   self.assertNotIn(first,a._pending_projection[second][1])
   self.assertIsNot(a._pending_projection[second][0],old_state)
   a.traced_allocate(r,1)
  self.assertEqual(snapshot(aa[0]),snapshot(aa[2]));self.assertEqual(stable_events(aa[0]),stable_events(aa[2]))

 def test_new_admission_with_or_without_shared_prefix_keeps_structural_transaction(self):
  for prefix in(False,True):
   aa=[fixture(False,False,5,True),fixture(False,False,5),fixture(True,False,5)]
   for a in aa:
    r=NS(request_id='z',num_prompt_tokens=16,max_tokens=16,num_computed_tokens=0,num_preemptions=0,output_token_ids=[])
    a.requests['z']=r
    kw={}
    if prefix:kw=dict(num_new_computed_tokens=16,new_computed_blocks=NS(blocks=[[a._pm.req_to_blocks[next(iter(a._pm.req_to_blocks))][0]]]))
    self.assertTrue(a._prc_allow(r,1,**kw));self.assertEqual(a._zero_projection_hits,0)
    self.assertEqual(a._authority.pending[0].pages,(1,))
    a.traced_allocate(r,1,**kw)
   self.assertEqual(snapshot(aa[0]),snapshot(aa[2]));self.assertEqual(stable_events(aa[0]),stable_events(aa[2]))

 def test_reused_state_and_groups_are_deeply_immutable(self):
  from dataclasses import FrozenInstanceError
  a=fixture(True);r=next(iter(a.requests.values()));self.assertTrue(a._prc_allow(r,1))
  post=a._pending_projection[r.request_id][0]
  with self.assertRaises(FrozenInstanceError):post.free=(999,)
  with self.assertRaises(FrozenInstanceError):post.groups[0].owners=frozenset({99})
  with self.assertRaises(AttributeError):post.groups[0].owners.add(99)
  self.assertTrue(a._authority.pending is not None)

if __name__=='__main__':unittest.main(verbosity=2)
