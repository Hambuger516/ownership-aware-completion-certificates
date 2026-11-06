"""Audit the diagnostic helper without importing vLLM or running a GPU."""
import ast
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest
from prc_nc.model import State, Group, canonical
from prc_nc.algorithms import indexed_closure

source=Path(__file__).resolve().parents[1]/'prc_nc/adapters/vllm_scheduler.py'
tree=ast.parse(source.read_text())
original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CertificateScheduler')
methods=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in ('_study_candidate_counterfactual','_log')]
module=ast.Module(body=[ast.ClassDef(name='AuditFixture',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
ns=dict(defaultdict=defaultdict,hashlib=hashlib,json=json,time=time,State=State,Group=Group,indexed_closure=indexed_closure)
exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)

def fixture():
    x=ns['AuditFixture']();x._audit=True;x._step=42;x._events=io.StringIO();x._event_buffer=[]
    x._pending_commit={'r2':NS(grant_id='actual-grant')};x._pending_projection={'keep':'unchanged'}
    x._authority=NS(version=7,controller=NS(stats={'oracle':11},state='unchanged'))
    return x

def positive():
    return State((0,),((0,),(0,),(1,)),(Group('shared',0,1,frozenset({0,1})),))

def emit(x,s,allow):
    return x._study_candidate_counterfactual(s,tuple('r'+str(i) for i in range(s.n)),
        request_id='r2',external_request_id='w0-r2',action_id='action42',proposal_version=7,actual_allow=allow)

class CandidateCounterfactualTests(unittest.TestCase):
    def test_known_information_gain_truth_and_actual_grant_binding(self):
        x=fixture();x._log('preceding');s=positive();emit(x,s,True)
        e=json.loads(x._events.getvalue().splitlines()[-1])
        self.assertTrue(e['shared_certificate']);self.assertFalse(e['exclusive_certificate'])
        self.assertFalse(e['sum_demand_certificate']);self.assertTrue(e['actual_allow'])
        self.assertEqual(e['actual_grant_id'],'actual-grant');self.assertEqual(e['action_id'],'action42')
        self.assertEqual(e['source_event_index'],1);self.assertEqual(e['source_candidate_event_index'],1)
        self.assertEqual(e['projected_post']['owner_classes'],[dict(pool=0,owner_ranks=[0,1],pinned=False,pages=1)])
        self.assertEqual(x._study_candidate_counts['diagnostic_oracle_calls'],2)

    def test_known_refusal_view_and_no_false_grant_link(self):
        x=fixture();s=State((0,),((1,),),(Group('private',0,1,frozenset({0})),));emit(x,s,False)
        e=json.loads(x._events.getvalue())
        self.assertFalse(e['shared_certificate']);self.assertFalse(e['exclusive_certificate'])
        self.assertFalse(e['sum_demand_certificate']);self.assertFalse(e['actual_allow'])
        self.assertIsNone(e['actual_grant_id']);self.assertEqual(x._study_candidate_counts['actual_refused'],1)

    def test_diagnostics_never_mutate_policy_or_physical_state_and_are_audit_only(self):
        x=fixture();s=positive();before=canonical(s);pending=x._pending_commit;projection=x._pending_projection
        self.assertIsNone(emit(x,s,True));self.assertIsNone(emit(x,s,False))
        self.assertEqual(canonical(s),before);self.assertIs(x._pending_commit,pending);self.assertIs(x._pending_projection,projection)
        self.assertEqual(x._authority.version,7);self.assertEqual(x._authority.controller.stats,{'oracle':11})
        self.assertEqual(x._authority.controller.state,'unchanged')
        self.assertEqual(x._study_candidate_counts['logged'],2)
        self.assertEqual(x._study_candidate_counts['shared_over_exclusive_gain'],2)
        log=x._events.getvalue();counts=dict(x._study_candidate_counts);x._audit=False
        emit(x,s,True);self.assertEqual(x._events.getvalue(),log);self.assertEqual(x._study_candidate_counts,counts)

if __name__=='__main__':unittest.main()
