"""CPU tests of real native hook code; these are not GPU timing evidence."""
import ast
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from prc_nc.analysis import _verify_cohort_certified_batches
from prc_nc.configuration import validate_config
from prc_nc.drain import DrainPolicy
from prc_nc.execution_meter import ExecutionTokenMeter
from test_witness_preserving_batch import (guarded_fixture,native_running_fixture,
    configuration,namespace,SOURCE)


def guarded_cohort(variant='common_full_c1'):
    a=guarded_fixture(variant)
    state,ids,_=a._snapshot()
    a._drain=DrainPolicy(1,scope='cohort',batching='cohort_certified')
    a._drain.observe(ids,tuple(range(len(ids))))
    return a


class CohortBatchTests(unittest.TestCase):
    def setUp(self):
        old=namespace.get('_instrumented_schedule')
        self.addCleanup(lambda:namespace.__setitem__('_instrumented_schedule',old))

    def test_config_disallows_shortcuts_pipeline_and_head_scope(self):
        self.assertEqual(validate_config(configuration(drain_batching='cohort_certified'))['drain_batching'],'cohort_certified')
        for change in (dict(drain_scope='head'),dict(drain_policy='none'),dict(rank_order=True),
                       dict(zero_growth_fast=True),dict(deny_cache=True),dict(enable_prc_pipeline=True)):
            with self.assertRaises(ValueError):validate_config(configuration(drain_batching='cohort_certified',**change))

    def test_epoch_witness_changes_but_membership_requires_terminal_receipt(self):
        d=DrainPolicy(1,batching='cohort_certified')
        d.observe(tuple('abc'),(0,1,2));d.observe(tuple('abc'),(2,0,1))
        self.assertEqual(d.order,tuple('cab'));self.assertFalse(d.allows('waiting'))
        with self.assertRaises(RuntimeError):d.observe(tuple('ab'),(0,1))
        with self.assertRaises(RuntimeError):d.observe(tuple('abcd'),(0,1,2,3))
        with self.assertRaises(RuntimeError):d.observe(tuple('abc'),(0,1))
        d.finished('b',cancelled=True,completed=False)
        d.observe(tuple('ac'),(1,0));self.assertEqual(d.order,tuple('ca'))
        self.assertEqual(d.head_completions,0)
        d.finished('c',completed=True);d.finished('a',completed=True)
        self.assertFalse(d.active);self.assertEqual(d.head_completions,2)

    def test_real_native_allows_other_safe_witness_without_fixed_scans(self):
        a=guarded_cohort();native_running_fixture(a,budget=18)
        a.requests['a'].num_computed_tokens=31;a.requests['a'].num_tokens_with_spec=32
        a.requests['b'].num_prompt_tokens=48;a.requests['b'].max_tokens=16
        a.requests['b'].num_tokens_with_spec=48
        with patch.dict(namespace,fixed_witness_pair=lambda *a,**k:(_ for _ in ()).throw(AssertionError('extra frozen scan'))):
            output=a.schedule()
        self.assertEqual(output.num_scheduled_tokens,{'a':1,'b':17})
        self.assertEqual(a._authority.drain_cut(),('b','a'))
        self.assertFalse(any(e['event']=='drain_frozen_witness_check' for e in a.events))
        frame=next(e for e in a.events if e['event']=='drain_epoch_witness')
        self.assertEqual(frame['head'],'a');self.assertEqual(frame['epoch_order'],['a','b'])
        self.assertEqual(a._authority.confirmed,a._snapshot()[0]);self.assertIsNone(a._authority.pending)
        with self.assertRaisesRegex(RuntimeError,'prior model batch confirmation'):a.schedule()

    def test_real_native_budget_one_selects_current_witness_head(self):
        a=guarded_cohort();native_running_fixture(a,budget=1)
        # The Authority certificate is authoritative, not manually altered drain order.
        a._drain.order=('b','a')
        output=a.schedule()
        self.assertEqual(output.num_scheduled_tokens,{'a':1})
        self.assertEqual([r.request_id for r in a.running],['a','b'])

    def test_new_mode_refuses_outside_and_follower_before_head(self):
        a=guarded_cohort()
        self.assertFalse(a._prc_allow(NS(request_id='waiting'),1,scheduled_tokens={'a':1}))
        self.assertFalse(a._prc_allow(a.requests['b'],1,scheduled_tokens={}))
        self.assertIsNone(a._authority.pending)

    def test_new_mode_empty_epoch_and_allocator_failure_are_not_progress(self):
        a=guarded_cohort();native_running_fixture(a,budget=2)
        a.requests['a'].num_tokens_with_spec=a.requests['a'].num_computed_tokens
        with self.assertRaisesRegex(RuntimeError,'no actual scheduled work'):a.schedule()
        a=guarded_cohort();native_running_fixture(a,budget=2,fail_allocation=True)
        with self.assertRaisesRegex(RuntimeError,'preemption is not permitted'):a.schedule()
        self.assertIsNone(a._authority.pending)

    def test_unsettled_execution_meter_batch_blocks_refresh(self):
        a=guarded_cohort();native_running_fixture(a,budget=1)
        a._execution_batches={17:object()}
        with self.assertRaisesRegex(RuntimeError,'prior model batch confirmation'):a.schedule()
        self.assertFalse(any(e['event']=='drain_epoch_witness' for e in a.events))

    def test_actual_model_join_counts_real_unique_execution_and_rejects_replay(self):
        original=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CertificateScheduler')
        method=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='update_from_output')
        calls=[]
        class Base:
            def update_from_output(self,output,result):
                calls.append('actual_base_return');self._execution_meter.confirm(self.batch_id,result.req_ids)
                return 'unchanged'
        definition=ast.ClassDef(name='Observed',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
        env=dict(Base=Base)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[])),str(SOURCE),'exec'),env)
        a=env['Observed']();events=[];a._log=lambda event,**kw:events.append(dict(event=event,**kw))
        a._execution_meter=ExecutionTokenMeter();a.batch_id=0
        output=NS(num_scheduled_tokens={'a':1,'b':2})
        proof=dict(episode=1,batch_index=2,head='a',batching='cohort_certified',frozen_order=['a','b'],
                   scheduled_tokens=output.num_scheduled_tokens,head_tokens=1,
                   order_semantics='current_epoch_authority_witness; membership_frozen_until_confirmed_terminal')
        a._execution_meter.prepare(0,[('a',0,1),('b',4,2)])
        a._drain_pending_batches={id(output):(output,proof)}
        self.assertEqual(a.update_from_output(output,NS(req_ids=['a','b'])),'unchanged')
        self.assertEqual(len(calls),1);self.assertFalse(a._drain_pending_batches)
        self.assertEqual(events[-1]['confirmed_real_positions'],{'a':1,'b':2})
        self.assertEqual(events[-1]['confirmed_new_unique_positions'],3)
        a.batch_id=1;a._execution_meter.prepare(1,[('a',0,1),('b',4,2)])
        a._drain_pending_batches={id(output):(output,proof)}
        with self.assertRaisesRegex(AssertionError,'unique confirmed execution'):a.update_from_output(output,NS(req_ids=['a','b']))
        self.assertTrue(a._drain_pending_batches)


SEMANTICS='current_epoch_authority_witness; membership_frozen_until_confirmed_terminal'

def receipts():
    events=[dict(event='drain_start',cohort=['a','b']),
        dict(event='drain_epoch_witness',episode=1,batch_index=1,head='a',epoch_order=['a','b'],cohort_members=['a','b'],authority_version=0,pending_grant=False,prior_model_batch_pending=False,order_semantics=SEMANTICS),
        dict(event='drain_head_first',episode=1,batch_index=1,head='a',frozen_order=['a','b'],before=['b','a'],after=['a','b'],order_semantics=SEMANTICS),
        dict(event='snapshot',cohort=['a','b'],free=3,hard_need=[1,1],groups=[dict(block='1',owners=['a']),dict(block='2',owners=['b'])]),
        dict(event='drain_batch',episode=1,batch_index=1,head='a',frozen_order=['a','b'],batching='cohort_certified',scheduled_tokens={'a':1,'b':1},head_tokens=1,empty=False,order_semantics=SEMANTICS),
        dict(event='drain_batch_confirmed',episode=1,batch_index=1,head='a',frozen_order=['a','b'],batching='cohort_certified',scheduled_tokens={'a':1,'b':1},head_tokens=1,physical_execution_observed=True,returned_request_ids=['a','b'],confirmed_real_positions={'a':1,'b':1},confirmed_new_unique_positions=2,order_semantics=SEMANTICS),
        dict(event='causal_finish',request_id='a',outcome='completed',release_observed=True),
        dict(event='causal_finish',request_id='b',outcome='completed',release_observed=True),
        dict(event='drain_segment_end',remaining_frozen_requests=[])]
    for i,e in enumerate(events):e.update(source_event_index=i,monotonic_ns=i+1,step=1)
    return events


class CohortReceiptTests(unittest.TestCase):
    def check(self,events):
        _verify_cohort_certified_batches(events,dict(drain_batching='cohort_certified',drain_scope='cohort',drain_policy='timer',audit=True,certificate_variant='common_full_c1'),5)
    def test_reference_replay_and_receipts_pass(self):self.check(receipts())
    def test_relative_follower_reordering_is_rejected(self):
        e=receipts()
        # A third resident follower makes a reversed follower order observable.
        e[0]['cohort']=['a','b','c'];e[1]['epoch_order']=['a','b','c'];e[1]['cohort_members']=['a','b','c']
        for index in (2,4,5):e[index]['frozen_order']=['a','b','c']
        e[2]['before']=['b','c','a'];e[2]['after']=['a','c','b']
        e[3]['cohort']=['a','b','c'];e[3]['hard_need']=[1,1,1];e[3]['free']=2
        e[3]['groups'].append(dict(block='3',owners=['c']))
        e.insert(-1,dict(event='causal_finish',request_id='c',outcome='completed',release_observed=True))
        for i,event in enumerate(e):event.update(source_event_index=i,monotonic_ns=i+1,step=1)
        with self.assertRaisesRegex(AssertionError,'priority was not stable'):self.check(e)
    def test_overlapping_epoch_refuses_forged_pending_false_flag(self):
        original=receipts()
        second=[copy.deepcopy(original[i]) for i in (1,2,3,4,5)]
        for event in second:
            event['step']=2
            if 'batch_index' in event:event['batch_index']=2
        e=original[:5]+second[:4]+[original[5],second[4]]+original[6:]
        for i,event in enumerate(e):event.update(source_event_index=i,monotonic_ns=i+1)
        with self.assertRaisesRegex(AssertionError,'prior model confirmation'):self.check(e)

    def test_actual_audit_without_capacity_is_rejected(self):
        with self.assertRaisesRegex(AssertionError,'measured physical capacity'):
            _verify_cohort_certified_batches(receipts(),dict(drain_batching='cohort_certified',drain_scope='cohort',drain_policy='timer',audit=True),None)
    def test_forged_ordinal_and_early_confirmation_fail(self):
        for variant in ('ordinal','order','count','outside','head','missing'):
            e=receipts()
            if variant=='ordinal':e[5]['source_event_index']=0
            if variant=='order':e[4],e[5]=e[5],e[4]
            if variant=='count':e[5]['confirmed_new_unique_positions']=0
            if variant=='outside':e[1]['epoch_order'].append('waiting')
            if variant=='head':e[4]['scheduled_tokens']={'b':1}
            if variant=='missing':e.pop(5)
            with self.assertRaises(AssertionError,msg=variant):self.check(e)


if __name__=='__main__':unittest.main()
