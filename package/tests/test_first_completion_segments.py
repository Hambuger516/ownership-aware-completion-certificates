"""CPU native-source/model-result/callback tests, never GPU performance claims."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch
from prc_nc.analysis import _verify_cohort_certified_batches, summarize
from prc_nc.algorithms import LinearBuildMarginIndex
from prc_nc.configuration import validate_config
from prc_nc.drain import DrainPolicy, terminal_flags
from prc_nc.execution_meter import ExecutionTokenMeter
from prc_nc.runtime import AllocationAuthority
from test_head_funded_segments import head_fixture, replay_events
from test_cohort_certified_batch import guarded_cohort, receipts
from test_witness_preserving_batch import native_running_fixture, install_real_allocate_body, namespace, configuration, SOURCE


CONFIG=dict(drain_scope='first_completion',drain_batching='cohort_certified',drain_policy='timer',audit=True,certificate_variant='common_full_c1')


def first_fixture():
    a=head_fixture(budget=2)
    # A genuine long head may remain alive while an ordinary follower hits EOS.
    a.requests['a'].max_tokens=16
    rename=dict(a='w0-r0',b='w0-r1',c='w0-r2')
    a._pm.req_to_blocks={rename[r]:v for r,v in a._pm.req_to_blocks.items()}
    a.requests={rename[r]:v for r,v in a.requests.items()}
    for rid,r in a.requests.items():r.request_id=rid
    state,ids,_=a._snapshot()
    a._authority=AllocationAuthority(state,ids,audit=True,certificate_variant='common_full_c1',on_event=lambda r:a._log('causal_'+r['kind'],**r))
    a._authority.controller.order=(0,1,2)
    a._authority.controller.index=LinearBuildMarginIndex(state,(0,1,2))
    a._drain=DrainPolicy(1,scope='first_completion',batching='cohort_certified')
    a._execution_meter=ExecutionTokenMeter()
    a._execution_schedule_ns=0;a._execution_schedule_calls=0
    a._release_epoch=0;a._deny_cache={};a._flush_events=lambda:None
    original=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CertificateScheduler')
    init=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    free=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name=='traced_free')
    def original_free(request):
        for block in a._pm.req_to_blocks.pop(request.request_id):block.ref_cnt-=1
        return 'unchanged-physical-release'
    env=dict(namespace,self=a,original_free=original_free,terminal_flags=terminal_flags)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[free],type_ignores=[])),str(SOURCE),'exec'),env)
    a._cpu_actual_free=env['traced_free']
    return a


def actual_model_callback(a,output,terminals=(),during_close=None):
    """Real outer observer + real free wrapper + real meter; CPU backend only."""
    original=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CertificateScheduler')
    method=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='update_from_output')
    class Base:
        def update_from_output(self,scheduled,result):
            for rid,status in terminals:
                self.requests[rid].status=status
                self._cpu_actual_free(self.requests[rid])
                if during_close is not None:during_close()
            self._execution_meter.confirm(self._cpu_batch,result.req_ids)
            del self._execution_batches[id(scheduled)]
            return 'unchanged-real-result'
    cls=ast.ClassDef(name='Observed',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
    env=dict(Base=Base)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(SOURCE),'exec'),env)
    observed=env['Observed']();observed.__dict__=a.__dict__
    a._cpu_batch=max(a._execution_meter.done,default=-1)+1
    a._execution_meter.prepare(a._cpu_batch,[(rid,a.requests[rid].num_computed_tokens,n) for rid,n in output.num_scheduled_tokens.items()])
    a._execution_batches={id(output):(output,a._cpu_batch)}
    return observed.update_from_output(output,NS(req_ids=list(output.num_scheduled_tokens)))


def positive_events(multifinish=False):
    a=first_fixture();out=a.schedule()
    terms=[('w0-r2','FINISHED_STOPPED')]+([('w0-r0','FINISHED_STOPPED')] if multifinish else [])
    actual_model_callback(a,out,terms)
    return a,replay_events(a)


class FirstCompletionTests(unittest.TestCase):
    def setUp(self):
        old=namespace.get('_instrumented_schedule');self.addCleanup(lambda:namespace.__setitem__('_instrumented_schedule',old))

    def test_configuration_restricts_reviewed_mode_and_keeps_old_defaults(self):
        for variant in ('common_full_c0','common_full_c1','shared_incremental'):
            c=configuration(**CONFIG)|dict(certificate_variant=variant,admission_policy='c0_first_shorter_cap')
            self.assertEqual(validate_config(c)['drain_scope'],'first_completion')
        for changes in (dict(drain_batching='serial'),dict(drain_batching='head_funded'),dict(drain_batching='witness_preserving'),dict(drain_policy='none'),dict(method='native_plain'),dict(certificate_variant='force_full'),dict(rank_order=True),dict(deny_cache=True),dict(zero_growth_fast=True),dict(enable_prc_pipeline=True)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_config(configuration(**CONFIG)|changes)
        self.assertEqual(validate_config({})['drain_scope'],'cohort')
        with self.assertRaises(ValueError):DrainPolicy(1,scope='first_completion',batching='serial')

    def test_nonhead_success_ends_once_and_does_not_invent_head_service(self):
        d=DrainPolicy(1,scope='first_completion',batching='cohort_certified');d.observe(tuple('abc'),(0,1,2))
        end=d.finished('b',completed=True)
        self.assertEqual(end['reason'],'first_member_completed');self.assertEqual(end['remaining_frozen_requests'],['a','c'])
        self.assertEqual(end['successful_member_completions'],1);self.assertEqual(end['successful_head_completions'],0)
        self.assertEqual((d.member_completions,d.head_completions,d.completed_segments),(1,0,1))
        self.assertIsNone(d.finished('a',completed=True));self.assertEqual(d.member_completions,1)
        self.assertFalse(d.active);self.assertEqual(d.elapsed,0)

    def test_cancellation_is_not_success_and_all_cancel_is_zero(self):
        d=DrainPolicy(1,scope='first_completion',batching='cohort_certified');d.observe(tuple('ab'),(0,1))
        self.assertIsNone(d.finished('a',cancelled=True,completed=False));self.assertTrue(d.active)
        d.observe(('b',),(0,));end=d.finished('b',cancelled=True,completed=False)
        self.assertEqual(end['reason'],'cohort_empty_without_member_completion')
        self.assertEqual((end['successful_member_completions'],end['successful_head_completions'],d.completed_segments),(0,0,0))

    def test_unknown_terminal_member_disappearance_and_outside_are_rejected(self):
        d=DrainPolicy(1,scope='first_completion',batching='cohort_certified');d.observe(tuple('ab'),(0,1))
        with self.assertRaises(RuntimeError):d.finished('a',completed=False,cancelled=False)
        with self.assertRaises(RuntimeError):d.observe(('a',),(0,))
        with self.assertRaises(RuntimeError):d.observe(tuple('abc'),(0,1,2))
        self.assertIsNone(d.finished('outside',completed=True));self.assertTrue(d.active)
        self.assertFalse(d.allows('outside'))

    def test_real_native_uses_new_authority_witness_without_funding_guard(self):
        for variant in ('common_full_c0','common_full_c1','shared_incremental'):
            a=guarded_cohort(variant);a._drain=DrainPolicy(1,scope='first_completion',batching='cohort_certified')
            native_running_fixture(a,budget=18)
            a.requests['a'].num_computed_tokens=31;a.requests['a'].num_tokens_with_spec=32
            a.requests['b'].num_prompt_tokens=48;a.requests['b'].max_tokens=16;a.requests['b'].num_tokens_with_spec=48
            caps={rid:r.max_tokens for rid,r in a.requests.items()}
            with patch.dict(namespace,fixed_witness_pair=lambda *a,**k:(_ for _ in ()).throw(AssertionError('extra fixed scans'))):out=a.schedule()
            self.assertEqual(out.num_scheduled_tokens,{'a':1,'b':17})
            self.assertEqual(a._authority.drain_cut(),('b','a'))
            self.assertEqual(caps,{rid:r.max_tokens for rid,r in a.requests.items()})
            self.assertFalse(any(e['event'] in ('drain_head_funding_check','drain_frozen_witness_check') for e in a.events))
            self.assertFalse(a._prc_allow(NS(request_id='outside'),1,scheduled_tokens={'a':1}))

    def test_actual_nonhead_free_callback_precedes_real_model_confirmation(self):
        a,events=positive_events();self.assertFalse(a._drain.active)
        _verify_cohort_certified_batches(events,CONFIG,5)
        end=next(e for e in events if e['event']=='drain_segment_end')
        self.assertEqual(end['completed_request_id'],'w0-r2');self.assertEqual(end['successful_head_completions'],0)
        finish=next(e for e in events if e['event']=='causal_finish');result=next(e for e in events if e['event']=='drain_batch_confirmed')
        self.assertLess(finish['source_event_index'],end['source_event_index']);self.assertLess(end['source_event_index'],result['source_event_index'])
        self.assertEqual(a._drain.completed_segments,1);self.assertFalse(a._drain_pending_batches)

    def test_multifinish_same_batch_keeps_first_winner_and_confirms_every_position(self):
        a,events=positive_events(multifinish=True)
        _verify_cohort_certified_batches(events,CONFIG,5)
        self.assertEqual(sum(e['event']=='causal_finish' for e in events),2)
        self.assertEqual(sum(e['event']=='drain_segment_end' for e in events),1)
        self.assertEqual((a._drain.member_completions,a._drain.head_completions,a._drain.completed_segments),(1,0,1))
        result=next(e for e in events if e['event']=='drain_batch_confirmed')
        self.assertEqual(result['confirmed_real_positions'],{'w0-r0':1,'w0-r2':1})
        self.assertEqual(set(a._authority.ids),{'w0-r1'})

    def test_cancelled_epoch_head_then_same_batch_follower_success_is_not_head_completion(self):
        a=first_fixture()
        a.running=[a.requests[r] for r in ('w0-r1','w0-r2','w0-r0')]
        out=a.schedule();self.assertEqual(out.num_scheduled_tokens,{'w0-r0':1,'w0-r1':1})
        actual_model_callback(a,out,[('w0-r0','FINISHED_ABORTED'),('w0-r1','FINISHED_STOPPED')])
        _verify_cohort_certified_batches(replay_events(a),CONFIG,5)
        end=next(e for e in a.events if e['event']=='drain_segment_end')
        self.assertEqual(end['completed_request_id'],'w0-r1')
        self.assertEqual((end['successful_member_completions'],end['successful_head_completions']),(1,0))
        self.assertEqual((a._drain.member_completions,a._drain.head_completions,a._drain.completed_segments),(1,0,1))

    def test_real_callback_cancellation_and_completion_matrix_preserves_epoch_head(self):
        matrix=[
            ([('w0-r0','FINISHED_ABORTED'),('w0-r1','FINISHED_ABORTED'),('w0-r2','FINISHED_STOPPED')],'w0-r2',0,['w0-r0']),
            ([('w0-r1','FINISHED_ABORTED'),('w0-r0','FINISHED_STOPPED')],'w0-r0',1,[]),
            ([('w0-r0','FINISHED_ABORTED'),('w0-r1','FINISHED_STOPPED')],'w0-r1',0,['w0-r0']),
            ([('w0-r1','FINISHED_STOPPED'),('w0-r0','FINISHED_STOPPED'),('w0-r2','FINISHED_STOPPED')],'w0-r1',0,[])]
        for terminals,winner,heads,cancelled_heads in matrix:
            with self.subTest(terminals=terminals):
                a=first_fixture();a.requests['w0-r1'].max_tokens=16
                state,ids,_=a._snapshot()
                a._authority=AllocationAuthority(state,ids,audit=True,certificate_variant='common_full_c1',on_event=lambda r:a._log('causal_'+r['kind'],**r))
                a._authority.controller.order=(0,1,2);a._authority.controller.index=LinearBuildMarginIndex(state,(0,1,2))
                a.max_num_scheduled_tokens=3;a.scheduler_config.max_num_batched_tokens=3
                a.running=[a.requests[r] for r in ('w0-r1','w0-r2','w0-r0')]
                out=a.schedule();self.assertEqual(out.num_scheduled_tokens,{'w0-r0':1,'w0-r1':1,'w0-r2':1})
                actual_model_callback(a,out,terminals)
                _verify_cohort_certified_batches(replay_events(a),CONFIG,5)
                endings=[e for e in a.events if e['event']=='drain_segment_end'];self.assertEqual(len(endings),1)
                self.assertEqual(endings[0]['completed_request_id'],winner)
                self.assertEqual((a._drain.member_completions,a._drain.head_completions,a._drain.completed_segments),(1,heads,1))
                cancellations=[e for e in a.events if e['event']=='drain_head_cancelled']
                self.assertEqual([e['request_id'] for e in cancellations],cancelled_heads)
                self.assertTrue(all(e['next_frozen_head'] is None for e in cancellations))
                self.assertEqual(a._execution_meter.report()['pending_batches'],0)
                if cancelled_heads:
                    original=replay_events(a)
                    for mutation in ('extra_follower_cancel','missing_head_cancel','premature_replacement'):
                        forged=copy.deepcopy(original)
                        receipt=next(e for e in forged if e['event']=='drain_head_cancelled')
                        if mutation=='extra_follower_cancel':
                            fake=copy.deepcopy(receipt);fake['request_id']='w0-r1'
                            forged.insert(forged.index(receipt)+1,fake)
                        if mutation=='missing_head_cancel':forged.remove(receipt)
                        if mutation=='premature_replacement':receipt['next_frozen_head']='w0-r1'
                        for i,e in enumerate(forged):e.update(source_event_index=i,monotonic_ns=i+1)
                        with self.subTest(mutation=mutation),self.assertRaises(AssertionError):_verify_cohort_certified_batches(forged,CONFIG,5)

    def test_early_inactive_state_cannot_resume_before_actual_model_confirmation(self):
        a=first_fixture();out=a.schedule();attempts=[]
        def attempt():
            self.assertFalse(a._drain.active)
            with self.assertRaisesRegex(RuntimeError,'prior model batch confirmation'):a.schedule()
            attempts.append('blocked_after_actual_free_before_model_return')
        actual_model_callback(a,out,[('w0-r2','FINISHED_STOPPED')],during_close=attempt)
        self.assertEqual(len(attempts),1)
        self.assertFalse(a._drain_pending_batches);self.assertFalse(a._execution_batches)
        a._drain.after_steps=256
        namespace['_instrumented_schedule']=lambda a:NS(num_scheduled_tokens={})
        a.schedule()  # Genuine post-confirmation settled cut is allowed.
        self.assertFalse(a._drain.active)

    def test_unsettled_resource_maps_block_even_after_segment_inactive(self):
        for name in ('_pending_commit','_pending_projection','_pending_index'):
            a=first_fixture();a._drain.active=False;setattr(a,name,{'ghost':object()})
            with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'unsettled resource'):a.schedule()
            self.assertFalse(any(e['event']=='drain_start' for e in a.events))

    def test_failed_allocation_empty_epoch_and_missing_head_never_claim_progress(self):
        a=first_fixture()
        def failed_allocate(request,num_new_tokens,num_lookahead_tokens=0):return None
        install_real_allocate_body(a,failed_allocate)
        with self.assertRaisesRegex(RuntimeError,'preemption is not permitted'):a.schedule()
        self.assertEqual(a._drain.completed_segments,0)
        a=first_fixture();a.running=[a.requests['w0-r1']]
        with self.assertRaises(RuntimeError):a.schedule()
        a=first_fixture();namespace['_instrumented_schedule']=lambda a:NS(num_scheduled_tokens={})
        with self.assertRaisesRegex(RuntimeError,'no actual scheduled work'):a.schedule()
        self.assertEqual(a._drain.member_completions,0)

    def test_receipt_negative_controls_bind_winner_survivors_callback_and_actual_work(self):
        a,original=positive_events()
        for mutation in ('winner','head_credit','member_count','survivors','outcome','missing_terminal','not_executed','missing_model','zero_work','late_end','early_outside_admission','ordinal'):
            events=copy.deepcopy(original);end=next(e for e in events if e['event']=='drain_segment_end');terminal=next(e for e in events if e['event']=='causal_finish');result=next(e for e in events if e['event']=='drain_batch_confirmed')
            if mutation=='winner':end['completed_request_id']='outside'
            if mutation=='head_credit':end['successful_head_completions']=1
            if mutation=='member_count':end['successful_member_completions']=0
            if mutation=='survivors':end['remaining_frozen_requests']=[]
            if mutation=='outcome':terminal['outcome']='cancelled'
            if mutation=='missing_terminal':events.remove(terminal)
            if mutation=='not_executed':terminal['request_id']=end['completed_request_id']='w0-r1';end['remaining_frozen_requests']=['w0-r0','w0-r2']
            if mutation=='missing_model':events.remove(result)
            if mutation=='zero_work':result['confirmed_real_positions']['w0-r2']=0
            if mutation=='late_end':events.remove(end);events.append(end)
            if mutation=='early_outside_admission':events.insert(events.index(result),dict(event='proposal',request_id='outside',allow=True,step=1))
            for i,e in enumerate(events):e.update(source_event_index=i,monotonic_ns=i+1)
            if mutation=='ordinal':end['source_event_index']=0
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):_verify_cohort_certified_batches(events,CONFIG,5)

    def test_all_cancel_actual_callbacks_replay_without_success(self):
        a=first_fixture();out=a.schedule();actual_model_callback(a,out)
        for rid in list(a._pm.req_to_blocks):
            a.requests[rid].status='FINISHED_ABORTED';a._cpu_actual_free(a.requests[rid])
        events=replay_events(a);_verify_cohort_certified_batches(events,CONFIG,5)
        end=next(e for e in events if e['event']=='drain_segment_end')
        self.assertEqual(end['successful_member_completions'],0);self.assertEqual(a._drain.completed_segments,0)

    def test_old_whole_cohort_still_rejects_early_survivors(self):
        a,events=positive_events()
        with self.assertRaisesRegex(AssertionError,'before confirmed terminals'):
            _verify_cohort_certified_batches(events,CONFIG|dict(drain_scope='cohort'),5)
        _verify_cohort_certified_batches(receipts(),CONFIG|dict(drain_scope='cohort'),5)

    def test_stats_persists_distinct_member_counter(self):
        tree=ast.parse(SOURCE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CertificateScheduler')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_flush_events')
        env=dict(json=json);exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(SOURCE),'exec'),env)
        a,_=positive_events();a._ownership=None;a._event_buffer=[];a._events=NS(flush=lambda:None)
        with tempfile.TemporaryDirectory() as t:
            a._study_stats_path=Path(t)/'stats.json';env['_flush_events'](a)
            state=json.loads(a._study_stats_path.read_text())['drain_policy_state']
            self.assertEqual((state['member_completions'],state['head_completions'],state['completed_segments']),(1,0,1))

    def test_full_summary_positive_and_forged_early_end_negative(self):
        a,events=positive_events()
        # Complete the remaining physical callbacks; this preserves the one
        # segment winner and all later release accounting.
        for rid in list(a._pm.req_to_blocks):
            a.requests[rid].status='FINISHED_STOPPED';a._cpu_actual_free(a.requests[rid])
        events=replay_events(a)
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);config=validate_config(configuration(**CONFIG)|dict(count=3,waves=1))
            manifest=dict(status='PASS',configuration=config,source_sha256={},physical_capacity_pages=5,request_caps=[16,48,16],completed=3,output_tokens=3)
            (folder/'manifest.json').write_text(json.dumps(manifest))
            records=[dict(request_id=rid,terminal='completed',tokens=[1],cap=a.requests[rid].max_tokens,submitted_at_s=0,offered_at_s=0) for rid in a.requests]
            (folder/'requests.json').write_text(json.dumps(records))
            log=folder/'allocator_events.jsonl';log.write_text('\n'.join(json.dumps(e) for e in events))
            self.assertEqual(summarize(folder)[0]['status'],'PASS')
            next(e for e in events if e['event']=='drain_segment_end')['successful_head_completions']=1
            log.write_text('\n'.join(json.dumps(e) for e in events))
            with self.assertRaisesRegex(AssertionError,'Nonhead winner'):summarize(folder)


if __name__=='__main__':unittest.main()
