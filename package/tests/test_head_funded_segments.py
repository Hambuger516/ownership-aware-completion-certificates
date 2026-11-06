"""CPU-only selected-head, native-hook, receipt and model-join fixtures."""
import ast
import copy
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from prc_nc.algorithms import indexed_closure,LinearBuildMarginIndex
from prc_nc.model import valid_witness
from prc_nc.drain import DrainPolicy
from prc_nc.configuration import validate_config
from prc_nc.runtime import AllocationAuthority
from prc_nc.execution_meter import ExecutionTokenMeter
from prc_nc.analysis import _verify_head_funded
from test_adapter_metadata import fixture,SOURCE,namespace
from test_witness_preserving_batch import native_running_fixture,configuration


def head_fixture(variant='common_full_c1',budget=2):
    a=fixture(audit=True)
    def log(event,**values):
        i=len(a.events);a._study_log_event_index=i+1
        a.events.append(dict(event=event,step=a._step,source_event_index=i,monotonic_ns=i+1,**values))
    a._log=log
    blocks=[NS(block_id=i,is_null=i==0,ref_cnt=0) for i in range(6)]
    mapping={rid:[blocks[i+1]] for i,rid in enumerate('abc')}
    for row in mapping.values():
        for b in row:b.ref_cnt+=1
    a._pm=NS(req_to_blocks=mapping,block_size=16)
    a.kv_cache_manager=NS(block_pool=NS(blocks=blocks,num_gpu_blocks=6,
        get_num_free_blocks=lambda:sum(not b.is_null and b.ref_cnt==0 for b in blocks)))
    a.requests={rid:NS(request_id=rid,num_prompt_tokens=15 if rid=='a' else 16,
        max_tokens={'a':1,'b':48,'c':16}[rid],num_computed_tokens=14 if rid=='a' else 16,
        num_preemptions=0,output_token_ids=[]) for rid in 'abc'}
    a._zero_growth_fast=a._deny_cache_enabled=a._rank_order=a._pipeline_enabled=False
    a._serial=0;a._pending_commit={};a._proposals={};a._online=None
    a._drain=DrainPolicy(1,scope='head',batching='head_funded')
    state,ids,_=a._snapshot()
    a._authority=AllocationAuthority(state,ids,audit=True,certificate_variant=variant,
        on_event=lambda r:a._log('causal_'+r['kind'],**r))
    # A complete retained Authority witness need not be the fresh closure order.
    # Install this independently valid abc witness/index at the settled cut.
    a._authority.controller.order=(0,1,2)
    a._authority.controller.index=LinearBuildMarginIndex(state,(0,1,2))
    native_running_fixture(a,budget=budget)
    a.running=[a.requests[r] for r in 'cba']
    return a


def complete_model_observer(a,output):
    """Execute the actual observer method over real ExecutionTokenMeter API."""
    original=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CertificateScheduler')
    method=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='update_from_output')
    class Base:
        def update_from_output(self,output,result):
            self._execution_meter.confirm(self._cpu_model_batch_id,result.req_ids);return 'same-base-result'
    cls=ast.ClassDef(name='Observed',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])
    env=dict(Base=Base)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(SOURCE),'exec'),env)
    observed=env['Observed']();observed.__dict__=a.__dict__
    if getattr(a,"_execution_meter",None) is None:a._execution_meter=ExecutionTokenMeter()
    a._execution_schedule_ns=getattr(a,"_execution_schedule_ns",0)
    a._execution_schedule_calls=getattr(a,"_execution_schedule_calls",0)
    a._cpu_model_batch_id=max(a._execution_meter.done,default=-1)+1
    a._execution_meter.prepare(a._cpu_model_batch_id,[(rid,a.requests[rid].num_computed_tokens,n) for rid,n in output.num_scheduled_tokens.items()])
    return observed.update_from_output(output,NS(req_ids=list(output.num_scheduled_tokens)))


def finish_selected(a):
    rid=a._drain.selected_head
    def release():
        for block in a._pm.req_to_blocks.pop(rid):block.ref_cnt-=1
    a._authority.release(rid,release,lambda:a._snapshot()[:2])
    ending=a._drain.finished(rid,completed=True)
    a._log('drain_segment_end',**ending)
    return ending


def replay_events(a):
    events=json.loads(json.dumps(a.events))
    for i,e in enumerate(events):e.update(source_event_index=i,monotonic_ns=i+1,step=e.get('step',1))
    return events


class HeadFundedPolicyTests(unittest.TestCase):
    def setUp(self):
        old=namespace.get('_instrumented_schedule');self.addCleanup(lambda:namespace.__setitem__('_instrumented_schedule',old))

    def test_config_and_filtered_admission_pairing(self):
        c=configuration(drain_scope='head',drain_batching='head_funded',admission_policy='c0_first_shorter_cap')
        self.assertEqual(validate_config(c)['drain_scope'],'head')
        for changes in (dict(drain_scope='cohort'),dict(drain_policy='none'),dict(rank_order=True),dict(deny_cache=True),dict(zero_growth_fast=True),dict(enable_prc_pipeline=True)):
            with self.assertRaises(ValueError):validate_config(c|changes)
        with self.assertRaises(ValueError):DrainPolicy(1,scope='cohort',batching='head_funded')

    def test_persistent_head_ignores_changed_witness_and_nonhead_completion(self):
        d=DrainPolicy(1,scope='head',batching='head_funded');d.observe(tuple('abc'),(0,1,2))
        d.observe(tuple('abc'),(2,1,0));self.assertEqual(d.selected_head,'a');self.assertEqual(d.order,tuple('abc'))
        d.finished('b',completed=True);self.assertTrue(d.active);self.assertEqual(d.head_completions,0)
        d.observe(tuple('ac'),(1,0));self.assertEqual(d.selected_head,'a')
        end=d.finished('a',completed=True)
        self.assertEqual(end['reason'],'selected_head_completed');self.assertEqual(end['remaining_frozen_requests'],['c'])
        self.assertFalse(d.active);self.assertEqual(d.head_completions,1);self.assertEqual(d.elapsed,0)

    def test_cancelled_head_waits_for_current_witness_replacement(self):
        d=DrainPolicy(1,scope='head',batching='head_funded');d.observe(tuple('abc'),(0,1,2))
        with self.assertRaises(RuntimeError):d.observe(tuple('bc'),(0,1))
        d.finished('a',cancelled=True,completed=False)
        self.assertIsNone(d.selected_head);self.assertTrue(d.replacement_pending);self.assertFalse(d.allows('b'))
        d.observe(tuple('bc'),(1,0));self.assertEqual(d.selected_head,'c');self.assertEqual(d.last_selection,'selected_head_cancelled')
        self.assertTrue(d.allows('b'));self.assertEqual(d.head_completions,0)
        d.finished('c',cancelled=True,completed=False);end=d.finished('b',cancelled=True,completed=False)
        self.assertEqual(end['successful_head_completions'],0);self.assertEqual(d.completed_segments,0);self.assertFalse(d.active)

    def test_no_unknown_disappearance_outside_admission_or_false_terminal(self):
        d=DrainPolicy(1,scope='head',batching='head_funded');d.observe(tuple('ab'),(0,1))
        with self.assertRaises(RuntimeError):d.observe(tuple('abc'),(0,1,2))
        with self.assertRaises(RuntimeError):d.observe(tuple('a'),(0,))
        with self.assertRaises(RuntimeError):d.finished('a',completed=False,cancelled=False)
        self.assertFalse(d.allows('waiting'));self.assertTrue(d.active)

    def test_real_native_accepts_follower_that_old_full_order_rejects(self):
        for variant in ('common_full_c0','common_full_c1','shared_incremental'):
            a=head_fixture(variant)
            output=a.schedule();self.assertEqual(output.num_scheduled_tokens,{'a':1,'c':1})
            state,ids,_=a._snapshot();self.assertFalse(valid_witness(state,tuple(ids.index(r) for r in 'abc')))
            self.assertTrue(valid_witness(state,tuple(ids.index(r) for r in 'acb')))
            self.assertEqual(a._drain.selected_head,'a')
            self.assertFalse(any(e['event']=='drain_frozen_witness_check' for e in a.events))
            funding=[e for e in a.events if e['event']=='drain_head_funding_check']
            self.assertEqual(len(funding),2);self.assertTrue(all(e['accepted'] for e in funding))
            with self.assertRaisesRegex(RuntimeError,'prior model batch'):a.schedule()

    def test_model_join_and_successful_head_end_with_survivors_replay(self):
        a=head_fixture();output=a.schedule()
        self.assertEqual(complete_model_observer(a,output),'same-base-result')
        self.assertFalse(a._drain_pending_batches)
        end=finish_selected(a);self.assertEqual(end['remaining_frozen_requests'],['b','c'])
        self.assertFalse(a._drain.active)
        self.assertEqual(_verify_head_funded(replay_events(a),dict(drain_batching='head_funded',drain_scope='head',drain_policy='timer',certificate_variant='common_full_c1',audit=True),5),set())

    def test_unfunded_follower_defer_is_not_global_certificate_rejection(self):
        # Use the original two-request fixture: head a needs2 of3 free pages.
        from test_witness_preserving_batch import guarded_fixture
        a=guarded_fixture();a._drain=DrainPolicy(1,scope='head',batching='head_funded')
        native_running_fixture(a,budget=18)
        a.requests['a'].num_computed_tokens=31;a.requests['a'].num_tokens_with_spec=32
        a.requests['b'].num_prompt_tokens=48;a.requests['b'].max_tokens=16;a.requests['b'].num_tokens_with_spec=48
        a._authority._on_event=lambda r:a._log('causal_'+r['kind'],**r)
        output=a.schedule();self.assertEqual(output.num_scheduled_tokens,{'a':1})
        check=next(e for e in a.events if e['event']=='drain_head_funding_check' and e['request_id']=='b')
        self.assertFalse(check['accepted']);self.assertEqual(check['certificate_status'],'not_evaluated_head_funding_defer')
        self.assertFalse(check['authority_checked']);self.assertIsNone(check['grant_id'])
        inc=next(e for e in a.events if e['event']=='incremental_proposal' and e['request_id']=='b')
        self.assertFalse(inc['oracle_checked']);self.assertIsNone(inc['equivalent']);self.assertTrue(inc['head_funding_defer'])
        cf=next(e for e in a.events if e['event']=='study_candidate_counterfactual' and e['request_id']=='b')
        self.assertTrue(cf['shared_certificate'])
        complete_model_observer(a,output);finish_selected(a)
        events=replay_events(a);verified=_verify_head_funded(events,dict(drain_batching='head_funded',drain_scope='head',drain_policy='timer',certificate_variant='common_full_c1',audit=True),6)
        self.assertEqual(len(verified),1)

    def test_cancel_replacement_next_settled_epoch_and_all_cancel_replay(self):
        config=dict(drain_batching='head_funded',drain_scope='head',drain_policy='timer',certificate_variant='common_full_c1',audit=True)
        def cancel(a,rid):
            def release():
                for block in a._pm.req_to_blocks.pop(rid):block.ref_cnt-=1
            a._authority.release(rid,release,lambda:a._snapshot()[:2],cancelled=True)
            end=a._drain.finished(rid,cancelled=True,completed=False)
            if end:a._log('drain_segment_end',**end)
            return end
        a=head_fixture();output=a.schedule();complete_model_observer(a,output)
        for rid,n in output.num_scheduled_tokens.items():
            a.requests[rid].num_computed_tokens+=n;a.requests[rid].num_tokens_with_spec=a.requests[rid].num_computed_tokens+1
        cancel(a,'a');self.assertIsNone(a._drain.selected_head);self.assertTrue(a._drain.replacement_pending)
        a.running=[a.requests[r] for r in 'cb']
        output=a.schedule();self.assertEqual(a._drain.selected_head,'c')
        complete_model_observer(a,output);finish_selected(a)
        epochs=[e for e in a.events if e['event']=='drain_head_funded_epoch']
        self.assertEqual(epochs[1]['head_selection_cause'],'selected_head_cancelled');self.assertTrue(epochs[1]['head_replacement'])
        self.assertEqual(_verify_head_funded(replay_events(a),config,5),set())
        b=head_fixture();output=b.schedule();complete_model_observer(b,output)
        for rid in 'acb':end=cancel(b,rid)
        self.assertEqual(end['successful_head_completions'],0)
        self.assertEqual(_verify_head_funded(replay_events(b),config,5),set())
        events=replay_events(b);next(e for e in events if e['event']=='drain_segment_end')['successful_head_completions']=1
        with self.assertRaises(AssertionError):_verify_head_funded(events,config,5)

    def test_forged_pending_false_cannot_overlap_epochs(self):
        a=head_fixture();output=a.schedule();complete_model_observer(a,output);finish_selected(a)
        events=replay_events(a)
        frame=copy.deepcopy(next(e for e in events if e['event']=='drain_head_funded_epoch'))
        frame['batch_index']=2;frame['head_selection_cause']='retained'
        j=next(i for i,e in enumerate(events) if e['event']=='drain_batch_confirmed')
        events.insert(j,frame)
        for i,event in enumerate(events):event.update(source_event_index=i,monotonic_ns=i+1)
        with self.assertRaisesRegex(AssertionError,'prior model confirmation'):
            _verify_head_funded(events,dict(drain_batching='head_funded',drain_scope='head',drain_policy='timer',certificate_variant='common_full_c1',audit=True),5)

    def test_full_summary_accepts_verified_unchecked_funding_deferral(self):
        import tempfile
        from pathlib import Path
        from prc_nc.analysis import summarize
        from prc_nc.drain import terminal_flags
        from test_witness_preserving_batch import guarded_fixture
        a=guarded_fixture();a._drain=DrainPolicy(1,scope='head',batching='head_funded')
        native_running_fixture(a,budget=18)
        a.requests['a'].num_computed_tokens=31;a.requests['a'].num_tokens_with_spec=32
        a.requests['b'].num_prompt_tokens=48;a.requests['b'].max_tokens=16;a.requests['b'].num_tokens_with_spec=48
        rename={'a':'w0-r0','b':'w0-r1'}
        a._pm.req_to_blocks={rename[r]:v for r,v in a._pm.req_to_blocks.items()}
        a.requests={rename[r]:v for r,v in a.requests.items()}
        for rid,r in a.requests.items():r.request_id=rid
        state,ids,_=a._snapshot()
        a._authority=AllocationAuthority(state,ids,audit=True,certificate_variant='common_full_c1',on_event=lambda r:a._log('causal_'+r['kind'],**r))
        output=a.schedule();self.assertEqual(output.num_scheduled_tokens,{'w0-r0':1});complete_model_observer(a,output)
        # Execute the real physical terminal wrapper, including Authority.release
        # and DrainPolicy.finished; only the physical backend is a CPU fixture.
        cls=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CertificateScheduler')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        free=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name=='traced_free')
        def original_free(request):
            for block in a._pm.req_to_blocks.pop(request.request_id):block.ref_cnt-=1
        env=dict(namespace,self=a,original_free=original_free,terminal_flags=terminal_flags)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[free],type_ignores=[])),str(SOURCE),'exec'),env)
        a._release_epoch=0;a._deny_cache={};a._flush_events=lambda:None
        for request in a.requests.values():request.status='FINISHED_STOPPED';env['traced_free'](request)
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t)
            config=validate_config(configuration(drain_scope='head',drain_batching='head_funded',count=2,waves=1))
            manifest=dict(status='PASS',configuration=config,source_sha256={},physical_capacity_pages=6,request_caps=[32,16],completed=2,output_tokens=2)
            (folder/'manifest.json').write_text(json.dumps(manifest))
            records=[dict(request_id=rid,terminal='completed',tokens=[1],cap=cap,submitted_at_s=0,offered_at_s=0) for rid,cap in zip(('w0-r0','w0-r1'),(32,16))]
            (folder/'requests.json').write_text(json.dumps(records))
            (folder/'allocator_events.jsonl').write_text('\n'.join(json.dumps(e) for e in replay_events(a)))
            report,_,_=summarize(folder)
            self.assertEqual(report['status'],'PASS');self.assertEqual(report['head_funding_deferrals_verified'],1)
            # The model meter alone cannot substitute for the physical grant
            # lifecycle. These mutate copies of the actual CPU native-hook logs.
            original=replay_events(a)
            mutations=('delete_execute_and_ack','delete_execute','delete_ack','duplicate_execute','duplicate_ack',
                       'execute_grant','ack_version','executed_post','projected_post_hash',
                       'head_ack_after_follower','execute_deferred','orphan_execute','orphan_ack')
            for mutation in mutations:
                events=copy.deepcopy(original)
                if mutation.startswith('delete_'):
                    kinds=({'causal_execute','causal_acknowledge'} if mutation=='delete_execute_and_ack'
                           else {'causal_execute'} if mutation=='delete_execute' else {'causal_acknowledge'})
                    events=[e for e in events if e['event'] not in kinds]
                if mutation in ('duplicate_execute','duplicate_ack'):
                    kind='causal_execute' if mutation=='duplicate_execute' else 'causal_acknowledge'
                    index=next(i for i,e in enumerate(events) if e['event']==kind)
                    events.insert(index,copy.deepcopy(events[index]))
                if mutation=='execute_grant':next(e for e in events if e['event']=='causal_execute')['grant_id']='ghost:grant'
                if mutation=='ack_version':next(e for e in events if e['event']=='causal_acknowledge')['state_version']+=1
                if mutation=='executed_post':next(e for e in events if e['event']=='causal_execute')['physical_free'][0]+=1
                if mutation=='projected_post_hash':next(e for e in events if e['event']=='drain_head_funding_check')['projected_post_sha256']='0'*64
                if mutation=='head_ack_after_follower':
                    moved=[e for e in events if e['event'] in ('causal_acknowledge','allocation')]
                    events=[e for e in events if e not in moved]
                    index=next(i for i,e in enumerate(events) if e['event']=='drain_batch')
                    events[index:index]=moved
                if mutation in ('orphan_execute','orphan_ack'):
                    kind='causal_execute' if mutation=='orphan_execute' else 'causal_acknowledge'
                    fake=copy.deepcopy(next(e for e in events if e['event']==kind));fake['grant_id']='orphan:grant'
                    index=next(i for i,e in enumerate(events) if e['event']=='drain_batch');events.insert(index,fake)
                if mutation=='execute_deferred':
                    deferred=next(e for e in events if e['event']=='drain_head_funding_check' and not e['accepted'])
                    fake=copy.deepcopy(next(e for e in events if e['event']=='causal_execute'))
                    fake.update(request_id=deferred['request_id'],grant_id=deferred['action_id']+':grant')
                    index=next(i for i,e in enumerate(events) if e['event']=='drain_batch');events.insert(index,fake)
                for i,event in enumerate(events):event.update(source_event_index=i,monotonic_ns=i+1)
                (folder/'allocator_events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
                with self.subTest(mutation=mutation),self.assertRaises(AssertionError):summarize(folder)

    def test_follower_authorization_must_follow_actual_head_ack(self):
        a=head_fixture();output=a.schedule();complete_model_observer(a,output);finish_selected(a)
        events=replay_events(a)
        moved=[e for e in events if e.get('request_id')=='a' and e['event'] in ('causal_acknowledge','allocation')]
        events=[e for e in events if e not in moved]
        index=next(i for i,e in enumerate(events) if e['event']=='causal_authorize' and e['request_id']=='c')
        events[index+1:index+1]=moved
        for i,event in enumerate(events):event.update(source_event_index=i,monotonic_ns=i+1)
        with self.assertRaisesRegex(AssertionError,'authorized before selected-head ACK'):
            _verify_head_funded(events,dict(drain_batching='head_funded',drain_scope='head',drain_policy='timer',certificate_variant='common_full_c1',audit=True),5)

    def test_empty_epoch_failed_allocation_and_missing_running_head_fail_closed(self):
        a=head_fixture();a.requests['a'].num_tokens_with_spec=a.requests['a'].num_computed_tokens
        with self.assertRaisesRegex(RuntimeError,'no actual scheduled work'):a.schedule()
        a=head_fixture();a.running=[a.requests['b'],a.requests['c']]
        with self.assertRaises(RuntimeError):a.schedule()
        a=head_fixture();native_running_fixture(a,budget=2,fail_allocation=True);a.running=[a.requests[r] for r in 'abc']
        with self.assertRaisesRegex(RuntimeError,'preemption is not permitted'):a.schedule()

    def test_receipt_forgery_ordinal_margin_grant_cf_and_head_replace_rejected(self):
        a=head_fixture();output=a.schedule();complete_model_observer(a,output);finish_selected(a)
        original=replay_events(a);config=dict(drain_batching='head_funded',drain_scope='head',drain_policy='timer',certificate_variant='common_full_c1',audit=True)
        for mutation in ('ordinal','margin','grant','cfhash','selection','modelcount','missing'):
            e=copy.deepcopy(original);fund=next(r for r in e if r['event']=='drain_head_funding_check')
            if mutation=='ordinal':fund['source_event_index']=0
            if mutation=='margin':fund['margin'][0]-=1
            if mutation=='grant':fund['grant_id']='ghost'
            if mutation=='cfhash':next(r for r in e if r['event']=='study_candidate_counterfactual')['projected_post_sha256']='0'*64
            if mutation=='selection':next(r for r in e if r['event']=='drain_head_funded_epoch')['head_replacement']=True
            if mutation=='modelcount':next(r for r in e if r['event']=='drain_batch_confirmed')['confirmed_new_unique_positions']=0
            if mutation=='missing':e.remove(fund)
            for i,event in enumerate(e):
                if mutation!='ordinal':event['source_event_index']=i
            with self.assertRaises(AssertionError,msg=mutation):_verify_head_funded(e,config,5)

if __name__=='__main__':unittest.main()
