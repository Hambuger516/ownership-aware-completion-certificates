"""Targeted fixes for the two independently reproduced lifecycle blockers."""
import ast
import copy
from pathlib import Path
from types import MethodType,SimpleNamespace as NS
import unittest
from itertools import product
from prc_nc.active_admission_analysis import verify_active_admission
from prc_nc.analysis import _verify_cohort_certified_batches,_verify_ordinary_priority
from test_active_drain_admission import active_fixture,actual_observer_with_terminals,finish_all
from test_head_funded_segments import replay_events
from test_witness_preserving_batch import namespace,SOURCE
from native_source_fixture import native_source

VENDOR=native_source('vllm/v1/core/sched/scheduler.py')
KV=native_source('vllm/v1/core/kv_cache_manager.py')


class Request(NS):
    __hash__=object.__hash__


def hashable_requests(a):
    rows={rid:Request(**vars(r)) for rid,r in a.requests.items()}
    running_ids={r.request_id for r in a.running}
    for rid,r in rows.items():
        if not hasattr(r,'status'):r.status='RUNNING' if rid in running_ids else 'WAITING'
        for name,value in dict(next_decode_eligible_step=0,num_output_placeholders=0,is_prefill_chunk=False).items():
            if not hasattr(r,name):setattr(r,name,value)
    a.running[:]=[rows[r.request_id] for r in a.running]
    a.waiting.rows[:]=[rows[r.request_id] for r in a.waiting.rows]
    a.skipped_waiting.rows[:]=[rows[r.request_id] for r in a.skipped_waiting.rows]
    a.requests=rows
    return rows


def vendor_methods():
    cls=next(n for n in ast.parse(VENDOR.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
    return {n.name:copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef)}


def strip_annotations(node):
    for n in ast.walk(node):
        if isinstance(n,ast.FunctionDef):n.returns=None
        elif isinstance(n,ast.arg):n.annotation=None
    return node


def install_native_abort_path(a):
    """Actual finish_requests/free/block dispatch and actual KV.free, CPU pages."""
    rows=hashable_requests(a)
    class Status:
        WAITING='WAITING';RUNNING='RUNNING';WAITING_FOR_STREAMING_REQ='STREAM';WAITING_FOR_REMOTE_KVS='REMOTE'
        @staticmethod
        def is_finished(value):return value in ('FINISHED_ABORTED','FINISHED_STOPPED','FINISHED_LENGTH_CAPPED')
    methods=vendor_methods()
    names=('finish_requests','_free_request','_free_blocks','_free_request_blocks')
    tree=ast.ClassDef(name='NativeAbort',bases=[],keywords=[],decorator_list=[],body=[strip_annotations(methods[n]) for n in names])
    env=dict(RequestStatus=Status,remove_all=lambda running,removed:[r for r in running if r not in removed])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[tree],type_ignores=[])),str(VENDOR),'exec'),env)
    for name in names:setattr(a,name,MethodType(getattr(env['NativeAbort'],name),a))
    for r in rows.values():
        r.is_finished=lambda r=r:Status.is_finished(r.status)
    a.waiting.remove_requests=lambda reqs:setattr(a.waiting,'rows',[r for r in a.waiting.rows if r not in reqs])
    a.skipped_waiting.remove_requests=lambda reqs:setattr(a.skipped_waiting,'rows',[r for r in a.skipped_waiting.rows if r not in reqs])
    a._connector_finished=lambda request:(False,None)
    a.encoder_cache_manager=NS(free=lambda request:None)
    a.finished_req_ids=set();a.finished_req_ids_dict=None;a.defer_block_free=False
    # Preserve the exact vLLM KVCacheManager.free body, with only coordinator
    # page reclamation supplied by the already checked CPU physical backend.
    old_free=a._cpu_actual_free.__globals__['original_free']
    a.kv_cache_manager.coordinator=NS(free=lambda rid:old_free(a.requests[rid]))
    cls=next(n for n in ast.parse(KV.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='KVCacheManager')
    free=strip_annotations(copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='free')))
    env={};exec(compile(ast.fix_missing_locations(ast.Module(body=[free],type_ignores=[])),str(KV),'exec'),env)
    a._cpu_actual_free.__globals__['original_free']=MethodType(env['free'],a.kv_cache_manager)
    a.kv_cache_manager.free=a._cpu_actual_free
    return rows


def confirm_via_native_cursor_and_actual_observers(a,out,terminals=()):
    """Actual vendor cursor update plus both real measurement/outer observers.

    Only the model result is a CPU double. It offers one subsequent token to
    keep already-running decode requests eligible in the next vendor loop.
    """
    methods=vendor_methods();cursor=strip_annotations(methods['_update_after_schedule'])
    class Backend:
        def update_from_output(self,scheduled,result):
            terminal_ids={rid for rid,status in terminals}
            for rid,status in terminals:
                self.requests[rid].status=status;self._cpu_actual_free(self.requests[rid])
                self.running[:]=[r for r in self.running if r.request_id!=rid]
            for rid in result.req_ids:
                if rid in terminal_ids:continue
                r=self.requests[rid]
                r.num_tokens_with_spec=r.num_computed_tokens+1
                r.num_tokens=max(r.num_tokens,r.num_computed_tokens+1)
            return 'CPU-model-result-not-GPU'
    native=ast.ClassDef(name='Native',bases=[ast.Name(id='Backend',ctx=ast.Load())],keywords=[],decorator_list=[],body=[cursor])
    measured_cls=next(n for n in ast.parse(SOURCE.with_name('execution_measurement.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ExecutionMeasurementMixin')
    measured=ast.ClassDef(name='Measured',bases=[ast.Name(id='Native',ctx=ast.Load())],keywords=[],decorator_list=[],body=[copy.deepcopy(n) for n in measured_cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_update_after_schedule','update_from_output')])
    adapter=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CertificateScheduler')
    observed=ast.ClassDef(name='Observed',bases=[ast.Name(id='Measured',ctx=ast.Load())],keywords=[],decorator_list=[],body=[copy.deepcopy(next(n for n in adapter.body if isinstance(n,ast.FunctionDef) and n.name=='update_from_output'))])
    env=dict(Backend=Backend,time=namespace['time'],os=__import__('os'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[native,measured,observed],type_ignores=[])),str(VENDOR),'exec'),env)
    observer=env['Observed']();observer.__dict__=a.__dict__
    a._execution_serial=max(a._execution_meter.done,default=-1)+1
    a._execution_batches={};a._execution_update_ns=a._execution_update_calls=0
    a.defer_block_free=False;a.enable_return_routed_experts=False
    for r in a.requests.values():
        r.num_in_flight_tokens=0;r.use_structured_output=False
        if not hasattr(r,"num_output_placeholders"):r.num_output_placeholders=0
        if not hasattr(r,'num_tokens'):r.num_tokens=r.num_tokens_with_spec
    out.has_structured_output_requests=False
    a.has_requests=lambda:bool(a._authority.ids);a._flush_execution_meter=lambda:None
    observer._update_after_schedule(out)
    return observer.update_from_output(out,NS(req_ids=list(out.num_scheduled_tokens)))


class LifecycleRevisionTests(unittest.TestCase):
    def setUp(self):
        old=namespace.get('_instrumented_schedule');self.addCleanup(lambda:namespace.__setitem__('_instrumented_schedule',old))

    def test_never_proposed_waiting_abort_twice_actual_vendor_path_binds_once(self):
        a,c=active_fixture();rows=install_native_abort_path(a);request=rows['w0-r2']
        before_ids=a._authority.ids;before_state=a._authority.confirmed
        self.assertNotIn('w0-r2',a._active_join.bound)
        first=a.finish_requests(['w0-r2'],'FINISHED_ABORTED')
        count=len(a.events);version=a._authority.version;release_epoch=a._release_epoch
        second=a.finish_requests(['w0-r2'],'FINISHED_ABORTED')
        self.assertEqual(first,[request]);self.assertEqual(second,[])
        self.assertEqual((len(a.events),a._authority.version,a._release_epoch),(count,version,release_epoch))
        self.assertEqual(a._authority.ids,before_ids);self.assertEqual(a._authority.confirmed,before_state)
        self.assertEqual(a._active_join.bound['w0-r2'],'w0-r2');self.assertEqual(a._active_join.terminal_outcomes['w0-r2'],'cancelled')
        self.assertIn('w0-r2',a._active_join.retired)
        self.assertFalse(any(e['event']=='active_drain_join' for e in a.events))
        self.assertEqual((a._drain.head_completions,a._drain.member_completions),(0,0))
        self.assertEqual(a._execution_meter.executed,{})
        # Direct duplicate callback is also idempotent, though native second
        # finish_requests already avoids calling KV.free again after deletion.
        a._cpu_actual_free(request)
        self.assertEqual((len(a.events),a._authority.version,a._release_epoch),(count,version,release_epoch))

    def test_never_proposed_cancel_invalid_identity_or_metadata_rejects_before_release(self):
        for mutation in ('outside','bad_uuid','cap','tokens','length'):
            a,c=active_fixture();r=copy.copy(a.requests['w0-r2'])
            if mutation=='outside':r.request_id='w0-r99-deadbeef'
            if mutation=='bad_uuid':r.request_id='w0-r2-notuuid'
            if mutation=='cap':r.max_tokens+=1
            if mutation=='tokens':r.prompt_token_ids=[99999]
            if mutation=='length':r.num_prompt_tokens+=1
            r.status='FINISHED_ABORTED';before=(a._authority.version,a._authority.ids,a._authority.confirmed)
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):a._cpu_actual_free(r)
            self.assertEqual((a._authority.version,a._authority.ids,a._authority.confirmed),before)
            self.assertNotIn(r.request_id,a._authority._terminals)

    def test_repeated_terminal_same_UUID_is_idempotent_but_reuse_or_contradiction_rejects(self):
        a,c=active_fixture();r=a.requests['w0-r2'];r.status='FINISHED_ABORTED';a._cpu_actual_free(r)
        count=len(a.events);version=a._authority.version
        for mutation in ('other_uuid','cap','tokens','contradiction','nonterminal'):
            forged=copy.copy(r)
            if mutation=='other_uuid':forged.request_id='w0-r2-deadbeef'
            if mutation=='cap':forged.max_tokens+=1
            if mutation=='tokens':forged.prompt_token_ids=[99999]
            if mutation=='contradiction':forged.status='FINISHED_STOPPED'
            if mutation=='nonterminal':forged.status='RUNNING'
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):a._cpu_actual_free(forged)
        with self.assertRaises(RuntimeError):a._prc_allow(r,1,scheduled_tokens={})
        self.assertEqual((len(a.events),a._authority.version),(count,version))
        self.assertEqual(a._authority._terminals[r.request_id],'cancelled')

    def test_interepoch_survivor_cancel_never_invents_a_captured_head(self):
        a,c=active_fixture();hashable_requests(a);out=a.schedule()
        confirm_via_native_cursor_and_actual_observers(a,out,[('w0-r0','FINISHED_STOPPED')])
        self.assertIsNone(a._active_join.epoch);self.assertIsNone(a._drain._active_epoch_head)
        install_native_abort_path(a)
        self.assertEqual(len(a.finish_requests(['w0-r1'],'FINISHED_ABORTED')),1)
        self.assertFalse(any(e['event']=='drain_head_cancelled' for e in a.events))
        # Actual next native schedule selects the surviving current witness;
        # real cursor/meter observers confirm a new, non-reexecuted interval.
        following=a.schedule()
        self.assertEqual(next(reversed([e for e in a.events if e['event']=='drain_epoch_witness']))['head'],'w0-r2')
        confirm_via_native_cursor_and_actual_observers(a,following,[('w0-r2','FINISHED_STOPPED')])
        counts=verify_active_admission(replay_events(a),c,12)
        self.assertTrue(counts['exact_work_evidence']);self.assertEqual(counts['epochs'],2)
        self.assertEqual(a._execution_meter.report()['reexecuted_tokens'],0)
        # A forged late label is still rejected rather than weakening replay.
        events=replay_events(a);finish=next(e for e in events if e['event']=='causal_finish' and e['request_id']=='w0-r1')
        events.insert(events.index(finish)+1,dict(event='drain_head_cancelled',request_id='w0-r1',step=finish['step'],next_frozen_head=None,successful_completion=False))
        for i,e in enumerate(events):e.update(source_event_index=i,monotonic_ns=i+1)
        with self.assertRaises(AssertionError):verify_active_admission(events,c,12)

    def test_same_old_head_cancel_between_epochs_is_not_a_live_head_tag(self):
        a,c=active_fixture();out=a.schedule();actual_observer_with_terminals(a,out,[],list(out.num_scheduled_tokens))
        a.requests['w0-r0'].status='FINISHED_ABORTED';a._cpu_actual_free(a.requests['w0-r0'])
        self.assertFalse(any(e['event']=='drain_head_cancelled' for e in a.events))
        finish_all(a);self.assertTrue(verify_active_admission(replay_events(a),c,12)['exact_work_evidence'])

    def test_ordinary_true_inactive_to_active_joined_to_settled_actual_vendor_observers(self):
        for variant,enabled in product(('common_full_c0','common_full_c1','shared_incremental'),(False,True)):
            with self.subTest(certificate_variant=variant,active_drain_admission=enabled):
                a,c=active_fixture(variant);hashable_requests(a)
                a._active_drain_admission=enabled;a._drain.active_admission=enabled
                a._ordinary_witness_priority=True;c['ordinary_witness_priority']=True;c['rank_order']=False
                c['active_drain_admission']=enabled;a._drain.after_steps=2
                queued=a.waiting.rows.pop(0)
                ordinary=a.schedule();self.assertFalse(a._drain.active)
                confirm_via_native_cursor_and_actual_observers(a,ordinary)
                self.assertFalse(a._ordinary_pending_batches);self.assertFalse(a._execution_batches)
                a.waiting.rows.append(queued)
                active=a.schedule();self.assertTrue(a._drain.active)
                self.assertEqual('w0-r2' in active.num_scheduled_tokens,enabled)
                confirm_via_native_cursor_and_actual_observers(a,active)
                self.assertFalse(a._drain_pending_batches);self.assertFalse(a._execution_batches)
                if enabled:self.assertIsNone(a._active_join.epoch)
                finish_all(a);events=replay_events(a)
                self.assertEqual(_verify_ordinary_priority(events,c,12)['confirmed_batches'],1)
                _verify_cohort_certified_batches(events,c,12)
                self.assertEqual(sum(e['event']=='active_drain_join' for e in events),int(enabled))
                self.assertEqual(a._execution_meter.report()['reexecuted_tokens'],0)


if __name__=='__main__':unittest.main()


class TerminalOutcomeIdempotenceTests(unittest.TestCase):
    setUp=LifecycleRevisionTests.setUp
    def test_same_verified_member_outcome_repeats_once_and_reverse_outcome_rejects(self):
        for status,opposite in (('FINISHED_STOPPED','FINISHED_ABORTED'),('FINISHED_ABORTED','FINISHED_STOPPED')):
            with self.subTest(status=status):
                a,c=active_fixture();out=a.schedule()
                actual_observer_with_terminals(a,out,[('w0-r0',status)],list(out.num_scheduled_tokens))
                r=a.requests['w0-r0'];before=(len(a.events),a._authority.version,a._release_epoch,a._drain.member_completions,a._drain.head_completions,a._drain.cancelled_members)
                a._cpu_actual_free(r)
                after=(len(a.events),a._authority.version,a._release_epoch,a._drain.member_completions,a._drain.head_completions,a._drain.cancelled_members)
                self.assertEqual(after,before)
                r.status=opposite
                with self.assertRaisesRegex(RuntimeError,'contradictory'):a._cpu_actual_free(r)
                self.assertEqual((len(a.events),a._authority.version),(before[0],before[1]))
                r.status=status;finish_all(a)
                self.assertTrue(verify_active_admission(replay_events(a),c,12)['exact_work_evidence'])
