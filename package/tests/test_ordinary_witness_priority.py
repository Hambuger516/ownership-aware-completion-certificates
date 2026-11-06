"""Contingent CPU-only rank reuse, actual vendor-loop hooks and model receipts."""
import ast
import contextlib
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from prc_nc.configuration import validate_config
from prc_nc.drain import DrainPolicy,terminal_flags
from prc_nc.execution_meter import ExecutionTokenMeter
from prc_nc.runtime import AllocationAuthority
from prc_nc.analysis import _verify_ordinary_priority
from test_cohort_certified_batch import guarded_cohort
from test_head_funded_segments import complete_model_observer,replay_events
from test_witness_preserving_batch import native_running_fixture,install_real_allocate_body,namespace,configuration,SOURCE

CONFIG=dict(method='shared_incremental',certificate_variant='common_full_c1',ordinary_witness_priority=True,
    rank_order=False,drain_scope='cohort',drain_batching='cohort_certified',drain_policy='timer',
    admission_policy='c0_first_shorter_cap',audit=True)


def ordinary_fixture(variant='common_full_c1',budget=2,extra_free=0):
    a=guarded_cohort(variant);native_running_fixture(a,budget=budget)
    rename={'a':'w0-r0','b':'w0-r1'}
    a._pm.req_to_blocks={rename[r]:v for r,v in a._pm.req_to_blocks.items()}
    a.requests={rename[r]:v for r,v in a.requests.items()}
    for rid,r in a.requests.items():r.request_id=rid
    pool=a.kv_cache_manager.block_pool
    for _ in range(extra_free):pool.blocks.append(NS(block_id=len(pool.blocks),is_null=False,ref_cnt=0))
    pool.num_gpu_blocks=len(pool.blocks)
    a._drain=DrainPolicy(256,scope='cohort',batching='cohort_certified')
    a._ordinary_witness_priority=True;a._ordinary_pending_batches={}
    a._execution_meter=ExecutionTokenMeter();a._execution_schedule_ns=a._execution_schedule_calls=0
    def log(event,**values):
        i=len(a.events);a.events.append(dict(event=event,step=a._step,source_event_index=i,monotonic_ns=i+1,**values))
    a._log=log
    state,ids,_=a._snapshot()
    a._authority=AllocationAuthority(state,ids,audit=True,certificate_variant=variant,admission_policy='c0_first_shorter_cap',on_event=lambda r:a._log('causal_'+r['kind'],**r))
    return a


class Queue:
    def __init__(self,rows=()):self.rows=list(rows)
    def __bool__(self):return bool(self.rows)
    def __len__(self):return len(self.rows)
    def peek_request(self):return self.rows[0]
    def pop_request(self):return self.rows.pop(0)
    def prepend_request(self,r):self.rows.insert(0,r)
    def prepend_requests(self,q):self.rows=list(q.rows)+self.rows


def install_actual_vendor_waiting_prefix(a):
    """Extend the existing reviewed vendor fixture through its actual waiting
    loop. Only physical pages, queue objects and encoder/connector absence are
    CPU doubles; no waiting admission or append logic is hand-reimplemented.
    """
    # Obtain exact transformed native source using the existing tested helper
    # before restoring the renamed fixture's actual request identities.
    scratch=guarded_cohort();source=native_running_fixture(scratch,budget=3)
    function=ast.parse(source).body[0]
    waiting=next(i for i,node in enumerate(function.body) if isinstance(node,ast.If)
        and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='_select_waiting_queue_for_scheduling' for x in ast.walk(node)))
    function.body=function.body[:waiting+1]+[ast.Return(value=ast.Call(func=ast.Name(id='NS',ctx=ast.Load()),args=[],keywords=[ast.keyword(arg='num_scheduled_tokens',value=ast.Name(id='num_scheduled_tokens',ctx=ast.Load()))]))]
    function.returns=None
    for arg in function.args.args:arg.annotation=None
    env=dict(NS=NS,time=namespace['time'],PauseState=NS(PAUSED_ALL='paused',UNPAUSED='unpaused'),
        RequestStatus=NS(WAITING='WAITING',PREEMPTED='PREEMPTED',RUNNING='RUNNING',WAITING_FOR_REMOTE_KVS='REMOTE'),
        create_request_queue=lambda policy:Queue(),record_function_or_nullcontext=lambda name:contextlib.nullcontext())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'<actual-vllm-running-plus-waiting>','exec'),env)
    namespace['_instrumented_schedule']=env['schedule']
    a.lora_config=None;a.policy='fcfs';a.waiting=Queue();a.skipped_waiting=Queue()
    a.num_waiting_for_streaming_input=0;a.max_num_running_reqs=16
    a._select_waiting_queue_for_scheduling=lambda:a.waiting if a.waiting else a.skipped_waiting
    a._is_blocked_waiting_status=lambda status:False
    a._get_local_prefix_cache_hit=lambda request:(NS(blocks=((),)),0,None,False)
    a.connector=a.ec_connector=None;a.connector_prefix_cache_stats=None
    a.num_spec_tokens=0;a.dynamic_sd_lookup=None;a.scheduler_config.enable_chunked_prefill=True
    a.is_encoder_decoder=False;a.scheduler_reserve_full_isl=False;a.log_stats=False;a._inflight_prefills=set()
    a.kv_cache_manager.empty_kv_cache_blocks=NS(blocks=((),))
    a.kv_cache_manager.record_prefix_cache_stats=lambda *args:None
    a.kv_cache_manager.get_blocks=lambda rid:NS(blocks=(tuple(a._pm.req_to_blocks.get(rid,())),))
    def allocate(request,num_new_tokens,num_lookahead_tokens=0,num_new_computed_tokens=0,new_computed_blocks=None,num_external_computed_tokens=0,**other):
        rows=a._pm.req_to_blocks.setdefault(request.request_id,[])
        target=request.num_computed_tokens+num_new_computed_tokens+num_new_tokens
        count=max(0,(target+15)//16-len(rows))
        free=[b for b in a.kv_cache_manager.block_pool.blocks if not b.is_null and b.ref_cnt==0]
        if count>len(free):return None
        for block in free[:count]:block.ref_cnt=1;rows.append(block)
        return NS(blocks=(tuple(rows),))
    install_real_allocate_body(a,allocate)
    r=NS(request_id='w0-r2',num_prompt_tokens=1,max_tokens=16,num_computed_tokens=0,
        output_token_ids=[],num_preemptions=0,status='WAITING',num_stale_output_tokens=0,drop_stale_output=False,
        lora_request=None,prefill_stats=None,num_tokens=1,has_encoder_inputs=False,spec_token_ids=[],mm_features=[])
    a.requests[r.request_id]=r;a.waiting.rows.append(r)
    return r


class OrdinaryPriorityTests(unittest.TestCase):
    def setUp(self):
        old=namespace.get('_instrumented_schedule');self.addCleanup(lambda:namespace.__setitem__('_instrumented_schedule',old))

    def test_flag_defaults_off_and_cannot_double_enable_or_change_service_factor(self):
        self.assertFalse(validate_config({})['ordinary_witness_priority'])
        self.assertTrue(validate_config(configuration(**CONFIG))['ordinary_witness_priority'])
        for change in (dict(rank_order=True),dict(drain_scope='first_completion'),dict(drain_batching='serial'),dict(drain_policy='none'),dict(admission_policy='unrestricted'),dict(enable_prc_pipeline=True),dict(zero_growth_fast=True),dict(deny_cache=True),dict(ordinary_witness_priority=1)):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_config(configuration(**CONFIG)|change)

    def test_real_vendor_reorders_and_real_model_meter_confirms_activation(self):
        a=ordinary_fixture();out=a.schedule();self.assertEqual(out.num_scheduled_tokens,{'w0-r0':1,'w0-r1':1})
        frame=next(e for e in a.events if e['event']=='ordinary_witness_priority')
        self.assertEqual(frame['prior_running'],['w0-r1','w0-r0']);self.assertTrue(frame['relative_order_changed'])
        self.assertEqual(frame['reordered_running'],frame['witness'])
        complete_model_observer(a,out)
        counts=_verify_ordinary_priority(replay_events(a),CONFIG,6)
        self.assertEqual(counts['activated_workload_reorders'],1);self.assertEqual(counts['positive_head_work_batches'],1)

    def test_common_arms_same_overlap_witness_sort_and_both_pay_dual_searches(self):
        observed=[];cost=[]
        for variant in ('common_full_c0','common_full_c1'):
            a=ordinary_fixture(variant);before=a._authority.controller.stats['oracle'];out=a.schedule()
            frame=next(e for e in a.events if e['event']=='ordinary_witness_priority')
            observed.append((frame['witness'],frame['reordered_running'],out.num_scheduled_tokens))
            cost.append(a._authority.controller.stats['oracle']-before)
        self.assertEqual(observed[0],observed[1]);self.assertEqual(cost,[4,4])

    def test_actual_vendor_waiting_can_admit_new_member_outside_resident_witness(self):
        a=ordinary_fixture(budget=3,extra_free=3);new=install_actual_vendor_waiting_prefix(a)
        out=a.schedule();self.assertIn(new.request_id,out.num_scheduled_tokens);self.assertIn(new,a.running)
        frame=next(e for e in a.events if e['event']=='ordinary_witness_priority')
        self.assertNotIn(new.request_id,frame['resident_ids'])
        budget=next(e for e in a.events if e['event']=='ordinary_witness_budget')
        self.assertEqual(budget['scheduled_outside_initial_residents'],[new.request_id])
        self.assertIn(new.request_id,a._authority.ids);self.assertIsNone(a._authority.pending)
        complete_model_observer(a,out)
        counts=_verify_ordinary_priority(replay_events(a),CONFIG,9)
        self.assertEqual(counts['confirmed_batches'],1)

    def test_actual_vendor_waiting_rejection_still_breaks_without_backfill(self):
        a=ordinary_fixture(budget=4,extra_free=3);long=install_actual_vendor_waiting_prefix(a)
        long.max_tokens=160
        brief=copy.copy(long);brief.request_id='w0-r3';brief.max_tokens=16
        a.requests[brief.request_id]=brief;a.waiting.rows.append(brief)
        out=a.schedule()
        self.assertNotIn(long.request_id,out.num_scheduled_tokens);self.assertNotIn(brief.request_id,out.num_scheduled_tokens)
        self.assertEqual([r.request_id for r in a.waiting.rows],[long.request_id,brief.request_id])
        self.assertTrue(any(e['event']=='proposal' and e['request_id']==long.request_id and e['allow'] is False for e in a.events))
        self.assertFalse(any(e.get('request_id')==brief.request_id for e in a.events))

    def test_active_and_timer_trigger_steps_do_not_use_new_full_sort(self):
        for active in (False,True):
            a=ordinary_fixture();a._drain=DrainPolicy(1,scope='cohort',batching='cohort_certified')
            if active:
                state,ids,_=a._snapshot();a._drain.observe(ids,tuple(range(len(ids))))
            out=a.schedule()
            self.assertTrue(a._drain.active);self.assertFalse(any(e['event'].startswith('ordinary_witness') for e in a.events))
            self.assertTrue(any(e['event']=='drain_head_first' for e in a.events))

    def test_pending_resource_model_and_meter_batches_reject_before_sort(self):
        for phase in ('ordinary','trigger','already_active'):
            for name in ('_ordinary_pending_batches','_pending_commit','_pending_projection','_pending_index','_execution_batches','meter_pending'):
                a=ordinary_fixture()
                if phase!='ordinary':
                    a._drain=DrainPolicy(1,scope='cohort',batching='cohort_certified')
                    if phase=='already_active':
                        state,ids,_=a._snapshot();a._drain.observe(ids,tuple(range(len(ids))))
                elapsed=a._drain.elapsed
                if name=='meter_pending':a._execution_meter.prepare(17,[('w0-r0',0,1)])
                else:setattr(a,name,{17:object()})
                with self.subTest(phase=phase,name=name),self.assertRaises(RuntimeError):a.schedule()
                self.assertEqual(a._drain.elapsed,elapsed)
                self.assertFalse(any(e['event'] in ('ordinary_witness_priority','drain_start','drain_dispatch') for e in a.events))

    def test_zero_head_budget_absent_head_and_empty_running_are_observations_not_failure(self):
        for kind in ('ineligible','absent','empty'):
            a=ordinary_fixture()
            if kind=='ineligible':a.requests['w0-r0'].num_tokens_with_spec=a.requests['w0-r0'].num_computed_tokens
            if kind=='absent':a.running=[a.requests['w0-r1']]
            if kind=='empty':a.running=[]
            out=a.schedule()
            if out.num_scheduled_tokens:complete_model_observer(a,out)
            counts=_verify_ordinary_priority(replay_events(a),CONFIG,6)
            self.assertEqual(counts['positive_head_work_batches'],0)
            budget=next(e for e in a.events if e['event']=='ordinary_witness_budget');self.assertEqual(budget['head_tokens'],0)

    def test_flag_off_retains_native_running_order_without_new_observations(self):
        a=ordinary_fixture();a._ordinary_witness_priority=False;out=a.schedule()
        self.assertEqual([r.request_id for r in a.running],['w0-r1','w0-r0'])
        self.assertFalse(any(e['event'].startswith('ordinary_witness') for e in a.events))
        self.assertEqual(_verify_ordinary_priority(replay_events(a),CONFIG|dict(ordinary_witness_priority=False),6)['observed_ordinary_steps'],0)

    def test_full_summary_joins_ordinary_actual_work_and_rejects_missing_model_receipt(self):
        from prc_nc.analysis import summarize
        a=ordinary_fixture();out=a.schedule();complete_model_observer(a,out)
        cls=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CertificateScheduler')
        init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        free=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name=='traced_free')
        def original_free(request):
            for block in a._pm.req_to_blocks.pop(request.request_id):block.ref_cnt-=1
        env=dict(namespace,self=a,original_free=original_free,terminal_flags=terminal_flags)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[free],type_ignores=[])),str(SOURCE),'exec'),env)
        a._release_epoch=0;a._deny_cache={};a._flush_events=lambda:None
        for request in a.requests.values():request.status='FINISHED_STOPPED';env['traced_free'](request)
        events=replay_events(a)
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);config=validate_config(configuration(**CONFIG)|dict(count=2,waves=1))
            manifest=dict(status='PASS',configuration=config,source_sha256={},physical_capacity_pages=6,request_caps=[32,48],completed=2,output_tokens=2,execution_measurement=True)
            (folder/'manifest.json').write_text(json.dumps(manifest))
            (folder/'execution_tokens.json').write_text(json.dumps(a._execution_meter.report()))
            records=[dict(request_id=rid,terminal='completed',tokens=[1],cap=r.max_tokens,submitted_at_s=0,offered_at_s=0) for rid,r in a.requests.items()]
            (folder/'requests.json').write_text(json.dumps(records));log=folder/'allocator_events.jsonl'
            log.write_text('\n'.join(json.dumps(e) for e in events))
            row,_,_=summarize(folder)
            self.assertEqual(row['ordinary_witness_priority_observation']['activated_workload_reorders'],1)
            events=[e for e in events if e['event']!='ordinary_witness_confirmed']
            for i,e in enumerate(events):e.update(source_event_index=i,monotonic_ns=i+1)
            log.write_text('\n'.join(json.dumps(e) for e in events))
            with self.assertRaisesRegex(AssertionError,'settlement'):summarize(folder)

    def test_receipt_forged_reorder_head_credit_budget_and_missing_model_are_rejected(self):
        a=ordinary_fixture();out=a.schedule();complete_model_observer(a,out);original=replay_events(a)
        for mutation in ('sort','activation','headcredit','work','missing','outside'):
            events=copy.deepcopy(original);frame=next(e for e in events if e['event']=='ordinary_witness_priority');result=next(e for e in events if e['event']=='ordinary_witness_confirmed')
            if mutation=='sort':frame['reordered_running'].reverse()
            if mutation=='activation':frame['relative_order_changed']=False
            if mutation=='headcredit':result['head_new_unique_positions']=0
            if mutation=='work':result['confirmed_executed_positions']['w0-r0']=0
            if mutation=='missing':events.remove(result)
            if mutation=='outside':result['scheduled_outside_initial_residents']=['ghost']
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):_verify_ordinary_priority(events,CONFIG,6)


if __name__=='__main__':unittest.main()
