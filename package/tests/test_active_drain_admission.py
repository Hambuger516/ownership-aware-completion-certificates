"""CPU-only contingent active joins; actual vendor waiting + Authority bodies."""
import ast,copy,hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from prc_nc.active_admission import ActiveAdmission,finite_universe,SEMANTICS
from prc_nc.active_admission_analysis import verify_active_admission
from prc_nc.configuration import validate_config
from prc_nc.drain import DrainPolicy,terminal_flags
from prc_nc.execution_meter import ExecutionTokenMeter
from test_ordinary_witness_priority import ordinary_fixture,install_actual_vendor_waiting_prefix
from test_witness_preserving_batch import namespace,SOURCE,configuration
from test_head_funded_segments import complete_model_observer,replay_events


def active_fixture(variant='common_full_c1',*,head_zero_charge=False):
    a=ordinary_fixture(variant,budget=4,extra_free=6)
    r=install_actual_vendor_waiting_prefix(a)
    a._ordinary_witness_priority=False;a._active_drain_admission=True
    a._drain=DrainPolicy(1,scope='cohort',batching='cohort_certified',active_admission=True)
    if head_zero_charge:
        a.requests['w0-r0'].num_computed_tokens=31;a.requests['w0-r0'].num_tokens_with_spec=32
    prompts=[[100+i]*(a.requests[f'w0-r{i}'].num_prompt_tokens if f'w0-r{i}' in a.requests else 1) for i in range(16)]
    caps=[a.requests[f'w0-r{i}'].max_tokens if f'w0-r{i}' in a.requests else 16 for i in range(16)]
    doc=finite_universe(prompts,caps,block_size=16,capacity=12)
    h=hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    a._active_join=ActiveAdmission(doc,sha256=h)
    namespace['ACTIVE_ADMISSION_SEMANTICS']=SEMANTICS
    for i in range(3):a.requests[f'w0-r{i}'].prompt_token_ids=prompts[i]
    for rid in a._authority.ids:
        req=a.requests[rid];key=a._active_join.bind(req);row=a._active_join.rows[key]
        a._log('finite_input_binding',request_id=rid,external_request_id=key,universe_sha256=h,
            **{k:row[k] for k in ('prompt_token_sha256','prompt_tokens','max_tokens')})
    # Install actual free callback as well as the actual allocate body already
    # used by vendor fixture. Physical backend only is a CPU double.
    cls=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CertificateScheduler')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    free=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name=='traced_free')
    def original_free(request):
        for block in a._pm.req_to_blocks.pop(request.request_id,[]):block.ref_cnt-=1
    env=dict(namespace,self=a,original_free=original_free,terminal_flags=terminal_flags)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[free],type_ignores=[])),str(SOURCE),'exec'),env)
    a._cpu_actual_free=env['traced_free'];a._release_epoch=0;a._deny_cache={}
    a._flush_events=lambda:None
    return a,dict(active_drain_admission=True,drain_scope='cohort',drain_batching='cohort_certified',
        drain_policy='timer',admission_policy='c0_first_shorter_cap',certificate_variant=variant,audit=True,
        _active_input_universe=doc,_active_input_universe_sha=h)


def finish_all(a):
    for rid in list(a._authority.ids):
        a.requests[rid].status='FINISHED_STOPPED';a._cpu_actual_free(a.requests[rid])


class ActiveAdmissionTests(unittest.TestCase):
    def setUp(self):
        old=namespace.get('_instrumented_schedule');self.addCleanup(lambda:namespace.__setitem__('_instrumented_schedule',old))

    def test_configuration_default_off_restricted_single_factor_and_boolean(self):
        self.assertFalse(validate_config({})['active_drain_admission'])
        base=configuration(active_drain_admission=True,count=16,waves=1,drain_batching='cohort_certified',admission_policy='c0_first_shorter_cap')
        self.assertTrue(validate_config(base)['active_drain_admission'])
        for change in (dict(active_drain_admission=1),dict(count=15),dict(waves=2),dict(drain_scope='first_completion'),dict(drain_scope='head',drain_batching='head_funded'),dict(drain_policy='none'),dict(rank_order=True),dict(admission_policy='unrestricted'),dict(enable_prc_pipeline=True)):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_config(base|change)

    def test_actual_vendor_waiting_NEW_ACK_joins_and_actual_model_confirms(self):
        a,c=active_fixture();out=a.schedule()
        self.assertEqual(out.num_scheduled_tokens,{'w0-r0':1,'w0-r1':1,'w0-r2':1})
        join=next(e for e in a.events if e['event']=='active_drain_join')
        self.assertEqual(join['before_members'],['w0-r0','w0-r1']);self.assertEqual(join['after_members'],['w0-r0','w0-r1','w0-r2'])
        self.assertEqual(set(a._authority.ids),set(a._drain.order));self.assertIsNone(a._authority.pending)
        with self.assertRaises(RuntimeError):a.schedule()
        complete_model_observer(a,out);self.assertIsNone(a._active_join.epoch)
        finish_all(a)
        result=verify_active_admission(replay_events(a),c,12)
        self.assertEqual((result['joins'],result['epochs'],result['exact_work_evidence']),(1,1,True))

    def test_head_positive_token_zero_page_charge_does_not_block_NEW(self):
        a,c=active_fixture(head_zero_charge=True);out=a.schedule()
        join=next(e for e in a.events if e['event']=='active_drain_join')
        head=next(e for e in a.events if e['event']=='causal_authorize' and e['grant_id']==join['head_grant_id'])
        self.assertEqual(head['authorized_pages'],(0,));self.assertEqual(join['head_allocated_tokens'],1)
        self.assertIn('w0-r2',out.num_scheduled_tokens)
        self.assertFalse(any(e['event']=='drain_head_funding_check' for e in a.events))

    def test_common_full_overlap_same_head_NEW_guard_and_dual_cost(self):
        observed=[]
        for variant in ('common_full_c0','common_full_c1'):
            a,c=active_fixture(variant);before=a._authority.controller.stats['oracle'];out=a.schedule()
            observed.append((out.num_scheduled_tokens,a._authority.controller.stats['oracle']-before,
                [(e['head'],e['before_members'],e['after_members']) for e in a.events if e['event']=='active_drain_join']))
        self.assertEqual(observed[0],observed[1])

    def test_current_head_zero_or_forged_stale_accumulator_never_allows_NEW(self):
        a,c=active_fixture();ids=list(a._authority.ids);head=a._authority.drain_cut()[0]
        a._drain.observe(ids,tuple(ids.index(r) for r in a._authority.drain_cut()))
        a._active_join.begin(episode=1,batch=1,head=head,members=list(a._drain.order),version=a._authority.version)
        candidate=a.requests['w0-r2']
        for fake in (None,{}, {head:0},{head:1}):self.assertFalse(a._prc_allow(candidate,1,scheduled_tokens=fake))
        self.assertIsNone(a._authority.pending);self.assertNotIn(candidate.request_id,a._drain.order)

    def test_finite_identity_token_cap_and_terminal_replay_fail(self):
        a,c=active_fixture();req=a.requests['w0-r2'];a._active_join.bind(req)
        forged=copy.copy(req);forged.request_id='w0-r2-a1b2c3d4'
        with self.assertRaises(RuntimeError):a._active_join.bind(forged)
        forged=copy.copy(req);forged.max_tokens+=1
        with self.assertRaises(RuntimeError):a._active_join.bind(forged)
        forged=copy.copy(req);forged.request_id='w0-r99-a1b2c3d4'
        with self.assertRaises(RuntimeError):a._active_join.bind(forged)
        a._active_join.retired.add('w0-r2')
        with self.assertRaises(RuntimeError):a._active_join.bind(req)

    def test_join_receipt_missing_ACK_ghost_grant_old_head_or_set_forgery_rejected(self):
        a,c=active_fixture();out=a.schedule();complete_model_observer(a,out);finish_all(a);original=replay_events(a)
        for mutation in ('ack','grant','version','members','head','post','model','headzero'):
            events=copy.deepcopy(original);join=next(e for e in events if e['event']=='active_drain_join')
            if mutation=='ack':events=[e for e in events if not(e['event']=='causal_acknowledge' and e.get('grant_id')==join['grant_id'])]
            if mutation=='grant':join['grant_id']='ghost'
            if mutation=='version':join['acked_version']+=1
            if mutation=='members':join['after_members'].append('outside')
            if mutation=='head':join['head']='w0-r1'
            if mutation=='post':join['physical_post_sha256']='0'*64
            if mutation=='model':events=[e for e in events if e['event']!='drain_batch_confirmed']
            if mutation=='headzero':join['head_allocated_tokens']=0
            for i,e in enumerate(events):e.update(source_event_index=i,monotonic_ns=i+1)
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):verify_active_admission(events,c,12)

    def test_all_pending_maps_and_meter_block_before_observe(self):
        for key in ('_pending_commit','_pending_projection','_pending_index','_execution_batches','_drain_pending_batches'):
            a,c=active_fixture();setattr(a,key,{'unsettled':object()})
            with self.subTest(key=key),self.assertRaises(RuntimeError):a.schedule()
        a,c=active_fixture();a._execution_meter.prepare(17,[('w0-r0',0,1)])
        with self.assertRaises(RuntimeError):a.schedule()

    def test_cancelled_unobserved_meter_settlement_never_claims_zero_work(self):
        m=ExecutionTokenMeter();m.prepare(1,[('head',0,1),('new',4,2)])
        with self.assertRaises(ValueError):m.settle_cancelled_unobserved(1,['new'],[])
        self.assertIn(1,m.pending)
        unknown=m.settle_cancelled_unobserved(1,['new'],['head'])
        self.assertEqual(unknown[0]['actual_work'],'UNKNOWN');self.assertNotIn('head',m.executed)
        self.assertEqual(m.executed['new'],2);self.assertFalse(m.pending)
        report=m.report(request_ids=['head','new']);self.assertFalse(report['exact_work_evidence'])
        self.assertEqual(report['unobserved_cancelled_spans'][0]['prepared_spans'],[('head',0,1),('new',4,2)])
        with self.assertRaises(ValueError):m.settle_cancelled_unobserved(1,['new'],['head'])
        legacy=ExecutionTokenMeter();legacy.prepare(1,[('a',0,1)])
        with self.assertRaises(ValueError):legacy.confirm(1,[])
        legacy.confirm(1,['a']);self.assertNotIn('exact_work_evidence',legacy.report())


if __name__=='__main__':unittest.main()


def actual_observer_with_terminals(a,out,terminals,returned):
    """Actual outer + measurement observer methods, CPU physical model double."""
    adapter=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='CertificateScheduler')
    outer=next(n for n in adapter.body if isinstance(n,ast.FunctionDef) and n.name=='update_from_output')
    measurement=SOURCE.with_name('execution_measurement.py')
    measured_class=next(n for n in ast.parse(measurement.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ExecutionMeasurementMixin')
    measured=next(n for n in measured_class.body if isinstance(n,ast.FunctionDef) and n.name=='update_from_output')
    class Native:
        def update_from_output(self,scheduled,result):
            for rid,status in terminals:
                self.requests[rid].status=status;self._cpu_actual_free(self.requests[rid])
                self.running[:]=[r for r in self.running if r.request_id!=rid]
            return 'actual-native-CPU-backend-return'
    m=ast.ClassDef(name='Measured',bases=[ast.Name(id='Native',ctx=ast.Load())],keywords=[],body=[measured],decorator_list=[])
    o=ast.ClassDef(name='Observed',bases=[ast.Name(id='Measured',ctx=ast.Load())],keywords=[],body=[outer],decorator_list=[])
    env=dict(Native=Native,time=namespace['time'])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[m,o],type_ignores=[])),str(SOURCE),'exec'),env)
    observer=env['Observed']();observer.__dict__=a.__dict__
    batch=max(a._execution_meter.done,default=-1)+1
    a._execution_meter.prepare(batch,[(rid,a.requests[rid].num_computed_tokens,n) for rid,n in out.num_scheduled_tokens.items()])
    a._execution_batches={id(out):(out,batch)};a._execution_update_ns=a._execution_update_calls=0
    a.has_requests=lambda:bool(a._authority.ids);a._flush_execution_meter=lambda:None
    return observer.update_from_output(out,NS(req_ids=returned))


class ActiveAdmissionBoundaryTests(unittest.TestCase):
    setUp=ActiveAdmissionTests.setUp
    def test_second_NEW_uses_current_ACKed_incumbents_and_version(self):
        a,c=active_fixture();r=copy.copy(a.requests['w0-r2']);r.request_id='w0-r3';r.prompt_token_ids=[103]
        a.requests[r.request_id]=r;a.waiting.rows.append(r)
        out=a.schedule();joins=[e for e in a.events if e['event']=='active_drain_join']
        self.assertEqual(len(joins),2);self.assertEqual(joins[1]['proposal_version'],joins[0]['acked_version'])
        self.assertIn('w0-r2',joins[1]['before_members'])
        decision=next(e['decision'] for e in a.events if e['event']=='admission_policy_decision' and e['request_id']=='w0-r3')
        self.assertEqual(set(rid for rid,cap in decision['context']['incumbent_caps']),{'w0-r0','w0-r1','w0-r2'})
        complete_model_observer(a,out);finish_all(a)
        self.assertEqual(verify_active_admission(replay_events(a),c,12)['joins'],2)

    def test_refunded_NEW_never_joins_and_does_not_preempt(self):
        a,c=active_fixture();wrapper=a.kv_cache_manager.allocate_slots
        original=wrapper.__globals__['original_allocate']
        wrapper.__globals__['original_allocate']=lambda request,*args,**kw:None if request.request_id=='w0-r2' else original(request,*args,**kw)
        with self.assertRaisesRegex(RuntimeError,'preemption is not permitted'):a.schedule()
        self.assertFalse(any(e['event']=='active_drain_join' for e in a.events))
        self.assertNotIn('w0-r2',a._authority.ids);self.assertNotIn('w0-r2',a._drain.order)
        self.assertTrue(any(e['event']=='causal_refund' and e['request_id']=='w0-r2' for e in a.events))

    def test_captured_head_cancel_then_follower_completes_is_not_head_success(self):
        a,c=active_fixture();out=a.schedule()
        actual_observer_with_terminals(a,out,[('w0-r0','FINISHED_ABORTED'),('w0-r2','FINISHED_STOPPED')],list(out.num_scheduled_tokens))
        self.assertEqual((a._drain.head_completions,a._drain.member_completions),(0,1))
        self.assertEqual([e['request_id'] for e in a.events if e['event']=='drain_head_cancelled'],['w0-r0'])
        self.assertIsNone(a._active_join.epoch)
        finish_all(a);self.assertTrue(verify_active_admission(replay_events(a),c,12)['exact_work_evidence'])

    def test_cancelled_missing_head_is_UNKNOWN_legal_control_and_fails_PASS_summary(self):
        from prc_nc.analysis import summarize
        a,c=active_fixture();out=a.schedule()
        returned=[rid for rid in out.num_scheduled_tokens if rid!='w0-r0']
        actual_observer_with_terminals(a,out,[('w0-r0','FINISHED_ABORTED')],returned)
        result=next(e for e in a.events if e['event']=='drain_batch_confirmed')
        self.assertIsNone(result['confirmed_real_positions']['w0-r0']);self.assertIsNone(result['confirmed_new_unique_positions'])
        self.assertEqual(result['actual_work_status'],'UNKNOWN_CANCELLED_SPAN');self.assertFalse(result['progress_credited'])
        self.assertIsNone(a._active_join.epoch);self.assertFalse(a._execution_batches);self.assertFalse(a._execution_meter.pending)
        finish_all(a);self.assertFalse(verify_active_admission(replay_events(a),c,12)['exact_work_evidence'])
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);doc=c['_active_input_universe'];raw=json.dumps(doc,sort_keys=True,separators=(',',':')).encode()
            (folder/'finite_request_universe.json').write_bytes(raw)
            configuration={k:v for k,v in c.items() if not k.startswith('_')}
            manifest=dict(status='PASS',configuration=configuration,physical_capacity_pages=12,source_sha256={},
                active_admission_contract=dict(universe_sha256=hashlib.sha256(raw).hexdigest()))
            (folder/'manifest.json').write_text(json.dumps(manifest));(folder/'allocator_events.jsonl').write_text('\n'.join(json.dumps(e) for e in replay_events(a)))
            with self.assertRaisesRegex(AssertionError,'UNKNOWN cancelled spans'):summarize(folder)

    def test_all_cancelled_missing_spans_settle_without_success_or_fake_execution(self):
        a,c=active_fixture();out=a.schedule()
        actual_observer_with_terminals(a,out,[(rid,'FINISHED_ABORTED') for rid in out.num_scheduled_tokens],[])
        self.assertFalse(a._drain.active);self.assertEqual((a._drain.head_completions,a._drain.member_completions),(0,0))
        self.assertEqual(a._execution_meter.executed,{})
        self.assertFalse(a._execution_meter.pending);self.assertFalse(a._drain_pending_batches)
        self.assertIsNone(a._active_join.epoch)
        self.assertFalse(verify_active_admission(replay_events(a),c,12)['exact_work_evidence'])

    def test_actual_measurement_observer_still_refuses_missing_noncancelled_member(self):
        a,c=active_fixture();out=a.schedule()
        with self.assertRaises(ValueError):actual_observer_with_terminals(a,out,[],['w0-r0','w0-r1'])
        self.assertIsNotNone(a._active_join.epoch)


class ActiveAdmissionFinalControls(unittest.TestCase):
    setUp=ActiveAdmissionTests.setUp
    def test_zero_charge_NEW_structural_ACK_accounting_not_native_execution_claim(self):
        from prc_nc.admission import AdmissionContext
        from prc_nc.model import State,Group
        a,c=active_fixture(head_zero_charge=True);a.waiting.rows=[];out=a.schedule()
        # Pure Authority ownership attachment control. It proves accounting has
        # no positive-page gate; it does NOT claim the vendor scheduler emitted
        # this NEW or that a GPU executed it. No model-work receipt is credited.
        before,ids,_=a._snapshot();new='w0-r2';post_ids=tuple(ids)+(new,);owner=0
        groups=list(before.groups);g=groups[0]
        groups[0]=Group(g.key,g.pool,g.count,g.owners|{len(ids)},g.pinned)
        post=State(before.free,before.need+((1,),),tuple(groups))
        version=a._authority.version
        context=AdmissionContext(new,version,a.requests[new].max_tokens,tuple(sorted((rid,a.requests[rid].max_tokens) for rid in ids)))
        grant=a._authority.authorize('zero-charge-attachment',new,post,post_ids,expected_version=version,admission_context=context)
        physical=[before,tuple(ids)]
        def allocate():physical[:]=[post,post_ids];return NS(success=True)
        a._authority.dispatch(grant,allocate,lambda:tuple(physical))
        joined=a._active_join.allocation_acked(a.requests[new],grant,ids,post_ids,a._authority,tokens=1,charge=(0,))
        self.assertEqual(joined['total_physical_charge'],[0]);self.assertEqual(grant.pages,(0,))
        self.assertEqual(joined['after_members'][-1],new)

    def test_head_ACK_omission_duplicate_join_transaction_and_unknown_binding_reject(self):
        a,c=active_fixture();out=a.schedule();complete_model_observer(a,out);finish_all(a);original=replay_events(a)
        for mutation in ('head_ack','duplicate_join','duplicate_execute','universe','binding','CF','early_join'):
            events=copy.deepcopy(original);join=next(e for e in events if e['event']=='active_drain_join')
            if mutation=='head_ack':events=[e for e in events if not(e['event']=='causal_acknowledge' and e.get('grant_id')==join['head_grant_id'])]
            if mutation=='duplicate_join':events.insert(events.index(join)+1,copy.deepcopy(join))
            if mutation=='duplicate_execute':
                event=next(e for e in events if e['event']=='causal_execute' and e.get('grant_id')==join['grant_id']);events.insert(events.index(event)+1,copy.deepcopy(event))
            if mutation=='universe':join['universe_sha256']='0'*64
            if mutation=='binding':next(e for e in events if e['event']=='finite_input_binding' and e['request_id']==join['request_id'])['max_tokens']+=1
            if mutation=='CF':next(e for e in events if e['event']=='study_candidate_counterfactual' and e.get('actual_grant_id')==join['grant_id'])['projected_post_sha256']='f'*64
            if mutation=='early_join':events.remove(join);events.insert(next(i for i,e in enumerate(events) if e['event']=='causal_execute' and e.get('grant_id')==join['grant_id']),join)
            for i,e in enumerate(events):e.update(source_event_index=i,monotonic_ns=i+1)
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):verify_active_admission(events,c,12)

    def test_new_cancelled_span_report_marks_exact_total_unknown(self):
        m=ExecutionTokenMeter();m.prepare(1,[('w0-r0',0,1)]);m.settle_cancelled_unobserved(1,[],['w0-r0'])
        report=m.report();self.assertIsNone(report['executed_tokens']);self.assertIsNone(report['reexecuted_tokens'])
        self.assertEqual(report['observed_executed_tokens'],0);self.assertEqual(report['actual_work_status'],'UNKNOWN_CANCELLED_SPAN')
        self.assertNotIn('w0-r0',report['requests'])
