"""CPU-only utility-filter safety/transaction/real-adapter fixtures, no GPU."""
from dataclasses import FrozenInstanceError
import copy
import json
import unittest
from types import SimpleNamespace as NS
from prc_nc.admission import AdmissionContext,AdmissionDeferred,decide
from prc_nc.controller import CertificateController,Refused,StaleState
from prc_nc.runtime import AllocationAuthority
from prc_nc.model import State,Group
from prc_nc.configuration import validate_config
from test_witness_preserving_batch import configuration
from test_adapter_metadata import fixture

FILTER='c0_first_shorter_cap'
IDS=('a','c','d');POSTIDS=('a','b','c','d')
BEFORE=State((1,),((1,),(3,),(3,)),(Group('prefix-a',0,2,frozenset({0})),Group('prefix-cd',0,2,frozenset({1,2}))))
POST=State((1,),((1,),(1,),(3,),(3,)),(Group('prefix-a',0,2,frozenset({0,1})),Group('prefix-cd',0,2,frozenset({2,3}))))

def context(cap=64,version=0):return AdmissionContext('b',version,cap,tuple((rid,512) for rid in IDS))


class AdmissionPureTests(unittest.TestCase):
    def test_strict_complete_c0_not_partial_or_per_candidate_and_shorter_rule(self):
        for cap,short in ((64,True),(512,False),(1024,False)):
            self.assertEqual(decide(FILTER,context(cap),False,True).policy_allows,short)
            self.assertTrue(decide(FILTER,context(cap),True,True).policy_allows)
            self.assertTrue(decide('unrestricted',context(cap),False,True).policy_allows)
        self.assertFalse(decide(FILTER,AdmissionContext('b',0,64,()),False,True).policy_allows)

    def test_deep_immutability_bool_duplicate_and_candidate_rejected(self):
        for kwargs in (dict(candidate_cap=True),dict(authority_version=True),dict(incumbent_caps=[('a',512)]),
                       dict(incumbent_caps=(('a',True),)),dict(incumbent_caps=(('a',512),('a',512))),
                       dict(incumbent_caps=(('b',512),))):
            d=dict(candidate_id='b',authority_version=0,candidate_cap=64,incumbent_caps=(('a',512),));d.update(kwargs)
            with self.assertRaises(ValueError):AdmissionContext(**d)
        c=context()
        with self.assertRaises(FrozenInstanceError):c.candidate_cap=512
        decision=decide(FILTER,c,False,True)
        with self.assertRaises(FrozenInstanceError):decision.policy_allows=False
        with self.assertRaises(ValueError):decide(FILTER,c,0,True)

    def test_configuration_and_environment_binding(self):
        config=configuration(drain_batching='cohort_certified',admission_policy=FILTER)
        self.assertEqual(validate_config(config)['admission_policy'],FILTER)
        for patch in (dict(admission_policy='typo'),dict(drain_batching='serial'),dict(enable_prc_pipeline=True)):
            with self.assertRaises(ValueError):validate_config(config|patch)
        from pathlib import Path
        text=(Path(__file__).resolve().parents[1]/'prc_nc/gpu_run.py').read_text()
        self.assertLess(text.index('"PRC_ADMISSION_POLICY": config["admission_policy"]'),text.index('from transformers import'))


class AdmissionControllerTests(unittest.TestCase):
    def test_common_arms_pay_same_decision_and_only_c1_policy_can_defer(self):
        outcomes=[]
        for variant in ('common_full_c0','common_full_c1'):
            c=CertificateController(BEFORE,certificate_variant=variant,admission_policy=FILTER)
            before=(c.state,c.version,c.order,len(c._tickets))
            try:c.prepare_state(POST,same_cohort=False,admission_context=context(512))
            except AdmissionDeferred as e:decision=e.decision;outcomes.append('policy')
            except Refused as e:decision=e.admission_decision;outcomes.append('certificate')
            self.assertFalse(decision.c0_complete);self.assertTrue(decision.c1_complete);self.assertFalse(decision.policy_allows)
            self.assertEqual((c.state,c.version,c.order,len(c._tickets)),before)
            self.assertEqual((c.stats['oracle'],c.stats['common_c0_checks'],c.stats['common_c1_checks'],c.stats['admission_checks']),(2,1,1,1))
        self.assertEqual(outcomes,['certificate','policy'])

    def test_incremental_new_admission_pays_both_full_checks_and_existing_growth_does_not_filter(self):
        c=CertificateController(BEFORE,admission_policy=FILTER)
        t=c.prepare_state(POST,same_cohort=False,admission_context=context())
        self.assertEqual((c.stats['oracle'],c.stats['admission_c0_checks'],c.stats['admission_c1_checks']),(2,1,1))
        self.assertTrue(t.admission_decision.policy_allows);c.commit(t,POST)
        old=c.stats['admission_checks'];growth=((0,),(1,),(0,),(0,))
        t=c.prepare_growth(growth);self.assertEqual(c.stats['admission_checks'],old);self.assertIsNone(t.admission_decision)
        c.abort(t)

    def test_direct_controller_cannot_omit_context_for_filtered_cohort_growth(self):
        c=CertificateController(BEFORE,certificate_variant='common_full_c1',admission_policy=FILTER)
        with self.assertRaisesRegex(ValueError,'immutable cap context'):c.prepare_state(POST,same_cohort=False)
        self.assertFalse(c._tickets);self.assertEqual(c.version,0)

    def test_constructor_release_rebuild_without_new_admission_are_not_filtered(self):
        c=CertificateController(POST,certificate_variant='common_full_c1',admission_policy=FILTER)
        t=c.prepare_state(POST,same_cohort=False)
        self.assertIsNone(t.admission_decision);self.assertEqual(c.stats['admission_checks'],0)


class AdmissionAuthorityTests(unittest.TestCase):
    def test_deferral_changes_no_state_version_ticket_grant_or_pending(self):
        a=AllocationAuthority(BEFORE,IDS,certificate_variant='common_full_c1',admission_policy=FILTER)
        before=(a.version,a.controller.version,a.confirmed,a.virtual,a.ids,a.controller.order)
        with self.assertRaises(AdmissionDeferred):a.authorize('long','b',POST,POSTIDS,expected_version=0,admission_context=context(512))
        self.assertEqual((a.version,a.controller.version,a.confirmed,a.virtual,a.ids,a.controller.order),before)
        self.assertFalse(a._grants);self.assertFalse(a.controller._tickets);self.assertIsNone(a.pending)
        with self.assertRaises(ValueError):a.authorize('omit','b',POST,POSTIDS,expected_version=0)

    def test_successful_ack_idempotent_fingerprint_binds_complete_cap_context(self):
        a=AllocationAuthority(BEFORE,IDS,certificate_variant='common_full_c1',admission_policy=FILTER)
        c=context();grant=a.authorize('brief','b',POST,POSTIDS,expected_version=0,admission_context=c)
        self.assertEqual(grant.admission_decision.context,c)
        snapshots=[(BEFORE,IDS),(POST,POSTIDS)]
        self.assertEqual(a.dispatch(grant,lambda:'physical-done',lambda:snapshots.pop(0)),'physical-done')
        self.assertEqual((a.version,a.controller.version),(2,1))
        self.assertEqual(a.authorize('brief','b',POST,POSTIDS,expected_version=0,admission_context=c),grant)
        with self.assertRaises(ValueError):a.authorize('brief','b',POST,POSTIDS,expected_version=0,admission_context=context(32))
        changed=AdmissionContext('b',0,64,(('a',256),('c',512),('d',512)))
        with self.assertRaises(ValueError):a.authorize('brief','b',POST,POSTIDS,expected_version=0,admission_context=changed)

    def test_wrong_authority_version_candidate_membership_and_multi_admission_rejected(self):
        for ctx in (context(version=1),AdmissionContext('wrong',0,64,tuple((rid,512) for rid in IDS)),AdmissionContext('b',0,64,(('a',512),))):
            a=AllocationAuthority(BEFORE,IDS,certificate_variant='common_full_c1',admission_policy=FILTER)
            with self.assertRaises(ValueError):a.authorize('bad','b',POST,POSTIDS,expected_version=0,admission_context=ctx)
            self.assertIsNone(a.pending)
        a=AllocationAuthority(BEFORE,IDS,admission_policy=FILTER)
        post=State(POST.free,POST.need+((0,),),POST.groups)
        with self.assertRaises(ValueError):a.authorize('two','b',post,POSTIDS+('e',),expected_version=0,admission_context=context())

    def test_refund_and_release_do_not_apply_new_admission_filter(self):
        a=AllocationAuthority(BEFORE,IDS,certificate_variant='common_full_c1',admission_policy=FILTER)
        grant=a.authorize('brief','b',POST,POSTIDS,expected_version=0,admission_context=context())
        checks=a.controller.stats['admission_checks'];a.stop_unexecuted(grant,lambda:(BEFORE,IDS))
        self.assertEqual(a.controller.stats['admission_checks'],checks);self.assertEqual(a.confirmed,BEFORE)




def actual_adapter_fixture(cap=64,variant='common_full_c1',policy=FILTER):
    """Real adapter projection/Authority code over CPU physical-page objects.

    K=93; incumbent a has33 private pages and zero residual page demand;
    c/d share28 pages and each need36. New request shares31 a-prefix pages,
    allocates1 private page. Full C0 then stalls at34<36; C1 releases the31
    shared pages on actual completion of both a/new. This is not GPU evidence.
    """
    from prc_nc.drain import DrainPolicy
    a=fixture(audit=True)
    blocks=[NS(block_id=i,is_null=i==0,ref_cnt=0) for i in range(94)]
    ids=('w0-r0','w0-r1','w0-r2');candidate='w0-r3'
    mapping={ids[0]:blocks[1:34],ids[1]:blocks[34:62],ids[2]:blocks[34:62]}
    for row in mapping.values():
        for block in row:block.ref_cnt+=1
    a._pm=NS(req_to_blocks=mapping,block_size=16)
    a.kv_cache_manager=NS(block_pool=NS(blocks=blocks,num_gpu_blocks=94,
        get_num_free_blocks=lambda:sum(not b.is_null and b.ref_cnt==0 for b in blocks)))
    a.requests={rid:NS(request_id=rid,num_prompt_tokens=16 if i==0 else 512,max_tokens=512,
        num_computed_tokens=527 if i==0 else 448,num_preemptions=0,output_token_ids=[])
        for i,rid in enumerate(ids)}
    request=NS(request_id=candidate,num_prompt_tokens=496,max_tokens=cap,num_computed_tokens=0,num_preemptions=0,output_token_ids=[])
    a.requests[candidate]=request
    a._drain=DrainPolicy(64,batching='cohort_certified')
    a._zero_growth_fast=a._deny_cache_enabled=a._rank_order=a._pipeline_enabled=False
    a._serial=0;a._pending_commit={};a._proposals={};a._online=None
    a.events=[]
    def log(event,**values):
        a.events.append(dict(event=event,step=1,source_event_index=len(a.events),monotonic_ns=len(a.events)+1,**values))
    a._log=log
    for rid in ids:log('causal_acknowledge',request_id=rid)
    state,residents,_=a._snapshot()
    a._authority=AllocationAuthority(state,residents,audit=True,certificate_variant=variant,
        admission_policy=policy,on_event=lambda record:log('causal_'+record['kind'],**record))
    return a,request,NS(blocks=[blocks[1:32]])


class AdmissionActualAdapterTests(unittest.TestCase):
    def test_real_projection_shorter_short_and_equal_long_with_cf_truth(self):
        from prc_nc.analysis import _verify_admission_policy
        for cap,expected in ((64,True),(512,False)):
            a,request,cached=actual_adapter_fixture(cap)
            before=(a._snapshot()[0],a._authority.version,a._authority.controller.version)
            self.assertEqual(a._prc_allow(request,1,num_new_computed_tokens=496,new_computed_blocks=cached),expected)
            receipt=next(e for e in a.events if e['event']=='admission_policy_decision')
            self.assertFalse(receipt['decision']['c0_complete']);self.assertTrue(receipt['decision']['c1_complete'])
            self.assertEqual(receipt['policy_deferred'],not expected)
            cf=next(e for e in a.events if e['event']=='study_candidate_counterfactual')
            self.assertFalse(cf['exclusive_certificate']);self.assertTrue(cf['shared_certificate'])
            proposal=a.events[-1];self.assertTrue(proposal['certificate']);self.assertEqual(proposal['allow'],expected)
            self.assertEqual(_verify_admission_policy(json.loads(json.dumps(a.events)),dict(admission_policy=FILTER,certificate_variant='common_full_c1',audit=True),[512,512,512,cap]),int(not expected))
            if expected:a._authority.stop_unexecuted(a._pending_commit.pop(request.request_id),lambda:a._snapshot()[:2])
            else:
                self.assertEqual((a._snapshot()[0],a._authority.version,a._authority.controller.version),before)
                self.assertIsNone(a._authority.pending)

    def test_both_c0_arms_report_certificate_refusal_not_policy_defer_and_pay_predicate(self):
        for policy in ('unrestricted',FILTER):
            a,request,cached=actual_adapter_fixture(512,'common_full_c0',policy)
            self.assertFalse(a._prc_allow(request,1,num_new_computed_tokens=496,new_computed_blocks=cached))
            receipt=next(e for e in a.events if e['event']=='admission_policy_decision')
            self.assertFalse(receipt['active_certificate']);self.assertFalse(receipt['policy_deferred'])
            self.assertEqual(a._authority.controller.stats['admission_checks'],1)
            self.assertIsNone(a._authority.pending)

    def test_unrestricted_c1_accepts_equal_cap_and_incremental_filtered_defers_it(self):
        for variant,policy,expected in (('common_full_c1','unrestricted',True),('shared_incremental',FILTER,False)):
            a,request,cached=actual_adapter_fixture(512,variant,policy)
            self.assertEqual(a._prc_allow(request,1,num_new_computed_tokens=496,new_computed_blocks=cached),expected)
            if expected:a._authority.stop_unexecuted(a._pending_commit.pop(request.request_id),lambda:a._snapshot()[:2])
            self.assertIsNone(a._authority.pending)

    def test_utility_defer_cannot_be_forged_or_mislabeled_in_replay(self):
        from prc_nc.analysis import _verify_admission_policy
        a,request,cached=actual_adapter_fixture(512)
        a._prc_allow(request,1,num_new_computed_tokens=496,new_computed_blocks=cached)
        config=dict(admission_policy=FILTER,certificate_variant='common_full_c1',audit=True)
        for mutation in ('cap','version','candidate','grant','certificate','unchecked','missing_cf','ordinal','predicate','cf_certificate'):
            e=json.loads(json.dumps(a.events))
            receipt=next(r for r in e if r['event']=='admission_policy_decision')
            if mutation=='cap':receipt['decision']['context']['candidate_cap']=64
            if mutation=='version':receipt['proposal_version']+=1
            if mutation=='candidate':receipt['request_id']='w0-r2'
            if mutation=='grant':receipt['grant_id']='forged'
            if mutation=='certificate':next(r for r in e if r['event']=='proposal')['certificate']=False
            if mutation=='unchecked':next(r for r in e if r['event']=='incremental_proposal')['oracle_checked']=False
            if mutation=='missing_cf':e=[r for r in e if r['event']!='study_candidate_counterfactual']
            if mutation=='ordinal':receipt['source_event_index']=0
            if mutation=='predicate':receipt['decision']['shorter_cap']=True
            if mutation=='cf_certificate':next(r for r in e if r['event']=='study_candidate_counterfactual')['shared_certificate']=False
            with self.assertRaises((AssertionError,ValueError),msg=mutation):_verify_admission_policy(e,config,[512,512,512,512])




def warmup_receipts(rid='warmup-8d123bc7'):
    """Actual adapter/Authority CPU warmup lifecycle, not a GPU service run."""
    from prc_nc.drain import DrainPolicy
    a=fixture(audit=True);blocks=[NS(block_id=i,is_null=i==0,ref_cnt=0) for i in range(3)]
    a._pm=NS(req_to_blocks={},block_size=16)
    a.kv_cache_manager=NS(block_pool=NS(blocks=blocks,num_gpu_blocks=3,
        get_num_free_blocks=lambda:sum(not b.is_null and b.ref_cnt==0 for b in blocks)))
    request=NS(request_id=rid,num_prompt_tokens=16,max_tokens=8,num_computed_tokens=0,num_preemptions=0,output_token_ids=[])
    a.requests={rid:request};a._drain=DrainPolicy(64,batching='cohort_certified')
    a._zero_growth_fast=a._deny_cache_enabled=a._rank_order=a._pipeline_enabled=False
    a._serial=0;a._pending_commit={};a._proposals={};a._online=None;a.events=[]
    def log(event,**values):a.events.append(dict(event=event,step=0,source_event_index=len(a.events),monotonic_ns=len(a.events)+1,**values))
    a._log=log;state,ids,_=a._snapshot()
    a._authority=AllocationAuthority(state,ids,audit=True,certificate_variant='common_full_c1',
        admission_policy='unrestricted',on_event=lambda r:log('causal_'+r['kind'],**r))
    assert a._prc_allow(request,16)
    grant=a._pending_commit.pop(rid)
    def allocate():blocks[1].ref_cnt=1;a._pm.req_to_blocks[rid]=[blocks[1]];return 'CPU-physical-fixture'
    a._authority.dispatch(grant,allocate,lambda:a._snapshot()[:2])
    def release():blocks[1].ref_cnt=0;del a._pm.req_to_blocks[rid]
    a._authority.release(rid,release,lambda:a._snapshot()[:2])
    return json.loads(json.dumps(a.events))


class AdmissionWarmupSchemaTests(unittest.TestCase):
    def test_reviewed_native_assignment_uses_exact_eight_lowercase_hex_chars(self):
        import ast
        from pathlib import Path
        from native_source_fixture import native_source
        native=native_source('vllm/v1/engine/input_processor.py')
        tree=ast.parse(native.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='InputProcessor')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='assign_request_id')
        method.decorator_list=[]
        for arg in method.args.args:arg.annotation=None
        method.returns=None
        env=dict(envs=NS(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False),random_uuid=lambda:'8d123bc7a4e603ff')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(native),'exec'),env)
        req=NS(request_id='warmup',external_req_id=None);env['assign_request_id'](req)
        self.assertEqual(req.request_id,'warmup-8d123bc7');self.assertEqual(req.external_req_id,'warmup')
        utility=native_source('vllm/utils/__init__.py').read_text()
        self.assertIn('uuid.uuid4().int & MASK_64_BITS:016x',utility)

    def test_real_suffix_warmup_receipts_survive_full_summarize(self):
        import tempfile
        from pathlib import Path
        from test_analysis_runner import ReplayTests
        from prc_nc.analysis import summarize
        for rid in ('warmup','warmup-8d123bc7','warmup-a3ab7b67'):
            with self.subTest(rid=rid),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);old=ReplayTests();work=old.fixture(folder)
                events=warmup_receipts(rid)+work
                for i,event in enumerate(events):event.update(source_event_index=i,monotonic_ns=i+1)
                manifest=json.loads((folder/'manifest.json').read_text());manifest['request_caps']=[2]
                (folder/'manifest.json').write_text(json.dumps(manifest));old.write_events(folder,events)
                report,_,_=summarize(folder)
                self.assertEqual(report['status'],'PASS');self.assertEqual(report['admission_policy_deferrals_verified'],0)

    def test_unknown_warmup_prefix_suffix_and_forged_cap_are_rejected(self):
        from prc_nc.analysis import _verify_admission_policy
        config=dict(admission_policy='unrestricted',certificate_variant='common_full_c1',audit=True)
        for rid in ('warmup-other','warmup-8d123bc','warmup-8d123bc77','warmup-8D123BC7',
                    'warmup-8d123bc7-extra','warmup8d123bc7','warmup-w0-r0-8d123bc7','wakeup-8d123bc7'):
            with self.subTest(rid=rid),self.assertRaisesRegex(AssertionError,'cap identity'):
                _verify_admission_policy(warmup_receipts(rid),config,[2])
        e=warmup_receipts();next(r for r in e if r['event']=='admission_policy_decision')['decision']['context']['candidate_cap']=64
        with self.assertRaises(AssertionError):_verify_admission_policy(e,config,[2])

    def test_missing_accepted_receipt_cannot_bypass_cap_truth_checks(self):
        from prc_nc.analysis import _verify_admission_policy
        e=[r for r in warmup_receipts() if r['event']!='admission_policy_decision']
        for i,event in enumerate(e):event.update(source_event_index=i,monotonic_ns=i+1)
        with self.assertRaisesRegex(AssertionError,'Unreceipted'):
            _verify_admission_policy(e,dict(admission_policy='unrestricted',certificate_variant='common_full_c1',audit=True),[2])

    def test_cf_must_precede_exact_proposal_and_match_grant_version_hash(self):
        from prc_nc.analysis import _verify_admission_policy
        config=dict(admission_policy='unrestricted',certificate_variant='common_full_c1',audit=True)
        for mutation in ('after_proposal','grant','version','hash','candidate'):
            e=warmup_receipts();cf=next(r for r in e if r['event']=='study_candidate_counterfactual')
            if mutation=='after_proposal':
                e.remove(cf);j=next(i for i,r in enumerate(e) if r['event']=='proposal');e.insert(j+1,cf)
            if mutation=='grant':cf['actual_grant_id']='ghost:grant'
            if mutation=='version':cf['proposal_version']+=1
            if mutation=='hash':cf['projected_post_sha256']='0'*64
            if mutation=='candidate':cf['request_id']='warmup-00000000'
            for i,event in enumerate(e):event.update(source_event_index=i,monotonic_ns=i+1)
            with self.subTest(mutation=mutation),self.assertRaises(AssertionError):_verify_admission_policy(e,config,[2])


if __name__=='__main__':unittest.main()
