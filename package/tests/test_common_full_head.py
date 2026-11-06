"""CPU correctness/progress checks; fixtures are not GPU performance evidence."""
import ast
from collections import Counter
import enum
import itertools
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from prc_nc.algorithms import indexed_closure, linear_margins, LinearBuildMarginIndex
from prc_nc.configuration import validate_config
from prc_nc.controller import CertificateController, Refused, StaleState
from prc_nc.drain import DrainPolicy, terminal_flags
from prc_nc.model import State, Group, canonical, brute_feasible, valid_witness, apply_private_growth, finish_request
from prc_nc.runtime import AllocationAuthority

VARIANTS=("common_full_c0", "common_full_c1")


def c0_view(s):
    return State(s.free, s.need, tuple(Group(g.key,g.pool,g.count,g.owners,g.pinned or len(g.owners)>1) for g in s.groups))


def config(**changes):
    return dict(method="shared_incremental",certificate_variant="common_full_c0",enable_prc_pipeline=False,
                rank_order=False,zero_growth_fast=False,deny_cache=False,drain_policy="timer",**changes)


def overlapping_state():
    # The two full oracles return DIFFERENT valid orders in this two-pool case.
    return State((0,0),((0,0),(0,0),(1,0),(0,1)),(
        Group("shared",0,1,frozenset({0,1})),
        Group("private1",1,1,frozenset({1})),
        Group("private3",0,1,frozenset({3}))))


class CommonFullTests(unittest.TestCase):
    def test_config_flags_are_explicit_and_old_default_is_cohort(self):
        self.assertEqual(validate_config(config())["drain_scope"],"cohort")
        for variant,scope in itertools.product(VARIANTS,("cohort","head")):
            c=config();c.update(certificate_variant=variant,drain_scope=scope)
            self.assertEqual(validate_config(c)["drain_scope"],scope)
        for key,value in (("certificate_variant","common_full_typo"),("drain_scope","partial"),("drain_scope",True)):
            c=config();c[key]=value
            with self.assertRaises(ValueError):validate_config(c)
        c=config(drain_scope="head");c["drain_policy"]="none"
        with self.assertRaises(ValueError):validate_config(c)

    def test_overlap_uses_common_c0_order_even_when_c1_oracle_order_differs(self):
        s=overlapping_state()
        self.assertNotEqual(indexed_closure(c0_view(s)),indexed_closure(s))
        controllers=[CertificateController(s,audit=False,certificate_variant=v) for v in VARIANTS]
        for c in controllers:
            self.assertEqual(c.order,indexed_closure(c0_view(s)))
            self.assertTrue(valid_witness(s,c.order))
            self.assertEqual(c.stats["initial_oracle"],2)
        self.assertEqual(controllers[0].index.table(),controllers[1].index.table())

    def test_every_prepare_pays_both_searches_and_same_fresh_index(self):
        s=overlapping_state()
        for variant in VARIANTS:
            c=CertificateController(s,audit=False,certificate_variant=variant)
            old_index=c.index
            with patch("prc_nc.controller.indexed_closure",wraps=indexed_closure) as oracle, \
                 patch("prc_nc.controller.LinearBuildMarginIndex",wraps=LinearBuildMarginIndex) as build, \
                 patch("prc_nc.controller.interval_trial",side_effect=AssertionError("cached acceptance")), \
                 patch("prc_nc.controller.event_index",side_effect=AssertionError("event acceptance")):
                t=c.prepare_growth(((0,0),)*4)
                self.assertEqual(oracle.call_count,2)
                self.assertEqual(build.call_count,1)
                self.assertEqual(oracle.call_args_list[0].args[0],c0_view(s))
                self.assertEqual(oracle.call_args_list[1].args[0],s)
                c.commit(t,c.candidate(t))
                self.assertIsNot(c.index,old_index)
                t=c.prepare_state(c.state)
                self.assertEqual(oracle.call_count,4)
                self.assertEqual(build.call_count,2)
                c.commit(t,c.candidate(t))
            self.assertEqual(c.stats["oracle"],4)
            self.assertEqual(c.stats["common_full_checks"],2)
            self.assertEqual(c.stats["common_c0_checks"],2)
            self.assertEqual(c.stats["common_c1_checks"],2)
            self.assertEqual(c.stats["fast"],0)
            self.assertEqual(c.stats["rebuild"],2)

    def test_old_cached_vs_fresh_head_counterexample_is_aligned(self):
        s=State((2,),((2,),(1,)),(Group("a",0,1,frozenset({0})),Group("b",0,1,frozenset({1}))))
        orders=[]
        for variant in VARIANTS:
            c=CertificateController(s,audit=True,certificate_variant=variant)
            self.assertEqual(c.order,(1,0))
            t=c.prepare_growth(((1,),(0,)));c.commit(t,c.candidate(t));orders.append(c.order)
            self.assertEqual(c.index.table(),linear_margins(c.state,c.order))
        self.assertEqual(orders,[(0,1),(0,1)])

    def test_true_information_divergence_changes_acceptance_only(self):
        before=State((1,),(),())
        post=State((0,),((0,),(0,),(1,)),(Group("g",0,1,frozenset({0,1})),))
        c0=CertificateController(before,audit=False,certificate_variant=VARIANTS[0])
        c1=CertificateController(before,audit=False,certificate_variant=VARIANTS[1])
        with self.assertRaises(Refused):c0.prepare_state(post,same_cohort=False)
        self.assertEqual(canonical(c0.state),canonical(before));self.assertEqual(c0.version,0)
        self.assertFalse(c0._tickets)
        t=c1.prepare_state(post,same_cohort=False)
        self.assertEqual(c1.state,before);c1.commit(t,c1.candidate(t))
        self.assertTrue(valid_witness(post,c1.order));self.assertEqual(c1.state,post)
        self.assertEqual(c0.stats["oracle"],c1.stats["oracle"])
        self.assertEqual(c0.stats["common_c1_only_checks"],1)
        self.assertEqual(c1.stats["common_c1_only_checks"],1)
        self.assertFalse(post.groups[0].pinned)

    def test_small_information_views_match_permutation_oracle(self):
        checked=0
        for owners,pinned,free,needs in itertools.product((frozenset({0}),frozenset({0,1}),frozenset({0,1,2})),(False,True),range(2),itertools.product(range(3),repeat=3)):
            s=State((free,),tuple((n,) for n in needs),(Group("g",0,1,owners,pinned),))
            expected0,expected1=brute_feasible(c0_view(s)),brute_feasible(s)
            for variant,expected in zip(VARIANTS,(expected0,expected1)):
                try:c=CertificateController(s,audit=False,certificate_variant=variant)
                except Refused:accepted=False
                else:
                    accepted=True;self.assertTrue(valid_witness(s,c.order));self.assertEqual(canonical(c.state),canonical(s))
                    if expected0:self.assertEqual(c.order,indexed_closure(c0_view(s)))
                self.assertEqual(accepted,expected);checked+=1
        self.assertEqual(checked,648)

    def test_stale_request_rejected_before_additional_work(self):
        c=CertificateController(State((2,),((1,),),()),audit=False,certificate_variant="common_full_c1")
        with self.assertRaises(StaleState):c.prepare_growth(((1,),),expected_version=8)
        self.assertEqual(c.stats["oracle"],0);self.assertFalse(c._tickets)


class Status(enum.IntEnum):
    RUNNING=1
    FINISHED_STOPPED=4
    FINISHED_LENGTH_CAPPED=5
    FINISHED_ABORTED=6
    FINISHED_IGNORED=7
    FINISHED_CANCELLED=8


class HeadDrainTests(unittest.TestCase):
    def test_intenum_terminal_names_not_numeric_str(self):
        self.assertEqual(str(Status.FINISHED_STOPPED),"4")
        for status in (Status.FINISHED_STOPPED,Status.FINISHED_LENGTH_CAPPED,"RequestStatus.FINISHED_STOPPED"):
            self.assertEqual(terminal_flags(status),(False,True))
        for status in (Status.FINISHED_ABORTED,Status.FINISHED_CANCELLED,"FINISHED_CANCELLED"):
            self.assertEqual(terminal_flags(status),(True,False))
        for status in (Status.RUNNING,Status.FINISHED_IGNORED,4,"4","UNKNOWN","NOT_ABORTED"):
            self.assertEqual(terminal_flags(status),(False,False))

    def test_default_drains_entire_cohort(self):
        d=DrainPolicy(1);d.observe(("a","b"),(0,1));d.finished("a")
        self.assertTrue(d.active);self.assertTrue(d.allows("b"));self.assertFalse(d.allows("waiting"))
        d.finished("b");self.assertFalse(d.active);self.assertEqual(d.completed_segments,1)

    def test_head_segment_completes_once_then_resets_timer(self):
        d=DrainPolicy(2,scope="head")
        self.assertFalse(d.observe(("a","b"),(0,1)))
        self.assertTrue(d.observe(("a","b"),(0,1)))
        for _ in range(5):self.assertFalse(d.observe(("a","b"),(0,1)))
        self.assertFalse(d.allows("b"))
        event=d.finished("a",completed=True)
        self.assertEqual(event["successful_head_completions"],1)
        self.assertEqual(event["remaining_frozen_requests"],["b"])
        self.assertFalse(d.active);self.assertEqual(d.elapsed,0);self.assertTrue(d.allows("b"))
        self.assertFalse(d.observe(("b",),(0,)))
        self.assertTrue(d.observe(("b",),(0,)))

    def test_cancelled_head_does_not_end_successfully(self):
        d=DrainPolicy(1,scope="head");d.observe(("a","b"),(0,1))
        self.assertIsNone(d.finished("a",cancelled=True,completed=False))
        self.assertTrue(d.active);self.assertEqual(d.order,("b",));self.assertEqual(d.head_completions,0)
        self.assertEqual(d.completed_segments,0);self.assertEqual(d.cancelled_members,1)
        self.assertFalse(d.allows("waiting"))
        d.finished("b",completed=True);self.assertFalse(d.active);self.assertEqual(d.completed_segments,1)

    def test_all_cancellations_are_not_successful_service(self):
        d=DrainPolicy(1,scope="head");d.observe(("a","b"),(0,1))
        d.finished("a",cancelled=True);event=d.finished("b",cancelled=True)
        self.assertFalse(d.active);self.assertEqual(d.head_completions,0);self.assertEqual(d.completed_segments,0)
        self.assertEqual(event["reason"],"cohort_empty_without_head_completion")

    def test_nonhead_terminal_and_missing_callback_cannot_abort_head(self):
        d=DrainPolicy(1,scope="head");d.observe(("a","b","c"),(0,1,2))
        d.finished("c",completed=True);self.assertTrue(d.active);self.assertEqual(d.order,("a","b"))
        with self.assertRaises(RuntimeError):d.observe(("b",),(0,))
        with self.assertRaises(RuntimeError):d.finished("a",completed=False,cancelled=False)
        self.assertTrue(d.active);self.assertEqual(d.head_completions,0)

    def test_finite_cohort_common_arms_complete_same_heads_with_real_authority(self):
        sequences=[]
        for variant in VARIANTS:
            initial=overlapping_state();ids=tuple("r"+str(i) for i in range(initial.n));live=[initial,ids]
            a=AllocationAuthority(initial,ids,audit=True,certificate_variant=variant)
            d=DrainPolicy(2,scope="head");done=[]
            while a.ids:
                self.assertFalse(d.observe(a.ids,tuple(a.ids.index(r) for r in a.drain_cut())))
                self.assertTrue(d.observe(a.ids,tuple(a.ids.index(r) for r in a.drain_cut())))
                head=d.order[0];rank=a.ids.index(head)
                growth=tuple(a.confirmed.need[i] if i==rank else (0,)*a.confirmed.p for i in range(a.confirmed.n))
                post=apply_private_growth(a.confirmed,growth)
                grant=a.authorize("grow-"+head,head,post,a.ids,expected_version=a.version,private_growth=growth)
                def allocate():live[:]=[post,a.ids];return (1,)
                a.dispatch(grant,allocate,lambda:tuple(live))
                after,_=finish_request(a.confirmed,rank);remaining=tuple(r for r in a.ids if r!=head)
                def release():live[:]=[after,remaining]
                a.release(head,release,lambda:tuple(live));event=d.finished(head,completed=True)
                self.assertEqual(event["successful_head_completions"],1)
                self.assertFalse(d.active);done.append(head)
            self.assertEqual(len(done),4);self.assertEqual(d.completed_segments,4);sequences.append(done)
        self.assertEqual(sequences[0],sequences[1])

    def test_adapter_real_free_body_propagates_intenum_reason(self):
        source=Path(__file__).resolve().parents[1]/"prc_nc/adapters/vllm_scheduler.py"
        tree=ast.parse(source.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="CertificateScheduler")
        init=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="__init__")
        function=next(x for x in init.body if isinstance(x,ast.FunctionDef) and x.name=="traced_free")
        for status,expected_active in ((Status.FINISHED_STOPPED,False),(Status.FINISHED_LENGTH_CAPPED,False),(Status.FINISHED_ABORTED,True)):
            blocks={"a":NS(ref_cnt=1,is_null=False),"b":NS(ref_cnt=1,is_null=False)};mapping={r:[b] for r,b in blocks.items()}
            pool=NS(get_num_free_blocks=lambda:sum(b.ref_cnt==0 for b in blocks.values()))
            events=[];cancel_flags=[]
            a=NS(_pm=NS(req_to_blocks=mapping),kv_cache_manager=NS(block_pool=pool),_drain_enabled=True,
                 _drain=DrainPolicy(1,scope="head"),_audit=True,_mode="shared_incremental",_release_epoch=0,
                 _deny_cache={},_pending_projection={},_pending_index={},_capture_request=lambda rid:None,
                 _log=lambda event,**kw:events.append(dict(event=event,**kw)),_flush_events=lambda:None)
            def snapshot():
                ids=tuple(sorted(mapping));return State((pool.get_num_free_blocks(),),tuple((0,) for _ in ids),tuple(Group(r,0,1,frozenset({i})) for i,r in enumerate(ids))),ids,{}
            a._snapshot=snapshot
            def original_free(request):
                mapping.pop(request.request_id);blocks[request.request_id].ref_cnt=0
            def release(rid,operation,observe,*,cancelled):cancel_flags.append(cancelled);operation();observe()
            a._authority=NS(release=release);a._drain.observe(("a","b"),(0,1))
            ns=dict(self=a,original_free=original_free,terminal_flags=terminal_flags,indexed_closure=indexed_closure,State=State,Group=Group)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(source),"exec"),ns)
            ns["traced_free"](NS(request_id="a",status=status))
            self.assertEqual(a._drain.active,expected_active)
            self.assertEqual(cancel_flags,[status==Status.FINISHED_ABORTED])
            self.assertEqual(a._drain.head_completions,0 if expected_active else 1)


if __name__=="__main__":unittest.main(verbosity=2)
