"""CPU fixtures execute audited native-loop hooks; no GPU/throughput evidence."""
import ast
import contextlib
import inspect
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from prc_nc.algorithms import indexed_closure, linear_margins
from prc_nc.configuration import validate_config
from prc_nc.drain import DrainPolicy
from prc_nc.drain_batching import fixed_witness_check, fixed_witness_pair, head_first_running
from prc_nc.model import State, Group, apply_private_growth, valid_witness
from prc_nc.runtime import AllocationAuthority
from test_adapter_metadata import fixture, namespace, SOURCE
from native_source_fixture import native_source

namespace.update(fixed_witness_pair=fixed_witness_pair, head_first_running=head_first_running)


def configuration(**changes):
    c=dict(method="shared_incremental", drain_scope="cohort", drain_policy="timer",
           certificate_variant="common_full_c1", enable_prc_pipeline=False,
           rank_order=False, zero_growth_fast=False, deny_cache=False)
    c.update(changes)
    return c


def guarded_fixture(variant="common_full_c1"):
    a=fixture(audit=True)
    a._zero_growth_fast=a._deny_cache_enabled=a._rank_order=False
    a._serial=0; a._pending_commit={}; a._proposals={}; a._online=None
    a._pipeline_enabled=False
    state, ids, _=a._snapshot()
    a._authority=AllocationAuthority(state, ids, audit=True, certificate_variant=variant)
    a._drain=DrainPolicy(1, scope="cohort", batching="witness_preserving")
    a._drain.observe(ids, (0,1))
    a._drain_batch_index=1
    return a


def install_real_allocate_body(a, original_allocate):
    """Execute actual nested traced_allocate, including Authority.dispatch."""
    cls=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.ClassDef) and n.name=="CertificateScheduler")
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="__init__")
    function=next(n for n in init.body if isinstance(n,ast.FunctionDef) and n.name=="traced_allocate")
    env=dict(namespace, self=a, original_allocate=original_allocate,
             allocation_signature=inspect.signature(original_allocate))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(SOURCE),"exec"),env)
    a.kv_cache_manager.allocate_slots=env["traced_allocate"]
    a._zero_growth_pass=set()
    a._capture_request=lambda rid: None  # fixture _snapshot performs the full physical scan


def native_running_fixture(a, *, budget=2, fail_allocation=False):
    """Transform real vLLM 0.29 source with real adapter hook code, then
    execute its unchanged running-loop prefix using CPU physical-page fixtures.
    Full waiting-loop execution, engine outputs and CUDA are outside this test.
    """
    native_path=native_source("vllm/v1/core/sched/scheduler.py")
    native_text=native_path.read_text()
    native_tree=ast.parse(native_text)
    cls=next(n for n in native_tree.body if isinstance(n,ast.ClassDef) and n.name=="Scheduler")
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="schedule")
    import textwrap
    original=textwrap.dedent("\n".join(native_text.splitlines()[method.lineno-1:method.end_lineno])+"\n")
    adapter_tree=ast.parse(SOURCE.read_text())
    start=next(i for i,n in enumerate(adapter_tree.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=="source" for t in n.targets))
    transform=ast.Module(body=adapter_tree.body[start:],type_ignores=[])
    env=dict(inspect=NS(getsource=lambda method:original),textwrap=textwrap,
             Scheduler=NS(schedule=None),base_module=NS(SchedulerOutput=object))
    exec(compile(ast.fix_missing_locations(transform),str(SOURCE),"exec"),env)
    # Both actual native entry hooks must carry the allocated-token dictionary.
    hooks=[n for n in ast.walk(ast.parse(env["source"])) if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute) and n.func.attr=="_prc_allow"]
    assert len(hooks)==2 and all(any(k.arg=="scheduled_tokens" and isinstance(k.value,ast.Name)
                                   and k.value.id=="num_scheduled_tokens" for k in n.keywords) for n in hooks)
    function=ast.parse(env["source"]).body[0]
    end=next(i for i,n in enumerate(function.body) if isinstance(n,ast.While))
    function.body=function.body[:end+1]+[ast.Return(value=ast.Call(func=ast.Name(id="NS",ctx=ast.Load()),
        args=[],keywords=[ast.keyword(arg="num_scheduled_tokens",value=ast.Name(id="num_scheduled_tokens",ctx=ast.Load()))]))]
    function.returns=None
    for arg in function.args.args: arg.annotation=None
    native_env=dict(NS=NS, time=namespace["time"], PauseState=NS(PAUSED_ALL="paused"),
                    record_function_or_nullcontext=lambda name:contextlib.nullcontext())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(native_path),"exec"),native_env)
    namespace["_instrumented_schedule"]=native_env["schedule"]
    a.current_step=0; a.max_num_scheduled_tokens=budget
    a.vllm_config=NS(speculative_config=None)
    a.scheduler_config=NS(max_num_batched_tokens=budget,long_prefill_token_threshold=0)
    a._pause_state="unpaused"; a.max_num_encoder_input_tokens=0
    a.num_sampled_tokens_per_step=1; a.need_mamba_block_aligned_split=False
    a.num_prefill_lookahead=0; a.num_lookahead_tokens=0
    a._reserve_prefill_lookahead=lambda request,computed,num:num
    a.kv_cache_manager.new_step_starts=lambda:None
    for r in a.requests.values():
        r.num_output_placeholders=0; r.next_decode_eligible_step=0
        r.num_tokens_with_spec=r.num_computed_tokens+1
        r.has_encoder_inputs=False; r.spec_token_ids=[]; r.is_prefill_chunk=False
    # Deliberately place native followers before the captured head.
    a.running=[a.requests["b"],a.requests["a"]]
    def allocate(request, num_new_tokens, num_lookahead_tokens=0):
        if fail_allocation:return None
        target=request.num_computed_tokens+num_new_tokens
        count=max(0,(target+15)//16-len(a._pm.req_to_blocks[request.request_id]))
        free=[b for b in a.kv_cache_manager.block_pool.blocks if not b.is_null and b.ref_cnt==0]
        if len(free)<count:return None
        for block in free[:count]:
            block.ref_cnt=1; a._pm.req_to_blocks[request.request_id].append(block)
        return NS(blocks=())
    install_real_allocate_body(a,allocate)
    return env["source"]


class WitnessBatchTests(unittest.TestCase):
    def setUp(self):
        old=namespace.get("_instrumented_schedule")
        self.addCleanup(lambda:namespace.__setitem__("_instrumented_schedule",old))

    def test_config_and_env_binding_default_serial(self):
        self.assertEqual(validate_config(configuration())["drain_batching"],"serial")
        self.assertEqual(validate_config(configuration(drain_batching="witness_preserving"))["drain_scope"],"cohort")
        for change in (dict(drain_scope="head"),dict(drain_policy="none"),dict(rank_order=True),
                       dict(zero_growth_fast=True),dict(deny_cache=True),dict(enable_prc_pipeline=True)):
            with self.assertRaises(ValueError):validate_config(configuration(drain_batching="witness_preserving",**change))
        with self.assertRaises(ValueError):validate_config(configuration(drain_batching="typo"))
        text=(SOURCE.parents[1]/"gpu_run.py").read_text()
        self.assertLess(text.index('"PRC_DRAIN_BATCHING": config["drain_batching"]'), text.index('from transformers import'))

    def test_existing_witness_elsewhere_is_not_captured_witness(self):
        s=State((2,),((2,),(1,)),(Group("a",0,1,frozenset({0})),Group("b",0,1,frozenset({1}))))
        self.assertTrue(valid_witness(s,(0,1)))
        post=apply_private_growth(s,((0,),(1,)))
        self.assertEqual(len(indexed_closure(post)),2)
        self.assertFalse(fixed_witness_check(post,("a","b"),("a","b"))["accepted"])
        self.assertTrue(fixed_witness_check(post,("a","b"),("b","a"))["accepted"])

    def test_dual_view_scans_are_paid_in_order(self):
        post=State((0,),((0,),(0,),(1,),(0,)),(
            Group("shared",0,1,frozenset({0,1})),Group("c",0,1,frozenset({2})),Group("d",0,2,frozenset({3}))))
        import prc_nc.drain_batching as module
        for active in (False,True):
            with patch.object(module,"fixed_witness_check",wraps=fixed_witness_check) as checks:
                pair=fixed_witness_pair(post,tuple("abcd"),tuple("abcd"),conservative=active)
            self.assertEqual(checks.call_count,2)
            self.assertEqual([c.kwargs["conservative"] for c in checks.call_args_list],[True,False])
            self.assertFalse(pair["c0"]["accepted"]);self.assertTrue(pair["c1"]["accepted"])
            self.assertEqual(pair["active"]["accepted"],not active)

    def test_actual_hook_selects_c0_vs_c1_fixed_view(self):
        for variant,expected in (("common_full_c0",False),("common_full_c1",True)):
            a=guarded_fixture(variant)
            blocks=[NS(block_id=i,is_null=i==0,ref_cnt=0) for i in range(5)]
            mapping={"a":[blocks[1]],"b":[blocks[1]],"c":[blocks[2]],"d":[blocks[3]]}
            for row in mapping.values():
                for block in row:block.ref_cnt+=1
            a._pm=NS(req_to_blocks=mapping,block_size=16)
            a.kv_cache_manager=NS(block_pool=NS(blocks=blocks,num_gpu_blocks=5,
                get_num_free_blocks=lambda:sum(not b.is_null and b.ref_cnt==0 for b in blocks)))
            a.requests={rid:NS(request_id=rid,num_prompt_tokens=15 if rid in "ab" else 16,
                              max_tokens=1 if rid in "ab" else 16,num_computed_tokens=14 if rid in "ab" else 16,
                              num_preemptions=0,output_token_ids=[]) for rid in "abcd"}
            state,ids,_=a._snapshot()
            a._authority=AllocationAuthority(state,ids,audit=True,certificate_variant=variant)
            a._drain=DrainPolicy(1,scope="cohort",batching="witness_preserving")
            a._drain.observe(ids,(0,1,2,3))
            self.assertEqual(a._prc_allow(a.requests["d"],1,scheduled_tokens={"a":1}),expected)
            record=next(e for e in a.events if e["event"]=="drain_frozen_witness_check")
            self.assertFalse(record["c0_fixed_accepted"]);self.assertTrue(record["c1_fixed_accepted"])
            self.assertEqual(record["active_view"],"C1_physical" if expected else "C0_pinned_shared")
            self.assertEqual((a._drain_fixed_checks,a._drain_c0_checks,a._drain_c1_checks),(1,1,1))
            self.assertEqual(a._authority.version,1 if expected else 0)
            if expected:
                grant=a._pending_commit.pop("d")
                a._authority.stop_unexecuted(grant,lambda:a._snapshot()[:2])
            self.assertIsNone(a._authority.pending)

    def test_exact_resident_identity_and_stable_followers(self):
        running=[NS(request_id=r) for r in ("b","c","a","d")]
        self.assertEqual([r.request_id for r in head_first_running(running,"a")],["a","b","c","d"])
        with self.assertRaises(RuntimeError):head_first_running(running,"waiting")
        with self.assertRaises(RuntimeError):fixed_witness_check(State((1,),((0,),),()),("a",),("waiting",))

    def test_cancellation_and_missing_callback_do_not_reorder(self):
        d=DrainPolicy(1,scope="cohort",batching="witness_preserving")
        d.observe(tuple("abc"),(0,1,2))
        self.assertTrue(d.allows("b"));self.assertFalse(d.allows("waiting"))
        with self.assertRaises(RuntimeError):d.observe(tuple("ac"),(0,1))
        d.finished("b",cancelled=True,completed=False)
        self.assertEqual(d.order,("a","c"));self.assertTrue(d.active)
        self.assertEqual(d.head_completions,0)
        with self.assertRaises(RuntimeError):d.finished("a",cancelled=False,completed=False)
        d.finished("a",cancelled=True,completed=False)
        self.assertTrue(d.active);self.assertEqual(d.order,("c",))
        d.finished("c",completed=True)
        self.assertFalse(d.active);self.assertEqual(d.head_completions,1)

    def test_actual_hook_defers_follower_without_allocated_head_budget(self):
        a=guarded_fixture()
        before=a._snapshot()[0]
        for scheduled in (None,{}, {"a":0}):
            self.assertFalse(a._prc_allow(a.requests["b"],1,scheduled_tokens=scheduled))
        self.assertEqual(a._snapshot()[0],before);self.assertIsNone(a._authority.pending)
        waiting=NS(request_id="waiting")
        self.assertFalse(a._prc_allow(waiting,1,scheduled_tokens={"a":1}))

    def test_real_native_loop_head_last_budget_one(self):
        a=guarded_fixture()
        native_running_fixture(a,budget=1)
        output=a.schedule()
        self.assertEqual(output.num_scheduled_tokens,{"a":1})
        self.assertEqual([r.request_id for r in a.running],["a","b"])
        self.assertIsNone(a._authority.pending)
        self.assertEqual(a._authority.confirmed,a._snapshot()[0])
        batch=next(e for e in a.events if e["event"]=="drain_batch")
        self.assertEqual(batch["head_tokens"],1);self.assertEqual(batch["nonhead_requests"],0)

    def test_real_native_loop_safe_follower_and_full_audit_state(self):
        for variant in ("common_full_c0","common_full_c1"):
            a=guarded_fixture(variant)
            native_running_fixture(a,budget=2)
            output=a.schedule()
            self.assertEqual(output.num_scheduled_tokens,{"a":1,"b":1})
            records=[e for e in a.events if e["event"]=="drain_frozen_witness_check"]
            self.assertEqual(len(records),2)
            self.assertEqual(records[1]["head_tokens_scheduled"],1)
            self.assertTrue(all(e["actual_allow"] and e["fixed_accepted"] for e in records))
            self.assertTrue(all(e["grant_id"] and e["post"]["owner_classes"] for e in records))
            self.assertEqual((a._drain_fixed_checks,a._drain_c0_checks,a._drain_c1_checks),(2,2,2))
            self.assertIsNone(a._authority.pending)
            self.assertEqual(a._authority.confirmed,a._snapshot()[0])

    def test_real_native_loop_rejects_alternative_order_follower(self):
        a=guarded_fixture()
        native_running_fixture(a,budget=18)
        a.requests["a"].num_computed_tokens=31
        a.requests["a"].num_tokens_with_spec=32 # positive head service in existing page
        a.requests["b"].num_prompt_tokens=48;a.requests["b"].max_tokens=16
        a.requests["b"].num_tokens_with_spec=48 # legitimate unfinished prefill
        output=a.schedule()
        self.assertEqual(output.num_scheduled_tokens,{"a":1})
        denied=next(e for e in a.events if e["event"]=="drain_frozen_witness_check" and e["request_id"]=="b")
        self.assertFalse(denied["fixed_accepted"]);self.assertFalse(denied["actual_allow"])
        post=denied["post"]
        state=State(tuple(post["free"]),tuple(map(tuple,post["need"])),tuple(
            Group(str(i),g["pool"],g["pages"],frozenset(g["owner_ranks"]),g["pinned"])
            for i,g in enumerate(post["owner_classes"])))
        self.assertEqual(len(indexed_closure(state)),state.n)
        self.assertEqual(len(a._pm.req_to_blocks["b"]),1)
        self.assertIsNone(a._authority.pending)

    def test_real_native_loop_zero_head_tokens_means_empty_batch(self):
        a=guarded_fixture();native_running_fixture(a,budget=2)
        a.requests["a"].num_tokens_with_spec=a.requests["a"].num_computed_tokens
        output=a.schedule()
        self.assertEqual(output.num_scheduled_tokens,{})
        batch=next(e for e in a.events if e["event"]=="drain_batch")
        self.assertTrue(batch["empty"]);self.assertFalse(batch["progress_credited"])
        self.assertEqual(a._drain.head_completions,0)

    def test_real_allocate_failure_stops_before_native_preemption(self):
        a=guarded_fixture();native_running_fixture(a,budget=2,fail_allocation=True)
        before=a._snapshot()[0]
        with self.assertRaisesRegex(RuntimeError,"preemption is not permitted"):a.schedule()
        self.assertEqual(a._snapshot()[0],before);self.assertIsNone(a._authority.pending)
        self.assertTrue(a._drain.active)

    def test_missing_running_head_is_explicit_failure(self):
        a=guarded_fixture();native_running_fixture(a,budget=2)
        a.running=[a.requests["b"]]
        with self.assertRaisesRegex(RuntimeError,"uniquely present"):a.schedule()
        self.assertEqual(a.events[-1]["event"],"drain_head_unavailable")

    def test_final_output_cannot_bypass_mandatory_head(self):
        a=guarded_fixture();native_running_fixture(a,budget=2)
        namespace["_instrumented_schedule"]=lambda a:NS(num_scheduled_tokens={"b":1})
        with self.assertRaisesRegex(AssertionError,"positive head budget"):a.schedule()

    def test_actual_result_observer_confirms_after_base_once(self):
        tree=ast.parse(SOURCE.read_text())
        original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="CertificateScheduler")
        method=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=="update_from_output")
        calls=[]
        class Base:
            def update_from_output(self,output,result):
                calls.append("base_return_after_execution");return "unchanged-base-result"
        definition=ast.ClassDef(name="Observed",bases=[ast.Name(id="Base",ctx=ast.Load())],
            keywords=[],body=[method],decorator_list=[])
        env=dict(Base=Base)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[])),str(SOURCE),"exec"),env)
        for batching in ("serial","witness_preserving"):
            a=env["Observed"](); events=[]
            a._log=lambda event,**kw:events.append(dict(event=event,**kw))
            output=NS(num_scheduled_tokens={"a":1,"b":1})
            proof=dict(episode=1,batch_index=2,head="a",batching=batching,
                       frozen_order=["a","b"],scheduled_tokens=output.num_scheduled_tokens,head_tokens=1)
            a._drain_pending_batches={id(output):(output,proof)}
            before=len(calls)
            self.assertEqual(a.update_from_output(output,NS(req_ids=["a","b"])),"unchanged-base-result")
            self.assertEqual(len(calls),before+1);self.assertFalse(a._drain_pending_batches)
            self.assertEqual(events[-1]["event"],"drain_batch_confirmed")
            self.assertTrue(events[-1]["physical_execution_observed"])
            self.assertEqual(a._drain_confirmed_batches,1)
            a._drain_pending_batches={id(output):(output,proof)}
            with self.assertRaisesRegex(AssertionError,"omits"):a.update_from_output(output,NS(req_ids=["b"]))
            self.assertTrue(a._drain_pending_batches)


if __name__=="__main__":unittest.main(verbosity=2)
