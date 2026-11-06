"""Execute unmodified adapter methods with metadata fixtures, NOT a GPU engine."""

import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest
from prc_nc.algorithms import State, Group, indexed_closure, LinearBuildMarginIndex, interval_trial
from prc_nc.drain import DrainPolicy
from prc_nc.controller import Refused
from prc_nc.admission import AdmissionContext, AdmissionDeferred
from prc_nc.runtime import AllocationAuthority
from prc_nc.online import ObservedRequest, OnlineProposalEngine
from prc_nc.physical import PageOwnershipIndex
from prc_nc.prediction import PredictionResult
from prc_nc.routing import RouteDevice


SOURCE = Path(__file__).resolve().parents[1] / "prc_nc/adapters/vllm_scheduler.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CertificateScheduler")
methods = [
    n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in ("_snapshot", "_prc_allow", "_online_proposal", "schedule", "_study_candidate_counterfactual")
]
extracted = ast.ClassDef(name="Adapter", bases=[], keywords=[], body=methods, decorator_list=[])
namespace = dict(
    defaultdict=defaultdict,
    re=re,
    hashlib=hashlib,
    json=json,
    time=time,
    State=State,
    Group=Group,
    indexed_closure=indexed_closure,
    LinearBuildMarginIndex=LinearBuildMarginIndex,
    interval_trial=interval_trial,
    Refused=Refused,
    AdmissionContext=AdmissionContext, AdmissionDeferred=AdmissionDeferred,
    ObservedRequest=ObservedRequest,
    RouteDevice=RouteDevice,
)
exec(compile(ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[])), str(SOURCE), "exec"), namespace)
Adapter = namespace["Adapter"]


def fixture(audit=True, mode="shared_incremental"):
    a = Adapter()
    blocks = [NS(block_id=i, is_null=i == 0, ref_cnt=0) for i in range(7)]
    mapping = {"a": [blocks[1], blocks[2]], "b": [blocks[3]]}
    for row in mapping.values():
        for block in row:
            block.ref_cnt += 1
    a._pm = NS(req_to_blocks=mapping, block_size=16)
    pool = NS(
        blocks=blocks,
        num_gpu_blocks=7,
        get_num_free_blocks=lambda: sum(not b.is_null and b.ref_cnt == 0 for b in blocks),
    )
    a.kv_cache_manager = NS(block_pool=pool)
    a.requests = {
        rid: NS(
            request_id=rid,
            num_prompt_tokens=16 * len(row),
            max_tokens=16 * need,
            num_computed_tokens=16 * len(row),
            num_preemptions=0,
            output_token_ids=[],
        )
        for (rid, row), need in zip(mapping.items(), (2, 3))
    }
    a.max_model_len = 1024
    a._mode = mode
    a._drain_enabled = mode in ("shared_incremental", "shared_closure")
    a._audit = audit
    a._step = 0
    a._drain = DrainPolicy(64)
    a._index = a._index_state = a._index_ids = None
    a._pending_projection = {}
    a._pending_index = {}
    a.events = []
    a._log = lambda event, **values: a.events.append(dict(event=event, **values))
    return a


class FixedPredictor:
    z_value = 2.
    calibration_state = "calibrated"

    def __init__(self, remaining):
        self.remaining = remaining
        self.queries = []

    def predict(self, query):
        self.queries.append(query)
        value = min(self.remaining, query.max_output_tokens-query.observed_output_tokens)
        return PredictionResult(query.request_id, value, 0., value, "calibrated", "test", "fixture", 1)


def with_authority(a, remaining):
    a._serial = 0
    a._pending_commit = {}
    a._proposals = {}
    a._pipeline_enabled = False
    a._last_batch_tokens = 32
    a.scheduler_config = NS(max_num_batched_tokens=64, max_num_seqs=4)
    a.waiting = ["waiting"]
    a._ownership = PageOwnershipIndex(6)
    for rid, blocks in a._pm.req_to_blocks.items():
        a._ownership.observe_request(rid, [b.block_id for b in blocks])
    s, ids, _ = a._snapshot()
    a._authority = AllocationAuthority(s, ids, audit=a._audit, pool_id="gpu0-kv")
    a._online = OnlineProposalEngine(FixedPredictor(remaining), block_size=16)
    return a


class AdapterMetadataTests(unittest.TestCase):
    def test_online_statistics_change_new_admission_without_changing_contracts(self):
        decisions = []
        for estimate in (0, 1000):
            a = with_authority(fixture(), estimate)
            request = NS(request_id="waiting", num_prompt_tokens=16, max_tokens=16,
                         num_computed_tokens=0, output_token_ids=[])
            a.requests["waiting"] = request
            before = a._snapshot()[0]
            decisions.append(a._prc_allow(request, 1))
            self.assertEqual(a._snapshot()[0], before)  # proposal is not allocation
            binding = next(e for e in a.events if e["event"] == "proposal_binding")
            self.assertEqual(binding["state_version"], 0)
            self.assertEqual(binding["telemetry"]["free_capacity"], 3)
            self.assertEqual(binding["telemetry"]["compute_pressure"], .5)
            self.assertEqual(binding["telemetry"]["memory_pressure"], .5)
            self.assertEqual(binding["telemetry"]["queue_pressure"], .25)
            if decisions[-1]:
                grant = a._pending_commit["waiting"]
                self.assertEqual(grant.action_id, binding["action_id"])
                self.assertEqual(grant.worker_id, binding["route"]["device_id"])
                self.assertEqual(a._authority.virtual.free, (2,))
        self.assertEqual(decisions, [True, False])

    def test_existing_request_progress_ignores_statistical_admission_gate(self):
        a = with_authority(fixture(), 1000)
        a.requests["a"].output_token_ids = [99]
        self.assertTrue(a._prc_allow(a.requests["a"], 1))
        binding = next(e for e in a.events if e["event"] == "proposal_binding")
        self.assertFalse(binding["statistical_reserve"]["fits"])
        self.assertFalse(binding["new_admission"])
        self.assertEqual(binding["observed_output_tokens"], 1)
        self.assertEqual(a._online.predictor.queries[0].observed_output_tokens, 1)

    def test_underprediction_cannot_authorize_an_unsafe_hard_state(self):
        a = with_authority(fixture(), 0)
        request = NS(request_id="waiting", num_prompt_tokens=16, max_tokens=80,
                     num_computed_tokens=0, output_token_ids=[])
        a.requests["waiting"] = request
        # Its first three pages consume all free pages; nobody can then finish.
        self.assertFalse(a._prc_allow(request, 48))
        self.assertIsNone(a._authority.pending)
        self.assertEqual(a._authority.version, 0)

    def test_authority_fast_path_does_not_call_adapter_oracle(self):
        a = with_authority(fixture(audit=False), 0)
        old = namespace["indexed_closure"]
        namespace["indexed_closure"] = lambda _: self.fail("redundant adapter oracle")
        try:
            self.assertTrue(a._prc_allow(a.requests["a"], 1))
        finally:
            namespace["indexed_closure"] = old
        self.assertEqual(a._authority.controller.stats["oracle"], 0)

    def test_online_authority_audit_performance_decisions_match(self):
        decisions = []
        for audit in (True, False):
            a = with_authority(fixture(audit=audit), 0)
            row = []
            for rid, tokens in (("a", 1), ("b", 17), ("a", 32)):
                before = a._snapshot()[0]
                allowed = a._prc_allow(a.requests[rid], tokens)
                row.append(allowed)
                if allowed:
                    grant = a._pending_commit.pop(rid)
                    a._authority.stop_unexecuted(grant, lambda: a._snapshot()[:2])
                    a._pending_projection.pop(rid)
                self.assertEqual(a._snapshot()[0], before)
            decisions.append(row)
        self.assertEqual(decisions[0], decisions[1])

    def test_actual_fallback_and_rebuild(self):
        a = fixture()
        s, ids, _ = a._snapshot()
        a._index = LinearBuildMarginIndex(s, (0, 1))
        a._index_state = s
        a._index_ids = tuple(ids)
        self.assertTrue(a._prc_allow(a.requests["b"], 17))
        event = next(e for e in a.events if e["event"] == "incremental_proposal")
        self.assertTrue(event["complete_fallback"])
        self.assertTrue(event["full_accepted"])
        self.assertEqual(a._pending_index["b"][0], "rebuild")
        self.assertEqual(a._snapshot()[0], s)

    def test_oracle_is_skipped_on_valid_fast_path(self):
        a = fixture(audit=False)
        s, ids, _ = a._snapshot()
        a._index = LinearBuildMarginIndex(s, (0, 1))
        a._index_state = s
        a._index_ids = tuple(ids)
        original = namespace["indexed_closure"]

        def forbidden(_):
            raise AssertionError("full oracle on fast path")

        namespace["indexed_closure"] = forbidden
        try:
            self.assertTrue(a._prc_allow(a.requests["a"], 1))
        finally:
            namespace["indexed_closure"] = original
        event = next(e for e in a.events if e["event"] == "incremental_proposal")
        self.assertFalse(event["oracle_checked"])
        self.assertIsNone(event["equivalent"])

    def test_waiting_not_protected_until_candidate(self):
        a = fixture()
        waiting = NS(request_id="waiting", num_prompt_tokens=16, max_tokens=16, num_computed_tokens=0)
        a.requests["waiting"] = waiting
        self.assertNotIn("waiting", a._snapshot()[1])
        self.assertIn("waiting", a._snapshot(waiting)[1])

    def test_resident_and_zero_owner_cached_prefix(self):
        for cached_id, touched in ((1, 0), (4, 1)):
            a = fixture()
            waiting = NS(request_id="waiting", num_prompt_tokens=16, max_tokens=16, num_computed_tokens=0)
            a.requests["waiting"] = waiting
            block = a.kv_cache_manager.block_pool.blocks[cached_id]
            self.assertTrue(a._prc_allow(waiting, 1, 16, NS(blocks=([block],))))
            event = next(e for e in a.events if e["event"] == "proposal")
            self.assertEqual(event["touched_free"], touched)
            self.assertEqual(event["free_after"], 2 - touched)

    def test_drain_defers_nonhead_without_allocation(self):
        a = fixture()
        a._drain = DrainPolicy(1)
        a._drain.observe(("a", "b"), (0, 1))
        before = a._snapshot()[0]
        self.assertFalse(a._prc_allow(a.requests["b"], 1))
        self.assertEqual(a._snapshot()[0], before)
        self.assertEqual(a.events[-1]["event"], "drain_defer")

    def test_schedule_triggers_finite_drain(self):
        a = fixture()
        a._drain = DrainPolicy(1)
        namespace["_instrumented_schedule"] = lambda self: tuple(r for r in self.requests if self._drain.allows(r))
        result = a.schedule()
        self.assertEqual(len(result), 1)
        self.assertTrue(a._drain.active)
        self.assertTrue(any(e["event"] == "drain_start" for e in a.events))
        snapshot = next(e for e in a.events if e["event"] == "snapshot")
        self.assertEqual(len(snapshot["request_phases"]), 2)

    def test_native_shadow_not_subject_to_drain(self):
        a = fixture(mode="native_shadow")
        a._drain = DrainPolicy(1)
        namespace["_instrumented_schedule"] = lambda self: tuple(self.requests)
        self.assertEqual(a.schedule(), ("a", "b"))
        self.assertFalse(a._drain.active)

    def test_hard_reserve_is_not_prc_drain_or_proposal_path(self):
        a = fixture(mode="hard_reserve")
        a._drain_enabled = False
        a._pipeline_enabled = False
        a._drain = DrainPolicy(1)
        a._drain.observe(("a", "b"), (0, 1))
        a._prc_allow(a.requests["b"], 1)
        self.assertFalse(any(e["event"] == "drain_defer" for e in a.events))
        self.assertTrue(any(e["event"] == "proposal" for e in a.events))

    def test_exclusive_closure_uses_conservative_release_certificate(self):
        a = fixture(mode="exclusive_closure")
        a._drain_enabled = False
        a._pipeline_enabled = False
        # The fixture starts with two independent requests.  A candidate that
        # reuses a resident block is checked with that group pinned, just as in
        # diagnostics.boundary_report's conservative C0 state.
        waiting = NS(request_id="waiting", num_prompt_tokens=16, max_tokens=16, num_computed_tokens=0)
        a.requests["waiting"] = waiting
        resident_block = a.kv_cache_manager.block_pool.blocks[1]
        self.assertTrue(a._prc_allow(waiting, 1, 16, NS(blocks=([resident_block],))))
        event = next(e for e in a.events if e["event"] == "proposal")
        self.assertTrue(event["exclusive_certificate"])

    def test_performance_snapshot_uses_event_index_without_physical_pool_scan(self):
        a = with_authority(fixture(audit=True), 0)
        expected = a._snapshot()

        class NoScan:
            def __iter__(self):
                raise AssertionError("physical page pool scanned")

            def items(self):
                raise AssertionError("all physical request mappings scanned")

        a.kv_cache_manager.block_pool.blocks = NoScan()
        a._pm.req_to_blocks = NoScan()
        a._audit = False
        self.assertEqual(a._snapshot(), expected)
        a._audit = True
        with self.assertRaisesRegex(AssertionError, "scanned"):
            a._snapshot()

    def test_unmodeled_reference_rejected(self):
        a = fixture()
        a.kv_cache_manager.block_pool.blocks[1].ref_cnt += 1
        with self.assertRaisesRegex(AssertionError, "Incomplete owner"):
            a._snapshot()

    def test_partial_prefix_external_and_speculative_rejected(self):
        a = fixture()
        for kw in (dict(num_new_computed_tokens=1), dict(num_external_computed_tokens=1), dict(num_lookahead_tokens=1)):
            with self.assertRaises(RuntimeError):
                a._prc_allow(a.requests["a"], 1, **kw)
