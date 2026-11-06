"""Experimental vLLM 0.29.0 single-full-attention-pool adapter.

Scope: TP=PP=1, synchronous scheduling, no speculative decoding/connectors,
full-block prefix hits, no sliding window. Unsupported states fail explicitly.
Audit mode checks incremental decisions against full closure; non-audit mode
skips that oracle on a valid fast path. Physical snapshots/projections are still
checked, so neither mode is advertised as a low-overhead production runtime.
Native mode retains native scheduling and preemption. Guarded mode skips an
uncertifiable running allocation rather than turning rejection into preemption.
"""

from __future__ import annotations
import hashlib
import atexit
import inspect
import json
import os
import re
import textwrap
import time
from pathlib import Path
from collections import defaultdict
import vllm
import vllm.v1.core.sched.scheduler as base_module
from vllm.v1.core.sched.scheduler import Scheduler
from prc_nc.algorithms import State, Group, indexed_closure, LinearBuildMarginIndex, linear_margins, interval_trial
from prc_nc.drain import DrainPolicy, terminal_flags
from prc_nc.active_admission import ActiveAdmission, SEMANTICS as ACTIVE_ADMISSION_SEMANTICS
from prc_nc.drain_batching import fixed_witness_pair, head_first_running
from prc_nc.runtime import AllocationAuthority
from prc_nc.physical import PageOwnershipIndex
from prc_nc.controller import Refused
from prc_nc.admission import AdmissionContext, AdmissionDeferred
from prc_nc.peer_admission import peer_admission_decision, pair_funding_exemption
from prc_nc.online import ObservedRequest, OnlineProposalEngine
from prc_nc.prediction import load_runtime_predictor
from prc_nc.routing import RouteDevice
from prc_nc.adapters.execution_measurement import ExecutionMeasurementMixin


class CertificateScheduler(ExecutionMeasurementMixin, Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if vllm.__version__ != "0.29.0":
            raise RuntimeError("Adapter requires the audited vLLM 0.29.0 source")
        if self.scheduler_config.async_scheduling or self.num_lookahead_tokens:
            raise RuntimeError("Async/speculative scheduling unsupported")
        if (
            self.connector is not None
            or self.parallel_config.tensor_parallel_size != 1
            or self.parallel_config.pipeline_parallel_size != 1
        ):
            raise RuntimeError("Connectors/distributed placement unsupported")
        managers = self.kv_cache_manager.coordinator.single_type_managers
        if len(managers) != 1 or type(managers[0]).__name__ != "FullAttentionManager":
            raise RuntimeError(f"Expected one full attention manager, got {[type(m).__name__ for m in managers]}")
        self._pm = managers[0]
        self._mode = os.environ.get("PRC_MODE", "native_shadow")
        if self._mode not in (
            "native_shadow",
            "exclusive_closure",
            "shared_closure",
            "shared_incremental",
            "hard_reserve",
        ):
            raise RuntimeError("Unknown PRC mode")
        self._step = 0
        self._audit = os.environ.get("PRC_AUDIT", "1") == "1"
        # CCMB (2026-10-12): PRC_DRAIN_POLICY=none disables the legacy timer-triggered
        # whole-cohort serial drain. Liveness then rests on the witness head never
        # being blocked by its own growth (T5: m_1 is invariant to head growth),
        # with per-allocation fixed-witness checks and complete fallback (T7).
        self._drain_policy = os.environ.get("PRC_DRAIN_POLICY", "timer")
        if self._drain_policy not in ("timer", "none"):
            raise RuntimeError("PRC_DRAIN_POLICY must be 'timer' or 'none'")
        self._drain_enabled = (
            self._mode in ("shared_closure", "shared_incremental") and self._drain_policy == "timer"
        )
        active_flag=os.environ.get("PRC_ACTIVE_DRAIN_ADMISSION","0")
        if active_flag not in ("0","1"):raise RuntimeError("active admission flag must be0/1")
        self._active_drain_admission=active_flag=="1"
        peer_flag = os.environ.get("PRC_PEER_RESIDENT_LIMIT", "0")
        if peer_flag not in ("0","2","4"):
            raise RuntimeError("peer resident limit must be0,2,4")
        self._peer_resident_limit = int(peer_flag)
        self._peer_gate_checks = self._peer_gate_deferrals = 0
        bypass_flag=os.environ.get("PRC_PEER_QUEUE_BYPASS","0")
        if bypass_flag not in ("0","1")or(bypass_flag=="1"and self._peer_resident_limit!=4):
            raise RuntimeError("queue bypass requires bool flag and fixed peerlimit4")
        self._peer_queue_bypass=bypass_flag=="1"
        adaptive_flag=os.environ.get('PRC_PEER_CAPACITY_ADAPTIVE','0')
        if adaptive_flag not in ('0','1') or (adaptive_flag=='1' and
                (self._peer_resident_limit!=4 or not self._peer_queue_bypass)):
            raise RuntimeError('adaptive pair funding requires fixed4 plus queue bypass')
        self._peer_capacity_adaptive=adaptive_flag=='1'
        self._peer_pair_checks=self._peer_pair_exemptions=0
        self._peer_queue_bypasses=0
        self._peer_last_deferred_id=None
        if self._peer_resident_limit and (
                self._mode != "shared_incremental" or self._active_drain_admission
                or os.environ.get("PRC_CERTIFICATE_VARIANT","shared_incremental") != "shared_incremental"
                or os.environ.get("PRC_PRIVATE_REFUSAL_CACHE","0") != "0"
                or os.environ.get("PRC_ORDINARY_WITNESS_PRIORITY","0") != "1"
                or os.environ.get("PRC_ADMISSION_POLICY","unrestricted") != "c0_first_shorter_cap"
                or os.environ.get("PRC_DRAIN_POLICY","timer") != "timer"
                or os.environ.get("PRC_DRAIN_SCOPE","cohort") != "cohort"
                or os.environ.get("PRC_DRAIN_BATCHING","serial") != "cohort_certified"
                or any(os.environ.get(k,"0") != "0"for k in ("PRC_RANK_ORDER","PRC_ZERO_GROWTH_FAST","PRC_DENY_CACHE"))):
            raise RuntimeError("unsupported peer-limit runtime composition")
        self._active_join=None
        self._drain = DrainPolicy(int(os.environ.get("PRC_DRAIN_AFTER_STEPS", "64")),
                                  scope=os.environ.get("PRC_DRAIN_SCOPE", "cohort"),
                                  batching=os.environ.get("PRC_DRAIN_BATCHING", "serial"),
                                  active_admission=self._active_drain_admission)
        self._drain_fixed_checks = self._drain_c0_checks = self._drain_c1_checks = 0
        self._drain_batch_index = 0
        # T9 greedy: visit running requests in witness-rank order so that growth is
        # granted to low ranks first (each rank r consumes margin only at k < r).
        self._rank_order = os.environ.get("PRC_RANK_ORDER", "0") == "1" and self._mode in (
            "shared_closure", "shared_incremental")
        self._rank_reorders = 0
        ordinary_flag = os.environ.get("PRC_ORDINARY_WITNESS_PRIORITY", "0")
        if ordinary_flag not in ("0","1"):
            raise RuntimeError("ordinary priority flag must be0/1")
        self._ordinary_witness_priority = ordinary_flag == "1"
        self._ordinary_reorders = self._ordinary_steps = self._ordinary_confirmed_batches = 0
        self._ordinary_pending_batches = {}
        self._zero_growth_fast = os.environ.get("PRC_ZERO_GROWTH_FAST", "1") == "1"
        self._zero_growth_hits = 0
        self._zero_growth_pass = set()
        # Separate from the legacy uncharged zero-growth shortcut. This opt-in
        # only reuses an unchanged immutable projection; every Authority and
        # allocator transaction below remains mandatory.
        projection_flag = os.environ.get("PRC_ZERO_PROJECTION_REUSE", "0")
        if projection_flag not in ("0", "1"):
            raise RuntimeError("zero projection reuse flag must be0/1")
        self._zero_projection_reuse = projection_flag == "1"
        self._zero_projection_checks = self._zero_projection_hits = 0
        self._zero_projection_audit_checks = 0
        if self._zero_projection_reuse and (
                self._mode != "shared_incremental" or not self._ordinary_witness_priority
                or self._peer_resident_limit != 4 or not self._peer_queue_bypass
                or not self._peer_capacity_adaptive
                or (self._drain.scope, self._drain.batching) != ("cohort", "cohort_certified")):
            raise RuntimeError("zero projection reuse supports only frozen Adaptive INC-Q")
        # Deny cache (v4): a refused growth/admission can only become admissible
        # after a release (T6) — other requests' growth only shrinks the feasible
        # set (T5) and leaves the hard-reserve slack sum(d)-f invariant. So a
        # refusal is cached under the current release epoch and re-evaluated
        # only after the next release. Decisions are unchanged; only repeated
        # snapshots/closures for deferred requests are avoided.
        self._deny_cache_enabled = os.environ.get("PRC_DENY_CACHE", "1") == "1"
        # Optional scheduler-side profiler (PRC_PROFILE=1): cProfile around schedule()
        # and the allocate/free wrappers, dumped to PRC_RUN_DIR/scheduler_profile.pstats.
        self._profiler = None
        if os.environ.get("PRC_PROFILE", "0") == "1":
            import cProfile
            self._profiler = cProfile.Profile()
            run_dir = os.environ.get("PRC_RUN_DIR")
            if run_dir:
                atexit.register(lambda: self._profiler.dump_stats(os.path.join(run_dir, "scheduler_profile.pstats")))
        self._deny_cache = {}
        self._deny_cache_hits = 0
        self._release_epoch = 0
        self._proposals = {}
        proposal_path = os.environ.get("PRC_PROPOSAL_PATH")
        if proposal_path:
            path = Path(proposal_path)
            if not path.is_file():
                raise RuntimeError("PRC proposal manifest is missing")
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
                raise RuntimeError("PRC proposal manifest is malformed")
            ids = [row.get("request_id") for row in raw]
            if any(type(rid) is not str or not rid for rid in ids) or len(set(ids)) != len(ids):
                raise RuntimeError("PRC proposal manifest has duplicate or malformed request IDs")
            self._proposals = {row["request_id"]: row for row in raw}
        self._pipeline_enabled = bool(self._proposals)
        if self._drain.batching in ("witness_preserving", "cohort_certified", "head_funded") and (
            not self._drain_enabled or self._mode != "shared_incremental"
            or self._rank_order or self._zero_growth_fast or self._deny_cache_enabled
            or self._pipeline_enabled
        ):
            raise RuntimeError("guarded batch modes require timer-enabled certificate adapter and disabled shortcuts/pipeline")
        if self._drain.batching in ("cohort_certified", "head_funded") and self._execution_meter is None:
            raise RuntimeError("cohort-certified batches require actual execution-position measurement")
        if self._ordinary_witness_priority and (
            self._rank_order or self._mode != "shared_incremental" or not self._drain_enabled
            or (self._drain.scope,self._drain.batching) != ("cohort","cohort_certified")
            or os.environ.get("PRC_ADMISSION_POLICY","unrestricted") != "c0_first_shorter_cap"
            or os.environ.get("PRC_CERTIFICATE_VARIANT","shared_incremental") not in ("shared_incremental","common_full_c0","common_full_c1", "force_full")
            or self._zero_growth_fast or self._deny_cache_enabled or self._pipeline_enabled):
            raise RuntimeError("ordinary priority draft requires isolated whole-cohort capfilter runtime without rank/shortcuts/pipeline")
        if self._active_drain_admission:
            if (not self._drain_enabled or self._mode!="shared_incremental" or self._execution_meter is None
                    or (self._drain.scope,self._drain.batching)!=("cohort","cohort_certified")
                    or os.environ.get("PRC_ADMISSION_POLICY","unrestricted")!="c0_first_shorter_cap"
                    or os.environ.get("PRC_CERTIFICATE_VARIANT","shared_incremental") not in ("shared_incremental","common_full_c0","common_full_c1", "force_full")
                    or self._rank_order or self._zero_growth_fast or self._deny_cache_enabled or self._pipeline_enabled):
                raise RuntimeError("active admission requires measured isolated whole-cohort capfilter runtime")
            self._active_join=ActiveAdmission.load(os.environ["PRC_FINITE_UNIVERSE_PATH"],os.environ["PRC_FINITE_UNIVERSE_SHA256"])
        self._online = None
        self._last_batch_tokens = 0
        if self._pipeline_enabled:
            first = next(iter(self._proposals.values()))
            predictor, digest = load_runtime_predictor(
                first["predictor_checkpoint"], z_value=first["predictor_z"])
            if digest != first["predictor_checkpoint_sha256"]:
                raise RuntimeError("predictor checkpoint changed after preflight")
            solver = None
            if first.get("patch_checkpoint"):
                from prc_nc.patch_neural import NeuralPatchSolver
                import torch
                checkpoint = Path(first["patch_checkpoint"])
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != first["patch_checkpoint_sha256"]:
                    raise RuntimeError("PATCH checkpoint changed after preflight")
                solver = NeuralPatchSolver.from_checkpoint(
                    checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"
                )
            self._online = OnlineProposalEngine(predictor, block_size=self._pm.block_size, patch_solver=solver)
        self._serial = 0
        self._pending_projection = {}
        self._pending_index = {}
        self._index = self._index_state = self._index_ids = None
        out = Path(os.environ["PRC_RUN_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        self._study_stats_path = out / "certificate_study_stats.json"
        self._events = (out / "allocator_events.jsonl").open("w", buffering=1)
        self._event_buffer = []
        atexit.register(self._flush_events)
        self._ownership = PageOwnershipIndex(self.kv_cache_manager.block_pool.num_gpu_blocks - 1,
            verify_cache=os.environ.get("PRC_VERIFY_CACHE", "0") == "1")
        for rid in self._pm.req_to_blocks:
            self._capture_request(rid)
        self._pending_commit = {}
        initial, initial_ids, _ = self._snapshot()
        refusal_flag = os.environ.get("PRC_PRIVATE_REFUSAL_CACHE", "0")
        if refusal_flag not in ("0", "1"):
            raise RuntimeError("private refusal cache flag must be0/1")
        self._authority = (
            AllocationAuthority(initial, initial_ids, audit=self._audit,
                certificate_variant=os.environ.get("PRC_CERTIFICATE_VARIANT", "shared_incremental"),
                admission_policy=os.environ.get("PRC_ADMISSION_POLICY", "unrestricted"),
                private_refusal_cache=refusal_flag == "1",
                worker_id=os.environ.get("PRC_WORKER_ID", "gpu0"),
                pool_id=os.environ.get("PRC_POOL_ID", "gpu0-kv"),
                on_event=lambda record: self._log("causal_" + record["kind"], **record))
            if self._mode in ("shared_closure", "shared_incremental") else None
        )
        self._log(
            "adapter",
            version=vllm.__version__,
            mode=self._mode,
            certificate_variant=os.environ.get("PRC_CERTIFICATE_VARIANT", "shared_incremental"),
            drain_scope=self._drain.scope,
            drain_batching=self._drain.batching,
            ordinary_witness_priority=self._ordinary_witness_priority,
            zero_projection_reuse_enabled=self._zero_projection_reuse,
            peer_resident_limit=self._peer_resident_limit,
            peer_queue_bypass=self._peer_queue_bypass,
            peer_capacity_adaptive=self._peer_capacity_adaptive,
            active_drain_admission=self._active_drain_admission,
            finite_universe_sha256=self._active_join.sha256 if self._active_join else None,
            admission_policy=os.environ.get("PRC_ADMISSION_POLICY", "unrestricted"),
            scheduler_sha256=hashlib.sha256(inspect.getsource(Scheduler).encode()).hexdigest(),
            block_size=self._pm.block_size,
            pools=1,
            capacity_excludes_null_block=True,
            hard_bound="ceil((prompt_tokens+max_output_tokens)/block_size)-allocated_blocks; may include one conservative final-token slot",
        )
        original_allocate = self.kv_cache_manager.allocate_slots
        allocation_signature = inspect.signature(original_allocate)

        def traced_allocate(request, *a, **kw):
            # CCMB zero-growth fast path: the outer precheck established that this
            # resident request needs no new page, so the allocator call cannot
            # change (f, d, groups). Bypass snapshots, authority grants and
            # projection checks, but verify that no page was in fact allocated.
            if request.request_id in self._zero_growth_pass:
                self._zero_growth_pass.discard(request.request_id)
                n_before = len(self._pm.req_to_blocks.get(request.request_id, ()))
                result = original_allocate(request, *a, **kw)
                n_after = len(self._pm.req_to_blocks.get(request.request_id, ()))
                if n_after != n_before:
                    raise AssertionError("zero-growth fast path allocated new pages")
                return result
            # Native preemption may retry allocation without revisiting schedule's
            # outer precheck. Recompute against the new state on every retry.
            if request.request_id not in self._pending_projection:
                if self._mode != "native_shadow":
                    raise AssertionError("Controlled allocation bypassed precheck")
                bound = allocation_signature.bind(request, *a, **kw)
                bound.apply_defaults()
                names = (
                    "num_new_tokens",
                    "num_new_computed_tokens",
                    "new_computed_blocks",
                    "num_external_computed_tokens",
                    "num_lookahead_tokens",
                )
                self._prc_allow(request, **{k: bound.arguments[k] for k in names})
            before, before_ids, _ = self._snapshot()
            def observe():
                self._capture_request(request.request_id)
                state, ids, _ = self._snapshot()
                return state, ids
            if self._authority is not None:
                grant = self._pending_commit.pop(request.request_id, None)
                if grant is None:
                    raise AssertionError("physical dispatch lacks a charged authority grant")
                result = self._authority.dispatch(grant, lambda: original_allocate(request, *a, **kw), observe)
            else:
                result = original_allocate(request, *a, **kw)
                self._capture_request(request.request_id)
            if result is not None:
                actual, ids, owner_map = self._snapshot()
                expected = self._pending_projection.pop(request.request_id, None)
                if expected is None:
                    raise AssertionError("Successful allocation has no projection")
                if expected is not None:
                    expected_state, expected_ids = expected

                    # Newly allocated block IDs differ; compare invariant quantities
                    # and canonical owner multisets, not synthetic block labels.
                    def canonical(s):
                        return sorted((tuple(sorted(g.owners)), g.count) for g in s.groups)

                    if (
                        ids != expected_ids
                        or actual.free != expected_state.free
                        or actual.need != expected_state.need
                        or canonical(actual) != canonical(expected_state)
                    ):
                        self._log(
                            "projection_mismatch",
                            request_id=request.request_id,
                            actual_free=actual.free,
                            expected_free=expected_state.free,
                        )
                        raise AssertionError("Projected allocation differs from physical allocator")
                pending_index = self._pending_index.pop(request.request_id, None)
                if self._mode == "shared_incremental" and self._authority is None:
                    if pending_index is None:
                        raise AssertionError("Incremental allocation has no transaction")
                    kind, payload = pending_index
                    if kind == "private":
                        req_index, pages = payload
                        self._index.private_growth(req_index, 0, pages)
                    else:
                        self._index = payload
                    if self._audit and self._index.table() != linear_margins(actual, self._index.order):
                        raise AssertionError("Committed incremental margins differ from physical state")
                    self._index_state, self._index_ids = actual, tuple(ids)
                    self._log("incremental_commit", request_id=request.request_id, kind=kind, margin_audit=self._audit)
                actual_certificate = (len(self._authority.controller.order) == actual.n
                    if self._authority is not None else
                    self._index.certificate_nonnegative
                    if self._mode == "shared_incremental" and not self._audit else
                    len(indexed_closure(actual)) == actual.n)
                if self._mode != "native_shadow" and not actual_certificate:
                    raise AssertionError("Controlled allocation lost completion certificate")
                self._log(
                    "allocation",
                    request_id=request.request_id,
                    free=actual.free[0],
                    resident_groups=len(actual.groups),
                    certificate=actual_certificate,
                    projection_verified=expected is not None,
                )
                if getattr(self,"_active_drain_admission",False) and self._drain.active:
                    bound=allocation_signature.bind(request,*a,**kw);bound.apply_defaults()
                    joined=self._active_join.allocation_acked(request,grant,before_ids,ids,self._authority,
                        tokens=bound.arguments["num_new_tokens"],charge=tuple(x-y for x,y in zip(before.free,actual.free)))
                    if joined is not None:
                        if tuple(self._drain.order)!=tuple(joined["before_members"]):raise RuntimeError("drain membership changed before ACK join")
                        self._drain.order=tuple(joined["after_members"])
                        classes=defaultdict(int)
                        for g in actual.groups:classes[(g.pool,tuple(sorted(g.owners)),g.pinned)]+=g.count
                        post_record=dict(free=list(actual.free),need=[list(row) for row in actual.need],cohort=list(ids),
                            owner_classes=[dict(pool=pool,owner_ranks=list(who),pinned=pin,pages=count) for (pool,who,pin),count in sorted(classes.items())])
                        joined.update(physical_post=post_record,physical_post_sha256=hashlib.sha256(json.dumps(post_record,sort_keys=True,separators=(",",":")).encode()).hexdigest())
                        self._log("active_drain_join",**joined)
                if self._drain_enabled and self._drain.active:
                    self._log("drain_service", request_id=request.request_id,
                              witness_head=self._drain.order[0], physical_allocation_confirmed=True)
            else:
                self._pending_projection.pop(request.request_id, None)
                self._pending_index.pop(request.request_id, None)
                after, after_ids, _ = self._snapshot()
                if (before, before_ids) != (after, after_ids):
                    raise AssertionError("Failed allocation changed physical state")
                self._log("native_allocation_refusal", request_id=request.request_id, rollback_audit=True)
                if self._drain_enabled and self._drain.active and self._drain.batching in ("witness_preserving", "cohort_certified", "head_funded"):
                    raise RuntimeError("frozen batch allocation failed; preemption is not permitted")
            return result

        self.kv_cache_manager.allocate_slots = traced_allocate
        original_free = self.kv_cache_manager.free

        def traced_free(request):
            cancelled, completed = terminal_flags(request.status)
            terminal_check=None
            if getattr(self,"_active_drain_admission",False):
                terminal_check=self._active_join.prepare_terminal(request,cancelled=cancelled,completed=completed)
                if terminal_check["new_binding"]:
                    external=terminal_check["external_request_id"];contract=self._active_join.rows[external]
                    self._log("finite_input_binding",request_id=request.request_id,external_request_id=external,
                        universe_sha256=self._active_join.sha256,binding_origin="trusted_never_proposed_cancellation",
                        **{key:contract[key] for key in ("prompt_token_sha256","prompt_tokens","max_tokens")})
            blocks = list(self._pm.req_to_blocks.get(request.request_id, ()))
            expected_release = sum(b.ref_cnt == 1 for b in blocks if not b.is_null)
            free_before = self.kv_cache_manager.block_pool.get_num_free_blocks()
            result = None
            def release():
                nonlocal result
                result = original_free(request)
            def observe():
                self._capture_request(request.request_id)
                state, ids, _ = self._snapshot()
                return state, ids
            if self._authority is not None:
                self._authority.release(request.request_id, release, observe,
                    cancelled=cancelled)
            else:
                release()
                self._capture_request(request.request_id)
            free_after = self.kv_cache_manager.block_pool.get_num_free_blocks()
            if free_after - free_before != expected_release:
                raise AssertionError("Last-owner release count mismatch")
            if terminal_check is not None and terminal_check["repeated"]:
                if self._authority._terminals.get(request.request_id)!=terminal_check["outcome"]:
                    raise RuntimeError("repeated terminal lacks matching authoritative terminal")
                return result
            after, ids, _ = self._snapshot()
            self._index = self._index_state = self._index_ids = None
            self._release_epoch += 1
            self._deny_cache.clear()
            self._pending_projection.pop(request.request_id, None)
            self._pending_index.pop(request.request_id, None)
            if getattr(self,"_active_drain_admission",False):
                self._active_join.terminal(request,cancelled=cancelled,completed=completed)
            if self._drain_enabled:
                if getattr(self,"_active_drain_admission",False):
                    was_head=(self._drain.active and self._active_join.epoch is not None
                        and request.request_id==self._active_join.epoch["head"])
                else:
                    was_head = self._drain.active and request.request_id == (
                        self._drain._first_completion_epoch_head if self._drain.scope == "first_completion"
                        else self._drain.order[0])
                segment_end = self._drain.finished(request.request_id, cancelled=cancelled, completed=completed)
                if cancelled and was_head:
                    self._log("drain_head_cancelled", request_id=request.request_id,
                              successful_completion=False, next_frozen_head=(self._drain.order[0] if self._drain.order else None) if self._drain.batching != "head_funded" and self._drain.scope != "first_completion" and not getattr(self,"_active_drain_admission",False) else None,
                              replacement_pending=getattr(self._drain,"replacement_pending",False))
                if segment_end is not None:
                    self._log("drain_segment_end", **segment_end)
            certificate_state = after
            if self._mode == "exclusive_closure":
                # C0: keep the same physical sharing, but conservatively treat
                # every currently shared group as pinned forever.  This is the
                # fair release-information baseline; it does not alter vLLM's
                # page allocation or smuggle an admission-policy comparison in.
                certificate_state = State(
                    after.free,
                    after.need,
                    tuple(Group(g.key, g.pool, g.count, g.owners, g.pinned or len(g.owners) > 1) for g in after.groups),
                )
            release_certificate = (
                len(indexed_closure(certificate_state)) == after.n
                if self._audit or self._authority is None
                else len(self._authority.controller.order) == after.n
            )
            if self._mode != "native_shadow" and not release_certificate:
                raise AssertionError("Controlled release lost completion certificate")
            self._log(
                "release",
                request_id=request.request_id,
                logical_pages=len(blocks),
                physical_pages=free_after - free_before,
                free_after=free_after,
                request_status=str(request.status),
                live_requests=len(ids),
                certificate=release_certificate,
                all_pages_reclaimable=not ids and free_after == after.capacity[0],
            )
            if not ids:
                self._flush_events()
            return result

        self.kv_cache_manager.free = traced_free

    def _preempt_request(self, request, *args, **kwargs):
        self._log("preemption", request_id=request.request_id, discarded_computed_tokens=request.num_computed_tokens)
        return super()._preempt_request(request, *args, **kwargs)

    def finish_requests(self, *args, **kwargs):
        result = super().finish_requests(*args, **kwargs)
        state, ids, _ = self._snapshot()
        self._log(
            "external_finish_audit",
            free=state.free[0],
            live_requests=len(ids),
            capacity=state.capacity[0],
            certificate=len(indexed_closure(state)) == state.n,
        )
        return result

    def _log(self, event, **values):
        source_index = getattr(self, "_study_log_event_index", 0)
        self._study_log_event_index = source_index + 1
        record = dict(event=event, source_event_index=source_index,
                      monotonic_ns=time.monotonic_ns(), step=self._step, **values)
        if self._audit:
            self._events.write(json.dumps(record) + "\n")
        elif event != "snapshot":
            # Performance mode: no per-event serialization or disk flush.
            # Preserve compact decisions/protocol/lifecycle evidence in memory.
            self._event_buffer.append(record)
            if event in ("causal_fault", "projection_mismatch"):
                self._flush_events()

    def _flush_events(self):
        authority = getattr(self, "_authority", None)
        stats_path = getattr(self, "_study_stats_path", None)
        if authority is not None and stats_path is not None:
            stats_path.write_text(json.dumps({
                "schema_version": 1,
                "certificate_variant": authority.controller.certificate_variant,
                "controller_stats": dict(authority.controller.stats),
                "drain_policy_state": dict(
                    scope=self._drain.scope, batching=getattr(self._drain, "batching", "serial"),
                    after_steps=self._drain.after_steps,
                    episodes=self._drain.episodes, head_completions=self._drain.head_completions,
                    member_completions=self._drain.member_completions,
                    completed_segments=self._drain.completed_segments,
                    empty_without_head_completion=self._drain.empty_without_head_completion,
                    cancelled_members=self._drain.cancelled_members,
                    active=self._drain.active, current_frozen_order=list(self._drain.order),
                    selected_head=getattr(self._drain,"selected_head",None),
                    replacement_pending=getattr(self._drain,"replacement_pending",False)) if getattr(self, "_drain", None) is not None else None,
                "drain_fixed_witness_stats": dict(
                    checks=getattr(self, "_drain_fixed_checks", 0), c0_checks=getattr(self, "_drain_c0_checks", 0),
                    c1_checks=getattr(self, "_drain_c1_checks", 0), batches=getattr(self, "_drain_batch_index", 0),
                    scope="extra full O(M+nP) C0 then C1 fixed-order scans, including warmup; separate from controller oracle counters"),
                "drain_head_funding_stats": dict(checks=getattr(self,"_drain_head_funding_checks",0),
                    deferrals=getattr(self,"_drain_head_funding_deferrals",0),
                    scope="extra all-pool selected-head funding arithmetic; O(P) after identity lookup; not whole-step complexity"),
                "peer_queue_bypass_stats": dict(enabled=getattr(self,"_peer_queue_bypass",False),
                    bypasses=getattr(self,"_peer_queue_bypasses",0),
                    scope="peer-only native waiting-queue transfers; no memory authorization"),
                "peer_admission_stats": dict(limit=getattr(self,"_peer_resident_limit",0),
                    checks=getattr(self,"_peer_gate_checks",0),deferrals=getattr(self,"_peer_gate_deferrals",0),
                    adaptive_enabled=getattr(self,'_peer_capacity_adaptive',False),
                    pair_checks=getattr(self,'_peer_pair_checks',0),
                    pair_exemptions=getattr(self,'_peer_pair_exemptions',0),
                    scope="cumulative including warmup; utility decisions, not certificate feasibility trials"),
                "zero_projection_reuse_stats": dict(
                    enabled=getattr(self,"_zero_projection_reuse",False),
                    checks=getattr(self,"_zero_projection_checks",0),
                    hits=getattr(self,"_zero_projection_hits",0),
                    audit_equivalence_checks=getattr(self,"_zero_projection_audit_checks",0),
                    authority_bypass=False,
                    scope="cumulative including warmup; resident zero-page projection construction only; full grants/dispatch/observation/ACK retained"),
                "ordinary_witness_priority_stats": dict(enabled=getattr(self,"_ordinary_witness_priority",False),
                    observed_steps=getattr(self,"_ordinary_steps",0), relative_order_changes=getattr(self,"_ordinary_reorders",0),
                    pending_batches=len(getattr(self,"_ordinary_pending_batches",{})),
                    confirmed_batches=getattr(self,"_ordinary_confirmed_batches",0),
                    scope="ordinary-only legacy full rank-sort composition; observations not a positive-head/speed guarantee"),
                "drain_execution_observation": dict(
                    pending_batches=len(getattr(self, "_drain_pending_batches", {})),
                    confirmed_batches=getattr(self, "_drain_confirmed_batches", 0)),
                "authority_version": authority.version,
                "authority_faulted": authority.faulted,
                "controller_faulted": authority.controller.faulted,
                "pending_grant": authority.pending is not None,
                "candidate_counterfactual_counts": dict(getattr(self, "_study_candidate_counts", {})),
                "candidate_counterfactual_scope": "audit-only diagnostics, separate from controller oracle statistics; cumulative including warmup",
                "scope": "cumulative controller checks including warmup; not independent trials or per-step timing",
            }, indent=2))
        tracker = getattr(self, "_ownership", None)
        if tracker is not None and tracker.verify_cache:
            self._log("snapshot_cache_verification", cache_hits=tracker.cache_hits,
                      verified_hits=tracker.cache_verifications, scope="cumulative including warmup")
        if self._event_buffer:
            self._events.write("".join(json.dumps(e)+"\n" for e in self._event_buffer))
            self._event_buffer.clear()
        self._events.flush()

    def _capture_request(self, request_id):
        blocks = list(self._pm.req_to_blocks.get(request_id, ()))
        if any(b.is_null for b in blocks):
            raise AssertionError("Null request block unsupported")
        old_ids = self._ownership.requests.get(request_id, ())
        self._ownership.observe_request(request_id, [b.block_id for b in blocks])
        pool = self.kv_cache_manager.block_pool
        changed = set(old_ids) | {b.block_id for b in blocks}
        self._ownership.verify_request_refs([pool.blocks[i] for i in changed])

    def _snapshot(self, extra=None):
        tracker = getattr(self, "_ownership", None)
        if not self._audit and tracker is not None:
            # v5: adapter-level memo keyed by tracker revision, free pages and the
            # provisional request identity. Contracts are static per request, so
            # an unchanged key means an identical (state, ids, owners) triple.
            free_now = self.kv_cache_manager.block_pool.get_num_free_blocks()
            key = (tracker._revision, free_now, None if extra is None else extra.request_id)
            memo = self.__dict__.setdefault("_snapshot_memo", {})
            hit = memo.get(key)
            if hit is not None:
                return hit
            request_ids = set(tracker.requests)
            if extra is not None:
                request_ids.add(extra.request_id)
            contracts = {}
            for rid in request_ids:
                r = extra if extra is not None and rid == extra.request_id else self.requests[rid]
                contracts[rid] = (min(r.num_prompt_tokens+r.max_tokens, self.max_model_len)+self._pm.block_size-1)//self._pm.block_size
            state, ids = tracker.snapshot(contracts, self.kv_cache_manager.block_pool.get_num_free_blocks(),
                                          extra=None if extra is None else extra.request_id)
            rank = {rid:i for i,rid in enumerate(ids)}
            owners = {bid:{rank[r] for r in who} for bid,who in tracker.owners.items()}
            if len(memo) >= 8:
                memo.clear()
            memo[key] = (state, ids, owners)
            return state, ids, owners
        mapping = {rid: list(bs) for rid, bs in self._pm.req_to_blocks.items() if bs}
        if extra is not None:
            mapping.setdefault(extra.request_id, [])
        ids = sorted(mapping)
        rank = {rid: i for i, rid in enumerate(ids)}
        owners = defaultdict(set)
        for rid, blocks in mapping.items():
            if len({b.block_id for b in blocks}) != len(blocks) or any(b.is_null for b in blocks):
                raise AssertionError("Duplicate or null request blocks unsupported")
            for block in blocks:
                owners[block.block_id].add(rank[rid])
        pool = self.kv_cache_manager.block_pool
        for block in pool.blocks:
            if block.is_null:
                continue
            if block.ref_cnt != len(owners.get(block.block_id, ())):
                raise AssertionError("Incomplete owner mapping or unmodeled pin/copy reference")
        free = pool.get_num_free_blocks()
        if free + len(owners) != pool.num_gpu_blocks - 1:
            raise AssertionError("Physical capacity mismatch")
        needs = []
        for rid in ids:
            r = extra if extra is not None and rid == extra.request_id else self.requests[rid]
            total = min(r.num_prompt_tokens + r.max_tokens, self.max_model_len)
            upper = (total + self._pm.block_size - 1) // self._pm.block_size
            if upper < len(mapping[rid]):
                raise AssertionError("Allocated pages exceed contractual upper bound")
            needs.append((upper - len(mapping[rid]),))
        groups = tuple(Group(str(bid), 0, 1, frozenset(who)) for bid, who in sorted(owners.items()))
        return State((free,), tuple(needs), groups), ids, owners

    def _study_candidate_counterfactual(self, post, ids, *, request_id, external_request_id,
                                       action_id, proposal_version, actual_allow, admission_decision=None, admission_policy_defer=False):
        """Inspect an already decided projected action without authorizing it.

        The two pure closure calls never use or mutate the authority controller,
        its witness, or its counters. Only audit diagnostics are written. A
        projected post-state is not an observed allocation or independent trial.
        """
        if not self._audit:
            return
        shared_order = indexed_closure(post)
        private_view = State(post.free, post.need, tuple(
            Group(g.key, g.pool, g.count, g.owners, g.pinned or len(g.owners) > 1)
            for g in post.groups))
        private_order = indexed_closure(private_view)
        shared = len(shared_order) == post.n
        private = len(private_order) == post.n
        full_reserve = all(sum(row[p] for row in post.need) <= post.free[p] for p in range(post.p))
        owner_classes = defaultdict(int)
        for group in post.groups:
            owner_classes[(group.pool, tuple(sorted(group.owners)), group.pinned)] += group.count
        state = dict(free=list(post.free), need=[list(row) for row in post.need],
            cohort=list(ids), owner_classes=[dict(pool=p, owner_ranks=list(owners), pinned=pinned, pages=count)
                for (p, owners, pinned), count in sorted(owner_classes.items())])
        counts = self.__dict__.setdefault("_study_candidate_counts", {})
        candidate_index = counts.get("logged", 0)
        for key, value in (("logged", 1), ("diagnostic_oracle_calls", 2),
                           ("actual_allowed", int(actual_allow)),
                           ("actual_refused", int(not actual_allow)),
                           ("shared_over_exclusive_gain", int(shared and not private)),
                           ("shared_over_sum_demand_gap", int(shared and not full_reserve))):
            counts[key] = counts.get(key, 0) + value
        grant = getattr(self, "_pending_commit", {}).get(request_id) if actual_allow else None
        source_index = getattr(self, "_study_log_event_index", 0)
        self._log("study_candidate_counterfactual", candidate_index=candidate_index,
            candidate_binding=f"{self._step}:{request_id}:{candidate_index}",
            request_id=request_id, external_request_id=external_request_id,
            action_id=action_id, proposal_version=proposal_version,
            admission_decision=admission_decision.summary() if admission_decision is not None else None,
            admission_policy_defer=bool(admission_policy_defer),
            actual_allow=bool(actual_allow), actual_grant_id=grant.grant_id if grant is not None else None,
            physical_fit=True, projected_post=state,
            projected_post_sha256=hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            shared_certificate=shared, exclusive_certificate=private,
            sum_demand_certificate=full_reserve,
            shared_closure_ranks=list(shared_order), exclusive_closure_ranks=list(private_order),
            diagnostic_oracle_calls=2,
            scope="projected post-state after actual decision and before allocator dispatch; not an observed allocation or admission-trial population",
            source_candidate_event_index=source_index)

    def _online_proposal(self, request, external_id, action_id, state_version, before, post, ids,
                         growth_pages, cached_tokens, new_admission):
        cohort = []
        for i, rid in enumerate(ids):
            current = request if rid == request.request_id else self.requests[rid]
            match = re.match(r"^(w[0-9]+-r[0-9]+)(?:-|$)", rid)
            contract = self._proposals.get(match.group(1) if match else rid, {})
            cohort.append(ObservedRequest(rid, current.num_prompt_tokens,
                len(current.output_token_ids), current.max_tokens,
                sum(g.count for g in post.groups if i in g.owners), post.need[i][0],
                contract.get("category", "default")))
        config = self.scheduler_config
        local_tokens = min(request.num_prompt_tokens, cached_tokens +
            len(self._pm.req_to_blocks.get(request.request_id, ())) * self._pm.block_size)
        device = RouteDevice(self._authority.worker_id, "default",
            compute_pressure=self._last_batch_tokens / max(1, config.max_num_batched_tokens),
            memory_pressure=(before.capacity[0]-before.free[0]) / before.capacity[0],
            queue_pressure=len(self.waiting) / max(1, config.max_num_seqs),
            locality=local_tokens / max(1, request.num_prompt_tokens),
            free_capacity=before.free[0], metadata={"pool_id": self._authority.pool_id,
                "state_version": state_version, "compute_source": "previous scheduled batch tokens"})
        value = self._online.propose(request.request_id, action_id, state_version, cohort,
            capacity=before.capacity[0], free_before=before.free[0], free_after=post.free[0],
            growth_pages=growth_pages, device=device, pool_id=self._authority.pool_id,
            new_admission=new_admission)
        if value["route"] and value["route"]["device_id"] != self._authority.worker_id:
            raise AssertionError("route does not identify the dispatch worker")
        self._log("proposal_binding", external_request_id=external_id,
            route_device=value["route"]["device_id"] if value["route"] else None, **value)
        return value

    def _peer_skip_rejected(self, request_queue, step_skipped_waiting, request):
        # Only a freshly recorded peer-filter refusal can take this path.
        # A capacity, certificate, drain or contract refusal keeps native break.
        if not getattr(self,"_peer_queue_bypass",False) or getattr(self,"_peer_last_deferred_id",None)!=request.request_id:
            return False
        if (self.policy.value!="fcfs" or request.has_encoder_inputs
                or self.connector is not None or self.ec_connector is not None
                or self.lora_config is not None):
            raise RuntimeError("peer queue bypass supports plain text FCFS only")
        authority=self._authority
        if (authority.pending is not None or authority.virtual!=authority.confirmed
                or request.request_id in authority.ids):
            raise RuntimeError("peer queue transfer requires settled nonresident state")
        if not(request_queue is self.waiting or request_queue is self.skipped_waiting):
            raise RuntimeError("unknown native waiting queue")
        if (step_skipped_waiting is self.waiting or step_skipped_waiting is self.skipped_waiting
                or request_queue.peek_request() is not request):
            raise RuntimeError("peer transfer must leave current-pass selection queues")
        status=request.status.name
        if status not in ("WAITING","PREEMPTED"):
            raise RuntimeError("unsupported peer transfer request status")
        before={}
        if self._audit:
            pack=lambda q:[[r.request_id,r.max_tokens]for r in q]
            before=dict(queue_before=pack(request_queue),step_skipped_before=pack(step_skipped_waiting))
        removed=request_queue.pop_request()
        if removed is not request:raise RuntimeError("native queue identity changed")
        step_skipped_waiting.prepend_request(request)
        self._peer_queue_bypasses+=1
        after={}
        if self._audit:
            after=dict(queue_after=pack(request_queue),step_skipped_after=pack(step_skipped_waiting))
        self._log("peer_waiting_bypass",request_id=request.request_id,cap=request.max_tokens,
            transfer_serial=self._peer_queue_bypasses,authority_version=authority.version,
            source_queue="waiting"if request_queue is self.waiting else"skipped_waiting",
            native_status=status,peer_proof_serial=self._peer_gate_checks,
            scope="native defer/restore only; no grant or physical allocation",**before,**after)
        self._peer_last_deferred_id=None
        return True

    def _prc_allow(
        self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_external_computed_tokens=0,
        num_lookahead_tokens=0,
        scheduled_tokens=None,
    ):
        start = time.perf_counter_ns()
        self._peer_last_deferred_id=None
        # A finite-step trigger freezes resident membership. Serial mode serves
        # only its captured head; batch modes require an actually allocated
        # positive head budget before followers. API-queued jobs are excluded.
        active_join=getattr(self,"_active_join",None) if getattr(self,"_active_drain_admission",False) else None
        if active_join is not None:
            previously_bound=request.request_id in active_join.bound.values()
            external=active_join.bind(request)
            if not previously_bound:
                contract=active_join.rows[external]
                self._log("finite_input_binding",request_id=request.request_id,external_request_id=external,
                    universe_sha256=active_join.sha256,**{key:contract[key] for key in ("prompt_token_sha256","prompt_tokens","max_tokens")})
            if self._drain.active:active_join.observe_accumulator(request,scheduled_tokens)
        joining=(active_join is not None and self._drain.active and request.request_id not in self._authority.ids)
        may_join=joining and active_join.may_join(request,scheduled_tokens)
        if getattr(self, "_drain_enabled", getattr(self, "_mode", "native_shadow") in ("shared_closure", "shared_incremental")) and not self._drain.allows(request.request_id) and not may_join:
            self._log("drain_defer", request_id=request.request_id)
            return False
        limit = getattr(self,"_peer_resident_limit",0)
        authority = getattr(self,"_authority",None)
        if limit and authority is not None and request.request_id not in authority.ids:
            if authority.pending is not None or authority.virtual != authority.confirmed:
                raise RuntimeError("peer admission gate requires settled resource rights")
            context = AdmissionContext(request.request_id,authority.version,request.max_tokens,
                tuple(sorted((rid,self.requests[rid].max_tokens)for rid in authority.ids)))
            gate = peer_admission_decision(limit,context)
            adaptive=getattr(self,'_peer_capacity_adaptive',False)
            if adaptive and not gate['allows']:
                state=authority.virtual
                if state.p!=1 or state.n!=len(authority.ids):
                    raise RuntimeError('adaptive pair funding requires the reviewed single settled pool')
                needs=tuple(sorted((rid,state.need[i][0])for i,rid in enumerate(authority.ids)))
                upper=(min(request.num_prompt_tokens+request.max_tokens,self.max_model_len)+self._pm.block_size-1)//self._pm.block_size
                gate=pair_funding_exemption(gate,state.free[0],needs,upper)
                self._peer_pair_checks+=1
                self._peer_pair_exemptions+=int(gate['capacity_exemption'])
            gate['adaptive_enabled']=adaptive
            self._peer_gate_checks += 1
            self._peer_gate_deferrals += int(not gate["allows"])
            self._log("peer_admission_gate",proof_serial=self._peer_gate_checks,
                pending_grant=False,resource_states_equal=True,actual_cap_source="native Request.max_tokens",
                scope="utility filter before certificate trial; pass does not authorize allocation",**gate)
            if not gate["allows"]:
                self._peer_last_deferred_id=request.request_id
                return False
        frozen_batch = (getattr(self, "_drain_enabled", False) and self._drain.active
                        and getattr(self._drain, "batching", "serial") in ("witness_preserving", "cohort_certified", "head_funded"))
        if frozen_batch and num_new_tokens <= 0:
            self._log("drain_defer", request_id=request.request_id, reason="no_positive_token_budget")
            return False
        if frozen_batch and request.request_id != self._drain.order[0]:
            # This dictionary is native schedule's actual successfully allocated
            # tokens, not an earlier permit or proposal. No follower-only batch.
            if scheduled_tokens is None or scheduled_tokens.get(self._drain.order[0], 0) <= 0:
                self._log("drain_defer", request_id=request.request_id,
                          reason="head_not_successfully_scheduled", episode=self._drain.episodes)
                return False
        if num_external_computed_tokens or num_lookahead_tokens:
            raise RuntimeError("External/speculative growth is outside this adapter")
        external_id = request.request_id
        match = re.match(r"^(w[0-9]+-r[0-9]+)(?:-|$)", external_id)
        if match:
            external_id = match.group(1)
        proposal = getattr(self, "_proposals", {}).get(external_id)
        if (
            proposal is None
            and getattr(self, "_pipeline_enabled", False)
            and request.request_id.startswith("w")
            and "-r" in request.request_id
            and self._mode in ("shared_closure", "shared_incremental")
        ):
            raise AssertionError("controlled request has no PRC proposal")
        if proposal is not None:
            expected_pages = (
                request.num_prompt_tokens + request.max_tokens + self._pm.block_size - 1
            ) // self._pm.block_size
            if proposal.get("hard_max_output_tokens") != request.max_tokens:
                raise AssertionError("proposal hard cap differs from request contract")
            if proposal.get("hard_need_pages") != expected_pages:
                raise AssertionError("proposal hard page envelope differs from request contract")
            if proposal.get("route", {}).get("device_id") != "gpu0":
                raise AssertionError("route is not the actual single GPU worker")
            if tuple(proposal.get("placement", {}).get("devices", ())) != ("gpu0",):
                raise AssertionError("placement is not installed on the actual worker")
        bs = self._pm.block_size
        cached = list(new_computed_blocks.blocks[0]) if new_computed_blocks is not None else []
        if num_new_computed_tokens % bs or len(cached) * bs != num_new_computed_tokens:
            raise RuntimeError("Only full-block prefix hits supported")
        current = list(self._pm.req_to_blocks.get(request.request_id, ()))
        if cached and current:
            raise RuntimeError("Unexpected prefix append to a resident request")
        # CCMB zero-growth fast path (2026-10-12). A resident request that needs no
        # new physical page this step leaves (f, d, groups) unchanged, so every
        # certificate/reserve decision equals the one that admitted its current
        # pages (T5 with z = 0). Skipping the snapshot here removes the per-step,
        # per-request O(pages) cost that dominated the archived comparisons.
        if getattr(self, "_zero_growth_fast", True) and current and not cached:
            target0 = request.num_computed_tokens + num_new_tokens
            required0 = (min(target0, self.max_model_len) + bs - 1) // bs
            if required0 - len(current) <= 0:
                self.__dict__.setdefault("_zero_growth_pass", set()).add(request.request_id)
                self._zero_growth_hits = getattr(self, "_zero_growth_hits", 0) + 1
                if self._zero_growth_hits % 1000 == 1:
                    self._log("zero_growth_fast", request_id=request.request_id, hits=self._zero_growth_hits,
                              controller_ns=time.perf_counter_ns() - start)
                return True
        self.__dict__.setdefault("_zero_growth_pass", set()).discard(request.request_id)  # a stale fast-path mark must never bypass a real growth
        deny_key = (len(current), len(cached), num_new_tokens, num_new_computed_tokens)
        if getattr(self, "_deny_cache_enabled", False) and self._mode != "native_shadow":
            cached_denial = self._deny_cache.get(request.request_id)
            if cached_denial == (self._release_epoch, deny_key):
                self._deny_cache_hits += 1
                if self._deny_cache_hits % 1000 == 1:
                    self._log("deny_cache_hit", request_id=request.request_id, hits=self._deny_cache_hits,
                              release_epoch=self._release_epoch, controller_ns=time.perf_counter_ns() - start)
                return False
        target = request.num_computed_tokens + num_new_computed_tokens + num_new_tokens
        required = (min(target, self.max_model_len) + bs - 1) // bs
        authority = getattr(self, "_authority", None)
        reuse_enabled = getattr(self, "_zero_projection_reuse", False)
        if reuse_enabled:
            self._zero_projection_checks += 1
        # No authorization is inferred from z=0. The current state must still
        # be observed; it is only reused as a projection if the Authority is
        # settled and the exact resident identities/version remain unchanged.
        reuse_version = authority.version if authority is not None else None
        reuse_candidate = (reuse_enabled and current and not cached
            and num_new_computed_tokens == 0 and required <= len(current)
            and authority is not None and authority.pending is None
            and authority.virtual == authority.confirmed
            and request.request_id in authority.ids
            and not any(getattr(self, name, {}) for name in
                ("_pending_commit", "_pending_projection", "_pending_index")))
        s, ids, owners = self._snapshot() if reuse_candidate else self._snapshot(request)
        i = ids.index(request.request_id)
        reuse_projection = (reuse_candidate and authority.version == reuse_version
            and authority.pending is None and authority.virtual == authority.confirmed
            and tuple(ids) == authority.ids and s == authority.confirmed)
        if reuse_projection:
            # State/Group are deeply immutable. A resident, no-prefix, zero-page
            # allocation changes neither free/need nor exact physical owners.
            # Keep the ordinary proposal, grant, dispatch and post-observation
            # path below; this branch only omits redundant object construction.
            post = s
            touched_free = new_count = 0
            new_free = s.free[0]
            physical_fit = True
            self._zero_projection_hits += 1
        else:
            projected = {g.key: set(g.owners) for g in s.groups}
            touched_free = 0
            for block in cached:
                if block.is_null:
                    raise RuntimeError("Null cache hit")
                if str(block.block_id) not in projected:
                    projected[str(block.block_id)] = set()
                    touched_free += 1
                projected[str(block.block_id)].add(i)
            new_count = max(0, required - len(current) - len(cached))
            for j in range(new_count):
                projected[f"prospective-{j}"] = {i}
            new_free = s.free[0] - touched_free - new_count
            needs = [list(row) for row in s.need]
            needs[i][0] -= len(cached) + new_count
            if needs[i][0] < 0:
                raise AssertionError("Action exceeds remaining hard bound")
            physical_fit = new_free >= 0
            post = (State((new_free,), tuple(map(tuple, needs)),
                tuple(Group(key, 0, 1, frozenset(who)) for key, who in projected.items()))
                if physical_fit else None)
        certificate = False
        oracle_checked = False
        if physical_fit:
            if reuse_projection and self._audit:
                # Differentially reconstruct the former projection in audit
                # mode. This is a proof check, not formal timing evidence.
                original_post = State((s.free[0],), tuple(tuple(row) for row in s.need),
                    tuple(Group(g.key, 0, 1, frozenset(g.owners)) for g in s.groups))
                if original_post != post:
                    raise AssertionError("zero projection reuse differs from original projection")
                self._zero_projection_audit_checks += 1
                self._log("zero_projection_reuse", request_id=request.request_id,
                    authority_version=reuse_version, resident_ids=list(ids),
                    free=s.free[0], growth_pages=0, projection_identity_reused=True,
                    original_projection_equal=True, authorization_bypass=False)
            if authority is None and (self._mode != "shared_incremental" or self._audit):
                certificate_state = post
                if self._mode == "exclusive_closure":
                    certificate_state = State(
                        post.free,
                        post.need,
                        tuple(
                            Group(g.key, g.pool, g.count, g.owners, g.pinned or len(g.owners) > 1)
                            for g in post.groups
                        ),
                    )
                certificate = len(indexed_closure(certificate_state)) == post.n
                oracle_checked = True
            reserve_fit = sum(row[0] for row in post.need) <= new_free
        else:
            post, reserve_fit = None, False
        frozen_pair = None
        if frozen_batch and self._drain.batching == "witness_preserving" and physical_fit:
            if authority is None:
                raise RuntimeError("witness-preserving batching requires the allocation authority")
            variant = authority.controller.certificate_variant
            frozen_pair = fixed_witness_pair(post, ids, self._drain.order,
                conservative=variant in ("common_full_c0", "exclusive_only"))
            self._drain_fixed_checks = getattr(self, "_drain_fixed_checks", 0) + 1
            self._drain_c0_checks = getattr(self, "_drain_c0_checks", 0) + 1
            self._drain_c1_checks = getattr(self, "_drain_c1_checks", 0) + 1
        frozen_allowed = frozen_pair is None or frozen_pair["active"]["accepted"]
        head_funding = None
        head_funded = frozen_batch and self._drain.batching == "head_funded"
        if head_funded and physical_fit:
            if authority is None:
                raise RuntimeError("head funding requires the original allocation Authority")
            selected = self._drain.selected_head
            if selected is None or selected not in ids or selected != self._drain.order[0]:
                raise RuntimeError("funding guard lacks a persistent selected head")
            h = ids.index(selected)
            margin = tuple(post.free[p]-post.need[h][p] for p in range(post.p))
            self._drain_head_funding_checks = getattr(self,"_drain_head_funding_checks",0) + 1
            head_funding = dict(selected_head=selected, post_free=post.free,
                head_need=post.need[h], margin=margin, accepted=all(v>=0 for v in margin))
            self._drain_head_funding_deferrals = getattr(self,"_drain_head_funding_deferrals",0) + int(not head_funding["accepted"])
            if request.request_id == selected and not head_funding["accepted"]:
                raise RuntimeError("selected head own growth lost funding invariant")
        head_funding_allowed = head_funding is None or head_funding["accepted"]
        allow = self._mode == "native_shadow" or (
            physical_fit
            and (reserve_fit if self._mode == "hard_reserve" else certificate)
        )
        if self._mode == "shared_incremental" and physical_fit and authority is None:
            fast_started = time.perf_counter_ns()
            eligible = (
                self._index is not None and self._index_state == s and self._index_ids == tuple(ids) and not cached
            )
            fixed = interval_trial(self._index, [(i, 0, new_count)]) if eligible else False
            if not oracle_checked:
                if fixed:
                    certificate = True  # Validated cached witness, not a guessed oracle result.
                else:
                    certificate = len(indexed_closure(post)) == post.n
                    oracle_checked = True
            decision = fixed or certificate
            if decision != certificate:
                raise AssertionError("Incremental plus complete fallback disagrees with full closure")
            if decision:
                if fixed:
                    self._pending_index[request.request_id] = ("private", (i, new_count))
                else:
                    order = indexed_closure(post)
                    self._pending_index[request.request_id] = ("rebuild", LinearBuildMarginIndex(post, order))
            self._log(
                "incremental_proposal",
                request_id=request.request_id,
                eligible=eligible,
                fixed_accepted=fixed,
                full_accepted=certificate,
                complete_fallback=eligible and not fixed,
                structural_rebuild=not eligible,
                equivalent=True if oracle_checked else None,
                oracle_checked=oracle_checked,
                arithmetic_and_build_ns=time.perf_counter_ns() - fast_started,
            )
            allow = decision
        action_id = proposal_version = None
        admission_decision = None
        admission_policy_defer = False
        if authority is not None and physical_fit:
            action_id = f"{request.request_id}:a{self._serial}"
            self._serial += 1
            proposal_version = authority.version
            online = None
            if getattr(self, "_online", None) is not None:
                online = self._online_proposal(request, external_id, action_id, proposal_version,
                    s, post, ids, touched_free+new_count, num_new_computed_tokens,
                    request.request_id not in authority.ids)
            online_allowed = online is None or online["proposal_allowed"]
            proposal_allowed = online_allowed and frozen_allowed and head_funding_allowed
            growth = None
            # Prefix/cohort changes always use a complete structural transaction.
            if tuple(ids) == authority.ids and not cached and self._mode == "shared_incremental":
                growth = tuple((new_count,) if k == i else (0,) for k in range(len(ids)))
            admission_context = None
            if (request.request_id not in authority.ids
                    and authority.controller.certificate_variant in ("shared_incremental", "common_full_c0", "common_full_c1", "force_full")):
                admission_context = AdmissionContext(request.request_id, proposal_version, request.max_tokens,
                    tuple(sorted((rid, self.requests[rid].max_tokens) for rid in authority.ids)))
            prior_stats = dict(authority.controller.stats)
            if proposal_allowed:
                try:
                    # The exact version carried by the online action is certified;
                    # never replace a stale proposal version with a fresh value.
                    grant = authority.authorize(action_id, request.request_id, post, ids,
                        expected_version=proposal_version, private_growth=growth, admission_context=admission_context)
                except AdmissionDeferred as exc:
                    admission_decision = exc.decision
                    admission_policy_defer = True
                    allow, certificate = False, True
                except Refused as exc:
                    admission_decision = getattr(exc, "admission_decision", None)
                    allow = certificate = False
                else:
                    allow = certificate = True
                    self._pending_commit[request.request_id] = grant
                    admission_decision = grant.admission_decision
            else:
                allow = certificate = False
            stats = authority.controller.stats
            oracle_checked = stats["oracle"] > prior_stats["oracle"]
            self._log("incremental_proposal", request_id=request.request_id,
                eligible=growth is not None, fixed_accepted=stats["fast"] > prior_stats["fast"],
                full_accepted=certificate, complete_fallback=stats["fallback"] > prior_stats["fallback"],
                structural_rebuild=growth is None, equivalent=True if oracle_checked else None,
                oracle_checked=oracle_checked, statistical_defer=not online_allowed,
                frozen_witness_defer=not frozen_allowed,
                admission_policy_defer=admission_policy_defer,
                head_funding_defer=not head_funding_allowed,
                authority_version=proposal_version, arithmetic_and_build_ns=0)
            if admission_decision is not None:
                self._log("admission_policy_decision", request_id=request.request_id,
                    action_id=action_id, proposal_version=proposal_version,
                    active_certificate=certificate, actual_allow=allow, policy_deferred=admission_policy_defer,
                    grant_id=getattr(getattr(self, "_pending_commit", {}).get(request.request_id), "grant_id", None),
                    decision=admission_decision.summary(),
                    cap_source="actual native Request.max_tokens; adapter attestation, not recovered from page needs",
                    scope="utility choice on new resident only; not certificate strength or a wall-time guarantee")
        if allow and physical_fit:
            self._pending_projection[request.request_id] = (post, ids)
        if frozen_pair is not None:
            record = dict(episode=self._drain.episodes, batch_index=getattr(self, "_drain_batch_index", 0),
                frozen_order=list(self._drain.order), head=self._drain.order[0],
                request_id=request.request_id, action_id=action_id, proposal_version=proposal_version,
                grant_id=getattr(getattr(self, "_pending_commit", {}).get(request.request_id), "grant_id", None),
                head_tokens_scheduled=(scheduled_tokens or {}).get(self._drain.order[0], 0),
                active_view=frozen_pair["active"]["view"], fixed_accepted=frozen_allowed,
                actual_allow=allow, c0_min_margin=frozen_pair["c0"]["min_margin"],
                c1_min_margin=frozen_pair["c1"]["min_margin"],
                c0_fixed_accepted=frozen_pair["c0"]["accepted"],
                c1_fixed_accepted=frozen_pair["c1"]["accepted"],
                reason="frozen_witness_valid" if frozen_allowed else "frozen_witness_negative_margin",
                authority_order=[authority.ids[k] for k in authority.controller.order] if authority is not None else None)
            if self._audit:
                classes = defaultdict(int)
                for g in post.groups:
                    classes[(g.pool, tuple(sorted(g.owners)), g.pinned)] += g.count
                record.update(post=dict(free=post.free, need=post.need, cohort=ids,
                    owner_classes=[dict(pool=p, owner_ranks=who, pinned=pin, pages=count)
                        for (p, who, pin), count in sorted(classes.items())]),
                    c0_margins=frozen_pair["c0"]["margins"], c1_margins=frozen_pair["c1"]["margins"])
            self._log("drain_frozen_witness_check", **record)
        if head_funding is not None:
            record=dict(episode=self._drain.episodes,batch_index=self._drain_batch_index,
                head=self._drain.selected_head,request_id=request.request_id,action_id=action_id,
                proposal_version=proposal_version,frozen_survivors=list(self._drain.order),
                head_tokens_scheduled=(scheduled_tokens or {}).get(self._drain.selected_head,0),
                post_cohort=list(ids),**head_funding,
                actual_allow=allow,active_certificate=certificate,
                authority_checked=head_funding_allowed,
                certificate_status=("not_evaluated_head_funding_defer" if not head_funding_allowed else "certified" if certificate else "refused_active_view"),
                grant_id=getattr(getattr(self,"_pending_commit",{}).get(request.request_id),"grant_id",None),
                guard_cost_scope="O(P) funding arithmetic after O(n) identity lookup; snapshot/Authority/log/model costs additional")
            if self._audit:
                classes=defaultdict(int)
                for g in post.groups:classes[(g.pool,tuple(sorted(g.owners)),g.pinned)]+=g.count
                state=dict(free=list(post.free),need=[list(row) for row in post.need],cohort=list(ids),
                    owner_classes=[dict(pool=p,owner_ranks=list(who),pinned=pin,pages=count)
                        for (p,who,pin),count in sorted(classes.items())])
                record.update(projected_post=state,projected_post_sha256=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(",", ":")).encode()).hexdigest())
            self._log("drain_head_funding_check",**record)
        if self._audit and physical_fit:
            self._study_candidate_counterfactual(post, ids, request_id=request.request_id,
                external_request_id=external_id, action_id=action_id,
                proposal_version=proposal_version, actual_allow=allow,
                admission_decision=admission_decision, admission_policy_defer=admission_policy_defer)
        self._log(
            "proposal",
            request_id=request.request_id,
            allow=allow,
            physical_fit=physical_fit,
            certificate=certificate,
            admission_policy_defer=admission_policy_defer,
            head_funding_defer=not head_funding_allowed,
            full_reserve_fit=reserve_fit,
            exclusive_certificate=certificate if self._mode == "exclusive_closure" else None,
            free_before=s.free[0],
            free_after=new_free,
            new_pages=new_count,
            reused_blocks=len(cached),
            touched_free=touched_free,
            controller_ns=time.perf_counter_ns() - start,
        )
        if getattr(self, "_deny_cache_enabled", False):
            if allow:
                self._deny_cache.pop(request.request_id, None)
            else:
                self._deny_cache[request.request_id] = (self._release_epoch, deny_key)
        return allow

    def schedule(self, *args, **kwargs):
        prof = getattr(self, "_profiler", None)
        if prof is not None:
            prof.enable()
        try:
            schedule_started = time.perf_counter_ns()
            self._step += 1
            if getattr(self, "_deny_cache_enabled", False):
                # Generic release detector: any increase in free pages (completion,
                # cancellation, cache reset) opens a new epoch for cached refusals.
                free_now = self.kv_cache_manager.block_pool.get_num_free_blocks()
                if free_now > getattr(self, "_last_free_seen", -1):
                    self._release_epoch += 1
                    self._deny_cache.clear()
                self._last_free_seen = free_now
            if getattr(self._drain, "scope", "cohort") == "first_completion":
                # Completion can close the segment inside model callbacks. It
                # cannot open the next admission cut before all old rights and
                # physical projections settle, even though active is now false.
                if any(getattr(self, name, {}) for name in
                       ("_pending_commit", "_pending_projection", "_pending_index")):
                    raise RuntimeError("first-completion admission cut has unsettled resource rights/projections")
            if getattr(self._drain, "batching", "serial") in ("cohort_certified", "head_funded"):
                if (getattr(self, "_drain_pending_batches", {})
                        or getattr(self, "_execution_batches", {})):
                    raise RuntimeError("cannot refresh cohort witness before prior model batch confirmation")
            if getattr(self,"_ordinary_witness_priority",False) or getattr(self,"_active_drain_admission",False):
                # The timer may activate at this cut. Settlement must precede
                # observe for ordinary, trigger and already-active steps alike.
                if (any(getattr(self,name,{}) for name in ("_ordinary_pending_batches","_drain_pending_batches",
                        "_execution_batches","_pending_commit","_pending_projection","_pending_index"))
                        or getattr(getattr(self,"_execution_meter",None),"pending",{})):
                    raise RuntimeError("ordinary priority requires fully settled prior model and resource rights")
            if getattr(self,"_active_drain_admission",False) and self._active_join.epoch is not None:
                raise RuntimeError("active admission requires previous captured epoch fully settled")
            s, ids, owners = self._snapshot()
            authority = getattr(self, "_authority", None)
            if authority is not None:
                witness = authority.drain_cut()
                if tuple(ids) != authority.ids:
                    raise AssertionError("physical cohort differs from authority at schedule boundary")
                order = tuple(ids.index(rid) for rid in witness)
                if self._audit and len(indexed_closure(s)) != s.n:
                    raise AssertionError("authority witness disagrees with physical oracle")
            elif (
                not self._audit
                and self._mode == "shared_incremental"
                and self._index is not None
                and self._index_state == s
                and self._index_ids == tuple(ids)
            ):
                order = self._index.order
            else:
                order = indexed_closure(s)
            if getattr(self, "_rank_order", False) and ids and len(order) == s.n and len(self.running) > 1:
                # T9: grant growth in witness order. Stable sort keeps native order among
                # requests outside the certified cohort (e.g. not yet resident).
                rank = {ids[i]: k for k, i in enumerate(order)}
                before = [r.request_id for r in self.running]
                self.running.sort(key=lambda r: rank.get(r.request_id, len(rank)))
                after = [r.request_id for r in self.running]
                if before != after:
                    self._rank_reorders += 1
                    self._log("rank_reorder", witness=[ids[i] for i in order], before=before, after=after)
            if getattr(self, "_drain_enabled", getattr(self, "_mode", "native_shadow") in ("shared_closure", "shared_incremental")) and self._drain.observe(ids, order):
                self._log(
                    "drain_start",
                    cohort=list(self._drain.order),
                    drain_scope=self._drain.scope,
                    protected_scope="ACK-expandable finite resident cohort" if getattr(self,"_active_drain_admission",False) else "resident requests; waiting API requests excluded",
                )
            active_drain = getattr(self, "_drain_enabled", False) and self._drain.active
            ordinary_proof = None
            if getattr(self,"_ordinary_witness_priority",False) and not active_drain:
                if getattr(self,"_rank_order",False):
                    raise RuntimeError("ordinary priority and legacy rank cannot both be enabled")
                if authority is None or authority.pending is not None or authority.virtual != authority.confirmed or len(order) != s.n:
                    raise RuntimeError("ordinary priority requires a settled complete Authority witness")
                rank = {ids[i]: k for k,i in enumerate(order)}
                before = [r.request_id for r in self.running]
                proposed = sorted(before,key=lambda rid:rank.get(rid,len(rank)))
                self.running.sort(key=lambda r:rank.get(r.request_id,len(rank)))
                after = [r.request_id for r in self.running]
                changed = before != after
                self._ordinary_steps = getattr(self,"_ordinary_steps",0)+1
                self._ordinary_reorders = getattr(self,"_ordinary_reorders",0)+int(changed)
                head = ids[order[0]] if order else None
                ordinary_proof = dict(authority_version_at_cut=authority.version,
                    resident_ids=list(ids),witness=[ids[i] for i in order],head=head,
                    prior_running=before,proposed_running=proposed,reordered_running=after,
                    relative_order_changed=changed,head_running_count=before.count(head) if head is not None else 0,
                    drain_active_at_cut=self._drain.active,
                    pending_grant=authority.pending is not None,resource_states_equal=authority.virtual == authority.confirmed,
                    pending_model_batches=len(getattr(self,"_execution_batches",{})),
                    pending_meter_batches=len(self._execution_meter.pending),
                    policy="ordinary_only_reuse_complete_witness_rank_sort")
                self._log("ordinary_witness_priority",**ordinary_proof)
            frozen_batch = active_drain and getattr(self._drain, "batching", "serial") in ("witness_preserving", "cohort_certified", "head_funded")
            epoch_certified = active_drain and self._drain.batching == "cohort_certified"
            funded_segment = active_drain and self._drain.batching == "head_funded"
            order_semantics = (ACTIVE_ADMISSION_SEMANTICS if getattr(self,"_active_drain_admission",False) and epoch_certified else "current_epoch_authority_witness; membership_frozen_until_confirmed_terminal"
                if epoch_certified else "persistent_selected_head; frozen_survivor_order_not_a_retained_witness" if funded_segment else "captured_cohort_witness")
            batch_head = self._drain.order[0] if active_drain else None
            if active_drain:
                self._drain_batch_index = getattr(self, "_drain_batch_index", 0) + 1
            if getattr(self,"_active_drain_admission",False) and active_drain:
                self._active_join.begin(episode=self._drain.episodes,batch=self._drain_batch_index,head=batch_head,
                    members=list(self._drain.order),version=authority.version)
            if epoch_certified:
                self._log("drain_epoch_witness", episode=self._drain.episodes,
                    batch_index=self._drain_batch_index, head=batch_head,
                    epoch_order=list(self._drain.order), cohort_members=sorted(self._drain.order),
                    authority_version=authority.version, pending_grant=False, prior_model_batch_pending=False,
                    order_semantics=order_semantics)
            if funded_segment:
                if self._drain.selected_head != batch_head or batch_head not in ids:
                    raise RuntimeError("selected head missing at settled cut")
                h=ids.index(batch_head)
                margin=tuple(s.free[p]-s.need[h][p] for p in range(s.p))
                if any(v<0 for v in margin):raise RuntimeError("persistent head is no longer funded at settled cut")
                self._log("drain_head_funded_epoch",episode=self._drain.episodes,batch_index=self._drain_batch_index,
                    head=batch_head,selected_head=batch_head,frozen_survivors=list(self._drain.order),
                    authority_witness=list(witness),authority_version=authority.version,
                    post_free=s.free,head_need=s.need[h],margin=margin,
                    head_selection_cause=self._drain.last_selection or "retained",
                    head_replacement=self._drain.last_selection == "selected_head_cancelled",
                    pending_grant=False,prior_model_batch_pending=False,order_semantics=order_semantics)
            if frozen_batch:
                before = [r.request_id for r in self.running]
                try:
                    self.running[:] = head_first_running(self.running, batch_head)
                except RuntimeError:
                    self._log("drain_head_unavailable", head=batch_head, running=before,
                              reason="head_not_uniquely_running", episode=self._drain.episodes)
                    raise
                self._log("drain_head_first", head=batch_head, before=before,
                          after=[r.request_id for r in self.running], episode=self._drain.episodes,
                          batch_index=self._drain_batch_index, frozen_order=list(self._drain.order),
                          order_semantics=order_semantics)
            if self._audit:
                self._log(
                    "snapshot", free=s.free[0], capacity=s.capacity[0], cohort=ids,
                    hard_need=[row[0] for row in s.need],
                    request_phases=[
                        {"request_id": rid, "prompt_tokens": self.requests[rid].num_prompt_tokens,
                         "max_output_tokens": self.requests[rid].max_tokens,
                         "num_computed_tokens": self.requests[rid].num_computed_tokens,
                         "output_tokens": len(self.requests[rid].output_token_ids),
                         "prefill_complete": self.requests[rid].num_computed_tokens >= self.requests[rid].num_prompt_tokens,
                         "allocated_pages": len(self._pm.req_to_blocks[rid])} for rid in ids],
                    groups=[{"block": g.key, "owners": [ids[i] for i in g.owners]} for g in s.groups],
                    certificate=len(order) == s.n,
                    shared_pages=sum(len(g.owners) > 1 for g in s.groups),
                    preemptions=sum(r.num_preemptions for r in self.requests.values()))
            output = _instrumented_schedule(self, *args, **kwargs)
            scheduled = getattr(output, "num_scheduled_tokens", {})
            self._last_batch_tokens = sum(scheduled.values())
            if ordinary_proof is not None:
                ordinary_proof = dict(ordinary_proof,scheduled_tokens=dict(scheduled),
                    head_tokens=scheduled.get(ordinary_proof["head"],0),empty=not bool(scheduled),
                    scheduled_outside_initial_residents=sorted(set(scheduled)-set(ordinary_proof["resident_ids"])),
                    evidence_scope="actual SchedulerOutput budget; ordinary admissions remain open; model execution not yet credited")
                self._log("ordinary_witness_budget",**ordinary_proof)
                if scheduled:
                    self._ordinary_pending_batches[id(output)] = (output,ordinary_proof)
            if (epoch_certified or funded_segment) and not scheduled:
                raise RuntimeError("cohort-certified eligible epoch produced no actual scheduled work")
            if self._drain_enabled and self._drain.active and scheduled:
                if frozen_batch:
                    if scheduled.get(batch_head, 0) <= 0 or any(rid not in self._drain.order for rid in scheduled):
                        raise AssertionError("frozen batch lacks positive head budget or schedules outside cohort")
                elif any(rid != self._drain.order[0] for rid in scheduled):
                    raise AssertionError("drain scheduled a non-head request")
                self._log("drain_dispatch", request_id=self._drain.order[0],
                          scheduled_tokens=self._last_batch_tokens, witness_head=self._drain.order[0])
            if active_drain:
                proof=dict(episode=self._drain.episodes,
                    batch_index=self._drain_batch_index, head=batch_head,
                    batching=getattr(self._drain, "batching", "serial"),
                    frozen_order=(list(self._active_join.epoch["initial_members"]) if getattr(self,"_active_drain_admission",False) else list(self._drain.order)), scheduled_tokens=dict(scheduled),
                    order_semantics=order_semantics,
                    head_tokens=scheduled.get(batch_head, 0), nonhead_requests=sum(rid != batch_head for rid in scheduled),
                    empty=not bool(scheduled), progress_credited=False,
                    evidence_scope="scheduler budget after physical allocation; actual model result requires update/execute join")
                if getattr(self,"_active_drain_admission",False):proof.update(self._active_join.proof())
                self._log("drain_batch", **proof)
                if scheduled:
                    self.__dict__.setdefault("_drain_pending_batches", {})[id(output)] = (output, proof)
            elapsed_ns = time.perf_counter_ns() - schedule_started
            if getattr(self, "_execution_meter", None) is not None:
                self._execution_schedule_ns += elapsed_ns
                self._execution_schedule_calls += 1
            self._log("schedule_timing", elapsed_ns=elapsed_ns)
            return output
        finally:
            if prof is not None:
                prof.disable()

    def update_from_output(self, scheduler_output, model_runner_output):
        # Observation only. Preserve the original model-result/Authority/free
        # path exactly once; its return is after the synchronous engine writes.
        pending = self.__dict__.setdefault("_drain_pending_batches", {})
        saved = pending.get(id(scheduler_output))
        new_mode = saved is not None and saved[1].get("batching") in ("cohort_certified", "head_funded")
        meter = getattr(self, "_execution_meter", None)
        if new_mode and meter is None:
            raise RuntimeError("cohort batch lacks actual execution meter")
        before_executed = ({rid: meter.executed.get(rid, 0) for rid in saved[1]["scheduled_tokens"]}
                           if new_mode else {})
        before_reexecuted = ({rid: meter.reexecuted.get(rid, 0) for rid in saved[1]["scheduled_tokens"]}
                             if new_mode else {})
        ordinary_pending = getattr(self,"_ordinary_pending_batches",{})
        ordinary_saved = ordinary_pending.get(id(scheduler_output))
        ordinary_before = ({rid:(meter.executed.get(rid,0),meter.reexecuted.get(rid,0))
                            for rid in ordinary_saved[1]["scheduled_tokens"]} if ordinary_saved is not None else {})
        result = super().update_from_output(scheduler_output, model_runner_output)
        if ordinary_saved is not None:
            ordinary_output, ordinary = ordinary_saved
            if ordinary_output is not scheduler_output:
                raise AssertionError("ordinary model batch identity mismatch")
            observed = {rid:meter.executed.get(rid,0)-ordinary_before[rid][0] for rid in ordinary["scheduled_tokens"]}
            repeated = {rid:meter.reexecuted.get(rid,0)-ordinary_before[rid][1] for rid in ordinary["scheduled_tokens"]}
            if observed != ordinary["scheduled_tokens"] or any(not 0<=repeated[rid]<=observed[rid] for rid in observed):
                raise AssertionError("ordinary model-result observation differs from actual execution meter")
            unique = {rid:observed[rid]-repeated[rid] for rid in observed}
            del ordinary_pending[id(scheduler_output)]
            self._ordinary_confirmed_batches = getattr(self,"_ordinary_confirmed_batches",0)+1
            head_unique = unique.get(ordinary["head"],0)
            confirmed_proof = dict(ordinary,evidence_scope="actual synchronous model-result and execution-meter join; no ordinary positive-head guarantee")
            self._log("ordinary_witness_confirmed",**confirmed_proof,
                returned_request_ids=list(model_runner_output.req_ids),confirmed_executed_positions=observed,
                confirmed_reexecuted_positions=repeated,confirmed_new_unique_positions=unique,
                head_new_unique_positions=head_unique,
                head_positive_unique_execution=ordinary["head_running_count"]==1 and head_unique>0,
                physical_execution_observed=True)
        if saved is not None:
            output, proof = saved
            if output is not scheduler_output:
                raise AssertionError("drain execution batch identity mismatch")
            returned = set(model_runner_output.req_ids)
            if getattr(self,"_active_drain_admission",False):
                active=self._active_join;epoch=active.epoch
                if epoch is None or (epoch["episode"],epoch["batch_index"],epoch["head"])!=(proof["episode"],proof["batch_index"],proof["head"]):
                    raise AssertionError("active membership result lacks captured epoch")
                missing=set(proof["scheduled_tokens"])-returned
                trusted={rid for rid in epoch["cancelled"] if self._authority._terminals.get(rid)=="cancelled"}
                if not missing<=trusted:raise AssertionError("noncancelled active member missing from model result")
                observed={rid:(None if rid in missing else meter.executed.get(rid,0)-before_executed[rid]) for rid in proof["scheduled_tokens"]}
                if any(observed[rid]!=count for rid,count in proof["scheduled_tokens"].items() if rid not in missing):
                    raise AssertionError("active member real work differs from allocated budget")
                if any(meter.reexecuted.get(rid,0)!=before_reexecuted[rid] for rid in proof["scheduled_tokens"]):
                    raise AssertionError("active membership replayed execution positions")
                if proof["head"] not in missing and observed[proof["head"]]<=0:
                    raise AssertionError("active epoch head lacks positive unique confirmed work")
                if missing:
                    unknown=getattr(meter,"unobserved_cancelled_spans",[])
                    if not unknown or {r["request_id"] for r in unknown[-1]["unknown_spans"]}!=missing:
                        raise AssertionError("cancelled missing spans were not explicitly settled as UNKNOWN")
                del pending[id(scheduler_output)]
                self._drain_confirmed_batches=getattr(self,"_drain_confirmed_batches",0)+1
                self._log("drain_batch_confirmed",**{key:proof[key] for key in ("episode","batch_index","batching","head","frozen_order","scheduled_tokens","head_tokens","order_semantics","initial_members","final_members","join_grant_ids","universe_sha256")},
                    returned_request_ids=sorted(returned),confirmed_real_positions=observed,
                    confirmed_new_unique_positions=None if missing else sum(observed.values()),
                    observed_confirmed_positions=sum(value for value in observed.values() if value is not None),
                    actual_work_status="UNKNOWN_CANCELLED_SPAN" if missing else "EXACT_CONFIRMED",
                    unobserved_cancelled_request_ids=sorted(missing),trusted_cancelled_ids=sorted(trusted),
                    physical_execution_observed=not bool(missing),model_result_observed=True,
                    progress_credited=not bool(missing),surviving_members=list(self._drain.order),
                    evidence_scope="Actual model/meter join; missing trusted-cancelled span work UNKNOWN, never zero; such runs cannot pass exact-work/performance evidence")
                active.epoch=None
                self._drain._active_epoch_head=None
                return result
            if not set(proof["scheduled_tokens"]) <= returned:
                raise AssertionError("model result omits a drain scheduled request")
            observed_positions = ({rid: meter.executed.get(rid, 0) - before_executed[rid]
                                  for rid in proof["scheduled_tokens"]} if new_mode else None)
            if new_mode and (observed_positions != proof["scheduled_tokens"]
                    or sum(observed_positions.values()) <= 0
                    or any(meter.reexecuted.get(rid, 0) != before_reexecuted[rid]
                           for rid in proof["scheduled_tokens"])):
                raise AssertionError("cohort batch lacks positive unique confirmed execution positions")
            del pending[id(scheduler_output)]
            self._drain_confirmed_batches = getattr(self, "_drain_confirmed_batches", 0) + 1
            self._log("drain_batch_confirmed", episode=proof["episode"],
                batch_index=proof["batch_index"], batching=proof["batching"],
                head=proof["head"], frozen_order=proof["frozen_order"],
                scheduled_tokens=proof["scheduled_tokens"], returned_request_ids=sorted(returned),
                head_tokens=proof["head_tokens"], physical_execution_observed=True,
                order_semantics=proof.get("order_semantics", "captured_cohort_witness"),
                confirmed_real_positions=observed_positions,
                confirmed_new_unique_positions=sum(observed_positions.values()) if new_mode else None,
                evidence_scope="synchronous model-result return; not proof of successful request completion")
        return result

# Two narrow pre-allocation insertions, verified against the installed source.
# Preserve native engine code and preemption policy; all changes live here.
source = textwrap.dedent(inspect.getsource(Scheduler.schedule))
anchor_running = "        # Schedule newly needed KV blocks for the request.\n"
insert_running = (
    "        if not self._prc_allow(request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens, scheduled_tokens=num_scheduled_tokens):\n"
    "            req_index += 1\n"
    "            continue\n"
)
anchor_waiting = "            new_blocks = self.kv_cache_manager.allocate_slots(\n                request,\n                num_new_tokens,\n                num_new_computed_tokens=num_new_local_computed_tokens,"
insert_waiting = (
    "            if not self._prc_allow(request, num_new_tokens, num_new_computed_tokens=num_new_local_computed_tokens,\n"
    "                                   new_computed_blocks=new_computed_blocks, num_external_computed_tokens=num_external_computed_tokens,\n"
    "                                   num_lookahead_tokens=effective_lookahead_tokens, scheduled_tokens=num_scheduled_tokens):\n"
    "                if getattr(self, \"_peer_queue_bypass\", False) and self._peer_skip_rejected(request_queue,step_skipped_waiting,request):\n"
    "                    continue\n"
    "                break\n"
)
if source.count(anchor_running) != 1 or source.count(anchor_waiting) != 1:
    raise RuntimeError("Scheduler source changed; review adapter before running")
# The installed native defer path must keep this pass separate and restore it.
if (source.count("step_skipped_waiting = create_request_queue(self.policy)")!=1
        or source.count("self.skipped_waiting.prepend_requests(step_skipped_waiting)")!=1):
    raise RuntimeError("Native waiting defer/restore changed; review before execution")
source = source.replace(anchor_running, insert_running + anchor_running)
source = source.replace(anchor_waiting, insert_waiting + anchor_waiting)
namespace = dict(vars(base_module))
exec(compile(source, "<PRC-audited-schedule-wrapper>", "exec"), namespace)
_instrumented_schedule = namespace["schedule"]

