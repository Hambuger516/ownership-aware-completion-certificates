"""Online PRC proposals from observed progress and current allocator telemetry.

The statistical gate is the GateServe mean/variance proposal (main_new.tex,
eq. U_n^+). It gates NEW admissions only. Empirical moments are not a tail
certificate; the hard completion controller must still authorize every action.
All memory quantities below are KV pages, with model weights excluded from
both the available pool capacity and the occupancy. Shared pages count once.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

from .patch_exact import PatchExactSolver, PatchGraph, PatchQuery, PatchStage
from .prediction import PredictionQuery
from .routing import NoFeasibleRoute, RouteRequest, TypedRoute


@dataclass(frozen=True)
class ObservedRequest:
    request_id: str
    prompt_tokens: int
    observed_output_tokens: int
    max_output_tokens: int
    projected_pages: int
    hard_remaining_pages: int
    category: str = "default"


def single_worker_graph(hard_pages, capacity, worker_id="gpu0"):
    # A deployed one-stage model has one possible placement. Zero latency/cost
    # labels and identity availability denote *unprofiled structural checks*,
    # never measured speed, price, reliability, or heterogeneous benefit.
    stage = PatchStage("deployed-model", worker_id, 0, 0, 0., 0., hard_pages, 0., 1.)
    return PatchGraph((stage,), (), 1, {worker_id: capacity}), PatchQuery(.5, 1., 0.)


class OnlineProposalEngine:
    def __init__(self, predictor, *, block_size, patch_solver=None, router=None):
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("positive block size required")
        self.predictor = predictor
        self.block_size = block_size
        self.patch = patch_solver or PatchExactSolver()
        self.router = router or TypedRoute()
        self._placements = {}

    def propose(self, request_id, action_id, state_version, cohort, *, capacity,
                free_before, free_after, growth_pages, device, pool_id, new_admission):
        started = time.perf_counter_ns()
        if type(state_version) is not int or state_version < 0 or not action_id:
            raise ValueError("a real authority version and action identity are required")
        rows = tuple(cohort)
        by_id = {row.request_id: row for row in rows}
        if len(by_id) != len(rows) or request_id not in by_id:
            raise ValueError("proposal cohort must contain unique request identities")
        if device.free_capacity != free_before or capacity <= 0:
            raise ValueError("route telemetry must match the observed physical pool")
        predictions = {
            row.request_id: self.predictor.predict(PredictionQuery(
                row.request_id, row.prompt_tokens, row.observed_output_tokens,
                row.category, row.max_output_tokens)) for row in rows
        }
        mean = sum(pred.remaining_mean_tokens for pred in predictions.values())
        variance = sum(pred.remaining_variance_tokens2 for pred in predictions.values())
        pending_prompt = sum(max(0, row.prompt_tokens-row.projected_pages*self.block_size) for row in rows)
        # Summing conditional variances is an empirical working model, not an
        # independence theorem. Page rounding is conservative. The current
        # materialized pages and this action's prompt pages are already charged
        # in ``free_after``; the statistical liability is only the remaining
        # residual. Cap it at the aggregate hard contractual remainder.
        raw_soft_tokens = pending_prompt + mean + self.predictor.z_value * math.sqrt(variance)
        soft_pages = min(sum(row.hard_remaining_pages for row in rows),
                         math.ceil(raw_soft_tokens / self.block_size))
        materialized_after = capacity - free_after
        reserve_fit = materialized_after + soft_pages <= capacity
        try:
            route = self.router.choose(RouteRequest(request_id, "default", growth_pages), [device])
        except NoFeasibleRoute:
            route = None
        placement = None
        row = by_id[request_id]
        if route is not None:
            hard_pages = (row.prompt_tokens+row.max_output_tokens+self.block_size-1)//self.block_size
            key = (request_id, route.device_id, hard_pages, capacity)
            reused = key in self._placements
            if not reused:
                graph, query = single_worker_graph(hard_pages, capacity, route.device_id)
                patch_start = time.perf_counter_ns()
                result = self.patch.solve(graph, query)
                elapsed = (time.perf_counter_ns()-patch_start)/1e9
                solution = result.solution if hasattr(result, "solution") else result
                if solution.assignments != (route.device_id,):
                    raise AssertionError("PATCH did not select the routed deployed worker")
                details = {
                    "devices": list(solution.assignments), "path": list(solution.path),
                    "objective": solution.objective, "solver": solution.solver,
                    "memory_by_device": dict(solution.memory_by_device),
                    "profile_scope": "single-worker structural degeneration; hardware objectives unprofiled",
                    "total_seconds": elapsed,
                    "exact_seconds": 0. if hasattr(result, "solution") else elapsed,
                }
                for name in ("inference_seconds", "decoding_seconds", "repair_seconds", "exact_fallback_seconds",
                             "candidate_count", "repair_trials", "repaired", "used_exact_fallback", "feature_provenance"):
                    details[name] = getattr(result, name, False if name in ("repaired", "used_exact_fallback") else 0)
                self._placements[key] = details
            placement = dict(self._placements[key], reused=reused)
            if reused:
                for name in ("total_seconds", "exact_seconds", "inference_seconds", "decoding_seconds",
                             "repair_seconds", "exact_fallback_seconds"):
                    placement[name] = 0.
        return {
            "request_id": request_id, "action_id": action_id, "state_version": state_version,
            "worker_id": device.device_id, "pool_id": pool_id,
            "new_admission": new_admission,
            "proposal_allowed": route is not None and (not new_admission or reserve_fit),
            "prediction": asdict(predictions[request_id]),
            "observed_output_tokens": row.observed_output_tokens,
            "soft_reserve_tokens": predictions[request_id].soft_upper_tokens,
            "statistical_reserve": {
                "z_value": self.predictor.z_value,
                "mean_tokens": mean, "variance_tokens2": variance, "pending_prompt_tokens": pending_prompt,
                "remaining_pages": soft_pages, "raw_soft_tokens": raw_soft_tokens,
                "materialized_after_pages": materialized_after, "free_after": free_after, "fits": reserve_fit,
                "formula": "U+=K+R+b+ceil((pending_prompt+sum(mu)+z*sqrt(sum(var)))/delta); K+R+b=capacity-free_after",
                "calibration_state": self.predictor.calibration_state,
                "safety_authority": False,
            },
            "route": asdict(route) if route else None, "placement": placement,
            "telemetry": asdict(device), "proposal_ns": time.perf_counter_ns()-started,
        }
