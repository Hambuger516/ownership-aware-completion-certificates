"""PATCH's constrained stage-DAG placement, with exact resource labels.

The paper objective is normalized TTFT versus monetary cost. TPOT is the
maximum stage decode time, and reliability is the paper's product of per-stage
availability scores. A placement consumes the SUM of its segments' memory on
each physical device. These are placement proposals, never allocator grants.

Stages may cover one fixed segment or a contiguous group of layers. Every
source-to-sink path covers all layers exactly once. Stage compute/cost is
charged on entry, avoiding the first-stage double count and missing final-stage
cost in the archived ``problem.py`` graph builder. No historical file is changed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
from types import MappingProxyType
from typing import Mapping, Sequence


class NoFeasiblePlacement(ValueError):
    """The supplied path, or the complete queried graph, is infeasible."""


class SearchLimitExceeded(RuntimeError):
    """An explicit search budget was reached; this is not infeasibility."""


def _number(value, name, *, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _name(value, name):
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class PatchStage:
    stage_id: str
    device_id: str
    start_layer: int
    end_layer: int
    prefill_seconds: float
    decode_seconds: float
    memory_pages: int
    gpu_cost: float
    availability: float = 1.0

    def __post_init__(self):
        _name(self.stage_id, "stage_id")
        _name(self.device_id, "device_id")
        _integer(self.start_layer, "start_layer")
        _integer(self.end_layer, "end_layer", self.start_layer)
        _integer(self.memory_pages, "memory_pages")
        for name in ("prefill_seconds", "decode_seconds", "gpu_cost", "availability"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.availability > 1:
            raise ValueError("availability must lie in [0, 1]")


@dataclass(frozen=True)
class PatchLink:
    source: str
    target: str
    transfer_seconds: float = 0.0
    transfer_cost: float = 0.0

    def __post_init__(self):
        _name(self.source, "source")
        _name(self.target, "target")
        for name in ("transfer_seconds", "transfer_cost"):
            object.__setattr__(self, name, _number(getattr(self, name), name))


@dataclass(frozen=True)
class PatchGraph:
    stages: tuple[PatchStage, ...]
    links: tuple[PatchLink, ...]
    layer_count: int
    device_memory_pages: Mapping[str, int]

    def __post_init__(self):
        _integer(self.layer_count, "layer_count", 1)
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "links", tuple(self.links))
        capacities = dict(self.device_memory_pages)
        for key, value in capacities.items():
            _name(key, "device_id")
            _integer(value, "device memory capacity")
        object.__setattr__(self, "device_memory_pages", MappingProxyType(capacities))
        if any(not isinstance(stage, PatchStage) for stage in self.stages):
            raise ValueError("stages must contain PatchStage objects")
        by_id = {stage.stage_id: stage for stage in self.stages}
        if len(by_id) != len(self.stages):
            raise ValueError("stage IDs must be unique")
        for stage in self.stages:
            if stage.end_layer >= self.layer_count:
                raise ValueError("stage extends beyond layer_count")
            if stage.device_id not in capacities:
                raise ValueError("every stage device needs an explicit memory capacity")
        edges = set()
        for link in self.links:
            if not isinstance(link, PatchLink):
                raise ValueError("links must contain PatchLink objects")
            if link.source not in by_id or link.target not in by_id:
                raise ValueError("link references an unknown stage")
            if (link.source, link.target) in edges:
                raise ValueError("duplicate directed link")
            edges.add((link.source, link.target))
            if by_id[link.source].end_layer + 1 != by_id[link.target].start_layer:
                raise ValueError("a link must join contiguous, non-overlapping segments")


@dataclass(frozen=True)
class PatchNormalization:
    ttft_min: float
    ttft_max: float
    cost_min: float
    cost_max: float
    epsilon: float = 1e-12

    def __post_init__(self):
        for name in ("ttft_min", "ttft_max", "cost_min", "cost_max", "epsilon"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.ttft_max < self.ttft_min or self.cost_max < self.cost_min or self.epsilon <= 0:
            raise ValueError("normalization requires ordered bounds and positive epsilon")

    @property
    def ttft_scale(self):
        return self.ttft_max - self.ttft_min + self.epsilon

    @property
    def cost_scale(self):
        return self.cost_max - self.cost_min + self.epsilon


@dataclass(frozen=True)
class PatchQuery:
    lambda_pref: float
    tpot_budget: float
    min_reliability: float = 0.0
    normalization: PatchNormalization | None = None

    def __post_init__(self):
        for name in ("lambda_pref", "tpot_budget", "min_reliability"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.lambda_pref > 1 or self.min_reliability > 1 or self.tpot_budget <= 0:
            raise ValueError("preference/reliability must be in [0, 1], TPOT budget must be positive")
        if self.normalization is not None and not isinstance(self.normalization, PatchNormalization):
            raise ValueError("normalization must be PatchNormalization or None")


@dataclass(frozen=True)
class PatchSolution:
    path: tuple[str, ...]
    assignments: tuple[str, ...]
    ttft_seconds: float
    tpot_seconds: float
    monetary_cost: float
    reliability: float
    memory_by_device: Mapping[str, int]
    objective: float
    solver: str = "PATCH-Exact"
    labels_expanded: int = 0
    labels_pruned: int = 0


def topological_order(graph: PatchGraph) -> tuple[str, ...]:
    """Kahn order; input arrays and names need not be topologically sorted."""
    degree = {stage.stage_id: 0 for stage in graph.stages}
    outgoing = {key: [] for key in degree}
    for link in graph.links:
        degree[link.target] += 1
        outgoing[link.source].append(link.target)
    ready = [key for key, value in degree.items() if value == 0]
    heapq.heapify(ready)
    result = []
    while ready:
        node = heapq.heappop(ready)
        result.append(node)
        for target in sorted(outgoing[node]):
            degree[target] -= 1
            if degree[target] == 0:
                heapq.heappush(ready, target)
    if len(result) != len(degree):
        raise ValueError("stage graph must be acyclic")
    return tuple(result)


def resolve_normalization(graph: PatchGraph, query: PatchQuery) -> PatchNormalization:
    """Instance-level min/max over complete, locally memory-feasible DAG paths.

    These bounds deliberately do not depend on preference, TPOT or reliability
    queries, or on cumulative-memory pruning. They define one fixed affine
    objective for the instance. Explicit paper/profile bounds override them.
    """
    if query.normalization is not None:
        return query.normalization
    by_id = {stage.stage_id: stage for stage in graph.stages}
    outgoing = {key: [] for key in by_id}
    for link in graph.links:
        outgoing[link.source].append(link)
    # Values are min TTFT, max TTFT, min cost, max cost.
    bounds = {}
    for node in topological_order(graph):
        stage = by_id[node]
        if stage.memory_pages > graph.device_memory_pages[stage.device_id]:
            continue
        if stage.start_layer == 0:
            bounds[node] = (stage.prefill_seconds, stage.prefill_seconds, stage.gpu_cost, stage.gpu_cost)
        if node not in bounds:
            continue
        low_t, high_t, low_c, high_c = bounds[node]
        for link in outgoing[node]:
            target = by_id[link.target]
            if target.memory_pages > graph.device_memory_pages[target.device_id]:
                continue
            dt = link.transfer_seconds + target.prefill_seconds
            dc = link.transfer_cost + target.gpu_cost
            new = (low_t + dt, high_t + dt, low_c + dc, high_c + dc)
            old = bounds.get(link.target)
            bounds[link.target] = new if old is None else (
                min(old[0], new[0]), max(old[1], new[1]), min(old[2], new[2]), max(old[3], new[3])
            )
    terminals = [bounds[s.stage_id] for s in graph.stages
                 if s.end_layer == graph.layer_count - 1 and s.stage_id in bounds]
    if not terminals:
        raise NoFeasiblePlacement("no complete, locally memory-feasible stage path")
    return PatchNormalization(
        min(row[0] for row in terminals), max(row[1] for row in terminals),
        min(row[2] for row in terminals), max(row[3] for row in terminals),
    )


def _risk(availability):
    return -math.log(availability) if availability else math.inf


def _risk_limit(query):
    return -math.log(query.min_reliability) if query.min_reliability else math.inf


def _objective(ttft, cost, query, normalization):
    return (
        query.lambda_pref * (ttft - normalization.ttft_min) / normalization.ttft_scale
        + (1 - query.lambda_pref) * (cost - normalization.cost_min) / normalization.cost_scale
    )


def evaluate_path(graph: PatchGraph, query: PatchQuery, path: Sequence[str]) -> PatchSolution:
    """Hard validation and metrics, shared by exact and learned proposers.

    Reliability is checked in log space, without a permissive tolerance. This
    function neither authorizes resources nor changes the request's hard cap.
    """
    path = tuple(path)
    by_id = {stage.stage_id: stage for stage in graph.stages}
    if not path or any(key not in by_id for key in path):
        raise NoFeasiblePlacement("path must contain known stages")
    if by_id[path[0]].start_layer != 0 or by_id[path[-1]].end_layer != graph.layer_count - 1:
        raise NoFeasiblePlacement("path must cover the complete layer range")
    links = {(link.source, link.target): link for link in graph.links}
    memory = {key: 0 for key in graph.device_memory_pages}
    ttft = cost = tpot = risk = 0.0
    for index, key in enumerate(path):
        stage = by_id[key]
        if index:
            link = links.get((path[index - 1], key))
            if link is None:
                raise NoFeasiblePlacement("path contains a missing or non-contiguous transition")
            ttft += link.transfer_seconds
            cost += link.transfer_cost
        ttft += stage.prefill_seconds
        cost += stage.gpu_cost
        tpot = max(tpot, stage.decode_seconds)
        risk += _risk(stage.availability)
        memory[stage.device_id] += stage.memory_pages
        if memory[stage.device_id] > graph.device_memory_pages[stage.device_id]:
            raise NoFeasiblePlacement("cumulative device memory capacity exceeded")
    if tpot > query.tpot_budget:
        raise NoFeasiblePlacement("TPOT budget exceeded")
    if risk > _risk_limit(query):
        raise NoFeasiblePlacement("reliability threshold not met")
    normalization = resolve_normalization(graph, query)
    return PatchSolution(
        path, tuple(by_id[key].device_id for key in path), ttft, tpot, cost,
        math.exp(-risk), memory, _objective(ttft, cost, query, normalization),
        solver="validated-path",
    )


@dataclass(frozen=True)
class _Label:
    path: tuple[str, ...]
    ttft: float
    cost: float
    risk: float
    memory: tuple[int, ...]
    scalar: float


def _dominates(left: _Label, right: _Label):
    # Only labels ending at the SAME graph vertex are compared. All future
    # extensions add identical nonnegative resources and additive objective.
    # No epsilon or beam truncation may discard a potentially optimal label.
    # Keep both physical objective components. This is conservative relative to
    # query-specific scalar dominance and avoids hiding a tiny monetary
    # difference behind an enormous, constant normalized latency component.
    return left.ttft <= right.ttft and left.cost <= right.cost and left.risk <= right.risk and all(
        a <= b for a, b in zip(left.memory, right.memory)
    )


class PatchExactSolver:
    """Resource-constrained DAG shortest path via non-dominated labels.

    Reliability and cumulative per-device memory require retaining alternatives;
    one distance/predecessor per vertex is not exact. Worst-case label growth
    is exponential. An optional max_labels limit raises SearchLimitExceeded,
    never silently returns an approximation or a false infeasibility result.
    """

    def __init__(self, *, max_labels: int | None = None):
        if max_labels is not None:
            _integer(max_labels, "max_labels", 1)
        self.max_labels = max_labels

    def solve(self, graph: PatchGraph, query: PatchQuery) -> PatchSolution:
        normalization = resolve_normalization(graph, query)
        by_id = {stage.stage_id: stage for stage in graph.stages}
        devices = tuple(sorted(graph.device_memory_pages))
        device_index = {key: i for i, key in enumerate(devices)}
        capacities = tuple(graph.device_memory_pages[key] for key in devices)
        outgoing = {key: [] for key in by_id}
        for link in graph.links:
            outgoing[link.source].append(link)
        labels = {key: [] for key in by_id}
        risk_limit = _risk_limit(query)
        expanded = pruned = 0

        def insert(key, label):
            nonlocal expanded, pruned
            expanded += 1
            if self.max_labels is not None and expanded > self.max_labels:
                raise SearchLimitExceeded("PATCH exact label budget reached before optimality was established")
            current = labels[key]
            if any(_dominates(old, label) for old in current):
                pruned += 1
                return
            keep = [old for old in current if not _dominates(label, old)]
            pruned += len(current) - len(keep)
            keep.append(label)
            labels[key] = keep

        def extend(old, stage, transfer_seconds=0.0, transfer_cost=0.0):
            if stage.decode_seconds > query.tpot_budget:
                return
            risk = old.risk + _risk(stage.availability)
            if risk > risk_limit:
                return
            memory = list(old.memory)
            memory[device_index[stage.device_id]] += stage.memory_pages
            if any(value > cap for value, cap in zip(memory, capacities)):
                return
            ttft = old.ttft + transfer_seconds + stage.prefill_seconds
            cost = old.cost + transfer_cost + stage.gpu_cost
            # The same affine offsets are used once in each prefix score, not
            # once per stage. In particular a constant complete-path objective
            # component is zero rather than an enormous epsilon-scaled offset.
            scalar = _objective(ttft, cost, query, normalization)
            insert(stage.stage_id, _Label(old.path + (stage.stage_id,), ttft, cost, risk, tuple(memory), scalar))

        empty = _Label((), 0.0, 0.0, 0.0, (0,) * len(devices), 0.0)
        for node in topological_order(graph):
            stage = by_id[node]
            if stage.start_layer == 0:
                extend(empty, stage)
            for label in labels[node]:
                for link in sorted(outgoing[node], key=lambda edge: edge.target):
                    extend(label, by_id[link.target], link.transfer_seconds, link.transfer_cost)
        terminals = [label for key, values in labels.items()
                     if by_id[key].end_layer == graph.layer_count - 1 for label in values]
        if not terminals:
            raise NoFeasiblePlacement("no path meets TPOT, reliability and cumulative device memory constraints")
        best = min(terminals, key=lambda label: (label.scalar, label.path))
        result = evaluate_path(graph, replace(query, normalization=normalization), best.path)
        return replace(result, solver="PATCH-Exact", labels_expanded=expanded, labels_pruned=pruned)
