"""Neural PATCH, ported from the historical GraphGPS/Transformer architecture.

Source: code/PATCH_source/PATCH_Experiment_Code_v1.0/code/{model,config}.py
in the archived source used by this TSMC artifact. Those files are never imported.
The original dimensions and optimizer defaults are retained. Unlike that source,
preference hypernetwork adapters are actually connected, path marginals use a
true forward/backward DAG calculation, and losses remain connected to parameters.

``temperature=0.1`` follows the historical code (inverse temperature 10 in the
manuscript's notation). Regret uses a differentiable expected-path relaxation;
reliability has an optional expected-risk constraint and calibrated auxiliary
head. Neither a soft loss nor a predicted gate certifies a returned placement.
The authoritative patch_exact evaluator validates every returned path.

This module requires PyTorch; the rest of prc_nc remains dependency-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import heapq
import json
import math
from pathlib import Path
import time
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .patch_exact import (
    NoFeasiblePlacement, PatchExactSolver, PatchGraph, PatchQuery,
    evaluate_path, resolve_normalization,
)


FORMAT = "prc-patch-neural-v1"
FEATURE_SCHEMA = "physical6-stage4-edge5-v1"


@dataclass(frozen=True)
class NeuralPATCHConfig:
    hidden_dim: int = 128
    gps_layers: int = 3
    transformer_layers: int = 2
    num_heads: int = 4
    adapter_rank: int = 16
    dropout: float = 0.1
    temperature: float = 0.1
    gate_temperature: float = 0.1
    gate_weight: float = 1.0
    candidates: int = 8

    def __post_init__(self):
        for key in ("hidden_dim", "gps_layers", "transformer_layers", "num_heads", "adapter_rank", "candidates"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        for key in ("temperature", "gate_temperature"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not 0 <= self.dropout < 1 or not math.isfinite(self.gate_weight) or self.gate_weight < 0:
            raise ValueError("invalid dropout/gate_weight")


@dataclass(frozen=True)
class NeuralTrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    epochs: int = 30
    regret_weight: float = 1.0
    bottleneck_weight: float = 1.0
    reliability_weight: float = 1.0
    gradient_clip: float = 5.0

    def __post_init__(self):
        for name in ("batch_size", "epochs"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("weight_decay", "regret_weight", "bottleneck_weight", "reliability_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite")


class LowRankAdapter(nn.Module):
    """Historical rank-16 hypernetwork generating a hidden-by-hidden adapter."""

    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.down = nn.Linear(2, rank)
        self.up = nn.Linear(rank, hidden_dim * hidden_dim)

    def forward(self, x, preference):
        matrix = self.up(F.relu(self.down(preference))).reshape(self.hidden_dim, self.hidden_dim)
        # Scaling controls the variance of the generated dense residual map.
        return x @ matrix / math.sqrt(self.hidden_dim)


class GraphGPSLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_dim
        self.conv = nn.Linear(2 * h, h)
        self.attention = nn.MultiheadAttention(h, config.num_heads, batch_first=True)
        self.adapter = LowRankAdapter(h, config.adapter_rank)
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)
        self.mlp = nn.Sequential(nn.Linear(h, 4 * h), nn.ReLU(), nn.Linear(4 * h, h))

    def forward(self, x, adjacency, preference):
        weights = adjacency.to(x.dtype)
        counts = weights.sum(-1, keepdim=True)
        neighbors = (weights @ x) / counts.clamp_min(1)
        neighbors = torch.where(counts > 0, neighbors, x)
        local = F.relu(self.conv(torch.cat((x, neighbors), -1)))
        global_value, _ = self.attention(x[None], x[None], x[None], need_weights=False)
        x = self.norm1(x + local + global_value[0] + self.adapter(x, preference))
        return self.norm2(x + self.mlp(x))


class GraphGPSEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Linear(6, config.hidden_dim)
        self.layers = nn.ModuleList(GraphGPSLayer(config) for _ in range(config.gps_layers))

    def forward(self, physical, adjacency, preference):
        x = F.relu(self.embed(physical))
        for layer in self.layers:
            x = layer(x, adjacency, preference)
        return x


class StageTokenTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.hidden_dim
        self.embed = nn.Linear(h + 6, h)
        layer = nn.TransformerEncoderLayer(h, config.num_heads, 4 * h, config.dropout,
                                          activation="relu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.transformer_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(h)

    def forward(self, tokens):
        return self.norm(self.transformer(self.embed(tokens)[None])[0])


def topological_order(node_count: int, edges: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    """Kahn order, independent of identifiers/source position; rejects cycles."""
    indegree = [0] * node_count
    outgoing = [[] for _ in range(node_count)]
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate directed edge")
    for u, v in edges:
        if not 0 <= u < node_count or not 0 <= v < node_count:
            raise ValueError("edge endpoint outside graph")
        indegree[v] += 1
        outgoing[u].append(v)
    ready = [i for i, d in enumerate(indegree) if not d]
    heapq.heapify(ready)
    order = []
    while ready:
        u = heapq.heappop(ready)
        order.append(u)
        for v in outgoing[u]:
            indegree[v] -= 1
            if not indegree[v]:
                heapq.heappush(ready, v)
    if len(order) != node_count:
        raise ValueError("PATCH graph must be acyclic")
    return tuple(order)


@dataclass
class PathDistribution:
    log_partition: torch.Tensor
    edge_marginals: torch.Tensor
    forward: tuple[torch.Tensor | None, ...]
    backward: tuple[torch.Tensor | None, ...]


def dag_path_distribution(log_weights, edges, node_count, source, sink, *, enabled=None):
    """Exact edge inclusion probabilities under a Gibbs law on DAG paths.

    Marginals are *not* independently row-normalized: each internal node's
    incoming and outgoing flow agrees, while source/sink flow equals one.
    Dead branches receive zero probability. Python None avoids logsumexp over
    exclusively negative infinities and the resulting NaN gradients.
    """
    if log_weights.ndim != 1 or log_weights.numel() != len(edges):
        raise ValueError("one log weight is required per edge")
    if not torch.isfinite(log_weights).all():
        raise ValueError("log weights must be finite")
    if enabled is None:
        enabled = [True] * len(edges)
    if len(enabled) != len(edges):
        raise ValueError("edge mask length mismatch")
    order = topological_order(node_count, edges)
    incoming = [[] for _ in range(node_count)]
    outgoing = [[] for _ in range(node_count)]
    for i, (u, v) in enumerate(edges):
        if enabled[i]:
            incoming[v].append((u, i))
            outgoing[u].append((v, i))
    zero = log_weights.sum() * 0
    alpha = [None] * node_count
    alpha[source] = zero
    for v in order:
        if v == source:
            continue
        terms = [alpha[u] + log_weights[i] for u, i in incoming[v] if alpha[u] is not None]
        if terms:
            alpha[v] = torch.logsumexp(torch.stack(terms), 0)
    if alpha[sink] is None:
        raise NoFeasiblePlacement("TPOT-pruned graph has no source-sink path")
    beta = [None] * node_count
    beta[sink] = zero
    for u in reversed(order):
        if u == sink:
            continue
        terms = [log_weights[i] + beta[v] for v, i in outgoing[u] if beta[v] is not None]
        if terms:
            beta[u] = torch.logsumexp(torch.stack(terms), 0)
    marginals = []
    for i, (u, v) in enumerate(edges):
        if enabled[i] and alpha[u] is not None and beta[v] is not None:
            marginals.append(torch.exp(alpha[u] + log_weights[i] + beta[v] - alpha[sink]))
        else:
            marginals.append(zero)
    return PathDistribution(alpha[sink], torch.stack(marginals), tuple(alpha), tuple(beta))


@dataclass
class PackedPATCHGraph:
    graph: PatchGraph
    query: PatchQuery
    physical: torch.Tensor
    adjacency: torch.Tensor
    stage_device: torch.Tensor
    stage_features: torch.Tensor
    preference: torch.Tensor
    edges: tuple[tuple[int, int], ...]
    edge_features: torch.Tensor
    edge_ttft: torch.Tensor
    edge_cost: torch.Tensor
    edge_risk: torch.Tensor
    stage_decode: torch.Tensor
    stage_availability: torch.Tensor
    enabled: tuple[bool, ...]
    stage_ids: tuple[str, ...]
    source: int
    sink: int
    ttft_scale: float
    cost_scale: float
    ttft_min: float
    cost_min: float
    constant_ttft: bool
    constant_cost: bool
    feature_provenance: str


def pack_graph(graph, query, *, device="cpu", dtype=torch.float32, physical_features=None):
    """Convert the authoritative graph; all stage costs belong to incoming arcs.

    Optional physical_features maps device IDs to the original six normalized
    hardware features [compute, bandwidth, memory, price, availability, carbon].
    Without measured features we derive explicit profile proxies from stage
    durations/capacities; provenance records that choice, never a hardware claim.
    """
    stages = tuple(graph.stages)
    ids = tuple(s.stage_id for s in stages)
    if len(set(ids)) != len(ids) or not stages:
        raise ValueError("stages require unique identifiers")
    devices = sorted(graph.device_memory_pages)
    device_index = {name: i for i, name in enumerate(devices)}
    if physical_features is not None:
        if set(physical_features) != set(devices):
            raise ValueError("physical feature devices must match graph devices")
        physical = [tuple(physical_features[d]) for d in devices]
        if any(len(row) != 6 or not all(math.isfinite(x) for x in row) for row in physical):
            raise ValueError("physical features require six finite values")
        provenance = "provided-normalized-physical6"
    else:
        physical = []
        for d in devices:
            members = [s for s in stages if s.device_id == d]
            pre = sum(s.prefill_seconds for s in members) / max(1, len(members))
            dec = sum(s.decode_seconds for s in members) / max(1, len(members))
            price = sum(s.gpu_cost for s in members) / max(1, len(members))
            availability = min((s.availability for s in members), default=1.0)
            physical.append((1 / (1 + pre), 1 / (1 + dec),
                             float(graph.device_memory_pages[d]), price, availability, 0.0))
        # Only scale the memory/price columns; do not fit using held-out data.
        memory_scale = max(1.0, max(row[2] for row in physical))
        price_scale = max(1.0, max(row[3] for row in physical))
        physical = [(a, b, c / memory_scale, d / price_scale, e, f) for a, b, c, d, e, f in physical]
        provenance = "derived-stage-profile-features-not-hardware-measurements"
    stage_map = {name: i for i, name in enumerate(ids)}
    source, sink = len(stages), len(stages) + 1
    adjacency = [[0.0] * len(devices) for _ in devices]
    edge_rows = []
    for s in stages:
        if s.start_layer == 0:
            edge_rows.append((source, stage_map[s.stage_id], 0.0, 0.0))
    for link in graph.links:
        u, v = stage_map[link.source], stage_map[link.target]
        edge_rows.append((u, v, link.transfer_seconds, link.transfer_cost))
        i, j = device_index[stages[u].device_id], device_index[stages[v].device_id]
        adjacency[i][j] = adjacency[j][i] = 1.0
    for s in stages:
        if s.end_layer == graph.layer_count - 1:
            edge_rows.append((stage_map[s.stage_id], sink, 0.0, 0.0))
    edges = tuple((u, v) for u, v, _, _ in edge_rows)
    topological_order(len(stages) + 2, edges)
    edge_features, ttft, cost, risk, enabled = [], [], [], [], []
    for u, v, transfer, transfer_cost in edge_rows:
        target = stages[v] if v < len(stages) else None
        source_stage = stages[u] if u < len(stages) else None
        edge_features.append((transfer, 0.0, transfer_cost, 0.0,
                              float(bool(target and source_stage and target.device_id != source_stage.device_id))))
        ttft.append(transfer + (target.prefill_seconds if target else 0.0))
        cost.append(transfer_cost + (target.gpu_cost if target else 0.0))
        risk.append(-math.log(max(target.availability, 1e-30)) if target else 0.0)
        enabled.append((target is None or target.decode_seconds <= query.tpot_budget)
                       and (source_stage is None or source_stage.decode_seconds <= query.tpot_budget))
    normalization = resolve_normalization(graph, query)
    actual_bounds = normalization if query.normalization is None else resolve_normalization(graph, replace(query, normalization=None))
    tensor = lambda x: torch.tensor(x, device=device, dtype=dtype)
    stage_features = [(float(s.memory_pages), s.prefill_seconds, s.decode_seconds,
                       (s.start_layer + s.end_layer) / (2 * graph.layer_count)) for s in stages]
    return PackedPATCHGraph(
        graph, query, tensor(physical), tensor(adjacency),
        torch.tensor([device_index[s.device_id] for s in stages], device=device, dtype=torch.long),
        tensor(stage_features), tensor([query.lambda_pref, query.tpot_budget]), edges,
        tensor(edge_features), tensor(ttft), tensor(cost), tensor(risk),
        tensor([s.decode_seconds for s in stages]), tensor([s.availability for s in stages]),
        tuple(enabled), ids, source, sink,
        normalization.ttft_scale, normalization.cost_scale,
        normalization.ttft_min, normalization.cost_min,
        actual_bounds.ttft_min == actual_bounds.ttft_max,
        actual_bounds.cost_min == actual_bounds.cost_max, provenance,
    )


@dataclass
class NeuralPATCHOutput:
    packed: PackedPATCHGraph
    edge_scores: torch.Tensor
    edge_gates: torch.Tensor
    stage_bottleneck: torch.Tensor
    stage_reliability: torch.Tensor
    log_weights: torch.Tensor
    distribution: PathDistribution | None


class NeuralPATCH(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or NeuralPATCHConfig()
        h = self.config.hidden_dim
        self.graph_encoder = GraphGPSEncoder(self.config)
        self.stage_transformer = StageTokenTransformer(self.config)
        self.cost_head = nn.Sequential(nn.Linear(2 * h + 7, h), nn.ReLU(), nn.Linear(h, 1))
        self.bottleneck_head = nn.Sequential(nn.Linear(2 * h + 5, h), nn.ReLU(), nn.Linear(h, 1))
        self.reliability_head = nn.Sequential(nn.Linear(2 * h + 7, h), nn.ReLU(), nn.Linear(h, 1), nn.Sigmoid())

    def forward(self, packed, *, compute_marginals=True):
        embeddings = self.graph_encoder(packed.physical, packed.adjacency, packed.preference)
        n = len(packed.stage_ids)
        pref = packed.preference.expand(n, -1)
        tokens = torch.cat((embeddings[packed.stage_device], packed.stage_features, pref), -1)
        stages = self.stage_transformer(tokens)
        augmented = torch.cat((stages, stages.new_zeros((2, self.config.hidden_dim))), 0)
        uv = torch.tensor(packed.edges, device=stages.device, dtype=torch.long)
        base = torch.cat((augmented[uv[:, 0]], augmented[uv[:, 1]], packed.edge_features), -1)
        scores = self.cost_head(torch.cat((base, packed.preference.expand(len(packed.edges), -1)), -1)).squeeze(-1)
        stage_features5 = torch.cat((packed.stage_features, stages.new_zeros((n, 1))), -1)
        stage_base = torch.cat((stages, stages, stage_features5), -1)
        bottleneck = F.softplus(self.bottleneck_head(stage_base).squeeze(-1))
        reliability = self.reliability_head(torch.cat((stage_base, pref), -1)).squeeze(-1)
        # Target-stage attribution counts every stage once. The sink costs zero.
        bottleneck_aug = torch.cat((bottleneck, bottleneck.new_zeros(2)))
        gates = torch.sigmoid((packed.query.tpot_budget - bottleneck_aug[uv[:, 1]]) / self.config.gate_temperature)
        energy = scores - self.config.gate_weight * F.logsigmoid(
            (packed.query.tpot_budget - bottleneck_aug[uv[:, 1]]) / self.config.gate_temperature)
        log_weights = -energy / self.config.temperature
        distribution = None
        if compute_marginals:
            distribution = dag_path_distribution(log_weights, packed.edges, n + 2,
                                                 packed.source, packed.sink, enabled=packed.enabled)
        return NeuralPATCHOutput(packed, scores, gates, bottleneck, reliability, log_weights, distribution)


def patch_losses(output, teacher_path, *, teacher_marginals=None, config=None):
    """KL imitation, relaxed regret, bottleneck calibration and risk constraint.

    Teacher paths must be feasible under the *same* query. Regret evaluates true
    additive graph costs weighted by student marginals, never teacher values as
    a stand-in for predictions. The optional reliability term uses true risk in
    its path relaxation plus supervision for the predicted reliability head.
    """
    config = config or NeuralTrainingConfig()
    p = output.packed
    teacher = evaluate_path(p.graph, p.query, teacher_path)
    if output.distribution is None:
        raise ValueError("training requires differentiable path marginals")
    marginals = output.distribution.edge_marginals
    positions = {name: i for i, name in enumerate(p.stage_ids)}
    path = (p.source,) + tuple(positions[x] for x in teacher_path) + (p.sink,)
    edge_index = {edge: i for i, edge in enumerate(p.edges)}
    if teacher_marginals is None:
        teacher_marginals = torch.zeros_like(marginals)
        for edge in zip(path, path[1:]):
            teacher_marginals[edge_index[edge]] = 1.0
    else:
        teacher_marginals = torch.as_tensor(teacher_marginals, device=marginals.device, dtype=marginals.dtype)
        if teacher_marginals.shape != marginals.shape or not torch.isfinite(teacher_marginals).all():
            raise ValueError("teacher marginals shape/value mismatch")
        if torch.any((teacher_marginals < 0) | (teacher_marginals > 1)):
            raise ValueError("teacher edge marginals must lie in [0,1]")
        teacher_marginals = teacher_marginals.detach()
        for node in range(p.sink + 1):
            incoming = sum(float(teacher_marginals[i]) for i, (_, v) in enumerate(p.edges) if v == node)
            outgoing = sum(float(teacher_marginals[i]) for i, (u, _) in enumerate(p.edges) if u == node)
            expected = 1.0 if node == p.source else (-1.0 if node == p.sink else 0.0)
            if not math.isclose(outgoing - incoming, expected, abs_tol=1e-6):
                raise ValueError("teacher marginals violate unit source-sink flow")
        if any(not allowed and float(teacher_marginals[i]) > 0 for i, allowed in enumerate(p.enabled)):
            raise ValueError("teacher places mass on TPOT-pruned edges")
    active = teacher_marginals > 0
    # epsilon only inside logarithms; it does not replace graph probabilities.
    imitation = (teacher_marginals[active] * (torch.log(teacher_marginals[active] + 1e-12)
                 - torch.log(marginals[active] + 1e-12))).sum()
    expected_ttft = (marginals * p.edge_ttft).sum()
    expected_cost = (marginals * p.edge_cost).sum()
    # A constant path metric has zero distribution-dependent gradient. Avoid
    # magnifying float32 flow roundoff by a zero-range epsilon denominator.
    if p.constant_ttft:
        expected_ttft = marginals.sum() * 0 + teacher.ttft_seconds
    if p.constant_cost:
        expected_cost = marginals.sum() * 0 + teacher.monetary_cost
    lam = p.query.lambda_pref
    expected_objective = lam * (expected_ttft - p.ttft_min) / p.ttft_scale
    expected_objective = expected_objective + (1 - lam) * (expected_cost - p.cost_min) / p.cost_scale
    regret = F.relu(expected_objective - teacher.objective)
    teacher_stages = torch.tensor(path[1:-1], device=marginals.device)
    predicted_bottleneck = output.stage_bottleneck[teacher_stages].max()
    bottleneck = torch.abs(predicted_bottleneck - teacher.tpot_seconds)
    expected_risk = (marginals * p.edge_risk).sum()
    risk_limit = -math.log(p.query.min_reliability) if p.query.min_reliability > 0 else float("inf")
    risk_constraint = F.relu(expected_risk - risk_limit) if math.isfinite(risk_limit) else expected_risk * 0
    reliability_calibration = F.mse_loss(output.stage_reliability, p.stage_availability)
    reliability = risk_constraint + reliability_calibration
    total = imitation + config.regret_weight * regret + config.bottleneck_weight * bottleneck
    total = total + config.reliability_weight * reliability
    return {"imitation": imitation, "regret": regret, "bottleneck": bottleneck,
            "reliability": reliability, "risk_constraint": risk_constraint,
            "reliability_calibration": reliability_calibration, "total": total}


def topology_id(graph):
    """Structural identity excludes costs, query preferences and user labels."""
    payload = {
        "stages": sorted((s.stage_id, s.device_id, s.start_layer, s.end_layer) for s in graph.stages),
        "links": sorted((x.source, x.target) for x in graph.links),
        "devices": sorted(graph.device_memory_pages), "layer_count": graph.layer_count,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_split_manifest(splits):
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("splits must contain train/validation/test topology IDs")
    seen = set()
    for name in ("train", "validation", "test"):
        ids = splits[name]
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError(f"{name} topology IDs must be nonempty and unique")
        if any(type(x) is not str or len(x) != 64 for x in ids):
            raise ValueError("topology IDs must be SHA-256 strings")
        if seen.intersection(ids):
            raise ValueError("topology leakage across dataset splits")
        seen.update(ids)


def save_checkpoint(model, path, provenance):
    """Save a versioned, strict checkpoint; caller supplies actual train evidence."""
    validate_split_manifest(provenance.get("topology_splits"))
    if type(provenance.get("optimizer_steps")) is not int or provenance["optimizer_steps"] <= 0:
        raise ValueError("trained checkpoint requires positive optimizer_steps")
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("nonfinite checkpoint parameter")
    payload = {"format": FORMAT, "feature_schema": FEATURE_SCHEMA,
               "config": asdict(model.config), "state_dict": state,
               "provenance": provenance, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_checkpoint(path, *, device="cpu"):
    """Never accept historical/random/linear state or partially loaded tensors."""
    target = Path(path)
    payload = torch.load(target, map_location="cpu", weights_only=True)
    required = {"format", "feature_schema", "config", "state_dict", "provenance", "source_sha256"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("neural PATCH checkpoint schema mismatch")
    if payload["format"] != FORMAT or payload["feature_schema"] != FEATURE_SCHEMA:
        raise ValueError("unsupported neural PATCH format/features")
    expected_source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if payload["source_sha256"] != expected_source:
        raise ValueError("neural PATCH checkpoint was trained with a different source revision")
    if set(payload["config"]) != set(asdict(NeuralPATCHConfig())):
        raise ValueError("architecture config fields mismatch")
    provenance = payload["provenance"]
    validate_split_manifest(provenance.get("topology_splits"))
    if type(provenance.get("optimizer_steps")) is not int or provenance["optimizer_steps"] <= 0:
        raise ValueError("checkpoint has no training steps")
    if "architecture" in provenance and provenance["architecture"] != payload["config"]:
        raise ValueError("checkpoint architecture provenance does not match model config")
    if any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in payload["state_dict"].values()):
        raise ValueError("checkpoint tensors are malformed/nonfinite")
    model = NeuralPATCH(NeuralPATCHConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    metadata = {"checkpoint_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "provenance": provenance, "training_source_sha256": payload["source_sha256"]}
    return model, metadata


def _k_paths(output, k):
    """Deterministic K shortest learned paths in a DAG; no exact objective oracle."""
    p = output.packed
    score = (-output.log_weights).detach().cpu().tolist()
    incoming = [[] for _ in range(p.sink + 1)]
    for i, (u, v) in enumerate(p.edges):
        if p.enabled[i]:
            incoming[v].append((u, i))
    paths = [[] for _ in range(p.sink + 1)]
    paths[p.source] = [(0.0, (p.source,))]
    for v in topological_order(p.sink + 1, p.edges):
        if v == p.source:
            continue
        candidates = [(value + score[i], path + (v,)) for u, i in incoming[v] for value, path in paths[u]]
        paths[v] = heapq.nsmallest(k, candidates)
    return [tuple(p.stage_ids[i] for i in path[1:-1]) for _, path in paths[p.sink]]


@dataclass(frozen=True)
class NeuralPatchResult:
    solution: object
    inference_seconds: float
    decoding_seconds: float
    repair_seconds: float
    exact_fallback_seconds: float
    candidate_count: int
    repair_trials: int
    repaired: bool
    used_exact_fallback: bool
    feature_provenance: str

    @property
    def path(self):
        return self.solution.path

    @property
    def objective(self):
        return self.solution.objective


class NeuralPatchSolver:
    """Real neural inference followed by hard validation and bounded local repair."""

    def __init__(self, model, *, checkpoint_metadata=None, allow_untrained=False, max_repair_trials=256):
        if checkpoint_metadata is None and not allow_untrained:
            raise ValueError("use from_checkpoint for evaluation; untrained experiments require explicit opt-in")
        if type(max_repair_trials) is not int or max_repair_trials < 0:
            raise ValueError("invalid max_repair_trials")
        self.model = model.eval()
        self.checkpoint_metadata = checkpoint_metadata
        self.max_repair_trials = max_repair_trials

    @classmethod
    def from_checkpoint(cls, path, *, device="cpu", max_repair_trials=256):
        model, metadata = load_checkpoint(path, device=device)
        return cls(model, checkpoint_metadata=metadata, max_repair_trials=max_repair_trials)

    def solve(self, graph, query, *, physical_features=None):
        parameter = next(self.model.parameters())

        def sync():
            if parameter.device.type == "cuda":
                torch.cuda.synchronize(parameter.device)

        sync()
        start = time.perf_counter()
        packed = pack_graph(graph, query, device=parameter.device, dtype=parameter.dtype,
                            physical_features=physical_features)
        with torch.inference_mode():
            output = self.model(packed, compute_marginals=False)
        sync()
        inference = time.perf_counter() - start
        start = time.perf_counter()
        candidates = _k_paths(output, self.model.config.candidates)
        feasible = []
        for path in candidates:
            try:
                feasible.append(evaluate_path(graph, query, path))
            except NoFeasiblePlacement:
                pass
        decoding = time.perf_counter() - start
        repair_time, fallback_time, trials = 0.0, 0.0, 0
        repaired, fallback = False, False
        if feasible:
            # Ranked candidates already follow neural scores; retaining the first
            # valid one avoids inserting an exact-objective optimizer here.
            solution = feasible[0]
        else:
            start = time.perf_counter()
            stage_map = {s.stage_id: s for s in graph.stages}
            # Deterministic one-coordinate repair in learned candidate order.
            for path in candidates:
                if trials >= self.max_repair_trials:
                    break
                for index, stage_id in enumerate(path):
                    current = stage_map[stage_id]
                    alternatives = sorted(s.stage_id for s in graph.stages
                                          if s.start_layer == current.start_layer and s.end_layer == current.end_layer
                                          and s.stage_id != stage_id)
                    for alternative in alternatives:
                        if trials >= self.max_repair_trials:
                            break
                        trials += 1
                        changed = path[:index] + (alternative,) + path[index + 1:]
                        try:
                            feasible.append(evaluate_path(graph, query, changed))
                        except NoFeasiblePlacement:
                            continue
                    if feasible or trials >= self.max_repair_trials:
                        break
                if feasible:
                    break
            repair_time = time.perf_counter() - start
            if feasible:
                solution = min(feasible, key=lambda s: (s.objective, s.path))
                repaired = True
            else:
                start = time.perf_counter()
                solution = PatchExactSolver().solve(graph, query)
                fallback_time = time.perf_counter() - start
                fallback = True
        # Defense against a repair/fallback implementation drifting from schema.
        evaluate_path(graph, query, solution.path)
        solution = replace(solution, solver="PATCH-Learned+exact-fallback" if fallback else "PATCH-Learned-Neural")
        return NeuralPatchResult(solution, inference, decoding, repair_time, fallback_time,
                                 len(candidates), trials, repaired, fallback, packed.feature_provenance)
