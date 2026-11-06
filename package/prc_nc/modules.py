"""The proposal side of PRC.

This module deliberately keeps soft proposals separate from the hard resource
contract used by :mod:`prc_nc.controller`.  A predictor may rank or size a
window, a typed route may select a worker, and PATCH may choose a placement;
none of those outputs can increase ``max_output_tokens`` or bypass the
certificate controller.

The implementations are dependency-free so the same objects can be exercised
in the CPU protocol tests and in the vLLM adapter.  The learned placement
solver is a small, auditable linear imitation model.  It is not presented as a
replacement for a neural model; it provides a real trainable learned path and
deterministic constraint repair while the larger historical checkpoint is
being audited separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, pvariance
from typing import Iterable, Mapping, Sequence


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class PredictionObservation:
    request_id: str
    prompt_tokens: int
    observed_output_tokens: int = 0
    max_output_tokens: int = 0
    category: str = "default"

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("request_id must be a nonempty string")
        for name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("observed_output_tokens", self.observed_output_tokens),
            ("max_output_tokens", self.max_output_tokens),
        ):
            _nonnegative_int(value, name)
        if type(self.category) is not str or not self.category:
            raise ValueError("category must be a nonempty string")


@dataclass(frozen=True)
class LengthPrediction:
    request_id: str
    remaining_mean_tokens: float
    remaining_variance_tokens2: float
    soft_upper_tokens: int
    confidence: float
    calibration_version: str
    support: str

    def __post_init__(self) -> None:
        _finite(self.remaining_mean_tokens, "remaining_mean_tokens")
        _finite(self.remaining_variance_tokens2, "remaining_variance_tokens2")
        if self.remaining_mean_tokens < 0 or self.remaining_variance_tokens2 < 0:
            raise ValueError("prediction moments must be nonnegative")
        _nonnegative_int(self.soft_upper_tokens, "soft_upper_tokens")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must lie in [0, 1]")


class ResidualLengthPredictor:
    """Conditional empirical predictor for *remaining* output tokens.

    ``fit`` only accepts completed historical records.  A record may be a
    mapping with ``final_output_tokens`` and ``observed_output_tokens`` or a
    four-tuple ``(category, observed, final, max_output)``.  At runtime
    ``predict`` never reads the final output; it uses only the observation and
    the learned category/global moments.  Hard contracts remain outside this
    class.
    """

    def __init__(self, *, z_value: float = 2.0, calibration_version: str = "empirical-v1"):
        self.z_value = _finite(z_value, "z_value")
        if self.z_value < 0:
            raise ValueError("z_value must be nonnegative")
        self.calibration_version = str(calibration_version)
        if not self.calibration_version:
            raise ValueError("calibration_version must be nonempty")
        self._groups: dict[str, tuple[float, float, int]] = {}
        self._global: tuple[float, float, int] = (0.0, 0.0, 0)

    @property
    def fitted(self) -> bool:
        return self._global[2] > 0

    @staticmethod
    def _record(record):
        if isinstance(record, Mapping):
            category = str(record.get("category", "default"))
            observed = _nonnegative_int(int(record.get("observed_output_tokens", 0)), "observed_output_tokens")
            final = _nonnegative_int(int(record["final_output_tokens"]), "final_output_tokens")
            if final < observed:
                raise ValueError("final output cannot precede observed output")
            return category, final - observed
        if isinstance(record, Sequence) and len(record) == 4:
            category, observed, final, _max_output = record
            observed = _nonnegative_int(int(observed), "observed_output_tokens")
            final = _nonnegative_int(int(final), "final_output_tokens")
            if final < observed:
                raise ValueError("final output cannot precede observed output")
            return str(category), final - observed
        raise ValueError("training records must be mappings or four-tuples")

    def fit(self, records: Iterable[Mapping | Sequence], *, calibration_version: str | None = None) -> "ResidualLengthPredictor":
        grouped: dict[str, list[int]] = {}
        all_values: list[int] = []
        for record in records:
            category, remaining = self._record(record)
            if not category:
                raise ValueError("training category must be nonempty")
            grouped.setdefault(category, []).append(remaining)
            all_values.append(remaining)
        if not all_values:
            raise ValueError("at least one completed record is required")
        self._groups = {
            key: (fmean(values), pvariance(values) if len(values) > 1 else 0.0, len(values))
            for key, values in grouped.items()
        }
        self._global = (fmean(all_values), pvariance(all_values) if len(all_values) > 1 else 0.0, len(all_values))
        if calibration_version is not None:
            if not calibration_version:
                raise ValueError("calibration_version must be nonempty")
            self.calibration_version = str(calibration_version)
        return self

    def predict(self, observation: PredictionObservation) -> LengthPrediction:
        if not isinstance(observation, PredictionObservation):
            raise TypeError("predict expects PredictionObservation")
        moments = self._groups.get(observation.category, self._global)
        mean, variance, count = moments
        if not count:
            # An unfitted predictor has no evidence.  It may propose zero soft
            # growth, but the caller must continue to enforce the hard cap.
            mean, variance, count = 0.0, 0.0, 0
        residual_cap = max(0, observation.max_output_tokens - observation.observed_output_tokens)
        upper = min(residual_cap, max(0, math.ceil(mean + self.z_value * math.sqrt(variance))))
        confidence = 0.0 if not count else min(1.0, count / 32.0)
        support = "category" if observation.category in self._groups else ("global" if count else "unfitted")
        return LengthPrediction(
            request_id=observation.request_id,
            remaining_mean_tokens=min(mean, residual_cap),
            remaining_variance_tokens2=variance,
            soft_upper_tokens=upper,
            confidence=confidence,
            calibration_version=self.calibration_version,
            support=support,
        )


@dataclass(frozen=True)
class DeviceState:
    device_id: str
    domain: str
    free_pages: int
    queue_depth: int = 0
    utilization: float = 0.0
    locality: float = 0.0
    feasible: bool = True
    native: bool = True

    def __post_init__(self) -> None:
        if not self.device_id or not self.domain:
            raise ValueError("device_id and domain must be nonempty")
        _nonnegative_int(self.free_pages, "free_pages")
        _nonnegative_int(self.queue_depth, "queue_depth")
        for name, value in (("utilization", self.utilization), ("locality", self.locality)):
            _finite(value, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class RouteRequest:
    request_id: str
    preferred_domain: str
    min_free_pages: int = 0

    def __post_init__(self) -> None:
        if not self.request_id or not self.preferred_domain:
            raise ValueError("route request identifiers/domains must be nonempty")
        _nonnegative_int(self.min_free_pages, "min_free_pages")


@dataclass(frozen=True)
class RouteProposal:
    request_id: str
    domain: str
    device_id: str
    score: float
    residual: float
    candidates: tuple[str, ...]


class TypedRoute:
    """Two-stage typed domain/device route with explicit feasibility filtering."""

    MACRO_WEIGHTS = (0.24, 0.38, 0.30, 0.08)  # compute, memory, queue, locality
    MICRO_WEIGHTS = (0.22, 0.32, 0.20, 0.14, 0.06, 0.06)

    def __init__(self, *, use_residual: bool = True):
        self.use_residual = bool(use_residual)

    def choose(self, request: RouteRequest, devices: Sequence[DeviceState]) -> RouteProposal:
        feasible = [d for d in devices if d.feasible and d.free_pages >= request.min_free_pages]
        if not feasible:
            raise ValueError("no feasible device satisfies the hard page requirement")
        domains = sorted({d.domain for d in feasible})

        def domain_score(domain: str) -> float:
            members = [d for d in feasible if d.domain == domain]
            best = min(members, key=lambda d: (d.queue_depth, d.device_id))
            # Lower is better. Locality is represented as a benefit, hence the
            # negative sign.  Preferred-domain residual is soft only.
            return (
                self.MACRO_WEIGHTS[0] * best.utilization
                + self.MACRO_WEIGHTS[1] * (1.0 / max(1, best.free_pages))
                + self.MACRO_WEIGHTS[2] * min(1.0, best.queue_depth / 32.0)
                - self.MACRO_WEIGHTS[3] * best.locality
                + (0.025 if self.use_residual and domain != request.preferred_domain else 0.0)
            )

        domain = min(domains, key=lambda d: (domain_score(d), d))
        candidates = [d for d in feasible if d.domain == domain]

        def device_score(d: DeviceState) -> float:
            return (
                self.MICRO_WEIGHTS[0] * d.utilization
                + self.MICRO_WEIGHTS[1] * (1.0 / max(1, d.free_pages))
                + self.MICRO_WEIGHTS[2] * min(1.0, d.queue_depth / 32.0)
                - self.MICRO_WEIGHTS[3] * d.locality
                - self.MICRO_WEIGHTS[4] * float(d.native)
                + self.MICRO_WEIGHTS[5] * (0.025 if self.use_residual and domain != request.preferred_domain else 0.0)
            )

        selected = min(candidates, key=lambda d: (device_score(d), d.device_id))
        return RouteProposal(request.request_id, domain, selected.device_id, device_score(selected),
                             0.025 if domain != request.preferred_domain else 0.0,
                             tuple(d.device_id for d in feasible))


@dataclass(frozen=True)
class PlacementProblem:
    stage_devices: tuple[tuple[str, ...], ...]
    edge_cost: Mapping[tuple[int, str], float]
    memory_pages: Mapping[tuple[int, str], int]
    device_memory_pages: Mapping[str, int]
    reliability: Mapping[tuple[int, str], float] = field(default_factory=dict)
    min_reliability: float = 0.0
    preferred_devices: tuple[str, ...] = ()
    preference_weight: float = 1.0
    cost_weight: float = 1.0
    reliability_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.stage_devices:
            raise ValueError("at least one placement stage is required")
        for stage, devices in enumerate(self.stage_devices):
            if not devices:
                raise ValueError(f"stage {stage} has no candidate devices")
        for key, value in self.memory_pages.items():
            _nonnegative_int(value, f"memory_pages{key}")
        for key, value in self.device_memory_pages.items():
            _nonnegative_int(value, f"device_memory_pages{key}")
        _finite(self.min_reliability, "min_reliability")
        if not 0 <= self.min_reliability <= 1:
            raise ValueError("min_reliability must lie in [0, 1]")


@dataclass(frozen=True)
class PlacementSolution:
    devices: tuple[str, ...]
    objective: float
    total_cost: float
    bottleneck_cost: float
    memory_by_device: Mapping[str, int]
    repaired: bool = False
    solver: str = "exact"


class PatchExactSolver:
    """Exact finite placement solver with cumulative memory/reliability checks."""

    def solve(self, problem: PlacementProblem) -> PlacementSolution:
        best: PlacementSolution | None = None
        for assignment in product(*problem.stage_devices):
            memory: dict[str, int] = {}
            valid = True
            costs: list[float] = []
            reliability = 1.0
            for stage, device in enumerate(assignment):
                memory[device] = memory.get(device, 0) + problem.memory_pages.get((stage, device), 0)
                if memory[device] > problem.device_memory_pages.get(device, 0):
                    valid = False
                    break
                cost = _finite(problem.edge_cost.get((stage, device), 0.0), "edge_cost")
                costs.append(cost)
                reliability *= _finite(problem.reliability.get((stage, device), 1.0), "reliability")
            if not valid or reliability < problem.min_reliability:
                continue
            total = sum(costs)
            bottleneck = max(costs, default=0.0)
            preference_penalty = sum(device not in problem.preferred_devices for device in assignment)
            objective = (problem.cost_weight * total + problem.reliability_weight * (1.0 - reliability)
                         + problem.preference_weight * preference_penalty)
            candidate = PlacementSolution(tuple(assignment), objective, total, bottleneck, dict(memory))
            if best is None or (candidate.objective, candidate.devices) < (best.objective, best.devices):
                best = candidate
        if best is None:
            raise ValueError("no feasible placement satisfies cumulative constraints")
        return best


class PatchLearnedSolver:
    """Small trainable imitation scorer with deterministic exact constraint repair.

    It learns a linear score over ``(cost, memory, reliability penalty,
    preferred)``.  ``gradient`` is exposed for tests and audit logs so the
    training path cannot silently replace model losses with detached constants.
    Feasibility is always checked before a learned assignment is returned.
    """

    def __init__(self, *, learning_rate: float = 0.05, epochs: int = 50):
        self.weights = [1.0, 0.2, 1.0, 1.0]
        self.learning_rate = _finite(learning_rate, "learning_rate")
        self.epochs = _nonnegative_int(epochs, "epochs")
        self.training_summary: dict[str, object] = {"epochs": 0, "examples": 0, "gradient_l1": 0.0}
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @staticmethod
    def _features(problem: PlacementProblem, stage: int, device: str) -> tuple[float, float, float, float]:
        return (float(problem.edge_cost.get((stage, device), 0.0)),
                float(problem.memory_pages.get((stage, device), 0)),
                1.0 - float(problem.reliability.get((stage, device), 1.0)),
                0.0 if device in problem.preferred_devices else 1.0)

    def gradient(self, problem: PlacementProblem, assignment: Sequence[str], target: Sequence[str]) -> tuple[float, ...]:
        if len(assignment) != len(target) or len(assignment) != len(problem.stage_devices):
            raise ValueError("assignment and target must cover every stage")
        grad = [0.0] * len(self.weights)
        for stage, (actual, wanted) in enumerate(zip(assignment, target)):
            if actual == wanted:
                continue
            a = self._features(problem, stage, actual)
            t = self._features(problem, stage, wanted)
            for j in range(len(grad)):
                grad[j] += a[j] - t[j]
        return tuple(grad)

    def fit(self, examples: Iterable[tuple[PlacementProblem, Sequence[str]]]) -> "PatchLearnedSolver":
        examples = list(examples)
        if not examples:
            raise ValueError("at least one placement example is required")
        last_l1 = 0.0
        for _ in range(self.epochs):
            total = [0.0] * len(self.weights)
            for problem, target in examples:
                target = tuple(target)
                exact = PatchExactSolver().solve(problem)
                if target != exact.devices:
                    # Training labels must be feasible exact labels; silently
                    # accepting an invalid target would corrupt the hard path.
                    raise ValueError("training target is not the exact feasible solution")
                grad = self.gradient(problem, tuple(self._predict_assignment(problem)), target)
                for j, value in enumerate(grad):
                    total[j] += value
            last_l1 = sum(abs(value) for value in total)
            scale = self.learning_rate / max(1, len(examples))
            for j, value in enumerate(total):
                self.weights[j] -= scale * value
        self.training_summary = {"epochs": self.epochs, "examples": len(examples), "gradient_l1": last_l1}
        return self

    def checkpoint_dict(self) -> dict:
        """Return the complete learned state with a strict schema."""
        return {
            "format": "prc-patch-learned-v1",
            "solver": "PatchLearnedSolver",
            "feature_names": ["edge_cost", "memory_pages", "reliability_penalty", "preference_penalty"],
            "weights": [float(value) for value in self.weights],
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "training_summary": dict(self.training_summary),
        }

    def save_checkpoint(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.checkpoint_dict()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        path.write_bytes(encoded)
        return hashlib.sha256(encoded).hexdigest()

    def load_checkpoint(self, path: str | Path) -> str:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"format", "solver", "feature_names", "weights", "learning_rate", "epochs", "training_summary"}
        if set(payload) != required or payload["format"] != "prc-patch-learned-v1" or payload["solver"] != "PatchLearnedSolver":
            raise ValueError("learned checkpoint schema mismatch")
        if payload["feature_names"] != ["edge_cost", "memory_pages", "reliability_penalty", "preference_penalty"]:
            raise ValueError("learned checkpoint feature schema mismatch")
        weights = payload["weights"]
        if type(weights) is not list or len(weights) != 4 or any(not math.isfinite(float(x)) for x in weights):
            raise ValueError("learned checkpoint weights are malformed")
        self.weights = [float(x) for x in weights]
        self.learning_rate = _finite(payload["learning_rate"], "learning_rate")
        self.epochs = _nonnegative_int(payload["epochs"], "epochs")
        if type(payload["training_summary"]) is not dict:
            raise ValueError("learned checkpoint training summary is malformed")
        self.training_summary = dict(payload["training_summary"])
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _predict_assignment(self, problem: PlacementProblem) -> tuple[str, ...]:
        out = []
        for stage, devices in enumerate(problem.stage_devices):
            out.append(min(devices, key=lambda d: (sum(w * x for w, x in zip(self.weights, self._features(problem, stage, d))), d)))
        return tuple(out)

    def solve(self, problem: PlacementProblem) -> PlacementSolution:
        # Enumerate only feasible assignments, but rank with learned scores.
        candidates = []
        for assignment in product(*problem.stage_devices):
            memory: dict[str, int] = {}
            costs: list[float] = []
            reliability = 1.0
            valid = True
            score = 0.0
            for stage, device in enumerate(assignment):
                memory[device] = memory.get(device, 0) + problem.memory_pages.get((stage, device), 0)
                if memory[device] > problem.device_memory_pages.get(device, 0):
                    valid = False
                    break
                score += sum(w * x for w, x in zip(self.weights, self._features(problem, stage, device)))
                costs.append(float(problem.edge_cost.get((stage, device), 0.0)))
                reliability *= float(problem.reliability.get((stage, device), 1.0))
            if valid and reliability >= problem.min_reliability:
                candidates.append((score, assignment, memory, costs, reliability))
        if not candidates:
            raise ValueError("learned placement has no feasible assignment")
        score, assignment, memory, costs, _reliability = min(candidates, key=lambda x: (x[0], x[1]))
        return PlacementSolution(tuple(assignment), score, sum(costs), max(costs, default=0.0), dict(memory), solver="learned")


@dataclass(frozen=True)
class PRCProposal:
    request_id: str
    state_version: int
    hard_max_output_tokens: int
    hard_need_pages: int
    soft_reserve_tokens: int
    predicted_mean_tokens: float
    predicted_variance_tokens2: float
    route: RouteProposal
    placement: PlacementSolution
    calibration_version: str

    def as_dict(self) -> dict:
        out = asdict(self)
        out["route"]["candidates"] = list(self.route.candidates)
        out["placement"]["memory_by_device"] = dict(self.placement.memory_by_device)
        return out


class PRCProposalPipeline:
    """Compose prediction, typed routing and PATCH before certificate auth."""

    def __init__(self, predictor: ResidualLengthPredictor, router: TypedRoute,
                 *, placement_solver: PatchExactSolver | PatchLearnedSolver | None = None,
                 block_size: int = 16, calibration_version: str = "empirical-v1"):
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be positive")
        self.predictor = predictor
        self.router = router
        self.placement_solver = placement_solver or PatchExactSolver()
        self.block_size = block_size
        self.calibration_version = calibration_version

    def propose(self, observation: PredictionObservation, devices: Sequence[DeviceState],
                *, placement_problem: PlacementProblem | None = None, state_version: int = 0) -> PRCProposal:
        if type(state_version) is not int or state_version < 0:
            raise ValueError("state_version must be nonnegative")
        prediction = self.predictor.predict(observation)
        route = self.router.choose(RouteRequest(observation.request_id, "default", 0), devices)
        if placement_problem is None:
            placement_problem = PlacementProblem(
                stage_devices=((route.device_id,),),
                edge_cost={(0, route.device_id): route.score},
                memory_pages={(0, route.device_id): 0},
                device_memory_pages={route.device_id: 1},
                preferred_devices=(route.device_id,),
            )
        placement = self.placement_solver.solve(placement_problem)
        hard_need = math.ceil((observation.prompt_tokens + observation.max_output_tokens) / self.block_size)
        return PRCProposal(observation.request_id, state_version, observation.max_output_tokens, hard_need,
                           prediction.soft_upper_tokens, prediction.remaining_mean_tokens,
                           prediction.remaining_variance_tokens2, route, placement,
                           prediction.calibration_version)
