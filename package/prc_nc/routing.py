"""Faithful two-stage DARC TypedRoute implementation.

The equations and validated coefficients are taken from the archived DARC
method section (``code/DARC_source/main.tex``).  A route is a typed action
``(assignment_domain, realised_device)``.  Feasibility is a hard filter; the
macro and micro scores only rank the surviving actions and never grant memory
or execution authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


def _finite(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _pressure(value: Any, name: str) -> float:
    value = _finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class RouteDevice:
    """Runtime device telemetry used by TypedRoute.

    Pressures are normalized by the telemetry producer.  The router accepts
    values above one (for overloaded devices) because the original DARC
    simulator clips only some pressure terms; no hidden clipping is performed
    here.  ``free_capacity`` and ``hard_need`` use the same unit (pages or
    tokens) and are checked before scoring.
    """

    device_id: str
    domain_id: str
    compute_pressure: float = 0.0
    memory_pressure: float = 0.0
    queue_pressure: float = 0.0
    locality: float = 0.0
    free_capacity: float = math.inf
    feasible: bool = True
    closed: bool = False
    native: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("device_id", "domain_id"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("compute_pressure", "memory_pressure", "queue_pressure", "locality", "free_capacity"):
            value = getattr(self, name)
            if name == "free_capacity" and math.isinf(float(value)):
                continue
            value = _finite(value, name)
            if name == "locality":
                if not 0.0 <= value <= 1.0:
                    raise ValueError("locality must lie in [0, 1]")
            elif value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class RouteRequest:
    request_id: str
    preferred_domain: str | None = None
    hard_need: float = 0.0
    allowed_domains: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if self.preferred_domain is not None and (type(self.preferred_domain) is not str or not self.preferred_domain):
            raise ValueError("preferred_domain must be non-empty when supplied")
        self_need = _finite(self.hard_need, "hard_need")
        if self_need < 0:
            raise ValueError("hard_need must be non-negative")
        if self.allowed_domains is not None and not all(isinstance(x, str) and x for x in self.allowed_domains):
            raise ValueError("allowed_domains must contain non-empty strings")


@dataclass(frozen=True)
class RouteProposal:
    request_id: str
    assignment_domain: str
    device_id: str
    macro_score: float
    micro_score: float
    total_score: float
    request_domain_residual: float
    domain_device_residual: float
    feasible_device_ids: tuple[str, ...]
    candidate_device_ids: tuple[str, ...]
    residual_enabled: bool
    score_components: Mapping[str, float] = field(default_factory=dict)

    # Existing adapter terminology uses ``domain`` and ``score``.  Keep these
    # read-only aliases while exposing the full typed action explicitly.
    @property
    def domain(self) -> str:
        return self.assignment_domain

    @property
    def score(self) -> float:
        return self.total_score

    @property
    def residual(self) -> float:
        return self.request_domain_residual + self.domain_device_residual


class NoFeasibleRoute(RuntimeError):
    """Raised when no open runtime device satisfies the hard request filter."""


class TypedRoute:
    """DARC's macro-domain then micro-device controller.

    Macro terms are ``(compute, memory, queue, locality deficit)`` with
    ``(0.24, 0.38, 0.30, 0.08)`` and ``alpha=0.025``.  Micro terms are
    ``(compute, memory, queue, locality deficit, domain queue, domain memory)``
    with ``(0.22, 0.32, 0.20, 0.14, 0.06, 0.06)`` and ``beta=0.08``.
    Residuals can be disabled for the no-residual ablation, but feasibility and
    the two-stage typed action remain active.
    """

    MACRO_WEIGHTS = (0.24, 0.38, 0.30, 0.08)
    MICRO_WEIGHTS = (0.22, 0.32, 0.20, 0.14, 0.06, 0.06)
    ALPHA = 0.025
    BETA = 0.08

    def __init__(self, *, use_residual: bool = True, residual_enabled: bool | None = None) -> None:
        self.use_residual = bool(use_residual if residual_enabled is None else residual_enabled)

    @staticmethod
    def _feasible_devices(request: RouteRequest, devices: Sequence[RouteDevice]) -> list[RouteDevice]:
        allowed = set(request.allowed_domains) if request.allowed_domains is not None else None
        return [
            d for d in devices
            if d.feasible and not d.closed and d.free_capacity >= request.hard_need
            and (allowed is None or d.domain_id in allowed)
        ]

    @staticmethod
    def _domain_pressure(members: Sequence[RouteDevice], field_name: str) -> float:
        # Equation (domain-pressure) in the DARC paper; the validated
        # simulator uses a uniform domain mean (the capacity weights sum to 1).
        if not members:
            return 0.0
        return sum(float(getattr(d, field_name)) for d in members) / len(members)

    @staticmethod
    def _best_locality(request: RouteRequest, domain: str, feasible: Sequence[RouteDevice]) -> float:
        members = [d for d in feasible if d.domain_id == domain]
        return max((d.locality for d in members), default=0.0)

    def _macro_score(
        self,
        request: RouteRequest,
        domain: str,
        all_devices: Sequence[RouteDevice],
        feasible: Sequence[RouteDevice],
    ) -> tuple[float, dict[str, float]]:
        members = [d for d in all_devices if d.domain_id == domain]
        c = self._domain_pressure(members, "compute_pressure")
        m = self._domain_pressure(members, "memory_pressure")
        q = self._domain_pressure(members, "queue_pressure")
        bestloc = self._best_locality(request, domain, feasible)
        residual = self.ALPHA if self.use_residual and request.preferred_domain is not None and domain != request.preferred_domain else 0.0
        score = self.MACRO_WEIGHTS[0] * c + self.MACRO_WEIGHTS[1] * m + self.MACRO_WEIGHTS[2] * q + self.MACRO_WEIGHTS[3] * (1.0 - bestloc) + residual
        return score, {
            "domain_compute": c, "domain_memory": m, "domain_queue": q,
            "best_locality": bestloc, "request_domain_residual": residual,
        }

    def _micro_score(
        self,
        request: RouteRequest,
        assignment_domain: str,
        device: RouteDevice,
        all_devices: Sequence[RouteDevice],
    ) -> tuple[float, dict[str, float]]:
        domain_members = [d for d in all_devices if d.domain_id == device.domain_id]
        dq = self._domain_pressure(domain_members, "queue_pressure")
        dm = self._domain_pressure(domain_members, "memory_pressure")
        residual = self.BETA if self.use_residual and device.domain_id != assignment_domain else 0.0
        score = (
            self.MICRO_WEIGHTS[0] * device.compute_pressure
            + self.MICRO_WEIGHTS[1] * device.memory_pressure
            + self.MICRO_WEIGHTS[2] * device.queue_pressure
            + self.MICRO_WEIGHTS[3] * (1.0 - device.locality)
            + self.MICRO_WEIGHTS[4] * dq
            + self.MICRO_WEIGHTS[5] * dm
            + residual
        )
        return score, {"device_compute": device.compute_pressure, "device_memory": device.memory_pressure,
                       "device_queue": device.queue_pressure, "locality_deficit": 1.0 - device.locality,
                       "realized_domain_queue": dq, "realized_domain_memory": dm,
                       "domain_device_residual": residual}

    def choose(self, request: RouteRequest, devices: Sequence[RouteDevice]) -> RouteProposal:
        if not isinstance(request, RouteRequest):
            raise TypeError("request must be RouteRequest")
        all_devices = tuple(devices)
        if not all_devices:
            raise NoFeasibleRoute("route received no runtime devices")
        feasible = self._feasible_devices(request, all_devices)
        if not feasible:
            raise NoFeasibleRoute("no feasible device satisfies hard route constraints")

        # The original DARC action set is D x F, so a domain with no currently
        # feasible GPU remains a valid macro assignment and incurs beta when
        # the micro stage realises on another domain.
        domains = sorted({d.domain_id for d in all_devices})
        if len(feasible) == 1 and len({d.device_id for d in all_devices if not d.closed}) == 1:
            # True single-device degeneration: do not invent an alternative
            # domain merely because its aggregate pressure scores lower.
            assignment_domain = feasible[0].domain_id
            macro_score, macro_parts = self._macro_score(request, assignment_domain, all_devices, feasible)
        else:
            macro_ranked = [(self._macro_score(request, d, all_devices, feasible), d) for d in domains]
            (macro_score, macro_parts), assignment_domain = min(macro_ranked, key=lambda item: (item[0][0], item[1]))

        micro_ranked = [(self._micro_score(request, assignment_domain, d, all_devices), d) for d in feasible]
        (micro_score, micro_parts), selected = min(micro_ranked, key=lambda item: (item[0][0], item[1].device_id))
        candidates = tuple(d.device_id for d in feasible if d.domain_id == assignment_domain)
        return RouteProposal(
            request_id=request.request_id,
            assignment_domain=assignment_domain,
            device_id=selected.device_id,
            macro_score=macro_score,
            micro_score=micro_score,
            total_score=macro_score + micro_score,
            request_domain_residual=macro_parts["request_domain_residual"],
            domain_device_residual=micro_parts["domain_device_residual"],
            feasible_device_ids=tuple(d.device_id for d in feasible),
            candidate_device_ids=candidates,
            residual_enabled=self.use_residual,
            score_components={**macro_parts, **micro_parts},
        )

    # ``route`` is a clear runtime synonym used by the adapter.
    route = choose


# Names used by some integration code.
DeviceState = RouteDevice
TypedRouteRequest = RouteRequest
TypedRouteProposal = RouteProposal

__all__ = [
    "RouteDevice", "DeviceState", "RouteRequest", "TypedRouteRequest", "RouteProposal",
    "TypedRouteProposal", "TypedRoute", "NoFeasibleRoute",
]
