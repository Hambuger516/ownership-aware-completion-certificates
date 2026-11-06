"""Versioned certificate authority. Tickets never authorize unverified state.

prepare() is side-effect-free on physical state. A real allocator must serialize
prepare/allocation/commit at the same authority. Commit mismatch is fail-stop:
we cannot pretend already allocated physical pages have been rolled back.
"""

from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
from threading import RLock
from uuid import uuid4
from .model import State, Group, canonical, validate_growth, apply_private_growth
from .admission import POLICIES, AdmissionContext, AdmissionDeferred, decide
from .algorithms import indexed_closure, LinearBuildMarginIndex, linear_margins, interval_trial


class Refused(ValueError):
    """Legal request, but no supported completion certificate was found."""


class StaleState(RuntimeError):
    """Retry against current authoritative state; do not commit the old ticket."""


@dataclass(frozen=True)
class Ticket:
    identity: str
    version: int
    path: str
    admission_decision: object = None


def event_index(index, before: State, after: State):
    """Exact fixed-rank Δf - Δd + ΔR, including owner/pin/free changes.

    Clone for transactional rejection. This costs O(nP) copying: do NOT label
    the whole transaction O(log n). Each individual tree edit is logarithmic.
    Structural cohort/rank changes must rebuild instead of calling this.
    """
    if (before.n, before.p) != (after.n, after.p):
        raise ValueError("fixed-rank transition required")
    candidate = deepcopy(index)
    old = {g.key: g for g in before.groups}
    new = {g.key: g for g in after.groups}
    for pool in range(before.p):
        candidate.delta_free(pool, after.free[pool] - before.free[pool])
    for i in range(before.n):
        for pool in range(before.p):
            candidate.delta_need(i, pool, after.need[i][pool] - before.need[i][pool])
    for key in old.keys() | new.keys():
        a, b = old.get(key), new.get(key)
        if a and b and (a.pool, a.count) == (b.pool, b.count):
            candidate.move_frontier(a.pool, a.count, candidate.frontier(a), candidate.frontier(b))
        else:
            # Remove/add future releases only; actual Δfree was counted above.
            if a:
                candidate.trees[a.pool].add(candidate.frontier(a), before.n - 1, -a.count)
            if b:
                candidate.trees[b.pool].add(candidate.frontier(b), before.n - 1, b.count)
    return candidate


class CertificateController:
    def __init__(self, state: State, *, audit=True, certificate_variant="shared_incremental", admission_policy="unrestricted", private_refusal_cache=False):
        if type(audit) is not bool:
            raise ValueError("audit must be bool")
        if certificate_variant not in ("shared_incremental", "force_full", "exclusive_only", "common_full_c0", "common_full_c1"):
            raise ValueError("unsupported certificate_variant")
        if admission_policy not in POLICIES:
            raise ValueError("unsupported admission policy")
        if type(private_refusal_cache) is not bool or (private_refusal_cache and certificate_variant != "shared_incremental"):
            raise ValueError("private refusal cache requires a bool and shared_incremental")
        self.private_refusal_cache = private_refusal_cache
        self._private_refusals = set()
        self.admission_policy = admission_policy
        self.certificate_variant = certificate_variant
        self._common_full = certificate_variant in ("common_full_c0", "common_full_c1")
        if self._common_full:
            c0, c1 = self._common_full_orders(state)
            order = self._common_full_choice(c0, c1, state.n)
        else:
            order = self._complete_order(state)
        if len(order) != state.n:
            raise Refused("initial state has no complete witness")
        self.state = state
        self.version = 0
        self.audit = audit
        self.faulted = False
        self.order = order
        self.index = LinearBuildMarginIndex(state, order) if state.n else None
        self._tickets = {}
        self._lock = RLock()
        self.stats = {"fast": 0, "fallback": 0, "structural": 0, "rebuild": 0, "oracle": 0, "commits": 0,
                      "forced_full": 0, "exclusive_checks": 0, "policy_refusals": 0,
                      "initial_oracle": 2 if self._common_full else 1,
                      "common_full_checks": 0, "common_c0_checks": 0, "common_c1_checks": 0,
                      "common_overlap_checks": 0, "common_c1_only_checks": 0,
                      "admission_checks": 0, "admission_deferrals": 0, "admission_c0_checks": 0, "admission_c1_checks": 0,
                      "private_refusal_cache_enabled": private_refusal_cache,
                      "private_refusal_hits": 0, "private_refusal_audited_hits": 0,
                      "private_refusal_stores": 0, "private_refusal_invalidations": 0}

    @staticmethod
    def _common_full_orders(state):
        """Same two complete searches, in the same order, in BOTH common arms.

        C0 pins currently shared groups only in its certificate view. The real
        physical state stays unchanged. Both searches are paid in the control
        path, not precomputed offline or omitted from the C0 cost.
        """
        c0_state = State(state.free, state.need, tuple(
            Group(g.key, g.pool, g.count, g.owners, g.pinned or len(g.owners) > 1)
            for g in state.groups))
        c0 = indexed_closure(c0_state)
        c1 = indexed_closure(state)
        if not set(c0) <= set(c1):
            raise AssertionError("conservative closure exceeds shared closure")
        return c0, c1

    def _common_full_choice(self, c0, c1, n):
        # A C0-valid order is also valid physically under C1. On the overlap,
        # select this same fresh order in BOTH arms, independent of old caches.
        if len(c0) == n:
            return c0
        if self.certificate_variant == "common_full_c1" and len(c1) == n:
            return c1
        raise Refused("post-state has no complete witness for common_full view")

    def _prepare_common_full(self, post, *, structural, admission_context=None):
        """Full C0/C1 pair plus identical fresh physical index maintenance.

        The active view changes acceptance only. In the overlapping feasible
        region, order/index/timer input are identical for an identical post.
        After true acceptance divergence, whole trajectories need not match.
        """
        self.stats["oracle"] += 2
        self.stats["forced_full"] += 1
        self.stats["exclusive_checks"] += 1
        self.stats["common_full_checks"] += 1
        self.stats["common_c0_checks"] += 1
        self.stats["common_c1_checks"] += 1
        c0, c1 = self._common_full_orders(post)
        self.stats["common_overlap_checks"] += len(c0) == post.n
        self.stats["common_c1_only_checks"] += len(c0) != post.n and len(c1) == post.n
        decision = self._admission_decision(admission_context, len(c0)==post.n, len(c1)==post.n)
        try:
            order = self._common_full_choice(c0, c1, post.n)
        except Refused as exc:
            self.stats["policy_refusals"] += 1
            exc.admission_decision = decision
            raise
        if decision is not None and not decision.policy_allows:
            self.stats["admission_deferrals"] += 1
            raise AdmissionDeferred(decision)
        index = LinearBuildMarginIndex(post, order) if post.n else None
        ticket = Ticket(uuid4().hex, self.version, "common_full", decision)
        self._tickets[ticket.identity] = (ticket, post, order, index, None)
        self.stats["rebuild"] += 1
        if structural:
            self.stats["structural"] += 1
        return ticket

    def _admission_decision(self, context, c0_complete, c1_complete):
        if context is None:
            return None
        self.stats["admission_checks"] += 1
        return decide(self.admission_policy, context, c0_complete, c1_complete)

    def _prepare_incremental_admission(self, post, context):
        # New membership is structural: pay complete C0 then C1 in both utility
        # arms. Existing resident growth retains its original incremental path.
        self.stats["oracle"] += 2
        self.stats["admission_c0_checks"] += 1
        self.stats["admission_c1_checks"] += 1
        c0,c1=self._common_full_orders(post)
        decision=self._admission_decision(context,len(c0)==post.n,len(c1)==post.n)
        if len(c1)!=post.n:
            exc=Refused("structural admission post-state has no complete shared witness")
            exc.admission_decision=decision
            raise exc
        if not decision.policy_allows:
            self.stats["admission_deferrals"] += 1
            raise AdmissionDeferred(decision)
        order=c1
        index=LinearBuildMarginIndex(post,order) if post.n else None
        ticket=Ticket(uuid4().hex,self.version,"structural_admission",decision)
        self._tickets[ticket.identity]=(ticket,post,order,index,None)
        self.stats["structural"] += 1;self.stats["rebuild"] += 1
        return ticket

    def _complete_order(self, state):
        """Policy view only; the authority retains the real physical ownership.

        exclusive_only cannot fund a successor from a currently shared group.
        A returned order is therefore also valid under the actual release map.
        This is a conservative information ablation, not altered physical KV.
        """
        if self.certificate_variant == "exclusive_only":
            state = State(state.free, state.need, tuple(
                Group(g.key, g.pool, g.count, g.owners, g.pinned or len(g.owners) > 1)
                for g in state.groups))
        return indexed_closure(state)

    def _prepare_full(self, post, *, structural):
        """Conservative information arm; no interval/event-index acceptance."""
        self.stats["oracle"] += 1
        self.stats["forced_full"] += 1
        self.stats["exclusive_checks"] += self.certificate_variant == "exclusive_only"
        order = self._complete_order(post)
        if len(order) != post.n:
            self.stats["policy_refusals"] += 1
            raise Refused("post-state has no complete witness for certificate_variant")
        index = LinearBuildMarginIndex(post, order) if post.n else None
        ticket = Ticket(uuid4().hex, self.version, "forced_full")
        self._tickets[ticket.identity] = (ticket, post, order, index, None)
        self.stats["rebuild"] += 1
        if structural:
            self.stats["structural"] += 1
        return ticket

    def _forced_shared_order(self, post):
        """Always run the shared oracle, preserving the common witness policy.

        Index maintenance below is identical to shared_incremental. An old
        valid witness is retained in both arms, so the timer head cannot change
        merely because one arm performs this additional full verification.
        """
        self.stats["oracle"] += 1
        self.stats["forced_full"] += 1
        order = indexed_closure(post)
        if len(order) != post.n:
            self.stats["policy_refusals"] += 1
            raise Refused("post-state has no complete shared witness")
        return order

    def _ready(self):
        if self.faulted:
            raise RuntimeError("allocator mismatch: authority is faulted")

    def prepare_growth(self, growth, *, expected_version=None):
        with self._lock:
            self._ready()
            if expected_version is not None and expected_version != self.version:
                raise StaleState("old version")
            z = validate_growth(self.state, growth)
            sparse = [(i, p, z[i][p]) for i in range(self.state.n) for p in range(self.state.p) if z[i][p]]
            refusal_key = tuple(sparse)
            # Every private-growth transition weakly decreases each fixed-order
            # prefix margin. Hence it cannot repair ANY previously infeasible
            # proposed growth in this structural epoch (proof in study notes).
            # Bounds/immediate capacity are validated above even on cache hits.
            if self.private_refusal_cache and refusal_key in self._private_refusals:
                self.stats["private_refusal_hits"] += 1
                if self.audit:
                    full = indexed_closure(apply_private_growth(self.state, z))
                    self.stats["oracle"] += 1
                    self.stats["private_refusal_audited_hits"] += 1
                    if len(full) == self.state.n:
                        self.faulted = True
                        raise AssertionError("monotone refusal cache disagrees with full closure")
                raise Refused("private growth loses every complete witness")
            post = apply_private_growth(self.state, z)
            if self._common_full:
                return self._prepare_common_full(post, structural=False)
            if self.certificate_variant == "exclusive_only":
                return self._prepare_full(post, structural=False)
            forced_order = self._forced_shared_order(post) if self.certificate_variant == "force_full" else None
            fixed = self.index is None or interval_trial(self.index, sparse)
            path = "fast" if fixed else "fallback"
            candidate = None
            order = self.order
            # Normal mode genuinely omits full closure when the fixed witness passes.
            if not fixed or self.audit:
                order_full = forced_order if forced_order is not None else indexed_closure(post)
                self.stats["oracle"] += forced_order is None
                if fixed and len(order_full) != post.n:
                    raise AssertionError("fixed guard disagrees with oracle")
                if not fixed:
                    if len(order_full) != post.n:
                        if self.private_refusal_cache:
                            if len(self._private_refusals) >= 512:
                                self._private_refusals.clear()
                                self.stats["private_refusal_invalidations"] += 1
                            self._private_refusals.add(refusal_key)
                            self.stats["private_refusal_stores"] += 1
                        raise Refused("private growth loses every complete witness")
                    order = order_full
                    candidate = LinearBuildMarginIndex(post, order) if post.n else None
            ticket = Ticket(uuid4().hex, self.version, path)
            self._tickets[ticket.identity] = (ticket, post, order, candidate, sparse)
            self.stats[path] += 1
            return ticket

    def prepare_state(self, post: State, *, same_cohort=True, expected_version=None, admission_context=None):
        """Caller attests unchanged rank identities when same_cohort=True.

        This validates the proposed state, not the safety of an external RPC,
        an uncharged temporary copy, or changing a request's output contract.
        """
        with self._lock:
            self._ready()
            if expected_version is not None and expected_version != self.version:
                raise StaleState("old version")
            if type(same_cohort) is not bool:
                raise ValueError("same_cohort must be bool")
            if post.capacity != self.state.capacity:
                raise ValueError("capacity changes need a new episode")
            if (post.n > self.state.n and self.admission_policy != "unrestricted"
                    and admission_context is None):
                raise ValueError("filtered new admission requires immutable cap context")
            if admission_context is not None:
                if type(admission_context) is not AdmissionContext or same_cohort or post.n != self.state.n + 1:
                    raise ValueError("admission context requires new resident structural transaction")
                if self.certificate_variant not in ("common_full_c0", "common_full_c1", "shared_incremental", "force_full"):
                    raise ValueError("admission policy unsupported for this certificate variant")
            if self._common_full:
                return self._prepare_common_full(post, structural=True, admission_context=admission_context)
            if admission_context is not None:
                return self._prepare_incremental_admission(post,admission_context)
            if self.certificate_variant == "exclusive_only":
                return self._prepare_full(post, structural=True)
            forced_order = self._forced_shared_order(post) if self.certificate_variant == "force_full" else None
            index = None
            order = self.order
            fixed = False
            if same_cohort and post.n == self.state.n and self.index is not None:
                index = event_index(self.index, self.state, post)
                fixed = index.certificate_nonnegative
            if not fixed or self.audit:
                full = forced_order if forced_order is not None else indexed_closure(post)
                self.stats["oracle"] += forced_order is None
                if fixed and len(full) != post.n:
                    raise AssertionError("event calculus disagrees with oracle")
                if not fixed:
                    if len(full) != post.n:
                        raise Refused("structural post-state has no complete witness")
                    order = full
                    index = LinearBuildMarginIndex(post, order) if post.n else None
                    self.stats["rebuild"] += 1
            ticket = Ticket(uuid4().hex, self.version, "structural")
            self._tickets[ticket.identity] = (ticket, post, order, index, None)
            self.stats["structural"] += 1
            return ticket

    def candidate(self, ticket):
        with self._lock:
            self._ready()
            entry = self._tickets.get(ticket.identity)
            if entry is None or entry[0] != ticket:
                raise ValueError("unknown ticket")
            if ticket.version != self.version:
                raise StaleState("stale candidate")
            return entry[1]  # State is deeply immutable.

    def abort(self, ticket):
        with self._lock:
            self._tickets.pop(ticket.identity, None)

    def commit(self, ticket, observed: State):
        with self._lock:
            post = self.candidate(ticket)
            if canonical(observed) != canonical(post):
                self.faulted = True
                raise RuntimeError("physical commit differs from certified post-state")
            _, _, order, index, sparse = self._tickets[ticket.identity]
            if ticket.path == "fast":
                index = self.index
                if index is not None:
                    for i, p, pages in sparse:
                        index.private_growth(i, p, pages)
            if self.audit and index is not None and index.table() != linear_margins(observed, order):
                self.faulted = True
                raise AssertionError("committed margins do not match physical state")
            # Any structural commit (even an identical-state one) invalidates:
            # release, ownership/pin/cohort/contract edits may improve margins.
            if sparse is None:
                self._private_refusals.clear()
                self.stats["private_refusal_invalidations"] += 1
            self.state = observed
            self.index = index
            self.order = order
            self.version += 1
            self._tickets.clear()
            self.stats["commits"] += 1
            return self.version
