"""Single-authority allocator transactions with charged execution permissions.

This is a synchronous physical-backend contract. The lock spans validation,
allocation and observation. A receipt is only produced after that callback has
returned and an unchanged physical snapshot proves that no allocation occurred.
No promise is made about stopping another process or asynchronous CUDA writes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from typing import Callable

from .controller import CertificateController, Refused, StaleState
from .admission import AdmissionContext, AdmissionDeferred
from .model import State, canonical, finish_request


@dataclass(frozen=True)
class AllocationGrant:
    grant_id: str
    action_id: str
    request_id: str
    worker_id: str
    pool_id: str
    state_version: int
    pages: tuple[int, ...]
    admission_decision: object = None


class AllocationAuthority:
    """Own confirmed/virtual state; only issued grants can call the allocator."""

    def __init__(self, state: State, request_ids=(), *, worker_id="gpu0", pool_id="kv0", audit=True, on_event=None,
                 certificate_variant="shared_incremental", admission_policy="unrestricted", private_refusal_cache=False):
        self.ids = self._ids(state, request_ids)
        self.controller = CertificateController(state, audit=audit, certificate_variant=certificate_variant, admission_policy=admission_policy, private_refusal_cache=private_refusal_cache)
        self.confirmed = state
        self.virtual = state
        self.worker_id, self.pool_id = worker_id, pool_id
        self.version = 0
        self.faulted = False
        self.pending = None
        self._grants = {}
        self._outcomes = {}
        self._stopped = set()
        self._terminals = {}
        self._lock = RLock()
        self._on_event = on_event or (lambda record: None)

    @staticmethod
    def _ids(state, ids):
        ids = tuple(ids)
        if len(ids) != state.n or len(set(ids)) != len(ids) or any(type(i) is not str or not i for i in ids):
            raise ValueError("unique request IDs must match state rows")
        return ids

    def _ready(self):
        if self.faulted:
            raise RuntimeError("physical mismatch: authority stopped")

    def _emit(self, kind, **values):
        self._on_event(dict(kind=kind, state_version=self.version, worker_id=self.worker_id,
                            pool_id=self.pool_id, **values))

    def _match(self, state, ids):
        return tuple(ids) == self.ids and canonical(state) == canonical(self.confirmed)

    def _fail(self, reason, **values):
        self.faulted = True
        self._emit("fault", reason=reason, **values)
        raise RuntimeError(reason)

    def authorize(self, action_id, request_id, post: State, post_ids, *, expected_version, private_growth=None, admission_context=None):
        """Certify and CHARGE post-state before returning an execution permission.

        Admission, shared-prefix attachment, changed placement and ownership use
        full structural transactions. Only caller-proven private growth may use
        the growth fast path, and its computed candidate must match post exactly.
        """
        with self._lock:
            self._ready()
            post_ids = self._ids(post, post_ids)
            if not action_id or request_id not in post_ids:
                raise ValueError("action/request identity missing")
            if admission_context is not None and type(admission_context) is not AdmissionContext:
                raise ValueError("immutable admission context required")
            payload = (request_id, post, post_ids, expected_version, admission_context)
            if action_id in self._grants:
                old_payload, grant = self._grants[action_id]
                if old_payload != payload:
                    raise ValueError("action identity reused with different payload")
                return grant
            if request_id in self._terminals:
                raise Refused("terminal request cannot be authorized again")
            if expected_version != self.version:
                raise StaleState("old authoritative state version")
            if self.pending is not None:
                raise Refused("synchronous allocator must settle current grant before another action")
            new_ids=set(post_ids)-set(self.ids)
            if not set(self.ids)<=set(post_ids) or (new_ids and new_ids!={request_id}):
                raise ValueError("allocation may add only its candidate and cannot remove incumbents")
            new_admission=bool(new_ids)
            if admission_context is not None:
                if (not new_admission or admission_context.candidate_id!=request_id
                        or admission_context.authority_version!=expected_version
                        or tuple(r for r,_ in admission_context.incumbent_caps)!=tuple(sorted(self.ids))):
                    raise ValueError("admission context candidate/version/incumbents differ from Authority")
            elif new_admission and self.controller.admission_policy!="unrestricted":
                raise ValueError("filtered admission requires actual immutable cap context")
            if post.capacity != self.confirmed.capacity:
                raise ValueError("capacity must remain fixed")
            pages = tuple(a-b for a,b in zip(self.confirmed.free, post.free))
            if any(x < 0 for x in pages):
                raise ValueError("allocation cannot claim an unobserved release")
            if private_growth is not None:
                if post_ids != self.ids:
                    raise ValueError("private growth cannot change cohort")
                ticket = self.controller.prepare_growth(private_growth)
                if canonical(self.controller.candidate(ticket)) != canonical(post):
                    self.controller.abort(ticket)
                    raise ValueError("private growth does not describe structural change")
            else:
                ticket = self.controller.prepare_state(post, same_cohort=post_ids == self.ids, admission_context=admission_context)
            # controller.state is the VIRTUAL charged state after this commit.
            # The confirmed physical state deliberately remains unchanged.
            before = self.confirmed
            before_ids = self.ids
            self.controller.commit(ticket, post)
            self.virtual = post
            self.version += 1
            grant = AllocationGrant(action_id+":grant", action_id, request_id, self.worker_id,
                                    self.pool_id, self.version, pages, ticket.admission_decision)
            self.pending = (grant, before, before_ids, post, post_ids)
            self._grants[action_id] = (payload, grant)
            self._emit("authorize", request_id=request_id, action_id=action_id,
                       grant_id=grant.grant_id, authorized_pages=pages,
                       confirmed_free=before.free, virtual_free=post.free,
                       execution_confirmed=False, transaction_path=ticket.path,
                       certificate_variant=self.controller.certificate_variant,
                       admission_decision=ticket.admission_decision.summary() if ticket.admission_decision is not None else None,
                       witness_sha256=hashlib.sha256(json.dumps(
                           [post_ids[i] for i in self.controller.order], separators=(",", ":")
                       ).encode()).hexdigest())
            return grant

    def _refund_failed(self, grant, observed, ids):
        if not self._match(observed, ids):
            self._fail("failed allocator changed physical state; refund forbidden", grant_id=grant.grant_id)
        self._stopped.add(grant.grant_id)
        self._emit("stop", request_id=grant.request_id, grant_id=grant.grant_id,
                   evidence="synchronous callback returned; physical state unchanged")
        ticket = self.controller.prepare_state(self.confirmed, same_cohort=False)
        self.controller.commit(ticket, self.confirmed)
        self.virtual = self.confirmed
        self.pending = None
        self.version += 1
        self._outcomes[grant.grant_id] = None
        self._emit("refund", request_id=grant.request_id, grant_id=grant.grant_id, refunded_pages=grant.pages)

    def dispatch(self, grant, allocate: Callable, observe: Callable):
        """Only this method dispatches; replay cannot execute an allocation twice."""
        with self._lock:
            self._ready()
            if not isinstance(grant, AllocationGrant) or self._grants.get(grant.action_id, (None,None))[1] != grant:
                raise ValueError("execution requires an authority-issued grant")
            if grant.grant_id in self._stopped:
                raise Refused("stopped grant cannot execute")
            if grant.grant_id in self._outcomes:
                return self._outcomes[grant.grant_id]
            if self.pending is None or self.pending[0] != grant:
                raise StaleState("grant no longer pending")
            observed, ids = observe()
            if not self._match(observed, ids):
                self._fail("physical state changed between authorization and dispatch", grant_id=grant.grant_id)
            post, post_ids = self.pending[3:]
            try:
                result = allocate()
            except BaseException:
                after, ids = observe()
                self._refund_failed(grant, after, ids)
                raise
            after, ids = observe()
            if result is None:
                self._refund_failed(grant, after, ids)
                return None
            if tuple(ids) != post_ids or canonical(after) != canonical(post):
                self._fail("authorized allocation differs from physical result", grant_id=grant.grant_id,
                           authorized_free=post.free, observed_free=after.free)
            self._emit("execute", request_id=grant.request_id, grant_id=grant.grant_id,
                       executed_pages=grant.pages, physical_free=after.free)
            self.confirmed = self.virtual = after
            self.ids = post_ids
            # Only physical page labels changed; margins/order are unchanged.
            self.controller.state = after
            self.pending = None
            self.version += 1
            self._outcomes[grant.grant_id] = result
            self._emit("acknowledge", request_id=grant.request_id, grant_id=grant.grant_id,
                       acknowledged_pages=grant.pages, confirmed_free=after.free, virtual_free=after.free)
            return result

    def stop_unexecuted(self, grant, observe):
        """Stop competes with dispatch under the same lock, and verifies pages."""
        with self._lock:
            self._ready()
            if self.pending is None or self.pending[0] != grant:
                raise Refused("only an outstanding unexecuted grant can be stopped")
            self._refund_failed(grant, *observe())

    def release(self, request_id, release: Callable, observe: Callable, *, cancelled=False):
        with self._lock:
            self._ready()
            if self.pending is not None:
                raise Refused("settle grant before physical release")
            if request_id in self._terminals:
                return self._terminals[request_id]
            if request_id in self.ids:
                expected, _ = finish_request(self.confirmed, self.ids.index(request_id))
                ids = tuple(x for x in self.ids if x != request_id)
            else:  # waiting cancellation has no pages and must keep state intact
                expected, ids = self.confirmed, self.ids
            release()
            actual, actual_ids = observe()
            if tuple(actual_ids) != ids or canonical(actual) != canonical(expected):
                self._fail("physical release lacks last-owner evidence", request_id=request_id)
            try:
                ticket = self.controller.prepare_state(actual, same_cohort=False)
            except Refused:
                # Physical release already happened. Never keep serving against
                # an older confirmed/controller state after a failed recheck.
                self._fail("released physical state has no policy witness", request_id=request_id)
            self.controller.commit(ticket, actual)
            self.confirmed = self.virtual = actual
            self.ids = ids
            self.version += 1
            outcome = "cancelled" if cancelled else "completed"
            self._terminals[request_id] = outcome
            self._emit("finish", request_id=request_id, outcome=outcome, confirmed_free=actual.free,
                       release_observed=True)
            return outcome

    def drain_cut(self):
        with self._lock:
            self._ready()
            if self.pending is not None or self.virtual != self.confirmed:
                raise Refused("resolve all in-flight rights before drain cut")
            return tuple(self.ids[i] for i in self.controller.order)
