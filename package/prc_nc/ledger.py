"""Executable single-authority grant/execute/ACK/fence/refund protocol (§7.1).

The included worker is an IN-PROCESS PAGE BACKEND, not a CUDA/RPC fence. Its
execute and stop transitions share this authority lock, so a stop receipt really
prevents later allocation in this backend. A GPU/network integration must provide
the same physical guarantee; a timeout or epoch number is never such evidence.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from threading import RLock
from uuid import uuid4
from .model import State, apply_private_growth, finish_request, canonical, valid_witness
from .algorithms import indexed_closure, linear_margins
from .controller import Refused, StaleState


@dataclass(frozen=True)
class Grant:
    identity: str
    request: str
    issued: tuple[int, ...]
    executed: tuple[int, ...]
    acknowledged: tuple[int, ...]
    refunded: tuple[int, ...]
    stopped: bool = False


@dataclass(frozen=True)
class StopReceipt:
    grant: str
    executed: tuple[int, ...]
    nonce: str


@dataclass(frozen=True)
class RuntimeGrant:
    """A synchronous adapter grant bound to one hard request envelope."""

    identity: str
    request: str
    hard_pages: int
    issued: int
    executed: int = 0
    acknowledged: int = 0
    refunded: int = 0
    stopped: bool = False


class CausalCommitBridge:
    """Small runtime-facing CausalCommit state machine.

    The full :class:`CommitmentLedger` is the mathematical multi-pool backend.
    This bridge is the adapter boundary: each real allocator action gets a
    grant before dispatch, then an execute/ack after the allocator reports the
    physical result.  It deliberately refuses timeout-only refunds and keeps
    duplicate event IDs idempotent.  It does not pretend to fence a remote
    worker; the vLLM adapter uses it only in its synchronous, same-process
    execution domain.
    """

    def __init__(self, *, on_event=None):
        self.grants: dict[str, RuntimeGrant] = {}
        self.terminals: dict[str, str] = {}
        self.receipts: dict[str, StopReceipt] = {}
        self.events: list[dict] = []
        self.version = 0
        self._messages = {}
        self._on_event = on_event
        self._lock = RLock()

    @staticmethod
    def _pages(value: int, name: str = "pages") -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value

    def _apply(self, event_id, payload, action):
        if type(event_id) is not str or not event_id:
            raise ValueError("nonempty event identity required")
        with self._lock:
            old = self._messages.get(event_id)
            if old is not None:
                if old[0] != payload:
                    raise ValueError("event identity reused with different payload")
                return old[1]
            result = action()
            self.version += 1
            record = {"event_id": event_id, "kind": payload[0], "version": self.version}
            if payload[0] == "authorize":
                record.update(grant_id=payload[1], request_id=payload[2])
            elif payload[0] in ("execute", "acknowledge", "stop"):
                record.update(grant_id=payload[1], request_id=self.grants[payload[1]].request)
            elif payload[0] == "refund":
                record.update(grant_id=payload[1].grant, request_id=self.grants[payload[1].grant].request)
            elif payload[0] == "finish":
                record.update(request_id=payload[1])
            self._messages[event_id] = (payload, result)
            self.events.append(record)
            if self._on_event is not None:
                self._on_event(record)
            return result

    def authorize(self, event_id, grant_id, request, amount, hard_pages, *, state_version=None):
        amount = self._pages(amount, "amount")
        hard_pages = self._pages(hard_pages, "hard_pages")
        if not grant_id or not request:
            raise ValueError("grant_id and request must be nonempty")

        def action():
            if state_version is not None and (type(state_version) is not int or state_version < 0):
                raise ValueError("state_version must be a nonnegative integer")
            existing = self.grants.get(grant_id)
            if existing is not None:
                if (existing.request, existing.issued, existing.hard_pages) != (request, amount, hard_pages):
                    raise ValueError("grant identity collision")
                return grant_id
            used = sum(g.issued - g.refunded for g in self.grants.values() if g.request == request)
            if used + amount > hard_pages:
                raise ValueError("authorization exceeds hard request envelope")
            self.grants[grant_id] = RuntimeGrant(grant_id, request, hard_pages, amount)
            return grant_id

        return self._apply(event_id, ("authorize", grant_id, request, amount, hard_pages, state_version), action)

    def execute(self, event_id, grant_id, cumulative):
        cumulative = self._pages(cumulative, "cumulative")

        def action():
            grant = self.grants[grant_id]
            if cumulative < grant.executed or cumulative > grant.issued - grant.refunded:
                raise ValueError("execution exceeds issued rights or is not cumulative")
            if grant.stopped and cumulative > grant.executed:
                raise ValueError("stopped grant cannot execute more pages")
            self.grants[grant_id] = replace(grant, executed=cumulative)
            return cumulative

        return self._apply(event_id, ("execute", grant_id, cumulative), action)

    def acknowledge(self, event_id, grant_id, cumulative):
        cumulative = self._pages(cumulative, "cumulative")

        def action():
            grant = self.grants[grant_id]
            if cumulative < grant.acknowledged or cumulative > grant.executed:
                raise ValueError("acknowledgement exceeds executed rights")
            self.grants[grant_id] = replace(grant, acknowledged=cumulative)
            return cumulative

        return self._apply(event_id, ("acknowledge", grant_id, cumulative), action)

    def stop(self, event_id, grant_id):
        def action():
            old = self.receipts.get(grant_id)
            if old is not None:
                return old
            grant = self.grants[grant_id]
            receipt = StopReceipt(grant_id, (grant.executed,), uuid4().hex)
            self.grants[grant_id] = replace(grant, stopped=True)
            self.receipts[grant_id] = receipt
            return receipt

        return self._apply(event_id, ("stop", grant_id), action)

    def refund(self, event_id, receipt: StopReceipt, amount):
        amount = self._pages(amount, "amount")
        if not isinstance(receipt, StopReceipt):
            raise ValueError("stop receipt required")

        def action():
            if self.receipts.get(receipt.grant) != receipt:
                raise ValueError("unknown stop receipt")
            grant = self.grants[receipt.grant]
            available = grant.issued - grant.executed - grant.refunded
            if amount > available:
                raise ValueError("cannot refund executed rights")
            self.grants[grant.identity] = replace(grant, refunded=grant.refunded + amount)
            return amount

        return self._apply(event_id, ("refund", receipt, amount), action)

    def finish(self, event_id, request, *, cancelled=False):
        if type(cancelled) is not bool:
            raise ValueError("cancelled must be Boolean")

        def action():
            outcome = "cancelled" if cancelled else "completed"
            if request in self.terminals:
                if self.terminals[request] != outcome:
                    raise ValueError("conflicting terminal outcome")
                return outcome
            active = [g for g in self.grants.values() if g.request == request]
            if any(g.issued - g.acknowledged - g.refunded for g in active):
                raise ValueError("pending grant must be acknowledged or refunded before finish")
            self.terminals[request] = outcome
            return outcome

        return self._apply(event_id, ("finish", request, cancelled), action)


class GuardedAuditLog(list):
    """Quarantine after an append failure; never roll back a physical transition.

    The optional sink is synchronous. This list remains an in-memory audit
    history; a successful sink write is not a crash-recovery guarantee.
    """
    def __init__(self,on_failure,sink=None):
        super().__init__();self.on_failure=on_failure;self.sink=sink;self.failed=False

    def append(self,record):
        if self.failed:raise RuntimeError('audit history requires reconciliation')
        try:
            if self.sink is not None:self.sink(record)
            super().append(record)
        except BaseException:
            self.failed=True;self.on_failure();raise


class CommitmentLedger:
    def __init__(self, base: State, request_ids, *, audit_history=False):
        if type(audit_history) is not bool:
            raise ValueError('audit_history must be Boolean')
        self.audit_history = audit_history
        self.history_audit_checks = 0
        ids = tuple(request_ids)
        if len(ids) != base.n or len(set(ids)) != len(ids) or any(type(i) is not str or not i for i in ids):
            raise ValueError("unique nonempty request IDs must match state rows")
        order = indexed_closure(base)
        if len(order) != base.n:
            raise Refused("initial cohort is uncertifiable")
        self.base = base
        self.physical = base
        self.ids = ids
        self.order = order
        self.grants = {}
        self._pending_grants = set()
        self._unfenced_grants = set()
        self.receipts = {}
        self.terminal = {}
        self.version = 0
        self.draining = False
        self.drain_order = ()
        self._messages = {}
        self.audit_fault = False
        self.events = GuardedAuditLog(self._audit_failed)
        self._lock = RLock()

    def _audit_failed(self):
        self.audit_fault = True

    def _vector(self, values):
        v = tuple(values)
        if len(v) != self.base.p or any(type(x) is not int or x < 0 for x in v):
            raise ValueError("nonnegative integer pool vector required")
        return v

    def _row(self, request, amount):
        if request not in self.ids:
            raise ValueError("request is not protected")
        return tuple(amount if rid == request else (0,) * self.base.p for rid in self.ids)

    def outstanding(self):
        rows = {rid: i for i, rid in enumerate(self.ids)}
        q = [[0] * self.base.p for _ in self.ids]
        x = [[0] * self.base.p for _ in self.ids]
        for key in self._pending_grants:
            grant = self.grants[key]
            i = rows.get(grant.request)
            if i is None:
                continue
            for p in range(self.base.p):
                q[i][p] += grant.issued[p] - grant.acknowledged[p] - grant.refunded[p]
                x[i][p] += grant.executed[p] - grant.acknowledged[p]
        return tuple(map(tuple, q)), tuple(map(tuple, x))

    def outstanding_full(self):
        """Independent full-history reference for audits and differential tests."""
        q = [[0] * self.base.p for _ in self.ids]
        x = [[0] * self.base.p for _ in self.ids]
        for grant in self.grants.values():
            if grant.request not in self.ids:
                continue
            i = self.ids.index(grant.request)
            for p in range(self.base.p):
                q[i][p] += grant.issued[p] - grant.acknowledged[p] - grant.refunded[p]
                x[i][p] += grant.executed[p] - grant.acknowledged[p]
        return tuple(map(tuple, q)), tuple(map(tuple, x))

    def virtual(self):
        q, _ = self.outstanding()
        return apply_private_growth(self.base, q)

    def audit_pending_index(self):
        """Full independent index audit; mismatch quarantines future grants."""
        with self._lock:
            if self.audit_fault:
                raise RuntimeError('ledger audit failure requires reconciliation')
            try:
                expected = set()
                for key, grant in self.grants.items():
                    q = tuple(a-b-c for a,b,c in zip(grant.issued, grant.acknowledged, grant.refunded))
                    x = tuple(a-b for a,b in zip(grant.executed, grant.acknowledged))
                    if any(q) or any(x):
                        expected.add(key)
                if expected != self._pending_grants or self.outstanding() != self.outstanding_full():
                    raise AssertionError('pending grant index differs from full history')
                if {key for key, grant in self.grants.items() if not grant.stopped} != self._unfenced_grants:
                    raise AssertionError('unfenced grant index differs from full history')
                self.history_audit_checks += 1
            except BaseException:
                self.audit_fault = True
                raise

    def has_unfenced_grants(self):
        """Zero outstanding pages does not establish that execution is stopped."""
        with self._lock:
            if self.audit_fault:
                raise RuntimeError('ledger audit failure requires reconciliation')
            if self.audit_history:
                self.audit_pending_index()
            return bool(self._unfenced_grants)

    def _verify(self):
        if self.audit_history:
            self.audit_pending_index()
        q, x = self.outstanding()
        virtual = apply_private_growth(self.base, q)
        if any(not 0 <= x[i][p] <= q[i][p] for i in range(self.base.n) for p in range(self.base.p)):
            raise AssertionError("x/q accounting violation")
        expected_physical = apply_private_growth(self.base, x)
        if self.physical is not expected_physical and canonical(self.physical) != canonical(expected_physical):
            raise AssertionError("physical pages and confirmed/unconfirmed ledger disagree")
        if self.physical.capacity != self.base.capacity:
            raise AssertionError("capacity changed")
        vm = linear_margins(virtual, self.order)
        if any(value < 0 for row in vm for value in row):
            raise AssertionError("virtual certificate lost")
        if self.audit_history and not valid_witness(virtual, self.order):
            raise AssertionError("reference virtual certificate lost")
        pm = linear_margins(self.physical, self.order)
        suffix = [0] * self.base.p
        for k in range(self.base.n - 1, -1, -1):
            for p in range(self.base.p):
                if pm[k][p] != vm[k][p] + suffix[p]:
                    raise AssertionError("virtual/physical margin identity lost")
                i = self.order[k]
                suffix[p] += q[i][p] - x[i][p]

    def _apply(self, event_id, payload, action, *, writes=()):
        if type(event_id) is not str or not event_id:
            raise ValueError("nonempty event identity required")
        with self._lock:
            if self.audit_fault:raise RuntimeError('ledger audit failure requires reconciliation')
            if self.audit_history:
                self.audit_pending_index()
            if event_id in self._messages:
                old, result = self._messages[event_id]
                if old != payload:
                    raise ValueError("event identity reused with different payload")
                return result
            # Only in-process immutable page transitions occur in action. We can
            # roll back this metadata backend; never use this to undo real DMA.
            names = ("base", "physical", "ids", "order", "version", "draining", "drain_order")
            previous = {name: getattr(self, name) for name in names}
            # Grant/receipt values are immutable; only declared entries change.
            # Keep historical identities for delayed and duplicate messages.
            missing = object()
            entries = [(getattr(self, name), key, getattr(self, name).get(key, missing))
                       for name, key in writes]
            pending_before = {key: key in self._pending_grants
                              for name, key in writes if name == 'grants'}
            unfenced_before = {key: key in self._unfenced_grants for key in pending_before}
            try:
                result = action()
                for key in pending_before:
                    grant = self.grants.get(key)
                    pending = grant is not None and any(
                        issued - ack - refund != 0 or executed - ack != 0
                        for issued, executed, ack, refund in zip(
                            grant.issued, grant.executed, grant.acknowledged, grant.refunded))
                    if pending:
                        self._pending_grants.add(key)
                    else:
                        self._pending_grants.discard(key)
                    if grant is not None and not grant.stopped:
                        self._unfenced_grants.add(key)
                    else:
                        self._unfenced_grants.discard(key)
                self._verify()
                self.version += 1
            except BaseException:
                for key, was_unfenced in unfenced_before.items():
                    if was_unfenced:
                        self._unfenced_grants.add(key)
                    else:
                        self._unfenced_grants.discard(key)
                for key, was_pending in pending_before.items():
                    if was_pending:
                        self._pending_grants.add(key)
                    else:
                        self._pending_grants.discard(key)
                for mapping, key, value in reversed(entries):
                    if value is missing:
                        mapping.pop(key, None)
                    else:
                        mapping[key] = value
                for name, value in previous.items():
                    setattr(self, name, value)
                raise
            try:
                self._messages[event_id] = (payload, result)
                self.events.append(dict(event_id=event_id, kind=payload[0], version=self.version))
            except BaseException:
                # The verified transition already occurred. A storage failure
                # cannot silently permit replay or pretend physical rollback.
                self._audit_failed()
                raise
            return result

    def grant(self, event_id, grant_id, request, amount, *, expected_version=None):
        amount = self._vector(amount)
        if type(grant_id) is not str or not grant_id:
            raise ValueError("nonempty grant identity required")

        def action():
            existing = self.grants.get(grant_id)
            if existing:
                if (existing.request, existing.issued) != (request, amount):
                    raise ValueError("grant identity collision")
                return grant_id
            if expected_version is not None and expected_version != self.version:
                raise StaleState("old ledger version")
            if self.draining and (not self.drain_order or request != self.drain_order[0]):
                raise Refused("drain freezes all non-head grants")
            post = apply_private_growth(self.virtual(), self._row(request, amount))
            # Correctness protocol prioritizes clarity, not low-cost throughput.
            if not valid_witness(post, self.order):
                order = indexed_closure(post)
                if len(order) != post.n:
                    raise Refused("authorization loses completion certificate")
                self.order = order
            zero = (0,) * self.base.p
            self.grants[grant_id] = Grant(grant_id, request, amount, zero, zero, zero)
            return grant_id

        return self._apply(event_id, ("grant", grant_id, request, amount, expected_version), action, writes=(("grants", grant_id),))

    def execute(self, event_id, grant_id, cumulative):
        cumulative = self._vector(cumulative)

        def action():
            g = self.grants[grant_id]
            if any(v > issued - refund for v, issued, refund in zip(cumulative, g.issued, g.refunded)):
                raise ValueError("worker allocation exceeds issued rights")
            target = tuple(max(a, b) for a, b in zip(g.executed, cumulative))
            delta = tuple(a - b for a, b in zip(target, g.executed))
            if any(delta):
                if g.stopped:
                    raise Refused("effective fence prohibits later worker allocation")
                self.physical = apply_private_growth(self.physical, self._row(g.request, delta))
                self.grants[grant_id] = replace(g, executed=target)
            return target

        return self._apply(event_id, ("execute", grant_id, cumulative), action, writes=(("grants", grant_id),))

    def acknowledge(self, event_id, grant_id, cumulative):
        cumulative = self._vector(cumulative)

        def action():
            g = self.grants[grant_id]
            if any(a > b for a, b in zip(cumulative, g.executed)):
                raise ValueError("cannot acknowledge unexecuted pages")
            target = tuple(max(a, b) for a, b in zip(cumulative, g.acknowledged))
            delta = tuple(a - b for a, b in zip(target, g.acknowledged))
            if any(delta):
                self.base = apply_private_growth(self.base, self._row(g.request, delta))
                self.grants[grant_id] = replace(g, acknowledged=target)
            return target

        return self._apply(event_id, ("acknowledge", grant_id, cumulative), action, writes=(("grants", grant_id),))

    def stop_worker(self, event_id, grant_id):
        def action():
            if grant_id in self.receipts:
                return self.receipts[grant_id]
            g = self.grants[grant_id]
            self.grants[grant_id] = replace(g, stopped=True)
            receipt = StopReceipt(grant_id, g.executed, uuid4().hex)
            self.receipts[grant_id] = receipt
            return receipt

        return self._apply(event_id, ("stop_worker", grant_id), action, writes=(("grants", grant_id), ("receipts", grant_id)))

    def refund(self, event_id, receipt: StopReceipt, amount):
        amount = self._vector(amount)
        if not isinstance(receipt, StopReceipt):
            raise ValueError("worker stop receipt required; timeout is not evidence")

        def action():
            if self.receipts.get(receipt.grant) != receipt:
                raise ValueError("unrecognized stop receipt")
            g = self.grants[receipt.grant]
            if not g.stopped or any(a > i - e - r for a, i, e, r in zip(amount, g.issued, g.executed, g.refunded)):
                raise ValueError("only reliably unexecuted rights may be refunded")
            self.grants[g.identity] = replace(g, refunded=tuple(a + b for a, b in zip(g.refunded, amount)))
            # No need mutation on base: reducing q restores virtual future need
            # automatically, avoiding the dangerous "refund U only" shortcut.
            return amount

        return self._apply(event_id, ("refund", receipt, amount), action, writes=(("grants", receipt.grant),))

    def finish(self, event_id, request, *, early=False, cancelled=False):
        if type(early) is not bool or type(cancelled) is not bool:
            raise ValueError("Boolean termination flags required")

        def action():
            if request in self.terminal:
                if self.terminal[request] != ("cancelled" if cancelled else "completed"):
                    raise ValueError("conflicting terminal outcome")
                return self.terminal[request]
            if request not in self.ids:
                raise ValueError("unknown request")
            i = self.ids.index(request)
            q, x = self.outstanding()
            if any(q[i]) or any(x[i]):
                raise Refused("resolve pending rights before completion")
            if not (early or cancelled) and any(self.physical.need[i]):
                raise Refused("hard contract not yet completed")
            self.base, remap = finish_request(self.base, i)
            self.physical, _ = finish_request(self.physical, i)
            self.ids = tuple(r for r in self.ids if r != request)
            self.order = tuple(remap[j] for j in self.order if j != i)
            self.drain_order = tuple(r for r in self.drain_order if r != request)
            self.terminal[request] = "cancelled" if cancelled else "completed"
            return self.terminal[request]

        return self._apply(event_id, ("finish", request, early, cancelled), action, writes=(("terminal", request),))

    def begin_drain(self, event_id):
        def action():
            if self.draining:
                return self.drain_order
            q, x = self.outstanding()
            if any(v for row in q + x for v in row):
                raise Refused("pending must resolve before drain cut")
            self.draining = True
            self.drain_order = tuple(self.ids[i] for i in self.order)
            return self.drain_order

        return self._apply(event_id, ("begin_drain",), action)

    def drain_local(self):
        """Finite T8 execution for this in-process backend; never called by GPU CLI."""
        self.begin_drain("drain:begin")
        while self.drain_order:
            rid = self.drain_order[0]
            amount = self.physical.need[self.ids.index(rid)]
            grant_id = "drain:" + rid
            self.grant(grant_id + ":grant", grant_id, rid, amount)
            self.execute(grant_id + ":execute", grant_id, amount)
            self.acknowledge(grant_id + ":ack", grant_id, amount)
            self.finish(grant_id + ":finish", rid)
        return dict(self.terminal)
