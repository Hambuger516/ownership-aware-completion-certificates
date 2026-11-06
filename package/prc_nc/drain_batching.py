"""Fixed-witness batch checks; no future outputs or alternative-order search."""

from .algorithms import linear_margins
from .model import State, Group


def fixed_witness_check(state, ids, frozen_order, *, conservative=False):
    """Check exactly the captured order on the current/projected physical state.

    The C0 view pins currently shared groups only in this temporary certificate
    view. The authority/physical state is never changed. Full O(M+nP) arithmetic
    is intentional in this development implementation and belongs in timing.
    """
    ids, frozen_order = tuple(ids), tuple(frozen_order)
    if len(ids) != state.n or len(set(ids)) != len(ids):
        raise RuntimeError("invalid resident identities for frozen witness")
    if len(frozen_order) != len(ids) or set(frozen_order) != set(ids):
        raise RuntimeError("frozen witness must cover exactly the resident cohort")
    if type(conservative) is not bool:
        raise ValueError("conservative must be bool")
    rank = {rid: i for i, rid in enumerate(ids)}
    order = tuple(rank[rid] for rid in frozen_order)
    view = State(state.free, state.need, tuple(
        Group(g.key, g.pool, g.count, g.owners,
              g.pinned or (conservative and len(g.owners) > 1))
        for g in state.groups))
    margins = linear_margins(view, order)
    minimum = tuple(min((row[p] for row in margins), default=0) for p in range(state.p))
    return dict(accepted=all(v >= 0 for v in minimum),
                view="C0_pinned_shared" if conservative else "C1_physical",
                order=order, margins=margins, min_margin=minimum)


def head_first_running(running, head):
    """Stable native order for followers; fail closed on an unsupported head."""
    matches = [r for r in running if r.request_id == head]
    if len(matches) != 1:
        raise RuntimeError("frozen head must be uniquely present in native running")
    return matches + [r for r in running if r.request_id != head]


def fixed_witness_pair(state, ids, frozen_order, *, conservative=False):
    """Pay C0 then C1 identically in both common-full arms (also secondary)."""
    c0 = fixed_witness_check(state, ids, frozen_order, conservative=True)
    c1 = fixed_witness_check(state, ids, frozen_order, conservative=False)
    if c0["accepted"] and not c1["accepted"]:
        raise AssertionError("conservative frozen witness cannot exceed physical view")
    return dict(c0=c0, c1=c1, active=c0 if conservative else c1)
