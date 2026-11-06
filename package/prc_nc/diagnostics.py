"""T2/T3 diagnostics: non-bridge is NOT evidence of useful ownership information."""

from dataclasses import replace
from .model import State, release
from .algorithms import indexed_closure


def boundary_report(state: State):
    # C0 retains guaranteed exclusive release, pins all shared groups forever.
    # Existing external owners and pins remain non-releasing in both models.
    conservative = State(
        state.free, state.need, tuple(replace(g, pinned=g.pinned or len(g.owners) > 1) for g in state.groups)
    )
    c0 = frozenset(indexed_closure(conservative))
    c1 = frozenset(indexed_closure(state))
    private = tuple(release(conservative, (i,)) for i in range(state.n))
    lower = tuple(max((row[p] for row in state.need), default=0) for p in range(state.p))
    upper = tuple(min((state.need[i][p] + private[i][p] for i in range(state.n)), default=0) for p in range(state.p))
    bridge = all(a <= b for a, b in zip(lower, upper)) if state.n else None
    r0, r1 = release(conservative, c0), release(state, c0)
    crossing = tuple(
        i
        for i in range(state.n)
        if i not in c0 and all(state.need[i][p] <= state.free[p] + r1[p] for p in range(state.p))
    )
    if not c0 <= c1 or bool(crossing) != (c0 != c1):
        raise AssertionError("T2 closure/crossing identity violated")
    if bridge and (c0 != c1 or len(c0) not in (0, state.n)):
        raise AssertionError("T3 bridge identity violated")
    return dict(
        cohort_size=state.n,
        conservative_closure=sorted(c0),
        exact_closure=sorted(c1),
        closure_gain=c0 != c1,
        full_feasibility_gain=len(c1) == state.n and len(c0) != state.n,
        bridge_holds=bridge,
        bridge_lower=lower,
        bridge_upper=upper,
        exclusive_release=private,
        threshold_crossing_requests=crossing,
        conservative_release_at_c0=r0,
        exact_release_at_c0=r1,
    )
