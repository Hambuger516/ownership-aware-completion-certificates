"""CPU model algorithms. No engine hooks, GPU experiments, or timing claims.

The original development package is imported read-only. New code strengthens
the full-closure baseline and builds fixed-order margins in linear incidence time.
"""

from __future__ import annotations

from collections import deque
from .model import Group as Group, State, validate_order
from .margins import MarginIndex, RangeAddMin


def indexed_closure(s: State) -> tuple[int, ...]:
    """Complete shared closure in O(P*n*log(n+1) + M) time and O(nP+M) space.

    Each pool has sorted, fixed need thresholds. Simulated free capacity only
    increases. Each threshold is crossed once, each owner incidence removed once.
    Pinned/external groups never release. This is an implementation improvement,
    not a historical novelty claim or an optimality lower bound.
    """
    sorted_ids = [sorted(range(s.n), key=lambda i: s.need[i][p]) for p in range(s.p)]
    ptr, work = [0] * s.p, list(s.free)
    satisfied = [0] * s.n
    queue: deque[int] = deque()
    adjacency: list[list[int]] = [[] for _ in range(s.n)]
    remaining = [0] * len(s.groups)
    for j, g in enumerate(s.groups):
        if g.pinned or any(i < 0 or i >= s.n for i in g.owners):
            continue
        remaining[j] = len(g.owners)
        for i in g.owners:
            adjacency[i].append(j)

    def advance(p: int) -> None:
        while ptr[p] < s.n:
            i = sorted_ids[p][ptr[p]]
            if s.need[i][p] > work[p]:
                break
            ptr[p] += 1
            satisfied[i] += 1
            if satisfied[i] == s.p:
                queue.append(i)

    for p in range(s.p):
        advance(p)
    order = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for j in adjacency[i]:
            remaining[j] -= 1
            if remaining[j] == 0:
                g = s.groups[j]
                work[g.pool] += g.count
                advance(g.pool)
    return tuple(order)


def linear_margins(s: State, order) -> tuple[tuple[int, ...], ...]:
    """O(M+nP) frontier aggregation, including order validation."""
    order = validate_order(order, s.n)
    rank = {i: k + 1 for k, i in enumerate(order)}
    releases = [[0] * (s.n + 1) for _ in range(s.p)]
    for g in s.groups:
        if g.pinned or any(i not in rank for i in g.owners):
            continue
        frontier = max(rank[i] for i in g.owners)
        releases[g.pool][frontier] += g.count
    work = list(s.free)
    out = []
    for k, i in enumerate(order, 1):
        for p in range(s.p):
            work[p] += releases[p][k - 1]
        out.append(tuple(work[p] - s.need[i][p] for p in range(s.p)))
    return tuple(out)


class LinearBuildMarginIndex(MarginIndex):
    """Original interval operations, with O(M+nP) initialization."""

    def __init__(self, state: State, order):
        if not state.n:
            raise ValueError("empty cohort needs no margin tree")
        table = linear_margins(state, order)
        self.n, self.p = state.n, state.p
        self.order = tuple(order)
        self.rank = {i: k + 1 for k, i in enumerate(order)}
        self.trees = [RangeAddMin([table[k][p] for k in range(self.n)]) for p in range(self.p)]


def cached_suffix_guard(table, order, growth, pools: int) -> bool:
    """Only certificate arithmetic; physical growth validation is separate."""
    suffix = [0] * pools
    for k in range(len(order) - 1, -1, -1):
        if any(suffix[p] > table[k][p] for p in range(pools)):
            return False
        i = order[k]
        for p in range(pools):
            suffix[p] += growth[i][p]
    return True


def interval_trial(index: LinearBuildMarginIndex, sparse_growth) -> bool:
    """Apply, query and roll back certificate deltas, even for rejected proposals.

    Not an atomic transaction. The caller must validate capacity/needs, provide
    exclusive access, and update physical ownership after an accepted action.
    """
    applied = []
    try:
        for i, p, pages in sparse_growth:
            index.private_growth(i, p, pages)
            applied.append((i, p, pages))
        return index.certificate_nonnegative
    finally:
        for i, p, pages in reversed(applied):
            index.trees[p].add(0, index.rank[i] - 2, pages)
