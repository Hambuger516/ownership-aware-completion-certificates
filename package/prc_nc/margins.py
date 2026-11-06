"""Fixed-order release-frontier margin index (research data structure).

Only maintains certificate arithmetic. Callers MUST validate physical capacity,
hard needs, complete ownership, and transactional runtime semantics separately.
Rebuild after changing the cohort or its order. Not a production vLLM adapter.
"""

from __future__ import annotations
from typing import Sequence
from .model import Group, State, margins, integer


class RangeAddMin:
    """Standard lazy segment tree: range addition and whole-array minimum."""

    def __init__(self, values: Sequence[int]):
        if not values or any(type(v) is not int for v in values):
            raise ValueError("nonempty integer values required")
        self.n = len(values)
        self.mn = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)

        def build(v: int, lo: int, hi: int) -> None:
            if lo == hi:
                self.mn[v] = values[lo]
            else:
                mid = (lo + hi) // 2
                build(2 * v, lo, mid)
                build(2 * v + 1, mid + 1, hi)
                self.mn[v] = min(self.mn[2 * v], self.mn[2 * v + 1])

        build(1, 0, self.n - 1)

    def add(self, left: int, right: int, delta: int) -> None:
        if any(type(v) is not int for v in (left, right, delta)):
            raise ValueError("integer interval and delta required")
        if left > right:
            return
        if not (0 <= left <= right < self.n) or type(delta) is not int:
            raise ValueError("invalid interval addition")

        def visit(v: int, lo: int, hi: int) -> None:
            if left <= lo and hi <= right:
                self.mn[v] += delta
                self.lazy[v] += delta
                return
            mid = (lo + hi) // 2
            if left <= mid:
                visit(2 * v, lo, mid)
            if right > mid:
                visit(2 * v + 1, mid + 1, hi)
            self.mn[v] = self.lazy[v] + min(self.mn[2 * v], self.mn[2 * v + 1])

        visit(1, 0, self.n - 1)

    @property
    def minimum(self) -> int:
        return self.mn[1]

    def values(self) -> tuple[int, ...]:
        out = [0] * self.n

        def visit(v: int, lo: int, hi: int, carry: int) -> None:
            if lo == hi:
                out[lo] = self.mn[v] + carry
            else:
                carry += self.lazy[v]
                mid = (lo + hi) // 2
                visit(2 * v, lo, mid, carry)
                visit(2 * v + 1, mid + 1, hi, carry)

        visit(1, 0, self.n - 1, 0)
        return tuple(out)


class MarginIndex:
    def __init__(self, state: State, order: Sequence[int]):
        if state.n < 1:
            raise ValueError("empty cohorts need no certificate index")
        table = margins(state, order)  # reference build; optimized build is separate
        self.n, self.p = state.n, state.p
        self.order = tuple(order)
        self.rank = {i: k + 1 for k, i in enumerate(order)}  # mathematical ranks 1..n
        self.trees = [RangeAddMin([table[k][p] for k in range(state.n)]) for p in range(state.p)]

    def frontier(self, group: Group) -> int:
        if group.pinned or any(i not in self.rank for i in group.owners):
            return self.n  # unreleased within every relevant prefix k-1
        return max(self.rank[i] for i in group.owners)

    def private_growth(self, request: int, pool: int, pages: int) -> None:
        self._coordinates(request, pool)
        if type(pages) is not int or pages < 0:
            raise ValueError("nonnegative integer growth required")
        r = self.rank[request]
        self.trees[pool].add(0, r - 2, -pages)

    def move_frontier(self, pool: int, pages: int, old: int, new: int) -> None:
        self._coordinates(None, pool)
        for value, name in ((pages, "pages"), (old, "old frontier"), (new, "new frontier")):
            integer(value, name)
        if not (1 <= old <= self.n and 1 <= new <= self.n) or pages < 0:
            raise ValueError("invalid frontier change")
        if old < new:
            self.trees[pool].add(old, new - 1, -pages)
        elif new < old:
            self.trees[pool].add(new, old - 1, pages)

    def release_group(self, pool: int, pages: int, old: int) -> None:
        """Physical release + removal of group from R; only after real release."""
        self._coordinates(None, pool)
        integer(pages, "pages")
        integer(old, "frontier", 1)
        if not 1 <= old <= self.n or pages < 0:
            raise ValueError("invalid release")
        self.trees[pool].add(0, old - 1, pages)

    def delta_need(self, request: int, pool: int, delta: int) -> None:
        self._coordinates(request, pool)
        if type(delta) is not int:
            raise ValueError("integer need delta required")
        r = self.rank[request] - 1
        self.trees[pool].add(r, r, -delta)

    def delta_free(self, pool: int, delta: int) -> None:
        self._coordinates(None, pool)
        self.trees[pool].add(0, self.n - 1, delta)

    def _coordinates(self, request, pool):
        integer(pool, "pool")
        if pool >= self.p:
            raise ValueError("pool outside index")
        if request is not None:
            integer(request, "request")
            if request not in self.rank:
                raise ValueError("request outside index")

    @property
    def certificate_nonnegative(self) -> bool:
        return all(t.minimum >= 0 for t in self.trees)

    def table(self) -> tuple[tuple[int, ...], ...]:
        values = [t.values() for t in self.trees]
        return tuple(tuple(values[p][k] for p in range(self.p)) for k in range(self.n))
