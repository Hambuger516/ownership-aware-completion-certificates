"""Ownership-aware completion certificates: small, auditable reference model.

Validated specification shared by controllers and the vLLM adapter. Fixed typed
page pools; a quiescent cut; immutable shared resident pages; append-only
PRIVATE new pages; finite hard remaining-growth bounds. See docs/METHOD_MAP_CN.md.
"""

from __future__ import annotations
import os
from collections import OrderedDict
from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Sequence


def integer(value, name, minimum=0):
    """bool is an int subclass in Python, but never a valid page count/rank."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_order(order, n):
    order = tuple(order)
    if len(order) != n or any(type(i) is not int for i in order) or set(order) != set(range(n)):
        raise ValueError("order must be a complete integer permutation")
    return order


_FAST_STATE = os.environ.get("PRC_FAST_STATE", "0") == "1"


@dataclass(frozen=True)
class Group:
    key: str
    pool: int
    count: int
    owners: frozenset[int]
    pinned: bool = False


@dataclass(frozen=True)
class State:
    free: tuple[int, ...]
    need: tuple[tuple[int, ...], ...]
    groups: tuple[Group, ...]

    def __post_init__(self) -> None:
        # Performance mode (PRC_FAST_STATE=1, set by gpu_run when audit is off):
        # skip the deep structural validation of internally constructed states.
        # Every state built by the adapter passes through the same constructors;
        # the audited mode re-enables the full checks.
        if _FAST_STATE:
            if type(self.free) is not tuple or type(self.need) is not tuple or type(self.groups) is not tuple:
                raise ValueError("state requires immutable tuples of rows and Groups")
            return
        # Immutable tuples/frozensets prevent mutation behind a versioned guard.
        if (
            type(self.free) is not tuple
            or type(self.need) is not tuple
            or type(self.groups) is not tuple
            or any(type(row) is not tuple for row in self.need)
            or any(not isinstance(g, Group) for g in self.groups)
        ):
            raise ValueError("state requires immutable tuples of rows and Groups")
        p = len(self.free)
        if p == 0 or any(type(v) is not int or v < 0 for v in self.free):
            raise ValueError("free must be a nonempty nonnegative integer vector")
        if any(len(row) != p or any(type(v) is not int or v < 0 for v in row) for row in self.need):
            raise ValueError("need must be an n-by-p nonnegative integer matrix")
        if any(type(g.key) is not str or not g.key for g in self.groups):
            raise ValueError("nonempty string group keys required")
        if len({g.key for g in self.groups}) != len(self.groups):
            raise ValueError("group keys must represent disjoint unique page groups")
        for g in self.groups:
            if (
                type(g.pool) is not int
                or not 0 <= g.pool < p
                or type(g.count) is not int
                or g.count <= 0
                or type(g.pinned) is not bool
                or type(g.owners) is not frozenset
                or not g.owners
                or any(type(i) is not int for i in g.owners)
            ):
                raise ValueError("invalid group")
            # Owners outside 0..n-1 are external and never finish in this cohort.

    @property
    def n(self) -> int:
        return len(self.need)

    @property
    def p(self) -> int:
        return len(self.free)

    @property
    def capacity(self) -> tuple[int, ...]:
        return tuple(self.free[p] + sum(g.count for g in self.groups if g.pool == p) for p in range(self.p))


def release(s: State, completed: Iterable[int]) -> tuple[int, ...]:
    done = frozenset(completed)
    if any(type(i) is not int or not 0 <= i < s.n for i in done):
        raise ValueError("completed contains a non-cohort request")
    return tuple(
        sum(g.count for g in s.groups if g.pool == p and not g.pinned and g.owners <= done) for p in range(s.p)
    )


def margins(s: State, order: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    order = validate_order(order, s.n)
    done: set[int] = set()
    out = []
    for i in order:
        rel = release(s, done)
        out.append(tuple(s.free[p] + rel[p] - s.need[i][p] for p in range(s.p)))
        done.add(i)
    return tuple(out)


def valid_witness(s: State, order: Sequence[int]) -> bool:
    return all(v >= 0 for row in margins(s, order) for v in row)


def closure(s: State) -> tuple[int, ...]:
    """Reference O(n^2 P + n M) search; deliberately simple for auditing.

    An incremental owner-incidence implementation can avoid recomputing release
    and achieve O(n^2 P + M), M = page-group/owner incidence count.
    """
    done: set[int] = set()
    order: list[int] = []
    while len(done) < s.n:
        rel = release(s, done)
        eligible = [
            i for i in range(s.n) if i not in done and all(s.need[i][p] <= s.free[p] + rel[p] for p in range(s.p))
        ]
        if not eligible:
            break
        i = eligible[0]
        done.add(i)
        order.append(i)
    return tuple(order)


def feasible(s: State) -> bool:
    return len(closure(s)) == s.n


def brute_feasible(s: State) -> bool:
    if s.n > 8:
        raise ValueError("brute-force comparator limited to n <= 8")
    return any(valid_witness(s, order) for order in permutations(range(s.n)))


def validate_growth(s: State, growth: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    z = tuple(tuple(row) for row in growth)
    if len(z) != s.n or any(len(row) != s.p for row in z):
        raise ValueError("growth shape mismatch")
    for i in range(s.n):
        for p in range(s.p):
            if type(z[i][p]) is not int or not 0 <= z[i][p] <= s.need[i][p]:
                raise ValueError("growth must lie within remaining hard bound")
    if any(sum(z[i][p] for i in range(s.n)) > s.free[p] for p in range(s.p)):
        raise ValueError("batch exceeds immediate free capacity")
    return z


def apply_private_growth(s: State, growth: Sequence[Sequence[int]]) -> State:
    z = validate_growth(s, growth)
    # State and its groups are immutable. A validated zero transition needs no
    # new group list, key set, or repeated validation of every resident group.
    if not any(value for row in z for value in row):
        return s
    groups = list(s.groups)
    keys = {g.key for g in groups}
    for i in range(s.n):
        for p in range(s.p):
            if z[i][p]:
                key = f"new-{len(groups)}-{i}-{p}"
                while key in keys:
                    key += "x"
                keys.add(key)
                groups.append(Group(key, p, z[i][p], frozenset({i})))
    return State(
        tuple(s.free[p] - sum(z[i][p] for i in range(s.n)) for p in range(s.p)),
        tuple(tuple(s.need[i][p] - z[i][p] for p in range(s.p)) for i in range(s.n)),
        tuple(groups),
    )


def fast_guard(s: State, order: Sequence[int], growth: Sequence[Sequence[int]]) -> bool:
    try:
        z = validate_growth(s, growth)
    except ValueError:
        return False
    m = margins(s, order)
    suffix = [0] * s.p
    for k in range(s.n - 1, -1, -1):
        if any(suffix[p] > m[k][p] for p in range(s.p)):
            return False
        i = order[k]
        for p in range(s.p):
            suffix[p] += z[i][p]
    return True


def predicted_updated_margins(
    s: State, order: Sequence[int], growth: Sequence[Sequence[int]]
) -> tuple[tuple[int, ...], ...]:
    z = validate_growth(s, growth)
    m = margins(s, order)
    return tuple(
        tuple(m[k][p] - sum(z[order[r]][p] for r in range(k + 1, s.n)) for p in range(s.p)) for k in range(s.n)
    )


def add_headroom(s: State, b: int, direction: Sequence[int] | None = None) -> State:
    w = tuple(direction) if direction is not None else (1,) * s.p
    if type(b) is not int or b < 0 or len(w) != s.p or any(type(v) is not int or v <= 0 for v in w):
        raise ValueError("nonnegative integer b and positive integer direction required")
    return State(tuple(s.free[p] + b * w[p] for p in range(s.p)), s.need, s.groups)


def headroom(s: State, direction: Sequence[int] | None = None) -> int:
    w = tuple(direction) if direction is not None else (1,) * s.p
    add_headroom(s, 0, w)  # validate
    if feasible(s):
        return 0
    lo, hi = 0, 1
    while not feasible(add_headroom(s, hi, w)):
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if feasible(add_headroom(s, mid, w)):
            hi = mid
        else:
            lo = mid
    return hi


def brute_headroom(s: State, direction: Sequence[int] | None = None) -> int:
    w = tuple(direction) if direction is not None else (1,) * s.p
    add_headroom(s, 0, w)
    best = None
    for order in permutations(range(s.n)):
        m = margins(s, order)
        need = max((max(0, (-m[k][p] + w[p] - 1) // w[p]) for k in range(s.n) for p in range(s.p)), default=0)
        best = need if best is None else min(best, need)
    return int(best or 0)


def weighted_unit_batch(s: State, order: Sequence[int], utilities: Sequence[float]) -> tuple[int, ...]:
    """Exact weighted greedy in the restricted single-pool, one-page-per-action
    laminar-matroid model. Not a general GPU batching optimizer.
    """
    if s.p != 1 or len(utilities) != s.n or not valid_witness(s, order):
        raise ValueError("single pool, correct utility length, valid witness required")
    import math

    if any(not math.isfinite(float(u)) for u in utilities):
        raise ValueError("finite utilities required")
    z = [[0] for _ in range(s.n)]
    for i in sorted(range(s.n), key=lambda i: (-utilities[i], i)):
        if utilities[i] < 0 or s.need[i][0] < 1:
            continue
        z[i][0] = 1
        if not fast_guard(s, order, z):
            z[i][0] = 0
    return tuple(i for i in range(s.n) if z[i][0])


def finish_request(s: State, request: int):
    """T6 after a trusted stop/completion; not a cancellation-intent handler.

    External positive owner labels must not alias a new internal rank after
    removing a request. Convert those labels to distinct negative labels.
    Pinned ownerless groups remain held by a synthetic external owner (L).
    """
    integer(request, "request")
    if request >= s.n:
        raise ValueError("request is outside cohort")
    remaining = [i for i in range(s.n) if i != request]
    remap = {i: j for j, i in enumerate(remaining)}
    external = {i for g in s.groups for i in g.owners if i < 0 or i >= s.n}
    used = {i for i in external if i < 0}
    next_external = min(used | {0}) - 1
    external_map = {}
    for owner in sorted(external):
        if owner < 0:
            external_map[owner] = owner
        else:
            external_map[owner] = next_external
            next_external -= 1
    free = list(s.free)
    groups = []
    for group in s.groups:
        owners = group.owners - {request}
        if not owners and not group.pinned:
            free[group.pool] += group.count
            continue
        mapped = frozenset(remap[i] if i in remap else external_map[i] for i in owners)
        if not mapped:  # unreclaimable ownerless storage cannot disappear
            mapped = frozenset({next_external})
            next_external -= 1
        groups.append(Group(group.key, group.pool, group.count, mapped, group.pinned))
    result = State(tuple(free), tuple(s.need[i] for i in remaining), tuple(groups))
    if result.capacity != s.capacity:
        raise AssertionError("completion changed capacity")
    return result, remap


_CANONICAL_MEMO: "OrderedDict[int, tuple]" = OrderedDict()
_CANONICAL_MEMO_SIZE = 256


def canonical(s: State):
    """Group labels and splitting do not change the represented physical state.

    States are immutable, so the canonical form is memoized per object identity
    (bounded LRU; the memo also keeps the state alive so an id cannot be reused
    while its entry exists)."""
    from collections import Counter

    hit = _CANONICAL_MEMO.get(id(s))
    if hit is not None and hit[0] is s:
        _CANONICAL_MEMO.move_to_end(id(s))
        return hit[1]
    counts = Counter()
    for g in s.groups:
        counts[(g.pool, tuple(sorted(g.owners)), g.pinned)] += g.count
    result = (s.free, s.need, tuple(sorted(counts.items())))
    _CANONICAL_MEMO[id(s)] = (s, result)
    if len(_CANONICAL_MEMO) > _CANONICAL_MEMO_SIZE:
        _CANONICAL_MEMO.popitem(last=False)
    return result
