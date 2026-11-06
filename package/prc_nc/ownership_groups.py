"""Certificate-only grouping; physical page identity evidence stays external."""
from collections import Counter
from .model import Group, State


def coalesce_ownership(state: State) -> State:
    """Merge only identical (pool, exact owner set, pin status) classes.

    This cannot authenticate or compare physical cuts. Callers must finish all
    page-identity checks before retaining this smaller certificate state.
    External owner labels are preserved, never collapsed into one sentinel.
    """
    if not isinstance(state, State):
        raise ValueError('validated State required')
    counts = Counter()
    for group in state.groups:
        counts[group.pool, tuple(sorted(group.owners)), group.pinned] += group.count
    if len(counts) == len(state.groups):
        return state
    groups = tuple(Group('ownership-class:'+str(i), pool, count, frozenset(owners), pinned)
                   for i, ((pool, owners, pinned), count) in enumerate(sorted(counts.items())))
    return State(state.free, state.need, groups)
