"""Event-maintained physical owner index for a single full-attention pool."""
from __future__ import annotations
from .model import Group, State


class PageOwnershipIndex:
    def __init__(self, capacity, *, verify_cache=False):
        if type(capacity) is not int or capacity < 0:
            raise ValueError("nonnegative physical capacity required")
        self.capacity = capacity
        self.requests = {}
        self.owners = {}
        self.events = 0
        self.verify_cache = verify_cache
        self.cache_hits = 0
        self.cache_verifications = 0
        self._revision = 0
        self._cached_key = None
        self._cached_state = None

    def observe_request(self, request_id, blocks):
        """Apply ONE actual allocator result; never consume a forecast page ID."""
        blocks = tuple(blocks)
        if len(set(blocks)) != len(blocks):
            raise ValueError("duplicate physical block")
        previous = self.requests.get(request_id, ())
        old = set(previous)
        new = set(blocks)
        for block in old-new:
            self.owners[block].remove(request_id)
            if not self.owners[block]:
                del self.owners[block]
        for block in new-old:
            self.owners.setdefault(block, set()).add(request_id)
        if blocks:
            self.requests[request_id] = blocks
        else:
            self.requests.pop(request_id, None)
        if len(self.owners) > self.capacity:
            raise RuntimeError("physical owner index exceeds capacity")
        self.events += 1
        if blocks != previous:
            self._revision += 1
            self._cached_key = None
            self._cached_state = None

    def snapshot(self, contracts, free_pages, *, extra=None):
        if type(free_pages) is not int or free_pages < 0:
            raise ValueError("free pages must be a nonnegative integer")
        if free_pages + len(self.owners) != self.capacity:
            raise AssertionError("event owner index differs from physical allocator capacity")
        ids = sorted(set(self.requests) | ({extra} if extra is not None else set()))
        # State and Group are immutable. Physical capacity is checked above on
        # every call, including cache hits. Contracts and provisional admission
        # identity are part of the key; neither may reuse an old certificate.
        key = (self._revision, free_pages, tuple((rid, contracts[rid]) for rid in ids))
        cache_hit = key == self._cached_key
        if cache_hit:
            self.cache_hits += 1
            if not self.verify_cache:
                return self._cached_state, ids
        rank = {rid:i for i,rid in enumerate(ids)}
        needs = []
        for rid in ids:
            left = contracts[rid]-len(self.requests.get(rid, ()))
            if left < 0:
                raise AssertionError("physical pages exceed request contract")
            needs.append((left,))
        groups = tuple(Group(str(block),0,1,frozenset(rank[r] for r in who))
                       for block,who in sorted(self.owners.items()))
        state = State((free_pages,),tuple(needs),groups)
        if cache_hit:
            if state != self._cached_state:
                raise AssertionError("cached ownership snapshot differs from full reconstruction")
            self.cache_verifications += 1
            return self._cached_state, ids
        self._cached_key, self._cached_state = key, state
        return state, ids

    def verify_request_refs(self, blocks):
        for block in blocks:
            if block.is_null or block.ref_cnt != len(self.owners.get(block.block_id, ())):
                raise AssertionError("changed physical block has unmodeled ownership")
