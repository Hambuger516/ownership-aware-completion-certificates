"""Independent materialized CPU pools; no claim of multi-GPU validation."""
import unittest

from prc_nc.model import Group, State
from prc_nc.runtime import AllocationAuthority


class PhysicalPools:
    def __init__(self, pools):
        self.capacity = (4,) * pools
        self.storage = [{} for _ in self.capacity]
        self.contracts = {}
        self.calls = 0

    def admit(self, request, cap):
        self.contracts[request] = cap

    def allocate(self, request, pools):
        self.calls += 1
        result = []
        for pool in pools:
            slots = self.storage[pool]
            slot = next(i for i in range(self.capacity[pool]) if i not in slots)
            data = bytearray(4096)
            data[:] = bytes([pool + 1]) * len(data)
            slots[slot] = (data, {request})
            result.append((pool, slot))
        return result

    def attach(self, request):
        for slots in self.storage:
            for _, owners in slots.values():
                owners.add(request)
        return True

    def release(self, request):
        for slots in self.storage:
            for slot, (_, owners) in list(slots.items()):
                owners.discard(request)
                if not owners:
                    del slots[slot]
        del self.contracts[request]

    def snapshot(self):
        ids = tuple(self.contracts)
        ranks = {request: i for i, request in enumerate(ids)}
        groups = tuple(
            Group(f"{pool}:{slot}", pool, 1, frozenset(ranks[r] for r in owners))
            for pool, slots in enumerate(self.storage)
            for slot, (_, owners) in slots.items()
        )
        need = tuple(tuple(
            self.contracts[r][pool] - sum(r in owners for _, owners in slots.values())
            for pool, slots in enumerate(self.storage)
        ) for r in ids)
        return State(tuple(c - len(s) for c, s in zip(self.capacity, self.storage)), need, groups), ids


class MultiPoolPhysicalTests(unittest.TestCase):
    def test_charged_growth_shared_attachment_and_last_owner_reclamation(self):
        for pools in (2, 3):
            with self.subTest(pools=pools):
                worker = PhysicalPools(pools)
                events = []
                authority = AllocationAuthority(worker.snapshot()[0], on_event=events.append)
                shared = tuple(Group(f"s{p}", p, 1, frozenset({0})) for p in range(pools))
                post = State((3,) * pools, ((1,) * pools,), shared)
                grant = authority.authorize("admit-a", "a", post, ("a",), expected_version=0)
                self.assertEqual(authority.virtual.free, (3,) * pools)
                self.assertEqual(worker.snapshot()[0].free, (4,) * pools)

                def allocate_a():
                    worker.admit("a", (2,) * pools)
                    return worker.allocate("a", range(pools))

                authority.dispatch(grant, allocate_a, worker.snapshot)
                authority.dispatch(grant, allocate_a, worker.snapshot)
                self.assertEqual(worker.calls, 1)
                shared_buffers = [slots[0][0] for slots in worker.storage]
                self.assertEqual(len({id(b) for b in shared_buffers}), pools)
                for p, data in enumerate(shared_buffers):
                    self.assertEqual(data, bytes([p + 1]) * 4096)

                shared = tuple(Group(f"s{p}", p, 1, frozenset({0, 1})) for p in range(pools))
                post = State((3,) * pools, ((1,) * pools,) * 2, shared)
                grant = authority.authorize("attach-b", "b", post, ("a", "b"), expected_version=authority.version)
                self.assertEqual(grant.pages, (0,) * pools)

                def attach_b():
                    worker.admit("b", (2,) * pools)
                    return worker.attach("b")

                authority.dispatch(grant, attach_b, worker.snapshot)
                private = tuple(Group(f"p{p}", p, 1, frozenset({0})) for p in range(pools))
                post = State((2,) * pools, ((0,) * pools, (1,) * pools), shared + private)
                grant = authority.authorize("grow-a", "a", post, ("a", "b"), expected_version=authority.version)
                authority.dispatch(grant, lambda: worker.allocate("a", range(pools)), worker.snapshot)
                authority.release("a", lambda: worker.release("a"), worker.snapshot)
                self.assertEqual(authority.confirmed.free, (3,) * pools)
                for p, slots in enumerate(worker.storage):
                    self.assertIs(slots[0][0], shared_buffers[p])
                    self.assertEqual(slots[0][1], {"b"})
                authority.release("b", lambda: worker.release("b"), worker.snapshot, cancelled=True)
                self.assertEqual(authority.confirmed.free, (4,) * pools)
                self.assertTrue(all(not slots for slots in worker.storage))
                self.assertEqual(authority.drain_cut(), ())
                self.assertEqual(sum(e["kind"] == "acknowledge" for e in events), 3)

    def test_partial_pool_failure_stops_without_refund(self):
        worker = PhysicalPools(2)
        authority = AllocationAuthority(worker.snapshot()[0])
        post = State((3, 3), ((1, 1),), (
            Group("p0", 0, 1, frozenset({0})), Group("p1", 1, 1, frozenset({0})),
        ))
        grant = authority.authorize("partial", "a", post, ("a",), expected_version=0)

        def partial():
            worker.admit("a", (2, 2))
            worker.allocate("a", [0])
            return None

        with self.assertRaisesRegex(RuntimeError, "refund forbidden"):
            authority.dispatch(grant, partial, worker.snapshot)
        self.assertEqual(worker.snapshot()[0].free, (3, 4))
        self.assertEqual(authority.virtual.free, (3, 3))
        self.assertTrue(authority.faulted)
        self.assertEqual(len(worker.storage[0]), 1)
        with self.assertRaisesRegex(RuntimeError, "authority stopped"):
            authority.dispatch(grant, partial, worker.snapshot)
        self.assertEqual(worker.calls, 1)


if __name__ == "__main__":
    unittest.main()
