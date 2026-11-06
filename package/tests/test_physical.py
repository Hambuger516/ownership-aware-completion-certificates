import unittest
from prc_nc.physical import PageOwnershipIndex
from prc_nc.model import canonical
from prc_nc.algorithms import indexed_closure


class OwnershipIndexTests(unittest.TestCase):
    def test_sharing_growth_release_and_reuse_track_distinct_physical_pages(self):
        tracker=PageOwnershipIndex(4)
        tracker.observe_request("a",(1,2))
        tracker.observe_request("b",(1,3))
        state,ids=tracker.snapshot({"a":3,"b":3},1)
        self.assertEqual(state.need,((1,),(1,)))
        self.assertEqual(len(indexed_closure(state)),2)
        tracker.observe_request("a",())
        state,ids=tracker.snapshot({"b":3},2)
        self.assertEqual(ids,["b"])
        self.assertEqual(len(tracker.owners),2)
        tracker.observe_request("b",(1,3,4))
        state,ids=tracker.snapshot({"b":3},1)
        self.assertEqual(state.need,((0,),))
        tracker.observe_request("b",())
        state,ids=tracker.snapshot({},4)
        self.assertEqual(canonical(state),((4,),(),()))
        tracker.observe_request("c",(1,2))
        state,ids=tracker.snapshot({"c":3},2)
        self.assertEqual(ids,["c"])

    def test_capacity_discrepancy_fails_instead_of_guessing_release(self):
        tracker=PageOwnershipIndex(4)
        tracker.observe_request("a",(1,2))
        with self.assertRaises(AssertionError):
            tracker.snapshot({"a":3},3)

    def test_snapshot_reuse_invalidates_on_contract_and_ownership_changes(self):
        tracker = PageOwnershipIndex(4)
        tracker.observe_request("a", (1, 2))
        first, ids = tracker.snapshot({"a": 3}, 2)
        ids.append("caller-mutation")
        tracker.observe_request("a", (1, 2))
        reused, ids = tracker.snapshot({"a": 3}, 2)
        self.assertIs(first, reused)
        self.assertEqual(ids, ["a"])
        changed, _ = tracker.snapshot({"a": 4}, 2)
        self.assertEqual(changed.need, ((2,),))
        extra, ids = tracker.snapshot({"a": 4, "b": 2}, 2, extra="b")
        self.assertEqual(ids, ["a", "b"])
        self.assertEqual(extra.need, ((2,), (2,)))
        tracker.observe_request("b", (1,))
        shared, _ = tracker.snapshot({"a": 4, "b": 2}, 2)
        self.assertEqual(shared.groups[0].owners, frozenset({0, 1}))
        tracker.observe_request("a", ())
        released, ids = tracker.snapshot({"b": 2}, 3)
        self.assertEqual(ids, ["b"])
        self.assertEqual(released.need, ((1,),))
        self.assertEqual(released.groups[0].owners, frozenset({0}))

    def test_cached_snapshot_still_checks_physical_capacity(self):
        tracker = PageOwnershipIndex(4)
        tracker.observe_request("a", (1, 2))
        tracker.snapshot({"a": 3}, 2)
        with self.assertRaises(AssertionError):
            tracker.snapshot({"a": 3}, 3)
        with self.assertRaises(ValueError):
            tracker.snapshot({"a": 3}, 2.0)

    def test_diagnostic_cache_recomputation_rejects_corrupted_cached_state(self):
        from prc_nc.model import State
        tracker = PageOwnershipIndex(4, verify_cache=True)
        tracker.observe_request("a", (1, 2))
        state, _ = tracker.snapshot({"a": 3}, 2)
        again, _ = tracker.snapshot({"a": 3}, 2)
        self.assertIs(state, again)
        self.assertEqual(tracker.cache_hits, 1)
        self.assertEqual(tracker.cache_verifications, 1)
        tracker._cached_state = State(state.free, ((0,),), state.groups)
        with self.assertRaisesRegex(AssertionError, "full reconstruction"):
            tracker.snapshot({"a": 3}, 2)
        self.assertEqual(tracker.cache_verifications, 1)
