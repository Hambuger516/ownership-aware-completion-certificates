import unittest
from dataclasses import replace
from prc_nc.runtime import AllocationAuthority
from prc_nc.controller import Refused, StaleState
from prc_nc.model import State, Group


class PhysicalPages:
    def __init__(self):
        self.state = State((4,), (), ())
        self.ids = ()
        self.calls = 0

    def snapshot(self):
        return self.state, self.ids

    def allocate(self):
        self.calls += 1
        self.state = State((3,), ((2,),), (Group("real-7", 0, 1, frozenset({0})),))
        self.ids = ("request-1",)
        return (7,)

    def free(self):
        self.state = State((4,), (), ())
        self.ids = ()


def candidate():
    return State((3,), ((2,),), (Group("prospective", 0, 1, frozenset({0})),))


class RuntimeAuthorityTests(unittest.TestCase):
    def test_grant_is_charged_before_physical_dispatch_and_duplicate_exec_is_idempotent(self):
        pages = PhysicalPages()
        events = []
        authority = AllocationAuthority(pages.state, on_event=events.append)
        grant = authority.authorize("a0", "request-1", candidate(), ("request-1",), expected_version=0)
        self.assertEqual(authority.confirmed.free, (4,))
        self.assertEqual(authority.virtual.free, (3,))
        self.assertEqual(pages.calls, 0)
        with self.assertRaises(Refused):
            authority.drain_cut()
        self.assertEqual(authority.dispatch(grant, pages.allocate, pages.snapshot), (7,))
        self.assertEqual(authority.dispatch(grant, pages.allocate, pages.snapshot), (7,))
        self.assertEqual(pages.calls, 1)
        self.assertEqual(authority.virtual, authority.confirmed)
        self.assertEqual(authority.drain_cut(), ("request-1",))
        self.assertEqual([e["kind"] for e in events], ["authorize", "execute", "acknowledge"])
        authority.release("request-1", pages.free, pages.snapshot)
        self.assertEqual(authority.confirmed.free, (4,))
        with self.assertRaises(Refused):
            authority.authorize("late", "request-1", candidate(), ("request-1",), expected_version=authority.version)

    def test_old_version_forged_grant_and_pending_dispatch_are_rejected(self):
        pages = PhysicalPages()
        authority = AllocationAuthority(pages.state)
        with self.assertRaises(StaleState):
            authority.authorize("old", "request-1", candidate(), ("request-1",), expected_version=99)
        grant = authority.authorize("a0", "request-1", candidate(), ("request-1",), expected_version=0)
        with self.assertRaises(ValueError):
            authority.dispatch(replace(grant, pages=(2,)), pages.allocate, pages.snapshot)
        with self.assertRaises(Refused):
            authority.authorize("a1", "request-1", candidate(), ("request-1",), expected_version=authority.version)
        self.assertEqual(pages.calls, 0)

    def test_failure_refund_requires_unchanged_physical_state_and_stops_late_execution(self):
        pages = PhysicalPages()
        authority = AllocationAuthority(pages.state)
        grant = authority.authorize("a0", "request-1", candidate(), ("request-1",), expected_version=0)
        authority.dispatch(grant, lambda: None, pages.snapshot)
        self.assertEqual(authority.virtual.free, (4,))
        with self.assertRaises(Refused):
            authority.dispatch(grant, pages.allocate, pages.snapshot)
        self.assertEqual(pages.calls, 0)

    def test_failed_partial_allocation_faults_instead_of_refunding(self):
        pages = PhysicalPages()
        authority = AllocationAuthority(pages.state)
        grant = authority.authorize("a0", "request-1", candidate(), ("request-1",), expected_version=0)
        def bad_allocate():
            pages.allocate()
            return None
        with self.assertRaisesRegex(RuntimeError, "refund forbidden"):
            authority.dispatch(grant, bad_allocate, pages.snapshot)
        self.assertTrue(authority.faulted)
        self.assertEqual(authority.virtual.free, (3,))

    def test_structure_cannot_be_called_private_growth(self):
        pages = PhysicalPages()
        authority = AllocationAuthority(pages.state)
        with self.assertRaisesRegex(ValueError, "cannot change cohort"):
            authority.authorize("a0", "request-1", candidate(), ("request-1",), expected_version=0, private_growth=())
        self.assertEqual(authority.virtual.free, (4,))

    def test_cancellation_does_not_release_until_observed(self):
        pages = PhysicalPages()
        authority = AllocationAuthority(pages.state)
        grant = authority.authorize("a0", "request-1", candidate(), ("request-1",), expected_version=0)
        authority.dispatch(grant, pages.allocate, pages.snapshot)
        with self.assertRaisesRegex(RuntimeError, "last-owner evidence"):
            authority.release("request-1", lambda: None, pages.snapshot, cancelled=True)
        self.assertTrue(authority.faulted)
        self.assertEqual(authority.confirmed.free, (3,))

    def test_stop_races_through_same_authority_before_dispatch(self):
        pages = PhysicalPages()
        authority = AllocationAuthority(pages.state)
        grant = authority.authorize("a0", "request-1", candidate(), ("request-1",), expected_version=0)
        authority.stop_unexecuted(grant, pages.snapshot)
        with self.assertRaises(Refused):
            authority.dispatch(grant, pages.allocate, pages.snapshot)
        self.assertEqual(pages.calls, 0)
