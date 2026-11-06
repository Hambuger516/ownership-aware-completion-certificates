import threading
import unittest
from prc_nc.model import State, Group, canonical
from prc_nc.ledger import CommitmentLedger, StopReceipt, CausalCommitBridge
from prc_nc.controller import Refused, StaleState
from prc_nc.drain import DrainPolicy


def ledger(pools=1):
    return CommitmentLedger(
        State((3,) * pools, ((2,) * pools,) * 2, tuple(Group(str(p), p, 2, frozenset({0, 1})) for p in range(pools))),
        ("a", "b"),
    )


class ProtocolTests(unittest.TestCase):
    def test_runtime_bridge_authorize_execute_ack_finish_and_idempotence(self):
        bridge = CausalCommitBridge()
        bridge.authorize("a0", "g0", "r0", 2, 4, state_version=0)
        bridge.authorize("a0", "g0", "r0", 2, 4, state_version=0)
        bridge.execute("e0", "g0", 2)
        bridge.acknowledge("k0", "g0", 2)
        bridge.finish("f0", "r0")
        self.assertEqual(bridge.terminals["r0"], "completed")
        self.assertEqual(len(bridge.events), 4)

    def test_runtime_bridge_rejects_over_envelope_and_timeout_refund(self):
        bridge = CausalCommitBridge()
        bridge.authorize("a0", "g0", "r0", 2, 2)
        with self.assertRaises(ValueError):
            bridge.authorize("a1", "g1", "r0", 1, 2)
        with self.assertRaises(ValueError):
            bridge.refund("rf", object(), 1)
        bridge.execute("e0", "g0", 1)
        receipt = bridge.stop("s0", "g0")
        bridge.refund("rf0", receipt, 1)
        bridge.acknowledge("k0", "g0", 1)
        bridge.finish("f0", "r0", cancelled=True)

    def test_runtime_bridge_stopped_grant_cannot_late_execute(self):
        bridge = CausalCommitBridge()
        bridge.authorize("a0", "g0", "r0", 2, 2)
        receipt = bridge.stop("s0", "g0")
        self.assertEqual(receipt.executed, (0,))
        with self.assertRaises(ValueError):
            bridge.execute("e0", "g0", 1)

    def test_full_grant_execute_ack_complete(self):
        m = ledger()
        m.grant("g", "one", "a", (2,))
        before = canonical(m.virtual())
        m.execute("e", "one", (2,))
        self.assertEqual(canonical(m.virtual()), before)
        physical = canonical(m.physical)
        m.acknowledge("a", "one", (2,))
        self.assertEqual(canonical(m.physical), physical)
        self.assertEqual(canonical(m.virtual()), before)
        m.finish("f", "a")
        m.drain_local()
        self.assertEqual(m.physical.free, (5,))

    def test_pending_early_completion_all_execution_vectors(self):
        import itertools

        for pools in (1, 2, 3):
            for executed in itertools.product((0, 1), repeat=pools):
                m = ledger(pools)
                m.grant("ga", "a1", "a", (2,) * pools)
                m.grant("gb", "b1", "b", (1,) * pools)
                m.execute("ea", "a1", (1,) * pools)
                m.execute("eb", "b1", executed)
                m.acknowledge("aa", "a1", (1,) * pools)
                receipt = m.stop_worker("sa", "a1")
                m.refund("ra", receipt, (1,) * pools)
                m.finish("fa", "a", early=True)
                m.acknowledge("ab", "b1", executed)
                m.execute("eb2", "b1", (1,) * pools)
                m.acknowledge("ab2", "b1", (1,) * pools)
                m.grant("gb2", "b2", "b", (1,) * pools)
                m.execute("eb3", "b2", (1,) * pools)
                m.acknowledge("ab3", "b2", (1,) * pools)
                m.finish("fb", "b")
                self.assertEqual(m.physical.free, (5,) * pools)

    def test_duplicate_event_is_idempotent(self):
        m = ledger()
        m.grant("g", "one", "a", (1,))
        version = m.version
        m.grant("g", "one", "a", (1,))
        self.assertEqual(m.version, version)
        with self.assertRaises(ValueError):
            m.grant("g", "two", "b", (1,))
        self.assertEqual(m.version, version)

    def test_duplicate_grant_identity_and_collision(self):
        m = ledger()
        m.grant("g", "one", "a", (1,))
        before = m.outstanding()
        m.grant("g2", "one", "a", (1,))
        self.assertEqual(m.outstanding(), before)
        with self.assertRaises(ValueError):
            m.grant("g3", "one", "a", (2,))

    def test_ack_cannot_invent_execution(self):
        m = ledger()
        m.grant("g", "one", "a", (1,))
        before = (m.base, m.physical, m.version)
        with self.assertRaises(ValueError):
            m.acknowledge("bad", "one", (1,))
        self.assertEqual((m.base, m.physical, m.version), before)

    def test_old_ack_is_safe_noop(self):
        m = ledger()
        m.grant("g", "one", "a", (2,))
        m.execute("e", "one", (2,))
        m.acknowledge("a", "one", (2,))
        before = m.base
        m.acknowledge("late", "one", (1,))
        self.assertEqual(m.base, before)

    def test_fence_then_refund_and_late_execute_rejected(self):
        m = ledger()
        m.grant("g", "one", "a", (2,))
        m.execute("e", "one", (1,))
        receipt = m.stop_worker("s", "one")
        m.refund("r", receipt, (1,))
        with self.assertRaises((ValueError, Refused)):
            m.execute("late", "one", (2,))
        before = m.physical
        m.execute("old", "one", (1,))
        self.assertEqual(m.physical, before)

    def test_timeout_or_forged_receipt_never_refunds(self):
        m = ledger()
        m.grant("g", "one", "a", (1,))
        for receipt in ("timeout", StopReceipt("one", (0,), "forged")):
            with self.assertRaises(ValueError):
                m.refund("bad", receipt, (1,))
        self.assertEqual(m.outstanding()[0], ((1,), (0,)))

    def test_executed_rights_cannot_be_refunded(self):
        m = ledger()
        m.grant("g", "one", "a", (2,))
        m.execute("e", "one", (1,))
        receipt = m.stop_worker("s", "one")
        with self.assertRaises(ValueError):
            m.refund("bad", receipt, (2,))

    def test_completion_requires_pending_resolution(self):
        m = ledger()
        m.grant("g", "one", "a", (1,))
        with self.assertRaises(Refused):
            m.finish("f", "a", early=True)
        with self.assertRaises(Refused):
            m.begin_drain("d")

    def test_unfinished_hard_contract_not_success(self):
        m = ledger()
        with self.assertRaises(Refused):
            m.finish("bad", "a")
        m.finish("cancel", "a", cancelled=True)
        self.assertEqual(m.terminal["a"], "cancelled")

    def test_failed_grant_rolls_back_everything(self):
        m = ledger()
        before = (m.base, m.physical, m.version, m.order, m.outstanding())
        with self.assertRaises(ValueError):
            m.grant("bad", "one", "a", (4,))
        self.assertEqual((m.base, m.physical, m.version, m.order, m.outstanding()), before)
        self.assertFalse(m.grants)

    def test_concurrent_stale_authorizations(self):
        m = ledger()
        barrier = threading.Barrier(2)
        out = []

        def offer(rid):
            barrier.wait()
            try:
                m.grant(rid, rid, rid, (2,), expected_version=0)
                out.append("pass")
            except StaleState:
                out.append("stale")

        workers = [threading.Thread(target=offer, args=(rid,)) for rid in ("a", "b")]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
        self.assertCountEqual(out, ["pass", "stale"])
        m._verify()

    def test_drain_freezes_nonhead(self):
        m = ledger()
        order = m.begin_drain("d")
        other = next(r for r in m.ids if r != order[0])
        with self.assertRaises(Refused):
            m.grant("bad", "bad", other, (1,))

    def test_local_drain_completes_without_cancellation(self):
        m = ledger(3)
        terminal = m.drain_local()
        self.assertEqual(terminal, {"a": "completed", "b": "completed"})
        self.assertEqual(m.physical.free, (5, 5, 5))
        self.assertFalse(m.ids)

    def test_boolean_vector_rejected(self):
        with self.assertRaises(ValueError):
            ledger().grant("g", "one", "a", (True,))


class DrainTests(unittest.TestCase):
    def test_finite_trigger_and_selected_head(self):
        d = DrainPolicy(2)
        self.assertFalse(d.observe(["a", "b"], (1, 0)))
        self.assertTrue(d.observe(["a", "b"], (1, 0)))
        self.assertTrue(d.allows("b"))
        self.assertFalse(d.allows("a"))
        self.assertFalse(d.allows("waiting"))
        d.finished("b")
        self.assertTrue(d.allows("a"))
        d.finished("a")
        self.assertFalse(d.active)

    def test_cancellation_removes_member(self):
        d = DrainPolicy(1)
        d.observe(["a", "b"], (0, 1))
        d.finished("b")
        self.assertEqual(d.order, ("a",))

    def test_illegal_new_member_during_drain(self):
        d = DrainPolicy(1)
        d.observe(["a"], (0,))
        with self.assertRaises(RuntimeError):
            d.observe(["a", "new"], (0, 1))

    def test_missing_witness_refused(self):
        with self.assertRaises(RuntimeError):
            DrainPolicy(1).observe(["a", "b"], (0,))

    def test_empty_cohort_resets_trigger(self):
        d = DrainPolicy(2)
        d.observe(["a"], (0,))
        d.observe([], ())
        self.assertEqual(d.elapsed, 0)

    def test_bad_trigger(self):
        for trigger in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                DrainPolicy(trigger)
