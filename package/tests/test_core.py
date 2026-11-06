import itertools
import random
import unittest
from unittest.mock import patch
from prc_nc.model import State, Group, finish_request, apply_private_growth, validate_growth, margins, valid_witness
from prc_nc.algorithms import indexed_closure, linear_margins, LinearBuildMarginIndex, interval_trial
from prc_nc.controller import CertificateController, Refused, StaleState, Ticket, event_index


def oracle_release(s, done):
    return tuple(
        sum(g.count for g in s.groups if g.pool == p and not g.pinned and g.owners <= done) for p in range(s.p)
    )


def oracle_closure(s):
    done = set()
    while True:
        released = oracle_release(s, done)
        next_ids = {
            i for i in range(s.n) if i not in done and all(s.need[i][p] <= s.free[p] + released[p] for p in range(s.p))
        }
        if not next_ids:
            return done
        done.update(next_ids)


def oracle_margins(s, order):
    done = set()
    out = []
    for i in order:
        released = oracle_release(s, done)
        out.append(tuple(s.free[p] + released[p] - s.need[i][p] for p in range(s.p)))
        done.add(i)
    return tuple(out)


def state(free=2):
    return State((free,), ((2,), (1,)), (Group("a", 0, 1, frozenset({0})), Group("b", 0, 1, frozenset({1}))))


class ModelTests(unittest.TestCase):
    def test_empty(self):
        s = State((3,), (), ())
        self.assertEqual(indexed_closure(s), ())
        self.assertTrue(valid_witness(s, ()))

    def test_pool_bool_and_pin_int_rejected(self):
        for g in (Group("a", True, 1, frozenset({0})), Group("a", 0, 1, frozenset({0}), 1)):
            with self.assertRaises(ValueError):
                State((1, 1), ((1, 1),), (g,))

    def test_order_strict(self):
        s = state()
        for order in ((), (0, 0), (0,), (1, 0, 2), (False, True), (0.0, 1.0), (0, -1)):
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    linear_margins(s, order)
                with self.assertRaises(ValueError):
                    margins(s, order)

    def test_growth_strict(self):
        for z in (((True,), (0,)), ((3,), (0,)), ((1,), (2,)), ((1, 1), (0, 0)), ((1,),)):
            with self.assertRaises(ValueError):
                validate_growth(state(), z)

    def test_immutable_state(self):
        with self.assertRaises(ValueError):
            State([1], ((1,),), ())
        with self.assertRaises(ValueError):
            State((1,), ((1,),), (Group("a", 0, 1, {0}),))

    def test_duplicate_groups(self):
        g = Group("a", 0, 1, frozenset({0}))
        with self.assertRaises(ValueError):
            State((1,), ((1,),), (g, g))

    def test_external_owner_survives_remap(self):
        s = State((1,), ((0,), (1,)), (Group("a", 0, 2, frozenset({0, 2})), Group("b", 0, 1, frozenset({1, -1}))))
        out, remap = finish_request(s, 0)
        self.assertEqual(out.capacity, s.capacity)
        self.assertEqual(out.free, s.free)
        self.assertTrue(all(i < 0 for i in out.groups[0].owners))
        self.assertEqual(remap, {1: 0})

    def test_last_owner_release_and_pins(self):
        for pinned in (False, True):
            s = State((1,), ((0,), (0,)), (Group("shared", 0, 2, frozenset({0, 1}), pinned),))
            first, _ = finish_request(s, 0)
            self.assertEqual(first.free, (1,))
            last, _ = finish_request(first, 0)
            self.assertEqual(last.free, (1 if pinned else 3,))
            self.assertEqual(last.capacity, (3,))

    def test_tree_coordinates_and_exception_rollback(self):
        index = LinearBuildMarginIndex(state(), (0, 1))
        before = index.table()
        with self.assertRaises(ValueError):
            interval_trial(index, [(1, 0, 1), (0, True, 1)])
        self.assertEqual(index.table(), before)
        for args in ((False, 0, 1), (0, -1, 1), (2, 0, 1), (0, 0, True)):
            with self.assertRaises(ValueError):
                index.private_growth(*args)


class DifferentialTests(unittest.TestCase):
    def test_5000_valid_states_and_private_growth(self):
        rng = random.Random(2026091802)
        for case in range(5000):
            n = rng.randrange(7)
            p = rng.randrange(1, 4)
            groups = []
            for j in range(rng.randrange(9)):
                owners = frozenset(rng.sample(list(range(n)) + [-1, n + 2], rng.randrange(1, min(3, n + 2) + 1)))
                groups.append(Group(str(j), rng.randrange(p), rng.randrange(1, 4), owners, rng.random() < 0.2))
            s = State(
                tuple(rng.randrange(7) for _ in range(p)),
                tuple(tuple(rng.randrange(5) for _ in range(p)) for _ in range(n)),
                tuple(groups),
            )
            order = indexed_closure(s)
            self.assertEqual(set(order), oracle_closure(s), case)
            if n <= 4 and case < 300:
                brute = any(
                    all(v >= 0 for row in oracle_margins(s, o) for v in row) for o in itertools.permutations(range(n))
                )
                self.assertEqual(brute, len(order) == n)
            if len(order) != n or not n:
                continue
            table = oracle_margins(s, order)
            self.assertEqual(linear_margins(s, order), table)
            index = LinearBuildMarginIndex(s, order)
            z = [[0] * p for _ in range(n)]
            for pool in range(p):
                remaining = s.free[pool]
                for i in rng.sample(list(range(n)), n):
                    z[i][pool] = rng.randrange(min(s.need[i][pool], remaining) + 1)
                    remaining -= z[i][pool]
            post = apply_private_growth(s, z)
            sparse = [(i, pool, z[i][pool]) for i in range(n) for pool in range(p) if z[i][pool]]
            fixed = interval_trial(index, sparse)
            self.assertEqual(index.table(), table)
            self.assertEqual(fixed, all(v >= 0 for row in oracle_margins(post, order) for v in row))
            self.assertEqual(fixed or len(indexed_closure(post)) == n, len(oracle_closure(post)) == n)
            for i, pool, pages in sparse:
                index.private_growth(i, pool, pages)
            self.assertEqual(index.table(), oracle_margins(post, order))

    def test_1000_general_event_deltas(self):
        rng = random.Random(901)
        for _ in range(1000):
            n = 4
            p = 2
            before_groups = []
            after_groups = []
            for j in range(5):
                for dest in (before_groups, after_groups):
                    if rng.random() < 0.25:
                        continue
                    owners = frozenset(rng.sample(list(range(n)) + [-1], rng.randrange(1, 4)))
                    dest.append(Group(str(j), rng.randrange(p), rng.randrange(1, 3), owners, rng.random() < 0.2))

            def make(groups):
                return State(
                    tuple(30 - sum(g.count for g in groups if g.pool == q) for q in range(p)),
                    tuple(tuple(rng.randrange(8) for _ in range(p)) for _ in range(n)),
                    tuple(groups),
                )

            before = make(before_groups)
            after = make(after_groups)
            order = tuple(rng.sample(range(n), n))
            original = LinearBuildMarginIndex(before, order)
            original_table = original.table()
            candidate = event_index(original, before, after)
            self.assertEqual(candidate.table(), oracle_margins(after, order))
            self.assertEqual(original.table(), original_table)


class ControllerTests(unittest.TestCase):
    def test_fast_mode_does_not_call_complete_oracle(self):
        c = CertificateController(state(3), audit=False)
        with patch("prc_nc.controller.indexed_closure", side_effect=AssertionError("unexpected oracle")):
            t = c.prepare_growth(((1,), (0,)))
            c.commit(t, c.candidate(t))
        self.assertEqual(c.version, 1)

    def test_audit_mode_checks_oracle(self):
        c = CertificateController(state(3), audit=True)
        with patch("prc_nc.controller.indexed_closure", wraps=indexed_closure) as oracle:
            c.prepare_growth(((1,), (0,)))
            self.assertEqual(oracle.call_count, 1)

    def test_full_fallback_rebuild_and_continue(self):
        # Equal initial needs select 0 first. Giving a page to 1 then requires
        # replacing that stored order; unequal needs would already pick 1 first.
        base = state()
        c = CertificateController(State(base.free, ((2,), (2,)), base.groups))
        t = c.prepare_growth(((0,), (1,)))
        self.assertEqual(t.path, "fallback")
        c.commit(t, c.candidate(t))
        self.assertEqual(c.order, (1, 0))
        t = c.prepare_growth(((0,), (1,)))
        c.commit(t, c.candidate(t))
        self.assertEqual(c.index.table(), margins(c.state, c.order))

    def test_capacity_fit_can_still_be_refused(self):
        s = State((4,), ((3,), (3,)), (Group("a", 0, 3, frozenset({0})), Group("b", 0, 3, frozenset({1}))))
        c = CertificateController(s)
        with self.assertRaises(Refused):
            c.prepare_growth(((2,), (2,)))
        self.assertEqual(c.state, s)
        self.assertEqual(c.version, 0)

    def test_stale_concurrent_tickets(self):
        c = CertificateController(state(3))
        a = c.prepare_growth(((1,), (0,)))
        b = c.prepare_growth(((0,), (1,)))
        c.commit(a, c.candidate(a))
        with self.assertRaises((ValueError, StaleState)):
            c.commit(b, c.state)
        with self.assertRaises(StaleState):
            c.prepare_growth(((0,), (0,)), expected_version=0)

    def test_commit_mismatch_fail_stops(self):
        c = CertificateController(state(3))
        t = c.prepare_growth(((1,), (0,)))
        with self.assertRaises(RuntimeError):
            c.commit(t, state(3))
        self.assertTrue(c.faulted)
        with self.assertRaises(RuntimeError):
            c.prepare_growth(((0,), (0,)))

    def test_abort_and_forged_ticket(self):
        c = CertificateController(state(3))
        t = c.prepare_growth(((1,), (0,)))
        with self.assertRaises(ValueError):
            c.candidate(Ticket(t.identity, 0, "forged"))
        c.abort(t)
        with self.assertRaises(ValueError):
            c.candidate(t)
        self.assertEqual(c.version, 0)

    def test_structural_owner_edit_uses_event_calculus(self):
        s = State((4,), ((1,), (2,)), (Group("a", 0, 1, frozenset({0})),))
        c = CertificateController(s, audit=False)
        post = State((4,), s.need, (Group("a", 0, 1, frozenset({0, 1})),))
        with patch("prc_nc.controller.indexed_closure", side_effect=AssertionError("unexpected full closure")):
            t = c.prepare_state(post)
            c.commit(t, post)
        self.assertEqual(c.index.table(), margins(post, c.order))

    def test_completion_rebuilds_rank(self):
        c = CertificateController(state(3))
        post, _ = finish_request(c.state, 0)
        t = c.prepare_state(post, same_cohort=False)
        c.commit(t, post)
        self.assertEqual(c.order, (0,))
        self.assertEqual(c.index.n, 1)

    def test_capacity_change_rejected(self):
        c = CertificateController(state())
        with self.assertRaises(ValueError):
            c.prepare_state(state(4))

    def test_empty_controller(self):
        c = CertificateController(State((2,), (), ()))
        t = c.prepare_growth(())
        c.commit(t, c.candidate(t))
        self.assertIsNone(c.index)
