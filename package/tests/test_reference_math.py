"""Small exhaustive comparators for secondary mathematical utilities."""

import itertools
import random
import unittest
from prc_nc.algorithms import indexed_closure
from prc_nc.model import (
    State,
    Group,
    headroom,
    brute_headroom,
    weighted_unit_batch,
    fast_guard,
)


class ReferenceMathTests(unittest.TestCase):
    def test_dedup_does_not_worsen_certificate(self):
        rng = random.Random(414)
        for _ in range(300):
            n, p = rng.randrange(2, 6), rng.randrange(1, 4)
            pool = rng.randrange(p)
            free = tuple(rng.randrange(5) for _ in range(p))
            need = tuple(tuple(rng.randrange(5) for _ in range(p)) for _ in range(n))
            common = tuple(Group(f"x{i}", rng.randrange(p), 1, frozenset({i})) for i in range(n))
            before = State(
                free, need, common + (Group("a", pool, 1, frozenset({0})), Group("b", pool, 1, frozenset({1})))
            )
            after = State(
                tuple(v + (q == pool) for q, v in enumerate(free)),
                need,
                common + (Group("ab", pool, 1, frozenset({0, 1})),),
            )
            self.assertEqual(before.capacity, after.capacity)
            self.assertLessEqual(set(indexed_closure(before)), set(indexed_closure(after)))

    def test_headroom_matches_exhaustive_orders(self):
        rng = random.Random(519)
        for _ in range(100):
            n, p = rng.randrange(0, 6), rng.randrange(1, 4)
            state = State(
                tuple(rng.randrange(3) for _ in range(p)),
                tuple(tuple(rng.randrange(5) for _ in range(p)) for _ in range(n)),
                tuple(Group(str(i), rng.randrange(p), 1, frozenset({i})) for i in range(n)),
            )
            direction = tuple(rng.randrange(1, 4) for _ in range(p))
            self.assertEqual(headroom(state, direction), brute_headroom(state, direction))

    def test_restricted_weighted_batch_matches_all_subsets(self):
        rng = random.Random(615)
        for _ in range(150):
            n = rng.randrange(1, 7)
            state = State(
                (rng.randrange(1, 6),),
                tuple((rng.randrange(1, 5),) for _ in range(n)),
                tuple(Group(str(i), 0, 2, frozenset({i})) for i in range(n)),
            )
            order = indexed_closure(state)
            if len(order) != n:
                continue
            weights = tuple(rng.randrange(-2, 8) for _ in range(n))
            selected = weighted_unit_batch(state, order, weights)
            choices = (
                bits
                for bits in itertools.product((0, 1), repeat=n)
                if fast_guard(state, order, tuple((b,) for b in bits))
            )
            optimum = max(sum(a * b for a, b in zip(bits, weights)) for bits in choices)
            self.assertEqual(sum(weights[i] for i in selected), optimum)

    def test_headroom_and_weight_reject_outside_domain(self):
        state = State((1,), ((1,),), ())
        for direction in ((0,), (True,), (1, 2)):
            with self.assertRaises(ValueError):
                headroom(state, direction)
        with self.assertRaises(ValueError):
            weighted_unit_batch(state, (0,), (float("nan"),))
