import itertools
import random
import unittest
from prc_nc.model import State, Group, canonical, margins, fast_guard, apply_private_growth, finish_request
from prc_nc.algorithms import linear_margins, indexed_closure
from prc_nc.ownership_groups import coalesce_ownership


class OwnershipGroupsTests(unittest.TestCase):
    def test_seeded_multipoool_shared_pinned_external_equivalence(self):
        rng = random.Random(20260928)
        for case in range(200):
            n=rng.randint(1,4); p=rng.randint(1,3)
            groups=[]
            for category in range(rng.randint(1,8)):
                pool=rng.randrange(p); pinned=bool(rng.randrange(2))
                owners=frozenset(rng.sample(list(range(n))+[-3,n+4],rng.randint(1,n+2)))
                for repeat in range(rng.randint(2,5)):
                    groups.append(Group(f'{category}:{repeat}',pool,rng.randint(1,3),owners,pinned))
            s=State(tuple(rng.randint(0,12) for _ in range(p)),
                    tuple(tuple(rng.randint(0,12) for _ in range(p)) for _ in range(n)),tuple(groups))
            c=coalesce_ownership(s)
            self.assertEqual(canonical(s),canonical(c));self.assertEqual(s.capacity,c.capacity)
            self.assertLess(len(c.groups),len(s.groups))
            self.assertIs(coalesce_ownership(c),c)
            self.assertEqual(indexed_closure(s),indexed_closure(c))
            z=[[0]*p for _ in range(n)]
            for pool in range(p):
                left=s.free[pool]
                for i in range(n):
                    z[i][pool]=rng.randint(0,min(left,s.need[i][pool]));left-=z[i][pool]
            self.assertEqual(canonical(apply_private_growth(s,z)),canonical(apply_private_growth(c,z)))
            for order in itertools.permutations(range(n)):
                self.assertEqual(margins(s,order),linear_margins(c,order))
                self.assertEqual(fast_guard(s,order,z),fast_guard(c,order,z))
                # Exercise complete removal sequences, including last owners.
                old,new=s,c;alive=list(range(n))
                for owner in order:
                    index=alive.index(owner);alive.pop(index)
                    old,_=finish_request(old,index);new,_=finish_request(new,index)
                    self.assertEqual(old.free,new.free);self.assertEqual(old.capacity,new.capacity)
                    self.assertEqual(old.need,new.need)
                    for suffix in itertools.permutations(range(old.n)):
                        self.assertEqual(margins(old,suffix),linear_margins(new,suffix))

    def test_exact_external_labels_pools_and_pins_remain_distinct(self):
        s=State((0,0),((0,0),),tuple(Group(str(i),pool,1,frozenset(owners),pin)
            for i,(pool,owners,pin) in enumerate([(0,{-1},False),(0,{-1},False),
                (0,{-2},False),(1,{-1},False),(0,{-1},True),(0,{0},False)])))
        c=coalesce_ownership(s)
        self.assertEqual(len(c.groups),5);self.assertEqual(canonical(s),canonical(c))


if __name__=='__main__':unittest.main()
