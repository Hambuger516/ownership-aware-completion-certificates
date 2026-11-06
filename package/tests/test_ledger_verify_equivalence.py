import random
import unittest
from prc_nc.model import State,Group,apply_private_growth,canonical,valid_witness
from prc_nc.algorithms import linear_margins
from prc_nc.ledger import CommitmentLedger


def reference(ledger):
    q,x=ledger.outstanding();base=ledger.base;physical=ledger.physical;order=ledger.order
    virtual=apply_private_growth(base,q)
    if any(not 0<=x[i][p]<=q[i][p] for i in range(base.n) for p in range(base.p)):raise AssertionError('x/q')
    if canonical(physical)!=canonical(apply_private_growth(base,x)):raise AssertionError('physical')
    if physical.capacity!=base.capacity:raise AssertionError('capacity')
    if not valid_witness(virtual,order):raise AssertionError('witness')
    vm=linear_margins(virtual,order);pm=linear_margins(physical,order)
    for k in range(base.n):
        for p in range(base.p):
            suffix=sum(q[i][p]-x[i][p] for i in order[k+1:])
            if pm[k][p]!=vm[k][p]+suffix:raise AssertionError('margin')


class VerifyEquivalenceTests(unittest.TestCase):
    def test_seeded_multipoool_shared_pinned_external_and_corrupt_states(self):
        rng=random.Random(20260928);outcomes={True:0,False:0}
        for case in range(600):
            n=rng.randrange(1,9);p=rng.randrange(1,4)
            groups=tuple(Group(str(k),rng.randrange(p),rng.randrange(1,5),
                frozenset(rng.sample(range(-1,n+1),rng.randrange(1,min(n+2,4)+1))),rng.choice([False,False,True])) for k in range(rng.randrange(20)))
            base=State(tuple(rng.randrange(8,30) for _ in range(p)),tuple(tuple(rng.randrange(12) for _ in range(p)) for _ in range(n)),groups)
            q=tuple(tuple(min(1,v) if case%3 else 0 for v in row) for row in base.need)
            x=tuple(tuple(rng.randrange(v+1) for v in row) for row in q)
            physical=apply_private_growth(base,x)
            if case%5==0:physical=State(tuple(v+1 for v in physical.free),physical.need,physical.groups)
            if case%7==0:x=tuple(tuple(v+2 for v in row) for row in q)
            order=list(range(n));rng.shuffle(order)
            if case%11==0:order=order[:-1]
            ledger=object.__new__(CommitmentLedger);ledger.base=base;ledger.physical=physical;ledger.order=tuple(order);ledger.audit_history=False;ledger.outstanding=lambda:(q,x)
            def accepted(fn):
                try:fn(ledger);return True
                except (AssertionError,ValueError):return False
            expected=accepted(reference);self.assertEqual(accepted(CommitmentLedger._verify),expected,case);outcomes[expected]+=1
        self.assertGreater(outcomes[True],0);self.assertGreater(outcomes[False],0)

    def test_zero_growth_still_validates_shape_types_and_bounds(self):
        state=State((2,),((2,),),(Group('g',0,1,frozenset({0})),))
        self.assertEqual(apply_private_growth(state,((0,),)),state)
        for growth in ((),((False,),),((0,0),),((-1,),),((3,),)):
            with self.assertRaises(ValueError):apply_private_growth(state,growth)


if __name__=='__main__':unittest.main()
