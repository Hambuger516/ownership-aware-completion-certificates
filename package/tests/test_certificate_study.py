"""Isolated study policy and shared-authority invariants; no engine/GPU."""
import itertools
import unittest
from unittest.mock import patch
from prc_nc.configuration import validate_config
from prc_nc.model import State, Group, brute_feasible, valid_witness, apply_private_growth, canonical, finish_request
from prc_nc.algorithms import indexed_closure
from prc_nc.controller import CertificateController, Refused, StaleState
from prc_nc.runtime import AllocationAuthority

VARIANTS = ('shared_incremental', 'force_full', 'exclusive_only')

def exclusive_view(s):
    return State(s.free, s.need, tuple(Group(g.key, g.pool, g.count, g.owners,
        g.pinned or len(g.owners)>1) for g in s.groups))

def configured(**changes):
    return dict(method='shared_incremental', enable_prc_pipeline=False,
        rank_order=False, zero_growth_fast=False, deny_cache=False,
        drain_policy='timer', drain_after_steps=256, **changes)

class StudyVariantTests(unittest.TestCase):
    def test_exact_variants_and_shared_policies_validate(self):
        for v in VARIANTS:
            self.assertEqual(validate_config(configured(certificate_variant=v))['certificate_variant'],v)
        with self.assertRaises(ValueError):validate_config(configured(certificate_variant='pretend_full'))
        for k,value in [('method','exclusive_closure'),('enable_prc_pipeline',True),
                        ('rank_order',True),('zero_growth_fast',True),('deny_cache',True)]:
            for v in ('force_full','exclusive_only'):
                c=configured(certificate_variant=v);c[k]=value
                with self.subTest(v=v,k=k),self.assertRaises(ValueError):validate_config(c)

    def test_unknown_controller_variant_fails_closed(self):
        with self.assertRaises(ValueError):CertificateController(State((1,),(),()),certificate_variant='typo')

    def test_private_only_refuses_shared_successor_without_changing_physical_state(self):
        post=State((0,),((0,),(0,),(1,)),(Group('shared',0,1,frozenset({0,1})),))
        for v in ('shared_incremental','force_full'):
            c=CertificateController(post,audit=False,certificate_variant=v)
            self.assertEqual(len(c.order),3);self.assertEqual(canonical(c.state),canonical(post))
        with self.assertRaises(Refused):CertificateController(post,audit=False,certificate_variant='exclusive_only')
        self.assertFalse(post.groups[0].pinned)

    def test_full_checks_do_not_use_interval_or_event_acceptance_shortcuts(self):
        start=State((3,),((2,),),(Group('private',0,1,frozenset({0})),))
        for v in ('exclusive_only',):
            c=CertificateController(start,audit=False,certificate_variant=v)
            with patch('prc_nc.controller.interval_trial',side_effect=AssertionError('interval shortcut')), \
                 patch('prc_nc.controller.event_index',side_effect=AssertionError('event shortcut')), \
                 patch('prc_nc.controller.indexed_closure',wraps=indexed_closure) as full:
                t=c.prepare_growth(((1,),));self.assertEqual(full.call_count,1);c.commit(t,c.candidate(t))
                t=c.prepare_state(c.state,same_cohort=True);self.assertEqual(full.call_count,2);c.commit(t,c.candidate(t))
            self.assertEqual(c.stats['forced_full'],2);self.assertEqual(c.stats['oracle'],2)
            self.assertEqual(c.stats['fast'],0);self.assertEqual(c.stats['commits'],2)

    def test_force_full_always_calls_oracle_but_preserves_incremental_witness_policy(self):
        # A fresh oracle changes its tie order after growth; both study arms
        # must retain the same earlier valid order, hence the same drain head.
        start=State((2,),((2,),(1,)),(Group('a',0,1,frozenset({0})),Group('b',0,1,frozenset({1}))))
        growth=((1,),(0,));post=apply_private_growth(start,growth)
        self.assertNotEqual(indexed_closure(start),indexed_closure(post))
        for structural in (True,False):
            inc=CertificateController(start,audit=False,certificate_variant='shared_incremental')
            full=CertificateController(start,audit=False,certificate_variant='force_full')
            old_order=inc.order
            with patch('prc_nc.controller.indexed_closure',wraps=indexed_closure) as oracle:
                tf=full.prepare_state(post) if structural else full.prepare_growth(growth)
                self.assertEqual(oracle.call_count,1)
            ti=inc.prepare_state(post) if structural else inc.prepare_growth(growth)
            inc.commit(ti,inc.candidate(ti));full.commit(tf,full.candidate(tf))
            self.assertEqual(inc.order,old_order);self.assertEqual(full.order,old_order)
            self.assertEqual(inc.index.table(),full.index.table())
            self.assertEqual(inc.state,full.state);self.assertEqual(full.stats['forced_full'],1)

    def test_incremental_keeps_fixed_witness_fast_path(self):
        s=State((3,),((2,),),(Group('private',0,1,frozenset({0})),))
        c=CertificateController(s,audit=False,certificate_variant='shared_incremental')
        with patch('prc_nc.controller.indexed_closure',side_effect=AssertionError('unnecessary oracle')):
            t=c.prepare_growth(((1,),));c.commit(t,c.candidate(t))
        self.assertEqual(c.stats['fast'],1);self.assertEqual(c.stats['oracle'],0)

    def test_exhaustive_views_agree_with_independent_permutation_search(self):
        owners=[frozenset(x) for n in (1,2,3) for x in itertools.combinations(range(3),n)]
        checked=0
        for who,pinned,free,needs in itertools.product(owners,(False,True),range(3),itertools.product(range(4),repeat=3)):
            s=State((free,),tuple((x,) for x in needs),(Group('g',0,1,who,pinned),))
            for v,view in [('force_full',s),('exclusive_only',exclusive_view(s))]:
                expected=brute_feasible(view)
                try:c=CertificateController(s,audit=False,certificate_variant=v)
                except Refused:accepted=False
                else:
                    accepted=True;self.assertEqual(canonical(c.state),canonical(s))
                    self.assertTrue(valid_witness(s,c.order))
                self.assertEqual(accepted,expected,(s,v));checked+=1
        self.assertEqual(checked,5376)

    def test_growth_decisions_agree_with_each_complete_information_view(self):
        checked=0
        for who,free,needs in itertools.product((frozenset({0}),frozenset({0,1})),range(1,4),itertools.product(range(3),repeat=2)):
            s=State((free,),tuple((x,) for x in needs),(Group('g',0,1,who),))
            if not brute_feasible(s):continue
            for target in range(2):
                if s.need[target][0]<1:continue
                z=tuple((int(i==target),) for i in range(2));post=apply_private_growth(s,z)
                for v in VARIANTS:
                    if not brute_feasible(exclusive_view(s) if v=='exclusive_only' else s):continue
                    c=CertificateController(s,audit=False,certificate_variant=v)
                    expected=brute_feasible(exclusive_view(post) if v=='exclusive_only' else post)
                    try:t=c.prepare_growth(z)
                    except Refused:accepted=False
                    else:accepted=True;c.commit(t,c.candidate(t));self.assertEqual(canonical(c.state),canonical(post))
                    self.assertEqual(accepted,expected,(s,z,v));checked+=1
        self.assertGreater(checked,100)

    def test_shared_authority_lifecycle_for_every_variant(self):
        before=State((4,),(),());post=State((3,),((2,),),(Group('prospective',0,1,frozenset({0})),))
        for v in VARIANTS:
            live=[before,()];calls=[];events=[]
            a=AllocationAuthority(before,audit=False,certificate_variant=v,on_event=events.append)
            g=a.authorize('a','r',post,('r',),expected_version=0)
            self.assertEqual(a.confirmed,before);self.assertEqual(a.virtual,post)
            with self.assertRaises(Refused):a.drain_cut()
            def allocate():live[:]=[post,('r',)];calls.append(1);return (7,)
            a.dispatch(g,allocate,lambda:tuple(live));a.dispatch(g,allocate,lambda:tuple(live))
            self.assertEqual(len(calls),1);self.assertEqual(a.drain_cut(),('r',))
            def release():live[:]=[before,()]
            a.release('r',release,lambda:tuple(live));self.assertEqual(a.confirmed,before)
            self.assertEqual([e['kind'] for e in events],['authorize','execute','acknowledge','finish'])

    def test_stale_and_bad_physical_execution_fail_closed(self):
        before=State((4,),(),());post=State((3,),((2,),),(Group('prospective',0,1,frozenset({0})),))
        for v in VARIANTS:
            a=AllocationAuthority(before,audit=False,certificate_variant=v)
            with self.assertRaises(StaleState):a.authorize('old','r',post,('r',),expected_version=9)
            g=a.authorize('a','r',post,('r',),expected_version=0);live=[before,()]
            def bad():live[:]=[State((2,),((1,),),(Group('bad',0,2,frozenset({0})),)),('r',)];return (1,2)
            with self.assertRaises(RuntimeError):a.dispatch(g,bad,lambda:tuple(live))
            self.assertTrue(a.faulted)

    def test_released_shared_group_can_become_private_without_early_free(self):
        s=State((1,),((0,),(1,),(1,)),(Group('shared',0,1,frozenset({0,1})),))
        c=CertificateController(s,audit=False,certificate_variant='exclusive_only')
        after,_=finish_request(s,0)
        self.assertEqual(after.free,s.free);self.assertEqual(after.groups[0].owners,frozenset({0}))
        t=c.prepare_state(after,same_cohort=False);c.commit(t,after);self.assertEqual(len(c.order),2)

if __name__=='__main__':unittest.main()
