"""Ordered-index differential checks against the original linear predicates."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from prc_nc.analysis import _EventIndex,summarize
import test_analysis_runner as replay_fixtures


class EventIndexEquivalenceTests(unittest.TestCase):
    def test_randomized_ordered_matches_equal_linear_before_window_and_full_scans(self):
        rng=random.Random(20261102)
        kinds=('causal_authorize','causal_execute','causal_acknowledge','incremental_proposal',
               'allocation','proposal','study_candidate_counterfactual','snapshot')
        for trial in range(100):
            events=[]
            for i in range(rng.randrange(1,100)):
                e=dict(event=rng.choice(kinds),step=rng.randrange(4),request_id='r'+str(rng.randrange(4)),
                       action_id=rng.choice(('a0','a1',None)),grant_id=rng.choice(('g0','g1',None)),marker=i)
                events.append(e)
                if rng.randrange(8)==0:events.append(copy.deepcopy(e))  # retain duplicates, never overwrite
            index=_EventIndex(events);a=rng.randrange(-1,len(events));b=rng.randrange(a+1,len(events)+1)
            for kind in kinds:
                self.assertEqual(index.matches(kind,start=a,stop=b),[(i,e) for i,e in enumerate(events) if a<i<b and e['event']==kind])
                after=next((i for i in range(a+1,len(events)) if events[i]['event']==kind),None)
                self.assertEqual(index.first_after(kind,a),after)
            for kind in ('causal_authorize','study_candidate_counterfactual'):
                for action in ('a0','a1',None):
                    self.assertEqual(index.values(kind,action=action,start=a,stop=b),[e for e in events[a+1:b] if e['event']==kind and e.get('action_id')==action])
            for kind in ('causal_authorize','causal_execute','causal_acknowledge'):
                for grant in ('g0','g1',None):
                    self.assertEqual(index.matches(kind,grant=grant),[(i,e) for i,e in enumerate(events) if e['event']==kind and e.get('grant_id')==grant])
                for step in range(4):
                    self.assertEqual(index.values(kind,step=step),[e for e in events if e['event']==kind and e.get('step')==step])
            for kind in ('incremental_proposal','allocation'):
                key=(rng.randrange(4),'r'+str(rng.randrange(4)))
                self.assertEqual(index.values(kind,request_step=key,stop=b),[e for e in events[:b] if e['event']==kind and (e['step'],e['request_id'])==key])
            expected=next((i for i in range(a+1,len(events)) if events[i]['event'] in ('proposal','incremental_proposal','allocation')),len(events))
            self.assertEqual(index.next_barrier(a),expected)

    def test_duplicates_and_identical_object_first_identity_preserved(self):
        shared=dict(event='causal_execute',step=1,request_id='r',grant_id='g')
        events=[shared,copy.deepcopy(shared),shared]
        index=_EventIndex(events)
        self.assertEqual(index.positions('causal_execute',grant='g'),[0,1,2])
        self.assertEqual(index.first_identity[id(shared)],0)
        self.assertEqual(len(index.matches('causal_execute',grant='g')),3)

    def test_wrong_first_proposal_is_not_skipped_and_missing_before_fields_still_fail(self):
        events=[dict(event='incremental_proposal',step=1,request_id='correct'),
                dict(event='proposal',step=1,request_id='wrong'),
                dict(event='proposal',step=1,request_id='correct'),
                dict(event='incremental_proposal',request_id='malformed')]
        index=_EventIndex(events)
        self.assertEqual(index.first_after('proposal',0),1)
        self.assertEqual(index.next_barrier(0),1)
        self.assertEqual(len(index.values('incremental_proposal',request_step=(1,'correct'),stop=3)),1)
        with self.assertRaises(KeyError):index.values('incremental_proposal',request_step=(1,'correct'))

    def test_summary_builds_one_shared_index_without_replacing_assertions(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);fixture=replay_fixtures.ReplayTests();fixture.write_events(folder,fixture.fixture(folder))
            with patch('prc_nc.analysis._EventIndex',wraps=_EventIndex) as constructor:
                row,_,_=summarize(folder)
            self.assertEqual(row['status'],'PASS');self.assertEqual(constructor.call_count,1)

if __name__=='__main__':unittest.main()
