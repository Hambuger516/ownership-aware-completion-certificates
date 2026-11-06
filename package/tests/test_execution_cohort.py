import unittest
from prc_nc.execution_cohort import select_execution_cohort


class CohortTests(unittest.TestCase):
    def test_exact_mapping_excludes_warmup_without_parsing_names(self):
        raw=dict(request_selection='all_internal_including_warmup',pending_batches=0,requests={
            'opaque-1':dict(executed_tokens=20,reexecuted_tokens=3),
            'pilot:0-lookalike':dict(executed_tokens=99,reexecuted_tokens=0)})
        rows={'pilot:0':dict(status='completed',engine_request_id='opaque-1'),
              'pilot:1':dict(status='rejected')}
        r=select_execution_cohort(raw,rows)
        self.assertEqual((r['executed_tokens'],r['reexecuted_tokens']),(20,3))
        self.assertEqual(r['excluded_internal_requests'],['pilot:0-lookalike'])
        rows['pilot:1']=dict(status='completed',engine_request_id='opaque-1')
        with self.assertRaises(ValueError):select_execution_cohort(raw,rows)
        raw['pending_batches']=1
        with self.assertRaises(ValueError):select_execution_cohort(raw,{})
