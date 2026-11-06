import unittest
from prc_nc.execution_trace import replay_execution


class ExecutionTraceTests(unittest.TestCase):
    def trace(self):
        return dict(status='PASS', records={'pilot:0': {}, 'pilot:1': {}}, events=[
            dict(kind='model_batch', batch_id='b', request_ids=['pilot:0','pilot:1'],
                 start_positions=[4,0], input_lengths=[3,1], actual_input_tokens=4, padded_query_tokens=6),
            dict(kind='prefill_chunk', batch_id='b', request_id='pilot:0', start_position=4, input_tokens=3),
            dict(kind='token', batch_id='b', request_id='pilot:1', start_position=0, input_tokens=1),
            dict(kind='token', request_id='pilot:0', start_position=5, input_tokens=4)])

    def test_no_double_count_of_batch_rows_and_recomputation(self):
        result = replay_execution(self.trace())
        self.assertEqual(result['executed_tokens'], 8)
        self.assertEqual(result['reexecuted_tokens'], 2)
        self.assertEqual(result['padded_query_positions'], 2)

    def test_missing_row_or_geometry_must_fail(self):
        trace = self.trace(); del trace['events'][2]
        with self.assertRaises(ValueError): replay_execution(trace)
        trace = self.trace(); trace['events'][0]['input_lengths'] = [3]
        with self.assertRaises(ValueError): replay_execution(trace)
        trace = self.trace(); trace['status'] = 'FAIL'
        with self.assertRaises(ValueError): replay_execution(trace)
