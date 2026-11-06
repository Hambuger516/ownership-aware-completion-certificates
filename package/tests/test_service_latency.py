import unittest
from prc_nc.service_latency import summarize


def row(rid,times,status='completed',terminal=100):
    return dict(request_id=rid,offered_ns=0,submitted_ns=0,cap=10,
                token_times_ns=times,status=status,terminal_ns=terminal)


class LatencyPopulationTests(unittest.TestCase):
    def test_rejection_and_single_output_are_not_zero_latency(self):
        result=summarize([row('a',[10,20,80]),row('b',[],status='rejected'),row('c',[30])],
                         start_ns=0,window_end_ns=50,cutoff_ns=100)
        self.assertEqual(result['arrived'],3)
        self.assertEqual(result['no_output_at_cutoff'],1)
        self.assertEqual(result['single_output_at_cutoff'],1)
        d=result['request_distributions']
        self.assertEqual(d['ttft_ns']['observed_count'],2)
        self.assertEqual(d['maximum_token_stall_ns']['unavailable_count'],2)
        self.assertEqual(d['maximum_token_stall_ns']['seconds']['p99'],60/1e9)
        self.assertEqual(result['pooled_inter_token_gaps']['observed_count'],2)

    def test_cutoff_censors_later_tokens_and_completion(self):
        result=summarize([row('a',[10,20,80],terminal=100)],
                         start_ns=0,window_end_ns=50,cutoff_ns=60)
        self.assertEqual(result['outcomes']['pending'],1)
        self.assertEqual(result['completed_end_to_end']['observed_count'],0)
        self.assertEqual(result['pooled_inter_token_gaps']['maximum_seconds'],10/1e9)

    def test_invalid_clock_order_remains_an_error(self):
        with self.assertRaises(ValueError):
            summarize([row('a',[20,10])],start_ns=0,window_end_ns=50,cutoff_ns=100)


if __name__=='__main__':
    unittest.main()
