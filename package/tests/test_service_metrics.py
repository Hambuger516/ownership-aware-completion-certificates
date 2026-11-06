import unittest
from prc_nc.service_metrics import measure,from_prc,from_vllm,latency_summary


def row(rid,status,times,terminal,offered=0,submitted=0):
    return dict(request_id=rid,status=status,token_times_ns=times,terminal_ns=terminal,offered_ns=offered,submitted_ns=submitted,cap=8)


class ServiceMetricTests(unittest.TestCase):
    def test_tail_rejection_cancellation_and_unfinished_stay_in_denominator(self):
        rows=[row('a','completed',[100,200],210),row('b','completed',[900,1100],1200),
            row('c','rejected',[],20),row('d','cancelled',[30],40),row('e','pending',[],None)]
        r=measure(rows,start_ns=0,window_end_ns=1000,cutoff_ns=1500,ttft_limit_ns=1000,tpot_limit_ns=200)
        self.assertEqual(r['arrived'],5);self.assertEqual(r['window_completed'],1)
        self.assertEqual(r['window_output_tokens'],4);self.assertEqual(r['cohort_slo_rate'],.4)
        self.assertEqual(r['outcomes']['pending'],1)
        self.assertEqual(r['window_slo_completed'],1)
        self.assertEqual(r['cohort_slo_completed'],2)
        self.assertEqual(r['window_slo_throughput']*2,r['cohort_slo_throughput'])

    def test_latency_tails_keep_missing_and_single_output_explicit(self):
        rows=[row('a','completed',[10,20],30),row('b','completed',[100],110),row('c','rejected',[],5),row('d','pending',[40,90],None)]
        result=measure(rows,start_ns=0,window_end_ns=1000,cutoff_ns=1500)
        tails=latency_summary(result)
        self.assertEqual(tails['zero_output_requests'],1)
        self.assertEqual(tails['single_output_requests'],1)
        self.assertEqual(tails['ttft']['p50_ns'],40)
        self.assertEqual(tails['ttft']['p95_ns'],100)
        self.assertEqual(tails['ttft']['undefined_requests'],1)
        self.assertEqual(tails['mean_tpot']['undefined_requests'],2)
        self.assertEqual(tails['maximum_observed_token_stall']['max_ns'],50)
        self.assertEqual(tails['completed_end_to_end']['observed_requests'],2)
        self.assertEqual(tails['completed_end_to_end']['p99_ns'],110)

    def test_window_boundary_completion_is_tail_only(self):
        result=measure([row('a','completed',[10,20],1000)],start_ns=0,window_end_ns=1000,cutoff_ns=1000)
        self.assertEqual(result['window_slo_completed'],0)
        self.assertEqual(result['cohort_slo_completed'],1)

    def test_empty_latency_summary_does_not_invent_zeros(self):
        result=measure([],start_ns=0,window_end_ns=1000,cutoff_ns=1000)
        tails=latency_summary(result)
        self.assertIsNone(tails['ttft']['p50_ns'])
        self.assertIsNone(tails['mean_tpot']['max_ns'])

    def test_later_completion_does_not_leak_through_cutoff(self):
        r=measure([row('a','completed',[100,2000],2100)],start_ns=0,window_end_ns=1000,cutoff_ns=1500)
        self.assertEqual(r['outcomes']['pending'],1);self.assertEqual(r['cohort_slo_completed'],0)
        self.assertEqual(r['rows'][0]['output_tokens_at_cutoff'],1)

    def test_delay_is_measured_from_offered_time_and_stall_is_not_hidden(self):
        r=measure([row('a','completed',[900,950,1500],1600,submitted=800)],start_ns=0,window_end_ns=1000,cutoff_ns=2000,ttft_limit_ns=500)
        self.assertFalse(r['rows'][0]['slo_met']);self.assertEqual(r['rows'][0]['ttft_ns'],900)
        self.assertEqual(r['rows'][0]['maximum_token_stall_ns'],550)

    def test_adapters_match_same_trajectory(self):
        prc=dict(records={'a':dict(request_id='a',scheduled_ns=0,arrival_ns=100_000_000,status='completed',terminal_ns=500_000_000,max_new_tokens=8,token_times_ns=[200_000_000,400_000_000])})
        v=[dict(request_id='a',wave=0,offered_at_s=0,submitted_at_s=.1,terminal='completed',finished_at_s=.5,cap=8,token_times_s=[.2,.4])]
        args=dict(start_ns=0,window_end_ns=1_000_000_000,cutoff_ns=2_000_000_000)
        self.assertEqual(measure(from_prc(prc),**args),measure(from_vllm(v,wave=0),**args))
        self.assertEqual(list(from_vllm(v,wave=1)),[])

    def test_client_basis_changes_window_and_stall_without_rewriting_production(self):
        record=dict(request_id='a',scheduled_ns=0,arrival_ns=0,status='completed',
            terminal_ns=45,max_new_tokens=2,token_times_ns=[10,40],
            client_token_times_ns=[100,105],client_output_token_count=2,client_terminal_ns=106)
        report=dict(records={'a':record})
        args=dict(start_ns=0,window_end_ns=50,cutoff_ns=110)
        old=measure(from_prc(report),**args)
        new=measure(from_prc(report,timing_basis='client'),**args)
        self.assertEqual(old['window_output_tokens'],2)
        self.assertEqual(new['window_output_tokens'],0)
        self.assertEqual(old['rows'][0]['maximum_token_stall_ns'],30)
        self.assertEqual(new['rows'][0]['maximum_token_stall_ns'],5)
        for key in ('client_token_times_ns','client_output_token_count','client_terminal_ns'):
            broken=dict(record);del broken[key]
            with self.assertRaises(ValueError):list(from_prc(dict(records={'a':broken}),timing_basis='client'))
        with self.assertRaises(ValueError):list(from_prc(report,timing_basis='guess'))

    def test_invalid_timeline_and_duplicate_rejected(self):
        for rows in ([row('a','completed',[200,100],300)],[row('a','completed',[100],90)],[row('a','completed',[],100)],[row('a','pending',[],None)]*2):
            with self.assertRaises(ValueError):measure(rows,start_ns=0,window_end_ns=1000,cutoff_ns=2000)


if __name__=='__main__':unittest.main()
