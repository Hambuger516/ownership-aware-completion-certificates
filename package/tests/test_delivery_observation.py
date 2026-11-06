import unittest
from prc_nc.delivery_observation import record_token_delivery,record_terminal_delivery


class DeliveryTests(unittest.TestCase):
    def test_buffered_delivery_keeps_actual_client_gap_and_bounded_summary(self):
        for retain in (True,False):
            record={}
            record_token_delivery(record,10,100,retain_history=retain)
            record_token_delivery(record,40,105,retain_history=retain)
            record_terminal_delivery(record,50,106)
            self.assertEqual(record['client_maximum_token_stall_ns'],5)
            self.assertEqual(record['client_output_token_count'],2)
            self.assertEqual(record['client_first_token_ns'],100)
            self.assertEqual(record['client_terminal_ns'],106)
            self.assertEqual(record.get('client_token_times_ns'),[100,105] if retain else None)

    def test_clock_reversal_does_not_mutate_record(self):
        record={}
        with self.assertRaises(ValueError):record_token_delivery(record,20,19,retain_history=True)
        self.assertEqual(record,{})
        record_token_delivery(record,10,100,retain_history=True);before=dict(record)
        with self.assertRaises(ValueError):record_token_delivery(record,40,99,retain_history=True)
        self.assertEqual(record,before)
        with self.assertRaises(ValueError):record_terminal_delivery(record,50,99)
