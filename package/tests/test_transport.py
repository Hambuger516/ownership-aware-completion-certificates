import unittest
from prc_nc.protocol_probe import run_probe


class LoopbackProtocolTests(unittest.TestCase):
    def test_real_socket_delay_duplicate_order_and_physical_release(self):
        result=run_probe()
        self.assertEqual(result["status"],"PASS")
        self.assertEqual(result["physical_allocations"],1)
        self.assertEqual(result["final_physical_free"],(4,))
        # The probe records one message per protocol transition.  Eleven
        # transitions cover the complete delayed/duplicate/stale/stop path;
        # keep the assertion tied to the actual protocol rather than a stale
        # count from an earlier probe revision.
        self.assertGreaterEqual(len(result["messages"]),11)
