import unittest
from prc_nc.execution_meter import ExecutionTokenMeter


class ExecutionMeterTests(unittest.TestCase):
    def test_explicit_cohort_does_not_infer_workload_from_names(self):
        m = ExecutionTokenMeter()
        m.prepare(0, [("pilot:0", 4, 8), ("continuous:1", 0, 3), ("warmup", 0, 7)])
        m.confirm(0, ["pilot:0", "continuous:1", "warmup"])
        m.prepare(1, [("pilot:0", 8, 8)])
        m.confirm(1, ["pilot:0"])
        report = m.report(request_ids=["pilot:0", "continuous:1", "rejected:2"])
        self.assertEqual(report['executed_tokens'], 19)
        self.assertEqual(report['reexecuted_tokens'], 4)
        self.assertEqual(set(report['requests']), {'pilot:0', 'continuous:1'})
        self.assertEqual(m.report(request_ids=[])['executed_tokens'], 0)
        self.assertEqual(m.report(request_ids=['warmup'])['executed_tokens'], 7)

    def test_confirmed_intersections_not_discarded_or_cached_positions(self):
        m = ExecutionTokenMeter()
        # Prefix [0,16) was reused from cache, not computed for this request.
        m.prepare(1, [("w0-r0", 16, 16)])
        self.assertEqual(m.report()["executed_tokens"], 0)
        m.confirm(1, ["w0-r0"])
        m.prepare(2, [("w0-r0", 8, 32)])
        m.confirm(2, ["w0-r0"])
        self.assertEqual(m.report()["executed_tokens"], 48)
        self.assertEqual(m.report()["reexecuted_tokens"], 16)
        self.assertFalse(m.confirm(2, ["w0-r0"]))
        m.prepare(3, [("w0-r0", 8, 32)])
        m.confirm(3, ["w0-r0"])
        self.assertEqual(m.report()["reexecuted_tokens"], 48)

    def test_failed_execution_does_not_count_and_validation_is_atomic(self):
        m = ExecutionTokenMeter()
        m.prepare(1, [("w0-r0", 0, 4), ("w0-r1", 0, 4)])
        with self.assertRaises(ValueError):
            m.confirm(1, ["w0-r0"])
        self.assertEqual(m.report()["executed_tokens"], 0)
        self.assertEqual(m.report()["pending_batches"], 1)
        m.confirm(1, ["w0-r0", "w0-r1"])
        self.assertEqual(m.report()["executed_tokens"], 8)

    def test_disjoint_intervals_and_warmup(self):
        m = ExecutionTokenMeter()
        for batch, start in enumerate((0, 8, 2)):
            m.prepare(batch, [("w0-r0", start, 4), ("warmup-1", start, 4)])
            m.confirm(batch, ["w0-r0", "warmup-1"])
        self.assertEqual(m.report()["executed_tokens"], 12)
        self.assertEqual(m.report()["reexecuted_tokens"], 2)
