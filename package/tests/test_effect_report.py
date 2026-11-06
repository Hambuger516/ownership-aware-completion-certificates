import copy
import unittest
from prc_nc.matrices import matrix
from tools.summarize_effect import validate_matrix


class EffectReportTests(unittest.TestCase):
    def rows(self):
        return [dict(case=c["case"], method=c["method"], replicate=c["replicate"], status="PASS",
                     config=c, prompt_token_sha256="fixture", caps_sha256="fixture",
                     execution_measurement=dict(pending_batches=0, executed_tokens=10, reexecuted_tokens=2,
                         schedule_seconds=0.2, update_seconds=0.1, scheduler_lifecycle_seconds=0.3,
                         scope="fixture", timing_scope="fixture")) for c in matrix("effect")]

    def test_complete_measured_matrix_and_corrupted_or_mixed_evidence(self):
        rows = self.rows()
        validate_matrix(rows, require_execution=True)
        for key, value in (("schedule_seconds", float("nan")), ("pending_batches", 1),
                           ("timing_scope", "different"), ("reexecuted_tokens", 11),
                           ("scheduler_lifecycle_seconds", 0.4)):
            with self.subTest(key=key):
                corrupted = copy.deepcopy(rows)
                corrupted[0]["execution_measurement"][key] = value
                with self.assertRaises(ValueError):
                    validate_matrix(corrupted, require_execution=True)
        rows[0]["execution_measurement"] = None
        with self.assertRaises(ValueError):
            validate_matrix(rows)

    def test_old_results_cannot_satisfy_required_execution_report(self):
        rows = self.rows()
        for row in rows:
            row["execution_measurement"] = None
        validate_matrix(rows)
        with self.assertRaises(ValueError):
            validate_matrix(rows, require_execution=True)
