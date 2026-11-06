"""Direct tests for the online PRC proposal layer, without vLLM."""

import unittest

from prc_nc.online import ObservedRequest, OnlineProposalEngine
from prc_nc.prediction import PredictionResult
from prc_nc.routing import RouteDevice


class Predictor:
    z_value = 2.0
    calibration_state = "calibrated"

    def __init__(self, mean):
        self.mean = mean

    def predict(self, query):
        mean = min(self.mean, query.max_output_tokens - query.observed_output_tokens)
        return PredictionResult(query.request_id, mean, 0.0, int(mean), "calibrated", "test", "fixture", 1)


def row(*, request_id="new", max_tokens=64, remaining_pages=4):
    return ObservedRequest(request_id, 16, 0, max_tokens, 1, remaining_pages, "default")


class OnlineProposalTests(unittest.TestCase):
    def device(self, free):
        return RouteDevice(
            "gpu0", "default", compute_pressure=.25, memory_pressure=.5,
            queue_pressure=.125, locality=.75, free_capacity=free,
            metadata={"pool_id": "gpu0-kv"},
        )

    def test_current_telemetry_and_version_are_carried_into_route_proposal(self):
        engine = OnlineProposalEngine(Predictor(0), block_size=16)
        proposal = engine.propose(
            "new", "new:a0", 11, [row()], capacity=8, free_before=2,
            free_after=1, growth_pages=1, device=self.device(2),
            pool_id="gpu0-kv", new_admission=True,
        )
        self.assertTrue(proposal["proposal_allowed"])
        self.assertEqual(proposal["state_version"], 11)
        self.assertEqual(proposal["route"]["device_id"], "gpu0")
        self.assertEqual(proposal["telemetry"]["queue_pressure"], .125)
        self.assertEqual(proposal["placement"]["solver"], "PATCH-Exact")

    def test_statistical_reserve_only_defers_new_admission(self):
        engine = OnlineProposalEngine(Predictor(64), block_size=16)
        proposal = engine.propose(
            "new", "new:a0", 0, [row()], capacity=8, free_before=2,
            free_after=1, growth_pages=1, device=self.device(2),
            pool_id="gpu0-kv", new_admission=True,
        )
        self.assertFalse(proposal["proposal_allowed"])
        self.assertFalse(proposal["statistical_reserve"]["fits"])
        existing = engine.propose(
            "new", "new:a1", 1, [row()], capacity=8, free_before=2,
            free_after=1, growth_pages=1, device=self.device(2),
            pool_id="gpu0-kv", new_admission=False,
        )
        self.assertTrue(existing["proposal_allowed"])

    def test_hard_route_filter_precedes_scoring(self):
        engine = OnlineProposalEngine(Predictor(0), block_size=16)
        proposal = engine.propose(
            "new", "new:a0", 0, [row(max_tokens=16, remaining_pages=1)], capacity=8,
            free_before=0, free_after=0, growth_pages=1, device=self.device(0),
            pool_id="gpu0-kv", new_admission=True,
        )
        self.assertIsNone(proposal["route"])
        self.assertFalse(proposal["proposal_allowed"])
        self.assertIsNone(proposal["placement"])


if __name__ == "__main__":
    unittest.main()
