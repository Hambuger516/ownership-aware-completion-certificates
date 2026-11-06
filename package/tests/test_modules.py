import unittest
import json
from pathlib import Path

from prc_nc.modules import (
    DeviceState,
    PatchExactSolver,
    PatchLearnedSolver,
    PlacementProblem,
    PredictionObservation,
    PRCProposalPipeline,
    ResidualLengthPredictor,
    RouteRequest,
    TypedRoute,
)
from prc_nc.gpu_run import _build_prc_proposals


class PredictorTests(unittest.TestCase):
    def test_prediction_uses_remaining_not_total_and_respects_cap(self):
        p = ResidualLengthPredictor(z_value=2.0).fit(
            [
                {"category": "chat", "observed_output_tokens": 0, "final_output_tokens": 10},
                {"category": "chat", "observed_output_tokens": 2, "final_output_tokens": 12},
                {"category": "code", "observed_output_tokens": 0, "final_output_tokens": 100},
            ]
        )
        out = p.predict(PredictionObservation("r", 20, observed_output_tokens=5, max_output_tokens=8, category="chat"))
        self.assertLessEqual(out.soft_upper_tokens, 3)
        self.assertGreaterEqual(out.remaining_mean_tokens, 0)
        self.assertEqual(out.support, "category")

    def test_unfitted_falls_back_to_explicit_zero_not_future_length(self):
        out = ResidualLengthPredictor().predict(PredictionObservation("r", 10, max_output_tokens=20))
        self.assertEqual(out.soft_upper_tokens, 0)
        self.assertEqual(out.support, "unfitted")


class RouteTests(unittest.TestCase):
    def test_feasibility_is_hard_and_single_device_is_deterministic(self):
        route = TypedRoute().choose(
            RouteRequest("r", "d0", min_free_pages=4),
            [
                DeviceState("gpu0", "d0", free_pages=2),
                DeviceState("gpu1", "d1", free_pages=8, locality=1.0),
            ],
        )
        self.assertEqual(route.device_id, "gpu1")
        self.assertEqual(route.candidates, ("gpu1",))


def problem():
    return PlacementProblem(
        stage_devices=(("g0", "g1"), ("g0", "g1")),
        edge_cost={(0, "g0"): 1.0, (0, "g1"): 4.0, (1, "g0"): 4.0, (1, "g1"): 1.0},
        memory_pages={(0, "g0"): 3, (0, "g1"): 3, (1, "g0"): 3, (1, "g1"): 3},
        device_memory_pages={"g0": 3, "g1": 6},
        reliability={(0, "g0"): 1.0, (0, "g1"): 1.0, (1, "g0"): 1.0, (1, "g1"): 1.0},
        preferred_devices=("g0",),
    )


class PatchTests(unittest.TestCase):
    def test_exact_checks_cumulative_memory(self):
        sol = PatchExactSolver().solve(problem())
        self.assertEqual(sol.devices, ("g0", "g1"))
        self.assertEqual(sol.memory_by_device, {"g0": 3, "g1": 3})

    def test_learned_path_has_nonzero_gradient_and_repairs_constraints(self):
        exact = PatchExactSolver().solve(problem())
        learned = PatchLearnedSolver(epochs=3)
        gradient = learned.gradient(problem(), ("g1", "g1"), exact.devices)
        self.assertTrue(any(abs(x) > 0 for x in gradient))
        learned.fit([(problem(), exact.devices)])
        sol = learned.solve(problem())
        self.assertLessEqual(sol.memory_by_device["g0"], 3)

    def test_pipeline_keeps_hard_contract_separate(self):
        predictor = ResidualLengthPredictor().fit(
            [{"category": "default", "observed_output_tokens": 0, "final_output_tokens": 4}]
        )
        pipeline = PRCProposalPipeline(predictor, TypedRoute(), block_size=16)
        proposal = pipeline.propose(
            PredictionObservation("r", prompt_tokens=17, max_output_tokens=33),
            [DeviceState("gpu0", "default", free_pages=100)],
            state_version=7,
        )
        self.assertEqual(proposal.hard_max_output_tokens, 33)
        self.assertEqual(proposal.hard_need_pages, 4)
        self.assertEqual(proposal.state_version, 7)
        self.assertEqual(proposal.route.device_id, "gpu0")

    def test_gpu_proposal_manifest_is_hard_contract_bound(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text.split())))

        config = {"predictor_z": 2.0, "proposal_mode": "exact", "waves": 2}
        proposals = _build_prc_proposals(config, Tokenizer(), [[1, 2, 3]], [33], 96, 16)
        self.assertEqual(len(proposals), 2)
        self.assertEqual(proposals[0]["hard_max_output_tokens"], 33)
        self.assertEqual(proposals[0]["hard_need_pages"], 3)
        self.assertEqual(proposals[1]["state_version"], 0)
        self.assertIn("offline preflight", proposals[1]["scope"])


if __name__ == "__main__":
    unittest.main()


class LearnedCheckpointTests(unittest.TestCase):
    def test_strict_checkpoint_roundtrip(self):
        import tempfile
        solver = PatchLearnedSolver(epochs=2)
        p = problem()
        target = PatchExactSolver().solve(p).devices
        solver.fit([(p, target)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patch.json"
            digest = solver.save_checkpoint(path)
            restored = PatchLearnedSolver(epochs=0)
            self.assertEqual(restored.load_checkpoint(path), digest)
            self.assertEqual(restored.weights, solver.weights)
            payload = json.loads(path.read_text())
            payload["extra"] = True
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                restored.load_checkpoint(path)
