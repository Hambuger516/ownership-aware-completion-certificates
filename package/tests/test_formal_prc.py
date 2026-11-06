import unittest

from prc_nc.gpu_run import _build_prc_proposals, _formal_predictor
from prc_nc.prediction import load_runtime_predictor, PredictionQuery


class FormalPRCProposalTests(unittest.TestCase):
    def test_configured_reserve_coefficient_survives_both_checkpoint_loads(self):
        low, path, digest = _formal_predictor({"predictor_z": 0.0})
        high, _, _ = _formal_predictor({"predictor_z": 2.0})
        query = PredictionQuery("r", 80, 0, "default", 1024)
        a, b = low.predict(query), high.predict(query)
        self.assertEqual(low.z_value, 0.0)
        self.assertEqual(a.remaining_mean_tokens, b.remaining_mean_tokens)
        self.assertEqual(a.remaining_variance_tokens2, b.remaining_variance_tokens2)
        self.assertLess(a.soft_upper_tokens, b.soft_upper_tokens)
        proposal = _build_prc_proposals(
            {"predictor_z": 0.0, "proposal_mode": "exact", "waves": 1},
            None, [[1] * 80], [128], 96, 16)[0]
        self.assertEqual(proposal["predictor_z"], 0.0)
        runtime, runtime_digest = load_runtime_predictor(
            proposal["predictor_checkpoint"], z_value=proposal["predictor_z"])
        self.assertEqual(runtime.z_value, 0.0)
        self.assertEqual(runtime_digest, digest)
        self.assertEqual(runtime.predict(query), a)
        for invalid in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                load_runtime_predictor(path, z_value=invalid)

    def test_formal_proposal_contains_all_boundaries_and_ignores_future_answer(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text.split())))

        config = {"predictor_z": 2.0, "proposal_mode": "exact", "waves": 1}
        prompts = [[1] * 17]
        caps = [33]
        first = _build_prc_proposals(config, Tokenizer(), prompts, caps, 96, 16,
                                     data_rows=[{"category": "open_qa", "reference_response": "one"}])
        second = _build_prc_proposals(config, Tokenizer(), prompts, caps, 96, 16,
                                      data_rows=[{"category": "open_qa", "reference_response": "" * 10000}])
        self.assertEqual(first[0]["hard_max_output_tokens"], 33)
        self.assertEqual(first[0]["hard_need_pages"], 4)
        self.assertEqual(first[0]["route"]["device_id"], "gpu0")
        self.assertEqual(first[0]["placement"]["devices"], ["gpu0"])
        self.assertEqual(first[0]["placement"]["solver"], "PATCH-Exact")
        self.assertEqual(first[0]["soft_reserve_tokens"], second[0]["soft_reserve_tokens"])
        self.assertIsNone(first[0]["patch_checkpoint"])
        self.assertTrue(first[0]["predictor_checkpoint_sha256"])
        self.assertTrue(first[0]["predictor_split_sha256"])
        self.assertTrue(first[0]["predictor_trace_sha256"])


if __name__ == "__main__":
    unittest.main()
