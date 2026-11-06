import json
import tempfile
import unittest
from pathlib import Path

from prc_nc.prediction import (
    ConditionalResidualPredictor,
    PredictionQuery,
    TraceRecord,
    load_real_generation_traces,
    make_split_manifest,
)
from prc_nc.routing import NoFeasibleRoute, RouteDevice, RouteRequest, TypedRoute


def _trace(source_index: int, observed: int, final: int, *, category: str = "qa") -> TraceRecord:
    return TraceRecord(
        source_index=source_index,
        run_id="real-run",
        request_id=f"r{source_index}",
        prompt_tokens=32 + source_index,
        observed_output_tokens=observed,
        final_output_tokens=final,
        category=category,
        trace_sha256=f"trace-{source_index}",
        manifest_sha256=f"manifest-{source_index}",
    )


class SplitAndPredictorTests(unittest.TestCase):
    def test_required_512_split_is_fixed_and_disjoint(self):
        first = make_split_manifest(range(512), seed=20260919)
        second = make_split_manifest(range(512), seed=20260919)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.counts, {"calibration": 307, "validation": 102, "test": 103})
        self.assertEqual(set(first.assignments), set(range(512)))

    def test_fit_requires_calibration_real_trace_and_clips_only_soft_reserve(self):
        manifest = make_split_manifest(range(8), seed=3)
        calibration_ids = [i for i in range(8) if manifest.split_of(i) == "calibration"]
        rows = [_trace(i, 0, 10 + i) for i in calibration_ids]
        predictor = ConditionalResidualPredictor(z_value=2.0).fit(rows, split_manifest=manifest)
        result = predictor.predict(PredictionQuery("new", 80, observed_output_tokens=2, category="qa", max_output_tokens=5))
        self.assertEqual(result.calibration_state, "calibrated")
        self.assertLessEqual(result.soft_upper_tokens, 3)
        self.assertGreaterEqual(result.remaining_mean_tokens, 0)
        with self.assertRaises(ValueError):
            predictor.fit([_trace(i, 0, 10) for i in range(8) if manifest.split_of(i) != "calibration"], split_manifest=manifest)

    def test_checkpoint_roundtrip_preserves_population_variance(self):
        manifest = make_split_manifest(range(8), seed=3)
        calibration_ids = [i for i in range(8) if manifest.split_of(i) == "calibration"]
        rows = [_trace(i, 0, 10 + i) for i in calibration_ids]
        predictor = ConditionalResidualPredictor(z_value=2.0).fit(rows, split_manifest=manifest)
        query = PredictionQuery("roundtrip", 32, category="qa")
        expected = predictor.predict(query)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictor.json"
            predictor.save_checkpoint(path)
            restored = ConditionalResidualPredictor()
            restored.load_checkpoint(path, expected_split_sha256=manifest.sha256)
            actual = restored.predict(query)
        self.assertAlmostEqual(actual.remaining_mean_tokens, expected.remaining_mean_tokens)
        self.assertAlmostEqual(actual.remaining_variance_tokens2, expected.remaining_variance_tokens2)

    def test_progress_records_from_one_trajectory_cannot_cross_split(self):
        manifest = make_split_manifest(range(32), seed=7)
        trajectory = "real-run/r5"
        records = [_trace(5, observed, 9) for observed in range(10)]
        self.assertEqual({manifest.split_of(x.source_index) for x in records}, {manifest.split_of(5)})
        self.assertEqual({x.trajectory_id for x in records}, {trajectory})

    def test_loader_uses_manifest_source_index_and_rejects_missing_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "good").mkdir()
            manifest = {
                "configuration": {"run_id": "good-run"},
                "selection": [{"source_index": 7, "prompt_tokens": 12}],
            }
            (root / "good" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "good" / "requests.json").write_text(json.dumps([{
                "request_id": "w0-r0", "index": 0, "terminal": "completed", "tokens": [1, 2, 3],
            }]), encoding="utf-8")
            corpus = load_real_generation_traces(root, source_rows=({"source_index": 7, "category": "qa"},), expected_source_count=None)
            self.assertEqual(len(corpus.records), 4)
            self.assertEqual({x.source_index for x in corpus.records}, {7})
            self.assertEqual({x.observed_output_tokens for x in corpus.records}, set(range(4)))

    def test_loader_commits_a_run_atomically_after_every_request_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "partly-bad"
            run.mkdir()
            (run / "manifest.json").write_text(json.dumps({
                "configuration": {"run_id": "partly-bad"},
                "selection": [
                    {"source_index": 1, "prompt_tokens": 12},
                    {"source_index": 2, "prompt_tokens": 13},
                ],
            }), encoding="utf-8")
            # The first request is valid; the second has no token list.  With
            # strict=False the entire run is rejected, including the valid
            # prefix, so calibration cannot depend on file ordering.
            (run / "requests.json").write_text(json.dumps([
                {"request_id": "w0-r0", "index": 0, "terminal": "completed", "tokens": [1, 2]},
                {"request_id": "w0-r1", "index": 1, "terminal": "completed"},
            ]), encoding="utf-8")
            manifest = make_split_manifest((1, 2), seed=4)
            corpus = load_real_generation_traces(
                root,
                source_rows=({"source_index": 1}, {"source_index": 2}),
                split_manifest=manifest,
                expected_source_count=None,
                strict=False,
            )
            self.assertEqual(corpus.records, ())
            self.assertEqual(len(corpus.rejected), 1)

    def test_checkpoint_rejects_extra_or_missing_fields_without_mutating_model(self):
        manifest = make_split_manifest(range(8), seed=3)
        calibration_ids = [i for i in range(8) if manifest.split_of(i) == "calibration"]
        predictor = ConditionalResidualPredictor().fit([_trace(i, 0, 10 + i) for i in calibration_ids], split_manifest=manifest)
        before = predictor.predict(PredictionQuery("before", 32, category="qa"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            payload = predictor.checkpoint_dict()
            payload["unexpected"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                predictor.load_checkpoint(path)
        after = predictor.predict(PredictionQuery("after", 32, category="qa"))
        self.assertEqual(before.remaining_mean_tokens, after.remaining_mean_tokens)
        self.assertEqual(before.remaining_variance_tokens2, after.remaining_variance_tokens2)
        self.assertEqual(before.soft_upper_tokens, after.soft_upper_tokens)
        self.assertEqual(before.support, after.support)


class TypedRouteTests(unittest.TestCase):
    def test_darc_two_stage_weights_and_feasible_filter(self):
        devices = (
            RouteDevice("g0", "d0", compute_pressure=.1, memory_pressure=.2, queue_pressure=.3, locality=.9, free_capacity=4),
            RouteDevice("g1", "d0", compute_pressure=.9, memory_pressure=.9, queue_pressure=.9, locality=.1, free_capacity=100),
            RouteDevice("g2", "d1", compute_pressure=.9, memory_pressure=.9, queue_pressure=.9, locality=.0, free_capacity=1),
        )
        proposal = TypedRoute().choose(RouteRequest("r", "d0", hard_need=4), devices)
        self.assertEqual(proposal.device_id, "g0")
        self.assertEqual(proposal.assignment_domain, "d0")
        self.assertEqual(proposal.feasible_device_ids, ("g0", "g1"))
        self.assertAlmostEqual(proposal.request_domain_residual, 0.0)
        self.assertAlmostEqual(proposal.domain_device_residual, 0.0)

    def test_cross_domain_realization_is_explicit_when_macro_domain_has_no_feasible_gpu(self):
        devices = (
            RouteDevice("g0", "d0", compute_pressure=.0, memory_pressure=.0, queue_pressure=.0, locality=1.0, free_capacity=0),
            RouteDevice("g1", "d1", compute_pressure=.2, memory_pressure=.2, queue_pressure=.2, locality=.0, free_capacity=8),
        )
        proposal = TypedRoute().choose(RouteRequest("r", "d0", hard_need=4), devices)
        self.assertEqual(proposal.device_id, "g1")
        self.assertEqual(proposal.domain_device_residual, TypedRoute.BETA)

    def test_no_feasible_and_single_device_degenerate(self):
        with self.assertRaises(NoFeasibleRoute):
            TypedRoute().choose(RouteRequest("r", "d0", hard_need=2), (RouteDevice("g0", "d0", free_capacity=1),))
        proposal = TypedRoute().choose(RouteRequest("r", "preferred", hard_need=2), (RouteDevice("g0", "d0", free_capacity=4),))
        self.assertEqual(proposal.assignment_domain, "d0")
        self.assertEqual(proposal.device_id, "g0")

    def test_residual_disabled_preserves_typed_feasibility(self):
        devices = (
            RouteDevice("g0", "d0", free_capacity=8),
            RouteDevice("g1", "d1", free_capacity=8),
        )
        proposal = TypedRoute(use_residual=False).choose(RouteRequest("r", "d0", hard_need=1), devices)
        self.assertFalse(proposal.residual_enabled)
        self.assertEqual(proposal.request_domain_residual, 0.0)
        self.assertEqual(proposal.domain_device_residual, 0.0)


if __name__ == "__main__":
    unittest.main()
