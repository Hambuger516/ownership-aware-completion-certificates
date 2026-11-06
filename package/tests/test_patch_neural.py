"""Independent mathematical and training checks; CPU PyTorch is sufficient."""

from dataclasses import replace
import importlib.util
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from prc_nc.patch_exact import PatchStage, PatchLink, PatchGraph, PatchQuery, PatchExactSolver, evaluate_path
    from prc_nc.patch_neural import (
        NeuralPATCH, NeuralPATCHConfig, NeuralPatchSolver, dag_path_distribution,
        pack_graph, patch_losses, save_checkpoint, load_checkpoint, topology_id,
        validate_split_manifest, topological_order,
    )


def graph_case(*, capacity=8):
    # Intentionally scrambled stage IDs/input order, and different safe/unsafe
    # reliability branches. The cheaper teacher is genuinely feasible.
    stages = (
        PatchStage("end-b", "b", 1, 1, 3.0, .15, 2, 3.0, .4),
        PatchStage("start-a", "a", 0, 0, 1.0, .1, 2, 1.0, .99),
        PatchStage("end-a", "a", 1, 1, 1.0, .1, 2, 1.0, .99),
        PatchStage("start-b", "b", 0, 0, 3.0, .15, 2, 3.0, .4),
    )
    links = tuple(PatchLink(u, v, .02, .01)
                  for u in ("start-a", "start-b") for v in ("end-a", "end-b"))
    return PatchGraph(stages, links, 2, {"a": capacity, "b": capacity})


def small_model():
    torch.manual_seed(17)
    return NeuralPATCH(NeuralPATCHConfig(hidden_dim=16, gps_layers=1, transformer_layers=1,
                                        num_heads=4, adapter_rank=2, dropout=0.0)).double()


@unittest.skipUnless(HAS_TORCH, "neural PATCH requires optional torch")
class PathMarginalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_enumeration_partition_marginals_and_gradient(self):
        # 5 is source, 0 is sink; 4 is a dead branch. Numeric index order is wrong.
        edges = ((5, 2), (5, 3), (2, 1), (3, 1), (1, 0), (2, 0), (5, 4))
        weights = torch.tensor([.4, -.1, -.2, .7, .5, -.4, 10.], dtype=torch.float64, requires_grad=True)
        result = dag_path_distribution(weights, edges, 6, 5, 0)
        paths = ((0, 2, 4), (1, 3, 4), (0, 5))
        masses = torch.stack([torch.exp(weights[list(path)].sum()) for path in paths])
        expected = torch.stack([sum((masses[j] for j, path in enumerate(paths) if i in path), weights.sum() * 0)
                                / masses.sum() for i in range(len(edges))])
        self.assertTrue(torch.allclose(result.edge_marginals, expected, atol=1e-12))
        self.assertAlmostEqual(result.log_partition.item(), torch.log(masses.sum()).item())
        gradient = torch.autograd.grad(result.log_partition, weights)[0]
        self.assertTrue(torch.allclose(gradient, expected, atol=1e-12))
        for node in (1, 2, 3, 4):
            incoming = sum((result.edge_marginals[i] for i, (_, v) in enumerate(edges) if v == node), weights.sum() * 0)
            outgoing = sum((result.edge_marginals[i] for i, (u, _) in enumerate(edges) if u == node), weights.sum() * 0)
            self.assertAlmostEqual(float(incoming.detach()), float(outgoing.detach()), places=10)

    def test_cycle_rejected_and_masked_edges_zero(self):
        with self.assertRaises(ValueError):
            topological_order(2, ((0, 1), (1, 0)))
        edges = ((2, 0), (0, 3), (2, 1), (1, 3))
        out = dag_path_distribution(torch.zeros(4, dtype=torch.float64), edges, 4, 2, 3,
                                    enabled=(True, True, False, True))
        self.assertEqual(out.edge_marginals.tolist(), [1., 1., 0., 0.])

    def test_packed_costs_charge_each_stage_exactly_once(self):
        graph, query = graph_case(), PatchQuery(.6, .2, .1)
        packed = pack_graph(graph, query, dtype=torch.float64)
        for start, end in itertools.product(("start-a", "start-b"), ("end-a", "end-b")):
            solution = evaluate_path(graph, query, (start, end))
            path = (packed.source, packed.stage_ids.index(start), packed.stage_ids.index(end), packed.sink)
            positions = [packed.edges.index(edge) for edge in zip(path, path[1:])]
            self.assertAlmostEqual(packed.edge_ttft[positions].sum().item(), solution.ttft_seconds)
            self.assertAlmostEqual(packed.edge_cost[positions].sum().item(), solution.monetary_cost)


@unittest.skipUnless(HAS_TORCH, "neural PATCH requires optional torch")
class NeuralGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_each_enabled_loss_has_finite_nonzero_model_gradient(self):
        graph, query = graph_case(), PatchQuery(.5, .2, .9)
        teacher = PatchExactSolver().solve(graph, query).path
        model = small_model().eval()
        packed = pack_graph(graph, query, dtype=torch.float64)
        for name in ("imitation", "regret", "bottleneck", "reliability"):
            with self.subTest(loss=name):
                model.zero_grad(set_to_none=True)
                losses = patch_losses(model(packed), teacher)
                self.assertTrue(torch.isfinite(losses[name]))
                self.assertGreater(losses[name].item(), 0)
                losses[name].backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients), 1e-9)

    def test_preference_adapter_connected_and_changes_output(self):
        graph = graph_case()
        model = small_model().eval()
        first = model(pack_graph(graph, PatchQuery(.1, .2), dtype=torch.float64))
        second = model(pack_graph(graph, PatchQuery(.9, .2), dtype=torch.float64))
        self.assertFalse(torch.allclose(first.edge_scores, second.edge_scores))
        first.edge_scores.sum().backward()
        gradients = [p.grad for name, p in model.named_parameters() if ".adapter." in name]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_optimizer_updates_real_parameters(self):
        graph, query = graph_case(), PatchQuery(.5, .2, .9)
        model = small_model()
        packed = pack_graph(graph, query, dtype=torch.float64)
        teacher = PatchExactSolver().solve(graph, query).path
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        optimizer.zero_grad()
        patch_losses(model(packed), teacher)["total"].backward()
        optimizer.step()
        changed = [name for name, value in model.named_parameters() if not torch.equal(before[name], value)]
        self.assertTrue(any("graph_encoder" in name for name in changed))
        self.assertTrue(any("stage_transformer" in name for name in changed))
        self.assertTrue(any("cost_head" in name for name in changed))
        self.assertTrue(any("bottleneck_head" in name for name in changed))
        self.assertTrue(any("reliability_head" in name for name in changed))

    def test_tpot_mask_is_ground_truth_not_predicted_gate(self):
        graph, query = graph_case(), PatchQuery(.5, .12)
        out = small_model().eval()(pack_graph(graph, query, dtype=torch.float64))
        for i, (u, v) in enumerate(out.packed.edges):
            if u in (0, 3) or v in (0, 3):
                self.assertEqual(out.distribution.edge_marginals[i].item(), 0.)

    def test_constant_objective_does_not_amplify_flow_roundoff(self):
        graph = graph_case()
        graph = replace(graph, stages=tuple(replace(s, prefill_seconds=1., gpu_cost=2.) for s in graph.stages))
        query = PatchQuery(.5, .2, .1)
        model = small_model().float().eval()
        out = model(pack_graph(graph, query))
        losses = patch_losses(out, ("start-a", "end-a"))
        self.assertEqual(losses["regret"].item(), 0.)

    def test_malformed_teacher_flow_rejected(self):
        graph, query = graph_case(), PatchQuery(.5, .2, .1)
        out = small_model()(pack_graph(graph, query, dtype=torch.float64))
        with self.assertRaises(ValueError):
            patch_losses(out, ("start-a", "end-a"), teacher_marginals=torch.ones(len(out.packed.edges)))


@unittest.skipUnless(HAS_TORCH, "neural PATCH requires optional torch")
class NeuralCheckpointAndSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def provenance():
        return {"optimizer_steps": 1, "topology_splits": {
            "train": ["0" * 64], "validation": ["1" * 64], "test": ["2" * 64]},
            "purpose": "checkpoint-format-unit-test"}

    def test_strict_checkpoint_roundtrip_missing_key_and_no_random_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pt"
            model = small_model().float().eval()
            digest = save_checkpoint(model, path, self.provenance())
            loaded, metadata = load_checkpoint(path)
            self.assertEqual(metadata["checkpoint_sha256"], digest)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
            payload = torch.load(path, weights_only=True)
            payload["state_dict"].pop(next(iter(payload["state_dict"])))
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path)
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(Path(folder) / "missing.pt")
            with self.assertRaises(ValueError):
                NeuralPatchSolver(model)

    def test_topology_identity_and_split_leakage(self):
        graph = graph_case()
        altered = replace(graph, stages=tuple(replace(s, gpu_cost=s.gpu_cost + .4) for s in graph.stages))
        self.assertEqual(topology_id(graph), topology_id(altered))
        splits = {"train": [topology_id(graph)], "validation": [topology_id(altered)], "test": ["f" * 64]}
        with self.assertRaises(ValueError):
            validate_split_manifest(splits)

    def test_inference_really_calls_network_and_never_unconditional_exact(self):
        graph, query = graph_case(), PatchQuery(.5, .2, .1)
        model = small_model()
        calls = []
        handle = model.register_forward_hook(lambda *args: calls.append(1))
        with patch("prc_nc.patch_neural.PatchExactSolver.solve", side_effect=AssertionError("unexpected oracle")):
            result = NeuralPatchSolver(model, allow_untrained=True).solve(graph, query)
        handle.remove()
        self.assertEqual(calls, [1])
        self.assertFalse(result.used_exact_fallback)
        self.assertEqual(result.exact_fallback_seconds, 0)
        self.assertGreater(result.inference_seconds, 0)
        evaluate_path(graph, query, result.path)

    def test_invalid_candidate_repaired_and_fallback_counted_separately(self):
        graph, query = graph_case(capacity=3), PatchQuery(.5, .2, .1)
        model = small_model()
        with patch("prc_nc.patch_neural._k_paths", return_value=[("start-a", "end-a")]):
            repaired = NeuralPatchSolver(model, allow_untrained=True).solve(graph, query)
            fallback = NeuralPatchSolver(model, allow_untrained=True, max_repair_trials=0).solve(graph, query)
        self.assertTrue(repaired.repaired)
        self.assertFalse(repaired.used_exact_fallback)
        self.assertTrue(fallback.used_exact_fallback)
        self.assertGreater(fallback.exact_fallback_seconds, 0)
        self.assertEqual(fallback.repair_trials, 0)
        evaluate_path(graph, query, repaired.path)
        evaluate_path(graph, query, fallback.path)


if __name__ == "__main__":
    unittest.main()
