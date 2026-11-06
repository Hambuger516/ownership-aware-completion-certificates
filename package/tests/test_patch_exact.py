"""Independent small-graph enumeration, not an oracle calling the solver."""

from dataclasses import replace
import math
import random
import unittest

from prc_nc.patch_exact import (
    NoFeasiblePlacement,
    PatchExactSolver,
    PatchGraph,
    PatchLink,
    PatchNormalization,
    PatchQuery,
    PatchStage,
    SearchLimitExceeded,
    evaluate_path,
    topological_order,
)


def stage(key, device, layer, pre=1.0, dec=1.0, memory=1, cost=1.0, availability=1.0, end=None):
    return PatchStage(key, device, layer, layer if end is None else end, pre, dec, memory, cost, availability)


def exhaustive_oracle(graph, query):
    """DFS all complete paths and evaluate the equations independently."""
    nodes = {s.stage_id: s for s in graph.stages}
    outgoing = {key: [] for key in nodes}
    for edge in graph.links:
        outgoing[edge.source].append(edge)
    complete = []

    def visit(path, transitions):
        last = nodes[path[-1]]
        if last.end_layer == graph.layer_count - 1:
            selected = [nodes[key] for key in path]
            if any(s.memory_pages > graph.device_memory_pages[s.device_id] for s in selected):
                return
            memory = {d: sum(s.memory_pages for s in selected if s.device_id == d)
                      for d in graph.device_memory_pages}
            complete.append(dict(
                path=path,
                ttft=sum(s.prefill_seconds for s in selected) + sum(e.transfer_seconds for e in transitions),
                cost=sum(s.gpu_cost for s in selected) + sum(e.transfer_cost for e in transitions),
                tpot=max(s.decode_seconds for s in selected),
                reliability=math.prod(s.availability for s in selected),
                memory=memory,
            ))
            return
        for edge in outgoing[last.stage_id]:
            visit(path + (edge.target,), transitions + (edge,))

    for candidate in graph.stages:
        if candidate.start_layer == 0:
            visit((candidate.stage_id,), ())
    if not complete:
        return []
    normal = query.normalization
    if normal is None:
        low_t, high_t = min(r["ttft"] for r in complete), max(r["ttft"] for r in complete)
        low_c, high_c = min(r["cost"] for r in complete), max(r["cost"] for r in complete)
        epsilon = 1e-12
    else:
        low_t, high_t, low_c, high_c, epsilon = (
            normal.ttft_min, normal.ttft_max, normal.cost_min, normal.cost_max, normal.epsilon
        )
    feasible = []
    for row in complete:
        if row["tpot"] > query.tpot_budget or row["reliability"] < query.min_reliability:
            continue
        if any(value > graph.device_memory_pages[d] for d, value in row["memory"].items()):
            continue
        row["objective"] = query.lambda_pref * (row["ttft"] - low_t) / (high_t - low_t + epsilon)
        row["objective"] += (1 - query.lambda_pref) * (row["cost"] - low_c) / (high_c - low_c + epsilon)
        feasible.append(row)
    return sorted(feasible, key=lambda row: (row["objective"], row["path"]))


class PatchExactTests(unittest.TestCase):
    def test_unsorted_graph_charges_each_stage_once(self):
        graph = PatchGraph(
            (stage("a-final", "a", 2, pre=7, cost=11),
             stage("z-first", "a", 0, pre=2, cost=3),
             stage("m-middle", "b", 1, pre=5, cost=7)),
            (PatchLink("m-middle", "a-final", 0.5, 2), PatchLink("z-first", "m-middle", 0.25, 1)),
            3, {"a": 2, "b": 1},
        )
        result = PatchExactSolver().solve(graph, PatchQuery(0.5, 1.0))
        self.assertEqual(topological_order(graph), ("z-first", "m-middle", "a-final"))
        self.assertEqual(result.path, ("z-first", "m-middle", "a-final"))
        self.assertEqual(result.ttft_seconds, 14.75)
        self.assertEqual(result.monetary_cost, 24)
        self.assertEqual(dict(result.memory_by_device), {"a": 2, "b": 1})

    def test_cost_preference_is_not_discarded(self):
        graph = PatchGraph((stage("fast", "a", 0, pre=1, cost=10),
                            stage("cheap", "b", 0, pre=10, cost=1)), (), 1, {"a": 1, "b": 1})
        self.assertEqual(PatchExactSolver().solve(graph, PatchQuery(1, 1)).path, ("fast",))
        self.assertEqual(PatchExactSolver().solve(graph, PatchQuery(0, 1)).path, ("cheap",))

    def test_tpot_is_stage_max_not_sum_or_transfer_delay(self):
        graph = PatchGraph((stage("x", "a", 0, dec=2), stage("y", "b", 1, dec=3)),
                           (PatchLink("x", "y", transfer_seconds=100),), 2, {"a": 1, "b": 1})
        result = PatchExactSolver().solve(graph, PatchQuery(1, 3))
        self.assertEqual(result.tpot_seconds, 3)
        self.assertEqual(result.ttft_seconds, 102)
        with self.assertRaises(NoFeasiblePlacement):
            PatchExactSolver().solve(graph, PatchQuery(1, 2.9))

    def test_locally_feasible_segments_can_exceed_cumulative_memory(self):
        graph = PatchGraph((stage("x", "a", 0, memory=2), stage("y", "a", 1, memory=2)),
                           (PatchLink("x", "y"),), 2, {"a": 3})
        with self.assertRaises(NoFeasiblePlacement):
            PatchExactSolver().solve(graph, PatchQuery(0.5, 1))
        with self.assertRaises(NoFeasiblePlacement):
            evaluate_path(graph, PatchQuery(0.5, 1), ("x", "y"))

    def test_reliability_nondominated_label_survives_cheaper_prefix(self):
        # Keeping only the cheaper path at the merge discards the only feasible solution.
        graph = PatchGraph(
            (stage("cheap-risky", "a", 0, pre=1, cost=1, availability=0.8),
             stage("costly-safe", "b", 0, pre=2, cost=2, availability=1),
             stage("merge", "c", 1, availability=1), stage("last", "d", 2, availability=0.9)),
            (PatchLink("cheap-risky", "merge"), PatchLink("costly-safe", "merge"), PatchLink("merge", "last")),
            3, {key: 2 for key in "abcd"},
        )
        result = PatchExactSolver().solve(graph, PatchQuery(0.5, 1, min_reliability=0.75))
        self.assertEqual(result.path, ("costly-safe", "merge", "last"))
        self.assertAlmostEqual(result.reliability, 0.9)

    def test_memory_nondominated_label_survives_cheaper_prefix(self):
        graph = PatchGraph(
            (stage("cheap", "a", 0, pre=1), stage("expensive", "b", 0, pre=2),
             stage("merge", "c", 1), stage("final", "a", 2)),
            (PatchLink("cheap", "merge"), PatchLink("expensive", "merge"), PatchLink("merge", "final")),
            3, {"a": 1, "b": 1, "c": 1},
        )
        result = PatchExactSolver().solve(graph, PatchQuery(1, 1))
        self.assertEqual(result.path, ("expensive", "merge", "final"))

    def test_repeated_device_reliability_is_per_stage_paper_score(self):
        graph = PatchGraph((stage("a", "gpu", 0, availability=0.9), stage("b", "gpu", 1, availability=0.9)),
                           (PatchLink("a", "b"),), 2, {"gpu": 2})
        result = PatchExactSolver().solve(graph, PatchQuery(1, 1, min_reliability=0.8))
        self.assertAlmostEqual(result.reliability, 0.81)
        with self.assertRaises(NoFeasiblePlacement):
            PatchExactSolver().solve(graph, PatchQuery(1, 1, min_reliability=0.85))

    def test_zero_availability_is_allowed_only_without_positive_threshold(self):
        graph = PatchGraph((stage("offline", "gpu", 0, availability=0),), (), 1, {"gpu": 1})
        self.assertEqual(PatchExactSolver().solve(graph, PatchQuery(1, 1)).reliability, 0)
        with self.assertRaises(NoFeasiblePlacement):
            PatchExactSolver().solve(graph, PatchQuery(1, 1, min_reliability=0.01))

    def test_grouped_layer_alternative_and_missing_links(self):
        graph = PatchGraph(
            (stage("whole", "a", 0, end=2, pre=6, memory=3),
             stage("first", "b", 0), stage("rest", "a", 1, end=2, pre=2, memory=2)),
            (PatchLink("first", "rest"),), 3, {"a": 3, "b": 1},
        )
        self.assertEqual(PatchExactSolver().solve(graph, PatchQuery(1, 1)).path, ("first", "rest"))
        with self.assertRaises(NoFeasiblePlacement):
            evaluate_path(graph, PatchQuery(1, 1), ("first", "whole"))
        with self.assertRaises(NoFeasiblePlacement):
            evaluate_path(graph, PatchQuery(1, 1), ("rest",))

    def test_input_capacities_are_snapshotted(self):
        capacity = {"a": 1}
        graph = PatchGraph((stage("x", "a", 0),), (), 1, capacity)
        capacity["a"] = 0
        self.assertEqual(PatchExactSolver().solve(graph, PatchQuery(1, 1)).assignments, ("a",))

    def test_explicit_normalization_matches_formula(self):
        graph = PatchGraph((stage("x", "a", 0, pre=4, cost=8),), (), 1, {"a": 1})
        normal = PatchNormalization(1, 10, 2, 20, epsilon=1e-9)
        result = PatchExactSolver().solve(graph, PatchQuery(0.3, 1, normalization=normal))
        self.assertAlmostEqual(result.objective, 0.3 * 3 / (9 + 1e-9) + 0.7 * 6 / (18 + 1e-9))

    def test_constant_latency_normalization_does_not_hide_cost_difference(self):
        graph = PatchGraph((stage("a-costly", "gpu", 0, pre=1e6, cost=1.000001),
                            stage("z-cheap", "gpu", 0, pre=1e6, cost=1.0)), (), 1, {"gpu": 1})
        result = PatchExactSolver().solve(graph, PatchQuery(0.5, 1))
        self.assertEqual(result.path, ("z-cheap",))
        self.assertEqual(result.objective, 0)

    def test_budget_exhaustion_is_not_infeasibility(self):
        graph = PatchGraph((stage("a", "x", 0), stage("b", "x", 1)),
                           (PatchLink("a", "b"),), 2, {"x": 2})
        with self.assertRaises(SearchLimitExceeded):
            PatchExactSolver(max_labels=1).solve(graph, PatchQuery(1, 1))

    def test_malformed_inputs(self):
        a, b = stage("a", "gpu", 0), stage("b", "gpu", 1)
        for links in ((PatchLink("a", "missing"),), (PatchLink("b", "a"),),
                      (PatchLink("a", "b"), PatchLink("a", "b"))):
            with self.assertRaises(ValueError):
                PatchGraph((a, b), links, 2, {"gpu": 2})
        with self.assertRaises(ValueError):
            PatchGraph((a, a), (), 1, {"gpu": 2})
        with self.assertRaises(ValueError):
            replace(a, memory_pages=True)
        with self.assertRaises(ValueError):
            replace(a, prefill_seconds=math.nan)
        for preference in (-0.1, 1.1, True, math.inf):
            with self.assertRaises(ValueError):
                PatchQuery(preference, 1)
        with self.assertRaises(NoFeasiblePlacement):
            PatchExactSolver().solve(PatchGraph((), (), 1, {}), PatchQuery(1, 1))

    def test_random_small_graphs_against_independent_exhaustive_oracle(self):
        rng = random.Random(20260919)
        feasible_cases = infeasible_cases = 0
        for case in range(180):
            count, num_devices = rng.randint(2, 5), rng.randint(2, 3)
            devices = tuple(f"gpu-{i}" for i in range(num_devices))
            candidates = []
            for layer in range(count):
                for device in devices:
                    candidates.append(stage(
                        f"{count - layer}-{device}", device, layer,
                        pre=rng.randint(1, 9), dec=rng.randint(1, 5), memory=rng.randint(1, 3),
                        cost=rng.randint(0, 9), availability=rng.choice((0.8, 0.9, 1.0)),
                    ))
            links = [PatchLink(a.stage_id, b.stage_id, rng.randint(0, 3), rng.randint(0, 4))
                     for a in candidates for b in candidates
                     if a.end_layer + 1 == b.start_layer and rng.random() > 0.15]
            rng.shuffle(candidates)
            rng.shuffle(links)
            graph = PatchGraph(tuple(candidates), tuple(links), count,
                               {key: rng.randint(3, 8) for key in devices})
            query = PatchQuery(rng.choice((0.0, 0.2, 0.5, 0.8, 1.0)), rng.randint(2, 5),
                               min_reliability=rng.choice((0.0, 0.5, 0.85, 1.0)))
            oracle = exhaustive_oracle(graph, query)
            with self.subTest(case=case, feasible=bool(oracle)):
                if not oracle:
                    infeasible_cases += 1
                    with self.assertRaises(NoFeasiblePlacement):
                        PatchExactSolver().solve(graph, query)
                    continue
                feasible_cases += 1
                result = PatchExactSolver().solve(graph, query)
                independent = next(row for row in oracle if row["path"] == result.path)
                self.assertAlmostEqual(result.objective, oracle[0]["objective"], places=12)
                self.assertAlmostEqual(result.ttft_seconds, independent["ttft"])
                self.assertAlmostEqual(result.monetary_cost, independent["cost"])
                self.assertAlmostEqual(result.reliability, independent["reliability"])
                self.assertEqual(result.tpot_seconds, independent["tpot"])
                self.assertEqual(dict(result.memory_by_device), independent["memory"])
        self.assertGreater(feasible_cases, 25)
        self.assertGreater(infeasible_cases, 25)


if __name__ == "__main__":
    unittest.main()
