"""Log fixtures and process mocks are NOT GPU execution results."""

import json
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from prc_nc.analysis import summarize, paired_outputs
from prc_nc.cli import run_plan
from prc_nc.configuration import PROJECT, validate_config
from prc_nc.diagnostics import boundary_report
from prc_nc.model import State, Group


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class BoundaryTests(unittest.TestCase):
    def test_topology_pair(self):
        def state(pairs):
            return State(
                (1,),
                ((1,), (1,), (3,), (3,)),
                tuple(Group(str(i), 0, 2, frozenset(pair)) for i, pair in enumerate(pairs)),
            )

        yes = boundary_report(state(((0, 1), (2, 3))))
        no = boundary_report(state(((0, 2), (1, 3))))
        self.assertTrue(yes["full_feasibility_gain"])
        self.assertFalse(no["closure_gain"])
        self.assertFalse(yes["bridge_holds"])
        self.assertFalse(no["bridge_holds"])  # Non-bridge does not imply a gain.

    def test_empty_is_not_nonempty_bridge_claim(self):
        self.assertIsNone(boundary_report(State((0,), (), ()))["bridge_holds"])

    def test_random_bridge_across_pools(self):
        rng = random.Random(73012)
        for _ in range(500):
            n, p = rng.randrange(1, 7), rng.randrange(1, 4)
            bound = [rng.randrange(1, 8) for _ in range(p)]
            need = tuple(tuple(rng.randrange(b + 1) for b in bound) for _ in range(n))
            groups = []
            for i in range(n):
                for q in range(p):
                    pages = bound[q] - need[i][q] + rng.randrange(2)
                    if pages:
                        groups.append(Group(f"{i}-{q}", q, pages, frozenset({i})))
            groups.append(Group("shared", 0, 2, frozenset(range(n))))
            report = boundary_report(State(tuple(rng.randrange(b + 1) for b in bound), need, tuple(groups)))
            self.assertTrue(report["bridge_holds"])
            self.assertEqual(report["conservative_closure"], report["exact_closure"])


class PairingTests(unittest.TestCase):
    def test_reference_matches_replicate_and_reports_length_difference(self):
        rows = [dict(run_id=f"{method}{rep}", case="case", replicate=rep,
                     method=method, status="PASS", prompt_token_sha256="prompt", caps_sha256="caps")
                for rep in (0, 1) for method in ("native_plain", "prc")]
        records = {"native_plain0": [dict(request_id="r", tokens=[1])],
                   "prc0": [dict(request_id="r", tokens=[1, 2])],
                   "native_plain1": [dict(request_id="r", tokens=[3])],
                   "prc1": [dict(request_id="r", tokens=[3])]}
        pairs = {p["run_id"]: p for p in paired_outputs(rows, records)}
        self.assertEqual(pairs["prc1"]["reference"], "native_plain1")
        self.assertEqual(pairs["prc1"]["exact_output_matches"], 1)
        self.assertEqual(pairs["prc0"]["output_length_matches"], 0)
        self.assertEqual(pairs["prc0"]["output_length_delta_tokens"], 1)


class ReplayTests(unittest.TestCase):
    def test_fail_before_engine_initialization_preserves_failure_without_capacity(self):
        for method in ("native_plain", "shared_incremental"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                c = validate_config(dict(method=method))
                save(folder / "manifest.json", dict(status="FAIL", configuration=c,
                     source_sha256={}, error="original pre-engine failure"))
                report, _, _ = summarize(folder)
                self.assertEqual(report["status"], "FAIL")
                self.assertEqual(json.loads((folder / "manifest.json").read_text())["error"],
                                 "original pre-engine failure")

    def test_pass_without_capacity_is_rejected_even_without_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            save(folder / "manifest.json", dict(status="PASS", configuration=validate_config({}), source_sha256={}))
            with self.assertRaisesRegex(AssertionError, "measured physical capacity"):
                summarize(folder)

    def fixture(self, folder):
        c = validate_config({"count": 1, "waves": 1})
        save(
            folder / "manifest.json", dict(status="PASS", configuration=c, source_sha256={}, physical_capacity_pages=2)
        )
        save(
            folder / "requests.json",
            [dict(request_id="w0-r0", terminal="completed", tokens=[1], cap=2, submitted_at_s=0, offered_at_s=0)],
        )
        events = [
            dict(
                event="incremental_proposal",
                request_id="w0-r0",
                step=1,
                fixed_accepted=True,
                oracle_checked=False,
                equivalent=None,
                complete_fallback=False,
                structural_rebuild=False,
            ),
            dict(
                event="proposal",
                request_id="w0-r0",
                step=1,
                new_pages=1,
                reused_blocks=0,
                touched_free=0,
                controller_ns=10,
            ),
            dict(event="allocation", request_id="w0-r0", step=1, projection_verified=True, certificate=True),
            dict(
                event="release",
                request_id="w0-r0",
                step=2,
                all_pages_reclaimable=True,
                certificate=True,
                request_status="FINISHED",
                logical_pages=1,
            ),
            dict(event="drain_start", step=1),
            dict(
                event="snapshot",
                cohort=["w0-r0"],
                free=1,
                hard_need=[1],
                step=1,
                groups=[dict(block="a", owners=["w0-r0"])],
                certificate=True,
            ),
        ]
        return events

    def write_events(self, folder, events):
        (folder / "allocator_events.jsonl").write_text("\n".join(map(json.dumps, events)), encoding="utf-8")

    def test_non_audit_is_not_counted_as_oracle_equality(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write_events(folder, self.fixture(folder))
            report, _, _ = summarize(folder)
            self.assertEqual(report["incremental_oracle_comparisons"], 0)
            self.assertEqual(report["drain_episodes"], 1)
            self.assertEqual(report["bridge_condition_snapshots"], 1)
            self.assertIsNone(report["all_prefill_complete_snapshots"])

    def test_unmeasured_equivalence_cannot_be_reported_true(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            events = self.fixture(folder)
            events[0]["equivalent"] = True
            self.write_events(folder, events)
            with self.assertRaises(AssertionError):
                summarize(folder)

    def test_proposal_binding_uses_explicit_external_identity(self):
        # Fixture only: exercise the real adapter's internal/external ID shape.
        for external_id in ("w0-r0", "w0-r9"):
            with self.subTest(external_id=external_id), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                events = self.fixture(folder)
                manifest = json.loads((folder / "manifest.json").read_text())
                manifest["configuration"]["method"] = "prc"
                manifest["configuration"]["proposal_mode"] = "exact"
                manifest["proposal_count"] = 1
                save(folder / "manifest.json", manifest)
                save(folder / "proposals.json", [dict(
                    request_id="w0-r0", action_id="w0-r0:preflight", worker_id="gpu0",
                    pool_id="kv0", state_version=0, hard_max_output_tokens=2,
                    hard_need_pages=1, soft_reserve_tokens=1, route=dict(device_id="gpu0"),
                    placement=dict(devices=["gpu0"], solver="PATCH-Exact"),
                    predictor_checkpoint_sha256="fixture", predictor_split_sha256="fixture")])
                events += [dict(event="proposal_binding", step=1,
                                request_id="w0-r0-engine-suffix", external_request_id=external_id),
                           dict(event="causal_authorize", step=1, request_id="w0-r0-engine-suffix")]
                self.write_events(folder, events)
                if external_id == "w0-r0":
                    report, _, _ = summarize(folder)
                    self.assertEqual(report["proposal_bindings"], 1)
                else:
                    with self.assertRaisesRegex(AssertionError, "unknown external request"):
                        summarize(folder)

    def test_only_verified_unexecuted_cancellation_may_lack_binding(self):
        for has_tokens in (False, True):
            with self.subTest(has_tokens=has_tokens), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                events = self.fixture(folder)
                manifest = json.loads((folder / "manifest.json").read_text())
                manifest["proposal_count"] = 1
                save(folder / "manifest.json", manifest)
                save(folder / "proposals.json", [dict(
                    request_id="w0-r0", action_id="preflight", worker_id="gpu0", pool_id="kv0",
                    state_version=0, hard_max_output_tokens=2, hard_need_pages=1,
                    soft_reserve_tokens=1, route=dict(device_id="gpu0"),
                    placement=dict(devices=["gpu0"]), predictor_checkpoint_sha256="fixture",
                    predictor_split_sha256="fixture")])
                records = json.loads((folder / "requests.json").read_text())
                records[0].update(terminal="cancelled", tokens=[1] if has_tokens else [])
                save(folder / "requests.json", records)
                save(folder / "lifecycle.json", [dict(event="waiting_abort_twice", request_id="w0-r0")])
                events = [e for e in events if e["event"] not in ("allocation", "proposal", "incremental_proposal")]
                for e in events:
                    if e["event"] == "release":
                        e.update(logical_pages=0, request_status="FINISHED_ABORTED")
                events.append(dict(event="causal_finish", step=2, request_id="w0-r0"))
                self.write_events(folder, events)
                if has_tokens:
                    with self.assertRaisesRegex(AssertionError, "bypassed proposal binding"):
                        summarize(folder)
                else:
                    report, _, _ = summarize(folder)
                    self.assertEqual(report["status"], "PASS")

    def test_statistical_deferral_requires_binding_and_no_allocation(self):
        for allowed in (False, True):
            with self.subTest(allowed=allowed), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                events = self.fixture(folder)
                events += [dict(event="proposal_binding", step=3, request_id="w0-r1", proposal_allowed=allowed),
                           dict(event="incremental_proposal", step=3, request_id="w0-r1",
                                fixed_accepted=False, oracle_checked=False, equivalent=None,
                                statistical_defer=True, complete_fallback=False, structural_rebuild=True),
                           dict(event="proposal", step=3, request_id="w0-r1", allow=False,
                                new_pages=1, reused_blocks=0, touched_free=0, controller_ns=10)]
                self.write_events(folder, events)
                if allowed:
                    with self.assertRaisesRegex(AssertionError, "verified proposal deferral"):
                        summarize(folder)
                else:
                    report, _, _ = summarize(folder)
                    self.assertEqual(report["status"], "PASS")

    def test_execution_evidence_requires_confirmation_and_request_coverage(self):
        for defect in (None, "pending", "missing_request"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                self.write_events(folder, self.fixture(folder))
                manifest = json.loads((folder / "manifest.json").read_text())
                manifest["execution_measurement"] = True
                save(folder / "manifest.json", manifest)
                rid = "w0-r9-suffix" if defect == "missing_request" else "w0-r0-suffix"
                save(folder / "execution_tokens.json", dict(
                    pending_batches=int(defect == "pending"), executed_tokens=4, reexecuted_tokens=1,
                    requests={rid: dict(executed_tokens=4, reexecuted_tokens=1)}))
                if defect:
                    with self.assertRaises(AssertionError):
                        summarize(folder)
                else:
                    report, _, _ = summarize(folder)
                    self.assertEqual(report["execution_measurement"]["reexecuted_tokens"], 1)

    def test_exclusive_closure_replays_pinned_certificate(self):
        """C0 logs the conservative certificate, not the exact C1 oracle."""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = validate_config({"method": "exclusive_closure", "count": 4, "run_id": "exclusive"})
            save(
                folder / "manifest.json",
                dict(status="PASS", configuration=config, source_sha256={}, physical_capacity_pages=5),
            )
            save(
                folder / "requests.json",
                [
                    dict(request_id=f"w0-r{i}", terminal="completed", tokens=[1], cap=3, submitted_at_s=0, offered_at_s=0)
                    for i in range(4)
                ],
            )
            events = [
                dict(
                    event="release",
                    request_id="w0-r0",
                    step=2,
                    all_pages_reclaimable=True,
                    certificate=True,
                    request_status="FINISHED",
                    logical_pages=1,
                ),
                dict(
                    event="snapshot",
                    cohort=[f"w0-r{i}" for i in range(4)],
                    free=1,
                    hard_need=[1, 1, 3, 3],
                    step=1,
                    groups=[
                        dict(block="a1", owners=["w0-r0", "w0-r1"]),
                        dict(block="a2", owners=["w0-r0", "w0-r1"]),
                        dict(block="b1", owners=["w0-r2", "w0-r3"]),
                        dict(block="b2", owners=["w0-r2", "w0-r3"]),
                    ],
                    # Exact C1 is feasible; conservative C0 is not.
                    certificate=False,
                ),
            ]
            self.write_events(folder, events)
            report, witnesses, _ = summarize(folder)
            self.assertEqual(report["method"], "exclusive_closure")
            self.assertEqual(report["ownership_useful_snapshots"], 1)
            self.assertEqual(len(witnesses), 1)

    def test_source_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write_events(folder, self.fixture(folder))
            manifest = json.loads((folder / "manifest.json").read_text())
            manifest["source_sha256"] = {"x.py": "invalid"}
            save(folder / "manifest.json", manifest)
            (folder / "source").mkdir()
            (folder / "source/x.py").write_text("pass")
            with self.assertRaisesRegex(AssertionError, "hash mismatch"):
                summarize(folder)

    def test_optimized_python_cannot_silently_disable_replay(self):
        script = "from pathlib import Path; from prc_nc.analysis import summarize; summarize(Path('missing'))"
        result = subprocess.run(
            [sys.executable, "-O", "-c", script], cwd=PROJECT, capture_output=True, text=True, timeout=10
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires assertions", result.stderr)


class RunnerTests(unittest.TestCase):
    def test_expected_fault_needs_injection_and_matching_error(self):
        from prc_nc.configuration import FAULTS

        for injected in (True, False):
            with self.subTest(injected=injected), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.prepare(root)
                c = json.loads((root / "one.json").read_text())
                c.update(fault="bypass_guard", expected_error=FAULTS["bypass_guard"])
                save(root / "one.json", c)

                def fake_launch(*args, **kwargs):
                    kwargs["stdout"].write(FAULTS["bypass_guard"])
                    out = root / "results/one"
                    out.mkdir(parents=True)
                    event = dict(event="fault_injection", kind="bypass_guard") if injected else dict(event="unrelated")
                    (out / "allocator_events.jsonl").write_text(json.dumps(event))
                    proc = Mock()
                    proc.wait.return_value = 1
                    return proc

                with (
                    patch("prc_nc.cli.doctor", return_value={"gpu_ready": True}),
                    patch("prc_nc.cli.subprocess.Popen", side_effect=fake_launch),
                    patch("prc_nc.analysis.summarize", return_value=({"status": "EXPECTED_REJECTION_PASS"}, [], [])),
                ):
                    if injected:
                        self.assertEqual(run_plan(root)["status"], "PASS")
                    else:
                        with self.assertRaises(RuntimeError):
                            run_plan(root)

    def prepare(self, root):
        model = root / "model"
        model.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
            (model / name).write_text("{}")
        save(root / "one.json", dict(model_path=str(model), output_root=str(root / "results"), run_id="one"))
        save(root / "plan.json", dict(configurations=["one.json"]))

    def test_mock_success_saves_replay_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root)
            proc = Mock()
            proc.wait.return_value = 0

            def fake_replay(folder):
                folder.mkdir(parents=True)
                return {"status": "PASS"}, [], []

            with (
                patch("prc_nc.cli.doctor", return_value={"gpu_ready": True}),
                patch("prc_nc.cli.subprocess.Popen", return_value=proc),
                patch("prc_nc.analysis.summarize", side_effect=fake_replay),
            ):
                self.assertEqual(run_plan(root)["status"], "PASS")
            self.assertTrue((root / "results/one/replay_summary.json").exists())

    def test_mock_timeout_stops_own_child_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.prepare(root)
            proc = Mock(pid=1234, returncode=-15)
            proc.wait.side_effect = [subprocess.TimeoutExpired("mock", 300), -15]
            with (
                patch("prc_nc.cli.doctor", return_value={"gpu_ready": True}),
                patch("prc_nc.cli.subprocess.Popen", return_value=proc),
                patch("prc_nc.cli.os.killpg", create=True) as stop,
            ):
                with self.assertRaisesRegex(RuntimeError, "experiment failed"):
                    run_plan(root)
            stop.assert_called_once_with(1234, signal.SIGTERM)
            status = json.loads((root / "execution/status.json").read_text())
            self.assertEqual(status["status"], "STOPPED_ON_FAILURE")
            self.assertTrue(status["results"][0]["timeout"])

    def test_empty_plan_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch("prc_nc.cli.doctor", return_value={"gpu_ready": True}):
            save(Path(tmp) / "plan.json", dict(configurations=[]))
            with self.assertRaises(ValueError):
                run_plan(Path(tmp))
