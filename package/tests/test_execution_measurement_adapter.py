"""Execute observer methods with a fake engine, not GPU performance evidence."""
import ast
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from prc_nc.execution_meter import ExecutionTokenMeter


class BaseScheduler:
    def __init__(self):
        self.scheduler_config = NS(async_scheduling=False)
        self.parallel_config = NS(tensor_parallel_size=1, pipeline_parallel_size=1)
        self.num_lookahead_tokens = 0
        self.connector = None
        self.requests = {"w0-r0": NS(num_computed_tokens=0)}
        self.calls = 0

    def schedule(self):
        self.calls += 1
        return "unchanged schedule result"

    def _update_after_schedule(self, output):
        for rid, count in output.num_scheduled_tokens.items():
            self.requests[rid].num_computed_tokens += count

    def update_from_output(self, *args):
        return "unchanged execution result"

    def has_requests(self):
        return False

    def reset_prefix_cache(self):
        return True


class ObserverAdapterTests(unittest.TestCase):
    def test_observer_preserves_results_and_excludes_warmup_timing_once(self):
        source = Path(__file__).resolve().parents[1] / "prc_nc/adapters/execution_measurement.py"
        classes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef)]
        ns = dict(atexit=NS(register=lambda callback: None), json=json, os=os, time=time,
                  Path=Path, vllm=NS(__version__="0.29.0"), Scheduler=BaseScheduler,
                  ExecutionTokenMeter=ExecutionTokenMeter)
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), ns)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, PRC_MEASURE_EXECUTION="1", PRC_RUN_DIR=tmp):
            scheduler = ns["MeasuredNativeScheduler"]()
            self.assertEqual(scheduler.schedule(), "unchanged schedule result")
            self.assertEqual(scheduler.calls, 1)
            scheduler._flush_execution_meter()
            before = json.loads((Path(tmp)/"execution_tokens.json").read_text())
            self.assertEqual(before["timing_status"], "unavailable")
            self.assertIsNone(before["schedule_seconds"])
            self.assertFalse(before["timing_boundary_started"])
            self.assertTrue(scheduler.reset_prefix_cache())
            self.assertEqual(scheduler._execution_schedule_ns, 0)
            empty = json.loads((Path(tmp)/"execution_tokens.json").read_text())
            self.assertTrue(empty["timing_boundary_started"])
            self.assertEqual(empty["timing_status"], "unavailable")
            self.assertIsNone(empty["scheduler_lifecycle_seconds"])
            scheduler.schedule()
            measured = scheduler._execution_schedule_ns
            out = NS(num_scheduled_tokens={"w0-r0": 4})
            scheduler._update_after_schedule(out)
            self.assertEqual(scheduler._execution_meter.report()["executed_tokens"], 0)
            self.assertEqual(scheduler.update_from_output(out, NS(req_ids=["w0-r0"])), "unchanged execution result")
            scheduler.reset_prefix_cache()
            self.assertEqual(scheduler._execution_schedule_ns, measured)
            report = json.loads((Path(tmp)/"execution_tokens.json").read_text())
            self.assertEqual(report["executed_tokens"], 4)
            self.assertEqual(report["timing_status"], "measured")
            self.assertEqual(report["measured_schedule_calls"], 1)
            self.assertEqual(report["measured_update_calls"], 1)
            self.assertEqual(report["pending_batches"], 0)
            self.assertAlmostEqual(report["scheduler_lifecycle_seconds"], report["schedule_seconds"]+report["update_seconds"])
