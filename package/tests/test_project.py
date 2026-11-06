import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from prc_nc.configuration import PROJECT, validate_config
from prc_nc.matrices import matrix
from prc_nc.cli import main, make_plan


class ConfigurationTests(unittest.TestCase):
    def test_all_declared_matrices_validate(self):
        for name, count in (("correctness", 22), ("drain", 4), ("performance", 12), ("proposal", 2), ("effect", 45), ("full", 40)):
            configs = matrix(name)
            self.assertEqual(len(configs), count)
            self.assertEqual(len({c["run_id"] for c in configs}), count)
            for config in configs:
                self.assertEqual(validate_config(config), config)

    def test_unknown_and_malformed_fields_fail_early(self):
        for config in (
            {"typo": 1},
            {"blocks": 64},
            {"audit": 1},
            {"method": "fake"},
            {"run_id": "../../escape"},
            {"count": True},
            {"arrival_interval": float("nan")},
            {"block_size": 8},
            {"min_prompt": 100, "max_prompt": 32},
            {"fault": "bypass_guard"},
            {"cancellation": True, "count": 1},
        ):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_missing_model_blocks_execution_not_planning(self):
        validate_config({})
        with self.assertRaises(ValueError):
            validate_config({}, require_paths=True)

    def test_destination_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
                (model / name).write_text("{}")
            (root / "already").mkdir()
            with self.assertRaises(FileExistsError):
                validate_config(dict(model_path=str(model), output_root=temp, run_id="already"), require_paths=True)

    def test_bundled_dataset_schema(self):
        data = Path(validate_config({})["data_path"])
        rows = json.loads(data.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 512)
        self.assertTrue(all({"instruction", "input", "source_index"} <= set(r) for r in rows))


class EntryPointTests(unittest.TestCase):
    def call(self, args):
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(args)
        return code, out.getvalue(), err.getvalue()

    def test_cpu_doctor(self):
        code, out, _ = self.call(["doctor"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["core_ready"])
        self.assertFalse(json.loads(out)["gpu_executed"])

    def test_gpu_doctor_imports_no_engine(self):
        with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
            code, out, _ = self.call(["doctor", "--gpu"])
        self.assertEqual(code, 0)
        self.assertFalse(json.loads(out)["gpu_ready"])

    def test_plan_prepares_no_subprocess(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch("subprocess.Popen", side_effect=AssertionError("unexpected execution")),
        ):
            folder = Path(temp) / "plan"
            result = make_plan("full", folder)
            self.assertEqual(result["runs"], 40)
            self.assertEqual(result["status"], "PLANNED_NOT_RUN")
            self.assertEqual(len(list(folder.glob("*.json"))), 41)
            with self.assertRaises(FileExistsError):
                make_plan("full", folder)

    def test_effect_randomization_is_reproducible_and_block_matched(self):
        with tempfile.TemporaryDirectory() as temp:
            a = make_plan("effect", Path(temp)/"a")
            b = make_plan("effect", Path(temp)/"b")
            self.assertEqual(a["configurations"], b["configurations"])
            self.assertEqual(a["execution_order_seed"], 20260919)
            configs = [json.loads((Path(temp)/"a"/name).read_text()) for name in a["configurations"]]
            self.assertNotEqual([x["run_id"] for x in configs], [x["run_id"] for x in matrix("effect")])
            for offset in range(0,45,3):
                group = configs[offset:offset+3]
                self.assertEqual({x["method"] for x in group}, {"prc", "native_plain", "hard_reserve"})
                for key in ("case","replicate","cap_profile","count","seed","arrival_interval","blocks"):
                    self.assertEqual(len({x[key] for x in group}),1)

    def test_plan_carries_strict_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            result = make_plan("proposal", Path(temp) / "plan", predictor_checkpoint="pred.json",
                               patch_checkpoint="patch.pt")
            self.assertEqual(result["runs"], 2)
            configs = [json.loads(p.read_text()) for p in (Path(temp) / "plan").glob("*.json") if p.name != "plan.json"]
            self.assertEqual({c["patch_checkpoint"] for c in configs}, {"patch.pt"})
            self.assertEqual({c["predictor_checkpoint"] for c in configs}, {"pred.json"})

    def test_run_without_opt_in_only_validates(self):
        with patch("subprocess.Popen", side_effect=AssertionError("unexpected execution")):
            code, out, _ = self.call(["run", str(PROJECT / "configs/smoke.json")])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["status"], "VALID_CONFIG_NOT_RUN")

    def test_gpu_missing_environment_is_actionable(self):
        with patch("prc_nc.cli.doctor", return_value={"gpu_ready": False}):
            code, _, err = self.call(["run", str(PROJECT / "configs/smoke.json"), "--execute-gpu"])
        self.assertEqual(code, 2)
        self.assertIn("environment not ready", err)

    def test_suite_requires_opt_in(self):
        code, _, err = self.call(["suite", "nonexistent"])
        self.assertEqual(code, 2)
        self.assertIn("--execute-gpu", err)

    def test_module_help_entrypoint(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "prc_nc", "--help"], cwd=PROJECT, capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("doctor", result.stdout)

    def test_gpu_driver_import_is_lazy(self):
        script = (
            "import sys; import prc_nc.gpu_run; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=PROJECT, capture_output=True, text=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)
