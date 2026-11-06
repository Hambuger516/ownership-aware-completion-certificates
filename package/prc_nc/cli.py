"""Safe entry points: validation/planning never load CUDA or download a model."""

from __future__ import annotations
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time
from .configuration import PROJECT, validate_config
from .matrices import matrix


def doctor(gpu=False):
    result = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "core_ready": sys.version_info >= (3, 12),
        "gpu_executed": False,
        "gpu_ready": False,
        "missing_or_mismatched": [],
        "gpu_ready_scope": "software versions only; driver, kernels and VRAM not tested",
    }
    if gpu:
        if sys.platform != "linux":
            result["missing_or_mismatched"].append("GPU adapter requires Linux")
        for package, expected in (("vllm", "0.29.0"), ("torch", "2.13.0+cu130"), ("transformers", "5.17.0")):
            try:
                # importlib.metadata drops the local CUDA build suffix from
                # torch (reports 2.13.0 for 2.13.0+cu130). Read torch's own
                # runtime version so the locked CUDA build is verified.
                if package == "torch":
                    import torch

                    actual = torch.__version__
                else:
                    actual = importlib.metadata.version(package)
            except (ImportError, importlib.metadata.PackageNotFoundError):
                actual = None
            result[package] = actual
            if actual != expected:
                result["missing_or_mismatched"].append(f"{package}: expected {expected}, got {actual}")
        result["gpu_ready"] = not result["missing_or_mismatched"]
    return result


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def make_plan(suite, folder, model_path="", output_root=None, predictor_checkpoint="", patch_checkpoint=""):
    configs = matrix(suite)
    order_seed = 20260919 if suite == "effect" else None
    if order_seed is not None:
        # Randomize within each (scenario, repetition) block, then block order.
        # All methods retain identical inputs, contracts, engine seed and arrivals.
        blocks = {}
        for config in configs:
            blocks.setdefault((config["case"], config["replicate"]), []).append(config)
        rng = random.Random(order_seed)
        block_order = list(blocks)
        rng.shuffle(block_order)
        configs = []
        for key in block_order:
            rng.shuffle(blocks[key])
            configs.extend(blocks[key])
    folder.mkdir(parents=True, exist_ok=False)
    for c in configs:
        c["model_path"] = str(model_path)
        c["predictor_checkpoint"] = str(predictor_checkpoint)
        c["patch_checkpoint"] = str(patch_checkpoint)
        c["output_root"] = str((output_root or folder / "results").resolve())
        c["data_path"] = ""  # Resolve bundled data on the target host, not this host.
        save_new(folder / (c["run_id"] + ".json"), c)
    manifest = {
        "suite": suite,
        "status": "PLANNED_NOT_RUN",
        "runs": len(configs),
        "execution_order_seed": order_seed,
        "execution_order_design": "randomized scenario/replicate blocks and method order" if order_seed else "declared",
        "configurations": [c["run_id"] + ".json" for c in configs],
        "scope": "Protocol core uses CPU tests; these are single-GPU configurations.",
    }
    save_new(folder / "plan.json", manifest)
    return manifest


def run_plan(folder):
    """Bounded serial GPU jobs, process-group cleanup, stop on first bad verdict."""
    if not doctor(True)["gpu_ready"]:
        raise RuntimeError("GPU environment not ready; run doctor --gpu")
    plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    if not isinstance(plan.get("configurations"), list) or not plan["configurations"]:
        raise ValueError("plan must contain at least one configuration")
    configs = []
    for name in plan["configurations"]:
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("unsafe configuration path")
        configs.append(validate_config(json.loads((folder / name).read_text(encoding="utf-8")), require_paths=True))
    if len({(c["output_root"], c["run_id"]) for c in configs}) != len(configs):
        raise ValueError("duplicate run destinations")
    execution = folder / "execution"
    execution.mkdir(exist_ok=False)
    state = {"status": "RUNNING", "results": []}

    def persist():
        temp = execution / "status.tmp"
        temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp.replace(execution / "status.json")

    persist()
    from .analysis import summarize

    try:
        for c in configs:
            config = execution / (c["run_id"] + ".json")
            save_new(config, c)
            log = execution / (c["run_id"] + ".log")
            started = time.monotonic()
            timeout = False
            with log.open("x", encoding="utf-8") as f:
                proc = subprocess.Popen(
                    [sys.executable, "-u", "-m", "prc_nc.gpu_run", str(config.resolve()), "--execute-gpu"],
                    cwd=PROJECT,
                    stdin=subprocess.DEVNULL,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = proc.wait(timeout=c["timeout_seconds"])
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass  # Child may have just exited.
                    try:
                        proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait()
                    if isinstance(exc, KeyboardInterrupt):
                        raise
                    timeout = True
                    code = proc.returncode
            output = Path(c["output_root"]) / c["run_id"]
            passed = not timeout and code == 0
            if c["fault"]:
                eventfile = output / "allocator_events.jsonl"
                events = [json.loads(l) for l in eventfile.read_text().splitlines()] if eventfile.exists() else []
                injected = any(e["event"] == "fault_injection" and e["kind"] == c["fault"] for e in events)
                passed = bool(code and not timeout and injected and c["expected_error"] in log.read_text())
                if output.exists():
                    save_new(
                        output / "expected_failure_verdict.json",
                        dict(passed=passed, fault_injected=injected, timeout=timeout),
                    )
            if passed:
                summary, _, _ = summarize(output)
                passed = summary["status"] in ("PASS", "EXPECTED_REJECTION_PASS")
                if c["case"] == "D1_drain":
                    passed = passed and summary["drain_episodes"] > 0
                save_new(output / "replay_summary.json", summary)
            state["results"].append(
                dict(
                    run_id=c["run_id"],
                    passed=passed,
                    exit_code=code,
                    timeout=timeout,
                    wall_seconds=time.monotonic() - started,
                )
            )
            persist()
            if not passed:
                raise RuntimeError("experiment failed: " + c["run_id"])
        state["status"] = "PASS"
    except BaseException as exc:
        state["status"] = "STOPPED_ON_FAILURE"
        state["error"] = str(exc)
        persist()
        raise
    finally:
        persist()
    return state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--gpu", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("config", type=Path)
    p = sub.add_parser("plan")
    p.add_argument("--suite", choices=["correctness", "drain", "performance", "proposal", "effect", "full"], default="correctness")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-path", default="")
    p.add_argument("--predictor-checkpoint", default="")
    p.add_argument("--patch-checkpoint", default="")
    p.add_argument("--output-root", type=Path)
    r = sub.add_parser("run")
    r.add_argument("config", type=Path)
    r.add_argument("--execute-gpu", action="store_true")
    s = sub.add_parser("suite")
    s.add_argument("plan", type=Path)
    s.add_argument("--execute-gpu", action="store_true")
    sub.add_parser("check")
    args = ap.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor(args.gpu)
        elif args.command in ("validate", "run"):
            c = validate_config(json.loads(args.config.read_text(encoding="utf-8")))
            if args.command == "run" and args.execute_gpu:
                if not doctor(True)["gpu_ready"]:
                    raise RuntimeError("GPU environment not ready; run doctor --gpu")
                # Single-run entry uses the same external timeout and process
                # cleanup as suites; observation_limit cannot interrupt a hung step.
                c = validate_config(c, require_paths=True)
                c["output_root"] = str(Path(c["output_root"]).resolve())
                c["model_path"] = str(Path(c["model_path"]).resolve())
                control = Path(c["output_root"]) / (c["run_id"] + "_control")
                control.mkdir(parents=True, exist_ok=False)
                save_new(control / "config.json", c)
                save_new(
                    control / "plan.json",
                    dict(suite="single", status="PLANNED_NOT_RUN", runs=1, configurations=["config.json"]),
                )
                result = run_plan(control)
                print(json.dumps(result, indent=2))
                return 0
            result = {"status": "VALID_CONFIG_NOT_RUN", "configuration": c}
        elif args.command == "plan":
            result = make_plan(args.suite, args.output, args.model_path, args.output_root,
                               args.predictor_checkpoint, args.patch_checkpoint)
        elif args.command == "suite":
            if not args.execute_gpu:
                raise ValueError("suite requires --execute-gpu; plan alone does not run anything")
            result = run_plan(args.plan.resolve())
        else:
            import unittest

            paths = list((PROJECT / "prc_nc").rglob("*.py")) + list((PROJECT / "tests").rglob("*.py"))
            for path in paths:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            suite = unittest.defaultTestLoader.discover(str(PROJECT / "tests"))
            if suite.countTestCases() == 0:
                raise RuntimeError("test suite not installed")
            run = unittest.TextTestRunner(verbosity=2).run(suite)
            return 0 if run.wasSuccessful() else 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
