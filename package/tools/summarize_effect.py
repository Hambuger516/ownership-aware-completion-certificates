"""Require the frozen 45-run matrix before emitting a final effect summary."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prc_nc.analysis import summarize, paired_outputs

CASES = {"EFFECT_dolly_pressure", "EFFECT_dolly_roomy", "EFFECT_hetero_forced"}
METHODS = {"prc", "native_plain", "hard_reserve"}


def validate_matrix(rows, *, require_execution=False):
    expected = {(case, method, rep) for case in CASES for method in METHODS for rep in range(5)}
    keys = [(r["case"], r["method"], r["replicate"]) for r in rows]
    if len(keys) != 45 or set(keys) != expected or any(r["status"] != "PASS" for r in rows):
        raise ValueError("Final report requires exactly 45 unique successful frozen configurations")
    modes = {r.get("execution_measurement") is not None for r in rows}
    if len(modes) != 1 or (require_execution and modes != {True}):
        raise ValueError("Missing or mixed execution measurement across matrix")
    if modes == {True}:
        scopes = set()
        for row in rows:
            e = row["execution_measurement"]
            if e.get("pending_batches") != 0:
                raise ValueError("Unconfirmed execution remains")
            for key in ("executed_tokens", "reexecuted_tokens"):
                if type(e.get(key)) is not int or e[key] < 0:
                    raise ValueError("Invalid execution count")
            if e["reexecuted_tokens"] > e["executed_tokens"]:
                raise ValueError("Repeated execution exceeds total execution")
            for key in ("schedule_seconds", "update_seconds", "scheduler_lifecycle_seconds"):
                if type(e.get(key)) not in (int, float) or not math.isfinite(e[key]) or e[key] < 0:
                    raise ValueError("Missing or invalid lifecycle timing")
            if not math.isclose(e["scheduler_lifecycle_seconds"], e["schedule_seconds"]+e["update_seconds"], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Lifecycle timing components disagree")
            scopes.add((e.get("scope"), e.get("timing_scope")))
        if len(scopes) != 1 or any(not item for pair in scopes for item in pair):
            raise ValueError("Missing or mixed measurement scopes")
    blocks = defaultdict(list)
    for row in rows:
        blocks[row["case"], row["replicate"]].append(row)
    for key, block in blocks.items():
        configs = [{k: v for k, v in r["config"].items() if k not in ("method", "run_id")}
                   for r in block]
        if any(c != configs[0] for c in configs[1:]):
            raise ValueError(f"Unmatched configuration in {key}")
        for field in ("prompt_token_sha256", "caps_sha256"):
            if not all(r[field] for r in block) or len({r[field] for r in block}) != 1:
                raise ValueError(f"Unmatched or missing {field} in {key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-execution", action="store_true")
    args = parser.parse_args()
    rows, records, sources = [], {}, set()
    for folder in sorted(args.results.iterdir()):
        if not (folder / "manifest.json").exists():
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["configuration"].get("case") not in CASES:
            continue
        row, _, outputs = summarize(folder)
        rows.append(row)
        records[row["run_id"]] = outputs
        sources.add(hashlib.sha256(json.dumps(manifest["source_sha256"], sort_keys=True).encode()).hexdigest())
    validate_matrix(rows, require_execution=args.require_execution)
    if len(sources) != 1:
        raise ValueError("Runtime source snapshots differ across the frozen comparison")
    pairs = paired_outputs(rows, records)
    aggregates = []
    for case in sorted(CASES):
        for method in sorted(METHODS):
            selected = [r for r in rows if r["case"] == case and r["method"] == method]
            values = {}
            for metric in ("measured_seconds", "ttft_p95_s", "mean_tpot_p95_s",
                           "proposal_controller_s", "instrumented_schedule_s"):
                measured = [r[metric] for r in selected if r[metric] is not None]
                values[metric] = (dict(n=len(measured), mean=statistics.mean(measured),
                                      min=min(measured), max=max(measured),
                                      stdev=statistics.stdev(measured) if len(measured)>1 else None)
                                  if measured else None)
            for metric in ("executed_tokens", "reexecuted_tokens", "schedule_seconds",
                           "update_seconds", "scheduler_lifecycle_seconds"):
                measured = [r["execution_measurement"][metric] for r in selected if r.get("execution_measurement")]
                values[metric] = (dict(n=len(measured), mean=statistics.mean(measured),
                                      min=min(measured), max=max(measured), total=sum(measured),
                                      stdev=statistics.stdev(measured) if len(measured)>1 else None)
                                  if measured else None)
            aggregates.append(dict(case=case, method=method, runs=len(selected),
                                   completed=sum(r["completed"] for r in selected),
                                   slo_completed=sum(r["slo_completed"] for r in selected), metrics=values))
    result = dict(status="COMPLETE_FROZEN_MATRIX",
                  confirmed_execution_measured=all(r.get("execution_measurement") is not None for r in rows), runs=rows, aggregates=aggregates,
                  paired_outputs=pairs, runtime_source_manifest_sha256=next(iter(sources)),
                  limitations=["Discarded tokens are not executed recomputation tokens",
                               "Missing native internal cost is not zero",
                               "Output sequence equality is not task-quality equivalence"])
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2)
    print(json.dumps(dict(status=result["status"], runs=len(rows))))


if __name__ == "__main__":
    main()
