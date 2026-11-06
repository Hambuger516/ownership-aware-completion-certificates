"""Validate all configurations before touching GPU state or creating results."""

from pathlib import Path
import math
import re

PROJECT = Path(__file__).resolve().parent.parent
DEFAULTS = dict(
    method="shared_incremental",
    workload="dolly",
    count=24,
    cap=128,
    cap_profile="uniform",
    blocks=96,
    block_size=16,
    batch_tokens=256,
    max_seqs=16,
    arrival_interval=0.025,
    ignore_eos=False,
    prefix=True,
    waves=1,
    reset_between=False,
    cancellation=False,
    busy_reset=False,
    dtype="float16",
    seed=20260917,
    min_prompt=32,
    max_prompt=384,
    observation_limit=120,
    replicate=0,
    audit=True,
    drain_after_steps=64,
    model_path="",
    data_path="",
    trace_path="",
    output_root="runs",
    run_id="smoke",
    case="custom",
    fault=None,
    expected_error=None,
    timeout_seconds=300,
    proposal_mode="exact",
    predictor_z=2.0,
    predictor_checkpoint="",
    patch_checkpoint="",
    enable_prc_pipeline=True,
    learned_checkpoint="",
    # 2026-10-12 CCMB additions. Defaults reproduce the archived 5090 behaviour.
    max_model_len=1024,
    gpu_memory_utilization=0.5,
    drain_policy="timer",   # "timer": legacy periodic whole-cohort serial drain; "none": CCMB (margin-governed deferral only)
    drain_scope="cohort",   # "head": one confirmed head completion per timer segment; finite-admission progress only
    drain_batching="serial",  # cohort_certified refreshes witness on confirmed closed-cohort epochs
    active_drain_admission=False,  # contingent finite-input ACK-joined active members
    ordinary_witness_priority=False,  # contingent ordinary-only reuse of legacy full witness sort
    rank_order=False,       # True: reorder the running queue by witness rank before each schedule (T9 greedy)
    zero_growth_fast=True,  # True: resident requests needing no new page this step bypass the snapshot/certificate (T5, z=0)
    deny_cache=True,        # True: a refused request is re-evaluated only after the next release epoch (T5/T6; decisions unchanged)
    admission_policy="unrestricted",  # utility filter only on C1-only NEW resident admission
    peer_queue_bypass=False,  # scan beyond peer-only refusals; original guard still mandatory
    peer_resident_limit=0,  # optional utility gate on new candidates at the current maximum hard cap
    peer_capacity_adaptive=False,  # release-independent funding of any two largest claims relaxes fixed4
    private_refusal_cache=False,  # exact monotone private-growth refusal reuse; structural commits invalidate
    certificate_variant="shared_incremental",  # isolated study: force_full or exclusive_only use the same authority lifecycle
)
FAULTS = {
    "unmodeled_ref": "Incomplete owner mapping or unmodeled pin/copy reference",
    "bypass_guard": "Controlled allocation bypassed precheck",
}


def validate_config(value, *, require_paths=False):
    if type(value) is not dict:
        raise ValueError("configuration must be a JSON object")
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError("unknown configuration fields: " + ", ".join(sorted(unknown)))
    c = DEFAULTS | value
    if c["method"] not in (
        "native_plain",
        "native_shadow",
        "exclusive_closure",
        "shared_closure",
        "shared_incremental",
        "prc",
        "hard_reserve",
    ):
        raise ValueError("unsupported method")
    if c["certificate_variant"] not in ("shared_incremental", "force_full", "exclusive_only", "common_full_c0", "common_full_c1"):
        raise ValueError("unsupported certificate_variant")
    if c["certificate_variant"] != "shared_incremental":
        if c["method"] != "shared_incremental":
            raise ValueError("isolated certificate variants require method=shared_incremental")
        if c["enable_prc_pipeline"] or c["rank_order"] or c["zero_growth_fast"] or c["deny_cache"]:
            raise ValueError("isolated certificate variants require pipeline/rank/zero-growth/deny-cache disabled")
    if c["admission_policy"] not in ("unrestricted", "c0_first_shorter_cap"):
        raise ValueError("unsupported admission_policy")
    if c["admission_policy"] != "unrestricted":
        if (c["method"] != "shared_incremental"
                or c["certificate_variant"] not in ("shared_incremental", "common_full_c0", "common_full_c1", "force_full")
                or (c["drain_scope"], c["drain_batching"]) not in (("cohort", "cohort_certified"), ("head", "head_funded"), ("first_completion", "cohort_certified"))
                or any(c[k] for k in ("enable_prc_pipeline", "rank_order", "zero_growth_fast", "deny_cache"))):
            raise ValueError("filtered admission requires guarded cohort runtime and disabled shortcuts/pipeline")
    if c["active_drain_admission"]:
        if (c["method"] != "shared_incremental"
                or c["certificate_variant"] not in ("shared_incremental","common_full_c0","common_full_c1", "force_full")
                or (c["drain_scope"],c["drain_batching"],c["drain_policy"]) != ("cohort","cohort_certified","timer")
                or c["admission_policy"] != "c0_first_shorter_cap" or c["count"] != 16 or c["waves"] != 1
                or any(c[k] for k in ("enable_prc_pipeline","rank_order","zero_growth_fast","deny_cache"))):
            raise ValueError("active admission draft requires exact16 single-wave cohort_certified capfilter and disabled shortcuts/pipeline/rank")
    if c["ordinary_witness_priority"]:
        if (c["rank_order"] or c["method"] != "shared_incremental"
                or c["certificate_variant"] not in ("shared_incremental", "common_full_c0", "common_full_c1", "force_full")
                or (c["drain_scope"],c["drain_batching"],c["drain_policy"]) != ("cohort","cohort_certified","timer")
                or c["admission_policy"] != "c0_first_shorter_cap"
                or any(c[k] for k in ("enable_prc_pipeline","zero_growth_fast","deny_cache"))):
            raise ValueError("ordinary priority draft requires cohort_certified whole-cohort/capfilter, reviewed certificate variants and disabled rank/shortcuts/pipeline")
    if c["private_refusal_cache"] and (c["method"] != "shared_incremental" or c["certificate_variant"] != "shared_incremental"):
        raise ValueError("private refusal cache supports only shared_incremental")
    if type(c['peer_capacity_adaptive']) is not bool or (c['peer_capacity_adaptive'] and
            (c['peer_resident_limit'] != 4 or not c['peer_queue_bypass'])):
        raise ValueError('adaptive pair funding requires fixed4 plus reviewed queue bypass')
    if type(c["peer_queue_bypass"]) is not bool or (c["peer_queue_bypass"] and c["peer_resident_limit"] != 4):
        raise ValueError("peer queue bypass requires explicit bool and fixed peer limit4")
    if type(c["peer_resident_limit"]) is not int or c["peer_resident_limit"] not in (0,2,4):
        raise ValueError("peer_resident_limit must be0,2,4")
    if c["peer_resident_limit"] and (
            c["method"] != "shared_incremental" or c["certificate_variant"] != "shared_incremental"
            or (c["drain_scope"],c["drain_batching"],c["drain_policy"]) != ("cohort","cohort_certified","timer")
            or c["admission_policy"] != "c0_first_shorter_cap" or not c["ordinary_witness_priority"]
            or any(c[k]for k in ("active_drain_admission","private_refusal_cache","enable_prc_pipeline","rank_order","zero_growth_fast","deny_cache"))):
        raise ValueError("peer limit requires isolated ordinary cohort profile without other optimizations")
    if c["workload"] not in ("dolly", "constructed_shared", "exact_context", "request_trace"):
        raise ValueError("unsupported workload")
    if c["cap_profile"] not in ("uniform", "mixed", "boundaries", "ownership_probe"):
        raise ValueError("unsupported cap profile")
    if c["dtype"] not in ("float16", "bfloat16"):
        raise ValueError("supported dtypes: float16, bfloat16")
    for k in (
        "count",
        "cap",
        "blocks",
        "block_size",
        "batch_tokens",
        "max_seqs",
        "waves",
        "drain_after_steps",
        "timeout_seconds",
    ):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    for k in ("seed", "replicate", "min_prompt", "max_prompt"):
        if type(c[k]) is not int or c[k] < 0:
            raise ValueError(f"{k} must be a nonnegative integer")
    for k in ("ignore_eos", "prefix", "reset_between", "cancellation", "busy_reset", "audit", "enable_prc_pipeline", "rank_order", "ordinary_witness_priority", "active_drain_admission", "zero_growth_fast", "deny_cache", "private_refusal_cache"):
        if type(c[k]) is not bool:
            raise ValueError(f"{k} must be Boolean")
    if type(c["max_model_len"]) is not int or c["max_model_len"] < 64:
        raise ValueError("max_model_len must be an integer >= 64")
    if type(c["gpu_memory_utilization"]) not in (int, float) or not (0.05 <= c["gpu_memory_utilization"] <= 0.98):
        raise ValueError("gpu_memory_utilization must lie in [0.05, 0.98]")
    if c["drain_policy"] not in ("timer", "none"):
        raise ValueError("drain_policy must be 'timer' or 'none'")
    if c["drain_scope"] not in ("cohort", "head", "first_completion"):
        raise ValueError("invalid drain_scope")
    if c["drain_scope"] == "head" and (c["method"] != "shared_incremental" or c["drain_policy"] != "timer"):
        raise ValueError("head drain requires method=shared_incremental and drain_policy=timer")
    if c["drain_batching"] not in ("serial", "witness_preserving", "cohort_certified", "head_funded"):
        raise ValueError("invalid drain_batching")
    if c["drain_scope"] == "first_completion":
        if (c["drain_batching"] != "cohort_certified" or c["drain_policy"] != "timer"
                or c["method"] != "shared_incremental"
                or c["certificate_variant"] not in ("shared_incremental", "common_full_c0", "common_full_c1")):
            raise ValueError("first_completion requires cohort_certified timer and reviewed shared_incremental certificate variants")
    if c["drain_batching"] in ("witness_preserving", "cohort_certified"):
        allowed_scopes = ("cohort", "first_completion") if c["drain_batching"] == "cohort_certified" else ("cohort",)
        if c["drain_scope"] not in allowed_scopes or c["drain_policy"] != "timer" or c["method"] != "shared_incremental":
            raise ValueError("witness-preserving batching requires shared_incremental, cohort scope and timer")
        if any(c[k] for k in ("enable_prc_pipeline", "rank_order", "zero_growth_fast", "deny_cache")):
            raise ValueError("witness-preserving batching requires pipeline/rank/zero-growth/deny-cache disabled")
    if c["drain_batching"] == "head_funded":
        if c["drain_scope"] != "head" or c["drain_policy"] != "timer" or c["method"] != "shared_incremental":
            raise ValueError("head-funded batching requires shared_incremental, head scope and timer")
        if any(c[k] for k in ("enable_prc_pipeline", "rank_order", "zero_growth_fast", "deny_cache")):
            raise ValueError("head-funded batching requires pipeline/rank/zero-growth/deny-cache disabled")
    for k in ("arrival_interval", "observation_limit"):
        if type(c[k]) not in (int, float) or not math.isfinite(c[k]) or c[k] < 0:
            raise ValueError(f"invalid {k}")
    if c["observation_limit"] == 0:
        raise ValueError("positive observation limit required")
    if c["proposal_mode"] not in ("exact", "learned"):
        raise ValueError("proposal_mode must be exact or learned")
    if type(c["predictor_z"]) not in (int, float) or not math.isfinite(c["predictor_z"]) or c["predictor_z"] < 0:
        raise ValueError("predictor_z must be a finite nonnegative number")
    if c["block_size"] not in (16, 32):
        raise ValueError("only reviewed block sizes 16/32 supported")
    if (c["blocks"] - 1) * c["block_size"] < c["max_model_len"]:
        raise ValueError("KV capacity must include one reserved null block and cover max_model_len")
    if c["cap"] > c["max_model_len"] or c["min_prompt"] > c["max_prompt"] or c["max_prompt"] > c["max_model_len"]:
        raise ValueError("invalid context limits")
    if c["cancellation"] and c["count"] < 2:
        raise ValueError("cancellation test requires at least two requests")
    if c["workload"] == "request_trace":
        if c["waves"] != 1 or c["cancellation"] or c["busy_reset"] or c["reset_between"]:
            raise ValueError("request_trace is a single-wave, non-cancellation workload")
    if c["fault"] is not None:
        if c["fault"] not in FAULTS or c["method"] != "shared_incremental":
            raise ValueError("invalid isolated fault test")
        if c["expected_error"] != FAULTS[c["fault"]]:
            raise ValueError("fault must declare its exact expected error")
    elif c["expected_error"] is not None:
        raise ValueError("expected_error without a fault")
    if type(c["case"]) is not str or not c["case"]:
        raise ValueError("case must be nonempty")
    if type(c["run_id"]) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}", c["run_id"]):
        raise ValueError("run_id must be a safe single path component")
    for name in (
        "model_path",
        "data_path",
        "trace_path",
        "output_root",
        "learned_checkpoint",
        "predictor_checkpoint",
        "patch_checkpoint",
    ):
        if type(c[name]) is not str:
            raise ValueError(f"{name} must be a path string")
    if not c["data_path"]:
        c["data_path"] = str(Path(__file__).resolve().parent / "data/dolly_pilot_sample.json")
    if c["workload"] == "request_trace" and not c["trace_path"]:
        raise ValueError("request_trace requires trace_path")
    if not c["output_root"]:
        raise ValueError("output_root is required")
    if require_paths:
        if not c["model_path"] or not Path(c["model_path"]).is_dir():
            raise ValueError("local model_path directory is required")
        # Resolve before the child process changes working directory. Relative
        # user paths are interpreted from the invoking shell, not site-packages.
        for name in ("model_path", "data_path", "output_root"):
            c[name] = str(Path(c[name]).resolve())
        if c["trace_path"]:
            c["trace_path"] = str(Path(c["trace_path"]).resolve())
        if c["learned_checkpoint"]:
            c["learned_checkpoint"] = str(Path(c["learned_checkpoint"]).resolve())
        for name in ("predictor_checkpoint", "patch_checkpoint"):
            if c[name]:
                c[name] = str(Path(c[name]).resolve())
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            if not (Path(c["model_path"]) / name).is_file():
                raise ValueError("missing model artifact: " + name)
        model_dir = Path(c["model_path"])
        if not ((model_dir / "model.safetensors").is_file() or (model_dir / "model.safetensors.index.json").is_file()):
            raise ValueError("missing model artifact: model.safetensors or sharded model.safetensors.index.json")
        if c["learned_checkpoint"] and not Path(c["learned_checkpoint"]).is_file():
            raise ValueError("learned_checkpoint file is required")
        if c["predictor_checkpoint"] and not Path(c["predictor_checkpoint"]).is_file():
            raise ValueError("predictor_checkpoint file is required")
        if c["patch_checkpoint"] and not Path(c["patch_checkpoint"]).is_file():
            raise ValueError("patch_checkpoint file is required")
        if c["proposal_mode"] == "learned" and not c["patch_checkpoint"]:
            raise ValueError("proposal_mode=learned requires patch_checkpoint")
        if c["workload"] == "dolly" and not Path(c["data_path"]).is_file():
            raise ValueError("Dolly sample not found")
        if c["workload"] == "request_trace" and not Path(c["trace_path"]).is_file():
            raise ValueError("request trace file not found")
        if (Path(c["output_root"]) / c["run_id"]).exists():
            raise FileExistsError("run_id already exists; refusing overwrite")
    return c
