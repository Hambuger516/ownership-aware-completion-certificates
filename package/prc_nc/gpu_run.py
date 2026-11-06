"""Real GPU contract/lifecycle validation, with partial results on failure."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import platform
import time
import traceback
from pathlib import Path

from .configuration import validate_config
from .patch_exact import PatchExactSolver
from .online import single_worker_graph
from .prediction import PredictionQuery, make_split_manifest, load_runtime_predictor
from .routing import RouteDevice, RouteRequest, TypedRoute
from .workloads import load_request_trace


def _checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_predictor(config):
    """Load the leakage-safe real-trace predictor used by the formal PRC path."""
    configured = config.get("predictor_checkpoint", "")
    path = Path(configured) if configured else Path(__file__).resolve().parent / "data/predictor_real_trace_v2.json"
    if not path.is_file():
        raise FileNotFoundError(f"real-trace predictor checkpoint is required: {path}")
    predictor, digest = load_runtime_predictor(path, z_value=config.get("predictor_z", 2.0))
    return predictor, path.resolve(), digest


def _single_gpu_patch_graph(prompt_tokens: int, hard_need_pages: int, blocks: int, block_size: int):
    """Build the one-worker physical PATCH instance used by the GPU adapter.

    The graph is intentionally a one-stage degeneration of the paper's stage
    DAG.  Its memory label is the hard request envelope; the predictor never
    changes this label and PATCH therefore cannot authorize extra KV pages.
    """
    return single_worker_graph(hard_need_pages, blocks - 1)


def _build_prc_proposals(config, tokenizer, prompts, caps, blocks, block_size, data_rows=None, request_metadata=None):
    """Create formal PRC proposals without using future output information.

    Prediction, routing and PATCH are proposal computations only.  The
    adapter's authority remains the sole source of execution permission.  The
    legacy ``tokenizer`` and ``data_rows`` parameters are retained for callers
    of the old helper signature but are intentionally ignored.
    """
    predictor, predictor_path, predictor_sha = _formal_predictor(config)
    patch_solver = PatchExactSolver()
    patch_path = None
    patch_sha = None
    if config.get("proposal_mode", "exact") == "learned":
        patch_path_value = config.get("patch_checkpoint", "")
        if not patch_path_value:
            raise ValueError("proposal_mode=learned requires a strict neural patch_checkpoint")
        patch_path = Path(patch_path_value)
        if not patch_path.is_file():
            raise FileNotFoundError(f"neural PATCH checkpoint is required: {patch_path}")
        from .patch_neural import NeuralPatchSolver
        import torch
        patch_solver = NeuralPatchSolver.from_checkpoint(
            patch_path, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        patch_sha = _checkpoint_sha256(patch_path)
    route = TypedRoute()
    rows = []
    for wave in range(config["waves"]):
        for i, (prompt, cap) in enumerate(zip(prompts, caps)):
            rid = f"w{wave}-r{i}"
            prompt_tokens = len(prompt)
            hard_need_pages = (prompt_tokens + cap + block_size - 1) // block_size
            metadata = request_metadata[i] if request_metadata else {}
            category = metadata.get("category", "default")
            prediction = predictor.predict(PredictionQuery(rid, prompt_tokens, 0, category, cap))
            route_proposal = route.choose(
                RouteRequest(rid, preferred_domain="default", hard_need=hard_need_pages),
                [RouteDevice("gpu0", "default", locality=1.0,
                             free_capacity=blocks - 1, metadata={"pool_id": "gpu0-kv"})],
            )
            graph, query = _single_gpu_patch_graph(prompt_tokens, hard_need_pages, blocks, block_size)
            patch_started = time.perf_counter_ns()
            patch_result = patch_solver.solve(graph, query)
            patch_seconds = (time.perf_counter_ns() - patch_started) / 1e9
            if hasattr(patch_result, "solution"):
                placement_solution = patch_result.solution
                patch_timing = {
                    "exact_seconds": 0.0,
                    "total_seconds": patch_seconds,
                    "inference_seconds": patch_result.inference_seconds,
                    "decoding_seconds": patch_result.decoding_seconds,
                    "repair_seconds": patch_result.repair_seconds,
                    "exact_fallback_seconds": patch_result.exact_fallback_seconds,
                    "candidate_count": patch_result.candidate_count,
                    "repair_trials": patch_result.repair_trials,
                    "repaired": patch_result.repaired,
                    "used_exact_fallback": patch_result.used_exact_fallback,
                    "feature_provenance": patch_result.feature_provenance,
                }
            else:
                placement_solution = patch_result
                patch_timing = {
                    "exact_seconds": patch_seconds,
                    "total_seconds": patch_seconds,
                    "inference_seconds": 0.0, "decoding_seconds": 0.0,
                    "repair_seconds": 0.0, "exact_fallback_seconds": 0.0,
                    "candidate_count": 0, "repair_trials": 0,
                    "repaired": False, "used_exact_fallback": False,
                    "feature_provenance": "exact-authoritative",
                }
            if tuple(placement_solution.assignments) != ("gpu0",):
                raise AssertionError("single-GPU PATCH placement did not select gpu0")
            rows.append({
                "request_id": rid,
                "category": category,
                "action_id": f"{rid}:preflight",
                "scope": "offline preflight; runtime actions use current authority version and telemetry",
                "worker_id": "gpu0",
                "pool_id": "gpu0-kv",
                "state_version": 0,
                "hard_max_output_tokens": cap,
                "hard_need_pages": hard_need_pages,
                "soft_reserve_tokens": prediction.soft_upper_tokens,
                "prediction": asdict(prediction),
                "predictor_checkpoint": str(predictor_path),
                "predictor_z": predictor.z_value,
                "predictor_checkpoint_sha256": predictor_sha,
                "predictor_split_sha256": predictor.fit_split_sha256,
                "predictor_trace_sha256": list(predictor.fit_trace_sha256),
                "route": asdict(route_proposal),
                "placement": {
                    "devices": list(placement_solution.assignments),
                    "path": list(placement_solution.path),
                    "objective": placement_solution.objective,
                    "solver": placement_solution.solver,
                    "memory_by_device": dict(placement_solution.memory_by_device),
                    "reliability": placement_solution.reliability,
                    "profile_scope": "single-worker structural degeneration; hardware objectives unprofiled",
                    **patch_timing,
                },
                "patch_checkpoint": str(patch_path) if patch_path else None,
                "patch_checkpoint_sha256": patch_sha,
            })
    return rows


def run(configuration):
    config = validate_config(configuration, require_paths=True)
    out = Path(config["output_root"]) / config["run_id"]
    out.mkdir(parents=True, exist_ok=False)
    # run() may be invoked twice in the same process; a baseline must never
    # inherit a prior PRC proposal manifest or fault-injection setting.
    for name in ("PRC_PROPOSAL_PATH", "PRC_TEST_FAULT", "PRC_FINITE_UNIVERSE_PATH", "PRC_FINITE_UNIVERSE_SHA256"):
        os.environ.pop(name, None)
    for key, value in {
        "PRC_MODE": "shared_incremental" if config["method"] == "prc" else config["method"],
        "PRC_RUN_DIR": str(out),
        "VLLM_NO_USAGE_STATS": "1",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "OMP_NUM_THREADS": "1",
        "PRC_AUDIT": "1" if config["audit"] else "0",
        "PRC_DRAIN_AFTER_STEPS": str(config["drain_after_steps"]),
        "PRC_DRAIN_POLICY": config["drain_policy"],
        "PRC_DRAIN_SCOPE": config["drain_scope"],
        "PRC_DRAIN_BATCHING": config["drain_batching"],
        "PRC_ADMISSION_POLICY": config["admission_policy"],
        "PRC_ACTIVE_DRAIN_ADMISSION": "1" if config["active_drain_admission"] else "0",
        "PRC_ORDINARY_WITNESS_PRIORITY": "1" if config["ordinary_witness_priority"] else "0",
        "PRC_RANK_ORDER": "1" if config["rank_order"] else "0",
        "PRC_ZERO_GROWTH_FAST": "1" if config["zero_growth_fast"] else "0",
        "PRC_DENY_CACHE": "1" if config["deny_cache"] else "0",
        "PRC_CERTIFICATE_VARIANT": config["certificate_variant"],
        "PRC_PEER_RESIDENT_LIMIT": str(config["peer_resident_limit"]),
        "PRC_PEER_CAPACITY_ADAPTIVE": "1" if config["peer_capacity_adaptive"] else "0",
        "PRC_PEER_QUEUE_BYPASS": "1" if config["peer_queue_bypass"] else "0",
        "PRC_PRIVATE_REFUSAL_CACHE": "1" if config["private_refusal_cache"] else "0",
        "PRC_FAST_STATE": "0" if config["audit"] else "1",
        "PRC_MAX_MODEL_LEN": str(config["max_model_len"]),
        "PRC_WORKER_ID": "gpu0",
        "PRC_POOL_ID": "gpu0-kv",
    }.items():
        os.environ[key] = value
    manifest = {
        "status": "STARTING",
        "configuration": config,
        "execution_measurement": os.environ.get("PRC_MEASURE_EXECUTION", "0") == "1",
        "started_unix": time.time(),
        "scope": "Single GPU synchronous page adapter; explicit resident drain scope; audit flag controls redundant oracles. No distributed worker protocol.",
        "ordinary_priority_contract": {
            "enabled": config["ordinary_witness_priority"],
            "operator": "Reuse complete existing witness rank stable sort only after observe finds no active drain; ordinary waiting/admissions unchanged",
            "limits": "No ordinary positive-head or speed guarantee; actual budgets/confirmed positions may be zero. Existing timer and all allocation guards remain. Not a new sorting algorithm.",
            "cost": "Same O(n log n) stable sort and observation in both common-full arms; existing common C0 overlap witness and dual solves retained; full service includes all instrumentation costs",
        },
        "drain_progress_contract": {
            "scope": config["drain_scope"], "trigger_steps": config["drain_after_steps"],
            "batching": config["drain_batching"],
            "applied": config["drain_policy"] == "timer" and config["method"] in ("shared_closure", "shared_incremental", "prc"),
            "obligation": "End after the first trusted successful completion of any frozen resident member; cancellation is not success; outside admission resumes only at the next fully settled model/resource cut" if config["drain_scope"] == "first_completion" else "Complete at least one selected witness head before resuming ordinary parallel service; cancellation does not satisfy completion" if config["drain_scope"] == "head" else "Freeze and drain the entire resident witness cohort",
            "head_scope_limit": "Conditional progress for finite offered cohorts / finitely many admissions, finite service/reclamation and repeated fair timer invocation; no open-arrival stability, SLO or inherited whole-cohort theorem",
            "batching_obligation": ("Freeze resident survivors; retain one funded selected head until its genuine completion; reliable head cancellation permits replacement only at next settled Authority cut; no outside admissions; actual positive head budget and model-confirmed unique positions; followers preserve post.free >= post.need[selected_head] and pass original Authority" if config["drain_batching"] == "head_funded" else "Frozen membership; refresh current Authority witness only after prior confirmed model work; positive first head; all followers re-certified by Authority; every active epoch has unique confirmed executed positions" if config["drain_batching"] == "cohort_certified" else "Every nonempty active batch has positive successfully allocated head tokens; head first; no outside admissions; every candidate preserves the captured full witness in the active C0/C1 view" if config["drain_batching"] == "witness_preserving" else "Original serial head-only service"),
            "batching_progress_limit": ("Finite frozen membership and fixed caps; each eligible epoch has positive unique confirmed work until one member completes; no specific head service promise; finite admitted population, fair timer and finite settlement required for eventual whole remainder; no open-arrival stability or seconds bound" if config["drain_scope"] == "first_completion" else "Conditional finite selected-head service, fair eligible synchronous epochs and finite rights/result/terminal settlement; no claim every external API job is admitted; finite admitted population required for eventual whole remainder; cancellation not successful completion" if config["drain_batching"] == "head_funded" else "Finite frozen membership; fixed hard caps; positive unique model-confirmed service positions per epoch; no rollback/reexecution/speculation; eligible synchronous fault-free service and finite settlement; numerical seconds additionally require epoch-time bounds" if config["drain_batching"] == "cohort_certified" else "Finite frozen cohort; finitely many head service units; finite epoch/settlement/reclamation times; numerical seconds require explicit epoch-time bounds, not just output/page caps"),
            "batching_cost": ("Extra O(P) funding arithmetic after O(n) identity lookup; existing snapshot, original Authority and logging costs remain charged; not O(P) whole-step cost" if config["drain_batching"] == "head_funded" else "Existing Authority C0/C1 common-full checks or declared incremental path; no extra frozen-order margin scans; snapshot and event work charged in service timing" if config["drain_batching"] == "cohort_certified" else "Each active candidate additionally pays full C0 then C1 fixed-order margin scans O(M+nP), separately counted; not logarithmic whole-step cost" if config["drain_batching"] == "witness_preserving" else "Original serial policy; existing controller, physical snapshot and event costs remain in service timing"),
        },
        "source_sha256": {
            p.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.rglob("*.py")
        },
        "pilot_slo": {"ttft_seconds": 2.0, "mean_tpot_seconds": 0.1},
        "certificate_study": {
            "variant": config["certificate_variant"],
            "variant_applied": config["method"] in ("shared_closure", "shared_incremental", "prc"),
            "counter_scope": "cumulative including warmup; oracle counts prepare searches, initial_oracle counts construction separately; diagnostic CF searches separate; not independent experimental units",
            "oracle_baseline": "two full C0/C1 searches in both arms on every prepare; common fresh C0 witness on overlap; identical fresh physical index; active guard view alone selects C0 versus C1 acceptance" if config["certificate_variant"] in ("common_full_c0", "common_full_c1") else "indexed_closure at every prepare; shared witness/index maintenance retained to match incremental timer-head policy" if config["certificate_variant"] == "force_full" else "complete private-release-only information view, actual ownership unchanged" if config["certificate_variant"] == "exclusive_only" else "fixed-witness plus complete fallback",
        },
    }
    (out / "source").mkdir()
    for p in Path(__file__).parent.rglob("*.py"):
        destination = out / "source" / p.relative_to(Path(__file__).parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    records, transitions, wave_results = {}, [], []
    engine = None
    started = time.monotonic()
    try:
        from transformers import AutoTokenizer
        import torch, vllm
        from vllm import EngineArgs, LLMEngine, SamplingParams

        model = Path(config["model_path"])
        tok = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        n = config["count"]
        caps = []
        for i in range(n):
            profile = config["cap_profile"]
            caps.append(
                {
                    "uniform": config["cap"],
                    "mixed": [32, 64, 128, 256][i % 4],
                    "boundaries": [1, 15, 16, 17, 31, 32, 33][i % 7],
                    "ownership_probe": 16 if i % 6 < 2 else 512,
                }[profile]
            )
        prompts, selection = [], []
        arrivals = [i * config["arrival_interval"] for i in range(n)]
        trace_info = None
        if config["workload"] == "request_trace":
            trace_info = load_request_trace(config["trace_path"], count=n, tokenizer=tok)
            prompts = trace_info["prompts"]
            caps = trace_info["caps"]
            arrivals = trace_info["arrivals"]
            selection = trace_info["selection"]
            manifest.update(
                trace_path=str(Path(config["trace_path"]).resolve()),
                trace_sha256=trace_info["data_sha256"],
                trace_schema_version=trace_info["schema_version"],
                trace_provenance=trace_info["provenance"],
                arrival_schedule_s=arrivals,
            )
        elif config["workload"] == "exact_context":
            unit = tok.encode("Physical page capacity verification. ", add_special_tokens=False)
            for i in range(n):
                ids = (unit * ((512 + len(unit) - 1) // len(unit)))[:512]
                prompts.append(ids)
                selection.append(
                    {
                        "constructed_id": i,
                        "prompt_tokens": 512,
                        "purpose": "exact 1024-token prompt plus output boundary",
                    }
                )
        elif config["workload"] == "dolly":
            data = Path(config["data_path"])
            manifest["data_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
            rows = json.loads(data.read_text(encoding="utf-8"))
            identity_split = make_split_manifest(row["source_index"] for row in rows)
            manifest["prediction_identity_split"] = identity_split.as_dict()
            manifest["workload_identity_partition"] = "test"
            for row in rows:
                if identity_split.split_of(row["source_index"]) != "test":
                    continue
                content = row["instruction"] + ("\n\n" + row["input"] if row["input"] else "")
                ids = tok.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a helpful assistant. Answer accurately and clearly."},
                        {"role": "user", "content": content},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                if config["min_prompt"] <= len(ids) <= config["max_prompt"] and len(ids) + max(caps) <= config["max_model_len"]:
                    prompts.append(ids)
                    selection.append({"source_index": row["source_index"], "prompt_tokens": len(ids),
                                      "category": row.get("category", "default")})
                if len(prompts) == n:
                    break
        elif config["workload"] == "constructed_shared":
            for i in range(n):
                group = (0 if i % 6 < 2 else 1) if config["cap_profile"] == "ownership_probe" else i % 2
                topic = "the water cycle" if group == 0 else "the solar system"
                context = f"This lesson concerns {topic}. Explain ideas with short examples. " * 12
                ids = tok.apply_chat_template(
                    [
                        {"role": "system", "content": "You are a helpful tutor."},
                        {
                            "role": "user",
                            "content": context + f"Question {i}: explain one important process in detail.",
                        },
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
                prompts.append(ids)
                selection.append({"constructed_id": i, "prefix_group": group, "prompt_tokens": len(ids)})
        if len(prompts) != n:
            raise AssertionError("Insufficient eligible prompts")
        if config["workload"] != "request_trace":
            manifest["arrival_schedule_s"] = arrivals
        if (config["blocks"] - 1) * config["block_size"] < config["max_model_len"]:
            raise ValueError("Engine max_model_len exceeds usable pool; account for the null block before startup")
        if any(len(p) + c > config["max_model_len"] for p, c in zip(prompts, caps)):
            raise AssertionError("Input exceeds context contract")
        if any(
            (len(p) + c + config["block_size"] - 1) // config["block_size"] > config["blocks"] - 1
            for p, c in zip(prompts, caps)
        ):
            raise AssertionError("A hard request contract exceeds total physical pool capacity")
        manifest.update(
            prompt_token_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
            request_caps=caps,
            selection=selection,
            torch=torch.__version__,
            vllm=vllm.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            python=platform.python_version(),
            physical_capacity_pages=config["blocks"] - 1,
        )
        controlled_method = config["method"] in ("shared_closure", "shared_incremental", "prc")
        proposals = []
        if controlled_method and config["enable_prc_pipeline"]:
            # The formal proposal path receives tokenized prompts and hard
            # contracts only.  Dataset rows (including any reference answer
            # metadata) are deliberately not passed into prediction.
            proposals = _build_prc_proposals(config, tok, prompts, caps, config["blocks"], config["block_size"],
                                             request_metadata=selection)
            checkpoints = out / "checkpoints"
            checkpoints.mkdir()
            for field in ("predictor_checkpoint", "patch_checkpoint"):
                source = proposals[0].get(field)
                if source:
                    source = Path(source)
                    target = checkpoints / source.name
                    target.write_bytes(source.read_bytes())
                    for row in proposals:
                        row[field] = str(target.resolve())
            if config["workload"] == "dolly" and proposals[0]["predictor_split_sha256"] != identity_split.sha256:
                raise AssertionError("calibration and evaluation identity partitions differ")
            proposal_path = out / "proposals.json"
            proposal_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
            os.environ["PRC_PROPOSAL_PATH"] = str(proposal_path)
            manifest.update(
                proposal_count=len(proposals),
                pipeline_schema="online-authority-v1",
                proposal_mode=config["proposal_mode"],
                pipeline_components=["real_trace_conditional_predictor", "typed_route", "patch_" + config["proposal_mode"], "certificate_controller", "causal_commit", "drain"],
                proposal_hard_contract="offline preflight contracts; online statistics gate new admissions; current authority version alone grants physical execution",
            )
            first_proposal = proposals[0]
            manifest.update(
                predictor_checkpoint=first_proposal["predictor_checkpoint"],
                predictor_checkpoint_sha256=first_proposal["predictor_checkpoint_sha256"],
                predictor_split_sha256=first_proposal["predictor_split_sha256"],
                predictor_trace_sha256=first_proposal["predictor_trace_sha256"],
                patch_checkpoint=first_proposal["patch_checkpoint"],
                patch_checkpoint_sha256=first_proposal["patch_checkpoint_sha256"],
                learned_exact_fallback_count=sum(
                    int(row["placement"].get("used_exact_fallback", False)) for row in proposals
                ),
            )
        else:
            manifest.update(proposal_count=0, pipeline_components=[])
        if config["active_drain_admission"]:
            from .active_admission import finite_universe
            universe=finite_universe(prompts,caps,block_size=config["block_size"],capacity=config["blocks"]-1)
            universe_path=out/"finite_request_universe.json"
            with universe_path.open("x") as stream:json.dump(universe,stream,sort_keys=True,separators=(",",":"))
            universe_sha=_checkpoint_sha256(universe_path)
            os.environ["PRC_FINITE_UNIVERSE_PATH"]=str(universe_path)
            os.environ["PRC_FINITE_UNIVERSE_SHA256"]=universe_sha
            manifest["active_admission_contract"]=dict(enabled=True,universe_path=str(universe_path),universe_sha256=universe_sha,
                scope="Finite16 workload plus separate warmup; joins after exact physical ACK, per-epoch fixed head and model-confirmed work; no streaming/fairness/time guarantee")
            manifest["drain_progress_contract"].update(obligation="Drain current members including exactly ACK-joined NEW; whole expanded set must terminate",batching_obligation="Fresh settled witness; fixed epoch head; NEW only after successful positive head allocation; actual work and cancellation receipts settle before next cut",batching_progress_limit="Finite bound input universe potential includes not-yet-admitted IDs; cancellation is not positive model work; no old frozen-membership theorem")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if config.get("fault"):
            os.environ["PRC_TEST_FAULT"] = config["fault"]
        scheduler = (
            None
            if config["method"] == "native_plain"
            else "prc_nc.adapters.fault_scheduler.FaultScheduler"
            if config.get("fault")
            else "prc_nc.adapters.vllm_scheduler.CertificateScheduler"
        )
        if config["method"] == "native_plain" and manifest["execution_measurement"]:
            scheduler = "prc_nc.adapters.execution_measurement.MeasuredNativeScheduler"
        args = EngineArgs(
            model=str(model),
            dtype=config["dtype"],
            max_model_len=config["max_model_len"],
            max_num_seqs=config["max_seqs"],
            max_num_batched_tokens=config["batch_tokens"],
            block_size=config["block_size"],
            num_gpu_blocks_override=config["blocks"],
            gpu_memory_utilization=config["gpu_memory_utilization"],
            enable_prefix_caching=config["prefix"],
            enable_chunked_prefill=True,
            enforce_eager=True,
            async_scheduling=False,
            seed=config["seed"],
            scheduler_cls=scheduler,
        )
        from .diagnostic_observer import install_engine
        install_engine(LLMEngine)
        engine = LLMEngine.from_engine_args(args)
        manifest["engine_startup_s"] = time.monotonic() - started
        engine.add_request("warmup", {"prompt_token_ids": prompts[0]}, SamplingParams(temperature=0, max_tokens=8))
        while engine.has_unfinished_requests():
            engine.step()
        if not engine.reset_prefix_cache():
            raise AssertionError("Warmup cache failed to reset")
        manifest["status"] = "RUNNING"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for wave in range(config["waves"]):
            if wave and config["reset_between"]:
                reset = engine.reset_prefix_cache()
                transitions.append({"event": "idle_cache_reset", "wave": wave, "success": reset})
                if not reset:
                    raise AssertionError("Idle prefix cache reset failed")
            service_started_unix = time.time()  # measured before monotonic service start
            clock_zero = time.monotonic()
            rids = [f"w{wave}-r{i}" for i in range(n)]
            for i, rid in enumerate(rids):
                records[rid] = {
                    "request_id": rid,
                    "wave": wave,
                    "index": i,
                    "cap": caps[i],
                    "offered_at_s": arrivals[i],
                    "submitted_at_s": None,
                    "first_token_s": None,
                    "finished_at_s": None,
                    "cancelled_at_s": None,
                    "tokens": [],
                    "token_times_s": [],
                    "terminal": None,
                }
            next_request = 0
            waiting_cancelled = False
            running_cancelled = False
            busy_reset_done = False
            while next_request < n or engine.has_unfinished_requests():
                now = time.monotonic() - clock_zero
                if now > config["observation_limit"]:
                    raise TimeoutError("Observation limit reached; partial requests retained")
                while next_request < n and now >= arrivals[next_request]:
                    rid = rids[next_request]
                    engine.add_request(
                        rid,
                        {"prompt_token_ids": prompts[next_request]},
                        SamplingParams(
                            temperature=0,
                            max_tokens=caps[next_request],
                            ignore_eos=config["ignore_eos"],
                            seed=config["seed"],
                        ),
                    )
                    records[rid]["submitted_at_s"] = time.monotonic() - clock_zero
                    next_request += 1
                if config["cancellation"] and next_request == n and not waiting_cancelled:
                    rid = rids[-1]
                    if records[rid]["terminal"] is None:
                        engine.abort_request([rid])
                        engine.abort_request([rid])
                        records[rid].update(cancelled_at_s=time.monotonic() - clock_zero, terminal="cancelled")
                        transitions.append({"event": "waiting_abort_twice", "request_id": rid})
                        waiting_cancelled = True
                if not engine.has_unfinished_requests():
                    if next_request < n:
                        time.sleep(0.001)
                    continue
                outputs = engine.step()
                stamp = time.monotonic() - clock_zero
                for result in outputs:
                    r = records[result.request_id]
                    if r["terminal"] is not None:
                        raise AssertionError("Output emitted after terminal state")
                    completion = result.outputs[0]
                    tokens = list(completion.token_ids)
                    if tokens[: len(r["tokens"])] != r["tokens"]:
                        raise AssertionError("Streaming token prefix changed")
                    if len(tokens) > r["cap"]:
                        raise AssertionError("Output cap exceeded")
                    added = len(tokens) - len(r["tokens"])
                    if added:
                        if r["first_token_s"] is None:
                            r["first_token_s"] = stamp
                        r["token_times_s"].extend([stamp] * added)
                        r["tokens"] = tokens
                    if result.finished:
                        if config["ignore_eos"] and len(tokens) != r["cap"]:
                            raise AssertionError("Forced-cap run terminated early")
                        r.update(
                            finished_at_s=stamp,
                            terminal="completed",
                            finish_reason=completion.finish_reason,
                            text=completion.text,
                        )
                active = [
                    r for rid, r in records.items() if rid in rids and r["terminal"] is None and len(r["tokens"]) >= 4
                ]
                if config["busy_reset"] and active and not busy_reset_done:
                    reset = engine.reset_prefix_cache()
                    transitions.append({"event": "busy_cache_reset", "success": reset, "wave": wave})
                    if reset:
                        raise AssertionError("Busy cache reset unexpectedly succeeded with live KV")
                    busy_reset_done = True
                if config["cancellation"] and active and not running_cancelled:
                    rid = active[0]["request_id"]
                    engine.abort_request([rid])
                    engine.abort_request([rid])
                    records[rid].update(cancelled_at_s=time.monotonic() - clock_zero, terminal="cancelled")
                    transitions.append({"event": "running_abort_twice", "request_id": rid})
                    running_cancelled = True
            if config["cancellation"] and not (waiting_cancelled and running_cancelled):
                raise AssertionError("Cancellation coverage not exercised")
            if config["busy_reset"] and not busy_reset_done:
                raise AssertionError("Busy reset path not exercised")
            if any(records[r]["terminal"] is None for r in rids):
                raise AssertionError("Unfinished request dropped from engine")
            service_finished_monotonic = time.monotonic()
            service_finished_unix = time.time()  # measured after monotonic service end
            wave_results.append(
                {
                    "wave": wave,
                    "duration_s": service_finished_monotonic - clock_zero,
                    "service_started_unix": service_started_unix,
                    "service_started_monotonic": clock_zero,
                    "service_finished_monotonic": service_finished_monotonic,
                    "service_finished_unix": service_finished_unix,
                    "service_unix_boundary_scope": "measured enclosing Unix endpoints; duration remains monotonic",
                    "completed": sum(records[r]["terminal"] == "completed" for r in rids),
                    "cancelled": sum(records[r]["terminal"] == "cancelled" for r in rids),
                }
            )
            (out / "requests.json").write_text(json.dumps(list(records.values()), indent=2))
        final_reset = engine.reset_prefix_cache()
        transitions.append({"event": "final_cache_reset", "success": final_reset})
        if not final_reset:
            raise AssertionError("Resident KV leaked after draining all requests")
        for r in records.values():
            r["ttft_s"] = r["first_token_s"] - r["offered_at_s"] if r["first_token_s"] is not None else None
            r["mean_tpot_s"] = (
                (r["finished_at_s"] - r["first_token_s"]) / (len(r["tokens"]) - 1)
                if r["terminal"] == "completed" and len(r["tokens"]) > 1
                else None
            )
            r["slo_met"] = (
                r["terminal"] == "completed"
                and r["ttft_s"] is not None
                and r["ttft_s"] <= 2
                and (r["mean_tpot_s"] is None or r["mean_tpot_s"] <= 0.1)
            )
        manifest.update(
            status="PASS",
            completed=sum(r["terminal"] == "completed" for r in records.values()),
            cancelled=sum(r["terminal"] == "cancelled" for r in records.values()),
            output_tokens=sum(len(r["tokens"]) for r in records.values()),
            slo_completed=sum(r["slo_met"] for r in records.values()),
            measured_seconds=sum(w["duration_s"] for w in wave_results),
            assertions="stream prefix, cap, forced length when requested, termination, idle reclamation, selected lifecycle coverage; allocator assertions in scheduler",
        )
    except BaseException as exc:
        manifest.update(status="FAIL", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        manifest.update(wall_seconds=time.monotonic() - started, finished_unix=time.time(), waves=wave_results)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (out / "requests.json").write_text(json.dumps(list(records.values()), indent=2))
        (out / "lifecycle.json").write_text(json.dumps(transitions, indent=2))
        print(
            json.dumps(
                {
                    k: manifest.get(k)
                    for k in ("status", "completed", "cancelled", "output_tokens", "wall_seconds", "error")
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--execute-gpu", action="store_true")
    args = ap.parse_args()
    if not args.execute_gpu:
        ap.error("GPU execution requires --execute-gpu")
    run(json.loads(args.config.read_text(encoding="utf-8")))
