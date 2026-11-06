"""Predeclared experiment configurations only; constructing a matrix runs nothing."""

from .configuration import validate_config, FAULTS


def matrix(suite):
    specs = []
    if suite == "correctness":
        for method in ("native_plain", "native_shadow", "shared_closure", "shared_incremental"):
            specs.append(dict(case="E1_smoke", method=method, count=4, cap=33, ignore_eos=True))
        for block, blocks in ((16, 96), (32, 64)):
            specs.append(
                dict(
                    case=f"E2_boundaries_{block}",
                    count=7,
                    cap_profile="boundaries",
                    prefix=False,
                    block_size=block,
                    blocks=blocks,
                    ignore_eos=True,
                )
            )
        for method in ("shared_closure", "shared_incremental"):
            specs.append(
                dict(
                    case="E3_capacity",
                    method=method,
                    workload="exact_context",
                    count=1,
                    cap=512,
                    blocks=65,
                    max_seqs=1,
                    batch_tokens=64,
                    ignore_eos=True,
                    waves=2,
                    reset_between=True,
                )
            )
        for method in ("native_shadow", "shared_incremental"):
            specs.append(
                dict(
                    case="E4_cancel",
                    method=method,
                    workload="constructed_shared",
                    count=12,
                    cap_profile="mixed",
                    blocks=80,
                    max_seqs=4,
                    arrival_interval=0,
                    ignore_eos=True,
                    waves=2,
                    cancellation=True,
                    busy_reset=True,
                    reset_between=True,
                )
            )
        for fault, error in FAULTS.items():
            specs.append(dict(case="E5_" + fault, count=1, fault=fault, expected_error=error))
        for method in ("shared_closure", "shared_incremental"):
            specs.append(
                dict(
                    case="E6_common_cap", method=method, workload="constructed_shared", count=8, cap=64, ignore_eos=True
                )
            )
        for forced in (True, False):
            for method in ("native_shadow", "shared_closure", "shared_incremental"):
                specs.append(
                    dict(
                        case="E7_" + ("forced" if forced else "eos"),
                        method=method,
                        workload="constructed_shared",
                        count=12,
                        cap_profile="ownership_probe",
                        blocks=65,
                        arrival_interval=0.01,
                        ignore_eos=forced,
                    )
                )
        for method in ("shared_closure", "shared_incremental"):
            specs.append(
                dict(
                    case="E8_chunked",
                    method=method,
                    count=8,
                    min_prompt=256,
                    max_prompt=640,
                    cap_profile="mixed",
                    blocks=96,
                    batch_tokens=64,
                )
            )
    elif suite == "performance":
        for blocks in (80, 512):
            for method in ("native_plain", "shared_closure", "shared_incremental"):
                for replicate in (0, 1):
                    specs.append(
                        dict(
                            case=f"P1_dolly_{blocks}",
                            method=method,
                            count=16,
                            cap_profile="mixed",
                            blocks=blocks,
                            replicate=replicate,
                            audit=False,
                        )
                    )
    elif suite == "drain":
        for method in ("shared_closure", "shared_incremental"):
            for audit in (True, False):
                specs.append(
                    dict(
                        case="D1_drain",
                        method=method,
                        count=8,
                        cap=64,
                        ignore_eos=True,
                        drain_after_steps=1,
                        audit=audit,
                        arrival_interval=0,
                    )
                )
    elif suite == "proposal":
        for proposal_mode in ("exact", "learned"):
            specs.append(
                dict(
                    case="P0_prc_proposal",
                    method="prc",
                    workload="constructed_shared",
                    count=8,
                    cap_profile="mixed",
                    blocks=80,
                    proposal_mode=proposal_mode,
                    ignore_eos=True,
                    drain_after_steps=1,
                    arrival_interval=0,
                )
            )
    elif suite == "effect":
        # Matched first-round comparison: complete PRC (Exact), native vLLM,
        # full hard reservation. Five
        # independent replicates per case use identical workload contracts.
        scenarios = (
            ("EFFECT_dolly_pressure", dict(workload="dolly", count=16, cap_profile="mixed", blocks=80, audit=False, arrival_interval=0.025)),
            ("EFFECT_dolly_roomy", dict(workload="dolly", count=16, cap_profile="mixed", blocks=512, audit=False, arrival_interval=0.025)),
            ("EFFECT_hetero_forced", dict(workload="constructed_shared", count=12, cap_profile="ownership_probe", blocks=65, ignore_eos=True, audit=False, arrival_interval=0.0)),
        )
        for case, common in scenarios:
            for method in ("prc", "native_plain", "hard_reserve"):
                for replicate in range(5):
                    # Keep the same physical input across all three methods.
                    specs.append(dict(case=case, method=method, replicate=replicate, **common))
    elif suite == "full":
        return matrix("correctness") + matrix("drain") + matrix("performance") + matrix("proposal")
    else:
        raise ValueError("unknown suite")
    return [
        validate_config(s | dict(run_id=f"nc_{suite}_{i:02d}_{s['case']}_{s.get('method', 'shared_incremental')}"))
        for i, s in enumerate(specs)
    ]
