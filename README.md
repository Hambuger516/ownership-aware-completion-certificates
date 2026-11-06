# Ownership-Aware Completion Certificates

Research implementation and analysis-ready measurements accompanying **Ownership-Aware Completion Certificates for Autoregressive Serving** by Hanbo Ma, Zhongguo Li, Junan Wang, and Zhengtao Ding.

The controller certifies finite retained-state cohorts using joint last-owner release, maintains a completion witness during growth, and charges unsettled execution rights. The repository also includes cap-aware admission, matched baseline profiles, and authorization-preserving zero-page projection reuse. The guarantees concern finite retained-state cohorts.

## Quick start (CPU only)

Python 3.12 or later is sufficient for the core tests and numerical reproduction; no model or GPU is loaded by these commands.

```bash
python scripts/run_cpu_tests.py
python analysis/reproduce_numbers.py --output results/reproduced_numbers.json
python scripts/verify_release.py
```

For the figure recipes, install the optional dependencies and a properly licensed Times New Roman font for the publication typography:

```bash
python -m pip install -r requirements-figures.txt
python scripts/reproduce_figures.py
```

The current main matrices show geometric means only, on a shared equality-centred ratio scale from 1/64 to 4. All four paired observations and observed minimum/maximum ranges remain in [the companion CSVs](source_data/v24_checks/matrices/README.txt).

Computer Modern is used for mathematics. A machine without Times New Roman can produce fallback-font figures, which do not claim identical typography.

## Repository contents

| Path | Contents |
|---|---|
| `package/prc_nc/` | Actual serving controller, allocation authority, ownership accounting, execution measurement and vLLM adapter |
| `package/tests/`, `verification/` | Existing CPU correctness, lifecycle, cap/headroom, reuse and native-queue controls |
| `benchmark/` | Exact tokenized inputs, frozen configuration templates, profiles and source/population map |
| `source_data/` | Analysis-ready same-block measurements and exact derived figure trajectories |
| `figure_source/` | Rebuildable scientific figure recipes; live-text SVG/vector PDF outputs |
| `analysis/` | Paired-log aggregation and runtime-accounting reproduction |
| `provenance/` | File hashes, source comparison and public-export transformations |

The unoptimized original implementation and the projection-reuse implementation are distinct source variants. Only the adapter differs between their 84 Python implementation/test files; `scripts/materialize_runtime.py` reconstructs either variant. The original fair-baseline, ownership-intervention and 14B results must not be attributed to the optimized variant. See [SOURCE_MAP.md](SOURCE_MAP.md).

## What the measurements show

The original deployment reduces cohort time against the conservative Banker and hard-reservation implementations on the two matched tight-quota baseline scenarios. It remains slower than Native in whole-cohort time. Brief-request benefits are input-conditional: instruction inputs and wider quotas do not show uniform gains. On separately frozen fresh inputs at fourfold quota, reuse reduces time relative to the same-block off implementation, while retaining approximately 10.2% overhead to Native. All adverse outcomes remain in the measurement tables.

The measurements use constructed closed cohorts on homogeneous A800 GPUs. Executed-position counts measure repeated model work; fixed-input process repetitions describe process variation.

## Reproduce and extend

[REPRODUCE.md](REPRODUCE.md) describes each executable layer and its limits. [BUILD.md](BUILD.md) records the observed software environment. `benchmark/make_plan.py` creates new portable local configurations without starting a GPU; a new run is not a replay of the original resource qualification.

The public release includes implementation, analysis-ready paired measurements and figure sources. Large original allocator/event/telemetry archives and model weights are **not included**. The original raw archives are retained locally by the authors; no public raw-archive download is asserted here. [RAW_AUDIT_SCOPE.md](RAW_AUDIT_SCOPE.md) distinguishes derived-number reproduction from independent physical/raw validation.

## Data and rights

Author implementation is source-available; no general open-source license has been granted for it. See [LICENSE](LICENSE). Databricks Dolly-derived data remains CC BY-SA 3.0; Stanford Alpaca-derived inputs remain CC BY-NC 4.0. The two vendored vLLM reference files retain Apache-2.0 notices. Models and other external packages retain their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

[DISCLOSURE.md](DISCLOSURE.md) describes AI assistance. The authors remain responsible for the implementation, verification and interpretation.
