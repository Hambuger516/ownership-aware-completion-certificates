# Ownership-Aware Completion Control for LLM Serving with Shared KV Memory

Artifact version **1.2.0** accompanies the TSMC manuscript by Hanbo Ma, Zhongguo Li, Junan Wang, and Zhengtao Ding.

The implementation certifies finite retained-state request cohorts using joint last-owner release, maintains a completion witness during growth, and accounts for unsettled execution rights. The artifact includes cap-aware admission, matched baseline profiles, projection reuse, CPU correctness tests, and the analysis-ready measurements supporting the current main paper and supplement.

## CPU and numerical reproduction

Use Python 3.12 or later from the repository root:

```bash
python scripts/run_cpu_tests.py
python analysis/reproduce_numbers.py --output results/reproduced_numbers.json
python scripts/verify_release.py
```

The CPU runner loads this checkout and reports both suites in `CPU_TEST_SUMMARY`. With no PyTorch installed, its 395 tests comprise 382 passes and 13 optional-PyTorch skips. Native-source AST fixtures are bundled and hash-pinned in [native_runtime_reference/MANIFEST.json](native_runtime_reference/MANIFEST.json). The protocol test binds a localhost socket. [REPRODUCE.md](REPRODUCE.md) gives the independent-directory procedure and the scope of each check.

## Current paper figures

```bash
python -m pip install -r requirements-figures.txt
python scripts/reproduce_figures.py
```

The bundled publication figures are in `figs/`. Rebuilds write figures to `results/figure_build/figures/`, generated readouts to `results/figure_build/main/` and `results/figure_build/supplementary/`, and a build registry to `results/figure_build/registry.json`. Frozen expected endpoints are retained in `source_data/plot_endpoints/`; rebuilding leaves the bundled figures and measurement files unchanged.

The default rebuild produces exactly **seven** experimental figures: four main-paper figures for ownership, standard baselines, 14B transfer and runtime accounting; and three supplementary figures for admission sensitivity, ready-growth width and projection reuse. [provenance/PAPER_FIGURE_REGISTRY.json](provenance/PAPER_FIGURE_REGISTRY.json) maps each figure to its measurement inputs and endpoint checks.

The two main matrices share the equality-centred ratio scale `(1/64, 1, 4)`. Their cells show geometric means; the accompanying measurements retain every process pair and its observed range. Ratios are recomputed from measured service endpoints before plotting. Publication typography uses a licensed local Times New Roman font and Computer Modern mathematics. A fallback-font rebuild reproduces the data but does not assert identical typography. Times New Roman is not distributed.

## Artifact contents

| Path | Contents |
|---|---|
| `package/prc_nc/` | Serving controller, allocation authority, ownership accounting, execution measurement and vLLM adapter |
| `package/tests/`, `verification/` | CPU correctness and native-source compatibility controls |
| `benchmark/` | Tokenized inputs, configuration templates, profiles and source/population records |
| `source_data/` | Matched service measurements and current figure records |
| `figure_source/` | Rebuildable sources for the seven experimental figures |
| `analysis/` | Numerical aggregation and recorded-time accounting |
| `provenance/` | Release hashes, measured-source mapping and figure endpoint checks |

`prc_nc` is the preserved internal compatibility namespace. The original deployment and projection-reuse implementation are distinct measured source variants. Their relationship and the applicable measurements are specified in [SOURCE_MAP.md](SOURCE_MAP.md); `scripts/materialize_runtime.py` exports either runtime variant.

## Measurement scope and access

The experiments use constructed closed cohorts on homogeneous A800 GPUs. The original deployment improves cohort time against Banker C0 and hard reservation in the matched tight-quota scenarios. Brief-request benefits depend on input and quota; whole-cohort time remains longer than Native. Projection reuse is evaluated separately on frozen inputs. The tables retain all measured outcomes, including unfavorable comparisons.

This compact artifact supplies source, selected tokenized inputs, analysis-ready paired measurements and figure records. Complete raw allocator/event/telemetry archives are scheduled for release upon acceptance; model weights are acquired separately. [RAW_AUDIT_SCOPE.md](RAW_AUDIT_SCOPE.md) distinguishes public numerical reproduction from original event-level replay.

## Rights and attribution

Author code is source-available under [LICENSE](LICENSE). Dolly-derived data retains CC BY-SA 3.0; Alpaca-derived inputs retain CC BY-NC 4.0. The five complete vLLM reference files retain Apache-2.0 notices and the exact upstream license. Model and external dependency terms remain applicable. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [DATA_PROVENANCE.md](DATA_PROVENANCE.md), and [DISCLOSURE.md](DISCLOSURE.md).
