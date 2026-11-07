# Reproduction guide

## 1. CPU correctness

```bash
python scripts/run_cpu_tests.py
```

Python 3.12 or later is required. Both suites run in separate interpreters and report their outcomes in `CPU_TEST_SUMMARY`; the command exits unsuccessfully if either suite fails. The checkout supplies the implementation and all five hash-pinned vLLM AST fixtures. No vLLM installation, model, GPU or network access is required. The protocol test needs permission to bind a localhost socket. The 13 neural PATCH tests require optional PyTorch and remain reported as skips when it is absent; they are separate from the native-source fixtures.

For an independent check, place the artifact in a directory outside any existing project tree and select the exact commit identified by the manuscript. Replace `COMMIT_FROM_PAPER` below with the full commit SHA in the printed repository URL:

```bash
git clone https://github.com/Hambuger516/ownership-aware-completion-certificates.git /tmp/tsmc-reproduction/repo
git -C /tmp/tsmc-reproduction/repo checkout --detach COMMIT_FROM_PAPER
python3.12 -m venv /tmp/tsmc-reproduction/venv
cd /tmp
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 /tmp/tsmc-reproduction/venv/bin/python /tmp/tsmc-reproduction/repo/scripts/run_cpu_tests.py
```

The unchanged CPU suites discover 363 package tests and 32 verification tests. Without PyTorch, 382 pass and 13 skip. The fixture resolver fails on missing or altered bundled source rather than searching outside the checkout.

## 2. Numerical comparisons

```bash
python analysis/reproduce_numbers.py --output results/reproduced_numbers.json
```

The script recomputes ownership ratios, baseline comparisons, projection-reuse comparisons and recorded-time accounting from the included process tables. Within-block ratios, scenario/input-family weighting and source populations are retained. This check verifies arithmetic on analysis-ready measurements; event-level qualification additionally uses the original raw archives described in [RAW_AUDIT_SCOPE.md](RAW_AUDIT_SCOPE.md).

## 3. Seven current experimental figures

```bash
python -m pip install -r requirements-figures.txt
python scripts/reproduce_figures.py
```

The default figure entry point rebuilds these seven PDF/SVG/PNG figures under `results/figure_build/figures/`. It writes generated readouts to `results/figure_build/main/` and `results/figure_build/supplementary/`, and the build registry to `results/figure_build/registry.json`. Bundled publication outputs in `figs/` and frozen expected endpoints in `source_data/plot_endpoints/` remain unchanged.

| Paper location | Figure stem | Measurement population |
|---|---|---|
| Main Fig. 3 | `fig_ownership_repeated_v30` | Repeated full-C1/full-C0 ownership intervention |
| Main Fig. 4 | `fig_fair_baselines_v30` | Matched 7B standard baselines |
| Main Fig. 5 | `fig_scale14b_v30` | Matched 14B model transfer |
| Main Fig. 6 | `fig_runtime_ledger_v30` | Fresh-input fourfold-quota runtime ledger |
| Supplement Fig. S1 | `fig_sensitivity_v16_v22` | Six-level admission sensitivity |
| Supplement Fig. S2 | `fig_width_ready_v19_v22` | Ready-growth width audit |
| Supplement Fig. S3 | `fig_projection_reuse_v19_v22` | Separate development and fresh-input reuse populations |

The registry in `provenance/PAPER_FIGURE_REGISTRY.json` binds the bundled publication figures to their measurement inputs; `results/figure_build/registry.json` records the new build and its endpoint comparisons. Endpoint comparisons retain process pairing, all displayed observations and the recorded summary statistics. Rendering is a presentation step and does not create experimental observations. Main matrices share the `(1/64, 1, 4)` colour range. The optional licensed Times New Roman font enables publication typography; a fallback-font build should be identified as such.

## 4. New GPU runs

```bash
python scripts/materialize_runtime.py --variant original --output runtime_original
python scripts/materialize_runtime.py --variant projection --output runtime_projection
python benchmark/make_plan.py --study projection-confirmation --model /your/local/model --output new_plan
```

Runtime materialization preserves the original/projection distinction. The original baseline, ownership and 14B timings belong to the original variant. Reuse timings belong to the projection variant, with their recorded off/on flags. The export contains the preserved test helper; running exported tests also requires the repository's bundled reference directory.

`make_plan.py` prepares new configurations and balanced method order without starting a GPU. Model acquisition and GPU execution are explicit subsequent operations through `--execute-gpu`. An independent experiment must also match the model/tokenizer bytes, caps, quotas, source variant, observation settings, exact outputs and resource/terminal checks specified in the supplement. A newly executed plan is a new measurement population.

## 5. File integrity

```bash
python scripts/verify_release.py
```

This command checks the release file hashes and scans the listed text files for private deployment paths, hosts, keys and credential tokens. It complements the CPU, numerical and figure checks.
