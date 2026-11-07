# Reproduction levels

## 1. CPU properties

`python scripts/run_cpu_tests.py` runs the original core tests and additional existing adapter-AST controls. They cover closure, margins, allocation rights, settlement, observation joins, cap/headroom truth tables, state reuse and native queue transfer. The neural PATCH tests require optional PyTorch; the runner preserves and reports their skips when PyTorch is absent. A skip is not a pass. The protocol probe binds only a localhost socket, so a sandbox that prohibits loopback binding cannot execute that test.

The entry point requires Python 3.12 or later and uses absolute checkout paths,
including when called from another working directory. Both suites run even if
the first reports failures, and the exit status remains nonzero if either suite
fails. `CPU_TEST_SUMMARY` records test/error/failure/skip counts and the Python
version. No installed copy of `prc_nc` is accepted in place of this checkout.

## 2. Published numerical comparisons

`python analysis/reproduce_numbers.py --output results/reproduced_numbers.json` recomputes matched geometric means from the included analysis-ready per-process tables. This is independent arithmetic on published rows, not a new replay of the absent physical event archives. It retains the Native gap and unfavorable cells.

## 3. Figures

`python scripts/reproduce_figures.py` rebuilds the scientific PDF/SVG/PNG figures using measured data and exact derived trajectories. It does not run an algorithm or generate observations. Figure metadata/output hashes are newly generated build artifacts. The compact two-worker timeline uses exact derived client-observation times and cumulative counts rather than reconstructed or invented raw events. The independent appendix figures keep their historical populations.

The current two main matrices can also be rebuilt alone with `python figure_source/redesign_main_matrices_v24.py --manuscript-root .`. Their 60 endpoint groups and 240 paired ratios are retained in `source_data/v24_checks/matrices/`; observed ranges describe the four process repetitions and are not confidence intervals.

## 4. New GPU runs

`python scripts/materialize_runtime.py --variant original --output runtime_original` reconstructs the unoptimized serving source. `--variant projection` reconstructs the zero-page reuse source. Do not use the optimized source to retroactively attribute the original fair-baseline or 14B timings.

`python benchmark/make_plan.py --study projection-confirmation --model /your/local/model --output new_plan` prepares the original same-input profile and balanced order without starting a GPU. The output is explicitly a new unexecuted plan. Model acquisition and GPU execution are opt-in subsequent operations under the frozen adapter's `--execute-gpu` interface. No private cluster addresses, credentials or SSH dispatcher are supplied.

A strict reproduction additionally needs the matched model bytes, tokenizer/weight hashes, resource/telemetry capture, terminal allocator/rights checks, exact outputs and source binding described in the supplement. Neither a new plan nor the bundled derived tables proves those conditions for a new run. Current raw-archive availability is detailed in RAW_AUDIT_SCOPE.md.
