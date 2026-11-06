# Reproduction levels

## 1. CPU properties

`python scripts/run_cpu_tests.py` runs the original core tests and additional existing adapter-AST controls. They cover closure, margins, allocation rights, settlement, observation joins, cap/headroom truth tables, state reuse and native queue transfer. The 13 tests requiring the installed vendor runtime are skipped when that runtime is absent; a skip is not a pass.

## 2. Published numerical comparisons

`python analysis/reproduce_numbers.py --output results/reproduced_numbers.json` recomputes matched geometric means from the included analysis-ready per-process tables. This is independent arithmetic on published rows, not a new replay of the absent physical event archives. It retains the Native gap and unfavorable cells.

## 3. Figures

`python scripts/reproduce_figures.py` rebuilds the scientific PDF/SVG/PNG figures using measured data and exact derived trajectories. It does not run an algorithm or generate observations. Figure metadata/output hashes are newly generated build artifacts. The compact two-worker timeline uses exact derived client-observation times and cumulative counts rather than reconstructed or invented raw events. The independent appendix figures keep their historical populations.

## 4. New GPU runs

`python scripts/materialize_runtime.py --variant original --output runtime_original` reconstructs the unoptimized serving source. `--variant projection` reconstructs the zero-page reuse source. Do not use the optimized source to retroactively attribute the original fair-baseline or 14B timings.

`python benchmark/make_plan.py --study projection-confirmation --model /your/local/model --output new_plan` prepares the original same-input profile and balanced order without starting a GPU. The output is explicitly a new unexecuted plan. Model acquisition and GPU execution are opt-in subsequent operations under the frozen adapter's `--execute-gpu` interface. No private cluster addresses, credentials or SSH dispatcher are supplied.

A strict reproduction additionally needs the matched model bytes, tokenizer/weight hashes, resource/telemetry capture, terminal allocator/rights checks, exact outputs and source binding described in the supplement. Neither a new plan nor the bundled derived tables proves those conditions for a new run. Current raw-archive availability is detailed in RAW_AUDIT_SCOPE.md.
