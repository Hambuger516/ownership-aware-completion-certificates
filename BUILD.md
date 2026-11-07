# Build and execution environment

## CPU checks

The CPU implementation and test runner require Python 3.12 or later. Core tests and numerical aggregation use the standard library; 13 neural PATCH tests additionally require optional PyTorch. The bundled vLLM source fixtures support AST compatibility checks without an installed vLLM engine.

```bash
python scripts/run_cpu_tests.py
python analysis/reproduce_numbers.py --output results/reproduced_numbers.json
```

## Scientific figures

```bash
python -m pip install -r requirements-figures.txt
python scripts/reproduce_figures.py
```

Bundled publication figures are in `figs/`. Rebuilt figures, readouts and the build registry are written under `results/figure_build/`, leaving release hashes and frozen measurement/endpoint files unchanged.

The figure recipe uses Matplotlib, NumPy and pypdf. Publication typography uses a licensed local Times New Roman font, which is not redistributed, and Computer Modern mathematics. The compatible STIXGeneral fallback preserves numerical data; main figures gain vertical padding for its font metrics and are marked as fallback builds. It does not assert identical publication typography. Set TSMC_FONT_DIR to a directory containing licensed Times New Roman .ttf files when they are not installed. Neither plotting nor numerical aggregation requires a model or CUDA.

## Recorded GPU environment

The measured stack is Linux on A800-SXM4-80GB, Python 3.12.13, vLLM 0.29.0, PyTorch 2.13.0+cu130, CUDA 13.0 and Transformers 5.17.0. `package/requirements-gpu.lock.txt` records the observed environment, including packages not required by the controller; it is not a minimal installation specification.

For a new deployment, follow the dependencies' official installation instructions and run `PYTHONPATH=package python -m prc_nc doctor --gpu`. Software/API compatibility is checked separately from GPU-kernel execution, memory isolation and experimental resource qualification. Model acquisition, GPU execution and resource capture are explicit experiment steps.
