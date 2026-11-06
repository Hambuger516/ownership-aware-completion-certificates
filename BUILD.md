# Build and environment

CPU implementation/tests: Python >=3.12, standard library only. The public staging validation used Python 3.13; the measured GPU environment used Python 3.12.13.

```bash
python -m pip install -e package
python scripts/run_cpu_tests.py
```

Optional figures: `python -m pip install -r requirements-figures.txt`. Exact publication typography additionally requires a licensed local Times New Roman font; it is not redistributed. Figure data reproduction does not need a model or CUDA.

Measured GPU stack: Linux, A800-SXM4-80GB, Python 3.12.13, vLLM 0.29.0, PyTorch 2.13.0+cu130, CUDA 13.0, Transformers 5.17.0. `package/requirements-gpu.lock.txt` records the complete observed environment, including packages not required by the controller. It is an environment record, not a minimal installer. Start from the GPU stack's official installation instructions, then check `PYTHONPATH=package python -m prc_nc doctor --gpu`. A successful software check is not a GPU-kernel, memory-isolation or experimental qualification test.

The runtime validates its supported engine API and rejects incompatible configurations. No new GPU execution, package installation on a GPU host, weight download or deployment was performed while preparing this release.
