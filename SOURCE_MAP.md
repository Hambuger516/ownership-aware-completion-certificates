# Source and timing population map

| Timing population | Runtime variant | Relevant data |
|---|---|---|
| Original 7B ownership intervention | `original`; both full C0/C1 views paid, selected view differs | `source_data/v18_verified/ownership/` |
| Original 7B fair baselines | `original`; actual production C0, hard reservation, Native, Adaptive profiles | `source_data/v18_verified/baselines/` |
| Original 14B transfer | `original`; fixed policy and model-specific Native | `source_data/v18_verified/scale14b/` |
| Eight-scenario Adaptive confirmation | `original`; INC/Fixed/Adaptive/Native | `source_data/v17_verified/` |
| Projection development and fresh-input confirmation | `projection`; reuse off/on flag, same-block Native | `source_data/v19_projection/` |
| Single cap negative control | Projection candidate source with reuse disabled | `source_data/v20_caps/` |
| Earlier diagnostics/hardware/CPU routing | Separate historical studies; not the present GPU source | Figure-specific data and captions |

The 84 Python implementation/test files recorded in `provenance/FROZEN_PYTHON_SOURCE.json` describe the historical measured projection package, not a claim that every current release test is byte-identical. In v1.1.1, 80 of those files remain byte-identical; three test files replace undeclared ancestor-directory lookups with bundled, hash-pinned vLLM fixtures, and `prc_nc/patch_neural.py` changes only its module docstring to name the TSMC artifact. Its executable AST and every other runtime source remain unchanged. The additional `tests/native_source_fixture.py` is a test-only resolver, bringing the current package to 85 Python files. `provenance/PORTABLE_TEST_RELEASE.json` records the exact changes and current hashes; the original 84 hashes are retained.

The original fair-baseline/ownership/14B packages have exactly one different runtime Python file: `prc_nc/adapters/vllm_scheduler.py`. Its original bytes remain in `verification/reference_adapter.py`, alongside the historical unified patch and hash manifest in `provenance/`. `scripts/materialize_runtime.py` retains this runtime-variant distinction; its exported tests use the same bundled-source resolver and need the repository fixture directory when run as tests. No historical source is relabeled as a newly measured implementation. Historical source-development logs and private deployment metadata are not copied into this public-facing TSMC artifact.
