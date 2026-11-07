# Measured source and evidence populations

| Evidence population | Runtime variant and contrast | Measurement location |
|---|---|---|
| Repeated 7B ownership intervention | `original`; both full C0/C1 views are computed, selected view differs | `source_data/v18_verified/ownership/` |
| 7B standard baselines | `original`; Adaptive, Banker C0, hard reservation and Native | `source_data/v18_verified/baselines/` |
| 14B transfer | `original`; INC, Fixed, Adaptive and model-specific Native | `source_data/v18_verified/scale14b/` |
| Frozen eight-scenario confirmation | `original`; INC, Fixed, Adaptive and Native | `source_data/v17_verified/` |
| Admission sensitivity | Original fixed-policy development configurations | `source_data/v16_verified/` |
| Ready-growth width | Settled cuts from the base-quota functional audits | `source_data/v19_closeout/width/` |
| Projection reuse | `projection`; off/on flag and same-block Native in two separate populations | `source_data/v19_projection/` |
| Bounded-cap perturbation | `projection` candidate with reuse disabled; inactive peer-gate contrast | `source_data/v20_caps/` |
| Runtime accounting | Recorded fourfold-quota scheduling intervals within the reuse populations | `source_data/runtime_accounting/` |

The original measured deployment and projection-reuse variant differ in `prc_nc/adapters/vllm_scheduler.py`. The original adapter is retained in `verification/reference_adapter.py`; the source hashes and exact adapter patch are in `provenance/`. `scripts/materialize_runtime.py` exports the selected variant. Original baseline, ownership and 14B measurements remain attributed to the original deployment.

The public runtime preserves executable controller logic. `provenance/FROZEN_PYTHON_SOURCE.json` retains the 84-file measured-package hash baseline. `provenance/PORTABLE_TEST_RELEASE.json` separately records the three test fixture-path changes, one module-docstring correction and added test-only resolver. The current package contains 85 Python files, with 80 of the original 84 byte-identical. These portability adjustments do not represent new GPU measurements or a changed serving algorithm. The internal `prc_nc` namespace is retained for compatibility with the measured implementation and configuration records.

The seven current experimental figures are indexed in `provenance/PAPER_FIGURE_REGISTRY.json`. Their sources and endpoint checks define the plotting scope of this artifact; measurement file dates and internal study identifiers identify populations, not manuscript versions.
