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

All 84 Python implementation/test files in the bundled `package` match the selected projection package byte for byte. The original fair-baseline/ownership/14B packages have exactly one different Python file: `prc_nc/adapters/vllm_scheduler.py`. Its original bytes are retained in `verification/reference_adapter.py`, alongside a unified patch and hash manifest in `provenance/`. No old source is silently relabeled as a measured new implementation. Historical source-development logs and private deployment metadata are intentionally not copied into the public-facing repository.
