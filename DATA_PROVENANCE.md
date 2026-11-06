# Data provenance and interpretation

The public tables contain matched measured processes, plus exact figure-level derived records. Private host/device names and deployment paths are removed or consistently pseudonymized. Numeric measurements, comparisons, case identities, caps and process-pair keys are preserved. Original/public file hashes and transformations are recorded in `provenance/`.

## Inputs

The bundled Dolly sample was drawn without replacement with seed 20260917 from 15,011 source records, then normalized to instruction/input/reference-response fields. Its source URL, original source SHA256 and selection rule remain in `package/prc_nc/data/provenance.json`. Reference responses are not used to choose winning methods.

`benchmark/inputs/` contains the original selected model-token inputs, caps and constructed arrival times. Prefix documents derive from Dolly (CC BY-SA 3.0). Distinct instruction inputs derive from Alpaca (CC BY-NC 4.0); generated reference answers were not used for input selection. Input IDs are constructed study IDs, not user identities. These are constructed evaluation cohorts, not production traffic. The confirmation selection records exclusion of 518 earlier fingerprints; the original frozen exclusion set is retained in `benchmark/input_exclusions.json`. No prompt or generated response from a private user conversation is included.

The cap perturbation is a single retained inactive-peer negative control, not evidence of robustness over a population of cap distributions.

## Statistical units

A process ratio pairs the same workload/configuration block. The original studies have four process repetitions per cell, and reuse studies have six. Form log ratios within blocks, then average within the specified input family and weight the families equally. Do not replace family weighting by a direct average of unequal numbers of scenarios. Brief completion is each run's arithmetic mean across the two designated brief requests, not an independent sample of two runs.

Observed ranges and empirical quartiles describe the retained fixed-input process observations. Sample SD is labeled only where defined. The posthoc conditional bootstrap does not imply workload-population or simultaneous coverage. Snapshot cuts within a run are dependent observations.

## Output/work and exclusions

Output equality means exact model-token sequences in the original audit, separately from token counts. The current matched datasets retain exact-output fields. Older legacy comparisons with sequence differences are retained and explicitly scoped. Zero repeated positions is a measured controller outcome in qualified runs, not a latency guarantee. Native's legal reexecution remains visible.

Resource-disqualified attempts are not silently relabeled as valid. The fresh-input reuse study originally recorded 87 actual services, of which 84 qualify; a whole three-arm block was excluded for resource interference/initialization failure and replaced only under the recorded resource-only continuation. Stage timing populations are never pooled. The full event/telemetry originals are outside this compact public release; see RAW_AUDIT_SCOPE.md.
