# Data provenance and interpretation

The analysis-ready tables preserve numeric measurements, case identities, caps, method labels and process-pair keys. Private host/device identifiers and deployment paths have been removed or consistently pseudonymized. File hashes and release transformations are recorded in `provenance/`.

## Selected inputs

The bundled Dolly sample was drawn without replacement with seed 20260917 from 15,011 source records, then normalized to instruction/input/reference-response fields. Source URL, source SHA-256 and selection rules are recorded in `package/prc_nc/data/provenance.json`. Reference responses do not select winning methods.

`benchmark/inputs/` and `benchmark/studies/` contain selected tokenized inputs, hard caps and constructed arrival schedules. Shared-prefix documents derive from Dolly (CC BY-SA 3.0); distinct instruction inputs derive from Alpaca (CC BY-NC 4.0). Generated reference answers were not used for input selection. The frozen selection and exclusion records retain the identities applicable to each confirmation population. The eight-scenario confirmation and projection-reuse confirmation have separate selections; their exclusion sets must not be interchanged. No private-user prompts or generated responses are included.

The bounded-cap perturbation retains one registered cap draw with an inactive peer gate. It describes that control population rather than independent draws from a general cap distribution.

## Statistical units

A process ratio pairs methods within the same workload/configuration block. The ownership, standard-baseline and model-transfer studies use four repetitions per cell; projection-reuse studies use six. Ratios are formed within blocks, aggregated within the stated input family, and then combined with equal family weight where specified. Brief completion is the arithmetic mean for the two designated brief requests within a run.

Observed ranges and empirical quartiles describe the recorded process observations on fixed inputs. Conditional bootstrap summaries have the scope stated in the supplement. Snapshot cuts within a run are dependent audit observations.

## Outputs, work and exclusions

Exact output equality concerns complete model-token sequences in the original audits, separately from token counts. The matched datasets retain their exact-output fields. Controller repeated-position counts and Native reexecution remain visible in the recorded tables. All comparison cells, including unfavorable results, are retained.

The fresh-input reuse population contains 87 attempted services and 84 qualified services. One whole three-arm block was resource-disqualified and replaced under the recorded resource-only continuation. Stage populations remain separate. Complete event/telemetry originals are outside this compact artifact; their access boundary is specified in [RAW_AUDIT_SCOPE.md](RAW_AUDIT_SCOPE.md).
