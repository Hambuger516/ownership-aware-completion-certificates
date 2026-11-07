# Artifact verification checklist

- [ ] Check the candidate file inventory, provenance mappings and recorded-source attribution.
- [ ] Run CPU tests in an unrelated directory with an empty Python 3.12+ environment; report passes and optional skips separately.
- [ ] Recompute numerical comparisons from the included paired service records.
- [ ] Rebuild exactly four main-paper and three supplementary experimental figures, then verify endpoint equivalence, vector output and the build registry under `results/figure_build/`.
- [ ] Verify release hashes and scan for private paths, hosts, credentials and model weights.
- [ ] Retain source/data license notices and the original/projection runtime distinction.
- [ ] Publish the verified commit, record its full SHA in the manuscript URL, and verify public code/data access at that commit. Preserve existing release tags.
- [ ] Keep the complete raw-event/telemetry access statement consistent with the supplement.
