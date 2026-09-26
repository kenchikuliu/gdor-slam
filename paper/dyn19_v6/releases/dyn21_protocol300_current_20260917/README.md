# DYN-21 Protocol-300 Current-Method Release

This directory is the Git-tracked release package for the completed
`DYN21_PROTOCOL300_CURRENT_20260917` comparison.

## Contents

- `comparison/index.html`: self-contained report entry point.
- `comparison/`: HTML, real comparison panels, tables, metrics, provenance,
  and validation screenshots.
- `run_artifacts/`: the complete non-comparison snapshot of the original run
  root, including all four run directories, logs, status files, trajectories,
  metrics, held-out renders, Gaussian maps, camera files, and PLY files.
- `run_contract/`: frozen plan, upstream protocol, campaign status, and the
  SHA-256 manifest used to verify `run_artifacts/`.
- `SHA256SUMS`: SHA-256 manifest for the Git-tracked comparison package.
- `TRACEABILITY.json`: machine-readable source, run, and artifact linkage.

## Complete snapshot

The source code, runner, builder, frozen plan, protocol, result summaries,
comparison package, and all files from the completed DYN-21 result root are
Git-tracked in this release. The original NAS path is retained as the provenance
source, and the release contains a complete content snapshot under
`run_artifacts/`.

Validate the complete snapshot with:

```bash
sha256sum -c comparison/checksums.sha256
(cd run_artifacts && sha256sum -c ../run_contract/raw_run_SHA256SUMS)
```

The related `DynaGS-SLAM` mapping-work commit `7d50616` is published separately
on the branch recorded in `TRACEABILITY.json`. The authoritative DYN-21 runner
and comparison registration are from the `gdor-slam` commits listed there.
