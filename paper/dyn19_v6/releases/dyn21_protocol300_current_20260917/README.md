# DYN-21 Protocol-300 Current-Method Release

This directory is the Git-tracked release package for the completed
`DYN21_PROTOCOL300_CURRENT_20260917` comparison.

## Contents

- `comparison/index.html`: self-contained report entry point.
- `comparison/`: HTML, real comparison panels, tables, metrics, provenance,
  and validation screenshots.
- `run_contract/`: frozen plan, upstream protocol, campaign status, and a
  SHA-256 manifest for every file in the original NAS run root except the
  copied comparison package.
- `SHA256SUMS`: SHA-256 manifest for the Git-tracked comparison package.
- `TRACEABILITY.json`: machine-readable source, run, and artifact linkage.

## Reproduction boundary

The source code, runner, builder, frozen plan, protocol, result summaries, and
published comparison package are Git-tracked. Dataset files, generated Gaussian
maps, trajectories, logs, and other large run outputs remain in the authoritative
NAS result root recorded in `TRACEABILITY.json`; their relative paths and hashes
are recorded in `run_contract/raw_run_SHA256SUMS`.

The raw run root must exist before validating that manifest:

```text
/mnt/nas_datasets/slam-experiments/DynaGS-SLAM/dyn21_protocol300_current_20260917_run01
```

The related `DynaGS-SLAM` mapping-work commit `7d50616` is published separately
on the branch recorded in `TRACEABILITY.json`. The authoritative DYN-21 runner
and comparison registration are from the `gdor-slam` commits listed there.
