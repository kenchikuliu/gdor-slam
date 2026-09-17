# DYN-21 Current-Method Protocol-300 Comparison

Frozen: September 17, 2026  
Experiment ID: `DYN-21_PROTOCOL300_CURRENT_20260917`

## Purpose

This experiment fills the one remaining FlowParse-minimum comparison cell:
current `dyn19_full` under the already frozen local Protocol-300 package. It
adds one current-method row to the existing SplaTAM, Photo-SLAM, and DynaGS
semantic evidence without modifying or relabeling the historical failed-boundary
`Ours` row.

The machine-readable plan is `DYN21_PROTOCOL300_CURRENT_PLAN.yaml`.

## Frozen Matrix

The denominator is exactly four real runs:

```text
dyn19_full x seed 0 x
  tum_walking_xyz
  tum_walking_halfsphere
  bonn_person_tracking
  bonn_crowd3
```

Every run consumes observations `[0, 300)`, uses the same DYN-19 Full binary
and runtime-library hashes as the registered DYN-19 main experiment, disables
realtime playback and the viewer, and records held-out candidates at stride 20.
Two GPU workers execute serially within their assigned queue.

## Evidence Contract

Tracking uses unique one-to-one timestamp association with a maximum absolute
difference of 0.02 seconds and fixed-scale SE(3) alignment. Mapping uses the
four already frozen Fig. 4 frames, public dataset GT poses mapped into each
final Gaussian-map frame at scale 1.0, 640x480 output, no crop, and no image
post-processing. The selected render pose is excluded from alignment support.

PSNR and SSIM remain full-frame diagnostics because dynamic foreground is
included. Runtime is end-to-end wall time and is not decomposed into tracking
and mapping components.

## Retention And Immutability

The two historical Protocol-300 roots are read-only inputs:

```text
/home/slam/experiments/dypho_skill_full_rerun_20260729/baseline_protocol_300/
/media/slam/My Passport/experiments/dypho_skill_full_rerun_20260729/
```

All new artifacts are written to:

```text
/mnt/nas_datasets/slam-experiments/DynaGS-SLAM/
  dyn21_protocol300_current_20260917_run01/
```

The runner refuses to start if that result root already exists.

## Execution

From a clean committed source tree:

```bash
scripts/run_dyn21_protocol300_current.sh
```

After the four run cells pass their artifact checks, the comparison builder
produces the tracking, mapping, runtime, provenance, and Fig. 4 package in the
same new result root.
