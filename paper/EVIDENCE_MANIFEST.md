# Public Evidence Manifest

## DYN-15: Main Tracking Evidence

- Scope: Semantic, GDOR, and Strict on TUM `walking_xyz` and six Bonn dynamic
  sequences, seeds 0/1/2.
- Completion: 63/63 runs.
- Public aggregate: `data/dyn15_tracking_per_sequence.csv`.
- Supported result: GDOR lowers All7 mean ATE from 11.331 cm to 3.798 cm and has
  lower mean ATE than Semantic on all seven declared sequences.

## DYN-18: Frozen TUM4 Validation

- Scope: four TUM sequences, three configurations, seeds 0/1/2.
- Completion: 36/36 cells after one exact Semantic repair run.
- Public cells: `data/dyn18_tum4_cells.csv`.
- Supported result: frozen GDOR improves all four sequence means and 9/12 paired
  cells relative to the same-source Semantic parent.
- Boundary: published DyPho-SLAM values are external reports; GDOR does not beat
  them on this local matrix.

## DYN-17: Map-Matched No-Track-Reuse Control

- Scope: TUM `walking_xyz`, Bonn `crowd2`, and Bonn `person_tracking2`, three
  seeds each.
- Completion: 9/9 runs.
- Public aggregate: `data/dyn17_mapmatched_control.csv`.
- Supported result: with persistent mapping weight held identical, the
  raw-mask no-track-reuse control does not match GDOR tracking performance on
  the tested subset.
- Boundary: this control isolates the combined temporal-recovery and adaptive
  tracking-replenishment path, not each cue separately.

## DYN-16: Common-View Mapping Diagnostic

- Scope: TUM `walking_xyz` and Bonn `crowd`, seed 0, 20 shared non-keyframe views
  per sequence.
- Public aggregate: `data/dyn16_commonview_mapping.csv`.
- Supported result: GDOR improves the declared online static-region rendering
  diagnostic over Semantic.
- Boundary: this is a single-seed diagnostic, not a multi-seed mapping
  superiority result.

## Provenance And Reproduction

- `reports/generated_figure_provenance.json` records SHA-256 hashes for figure
  inputs and generated outputs.
- `scripts/validate_gdor_tmm_evidence.py` recomputes the claim-bearing aggregate
  values from the public CSV files.
- Artifact URIs in the CSV files are stable logical identifiers for retained
  private experiment roots. They are not public URLs and do not imply that raw
  datasets or full generated maps are redistributed.

## Global Non-Claims

- No official DyPho-SLAM reproduction.
- No protocol-matched external SOTA ranking.
- No universal sequence or metric superiority.
- No multi-seed mapping-superiority certificate.
- No photorealistic-superiority claim.
- No archived empirical zero-overlap certificate for DYN-15 through DYN-18.
