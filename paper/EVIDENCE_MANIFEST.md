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
  the tested subset. The companion statistics are a 0.864 cm median sequence
  improvement, 3/0/0 wins/ties/losses, and Bonn `crowd2` as the failure-case
  contrast.
- Boundary: this control isolates the combined temporal-recovery and adaptive
  tracking-replenishment path, not each cue separately.

## DYN-19: Main, Ordered Ablation, And Mapping Diagnostic

- Main scope: Semantic, MapMatched, and Full on ten claim-bearing sequences,
  including TUM `sitting_halfsphere`, with seeds 0/1/2.
- Main completion: 90/90 runs. Public cells: `data/dyn19_main_90_cells.csv`.
- Full versus Semantic: mean ATE 10.119 cm to 4.930 cm; paired-cell median
  gain 0.213 cm; cell W/T/L 19/0/11; sequence-mean W/T/L 7/0/3; Bonn `crowd2`
  failure case 42.500 cm to 11.080 cm.
- Full versus MapMatched: mean ATE 7.392 cm to 4.930 cm; paired-cell median
  gain 0.016 cm; cell W/T/L 16/0/14; sequence-mean W/T/L 4/0/6; TUM
  `sitting_halfsphere` failure case 36.775 cm to 4.864 cm.
- Route certificate: all 30 Full cells have observed recovery and pass, covering
  3,751 recovered-support frames and 79,498,476 audited pixels with zero
  observed overlap against map-admissible support.
- Mechanism completion: 120/120 runs for T+M, T+F+M, T+F+R+M, and Full-NoM.
  Public cells: `data/dyn19_mechanism_120_cells.csv`.
- Ordered-ablation result: the tracking means are non-monotonic. T+M has the
  lowest mean ATE (3.859 cm); adjacent rows are bundle contrasts, not
  independent component effects. All 30 Full-NoM cells fail the route
  certificate by design and contain 90,790,462 overlapping pixels.
- Mapping completion: 72/72 cells over four scenes, seeds 0/1/2,
  Semantic/MapMatched/Full, and online/GT-aligned poses. Public cells:
  `data/dyn19_mapping_72_cells.csv`.
- Online static PSNR versus Semantic: mean gain 0.896 dB, paired median gain
  0.544 dB, W/T/L 9/0/3, and the lowest-Semantic cell at TUM
  `sitting_halfsphere` seed 1 (14.275 dB to 16.551 dB).
- Online static PSNR versus MapMatched: mean gain 0.773 dB, paired median gain
  0.515 dB, W/T/L 11/0/1, and the lowest-MapMatched cell at the same
  sequence/seed (14.009 dB to 16.551 dB).
- Public provenance: `data/dyn19_release_provenance.json`; full trajectories,
  masks, maps, renders, per-frame logs, certificates, plans, and summaries are
  in the public `dyn19-results-20260914` release.
- Boundary: DYN-19 is an ordered ablation, not a factorial design. Its mapping
  completeness and ghost-risk fields are proxies, not independent truth.

## DYN-19: Failure-Recovery-Control Episode Figure

- Scope: TUM `sitting_halfsphere`, seed 0, Semantic / MapMatched / Full from the
  completed DYN-19 main campaign.
- Public aggregate: `data/dyn19_failure_recovery_episode.csv`.
- Supported result: Semantic has one 89-frame failure interval, MapMatched has
  two intervals totaling 174 frames, and Full has no failure interval while
  observing 567 recovered-support frames.
- Route certificate: the Full episode records 11,918,366 audited recovered
  support pixels and zero observed recovered-support/map-admission overlap.
- Boundary: this is a representative route diagnostic, not independent
  ghost-contamination or complete-background truth.

## DYN-16: Common-View Mapping Diagnostic

- Scope: TUM `walking_xyz` and Bonn `crowd`, seed 0, 20 shared non-keyframe views
  per sequence.
- Public aggregate: `data/dyn16_commonview_mapping.csv`.
- Supported result: GDOR improves the declared online static-region rendering
  diagnostic over Semantic.
- Boundary: this is a limited preservation diagnostic with one seed and two
  scenes, not a zero-ghost proof, zero-contamination proof, or multi-seed
  mapping-superiority result.

## Provenance And Reproduction

- `reports/generated_figure_provenance.json` records SHA-256 hashes for figure
  inputs and generated outputs.
- `scripts/import_dyn19_release_metadata.py` imports compact DYN-19 cells from
  the hash-verified public metadata release.
- `scripts/validate_gdor_tmm_evidence.py` recomputes the claim-bearing aggregate
  values, medians, W/T/L counts, certificate totals, and mapping comparisons
  from the public CSV files.
- Artifact URIs in the CSV files are stable logical identifiers for retained
  private experiment roots. They are not public URLs and do not imply that raw
  datasets or full generated maps are redistributed.

## Global Non-Claims

- No official DyPho-SLAM reproduction.
- No protocol-matched external SOTA ranking.
- No universal sequence or metric superiority.
- No photorealistic-superiority claim.
- No archived empirical zero-overlap certificate for DYN-15 through DYN-18.
- No independent zero-ghost or zero-contamination proof.
- No factorial or interaction-effect claim for the DYN-19 ordered ablation.
- No protocol-matched DG-SLAM, DynaSLAM, or official DyPho-SLAM main-table row.
