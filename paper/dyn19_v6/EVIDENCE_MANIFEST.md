# DYN-19 Evidence Manifest

## Identity

- Experiment: `DYN-19_FULL_CAUSAL_MAP_INTEGRITY_20260913`
- Release: `DYN19_RESULTS_20260914_R1`
- Source commit: `14c6b2ca696a7eacda8f193e5c8a0a44e55d527e`
- Release manifest SHA-256:
  `b859bac34fa42e87905a6ba77b773490b640e95e5fbea0fa3b65343dd776721d`
- Design: ordered ablation, not factorial

`data/dyn19_provenance.json` binds the release manifest, source metadata,
benchmark plans, and all path-sanitized derived files by SHA-256.

## Tracking Evidence

- Main denominator: 90/90 runs, comprising Semantic, MapMatched, and Full on
  ten sequences and seeds 0/1/2.
- Mechanism denominator: 120/120 runs, comprising T+M, T+F+M, T+F+R+M, and
  Full-NoM on the same sequence/seed matrix.
- Public aggregate: `data/dyn19_tracking_sequence_means.csv`.
- Supported Full result: ten-sequence mean ATE 10.119 cm Semantic, 7.392 cm
  MapMatched, and 4.930 cm Full; Full wins 19/30 paired ATE cells and 7/10
  sequence means against Semantic.
- Boundary: the effect is aggregate-positive and sequence-dependent, not
  universal.

## Route-Separation Evidence

- Public cells: `data/dyn19_route_certificates.csv`.
- Full: 30/30 certificate passes; every cell has observed recovered support,
  mapping support for recovered frames, and zero recovered-support overlap
  with map admission.
- T+M, T+F+M, and T+F+R+M: 30/30 non-vacuous passes each.
- Semantic and MapMatched: 30 vacuous cells each because no support is
  recovered; these are not positive recovery evidence.
- Full-NoM: 30/30 expected certificate failures because the counterfactual
  deliberately lets recovered support overlap map admission.
- Boundary: pixel-route overlap is not an independent 3D ghost-contamination
  metric.

## Ordered Mechanism Evidence

- T+M has the lowest consumed-data aggregate: 3.859 cm ATE, 3.062 cm
  translation RPE, and 0.007% failure.
- It wins 20/30 ATE cells and 24/30 translation-RPE cells against Semantic and
  retains 30/30 zero-overlap passes.
- Boundary: row differences do not identify independent component effects or
  interactions. T+M is not a held-out final method.

## Mapping Evidence

- Denominator: 72/72 cells, comprising four scenes, three seeds, three main
  configurations, and online/GT-aligned pose modes.
- Public cells: `data/dyn19_mapping_cells.csv`.
- Online static region: 18.866 dB / 0.695 Semantic versus 19.763 dB / 0.728
  Full for PSNR/SSIM.
- GT-aligned static region: 12.082 dB / 0.492 Semantic versus 13.491 dB /
  0.533 Full.
- Boundary: common-view metrics are positive multi-seed reconstruction
  diagnostics, not proof of universal mapping superiority.
- Occluded-background fields are proxy-only because no independent revealed
  background or 3D ghost ground truth is available.

## External and Historical Boundary

- DYN-18 remains an independently frozen predecessor validation and is not
  merged into DYN-19 denominators.
- Published external baselines remain literature context unless an official
  implementation passes a protocol-matched audit.
- No official DyPho-SLAM reproduction or external SOTA claim is made.
- The frozen August 8, 2026 v5r3 PDF predates DYN-19 and remains unchanged.
