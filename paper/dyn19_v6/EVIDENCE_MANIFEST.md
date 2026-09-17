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
- `data/dypho_style_tracking_comparison.csv` combines local DYN-19 sequence
  means, local DYN-18 Guarded/Strict means, and four published DyPho-SLAM ATE
  values. The external values are positioning context and are not merged into
  local aggregates, win counts, or promotion gates.
- `data/dypho_style_mapping_comparison.csv` summarizes the local 72-cell
  mapping diagnostic. The checked DyPho-SLAM source reports no numeric PSNR,
  SSIM, or LPIPS mapping table, so no external mapping value is supplied.
- `data/external_quantitative_reference.csv` records the source table, metric,
  scope, and claim boundary for every external reference row.
- A protocol-matched official rerun is not required to report published values
  as external context, but direct external superiority still requires matched
  implementation and evaluation evidence.
- No official DyPho-SLAM reproduction or external SOTA claim is made.
- The frozen August 8, 2026 v5r3 PDF predates DYN-19 and remains unchanged.

## DyPho-Style Draft Package

- `dypho_style_assets/build/draft/table1.png` renders the local three-seed
  tracking rows and all 11 registered source-reported tracking rows, each
  marked `[ext]`.
- `dypho_style_assets/build/draft/table2.png` renders the local 72-cell static
  mapping diagnostic and contains no fabricated external mapping metric.
- `dypho_style_assets/build/draft/table3.png` renders local end-to-end runtime,
  mean per-run failure rate, relative runtime, and route-certificate status.
- The package was built by the installed `dypho-slam-figure-automation` CLI.
  Its audits pass, but every review YAML remains unapproved and release is
  blocked pending an identified human reviewer.
- `dypho_supplemental_assets/build/draft/table2.png` renders the seven-row
  ordered ablation on `w/xyz` and `w/half`.
- `dypho_supplemental_assets/build/draft/table3.png` renders 11 source-reported
  runtime rows plus the local DYN-19 Full end-to-end row. Mixed-hardware rows
  are descriptive only.
- `dypho_style_assets/mvp/reports/self_check.json` remains the structural
  self-check for the current DYN-19 freeze, not the global baseline inventory.
- The retained Protocol-300 package already contains complete local SplaTAM
  and Photo-SLAM tracking, diagnostic mapping, wall-time, and 20-panel Fig. 4
  evidence. Its historical `Ours` row is a failed-boundary ablation and cannot
  be relabeled as DYN-19 Full.
- The remaining qualitative gap is a co-registered current DYN-19 Full cell
  against those Protocol-300 baselines.
