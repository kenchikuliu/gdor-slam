# FlowParse-Style Minimum Experiment Matrix

Status date: 2026-09-17

This matrix treats the experiments in the user-provided
`/home/slam/Desktop/FlowParse-SLAM.tex` as the minimum comparison floor for the
GDOR/DYN-19 paper package. It does not promote source-reported values into local
reruns or protocol-matched rankings.

## Required minimum

| Evidence block | FlowParse-style floor | Current GDOR/DYN-19 status | Release gate |
|---|---|---|---|
| Tracking | Four TUM dynamic sequences: `fr3/w/xyz`, `fr3/w/half`, `fr3/w/static`, and `fr3/s/half`; ORB-SLAM3, DynaSLAM, NICE-SLAM, ESLAM, RoDyn-SLAM, SplaTAM, GS-SLAM, GassiDy, DGS-SLAM, Photo-SLAM, and DyPho-SLAM context rows | Source ledger imported; GDOR has 30 local cells over these four sequences as part of the 90-run main experiment | Verify every external row against the original cited paper; export local temporal trajectory dispersion before reproducing the FlowParse ATE/Std layout |
| Mechanism | At least the two hard TUM sequences `fr3/w/xyz` and `fr3/w/half`, with prior-mask/adaptive-feature context and method variants | 120 DYN-19 mechanism cells plus a seven-row ordered-ablation ledger exist over ten sequences and three seeds | Keep the design named `ordered ablation`; do not claim independent effects or interactions without a factorial experiment |
| Mapping visual | Common rows for `fr3/w/xyz`, `fr3/w/half`, `bonn/ps_track`, and `bonn/r3`; input, SplaTAM, Photo-SLAM, mask-only control, and the proposed method | Input, SplaTAM, Photo-SLAM, and mask-only panels exist; all four current GDOR render panels are missing | Fig.4 remains blocked until the four GDOR panels are generated from the frozen map/pose/crop contract and human-reviewed |
| Mapping quantitative | At minimum, a clearly bounded mapping diagnostic; stronger mapping claims require shared views, pose mode, masks, frame range, and metrics | 72 local mapping cells exist for online and GT-aligned pose modes; PSNR/SSIM plus proxy completeness/ghost risk are available | Photo-SLAM/SplaTAM and at least one dynamic Gaussian plus one classical dynamic RGB-D method still need a shared mapping protocol for a quantitative superiority claim |
| Runtime | Tracking ms, mapping ms, FPS, and sequence time on `fr3/w/xyz`, with external hardware caveats | Source-reported context imported; GDOR has 47.750 s mean full-sequence end-to-end time over 30 Full runs on one RTX 4090 | Export local tracking ms, mapping ms, FPS, sequence length/time, and inclusion policy for masks/segmentation |
| Reproducibility | Trajectory, mask, map, render, and timing evidence | DYN-19 compact cells, trajectories, map paths, logs, certificates, and provenance are registered | Publishable release still requires the missing renders, exact artifact bundle, and completed human-review YAMLs |

## Tracking parity result

The current GDOR Full local three-seed means on the four FlowParse TUM sequences
are 2.148, 3.609, 0.730, and 4.864 cm, with a four-sequence arithmetic mean of
2.838 cm. These are valid local DYN-19 means, but they are not the same statistic
as the source table's per-trajectory temporal Std. Empty local temporal-Std cells
therefore remain missing evidence.

The imported DyPho-SLAM source-report row averages 1.60 cm on the same four named
sequences. The present numbers support a mixed/failure-recovery claim, not a
uniform tracking-SOTA claim.

## Source and status files

- `paper/data/flowparse_tracking_context.csv`: exact tracking transcription from
  `tab:dynamic-tracking`, including corrected `ORB-SLAM3 fr3/w/half = 30.1`.
- `paper/data/flowparse_runtime_context.csv`: exact runtime transcription from
  `tab:dynamic-runtime`.
- `paper/data/flowparse_ablation_context.csv`: exact ablation transcription from
  `tab:dynamic-ablation`.
- `source_tables/table1_flowparse_context.csv`: external/reference rows plus the
  local GDOR anchor row.
- `source_tables/table3_flowparse_context.csv`: runtime context plus the local
  end-to-end-only GDOR row.
- `source_tables/flowparse_ablation_context.csv`: imported ablation context kept
  separate from the DYN-19 ordered ablation.

The bootstrap file `mvp_package/tables/table1_external_report.csv` is retained as
skill-generated provenance, but it is not the manuscript source for this parity
check. It contains an `ORB-SLAM3 fr3/w/half = 0.31` transcription inconsistent
with the provided TeX value `30.1` and omits several required rows.
