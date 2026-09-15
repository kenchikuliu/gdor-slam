# DYN-19 Causal Map-Integrity Result

Completed: September 14, 2026  
Experiment ID: `DYN-19_FULL_CAUSAL_MAP_INTEGRITY_20260913`  
Source commit: `14c6b2ca696a7eacda8f193e5c8a0a44e55d527e`  
Release: `DYN19_RESULTS_20260914_R1`

## Result Scope

DYN-19 completed the frozen denominator without missing cells:

- 90 main tracking runs: Semantic, MapMatched, and Full;
- 120 ordered-mechanism runs: T+M, T+F+M, T+F+R+M, and Full-NoM;
- 72 common-view render/metric cells: four scenes, three seeds, three main
  configurations, and online/GT-aligned pose modes.

The result package is stored under:

`/mnt/nas_datasets/slam-experiments/DynaGS-SLAM/dyn19_release_20260914_r1`

Its release manifest records checksums for the tracking, mapping, mechanism,
and metadata archives. Raw artifacts remain on NAS and are not committed.

## Main Tracking Result

The table averages the ten predeclared sequence means. Each sequence mean uses
seeds 0, 1, and 2.

| Configuration | ATE RMSE (cm) | 1-frame translation RPE (cm) | Failure rate | Runtime (s/run) |
| --- | ---: | ---: | ---: | ---: |
| Semantic | 10.119 | 3.420 | 1.497% | 37.90 |
| MapMatched | 7.392 | 3.362 | 2.046% | 34.36 |
| Full | **4.930** | **3.313** | **0.147%** | 47.75 |

Relative to Semantic, Full lowers the ten-sequence mean ATE by 51.3%. It wins
ATE in 19/30 sequence-seed cells and one-frame translation RPE in 18/30 cells.
The result is positive but not universal: Full wins 7/10 sequence means against
Semantic and 4/10 against MapMatched. Its runtime is 26.0% higher than Semantic.

## Route-Separation Certificate

Every Full cell contains observed recovered tracking support and a valid
persistent mapping weight:

- Full: 30/30 non-vacuous certificate passes;
- recovered tracking support overlapping map admission: zero pixels in all
  30 Full cells;
- Full-NoM counterfactual: 30/30 certificates fail because recovered support
  is deliberately allowed to overlap persistent mapping support.

This is direct evidence for the implemented routing contract: useful transient
tracking recovery can occur without granting the same observations persistent
map authority. The Full-NoM result shows that zero overlap is enforced by the
conservative mapping route rather than arising vacuously.

## Ordered Mechanism Result

DYN-19 is an ordered ablation, not a factorial design. The rows below therefore
describe complete configurations and do not identify independent component
effects or interactions.

| Configuration | ATE RMSE (cm) | 1-frame translation RPE (cm) | Failure rate |
| --- | ---: | ---: | ---: |
| T+M | **3.859** | **3.062** | **0.007%** |
| T+F+M | 6.377 | 3.213 | 2.069% |
| T+F+R+M | 4.907 | 3.186 | 0.719% |
| T+F+R+A+M (Full) | 4.930 | 3.313 | 0.147% |
| T+F+R+A+NoM | 4.060 | 3.181 | 0.179% |

T+M is the strongest aggregate tracking configuration in this consumed-data
matrix. It wins ATE in 20/30 cells and translation RPE in 24/30 cells against
Semantic, while retaining 30/30 zero-overlap certificate passes. This makes a
simpler temporal-recovery route the leading successor candidate, but not yet a
new held-out claim or default configuration.

## Multi-Seed Mapping Diagnostic

The mapping matrix uses four predeclared scenes, three seeds, and 20 common
non-keyframe views per sequence-seed cell. LPIPS was unavailable in this run.

| Pose mode | Configuration | Full PSNR | Static PSNR | Full SSIM | Static SSIM |
| --- | --- | ---: | ---: | ---: | ---: |
| Online | Semantic | 15.757 | 18.866 | 0.636 | 0.695 |
| Online | MapMatched | 15.761 | 18.990 | 0.639 | 0.700 |
| Online | Full | **16.339** | **19.763** | **0.665** | **0.728** |
| GT-aligned | Semantic | 11.477 | 12.082 | 0.460 | 0.492 |
| GT-aligned | MapMatched | 11.737 | 12.442 | 0.468 | 0.503 |
| GT-aligned | Full | **12.724** | **13.491** | **0.497** | **0.533** |

Against Semantic, Full improves online static PSNR by 0.896 dB and static SSIM
by 0.033, with wins in 9/12 cells for each metric. In GT-aligned evaluation it
improves static PSNR by 1.409 dB and static SSIM by 0.042. These are multi-seed
common-view reconstruction diagnostics, not proof of universal map superiority.

The GT-aligned occluded-background proxy also moves in the favorable direction
on aggregate: completeness increases by 0.038, mean color error decreases by
2.565, and the high-threshold ghost-risk proxy decreases by 0.038 relative to
Semantic. These values remain proxy evidence because no independent ghost or
revealed-background ground truth is available.

## Decision

DYN-19 strengthens the main GDOR-SLAM project rather than the terminated Schur
pose-prior route. It establishes three useful facts under the declared local
protocol:

1. transient recovery and persistent map admission are empirically separable;
2. the complete Full route improves aggregate tracking and multi-seed
   common-view reconstruction over Semantic, with sequence-dependent failures;
3. the simpler T+M configuration is the strongest tracking candidate in the
   ordered mechanism matrix and deserves a separately registered held-out test.

The result does not establish official DyPho-SLAM superiority, external SOTA,
factorial component causality, or independent 3D ghost ground truth.
