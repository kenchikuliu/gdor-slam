# DYN-21 Protocol-300 Current-Method Results

Completed: September 17, 2026  
Experiment ID: `DYN-21_PROTOCOL300_CURRENT_20260917`

## Completion

All four planned `dyn19_full`, seed-0, 300-frame runs completed from frozen
orchestration commit `f1c00ec69c8132f05d6ec1fc9e1303f84ce3e8d5`.
Every cell contains 300 processed frames, 300 tracked frames, zero failed
frames, a 300-pose trajectory, `frame_metrics.csv`, `run_summary.json`,
held-out evidence, and a final Gaussian PLY plus cameras file.

The authoritative result package is:

```text
/mnt/nas_datasets/slam-experiments/DynaGS-SLAM/
  dyn21_protocol300_current_20260917_run01/
```

## Tracking

ATE uses the frozen unique 0.02-second timestamp association and fixed-scale
SE(3) alignment.

| Method | fr3/w/xyz | fr3/w/half | bonn/ps_track | bonn/r3 | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| SplaTAM | 46.378 | 218.839 | 40.287 | 133.131 | 109.659 |
| Photo-SLAM | 45.614 | 25.011 | 92.505 | 24.531 | 46.915 |
| DynaGS semantic | 1.707 | 3.935 | 5.827 | 11.080 | 5.637 |
| **DYN-19 Full (current)** | **1.667** | **3.827** | **3.889** | **5.945** | **3.832** |

Values are ATE RMSE in centimeters. This is a seed-0, 300-frame local
comparison and has no variance estimate.

## Mapping Diagnostic

The four current-method panels were rendered from the final 300-observation
maps at the frozen public-GT poses, with the selected pose excluded from the
20-pose alignment support. All panels are real 640x480 outputs with no crop or
post-processing.

| Method | fr3/w/xyz | fr3/w/half | bonn/ps_track | bonn/r3 | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| SplaTAM | 11.771 / 0.30418 | 12.436 / 0.35140 | 13.016 / 0.58047 | 14.509 / 0.49772 | 12.933 / 0.43344 |
| Photo-SLAM | 14.830 / 0.40459 | 14.936 / 0.53600 | 15.316 / 0.66293 | 14.345 / 0.63369 | 14.857 / 0.55930 |
| DynaGS semantic | 15.491 / 0.45580 | 15.447 / 0.56482 | 12.156 / 0.60761 | 14.364 / 0.65523 | 14.364 / 0.57086 |
| **DYN-19 Full (current)** | **14.976 / 0.47566** | **12.009 / 0.54068** | **11.589 / 0.63100** | **13.776 / 0.65111** | **13.088 / 0.57461** |

Cells are PSNR in dB / SSIM. Dynamic foreground remains in the reference
images, so this table is a full-frame diagnostic, not a static-map superiority
claim. Full has the highest average SSIM in this four-method package but lower
average PSNR than Photo-SLAM and DynaGS semantic.

## Runtime

| Method | fr3/w/xyz | fr3/w/half | bonn/ps_track | bonn/r3 | Total | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SplaTAM | 2871.197 | 1568.175 | 1477.424 | 1178.818 | 7095.614 | 0.16912 |
| Photo-SLAM | 33.429 | 29.682 | 21.810 | 16.972 | 101.894 | 11.77699 |
| DynaGS semantic | 16.589 | 16.624 | 15.245 | 12.388 | 60.845 | 19.72212 |
| **DYN-19 Full (current)** | **38.769** | **38.788** | **27.135** | **24.369** | **129.061** | **9.29791** |

Times are end-to-end wall seconds. Throughput is frames per second over all
1,200 observations. No cross-method tracking/mapping component split is
available.

## Figure Evidence

The comparison package contains 20 real panels in the order Input, SplaTAM,
Photo-SLAM, DynaGS semantic, and DYN-19 Full. The correctly labeled contact
sheet is:

```text
comparison/fig4_current_comparison_contact_sheet.png
```

The DyPho pixel-locked internal layout draft passed the automated locked-pixel
audit with no violations. It is not released: human review is still required,
and the immutable reference column title says `DyPho-SLAM(Ours)` rather than
`DYN-19 Full`.

## Evidence Hashes

| Artifact | SHA-256 |
| --- | --- |
| `comparison/metrics.json` | `78b4fd45ff3390f15ae212ff60979fc60e0af8458fb276bd50d9be1e75573d69` |
| `comparison/evidence_manifest.json` | `7c869cbdacf60b8a58158da1900532d58b3f7c765e06fcd082534a365c247186` |
| `comparison/tables/table1_tracking.csv` | `67b0ea90b385f403adc4b0d3896b7add40ed1e2a56433384930e84513353f384` |
| `comparison/tables/table2_mapping.csv` | `0ba3d58d34296342b23d8207f46f7552b858d6f5b0b6d60ef80a2af741f1135e` |
| `comparison/tables/table3_runtime.csv` | `9d63a5bc08ef22ad21c73cd0fcc71ac7a67edc14a7d4338381bd69af19edf323` |
| correctly labeled contact sheet | `747e41854139ecbf4fba002a093d1675b4b811087da690c1cad0caa51873e7c8` |
| internal DyPho layout draft | `a70a63ddbaf0855472161c775ed081ad5c9377c27de22150ae5525a8f66b5c26` |

The historical failed-boundary `Ours` row remains unchanged and excluded from
the current-method tables and Fig. 4 column.
