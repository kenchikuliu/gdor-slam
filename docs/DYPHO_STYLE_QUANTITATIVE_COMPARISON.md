# DyPho-Style Quantitative Comparison

Updated: September 17, 2026

## Status

The project now has a FlowParse-minimum, source-labeled comparison package for
tracking, mapping, ablation, and runtime:

- **Broad tracking:** all 11 registered DyPho Table I methods marked `[ext]`,
  plus local DYN-19 Semantic, MapMatched, and Full.
- **Mapping:** local DYN-19 common-view PSNR/SSIM results for online and
  GT-aligned rendering, with the full-frame and fixed-static-region metrics.
- **Ablation:** seven ordered local configurations on `w/xyz` and `w/half`.
- **Runtime:** 11 source-reported rows plus a visibly scoped local DYN-19 Full
  end-to-end row.
- **Local baselines:** the retained Protocol-300 package contains complete
  SplaTAM/Photo-SLAM tracking, mapping diagnostics, wall time, and 20 real
  qualitative panels.

The installed `dypho-slam-figure-automation` skill now also renders the
paper-facing draft tables at
`paper/dyn19_v6/dypho_style_assets/build/draft/table1.png`,
`table2.png`, and `table3.png`. The companion
`paper/dyn19_v6/dypho_supplemental_assets/` package renders the two-sequence
ablation and broad runtime tables. Each raster has a source hash, automated
audit, and pending human-review form; none is released automatically.

Every `[ext]` row is an **external reference**, not a protocol-matched rerun.
Those values are not included in local means, win counts, or promotion gates.
This follows the project decision that source-reported methods do not need to
be rerun under the local protocol for descriptive positioning.

The exact requirement-level status is recorded in
`docs/FLOWPARSE_MINIMUM_EXPERIMENT_MATRIX.md` and its CSV companion. The one
material remaining gap is a co-registered current DYN-19 Full cell against the
Protocol-300 SplaTAM/Photo-SLAM package. The historical Protocol-300 `Ours` row
is a failed-boundary ablation and cannot be relabeled as DYN-19 Full.

## Tracking

All local rows below are three-seed full-sequence means. DYN-19 is the current
ten-sequence matrix; DYN-18 is the separately frozen four-sequence external-
positioning campaign. DyPho-SLAM values are the numbers reported in its Table I.
The paper-facing raster additionally includes every other method row from that
registered source. Lower ATE is better within each reported protocol; the
mixed-protocol table does not apply cross-row rank highlighting.

| Sequence | DYN-19 Semantic | DYN-19 MapMatched | DYN-19 Full | DYN-18 Guarded/GDOR | DYN-18 Strict control | DyPho-SLAM published |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TUM walking xyz | 2.3413 | 2.1032 | 2.1476 | 2.1187 | 2.1009 | 1.6 |
| TUM walking halfsphere | 3.3792 | 3.4234 | 3.6092 | 3.3441 | 3.3500 | 2.6 |
| TUM walking static | 0.7370 | 0.7901 | 0.7297 | 0.8429 | 0.7449 | 0.6 |
| TUM sitting halfsphere | 24.1754 | 36.7750 | 4.8639 | 7.3535 | 7.1821 | 1.6 |

The row-level source is
`paper/dyn19_v6/data/dypho_style_tracking_comparison.csv`. The local DYN-18
campaign is independently documented in
`/home/slam/3dgs_slam_ccfa_assets/DYN-18_GDOR_DYPHO_TUM4_EXTERNAL_POSITIONING_20260807.md`.
The published-reference ledger is
`paper/dyn19_v6/data/external_quantitative_reference.csv`.

The correct conclusions are:

1. GDOR/Guarded improves the same-source Semantic parent on all four sequence
   means in the DYN-18 local evaluation.
2. DYN-19 Full improves the difficult TUM sitting halfsphere case relative to
   the DYN-19 Semantic and MapMatched controls, while its four-sequence local
   values are not uniformly better than every local control.
3. DyPho-SLAM's published values are lower than the local GDOR values on all
   four rows, but this is an external positioning statement rather than an
   official implementation comparison.

The DYN-20 held-out result remains separate: its aggregate Full ATE is 2.464
cm and T+M is 4.014 cm on four held-out sequences. It is not mixed into the
four-sequence TUM table above because it uses a different frozen split.

## Mapping

The local mapping comparison is a 72-cell DYN-19 diagnostic: four scenes,
three seeds, three configurations, two pose modes, and 20 common non-keyframe
views per scene-seed cell. LPIPS was unavailable. Higher PSNR/SSIM is better.

| Pose mode | Configuration | Full PSNR [dB] | Static PSNR [dB] | Full SSIM | Static SSIM |
| --- | --- | ---: | ---: | ---: | ---: |
| Online | Semantic | 15.757 | 18.866 | 0.636 | 0.695 |
| Online | MapMatched | 15.761 | 18.990 | 0.639 | 0.700 |
| Online | Full | **16.339** | **19.763** | **0.665** | **0.728** |
| GT-aligned | Semantic | 11.477 | 12.082 | 0.460 | 0.492 |
| GT-aligned | MapMatched | 11.737 | 12.442 | 0.468 | 0.503 |
| GT-aligned | Full | **12.724** | **13.491** | **0.497** | **0.533** |

The row-level source is
`paper/dyn19_v6/data/dypho_style_mapping_comparison.csv`; the underlying
72-cell evidence remains `paper/dyn19_v6/data/dyn19_mapping_cells.csv`.
Relative to Semantic, Full gains `+0.896 dB` online static PSNR and `+0.033`
online static SSIM, and gains `+1.409 dB` GT-aligned static PSNR and `+0.042`
GT-aligned static SSIM.

The external reference boundary is important: the checked DyPho-SLAM source
reports qualitative novel-view rendering comparisons but no numeric PSNR,
SSIM, or LPIPS mapping table. Therefore there is no defensible external DyPho
mapping number to place beside the local table. The local table is a
quantitative common-view diagnostic, not a claim that GDOR universally maps
better than DyPho-SLAM.

DG-SLAM has published mesh accuracy/completion values in a different metric
family and on a different protocol. Those values are retained in the external
ledger for context and are not converted into PSNR/SSIM or ranked against the
local rendering results.

## Claim Boundary

This package supports the following wording:

> We compare our local tracking variants quantitatively on the four TUM
> sequences used by the DyPho-SLAM table and report the registered published
> baseline rows as visibly labeled external references. We additionally report
> a multi-seed common-view PSNR/SSIM mapping diagnostic and a two-sequence
> ordered module ablation for our local variants.

It does not support the following stronger claims:

- official DyPho-SLAM implementation superiority;
- apples-to-apples ranking across implementations, hardware, masks, or
  preprocessing;
- a numeric DyPho-SLAM mapping comparison;
- universal tracking or mapping superiority;
- independent 3D ghost-contamination ground truth.

## Runtime and Safety Draft

The DyPho-style Table III draft reports local DYN-19 application end-to-end
time, mean per-run failure rate, relative runtime, and route-certificate
status across the 90 main runs. Full records 47.750 s/run, 0.147% mean failure,
1.26x Semantic runtime, and 30/30 non-vacuous pass certificates. The frozen
manifests identify host `slam` and GPU identifier 1 under a serial GPU lock,
but do not record the GPU model. No tracking/mapping component split is
claimed.

The supplemental broad runtime draft adds ORB-SLAM3, Dyna-SLAM3, NICE-SLAM,
ESLAM, RoDyn-SLAM, SplaTAM, GS-SLAM, Photo-SLAM, DGS-SLAM, DG-SLAM, and
DyPho-SLAM as `[ext]` rows. These values are copied from the registered source
table and are descriptive because hardware and timing scope differ. DYN-19
Full is reported as 16.53 application end-to-end frames/s and 47.75 s/run over
30 runs; its tracking and mapping component cells remain unavailable.

## Protocol-300 Mapping Comparison

The retained Protocol-300 package provides real local SplaTAM and Photo-SLAM
comparison evidence over four sequences, including ATE, full-frame PSNR/SSIM,
end-to-end wall time, and a complete 20-panel qualitative page. It is a
seed-0, 300-frame historical package. Its `Ours` row is explicitly
failed-boundary evidence, so it supports baseline availability but not a claim
for the current DYN-19 Full method.
