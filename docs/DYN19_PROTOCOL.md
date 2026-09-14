# DYN-19 Ordered-Ablation Map-Integrity Protocol

Registered: September 13, 2026
Experiment ID: `DYN-19_FULL_CAUSAL_MAP_INTEGRITY_20260913`
Status: frozen ordered-ablation plan; no GPU run is authorized from an
uncommitted or dirty source tree.

The legacy experiment identifier is retained in manifests and run contracts
for provenance compatibility. It is not a claim that the design identifies
independent causal effects.

## Purpose

DYN-19 extends the DYN-17 map-matched control from its three-sequence subset
to the union of all DYN-15 and DYN-18 claim-bearing sequences. It separates
temporal depth recovery, residual-flow recovery veto, tracking-risk gating,
adaptive FAST, and conservative mapping weight rather than describing them as
one guarded bundle.

The protocol tests route separation with a predeclared **ordered ablation**.
It does not revive the terminated Schur or motion-reuse line, and it does not
turn external DyPho-SLAM reports into local evidence.

## Tracking Denominator

Every tracking phase uses these 10 sequences and seeds `0,1,2`:

```text
tum_walking_xyz
tum_walking_halfsphere
tum_walking_static
tum_sitting_halfsphere
bonn_balloon
bonn_crowd
bonn_crowd2
bonn_crowd3
bonn_person_tracking
bonn_person_tracking2
```

`tum_sitting_halfsphere` is mandatory. A missing, failed, or substituted
sequence/seed remains in the denominator and makes that phase incomplete.

### Main Phase: 90 Runs

```text
dyn19_semantic
dyn19_mapmatched
dyn19_full
```

This is `3 configs x 10 sequences x 3 seeds = 90` tracking runs.
`dyn19_semantic` exports frozen per-frame static masks; those masks are the
only static-mask source for DYN-19 common-view and proxy evaluation.

### Mechanisms Phase: 120 Runs

```text
dyn19_temporal_map
dyn19_temporal_flow_map
dyn19_temporal_flow_risk_map
dyn19_full_no_mapping
```

This is `4 configs x 10 sequences x 3 seeds = 120` tracking runs.

The two phases together yield a 210-run tracking matrix. They are frozen
separately so the main comparison can be completed and audited without
waiting for all incremental mechanism rows.

## Factor Matrix

The ordered ablation schedule is:

```text
Semantic
-> MapMatched
-> T + M
-> T + F + M
-> T + F + R + M
-> T + F + R + A + M
-> T + F + R + A + NoM
```

| Config | Temporal depth `T` | Flow veto `F` | Risk gate `R` | Adaptive FAST `A` | Conservative raw mapping `M` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dyn19_semantic` | 0 | 0 | 0 | 0 | 0 |
| `dyn19_mapmatched` | 0 | 0 | 0 | 0 | 1 |
| `dyn19_temporal_map` | 1 | 0 | 0 | 0 | 1 |
| `dyn19_temporal_flow_map` | 1 | 1 | 0 | 0 | 1 |
| `dyn19_temporal_flow_risk_map` | 1 | 1 | 1 | 0 | 1 |
| `dyn19_full` | 1 | 1 | 1 | 1 | 1 |
| `dyn19_full_no_mapping` | 1 | 1 | 1 | 1 | 0 |

`MapMatched` retains raw semantic-plus-flow exclusion for both tracking and
mapping, with no temporal recovery or adaptive feature replenishment. The
flow component `F` specifically denotes the residual-flow veto of temporal
recovery; raw semantic-plus-flow admission remains necessary for the
DYN-17-compatible map-matched route.

`R` and `A` have independent hold counters. They may use the same predeclared
previous-inlier threshold and hold length, but one state machine never opens
or closes the other.

`dyn19_full_no_mapping` is an intentional counterfactual: recovered tracking
support is permitted to enter persistent mapping. Its overlap certificate can
fail by design and is a negative/boundary result, not a runner failure.

### Interpretation Boundary

This matrix is an ordered ablation, not a factorial experiment. Adjacent
contrasts are protocol-defined bundle contrasts because several component
states change along the schedule. The results must not be described as
independent causal effects for temporal depth, flow, risk, adaptive FAST, or
mapping weight, and the matrix does not identify interaction effects. A claim
about an independent component effect or an interaction requires a separately
predeclared factorial or fractional-factorial design with an interaction
analysis.

## Recovered-Support Overlap Certificate

Every completed tracking run emits:

```text
frame_metrics.csv
run_summary.json
recovered_support_overlap_certificate.json
```

The certificate binds the run manifest to the frozen benchmark-plan hash and
checks:

1. frame-level and run-summary recovery/leak/mapping-weight counters agree;
2. every frame that has recovered tracking support has a mapping weight;
3. recovered tracking support has zero overlap with map-admissible support.

For `dyn19_full`, the main aggregate requires zero leaks across all 30
sequence/seed cells. A `pass` additionally requires observed recovery.
`vacuous` means no recovered support occurred and is retained as safety-only
evidence. A certificate `fail` is never positive evidence.

The aggregate records `status` as `pass`, `vacuous`, `fail`, or `incomplete`.
Its separate `safety_status` records whether all completed Full cells satisfy
the zero-leak invariant, so an all-vacuous aggregate can be safety-clean
without being positive recovery evidence.

## Multi-Seed Map-Integrity Evaluation

The mapping diagnostic reuses completed main-phase maps for:

```text
tum_walking_xyz
tum_sitting_halfsphere
bonn_crowd2
bonn_person_tracking2
```

It evaluates seeds `0,1,2` and:

```text
dyn19_semantic
dyn19_mapmatched
dyn19_full
```

For each of the 12 sequence/seed cells,
`prepare_mapping_heldout.py` creates one common non-keyframe manifest across
all three maps. Final maps are rendered at both `online` and `gt_aligned`
poses. The full result matrix is:

```text
4 scenes x 3 seeds x 3 configs x 2 pose modes = 72 render/metric cells
```

Normal rendering metrics remain full/static PSNR, SSIM, and LPIPS when LPIPS
is available. The `gt_aligned` mode is still an end-to-end reconstruction
diagnostic because online poses constructed the map.

## Occluded-Background Proxy

`evaluate_occluded_background_proxy.py build` creates a partial proxy once
per sequence/seed common manifest:

1. sample RGB-D source frames with frozen semantic-static masks;
2. transform source pixels using public dataset ground-truth poses;
3. for Bonn, convert marker poses with the existing
   `T_sensor = T_ROS^-1 T_marker T_ROS T_m` contract;
4. reproject source points into target pixels marked dynamic by the frozen
   semantic mask;
5. retain only points whose reprojected source depth is behind the target
   observed depth by the predeclared margin;
6. resolve competing source points with a nearest-depth z-buffer.

For a final-map render, the diagnostic reports:

- `proxy_coverage`: proxy-support pixels divided by target frozen-dynamic
  pixels;
- `background_proxy_color_error_mean`;
- `background_completeness_at_tau`;
- `ghost_risk_proxy_at_tau_high`.

These are explicitly **occluded-background proxy** metrics. They are not true
ghost-contamination counts, complete background ground truth, or a standalone
mapping-superiority claim. Empty proxy support is reported as a boundary,
not converted to zero ghost risk or complete background.

## Execution and Storage

Before any GPU run:

1. build the source in a dedicated build directory;
2. run all repository and DYN-19 tests;
3. commit the implementation and require a clean source snapshot;
4. inspect both tracking dry-run plans.

Tracking plans:

```bash
DYNAGS_DATASETS_ROOT=/mnt/nvme_data/datasets \
python3 scripts/run_reproducible_benchmark.py \
  --dyn19-phase main --gpus 0 --jobs 1 --dry-run \
  --output-root /mnt/nas_datasets/slam-experiments/DynaGS-SLAM/dyn19_main_dryrun_YYYYMMDD

DYNAGS_DATASETS_ROOT=/mnt/nvme_data/datasets \
python3 scripts/run_reproducible_benchmark.py \
  --dyn19-phase mechanisms --gpus 0 --jobs 1 --dry-run \
  --output-root /mnt/nas_datasets/slam-experiments/DynaGS-SLAM/dyn19_mechanisms_dryrun_YYYYMMDD
```

After a clean completed main tracking root exists, create the mapping
evaluation under a separate NAS root:

```bash
DYNAGS_DATASETS_ROOT=/mnt/nvme_data/datasets \
python3 scripts/run_dyn19_map_integrity.py \
  --tracking-root /mnt/nas_datasets/slam-experiments/DynaGS-SLAM/dyn19_main_YYYYMMDD \
  --output-root /mnt/nas_datasets/slam-experiments/DynaGS-SLAM/dyn19_map_integrity_YYYYMMDD \
  --gpus 0
```

Raw trajectories, maps, renderings, masks, proxy images, logs, and generated
metric artifacts remain on NAS. Git records source, configs, protocol,
manifests, summaries, and checksums only.

## Claim Boundary

DYN-19 can support only the result scope that completes with matching frozen
plans, source identities, and certificates. It may establish ordered-ablation
contrasts and whether a stated route keeps observed recovered tracking support
out of map admission under the declared protocol. It cannot, by itself,
establish independent component causality, interaction effects, external SOTA,
official DyPho-SLAM superiority, universal tracking improvement, or true 3D
ghost ground truth.
