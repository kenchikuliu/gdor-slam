# DYN-20 T+M Fresh Held-Out Protocol

Registered: September 15, 2026
Experiment ID: `DYN-20_TPLUSM_FRESH_HELDOUT_20260915`
Selection source: `DYN-19_FULL_CAUSAL_MAP_INTEGRITY_20260913`

## Hypothesis

DYN-19 identifies `dyn19_temporal_map` (T+M) as the strongest aggregate
tracking row on the consumed ten-sequence mechanism matrix. DYN-20 tests the
prospective hypothesis that this simpler route generalizes beyond every
DYN-19 sequence while retaining exact tracking-to-mapping separation.

T+M is not promoted before this protocol is complete and passes its frozen
gate. No threshold, sequence, seed, metric, or decision rule may be changed in
response to DYN-20 outcomes.

## Freshness Boundary

The held-out split contains no DYN-19 claim sequence:

```text
tum_sitting_static
tum_walking_rpy
tum_sitting_xyz
tum_sitting_rpy
```

`tum_sitting_static` is newly added to the project for DYN-20. The other three
sequences appeared in earlier non-T+M diagnostics, so this protocol is fresh
for the DYN-19-selected T+M candidate but is not described as globally unseen
project data. This disclosure is part of the frozen benchmark plan.

## Denominator

Configurations:

```text
dyn19_semantic
dyn19_mapmatched
dyn19_temporal_map
dyn19_full
```

Seeds are exactly `0`, `1`, and `2`. The denominator is:

```text
4 configs x 4 sequences x 3 seeds = 48 tracking runs
```

All configurations execute serially within each sequence/seed block under one
GPU lock, with alternating configuration order across blocks. Gaussian mapping
remains enabled, full association files are processed, and
`heldout_stride=0`. DYN-20 does not launch a mapping-rendering study or export
new static-mask snapshots.

## Frozen Input Identities

| Sequence | Association SHA-256 | Ground-truth SHA-256 |
| --- | --- | --- |
| `tum_sitting_static` | `1eb67a1054e1548f2037d47276f6b6f85725e80e056032be47744d8dae4878e6` | `13b03e2568833bb0a5ca901df587a882549b155682fff722eb94d726a3a4451c` |
| `tum_walking_rpy` | `34e8fb45667449f11d853de05f3a3eab9748c5559fc093e7004644b4c70607f3` | `9a565a8ea40b6c009c861c955341dd92c2c05924bca33909d0d5283f7bd59739` |
| `tum_sitting_xyz` | `e386b50e295c1ca6cd1d21d690e849218afb617b95b4b8006eef550e7452f6c8` | `17bfc4d03f725c7769eef468449ef0dae68b97309714f1561eec4a89807a0066` |
| `tum_sitting_rpy` | `ef1a296f30b5adfa3a8b3f8b98f0b45e06e708a4e0ac06a57b2564f9284cd5ee` | `b27e1cbe302cdc3998d47b66809f2fce5af4d8fc1b23713fa08cfcbc6831bfcd` |

The downloaded `rgbd_dataset_freiburg3_sitting_static.tgz` archive has
SHA-256
`6239d4b7f6edcab719b51c76434acdc238d03dad666758b51e6046992a7c3914`.
Its RGB/depth association file is generated once with the standard 20 ms TUM
timestamp matcher and then frozen by the benchmark plan's normal asset hashes.

## Metrics and Route Evidence

The tracking report retains all input frames and computes:

- SE(3)-aligned ATE RMSE;
- one-frame translation and rotation RPE RMSE;
- failure rate and trajectory coverage;
- end-to-end runtime;
- exact recovered-support overlap certificates.

For T+M, every sequence/seed cell must contain observed recovered support,
mapping support on recovered frames, and zero recovered-support overlap with
map-admissible support. A vacuous certificate is not a pass.

## Predeclared Promotion Gate

T+M is promoted only if all clauses pass:

1. all 48 planned runs complete with no duplicates, substitutions, or extra
   result identities;
2. T+M passes all 12 non-vacuous zero-overlap route certificates;
3. T+M aggregate ATE is no worse than Full and lower than Semantic;
4. T+M wins paired ATE against Full in at least 7 of 12 sequence-seed cells;
5. T+M mean failure rate is no more than 0.5 percentage points above Full.

Runtime, RPE, MapMatched, and per-sequence values are reported but are not
additional binary promotion clauses. A failed clause is a No-Go for promoting
T+M under this protocol; it does not authorize tuning on this split.

The completed runner writes `dyn20_tplusm_heldout_report.json`, which audits
the frozen plan and applies these clauses mechanically.

## Registration Command

Create the immutable plan from a clean committed source tree before any GPU
execution:

```bash
DYNAGS_DATASETS_ROOT=/mnt/nvme_data/datasets \
python3 scripts/run_reproducible_benchmark.py \
  --dyn20-tplusm-heldout \
  --gpus 0 1 \
  --jobs 2 \
  --dry-run \
  --output-root /mnt/nas_datasets/slam-experiments/DynaGS-SLAM/dyn20_tplusm_fresh_heldout_registered_20260915
```

After registration, remove `--dry-run` and use a new output root for actual
execution. The registered plan is never reused as a result directory.

## Completed Registration

The dry-run plan was frozen on September 15, 2026 from clean source commit
`a927a7959b2e4febedac13df01780107fe463e6b`. It contains exactly 48 tasks and
has SHA-256
`9b20aafd4760c1a8f92cc1583258e08280954fc02ceafa41d65a7170453b3145`.
The machine-readable registration, input hashes, and audit result are recorded
in `DYN20_TPLUSM_HELDOUT_REGISTRATION.json`.
