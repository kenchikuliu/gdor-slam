# Same-Frame Pixel-Admission Diagnostic

This directory publishes the pixel-level evidence used by slide 5 of
`paper/gdor_slam_briefing_ppt.html`.

## Scope

- Sequence: TUM `sitting_halfsphere`
- Seed: `0`
- Processed frame: `816` (zero-based association row)
- Configuration: DYN-19 Full
- Evidence kind: sequential diagnostic replay

Frame 816 was selected from the retained original DYN-19 per-frame log because
it has the largest `tracking_recovery_audit_pixels` value for this sequence and
seed: 98,018 pixels with zero recorded mapping leak pixels.

The replay exports 97,905 recovered Tracking-support pixels, 119,834 persistent
Mapping-admission pixels, and zero direct overlap pixels. Because the replay is
not byte-identical to the original run, the figure is not labeled as an
original DYN-19 release mask artifact.

## Runtime Definitions

```text
recovered_tracking_support = raw_dynamic_mask AND refined_static_mask
mapping_admission = staticWeight >= 0.5
recovered_admission_overlap = recovered_tracking_support AND mapping_admission
```

The definitions are identical to the DYN-19 recovered-support overlap
certificate. They audit direct pixel admission on this route. They do not prove
independent zero-ghost ground truth and do not exclude indirect Mapping changes
through an improved camera pose.

## Files

- `fig_pixel_admission_real_frame.png`: five-panel presentation figure.
- `fig_pixel_admission_real_frame.provenance.json`: selection record, source
  paths, pixel counts, exact definitions, and SHA-256 hashes.
- `source/`: same-frame RGB, binary masks, and runtime metadata used by the
  figure builder.
- `SHA256SUMS`: hashes for every published file in this directory.

The figure is built by `scripts/build_pixel_admission_figure.py`. The runtime
export is implemented in `examples/tum_rgbd_dynamic.cpp` and is diagnostic-only:
it reads the same masks already used by Tracking, Mapping, and the route
certificate without modifying them.

Rebuild command used for this asset:

```bash
python3 scripts/build_pixel_admission_figure.py \
  /home/slam/experiments/gdor_pixel_admission_replay_20261023/frame_000816_retry2/evidence \
  paper/figures/pixel_admission \
  --sequence tum_sitting_halfsphere --seed 0 \
  --original-frame-metrics /mnt/nas_datasets/slam-experiments/DynaGS-SLAM/dyn19_main_20260914_14c6b2c_run01/dyn19_full/tum_sitting_halfsphere/seed_0000/frame_metrics.csv \
  --replay-frame-metrics /home/slam/experiments/gdor_pixel_admission_replay_20261023/frame_000816_retry2/output/frame_metrics.csv \
  --mask-config config/mask_config_dyn19_full.yaml \
  --binary bin/tum_rgbd_dynamic \
  --source-file examples/tum_rgbd_dynamic.cpp \
  --source-commit 0e877f0bcebf78ea07d35e1da83b3b6000996ce7
```

The retained experiment and dataset paths are provenance pointers. The
published `source/` files are sufficient to regenerate the visual panel and
recompute its pixel counts without access to those private roots.
