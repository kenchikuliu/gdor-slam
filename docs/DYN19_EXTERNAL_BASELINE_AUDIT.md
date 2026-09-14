# DYN-19 External Baseline Audit

Date: September 14, 2026

Status: audit-only. No external-baseline number is promoted to a DYN-19
main table until the row passes the eligibility checks below.

## Dynamic Gaussian Baseline: DG-SLAM

- Official repository: <https://github.com/fudan-zvg/DG-SLAM>
- Audited checkout: `fb28a20caa74f376ebd4a5f3c473cd57e7f5d9b4`
- The shipped `run_tum.py` reads `groundtruth.txt` and passes each ground-truth
  pose into `dg_model.track`.
- `dg_model.track` stores those poses as `poses_gt`, converts them to
  `gt_pose_idx`, and passes them into the Gaussian tracking/mapping routine.
- The shipped mapping path uses that `gt_c2w` value for map optimization and
  keyframe construction. This is a ground-truth-pose mapping path, not a
  protocol-matched online-pose baseline.
- The official preprocessing instructions also require OneFormer semantic
  masks. The repository advertises Bonn downloads, but its shipped runner and
  config set do not provide a single protocol-matched Bonn execution path for
  the DYN-19 ten-sequence denominator.

**Decision:** the official runner is retained as provenance and diagnostic
evidence, but its numbers are ineligible for the DYN-19 main table. A fair
adapter must use online poses for both tracking and mapping, record independent
mask provenance, use the frozen associations and scene names, and emit a
manifest, trajectory, map, render, per-frame log, and metric certificate.

## Classical Dynamic RGB-D Baseline: DynaSLAM

- Official repository: <https://github.com/BertaBescos/DynaSLAM>
- Audited checkout: `8f894a8b9d63c0a608fd871d63c10796491b9312`
- The official RGB-D entry point documents TUM execution and supports
  geometry-only dynamic detection when no Mask R-CNN directory is supplied.
- It does not provide a standardized Bonn execution path for the DYN-19
  denominator.
- The local checkout contains diagnostic compatibility edits
  (`CMakeLists.txt`, `include/ORBextractor.h`, and `src/MaskNetNoOp.cc`) and a
  separate build directory. No completed official-compatible binary has been
  accepted as a protocol-matched DYN-19 result.

**Decision:** DynaSLAM remains a requested baseline and a compatibility
work item. A local diagnostic build or a partial trajectory is not a valid
main-table row. Acceptance requires a documented build identity, online
tracking, frozen associations, explicit dynamic-mask provenance, complete
trajectory and map artifacts, and a no-failure metric manifest.

## Eligibility Gate

An external baseline row is eligible only when all of the following are true:

1. The source repository and local adapter commit are recorded.
2. Camera poses used by the method are online estimates; ground truth is used
   only by the evaluator.
3. The scene, association file, frame policy, depth units, seed, and metric
   policy match the DYN-19 manifest.
4. Dynamic masks are either the method's documented external input or a
   clearly labeled no-mask/geometry variant; DYN-19 method masks are not
   silently reused.
5. The run has a non-empty trajectory, map output, render output where
   applicable, per-frame log, summary, and source/hash manifest.
6. Failed, interrupted, GT-aligned, contaminated, or otherwise diagnostic
   rows stay outside the main quantitative table.

Until these gates pass, the correct public statement is that the external
baseline comparison is pending rather than that DYN-19 wins against those
methods.
