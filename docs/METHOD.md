# GDOR-SLAM Method Summary

## Problem

Hard dynamic masks protect a persistent Gaussian map but can remove useful
background observations near moving objects. GDOR-SLAM treats this as a guarded
observation-recovery problem rather than re-admitting an entire dynamic region.

## Observation Routing Contract

GDOR maintains four distinct per-pixel signals instead of treating a dynamic
mask as a single decision:

- `D_raw`: semantic or residual-flow dynamic exclusion before recovery;
- `R_elig`: the subset of raw-excluded pixels qualified by runtime evidence;
- `S_track`: binary support supplied to ORB extraction and current-frame tracking;
- `W_map`: persistent mapping weight supplied to MapPoint and Gaussian mapping.

The claim-bearing route is:

```text
D_raw = semantic_exclusion OR residual_flow_exclusion
R_elig = D_raw AND temporal_depth_consistent AND NOT recovery_veto
S_track = NOT close(D_raw AND NOT R_elig)
W_map = 1 - close(D_raw)
```

Consequently, a qualified observation can be recovered in `S_track` without
being recovered in `W_map`. The runtime passes the two signals separately into
the RGB-D frontend and stores the mapping weight on the current frame.

## Selective Dynamic Observation Recovery

For each candidate pixel or support region, the implementation combines:

- temporal background support from the prior image and rendered depth;
- RGB-D depth consistency;
- camera-motion-compensated residual optical flow;
- dynamic-region risk and confidence-adaptive safety neighborhoods;
- conservative validity gates before the observation is used.

The recovery path is selective: failure of any required validity condition sets
`R_elig` to zero and falls back to the hard dynamic exclusion.

## Tracking and Mapping Decoupling

Recovered evidence is transient by default. It can contribute to current-frame
tracking or a guarded pose proposal, but dynamic tracklets and object states do
not receive automatic authority to initialize, update, densify, or otherwise
write persistent Gaussian primitives.

This boundary is enforced in the mapping path through the raw-mask-derived
`W_map`. Gaussian photometric loss, RGB-D densification, and MapPoint-to-Gaussian
admission consume this persistent weight rather than `S_track`.

The implementation exposes counters for recovered support, static mapping
pixels, rejected dynamic support, Gaussian admission decisions, and exact
tracking-recovery/mapping overlap. The overlap audit is diagnostic and does not
modify either route.

| Configuration | Tracking support | Persistent mapping weight |
| --- | --- | --- |
| Semantic | closed semantic-static support | default unit support |
| Matched control | closed raw semantic-plus-flow static support | the same raw static support |
| GDOR / Strict | raw support plus qualified recovery | closed raw semantic-plus-flow static support |

## Guarded Recovery

Naive semantic-plus-flow recovery is not treated as sufficient. The guarded
configuration first estimates dominant camera motion from static support,
evaluates residual motion in the candidate region, checks depth consistency, and
fails closed when support or the motion model is invalid. Adaptive feature
replenishment is only enabled after a positive tracking-support history and is
bounded by a short risk window.

## Reproducibility Contract

Claim-bearing runs should use a new output directory, a fixed seed, the recorded
configuration, and `scripts/run_reproducible_benchmark.py`. Runtime manifests
must report a clean source tree, complete return status, input protocol, command,
and artifact hashes. Ground truth is used only by offline evaluation, never by
the runtime recovery or mapping gates.

## Scope of the Current Evidence

The current local evidence supports the mechanism and a comparison against a
DyPho-compatible local reproduction. It does not support a claim against the
official DyPho-SLAM executable or source implementation.

The source release contains the exact overlap counter
`tracking_recovery_mapping_leak_pixels`. The archived DYN-15 through DYN-18
campaign summaries predate that field, so those historical results support
active mapping-gate behavior and source-level route separation rather than an
empirical zero-overlap count.

DYN-19 directly closes that newer audit path. All 30 Full sequence-seed cells
contain observed recovered tracking support and zero overlap with persistent
map admission, while all 30 Full-NoM counterfactual cells expose the expected
overlap. The same release adds a four-scene, three-seed common-view mapping
diagnostic. See [`DYN19_RESULTS.md`](DYN19_RESULTS.md) for numeric results and
the remaining external-baseline and independent-truth boundaries.
