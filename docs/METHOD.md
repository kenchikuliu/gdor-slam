# GDOR-SLAM Method Summary

## Problem

Hard dynamic masks protect a persistent Gaussian map but can remove useful
background observations near moving objects. GDOR-SLAM treats this as a guarded
observation-recovery problem rather than re-admitting an entire dynamic region.

## Selective Dynamic Observation Recovery

For each candidate pixel or support region, the implementation combines:

- temporal background support from the prior image and rendered depth;
- RGB-D depth consistency;
- camera-motion-compensated residual optical flow;
- dynamic-region risk and confidence-adaptive safety neighborhoods;
- conservative validity gates before the observation is used.

The recovery path is selective: failure of any required validity condition falls
back to the hard dynamic exclusion.

## Tracking and Mapping Decoupling

Recovered evidence is transient by default. It can contribute to current-frame
tracking or a guarded pose proposal, but dynamic tracklets and object states do
not receive automatic authority to initialize, update, densify, or otherwise
write persistent Gaussian primitives.

This boundary is enforced in the mapping path through static admission checks.
The implementation exposes counters for recovered support, static mapping
pixels, rejected dynamic support, and Gaussian admission decisions so that the
tracking and mapping effects can be audited separately.

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
