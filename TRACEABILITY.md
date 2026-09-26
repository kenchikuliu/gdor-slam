# GDOR / DYN-19 Traceability Record

Status date: 2026-09-26

This record answers a narrow question: which parts of the GDOR/DYN-19 work are
verifiably present on a Git remote, and which local engineering or experiment
directories are not yet covered by a remote commit. A local path or a Git remote
URL is not evidence that its current files have been pushed.

## Release status

| Work item | Local path | Remote | Branch / ref | HEAD | Local state | Traceability status |
|---|---|---|---|---|---|---|
| Public GDOR/DYN-19 release | `/home/slam/DynaGS-SLAM-dyn19-publish` | `https://github.com/kenchikuliu/gdor-slam.git` | `dyn19-publish-20260914` | `78cf347ac077df24edd604110de4f7e8e558afb5` | clean; `0 ahead / 0 behind` | **Pushed and verified** |
| DYN-19 causal-map-integrity development branch | `/home/slam/DynaGS-SLAM-dyn19-20260913` | `https://github.com/kenchikuliu/gdor-slam.git` | `dev/dyn19-causal-map-integrity-20260913` | `fd42a51626eac2d6cf3c761ad1d49f68ef63b18e` | clean; `0 ahead / 0 behind` | **Pushed and verified** |
| DYN-21 Protocol-300 current comparison package | `/home/slam/DynaGS-SLAM-dyn19-20260913/paper/dyn19_v6/releases/dyn21_protocol300_current_20260917` | `https://github.com/kenchikuliu/gdor-slam.git` | `traceability/dyn21-protocol300-release-20260926` | `d79c3fc408b6d1871b79d06a7ace09fdc4dc0fc5` | clean; `0 ahead / 0 behind` | **Pushed and verified on traceability branch** |
| Mapping evaluation engineering precursor | `/home/slam/DynaGS-SLAM-mapping-work` | `https://github.com/kenchikuliu/DynaGS-SLAM.git` | `traceability/mapping-evaluation-20260926` | `7d50616c9e7c86f29a11966e741fb0cd35295d40` | clean; `0 ahead / 0 behind` | **Pushed and verified on traceability branch** |
| Protocol-300 baseline/evidence root | `/home/slam/experiments/dypho_skill_full_rerun_20260729` | `https://github.com/kenchikuliu/dypho-slam-protocol300-evidence.git` | `traceability/protocol300-checkpoint-manifest-20260926` | `d66221c...` | clean; `0 ahead / 0 behind` | **Pushed and verified on traceability branch** |

## What is publicly traceable now

The public GDOR release commit contains the paper source, aggregate DYN-15 to
DYN-19 CSV ledgers, reproducible evaluation scripts, evidence manifests, the
DyPho-style table package, the FlowParse-style minimum comparison matrix, and
the HTML table/mapping previews. The separate DYN-21 traceability branch also
contains the current Protocol-300 comparison package: real render panels,
contact sheet, HTML report, metrics, evidence manifest, and SHA-256 records.
The release branch is directly verifiable with:

```bash
git clone https://github.com/kenchikuliu/gdor-slam.git
git checkout dyn19-publish-20260914
git rev-parse HEAD
# 78cf347ac077df24edd604110de4f7e8e558afb5
```

The current release does **not** claim that raw experiment roots, full generated
Gaussian maps, per-frame masks, or all local render directories are redistributed.
The public repository contains the aggregate values and provenance paths needed
to audit the paper claims; the raw roots remain local unless separately
published.

## Remaining publication boundary

1. The public GDOR release references retained local experiment roots in its
   freeze/provenance records. Those paths are provenance pointers, not public
   download links. A reviewer without the retained artifact bundle cannot
   independently retrieve those raw files from GitHub alone.
2. The mapping, DYN-21 comparison, and Protocol-300 updates are on explicit
   traceability branches, not merged into the default branches. Their exact
   commits are nevertheless publicly reachable and auditable.

## Verification commands

Run from the relevant repository:

```bash
git status --short --branch
git rev-parse HEAD
git branch -r --contains HEAD
git ls-remote origin refs/heads/<branch>
```

For the public release, the final check performed on 2026-09-26 was:

```text
local HEAD:  78cf347ac077df24edd604110de4f7e8e558afb5
remote ref:  78cf347ac077df24edd604110de4f7e8e558afb5
branch:      dyn19-publish-20260914
worktree:    clean
```

## Claim boundary

The accurate statement as of 2026-09-26 is:

> The identified GDOR/DYN-19 code, DYN-21 comparison package, mapping
> evaluation precursor, and Protocol-300 retention manifests are covered by
> named public Git refs with auditable commits. Raw experiment roots and
> generated Gaussian assets remain outside GitHub and are referenced by local
> provenance paths.

Therefore “all identified engineering is pushed” is accurate only with the
scope and branch names recorded in this file; it does not mean that all raw
experiment outputs are uploaded.
