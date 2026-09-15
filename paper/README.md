# GDOR-SLAM Paper Artifact

This directory contains the public, evidence-linked artifact for:

> GDOR-SLAM: Guarded Dynamic Observation Recovery for RGB-D Gaussian SLAM

The updated PDF is a ten-page TMM pre-submission manuscript rebuilt on
September 15, 2026.
It uses an unfinished author placeholder and is not a publication record.

## Contents

- `GDOR-SLAM_TMM_v5r4_preprint.pdf`: rebuilt manuscript PDF.
- `GDOR-SLAM_TMM_v5r3_preprint.pdf`: earlier frozen manuscript PDF retained
  for provenance.
- `main.tex`, `sections/`, `references.bib`: matching LaTeX source.
- `figures/`: the five figures used by the manuscript source.
- `data/`: aggregate claim-bearing CSV files for DYN-15 through DYN-18 and the
  DYN-19 failure-recovery-control episode diagnostic.
- `scripts/`: deterministic figure generation and evidence validation.
- `reports/evidence_validation.json`: computed values checked against the text.
- `FIGURE_TABLE_MANIFEST.yaml`: figure/table-to-evidence mapping.
- `EVIDENCE_MANIFEST.md`: public protocol and claim-boundary ledger.
- `SHA256SUMS`: hashes for the published artifact files.

Raw datasets, generated Gaussian maps, trajectories, per-frame masks, model
weights, and full experiment roots are not redistributed. The CSV files expose
the aggregate values used by the paper, while the repository contains the
runner and evaluation scripts required to produce equivalent manifests from
locally obtained datasets.

## Validate The Evidence

From the repository root:

```bash
python3 paper/scripts/validate_gdor_tmm_evidence.py
```

The validator recomputes the principal table values from the public CSV files,
checks that they occur in the manuscript source, verifies the expected five
figures and three tables, and writes `paper/reports/evidence_validation.json`.

The validator requires Python 3 and NumPy. Regenerating Figures 1 and 2 also
requires Matplotlib and Pillow:

```bash
python3 paper/scripts/generate_gdor_tmm_figures.py
```

`figures/fig3_trajectory_examples.png` and
`figures/fig4_commonview_mapping_audit.png` are archived experiment
visualizations and are included as fixed evidence images rather than
reconstructed from unpublished trajectories or rendered frame directories.
`figures/fig5_failure_recovery_control.png` is generated from the public DYN-19
episode summary CSV and visualizes failure intervals, recovered tracking
support, and the matched no-track-reuse control.

## Result Summary

| Evidence | Scope | Main result |
| --- | --- | --- |
| DYN-15 | 63 runs, TUM walking_xyz plus Bonn6 | All7 mean ATE: 11.331 cm Semantic, 3.798 cm GDOR |
| DYN-18 | 36 cells, frozen TUM4 validation | Mean ATE: 9.746 cm Semantic, 3.415 cm GDOR; 9/12 paired wins |
| DYN-17 | 9-run map-matched control | Mean ATE: 7.816 cm Matched, 4.053 cm GDOR |
| DYN-19 episode | TUM sitting_halfsphere seed 0 | Semantic 89 lost frames, Matched 174, GDOR 0; GDOR certificate pass |
| DYN-16 | Seed-0, 40 shared held-out views | Static PSNR: 19.068 dB Semantic, 20.154 dB GDOR |

## Evidence Boundary

The public artifact supports the concrete guarded recovery and separate
tracking/mapping routing mechanism against same-source local baselines. It does
not claim:

- an official DyPho-SLAM reproduction or protocol-matched superiority;
- universal sequence or metric superiority;
- multi-seed mapping superiority;
- archived empirical zero-overlap counts for DYN-15 through DYN-18;
- independent zero-ghost or zero-contamination proof.

The source release implements the newer tracking-to-mapping overlap audit, and
the completed DYN-19 episode diagnostic reports zero observed recovered-support
overlap with map admission. That audit is a route-invariant diagnostic, not an
independent ghost-contamination or complete-background ground truth.
