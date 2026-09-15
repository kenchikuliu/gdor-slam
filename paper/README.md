# GDOR-SLAM Paper Artifact

This directory contains the public, evidence-linked artifact for:

> GDOR-SLAM: Guarded Dynamic Observation Recovery for RGB-D Gaussian SLAM

The PDF is a nine-page TMM pre-submission manuscript frozen on August 8, 2026.
It uses an unfinished author placeholder and is not a publication record.
It predates the completed September 14, 2026 DYN-19 experiment and must not be
described as containing DYN-19 results.

## Contents

- `GDOR-SLAM_TMM_v5r3_preprint.pdf`: frozen manuscript PDF.
- `main.tex`, `sections/`, `references.bib`: matching LaTeX source.
- `figures/`: the four figures used by the manuscript.
- `data/`: aggregate claim-bearing CSV files for DYN-15 through DYN-18.
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
checks that they occur in the manuscript source, verifies the expected four
figures and three tables, and writes `paper/reports/evidence_validation.json`.

The validator requires Python 3 and NumPy. Regenerating Figures 1 and 2 also
requires Matplotlib and Pillow:

```bash
python3 paper/scripts/generate_gdor_tmm_figures.py
```

Figures 3 and 4 are archived experiment visualizations and are included as
fixed evidence images rather than reconstructed from unpublished trajectories
or rendered frame directories.

## Result Summary

| Evidence | Scope | Main result |
| --- | --- | --- |
| DYN-15 | 63 runs, TUM walking_xyz plus Bonn6 | All7 mean ATE: 11.331 cm Semantic, 3.798 cm GDOR |
| DYN-18 | 36 cells, frozen TUM4 validation | Mean ATE: 9.746 cm Semantic, 3.415 cm GDOR; 9/12 paired wins |
| DYN-17 | 9-run map-matched control | Mean ATE: 7.816 cm Matched, 4.053 cm GDOR |
| DYN-16 | Seed-0, 40 shared held-out views | Static PSNR: 19.068 dB Semantic, 20.154 dB GDOR |

## Evidence Boundary

The public artifact supports the concrete guarded recovery and separate
tracking/mapping routing mechanism against same-source local baselines. It does
not claim:

- an official DyPho-SLAM reproduction or protocol-matched superiority;
- universal sequence or metric superiority;
- multi-seed mapping superiority;
- archived empirical zero-overlap counts for DYN-15 through DYN-18.

The source release implements the newer tracking-to-mapping overlap audit, but
the archived experiments predate that counter. This distinction is retained in
the manuscript and public evidence manifest.

The later DYN-19 result is recorded separately in
[`docs/DYN19_RESULTS.md`](../docs/DYN19_RESULTS.md). It completes a 210-run
tracking matrix, 30/30 Full zero-overlap certificates, and a 72-cell multi-seed
mapping diagnostic. Integrating that evidence requires a new manuscript
revision and regenerated evidence manifest rather than editing the frozen PDF
in place.
