# GDOR-SLAM Paper Artifact

This directory contains the public, evidence-linked artifact for:

> GDOR-SLAM: Guarded Dynamic Observation Recovery for RGB-D Gaussian SLAM

The updated PDF is a ten-page TMM pre-submission manuscript rebuilt on
September 17, 2026.
It uses an unfinished author placeholder and is not a publication record.

## Contents

- `GDOR-SLAM_TMM_v5r6_preprint.pdf`: manuscript PDF with DYN-19 and clearly
  labeled published external-report positioning.
- `GDOR-SLAM_TMM_v5r5_preprint.pdf`: earlier DYN-19-integrated manuscript PDF
  retained for provenance.
- `GDOR-SLAM_TMM_v5r4_preprint.pdf`: earlier evidence-boundary revision retained
  for provenance.
- `GDOR-SLAM_TMM_v5r3_preprint.pdf`: earlier frozen manuscript PDF retained
  for provenance.
- `main.tex`, `sections/`, `references.bib`: matching LaTeX source.
- `figures/`: the five figures used by the manuscript source.
- `data/`: claim-bearing CSV files for DYN-15 through DYN-19, including the
  90 main cells, 120 mechanism cells, 72 mapping cells, release provenance,
  a separately labeled published external-report registry, and the
  `flowparse_*_context.csv` source ledgers transcribed from the supplied
  FlowParse-SLAM comparison tables.
- `scripts/`: deterministic figure generation and evidence validation.
- `reports/evidence_validation.json`: computed values checked against the text.
- `dypho_style_gdor_dyn19/`: DYN-19-anchored DyPho-style Table I/II/III drafts,
  freeze record, provenance, audits, external-report imports, baseline queues,
  FlowParse minimum experiment matrix, and Fig.4 release blockers.
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

The validator recomputes the principal means, medians, W/T/L counts, route
certificate totals, and mapping comparisons from the public CSV files, checks
that they occur in the manuscript source, verifies the expected five figures
and three tables, and writes `paper/reports/evidence_validation.json`.

The validator requires Python 3 and NumPy. Regenerating Figures 1 and 2 also
requires Matplotlib and Pillow:

```bash
python3 paper/scripts/generate_gdor_tmm_figures.py
```

Regenerate and audit the DyPho-style quantitative support package with:

```bash
python3 paper/scripts/build_dypho_gdor_dyn19_tables.py
dypho-pixel-skill mvp-package \
  --root /home/slam/.codex/skills/dypho-slam-figure-automation \
  --freeze-record paper/dypho_style_gdor_dyn19/protocol/freeze_record.yaml \
  --output paper/dypho_style_gdor_dyn19/mvp_package
dypho-pixel-skill build \
  --root /home/slam/.codex/skills/dypho-slam-figure-automation \
  --manifest paper/dypho_style_gdor_dyn19/build_tables.yaml \
  --output /home/slam/DynaGS-SLAM-dyn19-publish/paper/dypho_style_gdor_dyn19/table_drafts
```

The generated table drafts are not released assets. Their audit reports pass,
but the review YAML files remain pending. Fig.4 is blocked until current DYN-19
GDOR renders exist for all four locked comparison rows.

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
| DYN-19 main | 90 runs, ten sequences, three seeds | Mean ATE: 10.119 cm Semantic, 7.392 cm MapMatched, 4.930 cm Full |
| DYN-19 mechanisms | 120 runs, ordered ablation | Non-monotonic adjacent contrasts; Full-NoM has 30/30 expected certificate failures |
| DYN-19 mapping | 72 cells, four scenes and three seeds | Online static PSNR: 18.866 dB Semantic, 18.990 dB MapMatched, 19.763 dB Full |
| DYN-19 episode | TUM sitting_halfsphere seed 0 | Semantic 89 lost frames, Matched 174, GDOR 0; GDOR certificate pass |
| DYN-16 | Seed-0, 40 shared held-out views | Static PSNR: 19.068 dB Semantic, 20.154 dB GDOR |

## Evidence Boundary

The public artifact supports the concrete guarded recovery and separate
tracking/mapping routing mechanism against same-source local baselines. It does
not claim:

- an official DyPho-SLAM reproduction or protocol-matched superiority;
- universal sequence or metric superiority;
- independent component or interaction effects from the ordered ablation;
- archived empirical zero-overlap counts for DYN-15 through DYN-18;
- independent zero-ghost or zero-contamination proof.

The completed DYN-19 release reports 30/30 passing Full overlap certificates
and a 72-cell multi-seed mapping diagnostic. The certificate establishes the
declared route invariant; the occluded-background fields remain proxies rather
than independent ghost-contamination or complete-background ground truth.

The manuscript also includes paper-reported DynaSLAM, DG-SLAM, and DyPho-SLAM
values where the original papers provide them. Those rows preserve the source
dataset, metric, and statistic, are marked as external reports, and are not a
protocol-matched ranking. In particular, DG-SLAM reports BONN geometry metrics,
whereas DyPho-SLAM and DynaSLAM provide qualitative mapping results without a
PSNR/SSIM/LPIPS mapping table.
