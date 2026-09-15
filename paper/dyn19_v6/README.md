# GDOR-SLAM DYN-19 Manuscript Revision

This directory is the September 15, 2026 DYN-19 manuscript and evidence
package. It is a new revision derived from the frozen August 8, 2026 v5r3
manuscript. The frozen PDF remains at
`../GDOR-SLAM_TMM_v5r3_preprint.pdf` and is not modified by this package.

## Contents

- `main.tex`, `sections/`: DYN-19 manuscript source.
- `data/dyn19_tracking_sequence_means.csv`: 70 path-sanitized sequence means
  covering the 90-run main phase and 120-run ordered mechanism phase.
- `data/dyn19_route_certificates.csv`: all 210 route-certificate cells with
  bound plan, manifest, summary, frame-log, source, and release hashes.
- `data/dyn19_mapping_cells.csv`: all 72 common-view mapping metric cells with
  path fields removed and evidence hashes retained.
- `data/dyn19_provenance.json`: source-release and derived-file hashes.
- `scripts/export_dyn19_public_evidence.py`: deterministic importer from the
  retained DYN-19 release.
- `scripts/validate_dyn19_evidence.py`: manuscript-number, denominator,
  certificate, hash, figure, and table validator.
- `scripts/generate_dyn19_overview.py`: deterministic Figure 1 generator.
- `reports/`: generated validation and figure-provenance reports.
- `EVIDENCE_MANIFEST.md`: claim/evidence boundary ledger.
- `FIGURE_TABLE_MANIFEST.yaml`: figure/table-to-evidence mapping.
- `SHA256SUMS`: hashes for every published file in this revision package.

The package references `../IEEEtran.cls`, `../IEEEtran.bst`,
`../references.bib`, and the existing method figure in `../figures/`.

## Validate

From the repository root:

```bash
python3 paper/dyn19_v6/scripts/validate_dyn19_evidence.py
```

To reproduce the public CSV/JSON files from the retained release:

```bash
python3 paper/dyn19_v6/scripts/export_dyn19_public_evidence.py \
  /mnt/nas_datasets/slam-experiments/DynaGS-SLAM/dyn19_release_20260914_r1
```

To regenerate Figure 1:

```bash
python3 paper/dyn19_v6/scripts/generate_dyn19_overview.py
```

## Build

From `paper/dyn19_v6`:

```bash
bash build.sh
```

The script writes `GDOR-SLAM_TMM_v6_dyn19_preprint.pdf` in this directory and
copies the same versioned PDF to `paper/`. It does not overwrite v5r3.

## Claim Boundary

DYN-19 supports aggregate same-source tracking improvement, an exact 30/30
non-vacuous Full route-separation certificate, and positive multi-seed
common-view reconstruction diagnostics. It does not establish factorial
component causality, universal sequence or mapping superiority, official
DyPho-SLAM superiority, external SOTA, or independent 3D ghost truth. T+M is
the strongest consumed-data candidate and requires a new held-out experiment
before promotion.
