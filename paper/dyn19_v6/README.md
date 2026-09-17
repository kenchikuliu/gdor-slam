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
- `data/dypho_style_tracking_comparison.csv`: four-sequence tracking comparison
  combining local DYN-19/DYN-18 rows with DyPho-SLAM published ATE references.
- `data/dypho_style_mapping_comparison.csv`: compact summary of the 72-cell
  local mapping diagnostic and its external-reference boundary.
- `data/external_quantitative_reference.csv`: audited external tracking and
  mapping reference ledger; external rows are not protocol-matched reruns.
- `dypho_style_assets/`: skill-built Table I-III PNG drafts, CSV inputs,
  source/provenance manifests, automated audits, and pending review forms.
- `dypho_supplemental_assets/`: a second skill build for the two-sequence
  ordered ablation and broad source-reported runtime context.
- `data/dyn19_provenance.json`: source-release and derived-file hashes.
- `scripts/export_dyn19_public_evidence.py`: deterministic importer from the
  retained DYN-19 release.
- `scripts/validate_dyn19_evidence.py`: manuscript-number, denominator,
  certificate, hash, figure, and table validator.
- `scripts/validate_dypho_comparison.py`: local source-to-summary validator for
  the standalone tracking/mapping comparison package.
- `scripts/generate_dyn19_overview.py`: deterministic Figure 1 generator.
- `reports/`: generated validation and figure-provenance reports.
- `EVIDENCE_MANIFEST.md`: claim/evidence boundary ledger.
- `FIGURE_TABLE_MANIFEST.yaml`: figure/table-to-evidence mapping.
- `../../docs/DYPHO_STYLE_QUANTITATIVE_COMPARISON.md`: tracking/mapping
  comparison package and safe claim wording.
- `../../docs/FLOWPARSE_MINIMUM_EXPERIMENT_MATRIX.md`: requirement-by-
  requirement coverage, including the retained Protocol-300 local baselines.
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

To regenerate the DyPho-style raster tables:

```bash
python3 paper/dyn19_v6/scripts/prepare_dypho_style_assets.py

DYPHO=/home/slam/.codex/skills/dypho-slam-figure-automation
$DYPHO/.venv/bin/dypho-pixel-skill mvp-package \
  --root "$DYPHO" \
  --freeze-record paper/dyn19_v6/dypho_style_assets/freeze_record.yaml \
  --output paper/dyn19_v6/dypho_style_assets/mvp

$DYPHO/.venv/bin/dypho-pixel-skill build \
  --root "$DYPHO" \
  --manifest paper/dyn19_v6/dypho_style_assets/build.yaml

$DYPHO/.venv/bin/dypho-pixel-skill build \
  --root "$DYPHO" \
  --manifest paper/dyn19_v6/dypho_supplemental_assets/build.yaml
```

The generated tables are drafts. Their review YAML files remain unapproved
until an identified human reviewer completes the six required checks. The
skill-generated MVP subtree records the current-freeze rerun queue, full
baseline-library queue, Fig. 4 external-render policy, missing artifacts, and
package self-check. The global evidence ledger separately records the complete
historical Protocol-300 SplaTAM/Photo-SLAM package.

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
the strongest DYN-19 consumed-data candidate, but the completed DYN-20 held-out
gate did not promote it.

The DyPho-style comparison package reports published DyPho-SLAM tracking
values as external positioning context and does not require a protocol-matched
official rerun. DyPho-SLAM numeric mapping values were not reported in the
checked source, so the mapping comparison remains local-only.
