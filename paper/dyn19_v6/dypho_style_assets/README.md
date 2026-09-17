# DYN-19 DyPho-Style Draft Assets

This package is generated with the installed `dypho-slam-figure-automation`
skill. It contains actual DyPho-style raster table drafts, source hashes,
automated audit reports, and pending human-review forms.

## Evidence Boundary

- Table I: local DYN-19 three-seed tracking rows plus all 11 method rows from
  the registered DyPho Table I, each visibly labeled `[ext]` and recorded as
  `external-report`.
- Table II: local 72-cell static-region mapping diagnostic. DyPho-SLAM has no
  numeric PSNR/SSIM mapping row in the checked source.
- Table III: local application end-to-end runtime, the manuscript-matched mean
  per-run failure rate, and route-certificate status across all 90 DYN-19 main
  runs. No tracking and mapping runtime split is invented.
- All generated assets remain drafts until the review YAML files are completed
  by a human reviewer and the skill `release` command succeeds.

## Reproduce

From the repository root:

```bash
python3 paper/dyn19_v6/scripts/prepare_dypho_style_assets.py

DYPHO=/home/slam/.codex/skills/dypho-slam-figure-automation
$DYPHO/.venv/bin/dypho-pixel-skill validate-input \
  --root "$DYPHO" --kind freeze \
  --input paper/dyn19_v6/dypho_style_assets/freeze_record.yaml \
  --output paper/dyn19_v6/dypho_style_assets/reports/freeze_validation.json

$DYPHO/.venv/bin/dypho-pixel-skill mvp-package \
  --root "$DYPHO" \
  --freeze-record paper/dyn19_v6/dypho_style_assets/freeze_record.yaml \
  --output paper/dyn19_v6/dypho_style_assets/mvp

$DYPHO/.venv/bin/dypho-pixel-skill build \
  --root "$DYPHO" \
  --manifest paper/dyn19_v6/dypho_style_assets/build.yaml
```

The `mvp/` subtree is the skill-generated gap and self-check package for the
current DYN-19 freeze. Its literal importer and current-protocol rerun queue are
not the global baseline inventory. The final Table I generator imports every
real method row from the registered source by hash, relabels source `Ours` as
`DyPho-SLAM [ext]`, and keeps all external rows outside local aggregates.
The bootstrap external CSVs are therefore not the final paper tables.

The separately retained Protocol-300 package already contains complete local
SplaTAM/Photo-SLAM tracking, mapping, runtime, and 20-panel Fig. 4 evidence.
Its historical `Ours` row is failed-boundary evidence and cannot represent
DYN-19 Full. The figure availability report distinguishes that completed
historical package from the still-missing current-method co-registered cell.

The companion `../dypho_supplemental_assets/` build contains the FlowParse-
minimum two-sequence ordered ablation and the broad descriptive runtime table.
