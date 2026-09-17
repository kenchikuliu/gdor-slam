# DYN-19 DyPho-Style Draft Assets

This package is generated with the installed `dypho-slam-figure-automation`
skill. It contains actual DyPho-style raster table drafts, source hashes,
automated audit reports, and pending human-review forms.

## Evidence Boundary

- Table I: local DYN-19 three-seed tracking rows plus the published DyPho-SLAM
  row labeled `external-report`.
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

The `mvp/` subtree is the skill-generated gap and self-check package. Its
external-row importer performs literal method matching, so it records the
requested `DyPho-SLAM` row as missing because the reference CSV names the
paper's own row `Ours`. The final Table I generator binds that exact source row
by hash, relabels it `DyPho-SLAM [ext]`, and keeps it outside local aggregates.
The bootstrap external CSVs are therefore not the final paper tables.

The figure availability report records which locked Fig. 1-4 assets can be
built from retained evidence and which require new exports or real baselines.
