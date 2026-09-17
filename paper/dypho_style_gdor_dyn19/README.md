# GDOR/DYN-19 DyPho-Style Evidence Package

This directory defines the GDOR-specific evidence freeze used by the installed
`dypho-slam-figure-automation` skill.

The anchor is the completed DYN-19 release, not the older Protocol-300
failed-boundary placeholder. Source tables are generated only from the published
90-run main, 120-run ordered-ablation, and 72-cell mapping CSV files:

```bash
python3 paper/scripts/build_dypho_gdor_dyn19_tables.py
dypho-pixel-skill mvp-package \
  --root /home/slam/.codex/skills/dypho-slam-figure-automation \
  --freeze-record paper/dypho_style_gdor_dyn19/protocol/freeze_record.yaml \
  --output paper/dypho_style_gdor_dyn19/mvp_package
```

Evidence boundaries:

- DYN-19 is an ordered ablation, not a factorial design.
- Mapping PSNR/SSIM are held-out static-region diagnostics.
- Table II preserves DG-SLAM geometry and DyPho-SLAM/DynaSLAM qualitative
  mapping evidence in native external-report rows; incompatible DYN-19 metric
  cells remain blank and are not ranked.
- Completeness and ghost-risk fields are proxies, not independent truth.
- External-report rows are not local reruns or protocol-matched comparisons.
- The FlowParse-style tracking context contains the full eleven-baseline suite,
  but local GDOR temporal trajectory dispersion is still missing.
- The FlowParse-style runtime context is descriptive until local per-stage
  tracking/mapping latency and FPS are exported.
- Protocol-300 Photo-SLAM/SplaTAM assets are retained as local diagnostic assets;
  they are not DYN-19 protocol matches.
- `mvp_package/tables/table1_external_report.csv` is a retained bootstrap artifact,
  not the manuscript source: its ORB-SLAM3 `fr3/w/half` value conflicts with the
  user-provided TeX and it omits required comparison rows.
- The generated MVP remains draft-only until human provenance review.
