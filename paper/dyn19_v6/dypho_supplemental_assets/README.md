# DYN-19 DyPho-Style Supplemental Draft Assets

This independent skill build fills two FlowParse-minimum experiment surfaces
that do not fit the semantics of the main three-table package:

- Table II: a local ordered module ablation on `w/xyz` and `w/half`.
- Table III: broad source-reported runtime context plus a visibly scoped local
  DYN-19 Full end-to-end row.

The external runtime rows are descriptive only because hardware, software,
sequence length, and timing scope differ. The local row leaves tracking and
mapping component times as `x` rather than inventing a split.

Regenerate from the repository root:

```bash
python3 paper/dyn19_v6/scripts/prepare_dypho_style_assets.py

DYPHO=/home/slam/.codex/skills/dypho-slam-figure-automation
$DYPHO/.venv/bin/dypho-pixel-skill build \
  --root "$DYPHO" \
  --manifest paper/dyn19_v6/dypho_supplemental_assets/build.yaml
```

The PNGs remain drafts until their generated review YAML files are completed
by an identified human reviewer.
