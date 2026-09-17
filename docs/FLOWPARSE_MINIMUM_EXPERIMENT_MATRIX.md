# FlowParse-Minimum Experiment Coverage

Updated: September 17, 2026

This matrix translates the minimum experiment surface in
`/home/slam/Desktop/FlowParse-SLAM.tex` into source-labeled project evidence.
Strict same-protocol comparison is not required, so published rows are allowed
when they are visibly marked `[ext]` and kept outside local aggregate claims.

## Coverage Summary

| ID | Required surface | Current status | Authoritative evidence |
| --- | --- | --- | --- |
| M1 | Broad tracking on `fr3/w/xyz`, `w/half`, `w/static`, `s/half` | Complete, source-labeled | `paper/dyn19_v6/dypho_style_assets/data/table1_tracking.csv` |
| M2 | Module ablation on `w/xyz` and `w/half` | Complete, local three-seed ordered ablation | `paper/dyn19_v6/dypho_supplemental_assets/data/table2_ablation.csv` |
| M3 | Current-method quantitative mapping | Complete local diagnostic | `paper/dyn19_v6/dypho_style_assets/data/table2_mapping.csv` |
| M4-M5 | Local SplaTAM/Photo-SLAM tracking and mapping metrics | Complete historical Protocol-300 package | `/home/slam/experiments/dypho_skill_full_rerun_20260729/metrics/protocol300/` |
| M6 | Four-row qualitative mapping comparison | Historical package complete; current DYN-19 Full cell missing | `/home/slam/experiments/dypho_skill_full_rerun_20260729/figure_evidence_protocol300_v2/` |
| M7 | Broad tracking/mapping runtime context | Complete, source-labeled and descriptive | `paper/dyn19_v6/dypho_supplemental_assets/data/table3_runtime_broad.csv` |
| M8 | Local baseline end-to-end runtime | Complete historical Protocol-300 package | `/home/slam/experiments/dypho_skill_full_rerun_20260729/metrics/protocol300/tables/table3.csv` |
| M9 | Current DYN-19 Full co-registered with local SplaTAM/Photo-SLAM | Missing current-method cell | New run or render required under the frozen baseline package |

The machine-readable version with exact methods, scopes, evidence types, and
claim boundaries is `docs/FLOWPARSE_MINIMUM_EXPERIMENT_MATRIX.csv`.

## Broad Tracking

The draft Table I now includes the full registered comparison breadth:
ORB-SLAM3, Dyna-SLAM, NICE-SLAM, ESLAM, RoDyn-SLAM, SplaTAM, GS-SLAM,
GassiDy, DGS-SLAM, Photo-SLAM, DyPho-SLAM, and the three local DYN-19 rows.
All published methods carry `[ext]`; no best/second-best formatting is applied
across the mixed protocols.

## Local Baseline Evidence

Protocol-300 is already complete for SplaTAM, Photo-SLAM, DynaGS semantic, and
a historical `Ours` failed-boundary ablation over `fr3/w/xyz`, `fr3/w/half`,
Bonn `ps_track`, and Bonn `r3`. It contains:

- 16 verified method-sequence cells;
- seed 0 with a 300-frame horizon;
- tracking ATE, full-frame PSNR/SSIM diagnostics, and end-to-end wall time;
- 20 real 640x480 qualitative panels with zero pending entries.

This evidence is usable as a separately labeled historical local package. Its
`Ours` row must remain failed-boundary evidence and cannot be presented as the
current DYN-19 Full method.

## Remaining Gap

The only material FlowParse-style comparison gap is the current-method local
package: DYN-19 Full has not yet been co-registered with SplaTAM and Photo-SLAM
under the Protocol-300 tracking, mapping, runtime, and Fig. 4 evidence bundle.
Closing it requires a new DYN-19 Full run or render under that frozen package;
renaming the historical `Ours` panels is prohibited.

## Claim Boundary

- External rows are source-reported descriptive context, not matched reruns.
- Protocol-300 mapping metrics include dynamic foreground and are diagnostics.
- Broad runtime rows mix hardware and timing scopes and are not a speed rank.
- DYN-19 module rows are ordered configurations, not factorial effects.
- Every generated PNG remains a draft until human review is completed.
