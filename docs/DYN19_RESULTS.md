# DYN-19 Results

Release date: September 14, 2026

Public result assets:
[`dyn19-results-20260914`](https://github.com/kenchikuliu/gdor-slam/releases/tag/dyn19-results-20260914)

Source commit for all claim-bearing tracking runs:
`14c6b2ca696a7eacda8f193e5c8a0a44e55d527e` (`14c6b2c`)

## Completed Matrix

| Phase | Matrix | Status |
| --- | --- | --- |
| Main tracking | 3 configs x 10 sequences x 3 seeds | 90/90 complete |
| Mechanism tracking | 4 configs x 10 sequences x 3 seeds | 120/120 complete |
| Mapping integrity | 4 scenes x 3 seeds x 3 configs x 2 pose modes | 72/72 complete |

The tracking denominator includes `tum_sitting_halfsphere`.

The main-phase `dyn19_full` recovered-support aggregate has 30/30
certificates with `status=pass`, zero observed mapping leaks, and
`safety_status=pass`. `dyn19_semantic` and `dyn19_mapmatched` are retained as
their recorded `vacuous` safety certificates where no recovered support was
observed.

The mechanism phase has 30 passing certificates for each of
`dyn19_temporal_map`, `dyn19_temporal_flow_map`, and
`dyn19_temporal_flow_risk_map`. The 30 `dyn19_full_no_mapping` certificate
failures are intentional negative counterfactual evidence: recovered tracking
support is allowed into persistent mapping and the overlap invariant is
expected to fail.

## Claim-Bearing Aggregate Results

Across the 30 paired main cells, Full reduces mean ATE from 10.119 cm to
4.930 cm relative to Semantic. The paired-cell median gain is 0.213 cm, the
cell W/T/L count is 19/0/11, and the sequence-mean count is 7/0/3. The largest
failure-case rescue is Bonn `crowd2`, from 42.500 cm to 11.080 cm.

Relative to MapMatched, Full reduces the mean from 7.392 cm to 4.930 cm, but
the paired-cell median gain is only 0.016 cm, with 16/0/14 cell W/T/L and
4/0/6 sequence-mean W/T/L. This aggregate is driven by TUM
`sitting_halfsphere`, from 36.775 cm to 4.864 cm. The correct interpretation is
failure-regime rescue, not uniform improvement.

The ordered tracking means are non-monotonic: Semantic 10.119 cm, MapMatched
7.392 cm, T+M 3.859 cm, T+F+M 6.377 cm, T+F+R+M 4.907 cm, Full 4.930 cm, and
Full-NoM 4.060 cm. Full-NoM is not a positive replacement: all 30 certificates
fail by design, with 90,790,462 recovered-support pixels admitted to mapping.

For the 12 online mapping cells per method, Full static PSNR is 19.763 dB,
compared with 18.866 dB for Semantic and 18.990 dB for MapMatched. The Full
gain over Semantic has a 0.544 dB paired median and 9/0/3 W/T/L; the gain over
MapMatched has a 0.515 dB paired median and 11/0/1 W/T/L. These are static-view
rendering diagnostics, not independent ghost/completeness measurements.

Compact public rows used by the paper are under `paper/data/`:

- `dyn19_main_90_cells.csv`;
- `dyn19_mechanism_120_cells.csv`;
- `dyn19_mapping_72_cells.csv`;
- `dyn19_release_provenance.json`.

## Release Assets

The release contains:

- main tracking archives for `dyn19_semantic`, `dyn19_mapmatched`, and
  `dyn19_full`;
- mechanism tracking archives for all four mechanism configurations;
- mapping archives for `tum_walking_xyz`, `tum_sitting_halfsphere`,
  `bonn_crowd2`, and `bonn_person_tracking2`;
- a metadata archive with frozen plans, reports, aggregate CSV files, and
  `all_results.json`;
- `DYN19_RELEASE_MANIFEST.json` and `SHA256SUMS`.

The archives contain the recorded trajectories, static masks, Gaussian maps,
renders, per-frame logs, certificates, manifests, and metric artifacts.
Archive paths are relative to the corresponding phase/configuration root.

## Claim Boundary

DYN-19 is an **ordered ablation**, not a factorial or fractional-factorial
design. Its adjacent contrasts do not identify independent causal effects for
temporal depth, flow, risk, adaptive FAST, or mapping weight, and no
interaction effects are reported.

The occluded-background fields in the mapping package are
`occluded-background proxy` diagnostics. They are not independent
ghost-contamination ground truth or complete-background ground truth. The
independent-truth gate remains pending until approved external or blind human
annotations, agreement/adjudication, and revealed-background certificates are
available.

DG-SLAM and DynaSLAM remain audit-only external baselines until their
protocol-matched eligibility gates pass. Their numbers are not promoted into
the DYN-19 main table.
