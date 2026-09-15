# DYN-20 T+M Fresh Held-Out Result

Completed: September 15, 2026  
Experiment ID: `DYN-20_TPLUSM_FRESH_HELDOUT_20260915`  
Source commit: `a927a7959b2e4febedac13df01780107fe463e6b`  
Decision: **No-Go; T+M is not promoted**

## Completion and provenance

All 48 frozen runs completed in the new result root:

`/mnt/nas_datasets/slam-experiments/DynaGS-SLAM/dyn20_tplusm_fresh_heldout_run01_20260915`

The execution plan has SHA-256
`3c551b08924db1f60448b61c0b8b393435efa5d2bed21392d599275f87b67cbe`.
It differs bytewise from the immutable registered plan because `created_at`,
the output root, and task run paths are new. After removing `created_at` and
replacing each output-root prefix with a common placeholder, both plans have
the same canonical SHA-256
`04444ae87ea9189ad550e1fa0355d11978ae5bab38b68a9606bd4e8c7b042f28`.

The result audit found:

- 48 plan tasks, 48 unique result identities, and 48 discovered manifests;
- no missing, duplicate, substituted, or extra result identities;
- exact source, runtime snapshot, asset-contract, and execution-plan binding
  for every run;
- exact agreement between every `result.json` and `all_results.json`;
- exact certificate accounting against every `run_summary.json` and
  `frame_metrics.csv`;
- recomputed aggregates equal to the mechanical DYN-20 report.

The original runner issued each route certificate immediately before its
final manifest status write. Therefore the certificate's manifest digest
binds the reconstructable pre-final `status=running` payload rather than the
retained `status=complete` payload. All 48 digests reproduce exactly after
removing `return_code` and `finished_at` and restoring `status=running`; the
result manifest separately hashes all 48 retained final manifests. The runner
has been corrected for future experiments without rewriting these raw results.

## Aggregate result

Each row contains 12 runs: four held-out sequences and seeds 0, 1, and 2.

| Configuration | ATE RMSE (cm) | 1-frame translation RPE (cm) | Failure rate | Runtime (s/run) |
| --- | ---: | ---: | ---: | ---: |
| Full | **2.464** | 1.348 | 3.072% | 66.30 |
| Semantic | 3.959 | 1.330 | 5.629% | **39.77** |
| T+M | 4.014 | **1.228** | **2.723%** | 55.61 |
| MapMatched | 5.090 | 1.392 | 10.270% | 45.08 |

T+M passes all 12 non-vacuous route certificates with 518,737,122 recovered
support audit pixels and zero mapping-overlap pixels. It also wins paired ATE
against Full in 7/12 cells and keeps mean failure rate within the declared
tolerance. These facts do not override the failed aggregate ATE clauses.

## Frozen gate adjudication

| Clause | Result |
| --- | --- |
| Complete 48-run evidence | Pass |
| T+M route certificates, 12/12 non-vacuous zero-overlap | Pass |
| T+M mean ATE no worse than Full | **Fail**: 4.014 cm vs. 2.464 cm |
| T+M mean ATE lower than Semantic | **Fail**: 4.014 cm vs. 3.959 cm |
| T+M paired ATE wins over Full at least 7/12 | Pass: 7/12 |
| T+M failure rate within 0.5 percentage points of Full | Pass: 2.723% vs. 3.072% |

The largest candidate regression occurs on `tum_walking_rpy`, where T+M mean
ATE is 10.307 cm, compared with 3.146 cm for Full and 8.359 cm for Semantic.
The predeclared gate therefore returns `status=fail` and
`tplusm_promoted=false`.

## Decision boundary

DYN-20 is a complete and valid negative held-out result. T+M remains useful as
a DYN-19 consumed-data candidate and has favorable held-out translation RPE
and failure rate, but it is not the validated successor or default method.
This split must not be used for threshold tuning, selective reruns, or
post-hoc promotion.

The machine-readable result record is `DYN20_RESULTS.json`. The retained NAS
root contains `DYN20_RESULT_MANIFEST.json` and `SHA256SUMS` covering the
decision-bearing top-level files and five evidence files for each of 48 runs.
