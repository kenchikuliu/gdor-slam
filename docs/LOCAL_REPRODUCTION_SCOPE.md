# Local Comparison Scope

The clean local comparison uses seed 0 on four RGB-D sequences. The baseline is
explicitly named **DyPho-compatible local reproduction**; it is not the official
DyPho-SLAM implementation.

| Sequence | Compatible ATE [m] | GDOR ATE [m] | ATE reduction |
| --- | ---: | ---: | ---: |
| `tum_walking_xyz` | 0.058988 | 0.021431 | 63.669% |
| `tum_walking_halfsphere` | 1.190511 | 0.030602 | 97.430% |
| `bonn_person_tracking` | 0.603660 | 0.038859 | 93.563% |
| `bonn_crowd3` | 1.522230 | 0.037183 | 97.557% |

GDOR has lower translational RPE on all four local comparisons and removes the
non-zero compatible-baseline tracking failure rates on the three affected
sequences. Rotational RPE is not uniformly improved: Bonn
`person_tracking` is a documented counterexample.

These values are a local-reproduction result, not an official-baseline result.
The complete source, result, trajectory, camera, and point-cloud provenance is
kept in the private experiment asset ledger rather than in this source release.
