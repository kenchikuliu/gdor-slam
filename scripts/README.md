# Reproducible Evaluation Scripts

Claim-bearing experiments use `run_reproducible_benchmark.py`. The older
`run_statistical_validation.sh` and `compute_statistics.py` are retained only
for historical diagnostics and must not populate the active paper.

## Deterministic Replay Gates

Before any claim-bearing benchmark, capture shadow-only packets with
`--tracking-only --export-replay-packets --freeze-map-after-frame N`. A valid
freeze must acknowledge idle and stopped LocalMapping, idle and stopped
LoopClosing, stopped GBA, tracking-only mode, and one stable freeze epoch.

Replay every packet without starting YOLO, LocalMapping, LoopClosing, or the
Gaussian mapper:

```bash
./bin/frozen_same_state_replay run/replay_packets/frame_*.yml.gz
```

The current `frozen-same-state-replay-v3` packet includes the complete
Schur-reduced `6x6` pose information, its runtime scale, and the
initialization-only switch. The dynamic replay branch therefore executes the
same prior factor as the online branch. Historical v2 packets did not contain
these fields and are intentionally rejected by the v3 replayer; their fork
results are initialization-only diagnostics, not Full Schur evidence.

Packet export starts after one complete post-freeze settling frame. The first
frame after the freeze can still reference the final pre-freeze tracking
state, so it is excluded by contract rather than silently filtered during
evaluation. Replay scores are checked with
`1e-5 + 1e-5 * max(abs(expected), abs(actual))` tolerance; poses, support
counts, support hashes, and all gate decisions remain exact or retain their
separate strict pose tolerance.

Direct RGB-D raw scores use `1e-4` absolute tolerance because their
nearest-pixel depth lookup is discontinuous at pixel-rounding boundaries. The
direct preference and complete gate decision remain exact, so this tolerance
cannot turn a rejected intervention into an accepted one.

The null-intervention gate requires three independently executed run
directories:

```bash
python3 scripts/verify_null_intervention_equivalence.py \
  run_0 run_1 run_2 \
  --report null_equivalence_report.json
```

It fails on incomplete freeze acknowledgements, replay mismatch, a non-empty
selected intervention ID, a `would-use` ID mismatch, or any trajectory,
tracking-state, keyframe, MapPoint, support, map, or execution-fingerprint
difference. The first mismatch is reported as a field-level diff.

Only after the null gate passes may a one-time candidate initialization be
forked for 1/5/10 frames. Both branches subsequently use their own optimized
previous state and constant-velocity initialization:

```bash
./bin/checkpointed_fork_replay \
  --selection would-use --horizons 1,5,10 \
  --information-scale-multiplier 1 \
  --max-static-information-leverage 0.1 \
  --static-information-leverage-mode cap-only \
  --output forks.csv run_0/replay_packets/frame_*.yml.gz

python3 scripts/evaluate_checkpointed_forks.py forks.csv \
  --ground-truth /path/to/groundtruth.txt \
  --output-json fork_summary.json \
  --output-csv fork_rows.csv
```

Fork evaluation interpolates ground-truth translation and rotation at the
packet timestamps and rejects brackets wider than 0.1 seconds. Relative error
uses the standard `inverse(delta_gt) * delta_estimate` order without trajectory
alignment.

`--information-scale-multiplier` is a development-only covariance-calibration
control applied on top of the information scale serialized in each packet.
Do not select it on held-out sequences. A fork CSV contains exactly one
multiplier so results from different calibration cells cannot be pooled.

`--max-static-information-leverage` calibrates the translation-only Schur
factor against the rotation-marginalized static visual Hessian.
`--static-information-leverage-mode cap-only` preserves generalized
eigenvalues below the supplied value and clips stronger directions.
`normalize-to-target` instead applies one spectrum-preserving scale so the
largest statically supported generalized eigenvalue equals the supplied
target; it can therefore strengthen weak priors as well as temper strong
ones. Directions without static support are removed in both modes. The
default value remains zero, so existing configs and packet replay retain
their prior behavior.

`--gate-policy legacy-conjunction|direct-combined` selects the replayed
admission rule. The default is the policy serialized in the packet; old
packets without this field are `legacy-conjunction`. `direct-combined`
requires valid common RGB-D support, a lower dynamic combined robust cost,
and the packet's frozen translation/rotation innovation bounds. It fails
closed on invalid direct evidence and cannot be enabled through the legacy
bypass flag. In posterior-local-map mode, the posterior matched-support veto
must still pass before Schur commits.

To audit a new policy over every replayable checkpoint rather than prefiltering
with the serialized legacy decision:

```bash
./bin/checkpointed_fork_replay \
  --selection all --gate-policy direct-combined \
  --posterior-local-map --horizons 1,5,10 \
  --static-information-leverage-mode normalize-to-target \
  --max-static-information-leverage 0.025 \
  --output forks.csv run_0/replay_packets/frame_*.yml.gz
```

Here `all` expands checkpoint coverage; `gate.would_use` still controls Schur
candidacy and the posterior veto still controls commits. The frozen
three-seed TUM `walking_xyz` plus Bonn `crowd` study at targets `0.025` and
`0.05` failed its cross-dataset/horizon stop condition, so no held-out run was
launched. See `../MOTION_DIRECT_COMBINED_DEVELOPMENT_20260729.md`.

The fork CSV reports three matched rollouts: static, normal dynamic
propagation, and `velocity_neutral`. The latter applies the same Schur factor
at the checkpoint but preserves the incoming constant-velocity estimate for
the first successor prediction, instead of interpreting the prior correction
as a new physical velocity. Normal velocity updates resume after that
successor frame.

`would-use` prefilters checkpoints using the serialized method-level
intervention diagnostic. `all` includes every replayable checkpoint. In the
legacy rollout mode this is a forced candidate stress test; in
posterior-local-map mode the recomputed gate and posterior veto still control
the method action. Full synchronized mapping remains blocked unless the
method-level fork has enough events and stably improves translation/rotation
RPE, failure rate, and tail risk.

### Development-only recovery episode v2

The corrected scientific stop/go experiment is frozen in
`../MOTION_RECOVERY_EPISODE_V2_PROTOCOL_20260730.md`. First run a Static-only
H=10 scan; this stage does not execute Dynamic or read ground truth:

```bash
./bin/checkpointed_fork_replay \
  --output static_scan.csv --static-scan-only \
  --selection all --prior-subspace translation --horizons 10 \
  --posterior-local-map \
  --max-static-information-leverage 0.025 \
  --static-information-leverage-mode normalize-to-target \
  replay_packets/frame_*.yml.gz

python3 scripts/prepare_recovery_episodes.py static_scan.csv \
  --sequence tum_walking_xyz --seed 0 \
  --output-csv episodes.csv --output-json episodes.json
```

Freeze and hash `episodes.csv` before running either effect branch. The final
replay is restricted to that manifest and computes Static, velocity-neutral
Dynamic, and the true matched-sham contract:

```bash
./bin/checkpointed_fork_replay \
  --output forks.csv --episode-manifest episodes.csv \
  --selection all --prior-subspace translation --horizons 10 \
  --posterior-local-map \
  --max-static-information-leverage 0.025 \
  --static-information-leverage-mode normalize-to-target \
  replay_packets/frame_*.yml.gz

python3 scripts/evaluate_recovery_episode_v2.py forks.csv \
  --episode-manifest episodes.csv \
  --ground-truth /path/to/groundtruth.txt \
  --sequence tum_walking_xyz --seed 0 \
  --output-json summary.json --output-csv rows.csv
```

`Oracle-Delayed-H` chooses one complete H=10 branch using relative SE(3)
endpoint error without trajectory alignment. It is a diagnostic upper bound,
not a method result. Aggregate exactly the two development sequences and
three seeds with `summarize_recovery_episode_v2.py`; fewer than three events
in any cell is insufficient data, not a No-Go.

The registered motion ablation is:

```bash
python3 scripts/run_reproducible_benchmark.py \
  --configs semantic semantic_motion_init_rgbd \
    semantic_motion_ungated semantic_motion_rgbd \
    semantic_motion_rgbd_shadow semantic_motion_rgbd_shuffled \
  --sequences tum_walking_xyz bonn_crowd \
  --seeds 0 1 2
```

`motion_ablation_summary.json` is complete only when all six configurations
use the same seed plan and every planned run succeeds. The ungated row keeps
the Schur information factor but explicitly bypasses all reliability checks.
The internal `semantic_motion_rgbd_shuffled` row is displayed as `Lag-30`. It
uses the full gate with a causal 30-frame temporal shift of the pose proposal
and information matrix.

The DyPho-compatible engineering ablation is registered independently:

```bash
DYNAGS_DATASETS_ROOT=/path/to/datasets \
python3 scripts/run_reproducible_benchmark.py \
  --configs semantic dypho_raw dypho_temporal \
    dypho_feature dypho_compatible dypho_decoupled dypho_flow_guarded \
    dypho_flow_adaptive \
  --sequences tum_walking_xyz tum_walking_rpy \
    tum_walking_halfsphere tum_walking_static \
    bonn_crowd bonn_balloon bonn_person_tracking \
  --seeds 0 1 2 3 4 \
  --heldout-stride 0
```

Its aggregate rows include mask-pose prediction, temporal correction,
static-mask support, adaptive FAST threshold, extracted-feature statistics,
and flow-guard interventions. All `dypho_*` presets bind a hard static mask to Gaussian
keyframes. Raw/Feature use the raw semantic-plus-flow mask,
Temporal/Compatible use the refined mask, and Decoupled restores pixels only
for tracking while preserving the raw mask for mapping. Flow-Guarded adds a
full-image residual-flow veto to that recovery, retains the same raw mapping
mask, and activates adaptive feature replenishment for at least five frames
when the previous frame falls below 150 tracking inliers. The gate is armed
only after a valid positive inlier history exists, and the aggregate records
the number of active frames. Mapping-weight
coverage and rejection counters are included in `run_summary.json` and
`aggregate.csv`. For common-view mapping results, use
`scripts/summarize_mapping_ablation.py --profile dypho`; per-run held-out
frames are not a substitute for a frozen cross-method manifest. For the
focused Raw/Decoupled/Flow-Guarded engineering comparison, use the strict
`--profile dypho-core` matrix. Both profiles fail when any required method,
pose mode, or frozen frame is missing.

## Paired Trajectory Benchmark

```bash
python3 scripts/run_reproducible_benchmark.py \
  --configs semantic semantic_motion_shadow semantic_motion full \
  --sequences tum_walking_xyz bonn_crowd \
  --seeds 0 1 2 3 4 \
  --gpus 0 1 \
  --jobs 2 \
  --heldout-stride 0 \
  --output-root /path/to/new/benchmark_root
```

All configurations for one sequence/seed execute serially under one GPU lock.
The runner rejects dirty sources, duplicate identities, reused output roots,
and missing inputs. Each run records full-frame trajectories, per-frame state,
ATE/RPE artifacts, failure rate, end-to-end time, and source/input hashes.
Failed planned runs remain in the claim-gate denominator.

The runner also freezes `benchmark_plan.json` before scheduling, writes its
SHA-256 to `benchmark_plan.sha256`, and binds every run manifest to that exact
plan and task. Binary, runtime-library, vocabulary, sequence, association,
ground-truth, camera, Gaussian, mask-config, and YOLO TensorRT engine hashes
must still match at run start. YOLO-enabled configurations fail closed if the
engine is missing, empty, unloadable, or inference raises an error.

`--sync-local-mapping` and `--frame-period-ms N` are scheduling-sensitivity
diagnostics, not claim-runner defaults. The former waits for LocalMapping to
become fully idle; the latter enforces a minimum wall-clock frame period.
Consumed Bonn probes showed that either can turn a tens-of-seconds run into a
minutes-long run by allowing substantially more local BA. Results across
these execution semantics must not be pooled.

`semantic_motion_shadow` performs the complete motion-prior computation and
records `would_use_motion_priors`, but always restores the static pose branch.
Use it to estimate extra-compute and background-thread scheduling effects
before attributing a Semantic+Motion trajectory difference to the prior.
Runs with valid hypotheses also emit `motion_prior_counterfactual.csv`; the
runner evaluates both hypotheses against adjacent ground-truth poses without
trajectory alignment, so gate quality can be audited on the same map state.
Pairs outside the official GT temporal range are explicitly counted and
excluded rather than extrapolated.

`semantic_motion_init_rgbd` and its matched shadow are development-only. They
use the motion estimate as a search initialization, run the final pose
optimization on static MapPoint reprojection residuals without an external
SE(3) factor, and require consensus, combined RGB-D validation, lower robust
cost on exact common MapPoint support, and at least 3 mm translation
innovation. The consumed-data study did not establish an end-to-end benefit;
see `MOTION_INIT_ONLY_DEVELOPMENT_20260728.md`.

The counterfactual log also records an intervention-neutral diagnostic:
the exact MapPoint/feature correspondences independently recovered by both
pose initializations are intersected, then both optimized poses are scored on
that identical support with ORB's scale-normalized Huber reprojection cost.
`common_score_prefers_dynamic` in the evaluator summary measures whether this
matched evidence predicts lower ground-truth error. It is diagnostic only and
does not alter the active gate or camera trajectory.

An independent direct RGB-D diagnostic uses the previous valid frame's static
pixels, depth, and committed pose. Both counterfactual poses warp the same
sampled pixels into the current frame; only pixels valid and static under both
hypotheses enter the Huber depth and photometric scores. The evaluator reports
`direct_depth_prefers_dynamic`, `direct_photometric_prefers_dynamic`, and
`direct_combined_prefers_dynamic`. These fields remain diagnostic for all
legacy presets. In `semantic_motion_direct*`, the frozen consensus gate is
intersected with `direct_photometric_prefers_dynamic` inside tracking before
the pose branch is selected. The log fields `prior_consensus_pass`,
`direct_validation_valid`, and `direct_validation_pass` make that conjunction
auditable.

Development replay for the direct-validated shadow control is:

```bash
python3 scripts/run_reproducible_benchmark.py \
  --configs semantic_motion_direct_shadow \
  --sequences tum_walking_xyz bonn_crowd \
  --seeds 0 1 2 \
  --gpus 0 1 \
  --jobs 2 \
  --heldout-stride 0 \
  --output-root /path/to/new/direct_development_root
```

These are consumed development sequences. Do not treat this replay as
held-out evidence or launch mapping from it.

The frozen fresh mechanism test uses:

```bash
python3 scripts/run_reproducible_benchmark.py \
  --configs semantic_motion_direct_shadow \
  --sequences tum_sitting_xyz bonn_person_tracking \
  --seeds 0 1 2 \
  --gpus 0 1 \
  --jobs 2 \
  --heldout-stride 0 \
  --output-root /path/to/new/direct_heldout_root
```

The automatic decision is `motion_direct_counterfactual_gate.json`. Only if it
passes should the same sequences and seeds be run with `semantic`,
`semantic_motion_direct_shadow`, and `semantic_motion_direct`; that decision is
`motion_direct_end_to_end_gate.json`. The exact seed set is `0 1 2`; changing
or extending it makes this frozen gate insufficient. The end-to-end decision
also requires the matched Shadow mechanism gate to pass in the same output
root. The evidence boundary and stop conditions are frozen in
`../MOTION_DIRECT_GATE_PROTOCOL_20260728.md`.

The separately named equal-weight RGB-D candidate uses a new mechanism-specific
fresh test:

```bash
python3 scripts/run_reproducible_benchmark.py \
  --configs semantic_motion_rgbd_shadow \
  --sequences tum_sitting_rpy bonn_person_tracking2 \
  --seeds 0 1 2 \
  --gpus 0 1 \
  --jobs 2 \
  --heldout-stride 0 \
  --output-root /path/to/new/rgbd_heldout_root
```

The automatic decision is `motion_rgbd_counterfactual_gate.json`. Its exact
data identities and staged stop conditions are frozen in
`../MOTION_RGBD_GATE_PROTOCOL_20260728.md`.

The frozen consensus gate is evaluated first as a shadow-only mechanism test:

```bash
python3 scripts/run_reproducible_benchmark.py \
  --configs semantic_motion_consensus_shadow \
  --sequences tum_walking_halfsphere bonn_crowd2 \
  --seeds 0 1 2 \
  --gpus 0 1 \
  --jobs 2 \
  --heldout-stride 0 \
  --output-root /path/to/new/heldout_counterfactual_root
```

This writes `motion_consensus_counterfactual_gate.json`. Every held-out
sequence needs all three seeds, at least five selected same-state pairs, at
least 60% dynamic translation wins, pooled dynamic translation RMSE no worse
than static, and no selected pair worse by more than 1 cm. Do not run the
paired end-to-end intervention benchmark unless this gate passes.

The conditional paired benchmark uses `semantic`,
`semantic_motion_consensus_shadow`, and `semantic_motion_consensus` on the same
held-out sequences and seeds. Its predeclared decision is written to
`motion_consensus_end_to_end_gate.json`; exact conditions are frozen in
`../MOTION_CONSENSUS_GATE_PROTOCOL_20260728.md`.

The reliability-tempered rescue is a separate protocol. Its prior information
scale is frozen to `0.5`, while the consensus selection thresholds stay
unchanged. Run its fresh held-out mechanism test with:

```bash
python3 scripts/run_reproducible_benchmark.py \
  --configs semantic_motion_tempered_shadow \
  --sequences tum_walking_static bonn_crowd3 \
  --seeds 0 1 2 \
  --gpus 0 1 \
  --jobs 2 \
  --heldout-stride 0 \
  --output-root /path/to/new/tempered_heldout_root
```

The automatic decision is
`motion_tempered_counterfactual_gate.json`. Only if it passes should the same
sequences and seeds be run with `semantic`,
`semantic_motion_tempered_shadow`, and `semantic_motion_tempered`; that
decision is `motion_tempered_end_to_end_gate.json`. The frozen evidence
boundary is documented in `../MOTION_TEMPERED_GATE_PROTOCOL_20260728.md`.

The Full Fusion gate requires better mean ATE on every sequence, at least 70%
paired-seed wins, and no more than a 0.5 percentage-point failure-rate
increase. Semantic+Motion additionally must be selected in at least 70% of
planned seeds on every sequence.

## Held-Out Mapping Evaluation

Use `prepare_mapping_heldout.py` to freeze common non-keyframe views, final-map
pose alignment, and static-mask provenance. Use `view_result` for batch
rendering and `evaluate_heldout_rendering.py` for strict manifest-based
PSNR/SSIM/LPIPS. Missing, extra, unreadable, or shape-mismatched frames fail a
method group.

SplaTAM rendering requires an explicit `frame,checkpoint_index` mapping.
Per-frame-mapped SplaTAM checkpoints are seen-view qualitative references, not
held-out quantitative baselines. The complete commands and evidence contract
are documented in `../MAPPING_EVALUATION.md`.
