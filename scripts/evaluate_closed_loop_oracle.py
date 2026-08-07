#!/usr/bin/env python3
"""Evaluate real full-sequence Oracle/sham branches at sequence-seed level."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from evaluate_checkpointed_forks import (
    interpolate_pose,
    pose_matrix,
    read_ground_truth,
    relative_error,
)


ORACLE_CONFIG = "semantic_motion_rgbd_oracle"
SHAM_CONFIG = "semantic_motion_rgbd_oracle_sham"
REQUIRED_SEQUENCES = ("tum_walking_xyz", "bonn_crowd")
REQUIRED_SEEDS = (0, 1, 2)
FIRST_ACTION_IGNORED_FIELDS = {"prior_used", "oracle_applied"}
MATCHED_HASH_FIELDS = (
    "binary_sha256",
    "runtime_library_sha256",
    "orb_config_sha256",
    "gaussian_config_sha256",
    "mask_config_sha256",
    "association_sha256",
    "ground_truth_sha256",
    "vocabulary_sha256",
    "yolo_engine_sha256",
    "matched_oracle_pose_sha256",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def read_trajectory(path: Path) -> list[tuple[float, np.ndarray]]:
    trajectory: list[tuple[float, np.ndarray]] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 8:
                raise ValueError(
                    f"{path}:{line_number}: expected 8 TUM fields")
            values = [float(value) for value in fields]
            trajectory.append((values[0], pose_matrix(values[1:])))
    if not trajectory:
        raise ValueError(f"No trajectory poses in {path}")
    return trajectory


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def one_step_rpe(
    trajectory_path: Path,
    ground_truth_path: Path,
) -> dict[str, float | int | None]:
    trajectory = read_trajectory(trajectory_path)
    gt_timestamps, gt_poses = read_ground_truth(ground_truth_path)
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    previous: tuple[float, np.ndarray, np.ndarray] | None = None
    for timestamp, estimated_twc in trajectory:
        ground_truth_twc = interpolate_pose(
            timestamp, gt_timestamps, gt_poses, 0.1)
        if ground_truth_twc is None:
            previous = None
            continue
        if previous is not None:
            _, previous_estimated_twc, previous_ground_truth_twc = previous
            translation, rotation = relative_error(
                np.linalg.inv(previous_estimated_twc),
                np.linalg.inv(estimated_twc),
                previous_ground_truth_twc,
                ground_truth_twc)
            translation_errors.append(translation)
            rotation_errors.append(rotation)
        previous = (timestamp, estimated_twc, ground_truth_twc)
    return {
        "valid_pairs": len(translation_errors),
        "translation_rmse_m": (
            math.sqrt(statistics.mean(
                value * value for value in translation_errors))
            if translation_errors else None),
        "translation_p95_m": percentile(translation_errors, 95),
        "rotation_rmse_rad": (
            math.sqrt(statistics.mean(
                value * value for value in rotation_errors))
            if rotation_errors else None),
        "rotation_p95_rad": percentile(rotation_errors, 95),
    }


def failure_diagnostics(
    rows: list[dict[str, str]],
) -> dict[str, float | int]:
    failed_frames = 0
    longest_frames = 0
    longest_seconds = 0.0
    streak_start: float | None = None
    streak_frames = 0
    previous_timestamp = 0.0
    for row in rows:
        timestamp = float(row["timestamp"])
        failed = int(row["pose_valid"]) == 0 or int(row["lost"]) != 0
        if failed:
            failed_frames += 1
            if streak_start is None:
                streak_start = timestamp
                streak_frames = 0
            streak_frames += 1
            longest_frames = max(longest_frames, streak_frames)
            longest_seconds = max(
                longest_seconds, timestamp - streak_start)
        else:
            streak_start = None
            streak_frames = 0
        previous_timestamp = timestamp
    if streak_start is not None:
        longest_seconds = max(
            longest_seconds, previous_timestamp - streak_start)
    return {
        "failed_frames": failed_frames,
        "failure_rate": failed_frames / len(rows),
        "longest_failure_streak_frames": longest_frames,
        "longest_failure_streak_seconds": longest_seconds,
    }


def first_oracle_action(
    rows: list[dict[str, str]],
) -> int | None:
    for row in rows:
        if int(row["oracle_applied"]) != 0:
            return int(row["frame"])
    return None


def keyed(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {int(row["frame"]): row for row in rows}


def compare_prefix(
    oracle_execution: list[dict[str, str]],
    sham_execution: list[dict[str, str]],
    first_action: int,
) -> dict[str, Any]:
    oracle_by_frame = keyed(oracle_execution)
    sham_by_frame = keyed(sham_execution)
    mismatches: list[dict[str, str | int]] = []
    for frame in sorted(
            set(oracle_by_frame).intersection(sham_by_frame)):
        if frame >= first_action:
            break
        left = oracle_by_frame[frame]
        right = sham_by_frame[frame]
        for field in left:
            if field not in right or left[field] != right[field]:
                mismatches.append({
                    "frame": frame,
                    "field": field,
                    "oracle": left[field],
                    "sham": right.get(field, "<missing>"),
                })
                return {
                    "pass": False,
                    "compared_frames": frame,
                    "first_mismatch": mismatches[0],
                }
    expected = list(range(first_action))
    available = sorted(
        frame for frame in set(oracle_by_frame).intersection(sham_by_frame)
        if frame < first_action)
    missing = sorted(set(expected).difference(available))
    return {
        "pass": not missing,
        "compared_frames": len(available),
        "missing_frames": missing[:10],
    }


def compare_first_action_input(
    oracle_counterfactual: list[dict[str, str]],
    sham_counterfactual: list[dict[str, str]],
    first_action: int,
) -> dict[str, Any]:
    oracle = keyed(oracle_counterfactual).get(first_action)
    sham = keyed(sham_counterfactual).get(first_action)
    if oracle is None or sham is None:
        return {
            "pass": False,
            "reason": "first action is missing from counterfactual log",
        }
    for field, oracle_value in oracle.items():
        if field in FIRST_ACTION_IGNORED_FIELDS:
            continue
        sham_value = sham.get(field)
        if oracle_value != sham_value:
            return {
                "pass": False,
                "first_mismatch": {
                    "frame": first_action,
                    "field": field,
                    "oracle": oracle_value,
                    "sham": sham_value,
                },
            }
    return {
        "pass": (
            int(oracle["oracle_valid"]) == 1 and
            int(oracle["oracle_prefers_dynamic"]) == 1 and
            int(oracle["oracle_applied"]) == 1 and
            int(sham["oracle_applied"]) == 0),
        "oracle_valid": int(oracle["oracle_valid"]),
        "oracle_prefers_dynamic": int(oracle["oracle_prefers_dynamic"]),
        "oracle_applied": int(oracle["oracle_applied"]),
        "sham_applied": int(sham["oracle_applied"]),
    }


def map_diagnostics(
    oracle_rows: list[dict[str, str]],
    sham_rows: list[dict[str, str]],
    first_action: int,
) -> dict[str, Any]:
    oracle = keyed(oracle_rows)
    sham = keyed(sham_rows)
    common_after = sorted(
        frame for frame in set(oracle).intersection(sham)
        if frame >= first_action)
    divergent = [
        frame for frame in common_after
        if oracle[frame]["map_fingerprint"] !=
            sham[frame]["map_fingerprint"]
    ]
    oracle_final = oracle[max(oracle)]
    sham_final = sham[max(sham)]
    return {
        "compared_post_action_frames": len(common_after),
        "map_divergent_frames": len(divergent),
        "first_map_divergence_frame": divergent[0] if divergent else None,
        "oracle_final_keyframes": int(oracle_final["keyframe_count"]),
        "sham_final_keyframes": int(sham_final["keyframe_count"]),
        "oracle_final_map_points": int(oracle_final["map_point_count"]),
        "sham_final_map_points": int(sham_final["map_point_count"]),
        "oracle_final_map_fingerprint":
            oracle_final["map_fingerprint"],
        "sham_final_map_fingerprint": sham_final["map_fingerprint"],
    }


def verify_matched_manifests(
    oracle: dict[str, Any],
    sham: dict[str, Any],
) -> dict[str, Any]:
    mismatches = {}
    for field in MATCHED_HASH_FIELDS:
        left = oracle["hashes"].get(field)
        right = sham["hashes"].get(field)
        if left != right:
            mismatches[field] = {"oracle": left, "sham": right}
    return {"pass": not mismatches, "mismatches": mismatches}


def verify_replay_capture(
    oracle_summary: dict[str, Any],
    sham_summary: dict[str, Any],
) -> dict[str, Any]:
    oracle_enabled = bool(
        oracle_summary.get("replay_state_capture_enabled", False))
    sham_enabled = bool(
        sham_summary.get("replay_state_capture_enabled", False))
    return {
        "pass": oracle_enabled and sham_enabled,
        "oracle_enabled": oracle_enabled,
        "sham_enabled": sham_enabled,
    }


def evaluate_pair(
    root: Path,
    sequence: str,
    seed: int,
) -> dict[str, Any]:
    suffix = Path(sequence) / f"seed_{seed:04d}"
    oracle_dir = root / ORACLE_CONFIG / suffix
    sham_dir = root / SHAM_CONFIG / suffix
    required_files = (
        "result.json",
        "manifest.json",
        "run_summary.json",
        "frame_metrics.csv",
        "replay_execution_state.csv",
        "motion_prior_counterfactual.csv",
        "CameraTrajectory_AllFrames_TUM.txt",
    )
    for branch, directory in (("oracle", oracle_dir), ("sham", sham_dir)):
        missing_files = [
            name for name in required_files
            if not (directory / name).is_file()
        ]
        if missing_files:
            return {
                "sequence": sequence,
                "seed": seed,
                "status": "missing",
                "missing_branch": branch,
                "missing_run": str(directory),
                "missing_files": missing_files,
            }

    oracle_result = json.loads((oracle_dir / "result.json").read_text())
    sham_result = json.loads((sham_dir / "result.json").read_text())
    oracle_manifest = json.loads((oracle_dir / "manifest.json").read_text())
    sham_manifest = json.loads((sham_dir / "manifest.json").read_text())
    oracle_summary = json.loads(
        (oracle_dir / "run_summary.json").read_text())
    sham_summary = json.loads((sham_dir / "run_summary.json").read_text())
    oracle_frames = read_csv(oracle_dir / "frame_metrics.csv")
    sham_frames = read_csv(sham_dir / "frame_metrics.csv")
    oracle_execution = read_csv(
        oracle_dir / "replay_execution_state.csv")
    sham_execution = read_csv(sham_dir / "replay_execution_state.csv")
    oracle_counterfactual = read_csv(
        oracle_dir / "motion_prior_counterfactual.csv")
    sham_counterfactual = read_csv(
        sham_dir / "motion_prior_counterfactual.csv")
    first_action = first_oracle_action(oracle_frames)

    manifest_match = verify_matched_manifests(
        oracle_manifest, sham_manifest)
    replay_capture = verify_replay_capture(
        oracle_summary, sham_summary)
    prefix = (
        compare_prefix(oracle_execution, sham_execution, first_action)
        if first_action is not None
        else {"pass": False, "reason": "no Oracle action"})
    first_action_input = (
        compare_first_action_input(
            oracle_counterfactual, sham_counterfactual, first_action)
        if first_action is not None
        else {"pass": False, "reason": "no Oracle action"})
    maps = (
        map_diagnostics(
            oracle_execution, sham_execution, first_action)
        if first_action is not None
        else {})
    ground_truth = Path(oracle_manifest["paths"]["ground_truth"])
    oracle_rpe = one_step_rpe(
        oracle_dir / "CameraTrajectory_AllFrames_TUM.txt",
        ground_truth)
    sham_rpe = one_step_rpe(
        sham_dir / "CameraTrajectory_AllFrames_TUM.txt",
        ground_truth)
    oracle_failures = failure_diagnostics(oracle_frames)
    sham_failures = failure_diagnostics(sham_frames)
    oracle_metrics = oracle_result["metrics"]
    sham_metrics = sham_result["metrics"]
    oracle_actions = sum(
        int(row["oracle_applied"]) for row in oracle_frames)
    sham_actions = sum(
        int(row["oracle_applied"]) for row in sham_frames)

    return {
        "sequence": sequence,
        "seed": seed,
        "status": "complete",
        "first_oracle_action_frame": first_action,
        "oracle_actions": oracle_actions,
        "sham_actions": sham_actions,
        "oracle_preferences": sum(
            int(row["oracle_prefers_dynamic"])
            for row in oracle_frames),
        "matched_assets": manifest_match,
        "replay_capture_contract": replay_capture,
        "prefix_equivalence": prefix,
        "first_action_input_equivalence": first_action_input,
        "real_fork_pass": bool(
            first_action is not None and
            oracle_actions > 0 and sham_actions == 0 and
            manifest_match["pass"] and replay_capture["pass"] and
            prefix["pass"] and
            first_action_input["pass"]),
        "oracle": {
            "ate_rmse_m": oracle_metrics["ate_rmse_m"],
            "evo_rpe_translation_rmse_m":
                oracle_metrics["rpe_translation_rmse_m"],
            "evo_rpe_rotation_rmse_deg":
                oracle_metrics["rpe_rotation_rmse_deg"],
            "one_step_rpe": oracle_rpe,
            "failures": oracle_failures,
            "end_to_end_seconds":
                oracle_metrics["end_to_end_seconds"],
        },
        "sham": {
            "ate_rmse_m": sham_metrics["ate_rmse_m"],
            "evo_rpe_translation_rmse_m":
                sham_metrics["rpe_translation_rmse_m"],
            "evo_rpe_rotation_rmse_deg":
                sham_metrics["rpe_rotation_rmse_deg"],
            "one_step_rpe": sham_rpe,
            "failures": sham_failures,
            "end_to_end_seconds":
                sham_metrics["end_to_end_seconds"],
        },
        "map_diagnostics": maps,
        "delayed_mapping_evidence": {
            "status": "not_implemented",
            "accepted": None,
            "rejected": None,
        },
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def build_gate(
    rows: list[dict[str, Any]],
    planned_sequences: list[str],
    planned_seeds: list[int],
) -> dict[str, Any]:
    eligible = (
        tuple(sorted(planned_sequences)) ==
            tuple(sorted(REQUIRED_SEQUENCES)) and
        tuple(sorted(planned_seeds)) == REQUIRED_SEEDS)
    complete = all(row["status"] == "complete" for row in rows)
    real_fork = complete and all(
        row.get("real_fork_pass", False) for row in rows)
    sequence_checks = {}
    for sequence in planned_sequences:
        sequence_rows = [
            row for row in rows
            if row["status"] == "complete" and
            row["sequence"] == sequence
        ]
        if not sequence_rows:
            sequence_checks[sequence] = {
                "pass": False, "reason": "no complete pairs"}
            continue
        oracle_ate = [row["oracle"]["ate_rmse_m"]
                      for row in sequence_rows]
        sham_ate = [row["sham"]["ate_rmse_m"]
                    for row in sequence_rows]
        oracle_rpe = [
            row["oracle"]["one_step_rpe"]["translation_rmse_m"]
            for row in sequence_rows]
        sham_rpe = [
            row["sham"]["one_step_rpe"]["translation_rmse_m"]
            for row in sequence_rows]
        oracle_failure = [
            row["oracle"]["failures"]["failure_rate"]
            for row in sequence_rows]
        sham_failure = [
            row["sham"]["failures"]["failure_rate"]
            for row in sequence_rows]
        oracle_p95 = [
            row["oracle"]["one_step_rpe"]["translation_p95_m"]
            for row in sequence_rows]
        sham_p95 = [
            row["sham"]["one_step_rpe"]["translation_p95_m"]
            for row in sequence_rows]
        actions = sum(row["oracle_actions"] for row in sequence_rows)
        clauses = {
            "has_oracle_action": actions > 0,
            "mean_ate_strictly_better":
                mean(oracle_ate) < mean(sham_ate),
            "mean_one_step_translation_rpe_strictly_better":
                mean(oracle_rpe) < mean(sham_rpe),
            "mean_failure_rate_not_worse":
                mean(oracle_failure) <= mean(sham_failure),
            "mean_translation_p95_within_two_percent":
                mean(oracle_p95) <= 1.02 * mean(sham_p95),
        }
        sequence_checks[sequence] = {
            "pass": all(clauses.values()),
            "clauses": clauses,
            "oracle_actions": actions,
            "oracle_ate_mean_m": mean(oracle_ate),
            "sham_ate_mean_m": mean(sham_ate),
            "oracle_one_step_translation_rpe_mean_m":
                mean(oracle_rpe),
            "sham_one_step_translation_rpe_mean_m":
                mean(sham_rpe),
            "oracle_failure_rate_mean": mean(oracle_failure),
            "sham_failure_rate_mean": mean(sham_failure),
            "oracle_translation_p95_mean_m": mean(oracle_p95),
            "sham_translation_p95_mean_m": mean(sham_p95),
        }
    complete_rows = [row for row in rows if row["status"] == "complete"]
    paired_ate_wins = sum(
        row["oracle"]["ate_rmse_m"] < row["sham"]["ate_rmse_m"]
        for row in complete_rows)
    paired_win_rate = (
        paired_ate_wins / len(complete_rows) if complete_rows else 0.0)
    utility = bool(
        complete and
        all(check["pass"] for check in sequence_checks.values()) and
        paired_win_rate >= 2.0 / 3.0)
    passed = bool(eligible and complete and real_fork and utility)
    return {
        "protocol": "real-closed-loop-matched-sham-oracle-gate-v1",
        "claim_gate_eligible": eligible,
        "all_pairs_complete": complete,
        "real_closed_loop_fork_pass": real_fork,
        "sequence_level_utility_pass": utility,
        "paired_ate_wins": paired_ate_wins,
        "paired_ate_total": len(complete_rows),
        "paired_ate_win_rate": paired_win_rate,
        "sequence_checks": sequence_checks,
        "passed": passed,
        "decision": (
            "GO_TO_REVERSIBLE_TRACKING_AND_DELAYED_MAPPING"
            if passed else
            ("DIAGNOSTIC_ONLY_INCOMPLETE_PLAN" if not eligible
             else "NO_GO_TERMINATE_ROUTE")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    plan = json.loads(
        (args.root / "benchmark_plan.json").read_text())
    tasks = plan["tasks"]
    planned_sequences = sorted({task["sequence"] for task in tasks})
    planned_seeds = sorted({int(task["seed"]) for task in tasks})
    rows = [
        evaluate_pair(args.root, sequence, seed)
        for sequence in planned_sequences
        for seed in planned_seeds
    ]
    gate = build_gate(rows, planned_sequences, planned_seeds)
    report = {
        "protocol": "closed-loop-oracle-sequence-diagnostics-v1",
        "independent_unit": "sequence x seed",
        "rows": rows,
        "gate": gate,
    }
    output = args.root / "closed_loop_oracle_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    csv_path = args.root / "closed_loop_oracle_sequence_seed.csv"
    with csv_path.open("w", newline="") as stream:
        fieldnames = (
            "sequence", "seed", "status", "real_fork_pass",
            "first_oracle_action_frame", "oracle_actions",
            "oracle_ate_rmse_m", "sham_ate_rmse_m",
            "oracle_one_step_translation_rpe_rmse_m",
            "sham_one_step_translation_rpe_rmse_m",
            "oracle_translation_rpe_p95_m",
            "sham_translation_rpe_p95_m",
            "oracle_failure_rate", "sham_failure_rate",
            "oracle_end_to_end_seconds", "sham_end_to_end_seconds",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat: dict[str, object] = {
                "sequence": row["sequence"],
                "seed": row["seed"],
                "status": row["status"],
            }
            if row["status"] == "complete":
                flat.update({
                    "real_fork_pass": row["real_fork_pass"],
                    "first_oracle_action_frame":
                        row["first_oracle_action_frame"],
                    "oracle_actions": row["oracle_actions"],
                    "oracle_ate_rmse_m":
                        row["oracle"]["ate_rmse_m"],
                    "sham_ate_rmse_m": row["sham"]["ate_rmse_m"],
                    "oracle_one_step_translation_rpe_rmse_m":
                        row["oracle"]["one_step_rpe"]
                            ["translation_rmse_m"],
                    "sham_one_step_translation_rpe_rmse_m":
                        row["sham"]["one_step_rpe"]
                            ["translation_rmse_m"],
                    "oracle_translation_rpe_p95_m":
                        row["oracle"]["one_step_rpe"]
                            ["translation_p95_m"],
                    "sham_translation_rpe_p95_m":
                        row["sham"]["one_step_rpe"]
                            ["translation_p95_m"],
                    "oracle_failure_rate":
                        row["oracle"]["failures"]["failure_rate"],
                    "sham_failure_rate":
                        row["sham"]["failures"]["failure_rate"],
                    "oracle_end_to_end_seconds":
                        row["oracle"]["end_to_end_seconds"],
                    "sham_end_to_end_seconds":
                        row["sham"]["end_to_end_seconds"],
                })
            writer.writerow(flat)
    print(json.dumps(gate, indent=2, sort_keys=True))
    if not gate["claim_gate_eligible"]:
        return 0
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
