#!/usr/bin/env python3
"""Evaluate frozen 1/5/10-frame pose forks against timestamped TUM GT."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


POSE_SUFFIXES = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
EVALUATED_FIELDS = (
    "checkpoint",
    "selection",
    "gate_policy",
    "prior_subspace",
    "intervention_projection",
    "information_scale_multiplier",
    "max_static_information_leverage",
    "static_information_leverage_mode",
    "replay_mode",
    "lag_frames",
    "translation_blend",
    "posterior_min_score_improvement",
    "horizon",
    "start_timestamp",
    "end_timestamp",
    "static_success",
    "dynamic_success",
    "gt_valid",
    "static_translation_rpe_m",
    "static_rotation_rpe_rad",
    "dynamic_translation_rpe_m",
    "dynamic_rotation_rpe_rad",
    "velocity_neutral_available",
    "velocity_neutral_success",
    "velocity_neutral_translation_rpe_m",
    "velocity_neutral_rotation_rpe_rad",
    "shadow_available",
    "shadow_success",
    "shadow_translation_rpe_m",
    "shadow_rotation_rpe_rad",
    "lag30_available",
    "lag30_success",
    "lag30_translation_rpe_m",
    "lag30_rotation_rpe_rad",
    "schur_candidate",
    "schur_posterior_valid",
    "schur_posterior_pass",
    "schur_posterior_static_inliers",
    "schur_posterior_dynamic_inliers",
    "schur_posterior_common_support",
    "schur_posterior_static_score",
    "schur_posterior_dynamic_score",
    "schur_posterior_score_improvement",
    "schur_posterior_translation_innovation",
    "schur_static_translation_information_trace",
    "schur_dynamic_translation_information_trace_raw",
    "schur_dynamic_translation_information_trace_bounded",
    "schur_max_generalized_leverage_raw",
    "schur_max_generalized_leverage_bounded",
    "schur_leverage_normalization_scale",
    "lag30_candidate",
    "lag30_posterior_valid",
    "lag30_posterior_pass",
    "lag30_posterior_static_inliers",
    "lag30_posterior_dynamic_inliers",
    "lag30_posterior_common_support",
    "lag30_posterior_static_score",
    "lag30_posterior_dynamic_score",
    "lag30_posterior_score_improvement",
    "lag30_posterior_translation_innovation",
    "lag30_static_translation_information_trace",
    "lag30_dynamic_translation_information_trace_raw",
    "lag30_dynamic_translation_information_trace_bounded",
    "lag30_max_generalized_leverage_raw",
    "lag30_max_generalized_leverage_bounded",
    "lag30_leverage_normalization_scale",
)


def quaternion_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    quaternion = np.asarray((qw, qx, qy, qz), dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid quaternion")
    quaternion /= norm
    w, x, y, z = quaternion
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def pose_matrix(values: list[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_matrix(*values[3:])
    pose[:3, 3] = values[:3]
    return pose


def read_ground_truth(path: Path) -> tuple[list[float], list[np.ndarray]]:
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 8:
                raise ValueError(
                    f"{path}:{line_number}: expected 8 TUM fields"
                )
            values = [float(value) for value in fields]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"{path}:{line_number}: non-finite ground truth"
                )
            if timestamps and values[0] <= timestamps[-1]:
                raise ValueError(
                    f"{path}:{line_number}: ground-truth timestamps must be "
                    "strictly increasing"
                )
            timestamps.append(values[0])
            pose = np.asarray(values[1:], dtype=np.float64)
            quaternion_norm = float(np.linalg.norm(pose[3:]))
            if quaternion_norm < 1e-12:
                raise ValueError(
                    f"{path}:{line_number}: zero-norm ground-truth quaternion"
                )
            pose[3:] /= quaternion_norm
            poses.append(pose)
    if not timestamps:
        raise ValueError(f"No ground-truth poses in {path}")
    return timestamps, poses


def interpolate_pose(
    timestamp: float,
    timestamps: list[float],
    poses: list[np.ndarray],
    max_bracket_span: float,
) -> np.ndarray | None:
    position = bisect.bisect_left(timestamps, timestamp)
    if (
        position < len(timestamps)
        and abs(timestamps[position] - timestamp) <= 1e-9
    ):
        return pose_matrix(poses[position].tolist())
    if position == 0 or position == len(timestamps):
        return None
    lower = position - 1
    upper = position
    span = timestamps[upper] - timestamps[lower]
    if span > max_bracket_span:
        return None
    alpha = (timestamp - timestamps[lower]) / span
    translation = (
        (1.0 - alpha) * poses[lower][:3] + alpha * poses[upper][:3]
    )
    lower_quaternion = poses[lower][3:].copy()
    upper_quaternion = poses[upper][3:].copy()
    dot = float(np.dot(lower_quaternion, upper_quaternion))
    if dot < 0.0:
        upper_quaternion *= -1.0
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        quaternion = (
            (1.0 - alpha) * lower_quaternion + alpha * upper_quaternion
        )
        quaternion /= np.linalg.norm(quaternion)
    else:
        angle = math.acos(dot)
        denominator = math.sin(angle)
        quaternion = (
            math.sin((1.0 - alpha) * angle) / denominator * lower_quaternion
            + math.sin(alpha * angle) / denominator * upper_quaternion
        )
    return pose_matrix(np.concatenate((translation, quaternion)).tolist())


def row_pose(row: dict[str, str], prefix: str) -> np.ndarray:
    return pose_matrix(
        [float(row[f"{prefix}_{suffix}"]) for suffix in POSE_SUFFIXES]
    )


def relative_error(
    start_tcw: np.ndarray,
    end_tcw: np.ndarray,
    start_twc_gt: np.ndarray,
    end_twc_gt: np.ndarray,
) -> tuple[float, float]:
    estimated = end_tcw @ np.linalg.inv(start_tcw)
    ground_truth = np.linalg.inv(end_twc_gt) @ start_twc_gt
    error = np.linalg.inv(ground_truth) @ estimated
    translation = float(np.linalg.norm(error[:3, 3]))
    cosine = float(np.clip((np.trace(error[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    rotation = math.acos(cosine)
    return translation, rotation


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def evaluate(
    fork_csv: Path,
    ground_truth: Path,
    max_ground_truth_gap: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gt_timestamps, gt_poses = read_ground_truth(ground_truth)
    with fork_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return (
            {
                "protocol": "checkpointed-fork-evaluation-v2",
                "status": "insufficient_data",
                "reason": "no checkpoints matched the requested selection",
                "fork_csv": str(fork_csv.resolve()),
                "ground_truth": str(ground_truth.resolve()),
                "max_ground_truth_gap_seconds": max_ground_truth_gap,
                "input_rows": 0,
                "horizons": {},
            },
            [],
        )
    scale_multipliers = {
        float(row.get("information_scale_multiplier", "1.0"))
        for row in rows
    }
    if (
        len(scale_multipliers) != 1
        or not all(
            math.isfinite(value) and value >= 0.0
            for value in scale_multipliers
        )
    ):
        raise ValueError(
            "Fork CSV must contain one finite non-negative "
            "information_scale_multiplier"
        )
    information_scale_multiplier = next(iter(scale_multipliers))
    gate_policies = {
        row.get("gate_policy", "legacy-conjunction") for row in rows
    }
    if (
        len(gate_policies) != 1
        or not gate_policies.issubset(
            {"legacy-conjunction", "direct-combined"}
        )
    ):
        raise ValueError(
            "Fork CSV must contain one supported gate_policy"
        )
    gate_policy = next(iter(gate_policies))
    leverage_limits = {
        float(row.get("max_static_information_leverage", "0.0"))
        for row in rows
    }
    if (
        len(leverage_limits) != 1
        or not all(
            math.isfinite(value) and value >= 0.0
            for value in leverage_limits
        )
    ):
        raise ValueError(
            "Fork CSV must contain one finite non-negative "
            "max_static_information_leverage"
        )
    max_static_information_leverage = next(iter(leverage_limits))
    leverage_modes = {
        row.get("static_information_leverage_mode", "cap-only")
        for row in rows
    }
    if (
        len(leverage_modes) != 1
        or not leverage_modes.issubset(
            {"cap-only", "normalize-to-target"}
        )
    ):
        raise ValueError(
            "Fork CSV must contain one supported "
            "static_information_leverage_mode"
        )
    static_information_leverage_mode = next(iter(leverage_modes))
    if (
        static_information_leverage_mode == "normalize-to-target"
        and max_static_information_leverage <= 0.0
    ):
        raise ValueError(
            "normalize-to-target requires a positive "
            "max_static_information_leverage"
        )
    prior_subspaces = {
        row.get("prior_subspace", "full") for row in rows
    }
    if (
        len(prior_subspaces) != 1
        or not prior_subspaces.issubset({"full", "translation"})
    ):
        raise ValueError(
            "Fork CSV must contain one supported prior_subspace"
        )
    prior_subspace = next(iter(prior_subspaces))
    intervention_projections = {
        row.get("intervention_projection", "full") for row in rows
    }
    if (
        len(intervention_projections) != 1
        or not intervention_projections.issubset(
            {
                "full",
                "translation",
                "common-score",
                "shadow-anchor",
                "shadow-translation",
            }
        )
    ):
        raise ValueError(
            "Fork CSV must contain one supported intervention_projection"
        )
    intervention_projection = next(iter(intervention_projections))
    replay_modes = {
        row.get("replay_mode", "legacy-motion-model-v1")
        for row in rows
    }
    if len(replay_modes) != 1:
        raise ValueError("Fork CSV must contain one replay_mode")
    replay_mode = next(iter(replay_modes))
    lag_frames_values = {
        int(row.get("lag_frames", "30")) for row in rows
    }
    if len(lag_frames_values) != 1 or min(lag_frames_values) <= 0:
        raise ValueError("Fork CSV must contain one positive lag_frames")
    lag_frames = next(iter(lag_frames_values))
    translation_blends = {
        float(row.get("translation_blend", "1.0"))
        for row in rows
    }
    if (
        len(translation_blends) != 1
        or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in translation_blends
        )
    ):
        raise ValueError(
            "Fork CSV must contain one finite translation_blend in [0,1]"
        )
    translation_blend = next(iter(translation_blends))
    posterior_score_thresholds = {
        float(row.get("posterior_min_score_improvement", "0.0"))
        for row in rows
    }
    if (
        len(posterior_score_thresholds) != 1
        or not all(
            math.isfinite(value) and value >= 0.0
            for value in posterior_score_thresholds
        )
    ):
        raise ValueError(
            "Fork CSV must contain one finite non-negative "
            "posterior_min_score_improvement"
        )
    posterior_score_threshold = next(iter(posterior_score_thresholds))

    evaluated: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        horizon = int(row["horizon"])
        start_timestamp = float(row["start_timestamp"])
        end_timestamp = float(row["end_timestamp"])
        start_gt = interpolate_pose(
            start_timestamp, gt_timestamps, gt_poses, max_ground_truth_gap
        )
        end_gt = interpolate_pose(
            end_timestamp, gt_timestamps, gt_poses, max_ground_truth_gap
        )
        item: dict[str, Any] = {
            "checkpoint": row["checkpoint"],
            "selection": row["selection"],
            "gate_policy": gate_policy,
            "prior_subspace": prior_subspace,
            "intervention_projection": intervention_projection,
            "information_scale_multiplier": information_scale_multiplier,
            "max_static_information_leverage":
                max_static_information_leverage,
            "static_information_leverage_mode":
                static_information_leverage_mode,
            "replay_mode": replay_mode,
            "lag_frames": lag_frames,
            "translation_blend": translation_blend,
            "posterior_min_score_improvement":
                posterior_score_threshold,
            "horizon": horizon,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "static_success": int(row["static_success"]),
            "dynamic_success": int(row["dynamic_success"]),
            "gt_valid": int(start_gt is not None and end_gt is not None),
            "static_translation_rpe_m": None,
            "static_rotation_rpe_rad": None,
            "dynamic_translation_rpe_m": None,
            "dynamic_rotation_rpe_rad": None,
            "velocity_neutral_available": int(
                "velocity_neutral_success" in row
                and "velocity_neutral_tcw_tx" in row
            ),
            "velocity_neutral_success": int(
                row.get("velocity_neutral_success", "0")
            ),
            "velocity_neutral_translation_rpe_m": None,
            "velocity_neutral_rotation_rpe_rad": None,
            "shadow_available": int(
                "shadow_success" in row and "shadow_tcw_tx" in row
            ),
            "shadow_success": int(row.get("shadow_success", "0")),
            "shadow_translation_rpe_m": None,
            "shadow_rotation_rpe_rad": None,
            "lag30_available": int(
                "lag30_success" in row and "lag30_tcw_tx" in row
            ),
            "lag30_success": int(row.get("lag30_success", "0")),
            "lag30_translation_rpe_m": None,
            "lag30_rotation_rpe_rad": None,
            "schur_candidate": int(row.get("schur_candidate", "0")),
            "schur_posterior_valid": int(
                row.get("schur_posterior_valid", "0")
            ),
            "schur_posterior_pass": int(
                row.get("schur_posterior_pass", "0")
            ),
            "lag30_candidate": int(row.get("lag30_candidate", "0")),
            "lag30_posterior_valid": int(
                row.get("lag30_posterior_valid", "0")
            ),
            "lag30_posterior_pass": int(
                row.get("lag30_posterior_pass", "0")
            ),
        }
        for prefix in ("schur", "lag30"):
            for name in (
                "posterior_static_inliers",
                "posterior_dynamic_inliers",
                "posterior_common_support",
            ):
                item[f"{prefix}_{name}"] = int(
                    row.get(f"{prefix}_{name}", "-1")
                )
            for name in (
                "posterior_static_score",
                "posterior_dynamic_score",
                "posterior_score_improvement",
                "posterior_translation_innovation",
                "static_translation_information_trace",
                "dynamic_translation_information_trace_raw",
                "dynamic_translation_information_trace_bounded",
                "max_generalized_leverage_raw",
                "max_generalized_leverage_bounded",
                "leverage_normalization_scale",
            ):
                raw_value = row.get(f"{prefix}_{name}", "")
                item[f"{prefix}_{name}"] = (
                    float(raw_value) if raw_value else None
                )
        if start_gt is not None and end_gt is not None:
            start_tcw = row_pose(row, "start_tcw")
            if item["static_success"]:
                (
                    item["static_translation_rpe_m"],
                    item["static_rotation_rpe_rad"],
                ) = relative_error(
                    start_tcw, row_pose(row, "static_tcw"),
                    start_gt, end_gt
                )
            if item["dynamic_success"]:
                (
                    item["dynamic_translation_rpe_m"],
                    item["dynamic_rotation_rpe_rad"],
                ) = relative_error(
                    start_tcw, row_pose(row, "dynamic_tcw"),
                    start_gt, end_gt
                )
            if (
                item["velocity_neutral_available"]
                and item["velocity_neutral_success"]
            ):
                (
                    item["velocity_neutral_translation_rpe_m"],
                    item["velocity_neutral_rotation_rpe_rad"],
                ) = relative_error(
                    start_tcw,
                    row_pose(row, "velocity_neutral_tcw"),
                    start_gt,
                    end_gt,
                )
            if item["shadow_available"] and item["shadow_success"]:
                (
                    item["shadow_translation_rpe_m"],
                    item["shadow_rotation_rpe_rad"],
                ) = relative_error(
                    start_tcw,
                    row_pose(row, "shadow_tcw"),
                    start_gt,
                    end_gt,
                )
            if item["lag30_available"] and item["lag30_success"]:
                (
                    item["lag30_translation_rpe_m"],
                    item["lag30_rotation_rpe_rad"],
                ) = relative_error(
                    start_tcw,
                    row_pose(row, "lag30_tcw"),
                    start_gt,
                    end_gt,
                )
        evaluated.append(item)
        grouped.setdefault(horizon, []).append(item)

    summary: dict[str, Any] = {
        "protocol": "checkpointed-fork-evaluation-v2",
        "status": "complete",
        "fork_csv": str(fork_csv.resolve()),
        "ground_truth": str(ground_truth.resolve()),
        "max_ground_truth_gap_seconds": max_ground_truth_gap,
        "gate_policy": gate_policy,
        "prior_subspace": prior_subspace,
        "intervention_projection": intervention_projection,
        "information_scale_multiplier": information_scale_multiplier,
        "max_static_information_leverage":
            max_static_information_leverage,
        "static_information_leverage_mode":
            static_information_leverage_mode,
        "replay_mode": replay_mode,
        "lag_frames": lag_frames,
        "translation_blend": translation_blend,
        "posterior_min_score_improvement": posterior_score_threshold,
        "input_rows": len(rows),
        "horizons": {},
    }
    total_gt_pairs = 0
    for horizon, horizon_rows in sorted(grouped.items()):
        gt_rows = [row for row in horizon_rows if row["gt_valid"]]
        total_gt_pairs += len(gt_rows)
        static_translation = [
            row["static_translation_rpe_m"]
            for row in gt_rows
            if row["static_translation_rpe_m"] is not None
        ]
        dynamic_translation = [
            row["dynamic_translation_rpe_m"]
            for row in gt_rows
            if row["dynamic_translation_rpe_m"] is not None
        ]
        static_rotation = [
            row["static_rotation_rpe_rad"]
            for row in gt_rows
            if row["static_rotation_rpe_rad"] is not None
        ]
        dynamic_rotation = [
            row["dynamic_rotation_rpe_rad"]
            for row in gt_rows
            if row["dynamic_rotation_rpe_rad"] is not None
        ]
        velocity_neutral_translation = [
            row["velocity_neutral_translation_rpe_m"]
            for row in gt_rows
            if row["velocity_neutral_translation_rpe_m"] is not None
        ]
        velocity_neutral_rotation = [
            row["velocity_neutral_rotation_rpe_rad"]
            for row in gt_rows
            if row["velocity_neutral_rotation_rpe_rad"] is not None
        ]
        shadow_translation = [
            row["shadow_translation_rpe_m"]
            for row in gt_rows
            if row["shadow_translation_rpe_m"] is not None
        ]
        shadow_rotation = [
            row["shadow_rotation_rpe_rad"]
            for row in gt_rows
            if row["shadow_rotation_rpe_rad"] is not None
        ]
        lag30_translation = [
            row["lag30_translation_rpe_m"]
            for row in gt_rows
            if row["lag30_translation_rpe_m"] is not None
        ]
        lag30_rotation = [
            row["lag30_rotation_rpe_rad"]
            for row in gt_rows
            if row["lag30_rotation_rpe_rad"] is not None
        ]
        paired = [
            row for row in gt_rows
            if row["static_translation_rpe_m"] is not None
            and row["dynamic_translation_rpe_m"] is not None
        ]
        translation_wins = sum(
            row["dynamic_translation_rpe_m"]
            < row["static_translation_rpe_m"]
            for row in paired
        )
        rotation_wins = sum(
            row["dynamic_rotation_rpe_rad"]
            < row["static_rotation_rpe_rad"]
            for row in paired
        )
        neutral_paired = [
            row for row in gt_rows
            if row["static_translation_rpe_m"] is not None
            and row["velocity_neutral_translation_rpe_m"] is not None
        ]
        neutral_translation_wins = sum(
            row["velocity_neutral_translation_rpe_m"]
            < row["static_translation_rpe_m"]
            for row in neutral_paired
        )
        neutral_rotation_wins = sum(
            row["velocity_neutral_rotation_rpe_rad"]
            < row["static_rotation_rpe_rad"]
            for row in neutral_paired
        )
        lag30_paired = [
            row for row in gt_rows
            if row["dynamic_translation_rpe_m"] is not None
            and row["lag30_translation_rpe_m"] is not None
        ]
        dynamic_vs_lag30_translation_wins = sum(
            row["dynamic_translation_rpe_m"]
            < row["lag30_translation_rpe_m"]
            for row in lag30_paired
        )
        dynamic_vs_lag30_rotation_wins = sum(
            row["dynamic_rotation_rpe_rad"]
            < row["lag30_rotation_rpe_rad"]
            for row in lag30_paired
        )
        summary["horizons"][str(horizon)] = {
            "rows": len(horizon_rows),
            "gt_valid_pairs": len(gt_rows),
            "paired_successes": len(paired),
            "static_failures": sum(
                not row["static_success"] for row in horizon_rows
            ),
            "dynamic_failures": sum(
                not row["dynamic_success"] for row in horizon_rows
            ),
            "static_failure_rate": sum(
                not row["static_success"] for row in horizon_rows
            ) / len(horizon_rows),
            "dynamic_failure_rate": sum(
                not row["dynamic_success"] for row in horizon_rows
            ) / len(horizon_rows),
            "static_translation_rpe_m": distribution(static_translation),
            "dynamic_translation_rpe_m": distribution(dynamic_translation),
            "static_rotation_rpe_rad": distribution(static_rotation),
            "dynamic_rotation_rpe_rad": distribution(dynamic_rotation),
            "velocity_neutral_available": any(
                row["velocity_neutral_available"] for row in horizon_rows
            ),
            "velocity_neutral_failures": sum(
                row["velocity_neutral_available"]
                and not row["velocity_neutral_success"]
                for row in horizon_rows
            ),
            "velocity_neutral_translation_rpe_m": distribution(
                velocity_neutral_translation
            ),
            "velocity_neutral_rotation_rpe_rad": distribution(
                velocity_neutral_rotation
            ),
            "shadow_failures": sum(
                row["shadow_available"] and not row["shadow_success"]
                for row in horizon_rows
            ),
            "shadow_translation_rpe_m": distribution(
                shadow_translation
            ),
            "shadow_rotation_rpe_rad": distribution(shadow_rotation),
            "lag30_failures": sum(
                row["lag30_available"] and not row["lag30_success"]
                for row in horizon_rows
            ),
            "lag30_translation_rpe_m": distribution(
                lag30_translation
            ),
            "lag30_rotation_rpe_rad": distribution(lag30_rotation),
            "schur_candidate_count": sum(
                row["schur_candidate"] for row in horizon_rows
            ),
            "schur_posterior_valid_count": sum(
                row["schur_posterior_valid"] for row in horizon_rows
            ),
            "schur_posterior_pass_count": sum(
                row["schur_posterior_pass"] for row in horizon_rows
            ),
            "lag30_candidate_count": sum(
                row["lag30_candidate"] for row in horizon_rows
            ),
            "lag30_posterior_valid_count": sum(
                row["lag30_posterior_valid"] for row in horizon_rows
            ),
            "lag30_posterior_pass_count": sum(
                row["lag30_posterior_pass"] for row in horizon_rows
            ),
            "dynamic_translation_win_count": translation_wins,
            "dynamic_translation_win_rate": (
                translation_wins / len(paired) if paired else None
            ),
            "dynamic_rotation_win_count": rotation_wins,
            "dynamic_rotation_win_rate": (
                rotation_wins / len(paired) if paired else None
            ),
            "velocity_neutral_paired_successes": len(neutral_paired),
            "velocity_neutral_translation_win_count":
                neutral_translation_wins,
            "velocity_neutral_translation_win_rate": (
                neutral_translation_wins / len(neutral_paired)
                if neutral_paired else None
            ),
            "velocity_neutral_rotation_win_count":
                neutral_rotation_wins,
            "velocity_neutral_rotation_win_rate": (
                neutral_rotation_wins / len(neutral_paired)
                if neutral_paired else None
            ),
            "dynamic_vs_lag30_paired_successes": len(lag30_paired),
            "dynamic_vs_lag30_translation_win_count":
                dynamic_vs_lag30_translation_wins,
            "dynamic_vs_lag30_translation_win_rate": (
                dynamic_vs_lag30_translation_wins / len(lag30_paired)
                if lag30_paired else None
            ),
            "dynamic_vs_lag30_rotation_win_count":
                dynamic_vs_lag30_rotation_wins,
            "dynamic_vs_lag30_rotation_win_rate": (
                dynamic_vs_lag30_rotation_wins / len(lag30_paired)
                if lag30_paired else None
            ),
        }
    if total_gt_pairs == 0:
        raise ValueError("No fork endpoints matched ground truth")
    return summary, evaluated


def write_evaluated_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVALUATED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fork_csv", type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--max-ground-truth-gap", type=float, default=0.1)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    if (
        not math.isfinite(args.max_ground_truth_gap)
        or args.max_ground_truth_gap <= 0.0
    ):
        parser.error("--max-ground-truth-gap must be finite and positive")
    try:
        summary, rows = evaluate(
            args.fork_csv, args.ground_truth, args.max_ground_truth_gap
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        write_evaluated_rows(args.output_csv, rows)
    except (OSError, ValueError, KeyError) as error:
        print(f"checkpointed fork evaluation failed: {error}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
