#!/usr/bin/env python3
"""Compare static and motion-prior pose hypotheses on the same frame state."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def average_duplicate_poses(
    poses: list[tuple[float, ...]],
) -> tuple[float, ...]:
    translation = tuple(
        statistics.fmean(pose[index] for pose in poses)
        for index in range(3))
    reference = normalize_quaternion(poses[0][3:])
    quaternions = []
    for pose in poses:
        quaternion = normalize_quaternion(pose[3:])
        if sum(left * right for left, right in zip(reference, quaternion)) < 0:
            quaternion = tuple(-value for value in quaternion)
        quaternions.append(quaternion)
    quaternion = normalize_quaternion(tuple(
        statistics.fmean(values)
        for values in zip(*quaternions)))
    return translation + quaternion


def load_tum_poses(
    path: Path,
) -> tuple[list[float], list[tuple[float, ...]], dict[str, int]]:
    timestamp_groups: list[tuple[float, list[tuple[float, ...]]]] = []
    previous_timestamp = None
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 8:
                raise ValueError(
                    f"{path}:{line_number} must contain timestamp plus 7 pose values")
            values = tuple(float(value) for value in fields)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}:{line_number} contains a non-finite value")
            timestamp = values[0]
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(
                    f"Timestamps must be nondecreasing in {path}; "
                    f"{timestamp} follows {previous_timestamp}")
            if timestamp_groups and timestamp == timestamp_groups[-1][0]:
                timestamp_groups[-1][1].append(values[1:])
            else:
                timestamp_groups.append((timestamp, [values[1:]]))
            previous_timestamp = timestamp
    if not timestamp_groups:
        raise ValueError(f"No TUM poses found in {path}")
    timestamps = [timestamp for timestamp, _ in timestamp_groups]
    poses = [
        group[0] if len(group) == 1 else average_duplicate_poses(group)
        for _, group in timestamp_groups
    ]
    duplicate_groups = sum(len(group) > 1 for _, group in timestamp_groups)
    duplicate_rows_merged = sum(
        len(group) - 1 for _, group in timestamp_groups)
    return timestamps, poses, {
        "duplicate_timestamp_groups": duplicate_groups,
        "duplicate_rows_merged": duplicate_rows_merged,
    }


def normalize_quaternion(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-12:
        raise ValueError("Pose contains a zero-norm quaternion")
    return tuple(value / norm for value in values)


def interpolate_pose(
    timestamps: list[float],
    poses: list[tuple[float, ...]],
    timestamp: float,
    max_bracket_span: float,
) -> tuple[tuple[float, ...], float]:
    position = bisect.bisect_left(timestamps, timestamp)
    if position < len(timestamps) and abs(timestamps[position] - timestamp) < 1e-9:
        return poses[position], 0.0
    if position == 0 or position == len(timestamps):
        raise ValueError(
            f"Timestamp {timestamp:.9f} lies outside the ground-truth range")
    lower = position - 1
    upper = position
    span = timestamps[upper] - timestamps[lower]
    if span > max_bracket_span:
        raise ValueError(
            f"Ground-truth bracket around {timestamp:.9f} spans "
            f"{span:.6f}s, exceeding {max_bracket_span:.6f}s")
    alpha = (timestamp - timestamps[lower]) / span
    lower_pose = poses[lower]
    upper_pose = poses[upper]
    translation = tuple(
        (1.0 - alpha) * lower_pose[index] + alpha * upper_pose[index]
        for index in range(3)
    )
    lower_quaternion = normalize_quaternion(lower_pose[3:])
    upper_quaternion = normalize_quaternion(upper_pose[3:])
    dot = sum(
        left * right
        for left, right in zip(lower_quaternion, upper_quaternion))
    if dot < 0.0:
        upper_quaternion = tuple(-value for value in upper_quaternion)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        quaternion = normalize_quaternion(tuple(
            (1.0 - alpha) * left + alpha * right
            for left, right in zip(lower_quaternion, upper_quaternion)
        ))
    else:
        angle = math.acos(dot)
        denominator = math.sin(angle)
        lower_weight = math.sin((1.0 - alpha) * angle) / denominator
        upper_weight = math.sin(alpha * angle) / denominator
        quaternion = tuple(
            lower_weight * left + upper_weight * right
            for left, right in zip(lower_quaternion, upper_quaternion)
        )
    return translation + quaternion, span


def timestamp_has_ground_truth(timestamps: list[float], timestamp: float) -> bool:
    tolerance = 1e-9
    return (
        timestamps[0] - tolerance <= timestamp <=
        timestamps[-1] + tolerance)


def quaternion_matrix(pose: tuple[float, ...]) -> list[list[float]]:
    qx, qy, qz, qw = pose[3:]
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("Pose contains a zero-norm quaternion")
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(values) for values in zip(*matrix)]


def matmul(
    left: list[list[float]], right: list[list[float]]
) -> list[list[float]]:
    right_t = transpose(right)
    return [
        [sum(a * b for a, b in zip(row, column)) for column in right_t]
        for row in left
    ]


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def relative_pose(
    previous: tuple[float, ...], current: tuple[float, ...]
) -> tuple[list[list[float]], list[float]]:
    previous_rotation = quaternion_matrix(previous)
    current_rotation = quaternion_matrix(current)
    previous_rotation_t = transpose(previous_rotation)
    rotation = matmul(previous_rotation_t, current_rotation)
    translation_delta = [
        current[index] - previous[index] for index in range(3)
    ]
    translation = matvec(previous_rotation_t, translation_delta)
    return rotation, translation


def relative_error(
    ground_truth: tuple[tuple[float, ...], tuple[float, ...]],
    estimate: tuple[tuple[float, ...], tuple[float, ...]],
) -> tuple[float, float]:
    gt_rotation, gt_translation = relative_pose(*ground_truth)
    estimate_rotation, estimate_translation = relative_pose(*estimate)
    rotation_error = matmul(transpose(gt_rotation), estimate_rotation)
    translation_delta = [
        estimate_translation[index] - gt_translation[index]
        for index in range(3)
    ]
    translation_error = math.sqrt(
        sum(value * value for value in translation_delta))
    cosine = max(
        -1.0,
        min(
            1.0,
            (sum(rotation_error[index][index] for index in range(3)) - 1.0)
            / 2.0,
        ),
    )
    rotation_error_deg = math.degrees(math.acos(cosine))
    return translation_error, rotation_error_deg


def pose_from_row(row: dict[str, str], prefix: str) -> tuple[float, ...]:
    fields = [
        f"{prefix}_tx", f"{prefix}_ty", f"{prefix}_tz",
        f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz", f"{prefix}_qw",
    ]
    try:
        pose = tuple(float(row[field]) for field in fields)
    except KeyError as exc:
        raise ValueError(f"Counterfactual CSV is missing {exc.args[0]}") from exc
    if not all(math.isfinite(value) for value in pose):
        raise ValueError(f"Counterfactual CSV contains a non-finite {prefix} pose")
    return pose


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "pairs": 0,
            "static_translation_rmse_m": None,
            "dynamic_translation_rmse_m": None,
            "static_rotation_rmse_deg": None,
            "dynamic_rotation_rmse_deg": None,
            "dynamic_translation_win_rate": None,
            "dynamic_rotation_win_rate": None,
            "mean_inlier_gain": None,
            "mean_translation_utility_m": None,
            "mean_rotation_utility_deg": None,
            "hypothesis_translation_delta_rmse_m": None,
            "hypothesis_rotation_delta_rmse_deg": None,
            "static_translation_sum_squared_error_m2": 0.0,
            "dynamic_translation_sum_squared_error_m2": 0.0,
            "dynamic_translation_wins": 0,
            "worst_translation_degradation_m": None,
        }

    def rmse(field: str) -> float:
        return math.sqrt(statistics.fmean(row[field] ** 2 for row in rows))

    return {
        "pairs": len(rows),
        "static_translation_rmse_m": rmse("static_translation_error_m"),
        "dynamic_translation_rmse_m": rmse("dynamic_translation_error_m"),
        "static_rotation_rmse_deg": rmse("static_rotation_error_deg"),
        "dynamic_rotation_rmse_deg": rmse("dynamic_rotation_error_deg"),
        "dynamic_translation_win_rate": statistics.fmean(
            row["dynamic_translation_error_m"] <
            row["static_translation_error_m"]
            for row in rows),
        "dynamic_rotation_win_rate": statistics.fmean(
            row["dynamic_rotation_error_deg"] <
            row["static_rotation_error_deg"]
            for row in rows),
        "mean_inlier_gain": statistics.fmean(row["inlier_gain"] for row in rows),
        "mean_translation_utility_m": statistics.fmean(
            row["translation_utility_m"] for row in rows),
        "mean_rotation_utility_deg": statistics.fmean(
            row["rotation_utility_deg"] for row in rows),
        "hypothesis_translation_delta_rmse_m": rmse(
            "hypothesis_translation_delta_m"),
        "hypothesis_rotation_delta_rmse_deg": rmse(
            "hypothesis_rotation_delta_deg"),
        "static_translation_sum_squared_error_m2": sum(
            row["static_translation_error_m"] ** 2 for row in rows),
        "dynamic_translation_sum_squared_error_m2": sum(
            row["dynamic_translation_error_m"] ** 2 for row in rows),
        "dynamic_translation_wins": sum(
            row["dynamic_translation_error_m"] <
            row["static_translation_error_m"]
            for row in rows),
        "worst_translation_degradation_m": max(
            row["dynamic_translation_error_m"] -
            row["static_translation_error_m"]
            for row in rows),
    }


def evaluate(
    counterfactual_path: Path,
    ground_truth_path: Path,
    max_bracket_span: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gt_timestamps, gt_poses, gt_load = load_tum_poses(ground_truth_path)
    with counterfactual_path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"No counterfactual pose pairs found in {counterfactual_path}")

    evaluated_rows = []
    outside_ground_truth_frames = []
    seen_frames = set()
    for source in source_rows:
        frame = int(source["frame"])
        if frame in seen_frames:
            raise ValueError(f"Duplicate counterfactual frame {frame}")
        seen_frames.add(frame)
        timestamp = float(source["timestamp"])
        previous_timestamp = float(source["previous_timestamp"])
        if not (
            timestamp_has_ground_truth(gt_timestamps, previous_timestamp) and
            timestamp_has_ground_truth(gt_timestamps, timestamp)
        ):
            outside_ground_truth_frames.append(frame)
            continue
        gt_previous, previous_span = interpolate_pose(
            gt_timestamps, gt_poses, previous_timestamp, max_bracket_span)
        gt_current, current_span = interpolate_pose(
            gt_timestamps, gt_poses, timestamp, max_bracket_span)
        previous = pose_from_row(source, "previous")
        static = pose_from_row(source, "static")
        dynamic = pose_from_row(source, "dynamic")
        static_translation, static_rotation = relative_error(
            (gt_previous, gt_current), (previous, static))
        dynamic_translation, dynamic_rotation = relative_error(
            (gt_previous, gt_current), (previous, dynamic))
        hypothesis_translation_delta, hypothesis_rotation_delta = relative_error(
            (previous, static), (previous, dynamic))
        static_inliers = int(source["static_inliers"])
        dynamic_inliers = int(source["dynamic_inliers"])
        common_support = int(source.get("common_support", 0) or 0)
        static_common_score = float(
            source.get("static_common_score", -1.0) or -1.0)
        dynamic_common_score = float(
            source.get("dynamic_common_score", -1.0) or -1.0)
        common_score_available = int(
            common_support > 0 and
            math.isfinite(static_common_score) and
            math.isfinite(dynamic_common_score) and
            static_common_score >= 0.0 and
            dynamic_common_score >= 0.0)
        direct_common_support = int(
            source.get("direct_common_support", 0) or 0)
        direct_scores = {
            name: float(source.get(name, -1.0) or -1.0)
            for name in (
                "direct_static_depth_score",
                "direct_dynamic_depth_score",
                "direct_static_photometric_score",
                "direct_dynamic_photometric_score",
                "direct_static_combined_score",
                "direct_dynamic_combined_score",
            )
        }
        direct_score_available = int(
            direct_common_support > 0 and
            all(
                math.isfinite(value) and value >= 0.0
                for value in direct_scores.values()))
        evaluated_rows.append({
            "frame": frame,
            "timestamp": timestamp,
            "previous_timestamp": previous_timestamp,
            "gt_previous_bracket_span_s": previous_span,
            "gt_current_bracket_span_s": current_span,
            "prior_would_use": int(source["prior_would_use"]),
            "prior_used": int(source["prior_used"]),
            "static_inliers": static_inliers,
            "dynamic_inliers": dynamic_inliers,
            "inlier_gain": dynamic_inliers - static_inliers,
            "common_support": common_support,
            "static_common_score": static_common_score,
            "dynamic_common_score": dynamic_common_score,
            "common_score_available": common_score_available,
            "common_score_prefers_dynamic": int(
                common_score_available and
                dynamic_common_score < static_common_score),
            "direct_common_support": direct_common_support,
            **direct_scores,
            "direct_score_available": direct_score_available,
            "direct_depth_prefers_dynamic": int(
                direct_score_available and
                direct_scores["direct_dynamic_depth_score"] <
                direct_scores["direct_static_depth_score"]),
            "direct_photometric_prefers_dynamic": int(
                direct_score_available and
                direct_scores["direct_dynamic_photometric_score"] <
                direct_scores["direct_static_photometric_score"]),
            "direct_combined_prefers_dynamic": int(
                direct_score_available and
                direct_scores["direct_dynamic_combined_score"] <
                direct_scores["direct_static_combined_score"]),
            "static_translation_error_m": static_translation,
            "dynamic_translation_error_m": dynamic_translation,
            "static_rotation_error_deg": static_rotation,
            "dynamic_rotation_error_deg": dynamic_rotation,
            "translation_utility_m": static_translation - dynamic_translation,
            "rotation_utility_deg": static_rotation - dynamic_rotation,
            "hypothesis_translation_delta_m": hypothesis_translation_delta,
            "hypothesis_rotation_delta_deg": hypothesis_rotation_delta,
            "dynamic_translation_win": int(dynamic_translation < static_translation),
            "dynamic_rotation_win": int(dynamic_rotation < static_rotation),
        })

    if not evaluated_rows:
        raise ValueError(
            "No counterfactual pose pairs lie within the ground-truth "
            "timestamp range")

    gate_rows = [row for row in evaluated_rows if row["prior_would_use"]]
    strong_static_rows = [
        row for row in evaluated_rows if row["static_inliers"] >= 50]
    common_score_rows = [
        row for row in evaluated_rows if row["common_score_available"]]
    common_score_dynamic_rows = [
        row for row in common_score_rows
        if row["common_score_prefers_dynamic"]]
    common_score_static_rows = [
        row for row in common_score_rows
        if not row["common_score_prefers_dynamic"]]
    direct_score_rows = [
        row for row in evaluated_rows if row["direct_score_available"]]
    summary = {
        "protocol": (
            "Alignment-free adjacent-ground-truth SE(3) relative-pose error. "
            "Static and dynamic hypotheses come from the same frame/map state. "
            "Pairs requiring ground-truth extrapolation are explicitly excluded."
        ),
        "source_pairs": len(source_rows),
        "ground_truth_supported_pairs": len(evaluated_rows),
        "outside_ground_truth_range_pairs": len(outside_ground_truth_frames),
        "outside_ground_truth_range_frames": outside_ground_truth_frames,
        "max_ground_truth_bracket_span_s": max_bracket_span,
        "counterfactual_csv_sha256": sha256(counterfactual_path),
        "ground_truth_sha256": sha256(ground_truth_path),
        "ground_truth_duplicate_timestamp_groups":
            gt_load["duplicate_timestamp_groups"],
        "ground_truth_duplicate_rows_merged":
            gt_load["duplicate_rows_merged"],
        "ground_truth_duplicate_policy": (
            "Equal-timestamp translations are arithmetically averaged; "
            "quaternions are sign-aligned, averaged, and normalized."
        ),
        "all_candidates": summarize(evaluated_rows),
        "gate_would_use": summarize(gate_rows),
        "strong_static_support": summarize(strong_static_rows),
        "common_score_available": summarize(common_score_rows),
        "common_score_prefers_dynamic": summarize(
            common_score_dynamic_rows),
        "common_score_prefers_static": summarize(
            common_score_static_rows),
        "direct_score_available": summarize(direct_score_rows),
        "direct_depth_prefers_dynamic": summarize([
            row for row in direct_score_rows
            if row["direct_depth_prefers_dynamic"]]),
        "direct_photometric_prefers_dynamic": summarize([
            row for row in direct_score_rows
            if row["direct_photometric_prefers_dynamic"]]),
        "direct_combined_prefers_dynamic": summarize([
            row for row in direct_score_rows
            if row["direct_combined_prefers_dynamic"]]),
    }
    return evaluated_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("--max-ground-truth-gap", type=float, default=0.1)
    args = parser.parse_args()
    if args.max_ground_truth_gap <= 0:
        parser.error("--max-ground-truth-gap must be positive")
    counterfactual_path = args.run_dir / "motion_prior_counterfactual.csv"
    rows, summary = evaluate(
        counterfactual_path, args.ground_truth, args.max_ground_truth_gap)
    row_path = args.run_dir / "motion_prior_counterfactual_errors.csv"
    with row_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.run_dir / "motion_prior_counterfactual_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
