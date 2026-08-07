#!/usr/bin/env python3
"""Seal and evaluate a background-support shadow confirmation window.

The ``select`` phase reads only the DyPho-compatible Parent frame log.  It
must run before the candidate pose log or ground truth is inspected.  The
``evaluate`` phase then verifies the sealed window against ground truth.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    malformed = [
        index for index, row in enumerate(rows)
        if any(value is None for value in row.values())
    ]
    if malformed:
        if malformed != [len(rows) - 1]:
            raise ValueError(
                f"Malformed CSV rows are not confined to the tail: {path}"
            )
        rows.pop()
    return rows


def require_contiguous(rows: Sequence[Dict[str, str]]) -> None:
    frames = [int(row["frame"]) for row in rows]
    for previous, current in zip(frames, frames[1:]):
        if current != previous + 1:
            raise ValueError(
                f"Non-contiguous Parent frame log: {previous} -> {current}"
            )


def linear_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile over an empty set")
    return float(np.percentile(
        np.asarray(values, dtype=np.float64),
        percentile,
        method="linear",
    ))


def select_window(args: argparse.Namespace) -> None:
    frame_metrics = args.frame_metrics.resolve()
    rows = read_csv(frame_metrics)
    require_contiguous(rows)
    by_frame = {int(row["frame"]): row for row in rows}
    maximum_frame = max(by_frame)

    candidates: List[Dict[str, float]] = []
    last_start = maximum_frame - args.window_length + 1
    for start in range(args.minimum_start_frame, last_start + 1):
        window = [by_frame.get(frame) for frame in range(
            start, start + args.window_length
        )]
        if any(row is None for row in window):
            continue
        assert all(row is not None for row in window)
        typed_window = [row for row in window if row is not None]
        if any(int(row["pose_valid"]) != 1 or int(row["lost"]) != 0
               for row in typed_window):
            continue
        temporal_ratio = sum(
            int(row["temporal_refinement_applied"])
            for row in typed_window
        ) / args.window_length
        if temporal_ratio < args.minimum_temporal_ratio:
            continue
        dynamic_ratios = [
            float(row["dynamic_ratio"]) for row in typed_window
        ]
        candidates.append({
            "start_frame": float(start),
            "end_frame": float(start + args.window_length - 1),
            "mean_dynamic_ratio": float(np.mean(dynamic_ratios)),
            "minimum_dynamic_ratio": float(np.min(dynamic_ratios)),
            "maximum_dynamic_ratio": float(np.max(dynamic_ratios)),
            "temporal_refinement_ratio": temporal_ratio,
        })

    if not candidates:
        raise ValueError("No Parent-only window satisfies the frozen rules")

    dynamic_threshold = linear_percentile(
        [row["mean_dynamic_ratio"] for row in candidates],
        args.dynamic_percentile,
    )
    selected = next(
        row for row in candidates
        if row["mean_dynamic_ratio"] >= dynamic_threshold
    )
    start = int(selected["start_frame"])
    end = int(selected["end_frame"])
    selected_rows = [by_frame[frame] for frame in range(start, end + 1)]

    manifest = {
        "schema_version": 1,
        "selection_contract": "parent_only_before_candidate_or_gt_inspection",
        "frame_metrics": {
            "path": str(frame_metrics),
            "sha256": sha256_file(frame_metrics),
            "row_count": len(rows),
        },
        "rules": {
            "window_length": args.window_length,
            "minimum_start_frame": args.minimum_start_frame,
            "require_pose_valid_all_frames": True,
            "require_lost_zero_all_frames": True,
            "minimum_temporal_refinement_ratio":
                args.minimum_temporal_ratio,
            "dynamic_window_statistic": "mean_dynamic_ratio",
            "dynamic_percentile": args.dynamic_percentile,
            "dynamic_percentile_method": "linear",
            "selection": (
                "earliest eligible window at or above the frozen "
                "dynamic percentile"
            ),
        },
        "eligible_window_count": len(candidates),
        "dynamic_threshold": dynamic_threshold,
        "selected": {
            **selected,
            "start_frame": start,
            "end_frame": end,
            "start_timestamp": float(selected_rows[0]["timestamp"]),
            "end_timestamp": float(selected_rows[-1]["timestamp"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def quaternion_matrix(values: Iterable[float]) -> np.ndarray:
    x, y, z, w = np.asarray(list(values), dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Invalid quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray([
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ], dtype=np.float64)


def make_pose(
    translation: Iterable[float],
    quaternion_xyzw: Iterable[float],
) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_matrix(quaternion_xyzw)
    pose[:3, 3] = np.asarray(list(translation), dtype=np.float64)
    if not np.isfinite(pose).all():
        raise ValueError("Non-finite pose")
    return pose


def shadow_pose(
    row: Dict[str, str],
    prefix: str,
) -> np.ndarray:
    return make_pose(
        [
            float(row[f"{prefix}_twc_tx"]),
            float(row[f"{prefix}_twc_ty"]),
            float(row[f"{prefix}_twc_tz"]),
        ],
        [
            float(row[f"{prefix}_twc_qx"]),
            float(row[f"{prefix}_twc_qy"]),
            float(row[f"{prefix}_twc_qz"]),
            float(row[f"{prefix}_twc_qw"]),
        ],
    )


def read_tum_groundtruth(path: Path) -> Tuple[List[float], List[np.ndarray]]:
    timestamps: List[float] = []
    poses: List[np.ndarray] = []
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(value) for value in line.split()]
            if len(values) != 8:
                continue
            if timestamps and values[0] < timestamps[-1]:
                raise ValueError(
                    f"Ground-truth timestamps are not strictly increasing: "
                    f"{path}"
                )
            pose = np.asarray(values[1:], dtype=np.float64)
            if not np.isfinite(pose).all():
                raise ValueError(f"Non-finite ground-truth pose: {path}")
            quaternion_norm = float(np.linalg.norm(pose[3:]))
            if quaternion_norm <= 1e-12:
                raise ValueError(f"Zero-norm ground-truth quaternion: {path}")
            pose[3:] /= quaternion_norm
            if timestamps and values[0] == timestamps[-1]:
                previous = poses[-1]
                translation_delta = float(np.linalg.norm(
                    pose[:3] - previous[:3]
                ))
                quaternion = pose[3:].copy()
                dot = float(np.dot(previous[3:], quaternion))
                if dot < 0.0:
                    quaternion *= -1.0
                    dot = -dot
                rotation_delta = 2.0 * math.acos(float(np.clip(
                    dot, -1.0, 1.0
                )))
                if translation_delta > 0.01 or rotation_delta > 0.02:
                    raise ValueError(
                        "Conflicting ground-truth poses share timestamp "
                        f"{values[0]} in {path}"
                    )
                previous[:3] = 0.5 * (previous[:3] + pose[:3])
                previous[3:] += quaternion
                previous[3:] /= np.linalg.norm(previous[3:])
                continue
            timestamps.append(values[0])
            poses.append(pose)
    if not timestamps:
        raise ValueError(f"No ground-truth poses found in {path}")
    return timestamps, poses


def interpolate_pose(
    timestamp: float,
    timestamps: Sequence[float],
    poses: Sequence[np.ndarray],
    maximum_bracket_span: float,
) -> np.ndarray | None:
    upper = bisect.bisect_left(timestamps, timestamp)
    if upper < len(timestamps) and abs(timestamps[upper] - timestamp) <= 1e-9:
        return make_pose(poses[upper][:3], poses[upper][3:])
    if upper == 0 or upper == len(timestamps):
        return None
    lower = upper - 1
    span = timestamps[upper] - timestamps[lower]
    if span <= 0.0 or span > maximum_bracket_span:
        return None
    alpha = (timestamp - timestamps[lower]) / span
    translation = (
        (1.0 - alpha) * poses[lower][:3]
        + alpha * poses[upper][:3]
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
            (1.0 - alpha) * lower_quaternion
            + alpha * upper_quaternion
        )
        quaternion /= np.linalg.norm(quaternion)
    else:
        angle = math.acos(dot)
        denominator = math.sin(angle)
        quaternion = (
            math.sin((1.0 - alpha) * angle) / denominator
            * lower_quaternion
            + math.sin(alpha * angle) / denominator
            * upper_quaternion
        )
    return make_pose(translation, quaternion)


def relative_error(
    previous_estimate: np.ndarray,
    current_estimate: np.ndarray,
    previous_groundtruth: np.ndarray,
    current_groundtruth: np.ndarray,
) -> Tuple[float, float]:
    estimate_delta = np.linalg.inv(previous_estimate) @ current_estimate
    groundtruth_delta = (
        np.linalg.inv(previous_groundtruth) @ current_groundtruth
    )
    error = np.linalg.inv(groundtruth_delta) @ estimate_delta
    translation = float(np.linalg.norm(error[:3, 3]))
    cosine = float(np.clip(
        (np.trace(error[:3, :3]) - 1.0) * 0.5,
        -1.0,
        1.0,
    ))
    rotation = math.acos(cosine)
    return translation, rotation


def mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def evaluate_window(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = manifest["selected"]
    start = int(selected["start_frame"])
    end = int(selected["end_frame"])

    pose_rows = read_csv(args.pose_shadow)
    pose_by_frame = {int(row["frame"]): row for row in pose_rows}
    required = range(start - 1, end + 1)
    missing = [frame for frame in required if frame not in pose_by_frame]
    if missing:
        raise ValueError(
            f"Candidate pose log is missing required frames: {missing[:8]}"
        )

    gt_timestamps, gt_poses = read_tum_groundtruth(args.groundtruth)
    gate_records = []
    gt_unevaluable_records = []
    all_gate_rows = []
    for frame in range(start, end + 1):
        row = pose_by_frame[frame]
        if int(row["gate_pass"]) != 1:
            continue
        all_gate_rows.append(row)
        previous = pose_by_frame[frame - 1]
        previous_parent = shadow_pose(previous, "parent")
        parent = shadow_pose(row, "parent")
        candidate = shadow_pose(row, "candidate")
        previous_gt = interpolate_pose(
            float(previous["timestamp"]),
            gt_timestamps,
            gt_poses,
            args.maximum_gt_bracket_span,
        )
        current_gt = interpolate_pose(
            float(row["timestamp"]),
            gt_timestamps,
            gt_poses,
            args.maximum_gt_bracket_span,
        )
        if previous_gt is None or current_gt is None:
            gt_unevaluable_records.append({
                "frame": frame,
                "timestamp": float(row["timestamp"]),
                "reason": "groundtruth_bracket_unavailable_or_too_wide",
            })
            continue
        parent_translation, parent_rotation = relative_error(
            previous_parent, parent, previous_gt, current_gt
        )
        candidate_translation, candidate_rotation = relative_error(
            previous_parent, candidate, previous_gt, current_gt
        )
        gate_records.append({
            "frame": frame,
            "timestamp": float(row["timestamp"]),
            "translation_correction_m":
                float(row["translation_correction"]),
            "rotation_correction_rad":
                float(row["rotation_correction"]),
            "parent_translation_rpe_m": parent_translation,
            "candidate_translation_rpe_m": candidate_translation,
            "parent_rotation_rpe_rad": parent_rotation,
            "candidate_rotation_rpe_rad": candidate_rotation,
            "translation_win":
                candidate_translation < parent_translation,
            "rotation_non_regression": (
                candidate_rotation
                <= parent_rotation + args.rotation_tolerance
            ),
        })

    translation_wins = sum(
        int(row["translation_win"]) for row in gate_records
    )
    rotation_non_regressions = sum(
        int(row["rotation_non_regression"]) for row in gate_records
    )
    gate_count = len(all_gate_rows)
    gt_evaluable_count = len(gate_records)
    if gate_records:
        parent_translation = mean([
            row["parent_translation_rpe_m"] for row in gate_records
        ])
        candidate_translation = mean([
            row["candidate_translation_rpe_m"] for row in gate_records
        ])
        maximum_translation_correction = max(
            row["translation_correction_m"] for row in gate_records
        )
        maximum_rotation_correction = max(
            row["rotation_correction_rad"] for row in gate_records
        )
    else:
        parent_translation = None
        candidate_translation = None
        maximum_translation_correction = None
        maximum_rotation_correction = None

    maximum_translation_correction_all = (
        max(float(row["translation_correction"]) for row in all_gate_rows)
        if all_gate_rows else None
    )
    maximum_rotation_correction_all = (
        max(float(row["rotation_correction"]) for row in all_gate_rows)
        if all_gate_rows else None
    )
    minimum_evaluable_count = max(
        args.minimum_gt_evaluable_gate_frames,
        int(math.ceil(args.minimum_gt_evaluable_ratio * gate_count)),
    )
    gt_coverage_pass = (
        gate_count > 0 and gt_evaluable_count >= minimum_evaluable_count
    )
    local_confirmation_pass = (
        gt_coverage_pass
        and translation_wins / gt_evaluable_count > 0.5
        and parent_translation is not None
        and candidate_translation is not None
        and candidate_translation < parent_translation
        and rotation_non_regressions == gt_evaluable_count
        and maximum_translation_correction_all is not None
        and maximum_translation_correction_all
            <= args.maximum_translation_correction + 1e-6
        and maximum_rotation_correction_all is not None
        and maximum_rotation_correction_all <= args.rotation_tolerance
    )
    report = {
        "schema_version": 1,
        "decision_scope": (
            "authorize_or_reject implementation of the frozen four-frame "
            "verified correction state; not a complete method claim"
        ),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
        },
        "pose_shadow": {
            "path": str(args.pose_shadow.resolve()),
            "sha256": sha256_file(args.pose_shadow),
        },
        "groundtruth": {
            "path": str(args.groundtruth.resolve()),
            "sha256": sha256_file(args.groundtruth),
        },
        "window": selected,
        "gate_frame_count": gate_count,
        "gt_evaluable_gate_frame_count": gt_evaluable_count,
        "gt_unevaluable_gate_frame_count": len(gt_unevaluable_records),
        "minimum_gt_evaluable_gate_frame_count": minimum_evaluable_count,
        "gt_evaluable_gate_ratio": (
            gt_evaluable_count / gate_count if gate_count else 0.0
        ),
        "translation_wins": translation_wins,
        "translation_win_rate":
            translation_wins / gt_evaluable_count
            if gt_evaluable_count else 0.0,
        "rotation_non_regressions": rotation_non_regressions,
        "parent_gate_mean_translation_rpe_m": parent_translation,
        "candidate_gate_mean_translation_rpe_m": candidate_translation,
        "relative_gate_mean_translation_improvement": (
            (parent_translation - candidate_translation)
            / parent_translation
            if parent_translation is not None
            and candidate_translation is not None
            and parent_translation > 0.0
            else None
        ),
        "maximum_translation_correction_m":
            maximum_translation_correction_all,
        "maximum_rotation_correction_rad": maximum_rotation_correction_all,
        "frozen_gate": {
            "gate_frame_count_nonzero": gate_count > 0,
            "gt_evaluable_gate_count_minimum":
                args.minimum_gt_evaluable_gate_frames,
            "gt_evaluable_gate_ratio_minimum":
                args.minimum_gt_evaluable_ratio,
            "gt_coverage_pass": gt_coverage_pass,
            "translation_win_rate_strict_majority": (
                gt_evaluable_count > 0
                and translation_wins / gt_evaluable_count > 0.5
            ),
            "mean_translation_rpe_improves": (
                parent_translation is not None
                and candidate_translation is not None
                and candidate_translation < parent_translation
            ),
            "rotation_non_regression_all_gate_frames": (
                gt_evaluable_count > 0
                and rotation_non_regressions == gt_evaluable_count
            ),
            "maximum_translation_correction_m":
                args.maximum_translation_correction,
            "rotation_tolerance_rad": args.rotation_tolerance,
        },
        "decision": (
            "LOCAL_CONFIRMATION_PASS_AUTHORIZE_VERIFIED_STATE"
            if local_confirmation_pass
            else "LOCAL_CONFIRMATION_FAIL_STOP_BACKGROUND_RESCUE"
        ),
        "gate_records": gate_records,
        "gt_unevaluable_gate_records": gt_unevaluable_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--frame-metrics", type=Path, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument("--window-length", type=int, default=150)
    select_parser.add_argument("--minimum-start-frame", type=int, default=300)
    select_parser.add_argument(
        "--minimum-temporal-ratio", type=float, default=0.90
    )
    select_parser.add_argument(
        "--dynamic-percentile", type=float, default=75.0
    )
    select_parser.set_defaults(function=select_window)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument("--pose-shadow", type=Path, required=True)
    evaluate_parser.add_argument("--groundtruth", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument(
        "--maximum-gt-bracket-span", type=float, default=0.1
    )
    evaluate_parser.add_argument(
        "--minimum-gt-evaluable-gate-frames", type=int, default=10
    )
    evaluate_parser.add_argument(
        "--minimum-gt-evaluable-ratio", type=float, default=0.5
    )
    evaluate_parser.add_argument(
        "--maximum-translation-correction", type=float, default=0.02
    )
    evaluate_parser.add_argument(
        "--rotation-tolerance", type=float, default=1e-6
    )
    evaluate_parser.set_defaults(function=evaluate_window)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
