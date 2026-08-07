#!/usr/bin/env python3
"""Evaluate frozen recovery episodes with a delayed whole-branch Oracle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_checkpointed_forks import (  # noqa: E402
    POSE_SUFFIXES,
    distribution,
    interpolate_pose,
    read_ground_truth,
    relative_error,
    row_pose,
)


PROTOCOL = "development-only-recovery-v2-20260730"
TRANSLATION_TIE_EPS_M = 1e-9
ROTATION_TIE_EPS_RAD = 1e-12
OUTPUT_FIELDS = (
    "event_id",
    "checkpoint",
    "horizon",
    "start_timestamp",
    "end_timestamp",
    "static_success",
    "dynamic_success",
    "matched_sham_success",
    "oracle_success",
    "oracle_selected_branch",
    "static_translation_rpe_m",
    "static_rotation_rpe_rad",
    "dynamic_translation_rpe_m",
    "dynamic_rotation_rpe_rad",
    "matched_sham_translation_rpe_m",
    "matched_sham_rotation_rpe_rad",
    "oracle_translation_rpe_m",
    "oracle_rotation_rpe_rad",
    "static_first_inliers",
    "static_min_inliers",
    "static_final_inliers",
    "dynamic_first_inliers",
    "dynamic_min_inliers",
    "dynamic_final_inliers",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")
    return rows


def exact_pose_equal(row: dict[str, str], left: str, right: str) -> bool:
    return all(
        row[f"{left}_{suffix}"] == row[f"{right}_{suffix}"]
        for suffix in POSE_SUFFIXES
    )


def choose_oracle(
    static_success: bool,
    dynamic_success: bool,
    static_translation: float | None,
    static_rotation: float | None,
    dynamic_translation: float | None,
    dynamic_rotation: float | None,
) -> str | None:
    if dynamic_success and not static_success:
        return "dynamic"
    if static_success and not dynamic_success:
        return "static"
    if not static_success and not dynamic_success:
        return None
    assert static_translation is not None and static_rotation is not None
    assert dynamic_translation is not None and dynamic_rotation is not None
    if dynamic_translation < static_translation - TRANSLATION_TIE_EPS_M:
        return "dynamic"
    if (
        abs(dynamic_translation - static_translation)
        <= TRANSLATION_TIE_EPS_M
        and dynamic_rotation < static_rotation - ROTATION_TIE_EPS_RAD
    ):
        return "dynamic"
    return "static"


def method_summary(
    rows: list[dict[str, Any]], prefix: str
) -> dict[str, Any]:
    successful = [row for row in rows if row[f"{prefix}_success"]]
    return {
        "successes": len(successful),
        "failures": len(rows) - len(successful),
        "failure_rate": (len(rows) - len(successful)) / len(rows),
        "translation_rpe_m": distribution(
            [row[f"{prefix}_translation_rpe_m"] for row in successful]
        ),
        "rotation_rpe_rad": distribution(
            [row[f"{prefix}_rotation_rpe_rad"] for row in successful]
        ),
    }


def evaluate(
    fork_csv: Path,
    episode_manifest: Path,
    ground_truth: Path,
    sequence: str,
    seed: int,
    max_ground_truth_gap: float = 0.1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not sequence or seed < 0:
        raise ValueError("Sequence and seed must identify one frozen cell")
    manifest_rows = load_csv(episode_manifest)
    fork_rows = load_csv(fork_csv)
    manifest = {
        (row["checkpoint"], int(row["horizon"])): row
        for row in manifest_rows
    }
    if len(manifest) != len(manifest_rows):
        raise ValueError("Episode manifest has duplicate checkpoint/horizon rows")
    selected: dict[tuple[str, int], dict[str, str]] = {}
    for row in fork_rows:
        key = (row["checkpoint"], int(row["horizon"]))
        if key not in manifest:
            raise ValueError("Fork CSV contains an event outside the manifest")
        if key in selected:
            raise ValueError("Fork CSV contains a duplicate event")
        selected[key] = row
    if set(selected) != set(manifest):
        missing = sorted(set(manifest) - set(selected))
        raise ValueError(f"Fork CSV is missing manifest events: {missing}")

    gt_timestamps, gt_poses = read_ground_truth(ground_truth)
    evaluated: list[dict[str, Any]] = []
    for key, manifest_row in manifest.items():
        row = selected[key]
        if row.get("event_id") != manifest_row["event_id"]:
            raise ValueError("Fork event ID does not match the frozen manifest")
        if (
            row["static_success"] != "1"
            or int(row["static_min_inliers"])
            != int(manifest_row["static_min_inliers"])
        ):
            raise ValueError("Static replay changed after the episode freeze")
        if row.get("static_scan_only") != "0":
            raise ValueError("Final replay cannot be a Static-only scan")
        if (
            row.get("prior_subspace") != "translation"
            or row.get("replay_mode") != "posterior-local-map-v1"
            or row.get("static_information_leverage_mode")
            != "normalize-to-target"
            or abs(float(row.get("max_static_information_leverage", "nan")) - 0.025)
            > 1e-8
        ):
            raise ValueError("Final replay does not match the frozen v2 method")
        required_execution = (
            "static_executed",
            "dynamic_executed",
            "matched_sham_static_executed",
            "matched_sham_dynamic_executed",
        )
        if any(row.get(field) != "1" for field in required_execution):
            raise ValueError("Matched-sham did not execute both branches")
        if row.get("matched_sham_selected_branch") != "static":
            raise ValueError("Matched-sham must always publish Static")
        if row["shadow_success"] != row["static_success"] or not exact_pose_equal(
            row, "shadow_tcw", "static_tcw"
        ):
            raise ValueError("Matched-sham is not exactly equivalent to Static")

        start_timestamp = float(row["start_timestamp"])
        end_timestamp = float(row["end_timestamp"])
        start_gt = interpolate_pose(
            start_timestamp, gt_timestamps, gt_poses, max_ground_truth_gap
        )
        end_gt = interpolate_pose(
            end_timestamp, gt_timestamps, gt_poses, max_ground_truth_gap
        )
        if start_gt is None or end_gt is None:
            raise ValueError("A frozen event endpoint did not match ground truth")

        start_tcw = row_pose(row, "start_tcw")
        static_success = row["static_success"] == "1"
        dynamic_success = row["velocity_neutral_success"] == "1"
        static_translation = static_rotation = None
        dynamic_translation = dynamic_rotation = None
        if static_success:
            static_translation, static_rotation = relative_error(
                start_tcw,
                row_pose(row, "static_tcw"),
                start_gt,
                end_gt,
            )
        if dynamic_success:
            dynamic_translation, dynamic_rotation = relative_error(
                start_tcw,
                row_pose(row, "velocity_neutral_tcw"),
                start_gt,
                end_gt,
            )
        oracle_branch = choose_oracle(
            static_success,
            dynamic_success,
            static_translation,
            static_rotation,
            dynamic_translation,
            dynamic_rotation,
        )
        oracle_translation = (
            dynamic_translation if oracle_branch == "dynamic" else static_translation
        )
        oracle_rotation = (
            dynamic_rotation if oracle_branch == "dynamic" else static_rotation
        )
        evaluated.append(
            {
                "event_id": manifest_row["event_id"],
                "checkpoint": row["checkpoint"],
                "horizon": int(row["horizon"]),
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "static_success": int(static_success),
                "dynamic_success": int(dynamic_success),
                "matched_sham_success": int(static_success),
                "oracle_success": int(oracle_branch is not None),
                "oracle_selected_branch": oracle_branch or "failure",
                "static_translation_rpe_m": static_translation,
                "static_rotation_rpe_rad": static_rotation,
                "dynamic_translation_rpe_m": dynamic_translation,
                "dynamic_rotation_rpe_rad": dynamic_rotation,
                "matched_sham_translation_rpe_m": static_translation,
                "matched_sham_rotation_rpe_rad": static_rotation,
                "oracle_translation_rpe_m": oracle_translation,
                "oracle_rotation_rpe_rad": oracle_rotation,
                "static_first_inliers": int(row["static_first_inliers"]),
                "static_min_inliers": int(row["static_min_inliers"]),
                "static_final_inliers": int(row["static_final_inliers"]),
                "dynamic_first_inliers": int(
                    row["velocity_neutral_first_inliers"]
                ),
                "dynamic_min_inliers": int(row["velocity_neutral_min_inliers"]),
                "dynamic_final_inliers": int(
                    row["velocity_neutral_final_inliers"]
                ),
            }
        )

    evaluated.sort(key=lambda row: row["start_timestamp"])
    methods = {
        name: method_summary(evaluated, name)
        for name in ("static", "dynamic", "matched_sham", "oracle")
    }
    static_mean = methods["static"]["translation_rpe_m"]["mean"]
    oracle_mean = methods["oracle"]["translation_rpe_m"]["mean"]
    static_rotation_mean = methods["static"]["rotation_rpe_rad"]["mean"]
    oracle_rotation_mean = methods["oracle"]["rotation_rpe_rad"]["mean"]
    summary = {
        "protocol": PROTOCOL,
        "sequence": sequence,
        "seed": seed,
        "horizon": int(manifest_rows[0]["horizon"]),
        "events": len(evaluated),
        "relative_error": "inverse(delta_gt) * delta_estimate",
        "trajectory_alignment": "none",
        "dynamic_branch": "Schur prior at event start; velocity-neutral H rollout",
        "max_static_information_leverage": 0.025,
        "static_information_leverage_mode": "normalize-to-target",
        "oracle_rule": (
            "lower H-frame translation RPE; rotation breaks translation ties; "
            "Static wins residual ties"
        ),
        "translation_tie_epsilon_m": TRANSLATION_TIE_EPS_M,
        "rotation_tie_epsilon_rad": ROTATION_TIE_EPS_RAD,
        "fork_csv_sha256": sha256(fork_csv),
        "episode_manifest_sha256": sha256(episode_manifest),
        "ground_truth_sha256": sha256(ground_truth),
        "matched_sham_exact_static": True,
        "oracle_dynamic_selections": sum(
            row["oracle_selected_branch"] == "dynamic" for row in evaluated
        ),
        "methods": methods,
        "oracle_strict_translation_improvement": (
            static_mean is not None
            and oracle_mean is not None
            and oracle_mean < static_mean - TRANSLATION_TIE_EPS_M
        ),
        "oracle_mean_rotation_non_regression": (
            static_rotation_mean is not None
            and oracle_rotation_mean is not None
            and oracle_rotation_mean <= static_rotation_mean + ROTATION_TIE_EPS_RAD
        ),
    }
    return summary, evaluated


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fork_csv", type=Path)
    parser.add_argument("--episode-manifest", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-ground-truth-gap", type=float, default=0.1)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.max_ground_truth_gap) or args.max_ground_truth_gap <= 0:
        parser.error("--max-ground-truth-gap must be finite and positive")
    try:
        summary, rows = evaluate(
            args.fork_csv,
            args.episode_manifest,
            args.ground_truth,
            args.sequence,
            args.seed,
            args.max_ground_truth_gap,
        )
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        write_rows(args.output_csv, rows)
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"recovery v2 evaluation failed: {error}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
