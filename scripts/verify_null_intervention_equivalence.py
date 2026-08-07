#!/usr/bin/env python3
"""Fail-closed equivalence gate for repeated shadow-only SLAM runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


EXECUTION_FIELDS = (
    "frame",
    "timestamp",
    "valid",
    "tracking_state",
    "is_keyframe",
    "prior_would_use_id",
    "prior_selected_id",
    "internal_frame_id",
    "map_id",
    "map_generation",
    "reference_kf_id",
    "current_gray_hash",
    "current_depth_hash",
    "current_static_mask_hash",
    "current_descriptors_hash",
    "feature_count",
    "motion_model_ran",
    "motion_previous_pose_hash",
    "motion_static_initial_pose_hash",
    "motion_static_support_hash",
    "motion_static_support_count",
    "motion_static_inliers",
    "motion_static_optimized_pose_hash",
    "motion_dynamic_initial_pose_hash",
    "motion_dynamic_support_hash",
    "motion_dynamic_support_count",
    "motion_dynamic_inliers",
    "motion_dynamic_optimized_pose_hash",
    "motion_output_pose_hash",
    "local_map_ran",
    "local_map_entry_pose_hash",
    "local_keyframe_sequence_hash",
    "local_keyframe_count",
    "local_candidate_state_hash",
    "local_candidate_id_hash",
    "local_candidate_position_hash",
    "local_candidate_normal_hash",
    "local_candidate_distance_hash",
    "local_candidate_descriptor_hash",
    "local_candidate_observation_hash",
    "local_candidate_count",
    "local_map_support_hash",
    "local_map_support_count",
    "local_map_optimizer_inliers",
    "local_map_optimized_pose_hash",
    "current_pose_hash",
    "map_point_count",
    "keyframe_count",
    "map_point_hash",
    "keyframe_hash",
    "map_fingerprint",
    "freeze_protocol",
    "freeze_requested",
    "local_mapping_idle_ack",
    "local_mapping_stopped_ack",
    "loop_closing_idle_ack",
    "loop_closing_stopped_ack",
    "gba_stopped_ack",
    "freeze_epoch",
)

PACKET_INDEX_FIELDS = (
    "frame",
    "timestamp",
    "previous_frame_id",
    "map_id",
    "map_generation",
    "reference_kf_id",
    "map_point_state_hash",
    "static_support_hash",
    "dynamic_support_hash",
    "common_support_hash",
    "freeze_protocol",
    "freeze_requested",
    "local_mapping_idle_ack",
    "local_mapping_stopped_ack",
    "loop_closing_idle_ack",
    "loop_closing_stopped_ack",
    "gba_stopped_ack",
    "freeze_epoch",
)

FREEZE_PROTOCOL = "deterministic-map-freeze-v2"
FREEZE_ACK_FIELDS = (
    "freeze_requested",
    "local_mapping_idle_ack",
    "local_mapping_stopped_ack",
    "loop_closing_idle_ack",
    "loop_closing_stopped_ack",
    "gba_stopped_ack",
)


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Missing required artifact: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
        return list(reader)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_table_difference(
    baseline: list[dict[str, str]],
    candidate: list[dict[str, str]],
    fields: tuple[str, ...],
    artifact: str,
) -> dict[str, Any] | None:
    if len(baseline) != len(candidate):
        return {
            "artifact": artifact,
            "field": "row_count",
            "baseline": len(baseline),
            "candidate": len(candidate),
            "row": min(len(baseline), len(candidate)),
        }
    for row_index, (left, right) in enumerate(zip(baseline, candidate)):
        for field in fields:
            if left[field] != right[field]:
                return {
                    "artifact": artifact,
                    "row": row_index,
                    "frame": left.get("frame", ""),
                    "field": field,
                    "baseline": left[field],
                    "candidate": right[field],
                }
    return None


def verify_replay_packets(
    run_dir: Path,
    index_rows: list[dict[str, str]],
    replayer: Path,
) -> None:
    if not replayer.is_file():
        raise ValueError(f"Missing frozen replayer: {replayer}")
    packet_dir = run_dir / "replay_packets"
    packet_paths = [packet_dir / row["packet"] for row in index_rows]
    missing = [str(path) for path in packet_paths if not path.is_file()]
    if missing:
        raise ValueError("Missing replay packets: " + ", ".join(missing[:3]))
    for start in range(0, len(packet_paths), 100):
        command = [
            str(replayer),
            *(str(path) for path in packet_paths[start : start + 100]),
        ]
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        if completed.returncode != 0:
            output = (completed.stderr or completed.stdout).strip()
            raise ValueError(
                f"Frozen replay failed for {run_dir}: {output[:2000]}"
            )


def verify_run(run_dir: Path, replayer: Path) -> dict[str, Any]:
    execution = read_csv(
        run_dir / "replay_execution_state.csv", EXECUTION_FIELDS
    )
    if not execution:
        raise ValueError(f"No execution-state rows in {run_dir}")
    selected = [
        row for row in execution if int(row["prior_selected_id"]) >= 0
    ]
    if selected:
        raise ValueError(
            f"Null intervention violated in {run_dir} at frame "
            f"{selected[0]['frame']}"
        )
    frame_metrics = read_csv(
        run_dir / "frame_metrics.csv", ("frame", "prior_used")
    )
    used = [row for row in frame_metrics if int(row["prior_used"]) != 0]
    if used:
        raise ValueError(
            f"frame_metrics reports intervention in {run_dir} at frame "
            f"{used[0]['frame']}"
        )
    packet_index = read_csv(
        run_dir / "replay_packets" / "index.csv",
        ("packet", *PACKET_INDEX_FIELDS),
    )
    if not packet_index:
        raise ValueError(f"No frozen replay packets in {run_dir}")
    for row in packet_index:
        if row["freeze_protocol"] != FREEZE_PROTOCOL:
            raise ValueError(
                f"Unsupported freeze protocol in {run_dir} at frame "
                f"{row['frame']}: {row['freeze_protocol']}"
            )
        missing_ack = [
            field for field in FREEZE_ACK_FIELDS
            if row[field] != "1"
        ]
        if missing_ack or int(row["freeze_epoch"]) <= 0:
            raise ValueError(
                f"Incomplete map-freeze acknowledgement in {run_dir} "
                f"at frame {row['frame']}: "
                f"{', '.join(missing_ack) or 'freeze_epoch'}"
            )
    verify_replay_packets(run_dir, packet_index, replayer)
    trajectory = run_dir / "CameraTrajectory_AllFrames_TUM.txt"
    if not trajectory.is_file() or trajectory.stat().st_size == 0:
        raise ValueError(f"Missing all-frame trajectory: {trajectory}")
    return {
        "run_dir": str(run_dir.resolve()),
        "execution": execution,
        "packet_index": packet_index,
        "trajectory_sha256": sha256(trajectory),
        "execution_rows": len(execution),
        "packet_count": len(packet_index),
        "would_use_ids": [
            row["prior_would_use_id"]
            for row in execution
            if int(row["prior_would_use_id"]) >= 0
        ],
    }


def evaluate(run_dirs: list[Path], replayer: Path) -> dict[str, Any]:
    if len(run_dirs) < 3:
        raise ValueError("Null equivalence requires at least three repeats")
    runs = [verify_run(path, replayer) for path in run_dirs]
    baseline = runs[0]
    for candidate in runs[1:]:
        difference = first_table_difference(
            baseline["execution"],
            candidate["execution"],
            EXECUTION_FIELDS,
            "replay_execution_state.csv",
        )
        if difference is None:
            difference = first_table_difference(
                baseline["packet_index"],
                candidate["packet_index"],
                PACKET_INDEX_FIELDS,
                "replay_packets/index.csv",
            )
        if difference is None and (
            baseline["trajectory_sha256"] != candidate["trajectory_sha256"]
        ):
            difference = {
                "artifact": "CameraTrajectory_AllFrames_TUM.txt",
                "field": "sha256",
                "baseline": baseline["trajectory_sha256"],
                "candidate": candidate["trajectory_sha256"],
            }
        if difference is not None:
            return {
                "status": "fail",
                "protocol": "null-intervention-equivalence-v1",
                "baseline_run": baseline["run_dir"],
                "candidate_run": candidate["run_dir"],
                "first_difference": difference,
                "runs": [
                    {
                        key: run[key]
                        for key in (
                            "run_dir",
                            "execution_rows",
                            "packet_count",
                            "trajectory_sha256",
                            "would_use_ids",
                        )
                    }
                    for run in runs
                ],
            }
    return {
        "status": "pass",
        "protocol": "null-intervention-equivalence-v1",
        "runs": [
            {
                key: run[key]
                for key in (
                    "run_dir",
                    "execution_rows",
                    "packet_count",
                    "trajectory_sha256",
                    "would_use_ids",
                )
            }
            for run in runs
        ],
    }


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--replayer",
        type=Path,
        default=project / "bin" / "frozen_same_state_replay",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(args.run_dirs, args.replayer)
    except (OSError, ValueError) as error:
        report = {
            "status": "error",
            "protocol": "null-intervention-equivalence-v1",
            "error": str(error),
        }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    sys.stdout.write(serialized)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
