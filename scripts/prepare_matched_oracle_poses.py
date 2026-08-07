#!/usr/bin/env python3
"""Bind every association row to an interpolated GT Tcw pose."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from evaluate_checkpointed_forks import (
    interpolate_pose,
    read_ground_truth,
)


HEADER = "frame,timestamp,valid,tx,ty,tz,qx,qy,qz,qw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_association_timestamps(path: Path) -> list[float]:
    timestamps: list[float] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise ValueError(
                    f"{path}:{line_number}: malformed association row")
            timestamp = float(fields[0])
            if not math.isfinite(timestamp):
                raise ValueError(
                    f"{path}:{line_number}: non-finite timestamp")
            timestamps.append(timestamp)
    if not timestamps:
        raise ValueError(f"No association rows in {path}")
    return timestamps


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Invalid rotation matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray((qx, qy, qz, qw), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def prepare(
    association: Path,
    ground_truth: Path,
    output: Path,
    metadata_output: Path,
    max_ground_truth_gap: float,
) -> dict[str, object]:
    if output.exists() or metadata_output.exists():
        raise FileExistsError("Refusing to overwrite matched-oracle outputs")
    association_timestamps = read_association_timestamps(association)
    gt_timestamps, gt_poses = read_ground_truth(ground_truth)

    rows: list[str] = [HEADER]
    valid_count = 0
    for frame, timestamp in enumerate(association_timestamps):
        twc = interpolate_pose(
            timestamp, gt_timestamps, gt_poses, max_ground_truth_gap)
        if twc is None:
            rows.append(
                f"{frame},{timestamp:.9f},0,"
                "0.000000000,0.000000000,0.000000000,"
                "0.000000000,0.000000000,0.000000000,1.000000000")
            continue
        tcw = np.linalg.inv(twc)
        quaternion = rotation_matrix_to_quaternion(tcw[:3, :3])
        translation = tcw[:3, 3]
        values = [*translation.tolist(), *quaternion.tolist()]
        rows.append(
            f"{frame},{timestamp:.9f},1," +
            ",".join(f"{value:.9f}" for value in values))
        valid_count += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n")
    metadata = {
        "protocol": "matched-sham-oracle-pose-manifest-v1",
        "pose_convention": "Tcw",
        "selection_metric": (
            "instantaneous camera-center translation error; dynamic must "
            "improve by more than 1e-6 m; ties choose static"),
        "interpolation": (
            "linear translation and shortest-arc quaternion interpolation"),
        "max_ground_truth_bracket_seconds": max_ground_truth_gap,
        "association": str(association.resolve()),
        "ground_truth": str(ground_truth.resolve()),
        "association_sha256": sha256(association),
        "ground_truth_sha256": sha256(ground_truth),
        "manifest": str(output.resolve()),
        "manifest_sha256": sha256(output),
        "input_frames": len(association_timestamps),
        "valid_ground_truth_frames": valid_count,
        "invalid_ground_truth_frames": (
            len(association_timestamps) - valid_count),
    }
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("association", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--max-ground-truth-gap", type=float, default=0.1)
    args = parser.parse_args()
    if args.max_ground_truth_gap <= 0.0:
        parser.error("--max-ground-truth-gap must be positive")
    metadata_output = args.metadata_output or args.output.with_suffix(
        ".metadata.json")
    metadata = prepare(
        args.association, args.ground_truth, args.output,
        metadata_output, args.max_ground_truth_gap)
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
