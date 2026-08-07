#!/usr/bin/env python3
"""Prepare common held-out views and poses for final Gaussian-map evaluation.

The script selects RGB observations that are not mapping keyframes in any
compared run. For every run it writes two pose files:

* online: the run's estimated pose, for end-to-end self-consistency;
* gt_aligned: the ground-truth pose rigidly aligned into that run's map frame.

GT-aligned final-map rendering is deliberately not called "mapping-only": the
map was still constructed with online estimated poses, so tracking errors can
remain baked into its geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2


BONN_DATASET_SOURCE = (
    "https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/"
)
BONN_T_ROS = np.asarray([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)
BONN_T_ROS_INVERSE = np.linalg.inv(BONN_T_ROS)
BONN_T_M = np.asarray([
    [1.0157, 0.1828, -0.2389, 0.0113],
    [0.0009, -0.8431, -0.6413, -0.0098],
    [-0.3009, 0.6147, -0.8085, 0.0111],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


@dataclass(frozen=True)
class Pose:
    timestamp: float
    translation: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class RunArtifacts:
    name: str
    run_dir: Path
    manifest_path: Path
    manifest: dict
    trajectory_path: Path
    orb_config: Path
    gaussian_config: Path
    cameras_path: Path
    ply_path: Path
    keyframe_timestamps: np.ndarray
    trajectory: list[Pose]
    final_map_keyframes: list[Pose]


def load_tum_trajectory(path: Path) -> list[Pose]:
    poses: list[Pose] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 8:
                raise ValueError(f"{path}:{line_number}: expected 8 TUM fields")
            values = [float(value) for value in fields[:8]]
            quaternion = np.asarray(values[4:8], dtype=np.float64)
            norm = float(np.linalg.norm(quaternion))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise ValueError(f"{path}:{line_number}: invalid quaternion")
            poses.append(Pose(
                timestamp=values[0],
                translation=np.asarray(values[1:4], dtype=np.float64),
                quaternion_xyzw=quaternion / norm,
            ))
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return sorted(poses, key=lambda pose: pose.timestamp)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_associations(path: Path) -> list[tuple[int, float, str]]:
    rows: list[tuple[int, float, str]] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                continue
            rows.append((len(rows), float(fields[0]), fields[1]))
    if not rows:
        raise ValueError(f"No RGB associations found in {path}")
    return rows


def nearest_pose(poses: list[Pose], timestamp: float, max_delta: float) -> Pose | None:
    times = np.fromiter((pose.timestamp for pose in poses), dtype=np.float64)
    index = int(np.searchsorted(times, timestamp))
    choices = [candidate for candidate in (index - 1, index)
               if 0 <= candidate < len(poses)]
    best = min(choices, key=lambda candidate: abs(times[candidate] - timestamp))
    return poses[best] if abs(times[best] - timestamp) <= max_delta else None


def quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray([
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            ])
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray([
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            ])
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray([
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ])
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion


def closest_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    left, _, right_transpose = np.linalg.svd(matrix)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return rotation


def transform_bonn_rgbd_sensor_pose(pose: Pose) -> Pose:
    marker_pose = np.eye(4, dtype=np.float64)
    marker_pose[:3, :3] = quaternion_to_matrix(pose.quaternion_xyzw)
    marker_pose[:3, 3] = pose.translation
    sensor_pose = (
        BONN_T_ROS_INVERSE @ marker_pose @ BONN_T_ROS @ BONN_T_M
    )
    sensor_rotation = closest_rotation_matrix(sensor_pose[:3, :3])
    return Pose(
        timestamp=pose.timestamp,
        translation=sensor_pose[:3, 3],
        quaternion_xyzw=matrix_to_quaternion(sensor_rotation),
    )


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Alignment inputs must have matching Nx3 shapes")
    if len(source) < 3:
        raise ValueError("At least three pose pairs are required for alignment")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transpose[-1, :] *= -1.0
        rotation = right_transpose.T @ left.T
    translation = target_center - rotation @ source_center
    aligned = (rotation @ source.T).T + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return rotation, translation, rmse


def final_map_directory(run_dir: Path) -> Path:
    candidates = []
    for path in run_dir.glob("*_shutdown"):
        match = re.fullmatch(r"(\d+)_shutdown", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No *_shutdown final-map directory in {run_dir}")
    return max(candidates)[1]


def final_gaussian_ply(map_dir: Path) -> Path:
    candidates = []
    point_cloud_root = map_dir / "ply/point_cloud"
    for path in point_cloud_root.glob("iteration_*/point_cloud.ply"):
        match = re.fullmatch(r"iteration_(\d+)", path.parent.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"No final Gaussian point_cloud.ply found under {point_cloud_root}")
    return max(candidates)[1]


def load_final_map_cameras(cameras_path: Path) -> list[Pose]:
    cameras = json.loads(cameras_path.read_text())
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"Invalid or empty cameras.json: {cameras_path}")
    poses = []
    for camera in cameras:
        image_name = str(camera.get("img_name", ""))
        try:
            timestamp = float(Path(image_name).stem)
        except ValueError as exc:
            raise ValueError(
                f"Camera image name is not a timestamp: {image_name}") from exc
        translation = np.asarray(camera.get("position"), dtype=np.float64)
        rotation = np.asarray(camera.get("rotation"), dtype=np.float64)
        if translation.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError(f"Invalid final-map camera pose for {image_name}")
        if not np.isfinite(translation).all() or not np.isfinite(rotation).all():
            raise ValueError(f"Non-finite final-map camera pose for {image_name}")
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-4)
        ):
            raise ValueError(f"Non-rigid final-map camera rotation for {image_name}")
        poses.append(Pose(
            timestamp=timestamp,
            translation=translation,
            quaternion_xyzw=matrix_to_quaternion(rotation),
        ))
    poses.sort(key=lambda pose: pose.timestamp)
    timestamps = [pose.timestamp for pose in poses]
    if len(timestamps) != len(set(timestamps)):
        raise ValueError(f"Duplicate timestamps in {cameras_path}")
    return poses


def camera_model(path: Path) -> dict[str, float]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"Cannot open camera config: {path}")
    aliases = {
        "width": (
            "Camera.w", "Camera.width", "Camera1.w", "Camera1.width"),
        "height": (
            "Camera.h", "Camera.height", "Camera1.h", "Camera1.height"),
        "fx": ("Camera.fx", "Camera1.fx"),
        "fy": ("Camera.fy", "Camera1.fy"),
        "cx": ("Camera.cx", "Camera1.cx"),
        "cy": ("Camera.cy", "Camera1.cy"),
        "k1": ("Camera.k1", "Camera1.k1"),
        "k2": ("Camera.k2", "Camera1.k2"),
        "p1": ("Camera.p1", "Camera1.p1"),
        "p2": ("Camera.p2", "Camera1.p2"),
        "k3": ("Camera.k3", "Camera1.k3"),
    }
    values = {}
    try:
        for name, keys in aliases.items():
            value = None
            for key in keys:
                node = storage.getNode(key)
                if not node.empty():
                    value = float(node.real())
                    break
            if value is None:
                value = 0.0 if name == "k3" else None
            if value is None or not math.isfinite(value):
                raise ValueError(f"Missing or invalid {name} in {path}")
            values[name] = value
    finally:
        storage.release()
    return values


def validate_camera_compatibility(
    view_model: dict[str, float],
    orb_model: dict[str, float],
) -> None:
    for label, model in (("view", view_model), ("ORB", orb_model)):
        distortion = [model[key] for key in ("k1", "k2", "p1", "p2", "k3")]
        if any(abs(value) > 1e-12 for value in distortion):
            raise ValueError(
                f"Held-out preparation rejects non-zero distortion in the "
                f"{label} camera until RGB, masks, and render intrinsics share "
                "an explicit undistortion transform")
    for field in ("width", "height", "fx", "fy", "cx", "cy"):
        if not math.isclose(
            view_model[field], orb_model[field], rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                f"View/ORB camera mismatch for {field}: "
                f"{view_model[field]} != {orb_model[field]}")


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use NAME=RUN_DIR")
    name, raw_path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError(f"Unsafe run name: {name}")
    return name, Path(raw_path).resolve()


def validate_groundtruth_pose_model(sequence: str, pose_model: str) -> None:
    is_bonn = sequence.startswith("bonn_")
    if is_bonn and pose_model == "bonn_rgbd_sensor":
        return
    if is_bonn and pose_model == "camera":
        raise ValueError(
            "Bonn GT-aligned rendering requires --groundtruth-pose-model "
            "bonn_rgbd_sensor so marker poses are converted to optical RGB-D "
            "sensor poses. Raw groundtruth.txt remains correct for evo.")
    if pose_model == "bonn_mocap":
        raise ValueError(
            "bonn_mocap is deprecated because it conflated trajectory and "
            "rendering semantics; use bonn_rgbd_sensor for Bonn rendering")
    if not is_bonn and pose_model == "camera":
        return
    raise ValueError(
        "The Bonn marker-to-sensor transform cannot be applied to a non-Bonn "
        "camera trajectory")


def validate_run_identities(
    runs: list[RunArtifacts],
    dataset_dir: Path,
    association_path: Path,
    groundtruth_path: Path,
    view_camera_config: Path,
) -> dict:
    if not runs:
        raise ValueError("At least one run is required")
    expected_paths = {
        "dataset": dataset_dir,
        "association": association_path,
        "ground_truth": groundtruth_path,
    }
    shared_fields = ("sequence", "seed")
    reference = runs[0].manifest
    reference_source = reference.get("source", {}).get(
        "runtime_source_snapshot_sha256")
    if not reference_source:
        raise ValueError("Run manifest lacks runtime source snapshot identity")
    for run in runs:
        manifest = run.manifest
        for field in shared_fields:
            if manifest.get(field) != reference.get(field):
                raise ValueError(
                    f"Run identity mismatch for {field}: {run.name}")
        source_snapshot = manifest.get("source", {}).get(
            "runtime_source_snapshot_sha256")
        if source_snapshot != reference_source:
            raise ValueError(f"Source snapshot mismatch: {run.name}")
        for field, expected in expected_paths.items():
            recorded = Path(manifest.get("paths", {}).get(field, "")).resolve()
            if recorded != expected:
                raise ValueError(
                    f"Run {run.name} {field} mismatch: {recorded} != {expected}")
        hashes = manifest.get("hashes", {})
        required_hashes = {
            "association_sha256": association_path,
            "ground_truth_sha256": groundtruth_path,
            "orb_config_sha256": run.orb_config,
            "gaussian_config_sha256": run.gaussian_config,
        }
        for field, path in required_hashes.items():
            expected_hash = sha256(path)
            if hashes.get(field) != expected_hash:
                raise ValueError(
                    f"Run {run.name} has missing or stale {field}")

    reference_hashes = reference["hashes"]
    for run in runs[1:]:
        hashes = run.manifest["hashes"]
        for field in (
            "association_sha256",
            "ground_truth_sha256",
            "orb_config_sha256",
            "gaussian_config_sha256",
        ):
            if hashes[field] != reference_hashes[field]:
                raise ValueError(
                    f"Compared runs disagree on {field}: {run.name}")

    view_model = camera_model(view_camera_config)
    orb_model = camera_model(runs[0].orb_config)
    validate_camera_compatibility(view_model, orb_model)
    return {
        "sequence": reference["sequence"],
        "seed": reference["seed"],
        "runtime_source_snapshot_sha256": reference_source,
        "association_sha256": reference_hashes["association_sha256"],
        "ground_truth_sha256": reference_hashes["ground_truth_sha256"],
        "orb_config_sha256": reference_hashes["orb_config_sha256"],
        "gaussian_config_sha256": reference_hashes["gaussian_config_sha256"],
        "view_camera_config_sha256": sha256(view_camera_config),
        "view_camera_model": view_model,
    }


def collect_run(name: str, run_dir: Path) -> RunArtifacts:
    map_dir = final_map_directory(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("return_code") != 0:
        raise ValueError(f"Run manifest is not a completed successful run: {manifest_path}")
    paths = manifest.get("paths", {})
    orb_config = Path(paths.get("orb_config", "")).resolve()
    gaussian_config = Path(paths.get("gaussian_config", "")).resolve()
    trajectory_path = run_dir / "CameraTrajectory_TUM.txt"
    cameras_path = map_dir / "ply/cameras.json"
    ply_path = final_gaussian_ply(map_dir)
    required = [
        trajectory_path, cameras_path, ply_path, orb_config, gaussian_config,
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing run artifact: {path}")
    final_map_keyframes = load_final_map_cameras(cameras_path)
    return RunArtifacts(
        name=name,
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        trajectory_path=trajectory_path,
        orb_config=orb_config,
        gaussian_config=gaussian_config,
        cameras_path=cameras_path,
        ply_path=ply_path,
        keyframe_timestamps=np.asarray(
            [pose.timestamp for pose in final_map_keyframes], dtype=np.float64),
        trajectory=load_tum_trajectory(trajectory_path),
        final_map_keyframes=final_map_keyframes,
    )


def is_keyframe(run: RunArtifacts, timestamp: float, tolerance: float) -> bool:
    index = int(np.searchsorted(run.keyframe_timestamps, timestamp))
    return any(
        abs(float(run.keyframe_timestamps[candidate]) - timestamp) <= tolerance
        for candidate in (index - 1, index)
        if 0 <= candidate < len(run.keyframe_timestamps)
    )


def alignment_for_run(
    ground_truth: list[Pose],
    run: RunArtifacts,
    max_delta: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    ground_truth_points = []
    estimated_points = []
    for estimated in run.final_map_keyframes:
        reference = nearest_pose(ground_truth, estimated.timestamp, max_delta)
        if reference is None:
            continue
        ground_truth_points.append(reference.translation)
        estimated_points.append(estimated.translation)
    rotation, translation, rmse = rigid_alignment(
        np.asarray(ground_truth_points), np.asarray(estimated_points))
    return rotation, translation, rmse, len(ground_truth_points)


def write_pose_file(path: Path, rows: list[tuple[int, float, Pose]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "timestamp", "tx", "ty", "tz",
                         "qx", "qy", "qz", "qw"])
        for frame_id, timestamp, pose in rows:
            writer.writerow([
                frame_id, f"{timestamp:.9f}",
                *(f"{value:.9f}" for value in pose.translation),
                *(f"{value:.9f}" for value in pose.quaternion_xyzw),
            ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--view-camera-config", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True,
                        metavar="NAME=RUN_DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--mask-model", type=Path, required=True)
    parser.add_argument("--mask-config", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--pose-max-delta", type=float, default=0.03)
    parser.add_argument("--keyframe-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--groundtruth-pose-model",
        choices=("camera", "bonn_rgbd_sensor", "bonn_mocap"),
        default="camera",
        help=(
            "camera: ground truth already uses renderer camera axes; "
            "bonn_rgbd_sensor: convert each Bonn marker pose to the optical "
            "RGB-D sensor pose; bonn_mocap: deprecated fail-closed alias"
        ),
    )
    args = parser.parse_args()

    if args.warmup_frames < 0 or args.stride < 1 or args.max_frames < 1:
        parser.error("warmup must be non-negative; stride/max-frames must be positive")
    if args.pose_max_delta <= 0.0 or args.keyframe_tolerance < 0.0:
        parser.error("pose delta must be positive and keyframe tolerance non-negative")

    dataset_dir = args.dataset_dir.resolve()
    association_path = args.association.resolve()
    groundtruth_path = args.groundtruth.resolve()
    view_camera_config = args.view_camera_config.resolve()
    if not view_camera_config.is_file():
        parser.error(f"view camera config does not exist: {view_camera_config}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")
    run_specs = args.run
    if len({name for name, _ in run_specs}) != len(run_specs):
        parser.error("run names must be unique")

    mask_dir = args.mask_dir.resolve() if args.mask_dir else None
    mask_model = args.mask_model.resolve() if args.mask_model else None
    mask_config = args.mask_config.resolve() if args.mask_config else None
    if mask_dir is not None and (mask_model is None or mask_config is None):
        parser.error("--mask-dir requires --mask-model and --mask-config")
    if mask_dir is None and (mask_model is not None or mask_config is not None):
        parser.error("--mask-model/--mask-config require --mask-dir")
    for path, label in (
        (mask_model, "mask model"),
        (mask_config, "mask config"),
    ):
        if path is not None and (not path.is_file() or path.stat().st_size == 0):
            parser.error(f"{label} does not exist or is empty: {path}")

    runs = [collect_run(name, path) for name, path in run_specs]
    identity = validate_run_identities(
        runs, dataset_dir, association_path, groundtruth_path,
        view_camera_config)
    validate_groundtruth_pose_model(
        str(identity["sequence"]), args.groundtruth_pose_model)
    associations = load_associations(association_path)
    ground_truth = load_tum_trajectory(groundtruth_path)
    if args.groundtruth_pose_model == "bonn_rgbd_sensor":
        ground_truth = [
            transform_bonn_rgbd_sensor_pose(pose) for pose in ground_truth
        ]
    candidates: list[tuple[int, float, str]] = []
    last_selected = -args.stride
    for frame_id, timestamp, rgb_relative in associations:
        if frame_id <= args.warmup_frames or frame_id - last_selected < args.stride:
            continue
        if any(is_keyframe(run, timestamp, args.keyframe_tolerance) for run in runs):
            continue
        if nearest_pose(ground_truth, timestamp, args.pose_max_delta) is None:
            continue
        if any(nearest_pose(run.trajectory, timestamp, args.pose_max_delta) is None
               for run in runs):
            continue
        mask_path = mask_dir / f"{frame_id:06d}.png"
        if not mask_path.is_file() or mask_path.stat().st_size == 0:
            raise FileNotFoundError(
                "Selected held-out frame lacks its fixed evaluation mask: "
                f"{mask_path}")
        candidates.append((frame_id, timestamp, rgb_relative))
        last_selected = frame_id
        if len(candidates) >= args.max_frames:
            break
    if not candidates:
        raise RuntimeError("No common non-keyframe held-out observations were selected")

    ground_truth_dir = output_dir / "ground_truth"
    fixed_mask_dir = output_dir / "static_mask"
    poses_dir = output_dir / "poses"
    ground_truth_dir.mkdir(parents=True, exist_ok=False)
    poses_dir.mkdir()
    if mask_dir is not None:
        fixed_mask_dir.mkdir()

    alignment_records = {}
    alignments = {}
    for run in runs:
        rotation, translation, rmse, pairs = alignment_for_run(
            ground_truth, run, args.pose_max_delta)
        alignments[run.name] = (rotation, translation)
        alignment_records[run.name] = {
            "pairs": pairs,
            "translation_rmse_m": rmse,
            "rotation_map_from_gt": rotation.tolist(),
            "translation_map_from_gt_m": translation.tolist(),
            "pose_source": "final Gaussian-map keyframes from cameras.json",
            "heldout_views_used_for_alignment": False,
        }

    manifest_rows = []
    for frame_id, timestamp, rgb_relative in candidates:
        source = dataset_dir / rgb_relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing associated RGB image: {source}")
        target = ground_truth_dir / f"{frame_id:06d}.png"
        shutil.copy2(source, target)
        mask_target = None
        mask_target = fixed_mask_dir / f"{frame_id:06d}.png"
        shutil.copy2(
            mask_dir / f"{frame_id:06d}.png",
            mask_target)
        manifest_rows.append({
            "frame": frame_id,
            "timestamp": f"{timestamp:.9f}",
            "rgb_relative": rgb_relative,
            "ground_truth": str(target.relative_to(output_dir)),
            "ground_truth_sha256": sha256(target),
            "static_mask": (
                str(mask_target.relative_to(output_dir))
                if mask_target is not None else ""
            ),
            "static_mask_sha256": (
                sha256(mask_target) if mask_target is not None else ""
            ),
        })

    with (output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    for run in runs:
        online_rows = []
        gt_rows = []
        rotation, translation = alignments[run.name]
        for frame_id, timestamp, _ in candidates:
            online = nearest_pose(run.trajectory, timestamp, args.pose_max_delta)
            reference = nearest_pose(ground_truth, timestamp, args.pose_max_delta)
            if online is None or reference is None:
                raise RuntimeError(
                    "A prevalidated held-out pose association became unavailable")
            mapped_rotation = rotation @ quaternion_to_matrix(reference.quaternion_xyzw)
            gt_aligned = Pose(
                timestamp=timestamp,
                translation=rotation @ reference.translation + translation,
                quaternion_xyzw=matrix_to_quaternion(mapped_rotation),
            )
            online_rows.append((frame_id, timestamp, online))
            gt_rows.append((frame_id, timestamp, gt_aligned))
        write_pose_file(poses_dir / f"{run.name}_online.csv", online_rows)
        write_pose_file(poses_dir / f"{run.name}_gt_aligned.csv", gt_rows)

    protocol = {
        "protocol": (
            "Common final-map non-keyframe rendering at each run's online "
            "estimated poses and at ground-truth poses rigidly aligned into "
            "each run's map coordinate frame."
        ),
        "gt_aligned_scope": (
            "World-coordinate end-to-end reconstruction check; not an oracle "
            "mapping-only evaluation because online tracking poses constructed the map."
        ),
        "identity": identity,
        "selection": {
            "warmup_frames": args.warmup_frames,
            "stride": args.stride,
            "max_frames": args.max_frames,
            "selected_frames": len(candidates),
            "keyframe_tolerance_seconds": args.keyframe_tolerance,
            "pose_association_max_delta_seconds": args.pose_max_delta,
            "common_non_keyframe_across_all_runs": True,
            "fixed_static_mask": mask_dir is not None,
            "alignment_uses_mapping_views_only": True,
        },
        "dataset": {
            "directory": str(dataset_dir),
            "association": str(association_path),
            "groundtruth": str(groundtruth_path),
            "groundtruth_pose_model": args.groundtruth_pose_model,
            "groundtruth_calibration": (
                {
                    "source": BONN_DATASET_SOURCE,
                    "formula": (
                        "T_sensor(t) = T_ROS^-1 * T_marker(t) * T_ROS * T_m"
                    ),
                    "T_ROS": BONN_T_ROS.tolist(),
                    "T_m": BONN_T_M.tolist(),
                    "rotation_handling": (
                        "project the calibrated 3x3 block to the nearest SO(3) "
                        "rotation before quaternion conversion"
                    ),
                }
                if args.groundtruth_pose_model == "bonn_rgbd_sensor" else None
            ),
            "groundtruth_semantics": (
                {
                    "source": BONN_DATASET_SOURCE,
                    "trajectory_evaluation": (
                        "raw groundtruth.txt is used directly by the TUM/evo "
                        "trajectory evaluator"
                    ),
                    "rendering_pose": (
                        "each published marker pose is converted to the optical "
                        "RGB-D sensor pose using the fixed T_ROS/T_m calibration"
                    ),
                    "model_alignment_note": (
                        "T_ROS/T_m and T_g = T_ROS^-1 T_0 T_ROS T_m align a "
                        "sensor-frame reconstruction to the Leica scan using "
                        "only the first pose T_0; that separate model transform "
                        "is not applied to every map point here"
                    ),
                }
                if str(identity["sequence"]).startswith("bonn_") else {
                    "trajectory": "groundtruth.txt camera trajectory used directly",
                    "applied_transform": "identity",
                }
            ),
            "view_camera_config": str(view_camera_config),
            "view_camera_config_sha256": sha256(view_camera_config),
            "association_sha256": sha256(association_path),
            "groundtruth_sha256": sha256(groundtruth_path),
            "static_mask_provenance": (
                {
                    "directory": str(mask_dir),
                    "model": str(mask_model),
                    "model_sha256": sha256(mask_model),
                    "config": str(mask_config),
                    "config_sha256": sha256(mask_config),
                }
                if mask_dir is not None else None
            ),
        },
        "runs": {
            run.name: {
                "run_dir": str(run.run_dir),
                "manifest": str(run.manifest_path),
                "manifest_sha256": sha256(run.manifest_path),
                "trajectory": str(run.trajectory_path),
                "trajectory_sha256": sha256(run.trajectory_path),
                "trajectory_pose_source": "shutdown optimized CameraTrajectory_TUM",
                "orb_config": str(run.orb_config),
                "orb_config_sha256": sha256(run.orb_config),
                "gaussian_config": str(run.gaussian_config),
                "gaussian_config_sha256": sha256(run.gaussian_config),
                "cameras": str(run.cameras_path),
                "cameras_sha256": sha256(run.cameras_path),
                "ply": str(run.ply_path),
                "ply_sha256": sha256(run.ply_path),
            }
            for run in runs
        },
        "alignment": alignment_records,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "output_dir": str(output_dir),
        "selected_frames": len(candidates),
        "runs": [run.name for run in runs],
        "fixed_static_mask": mask_dir is not None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
