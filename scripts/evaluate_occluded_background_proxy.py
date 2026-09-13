#!/usr/bin/env python3
"""Build and score an occluded-background rendering proxy.

The proxy is deliberately weaker than true ghost ground truth. It takes
semantic-static RGB-D pixels from other frames, reprojects them with dataset
ground-truth poses into target pixels that the frozen semantic mask marks
dynamic, and retains only samples whose reprojected depth lies behind the
target observation. The resulting source colour is a partial background
reference for those occluded target pixels.

This script has two subcommands:

* ``build`` creates frozen proxy/support images once for a common held-out
  manifest;
* ``evaluate`` compares one final-map render set to that frozen proxy.

The reported ``ghost_risk_proxy`` is not a count of true ghost geometry.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import prepare_mapping_heldout as mapping  # noqa: E402


PROXY_CONTRACT = "occluded-background-proxy-v1"
METRIC_CONTRACT = "occluded-background-proxy-metrics-v1"


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_factor: float


@dataclass(frozen=True)
class RgbdAssociation:
    frame: int
    rgb_timestamp: float
    rgb_relative: str
    depth_timestamp: float
    depth_relative: str


@dataclass(frozen=True)
class SourceSample:
    frame: int
    timestamp: float
    points_world: np.ndarray
    colors_bgr: np.ndarray
    rgb_path: Path
    depth_path: Path
    mask_path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def open_cv_yaml_value(path: Path, key: str) -> str:
    prefix = f"{key}:"
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line.startswith(prefix):
                continue
            value = line[len(prefix):].strip()
            if len(value) >= 2 and value[0] == value[-1]:
                if value[0] in {"'", '"'}:
                    value = value[1:-1]
            if value:
                return value
    raise ValueError(f"{path} lacks required key {key}")


def open_cv_yaml_first(path: Path, keys: tuple[str, ...]) -> str:
    for key in keys:
        try:
            return open_cv_yaml_value(path, key)
        except ValueError:
            continue
    raise ValueError(f"{path} lacks any of {list(keys)}")


def load_camera_model(path: Path) -> CameraModel:
    path = path.resolve()
    width = int(float(open_cv_yaml_first(path, ("Camera.width", "Camera.w"))))
    height = int(float(open_cv_yaml_first(path, ("Camera.height", "Camera.h"))))
    fx = float(open_cv_yaml_first(path, ("Camera1.fx", "Camera.fx")))
    fy = float(open_cv_yaml_first(path, ("Camera1.fy", "Camera.fy")))
    cx = float(open_cv_yaml_first(path, ("Camera1.cx", "Camera.cx")))
    cy = float(open_cv_yaml_first(path, ("Camera1.cy", "Camera.cy")))
    try:
        depth_factor = float(open_cv_yaml_value(path, "RGBD.DepthMapFactor"))
    except ValueError:
        depth_factor = 1.0
    values = (fx, fy, cx, cy, depth_factor)
    if (
        width < 1 or height < 1 or depth_factor <= 0.0 or
        not all(math.isfinite(value) for value in values) or
        fx <= 0.0 or fy <= 0.0
    ):
        raise ValueError(f"Invalid camera model in {path}")
    return CameraModel(width, height, fx, fy, cx, cy, depth_factor)


def load_rgbd_associations(path: Path) -> list[RgbdAssociation]:
    rows: list[RgbdAssociation] = []
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(
                    f"{path}:{line_number}: expected RGB-D association with "
                    "timestamp/path pairs")
            try:
                rows.append(RgbdAssociation(
                    frame=len(rows),
                    rgb_timestamp=float(fields[0]),
                    rgb_relative=fields[1],
                    depth_timestamp=float(fields[2]),
                    depth_relative=fields[3],
                ))
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid association timestamps") from exc
    if not rows:
        raise ValueError(f"No RGB-D associations found in {path}")
    return rows


def load_heldout_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Held-out manifest does not exist: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"frame", "timestamp", "ground_truth", "static_mask"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"Held-out manifest must contain {sorted(required)}: {path}")
    frames = [int(row["frame"]) for row in rows]
    if len(frames) != len(set(frames)):
        raise ValueError(f"Held-out manifest has duplicate frames: {path}")
    if any(not row["static_mask"] for row in rows):
        raise ValueError(f"Held-out manifest has an empty static mask: {path}")
    return rows


def pose_to_matrix(pose: mapping.Pose) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = mapping.quaternion_to_matrix(pose.quaternion_xyzw)
    result[:3, 3] = pose.translation
    return result


def load_groundtruth_poses(
    path: Path,
    pose_model: str,
) -> list[mapping.Pose]:
    poses = mapping.load_tum_trajectory(path)
    if pose_model == "bonn_rgbd_sensor":
        poses = [mapping.transform_bonn_rgbd_sensor_pose(pose) for pose in poses]
    elif pose_model != "camera":
        raise ValueError(f"Unsupported ground-truth pose model: {pose_model}")
    return poses


def frame_poses(
    associations: list[RgbdAssociation],
    groundtruth: list[mapping.Pose],
    max_delta: float,
) -> dict[int, mapping.Pose]:
    result = {}
    for association in associations:
        pose = mapping.nearest_pose(
            groundtruth, association.rgb_timestamp, max_delta)
        if pose is not None:
            result[association.frame] = pose
    return result


def read_rgbd(
    dataset_dir: Path,
    association: RgbdAssociation,
    camera: CameraModel,
) -> tuple[np.ndarray, np.ndarray]:
    rgb_path = dataset_dir / association.rgb_relative
    depth_path = dataset_dir / association.depth_relative
    color = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or raw_depth is None:
        raise FileNotFoundError(
            f"Cannot load RGB-D frame {association.frame}: "
            f"rgb={rgb_path}, depth={depth_path}")
    if color.shape[:2] != (camera.height, camera.width):
        raise ValueError(
            f"RGB frame {rgb_path} has shape {color.shape[:2]}, expected "
            f"{(camera.height, camera.width)}")
    if raw_depth.shape[:2] != (camera.height, camera.width):
        raise ValueError(
            f"Depth frame {depth_path} has shape {raw_depth.shape[:2]}, "
            f"expected {(camera.height, camera.width)}")
    if raw_depth.dtype == np.uint16:
        depth = raw_depth.astype(np.float32) / camera.depth_factor
    else:
        depth = raw_depth.astype(np.float32)
    return color, depth


def read_static_mask(path: Path, camera: CameraModel) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot load frozen static mask: {path}")
    if image.shape != (camera.height, camera.width):
        raise ValueError(
            f"Mask {path} has shape {image.shape}, expected "
            f"{(camera.height, camera.width)}")
    return image > 0


def source_world_points(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    static_mask: np.ndarray,
    pose: mapping.Pose,
    camera: CameraModel,
    pixel_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be positive")
    rows, columns = np.indices(depth.shape)
    sampled = (
        (rows % pixel_stride == 0) &
        (columns % pixel_stride == 0) &
        static_mask &
        np.isfinite(depth) &
        (depth > 0.0)
    )
    y, x = np.nonzero(sampled)
    if x.size == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )
    z = depth[y, x].astype(np.float64)
    camera_points = np.column_stack((
        (x.astype(np.float64) - camera.cx) * z / camera.fx,
        (y.astype(np.float64) - camera.cy) * z / camera.fy,
        z,
    ))
    twc = pose_to_matrix(pose)
    world_points = (
        camera_points @ twc[:3, :3].T + twc[:3, 3]
    ).astype(np.float32)
    return world_points, color_bgr[y, x].copy()


def update_proxy_zbuffer(
    proxy_depth: np.ndarray,
    proxy_bgr: np.ndarray,
    source_points_world: np.ndarray,
    source_colors_bgr: np.ndarray,
    target_twc: np.ndarray,
    target_depth: np.ndarray,
    target_dynamic: np.ndarray,
    camera: CameraModel,
    occlusion_margin_m: float,
) -> int:
    """Update a target proxy with one source using a nearest-depth z-buffer."""
    if source_points_world.size == 0:
        return 0
    tcw_rotation = target_twc[:3, :3].T
    tcw_translation = -tcw_rotation @ target_twc[:3, 3]
    target_points = (
        source_points_world.astype(np.float64) @ tcw_rotation.T +
        tcw_translation
    )
    z = target_points[:, 2]
    valid = np.isfinite(z) & (z > 0.0)
    if not np.any(valid):
        return 0
    target_points = target_points[valid]
    z = z[valid]
    colors = source_colors_bgr[valid]
    u = np.rint(
        camera.fx * target_points[:, 0] / z + camera.cx).astype(np.int64)
    v = np.rint(
        camera.fy * target_points[:, 1] / z + camera.cy).astype(np.int64)
    inside = (
        (u >= 0) & (u < camera.width) &
        (v >= 0) & (v < camera.height)
    )
    if not np.any(inside):
        return 0
    u = u[inside]
    v = v[inside]
    z = z[inside]
    colors = colors[inside]
    observed_depth = target_depth[v, u]
    target_is_dynamic = target_dynamic[v, u]
    occluded = (
        target_is_dynamic &
        np.isfinite(observed_depth) &
        (observed_depth > 0.0) &
        (observed_depth + occlusion_margin_m < z)
    )
    if not np.any(occluded):
        return 0
    u = u[occluded]
    v = v[occluded]
    z = z[occluded]
    colors = colors[occluded]
    flat = v * camera.width + u

    # Sort by target pixel then depth. The first item in each pixel group is
    # the closest reprojected static source point from this source frame.
    order = np.lexsort((z, flat))
    flat = flat[order]
    z = z[order]
    colors = colors[order]
    unique_start = np.r_[True, flat[1:] != flat[:-1]]
    flat = flat[unique_start]
    z = z[unique_start]
    colors = colors[unique_start]

    depth_view = proxy_depth.reshape(-1)
    replace = z < depth_view[flat]
    if not np.any(replace):
        return 0
    flat = flat[replace]
    depth_view[flat] = z[replace].astype(np.float32)
    proxy_bgr.reshape(-1, 3)[flat] = colors[replace]
    return int(flat.size)


def source_manifest_rows(
    dataset_dir: Path,
    associations: list[RgbdAssociation],
    poses: dict[int, mapping.Pose],
    static_mask_dir: Path,
    source_stride: int,
) -> list[dict[str, str]]:
    if source_stride < 1:
        raise ValueError("source_stride must be positive")
    rows = []
    for association in associations:
        if association.frame % source_stride != 0:
            continue
        if association.frame not in poses:
            continue
        rgb_path = dataset_dir / association.rgb_relative
        depth_path = dataset_dir / association.depth_relative
        mask_path = static_mask_dir / f"{association.frame:06d}.png"
        for label, path in (
            ("RGB", rgb_path), ("depth", depth_path), ("static mask", mask_path),
        ):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Selected proxy source lacks {label}: {path}")
        rows.append({
            "frame": str(association.frame),
            "timestamp": f"{association.rgb_timestamp:.9f}",
            "rgb_relative": association.rgb_relative,
            "depth_relative": association.depth_relative,
            "rgb_sha256": sha256(rgb_path),
            "depth_sha256": sha256(depth_path),
            "static_mask_sha256": sha256(mask_path),
        })
    if not rows:
        raise RuntimeError("No valid source RGB-D frames selected for proxy")
    return rows


def build_proxy(
    *,
    dataset_dir: Path,
    association_path: Path,
    groundtruth_path: Path,
    camera_config: Path,
    static_mask_dir: Path,
    heldout_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    groundtruth_pose_model: str,
    pose_max_delta: float = 0.03,
    source_stride: int = 15,
    pixel_stride: int = 2,
    minimum_frame_gap: int = 30,
    occlusion_margin_m: float = 0.05,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    association_path = association_path.resolve()
    groundtruth_path = groundtruth_path.resolve()
    camera_config = camera_config.resolve()
    static_mask_dir = static_mask_dir.resolve()
    heldout_dir = heldout_dir.resolve()
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse proxy output directory: {output_dir}")
    if (
        pose_max_delta <= 0.0 or source_stride < 1 or pixel_stride < 1 or
        minimum_frame_gap < 1 or occlusion_margin_m <= 0.0
    ):
        raise ValueError("Invalid proxy sampling or occlusion parameters")

    camera = load_camera_model(camera_config)
    associations = load_rgbd_associations(association_path)
    association_by_frame = {
        association.frame: association for association in associations
    }
    heldout_rows = load_heldout_manifest(manifest_path)
    groundtruth = load_groundtruth_poses(
        groundtruth_path, groundtruth_pose_model)
    poses = frame_poses(associations, groundtruth, pose_max_delta)
    source_rows = source_manifest_rows(
        dataset_dir, associations, poses, static_mask_dir, source_stride)

    source_samples = []
    for row in source_rows:
        frame = int(row["frame"])
        association = association_by_frame[frame]
        color, depth = read_rgbd(dataset_dir, association, camera)
        mask_path = static_mask_dir / f"{frame:06d}.png"
        static_mask = read_static_mask(mask_path, camera)
        points_world, colors_bgr = source_world_points(
            color, depth, static_mask, poses[frame], camera, pixel_stride)
        source_samples.append(SourceSample(
            frame=frame,
            timestamp=association.rgb_timestamp,
            points_world=points_world,
            colors_bgr=colors_bgr,
            rgb_path=dataset_dir / association.rgb_relative,
            depth_path=dataset_dir / association.depth_relative,
            mask_path=mask_path,
        ))
    if not source_samples:
        raise RuntimeError("No usable proxy source samples were prepared")

    output_dir.mkdir(parents=True)
    proxy_dir = output_dir / "proxy"
    support_dir = output_dir / "support"
    proxy_dir.mkdir()
    support_dir.mkdir()
    source_manifest_path = output_dir / "source_frames.csv"
    with source_manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    per_frame_rows = []
    for heldout_row in heldout_rows:
        frame = int(heldout_row["frame"])
        if frame not in association_by_frame or frame not in poses:
            raise RuntimeError(
                f"Held-out frame {frame} lacks an RGB-D association or GT pose")
        association = association_by_frame[frame]
        target_color, target_depth = read_rgbd(
            dataset_dir, association, camera)
        del target_color
        source_mask_path = static_mask_dir / f"{frame:06d}.png"
        heldout_mask_path = heldout_dir / heldout_row["static_mask"]
        if not source_mask_path.is_file():
            raise FileNotFoundError(
                f"Held-out frame {frame} lacks a frozen semantic mask: "
                f"{source_mask_path}")
        if not heldout_mask_path.is_file():
            raise FileNotFoundError(
                f"Held-out manifest static mask is missing: {heldout_mask_path}")
        expected_mask_hash = heldout_row.get("static_mask_sha256", "")
        source_mask_hash = sha256(source_mask_path)
        if expected_mask_hash and source_mask_hash != expected_mask_hash:
            raise ValueError(
                f"Held-out frame {frame} semantic mask does not match the "
                "frozen manifest")
        target_static = read_static_mask(source_mask_path, camera)
        target_dynamic = ~target_static
        proxy_depth = np.full(
            (camera.height, camera.width), np.inf, dtype=np.float32)
        proxy_bgr = np.zeros(
            (camera.height, camera.width, 3), dtype=np.uint8)
        target_twc = pose_to_matrix(poses[frame])
        source_updates = 0
        used_sources = 0
        for sample in source_samples:
            if abs(sample.frame - frame) < minimum_frame_gap:
                continue
            updates = update_proxy_zbuffer(
                proxy_depth,
                proxy_bgr,
                sample.points_world,
                sample.colors_bgr,
                target_twc,
                target_depth,
                target_dynamic,
                camera,
                occlusion_margin_m,
            )
            source_updates += updates
            used_sources += int(updates > 0)
        support = np.isfinite(proxy_depth)
        target_dynamic_pixels = int(np.count_nonzero(target_dynamic))
        support_pixels = int(np.count_nonzero(support))
        stem = f"{frame:06d}"
        proxy_path = proxy_dir / f"{stem}.png"
        support_path = support_dir / f"{stem}.png"
        if not cv2.imwrite(str(proxy_path), proxy_bgr):
            raise RuntimeError(f"Cannot write proxy image: {proxy_path}")
        if not cv2.imwrite(
            str(support_path), support.astype(np.uint8) * 255):
            raise RuntimeError(f"Cannot write proxy support: {support_path}")
        per_frame_rows.append({
            "frame": frame,
            "timestamp": float(heldout_row["timestamp"]),
            "target_dynamic_pixels": target_dynamic_pixels,
            "proxy_support_pixels": support_pixels,
            "proxy_coverage": (
                support_pixels / target_dynamic_pixels
                if target_dynamic_pixels else 0.0
            ),
            "source_zbuffer_updates": source_updates,
            "source_frames_with_updates": used_sources,
            "proxy": str(proxy_path.relative_to(output_dir)),
            "proxy_sha256": sha256(proxy_path),
            "support": str(support_path.relative_to(output_dir)),
            "support_sha256": sha256(support_path),
        })

    per_frame_path = output_dir / "occluded_background_proxy_per_frame.csv"
    with per_frame_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_frame_rows[0]))
        writer.writeheader()
        writer.writerows(per_frame_rows)
    total_dynamic_pixels = sum(
        int(row["target_dynamic_pixels"]) for row in per_frame_rows)
    total_support_pixels = sum(
        int(row["proxy_support_pixels"]) for row in per_frame_rows)
    output = {
        "contract": PROXY_CONTRACT,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": (
            "Partial occluded-background proxy: source semantic-static RGB-D "
            "points are reprojected with GT poses into target frozen-semantic "
            "dynamic pixels; target observed depth must be nearer than the "
            "reprojected source depth, and a nearest-depth z-buffer selects "
            "the source colour."
        ),
        "evidence_boundary": (
            "This is an occluded-background proxy, not true ghost geometry "
            "ground truth or a complete background reconstruction metric."
        ),
        "heldout_manifest": str(manifest_path),
        "heldout_manifest_sha256": sha256(manifest_path),
        "source_frames": str(source_manifest_path),
        "source_frames_sha256": sha256(source_manifest_path),
        "parameters": {
            "groundtruth_pose_model": groundtruth_pose_model,
            "pose_max_delta_seconds": pose_max_delta,
            "source_stride": source_stride,
            "pixel_stride": pixel_stride,
            "minimum_frame_gap": minimum_frame_gap,
            "occlusion_margin_m": occlusion_margin_m,
        },
        "camera": {
            "config": str(camera_config),
            "config_sha256": sha256(camera_config),
            "width": camera.width,
            "height": camera.height,
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "depth_factor": camera.depth_factor,
        },
        "dataset": {
            "directory": str(dataset_dir),
            "association": str(association_path),
            "association_sha256": sha256(association_path),
            "groundtruth": str(groundtruth_path),
            "groundtruth_sha256": sha256(groundtruth_path),
            "semantic_static_mask_directory": str(static_mask_dir),
        },
        "frames": per_frame_rows,
        "summary": {
            "heldout_frames": len(per_frame_rows),
            "source_frames": len(source_samples),
            "target_dynamic_pixels": total_dynamic_pixels,
            "proxy_support_pixels": total_support_pixels,
            "proxy_coverage": (
                total_support_pixels / total_dynamic_pixels
                if total_dynamic_pixels else 0.0
            ),
        },
    }
    manifest_output = output_dir / "occluded_background_proxy_manifest.json"
    manifest_output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def validate_proxy(
    proxy_dir: Path,
    heldout_manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    proxy_manifest_path = proxy_dir / "occluded_background_proxy_manifest.json"
    if not proxy_manifest_path.is_file():
        raise FileNotFoundError(f"Proxy manifest is missing: {proxy_manifest_path}")
    proxy_manifest = json.loads(proxy_manifest_path.read_text())
    if proxy_manifest.get("contract") != PROXY_CONTRACT:
        raise ValueError(f"Invalid proxy contract: {proxy_manifest_path}")
    if proxy_manifest.get("heldout_manifest_sha256") != sha256(
            heldout_manifest_path):
        raise ValueError("Proxy was built against a different held-out manifest")
    heldout_rows = load_heldout_manifest(heldout_manifest_path)
    proxy_frames = proxy_manifest.get("frames")
    if not isinstance(proxy_frames, list):
        raise ValueError("Proxy manifest has no frame list")
    proxy_by_frame = {
        int(row["frame"]): row for row in proxy_frames
        if isinstance(row, dict) and "frame" in row
    }
    heldout_ids = [int(row["frame"]) for row in heldout_rows]
    if len(proxy_by_frame) != len(proxy_frames) or set(proxy_by_frame) != set(heldout_ids):
        raise ValueError("Proxy frame IDs do not exactly match the held-out manifest")
    for frame in heldout_ids:
        row = proxy_by_frame[frame]
        for name, hash_name in (
            ("proxy", "proxy_sha256"),
            ("support", "support_sha256"),
        ):
            path = proxy_dir / str(row[name])
            if not path.is_file() or sha256(path) != row[hash_name]:
                raise ValueError(
                    f"Proxy {name} file does not match its manifest for frame {frame}")
    return proxy_manifest, heldout_rows


def evaluate_proxy(
    *,
    proxy_dir: Path,
    heldout_manifest_path: Path,
    render_dir: Path,
    output_dir: Path,
    tau: float,
    tau_high: float,
    variant: str,
    pose_mode: str,
) -> dict[str, Any]:
    proxy_dir = proxy_dir.resolve()
    heldout_manifest_path = heldout_manifest_path.resolve()
    render_dir = render_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse proxy metrics output directory: {output_dir}")
    if tau < 0.0 or tau_high < tau:
        raise ValueError("Require 0 <= tau <= tau_high")
    proxy_manifest, heldout_rows = validate_proxy(
        proxy_dir, heldout_manifest_path)
    proxy_by_frame = {
        int(row["frame"]): row for row in proxy_manifest["frames"]
    }
    expected_renders = {f"{int(row['frame']):06d}.png" for row in heldout_rows}
    observed_renders = {path.name for path in render_dir.glob("*.png")}
    missing = sorted(expected_renders - observed_renders)
    unexpected = sorted(observed_renders - expected_renders)
    if missing or unexpected:
        raise RuntimeError(
            "Render set does not exactly match the proxy manifest; "
            f"missing={missing}, unexpected={unexpected}")

    rows = []
    total_dynamic_pixels = 0
    total_support_pixels = 0
    total_color_error = 0.0
    total_complete = 0
    total_ghost_risk = 0
    for heldout_row in heldout_rows:
        frame = int(heldout_row["frame"])
        proxy_row = proxy_by_frame[frame]
        proxy = cv2.imread(
            str(proxy_dir / proxy_row["proxy"]), cv2.IMREAD_COLOR)
        support = cv2.imread(
            str(proxy_dir / proxy_row["support"]), cv2.IMREAD_GRAYSCALE)
        render = cv2.imread(
            str(render_dir / f"{frame:06d}.png"), cv2.IMREAD_COLOR)
        if proxy is None or support is None or render is None:
            raise RuntimeError(f"Unreadable proxy or render image for frame {frame}")
        if proxy.shape != render.shape or proxy.shape[:2] != support.shape:
            raise RuntimeError(
                f"Shape mismatch for frame {frame}: proxy={proxy.shape}, "
                f"support={support.shape}, render={render.shape}")
        support_mask = support > 0
        support_pixels = int(np.count_nonzero(support_mask))
        color_error = (
            np.abs(render.astype(np.float32) - proxy.astype(np.float32))
            .mean(axis=2)
        )
        support_errors = color_error[support_mask]
        mean_error = (
            float(support_errors.mean()) if support_errors.size else None)
        completeness = (
            float(np.mean(support_errors <= tau))
            if support_errors.size else None)
        ghost_risk = (
            float(np.mean(support_errors >= tau_high))
            if support_errors.size else None)
        dynamic_pixels = int(proxy_row["target_dynamic_pixels"])
        total_dynamic_pixels += dynamic_pixels
        total_support_pixels += support_pixels
        if support_errors.size:
            total_color_error += float(support_errors.sum())
            total_complete += int(np.count_nonzero(support_errors <= tau))
            total_ghost_risk += int(
                np.count_nonzero(support_errors >= tau_high))
        rows.append({
            "frame": frame,
            "target_dynamic_pixels": dynamic_pixels,
            "proxy_support_pixels": support_pixels,
            "proxy_coverage": (
                support_pixels / dynamic_pixels if dynamic_pixels else 0.0),
            "background_proxy_color_error": mean_error,
            "background_completeness_at_tau": completeness,
            "ghost_risk_proxy_at_tau_high": ghost_risk,
        })

    output_dir.mkdir(parents=True)
    per_frame_path = output_dir / "occluded_background_proxy_metrics_per_frame.csv"
    with per_frame_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "contract": METRIC_CONTRACT,
        "protocol": (
            "Final map render compared to a frozen partial "
            "occluded-background proxy; not a true ghost-count metric."
        ),
        "evidence_boundary": proxy_manifest["evidence_boundary"],
        "variant": variant,
        "pose_mode": pose_mode,
        "heldout_manifest": str(heldout_manifest_path),
        "heldout_manifest_sha256": sha256(heldout_manifest_path),
        "proxy_manifest": str(
            proxy_dir / "occluded_background_proxy_manifest.json"),
        "proxy_manifest_sha256": sha256(
            proxy_dir / "occluded_background_proxy_manifest.json"),
        "render_dir": str(render_dir),
        "frames": len(rows),
        "frame_ids": [int(row["frame"]) for row in rows],
        "tau": tau,
        "tau_high": tau_high,
        "target_dynamic_pixels": total_dynamic_pixels,
        "proxy_support_pixels": total_support_pixels,
        "proxy_coverage": (
            total_support_pixels / total_dynamic_pixels
            if total_dynamic_pixels else 0.0
        ),
        "frames_with_proxy_support": sum(
            row["proxy_support_pixels"] > 0 for row in rows),
        "background_proxy_color_error_mean": (
            total_color_error / total_support_pixels
            if total_support_pixels else None
        ),
        "background_completeness_at_tau": (
            total_complete / total_support_pixels
            if total_support_pixels else None
        ),
        "ghost_risk_proxy_at_tau_high": (
            total_ghost_risk / total_support_pixels
            if total_support_pixels else None
        ),
        "per_frame_background_proxy_color_error_mean": finite_mean([
            row["background_proxy_color_error"]
            for row in rows
            if row["background_proxy_color_error"] is not None
        ]),
        "per_frame_background_completeness_at_tau_mean": finite_mean([
            row["background_completeness_at_tau"]
            for row in rows
            if row["background_completeness_at_tau"] is not None
        ]),
        "per_frame_ghost_risk_proxy_at_tau_high_mean": finite_mean([
            row["ghost_risk_proxy_at_tau_high"]
            for row in rows
            if row["ghost_risk_proxy_at_tau_high"] is not None
        ]),
        "proxy_parameters": proxy_manifest["parameters"],
    }
    summary_path = output_dir / "occluded_background_proxy_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or score the DYN-19 occluded-background proxy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--dataset-dir", type=Path, required=True)
    build.add_argument("--association", type=Path, required=True)
    build.add_argument("--groundtruth", type=Path, required=True)
    build.add_argument("--camera-config", type=Path, required=True)
    build.add_argument("--static-mask-dir", type=Path, required=True)
    build.add_argument("--heldout-dir", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument(
        "--groundtruth-pose-model",
        choices=("camera", "bonn_rgbd_sensor"),
        required=True,
    )
    build.add_argument("--pose-max-delta", type=float, default=0.03)
    build.add_argument("--source-stride", type=int, default=15)
    build.add_argument("--pixel-stride", type=int, default=2)
    build.add_argument("--minimum-frame-gap", type=int, default=30)
    build.add_argument("--occlusion-margin-m", type=float, default=0.05)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--proxy-dir", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--render-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--tau", type=float, default=30.0)
    evaluate.add_argument("--tau-high", type=float, default=60.0)
    evaluate.add_argument("--variant", required=True)
    evaluate.add_argument("--pose-mode", choices=("online", "gt_aligned"),
                          required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "build":
        summary = build_proxy(
            dataset_dir=args.dataset_dir,
            association_path=args.association,
            groundtruth_path=args.groundtruth,
            camera_config=args.camera_config,
            static_mask_dir=args.static_mask_dir,
            heldout_dir=args.heldout_dir,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            groundtruth_pose_model=args.groundtruth_pose_model,
            pose_max_delta=args.pose_max_delta,
            source_stride=args.source_stride,
            pixel_stride=args.pixel_stride,
            minimum_frame_gap=args.minimum_frame_gap,
            occlusion_margin_m=args.occlusion_margin_m,
        )
    else:
        summary = evaluate_proxy(
            proxy_dir=args.proxy_dir,
            heldout_manifest_path=args.manifest,
            render_dir=args.render_dir,
            output_dir=args.output_dir,
            tau=args.tau,
            tau_high=args.tau_high,
            variant=args.variant,
            pose_mode=args.pose_mode,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
