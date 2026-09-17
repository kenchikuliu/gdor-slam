#!/usr/bin/env python3
"""Build the current DYN-19 Full Protocol-300 comparison package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(
    "/mnt/nas_datasets/slam-experiments/DynaGS-SLAM/"
    "dyn21_protocol300_current_20260917_run01"
)
HISTORICAL = Path("/home/slam/experiments/dypho_skill_full_rerun_20260729")
PROTOCOL = HISTORICAL / "protocol.yaml"
HISTORICAL_COMPARISON = (
    HISTORICAL / "figure_evidence_protocol300_v2/comparison_manifest.json"
)
HISTORICAL_TABLES = HISTORICAL / "metrics/protocol300/tables"
TRAJECTORY_HELPERS = HISTORICAL / "scripts/prepare_trajectories.py"
FIGURE_HELPERS = HISTORICAL / "scripts/prepare_figure_evidence.py"
VIEW_RESULT = REPO / "bin/view_result"
PROTOCOL_SHA = "4ecc0b1dd48d5462324888b1c9d94d8ef18c4d83ce4f057b2ed65a264fbf9bbe"
FRAME_END = 300
MAX_DELTA = 0.02
ALIGNMENT_SUPPORT = 20

SEQUENCES = (
    "tum_walking_xyz",
    "tum_walking_halfsphere",
    "bonn_person_tracking",
    "bonn_crowd3",
)
LABELS = {
    "tum_walking_xyz": "fr3/w/xyz",
    "tum_walking_halfsphere": "fr3/w/half",
    "bonn_person_tracking": "bonn/ps_track",
    "bonn_crowd3": "bonn/r3",
}
ROW_IDS = {
    "tum_walking_xyz": "fr3_w_xyz",
    "tum_walking_halfsphere": "fr3_w_half",
    "bonn_person_tracking": "bonn_ps_track",
    "bonn_crowd3": "bonn_r3",
}
COLUMNS = (
    ("input", "Input"),
    ("splatam", "SplaTAM"),
    ("photo_slam", "Photo-SLAM"),
    ("photo_slam_mask", "DynaGS semantic"),
    ("dyn19_full", "DYN-19 Full"),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


traj = load_module("dyn21_protocol300_trajectory", TRAJECTORY_HELPERS)
fig = load_module("dyn21_protocol300_figure", FIGURE_HELPERS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def key_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def final_map(run_dir: Path) -> tuple[Path, Path, Path]:
    maps = []
    for path in run_dir.glob("*_shutdown"):
        match = re.fullmatch(r"(\d+)_shutdown", path.name)
        if match:
            maps.append((int(match.group(1)), path))
    if not maps:
        raise FileNotFoundError(f"no final map under {run_dir}")
    map_dir = max(maps)[1]
    plys = []
    for path in (map_dir / "ply/point_cloud").glob("iteration_*/point_cloud.ply"):
        match = re.fullmatch(r"iteration_(\d+)", path.parent.name)
        if match:
            plys.append((int(match.group(1)), path))
    if not plys:
        raise FileNotFoundError(f"no final PLY under {map_dir}")
    cameras = map_dir / "ply/cameras.json"
    record(cameras)
    return map_dir, max(plys)[1], cameras


def bonn_ground_truth(dataset: Path):
    source = traj.read_tum_trajectory(dataset / "groundtruth.txt", duplicate_policy="first")
    poses = source.poses_c2w.copy()
    for index, (timestamp, matrix) in enumerate(zip(source.timestamps, source.poses_c2w)):
        pose = fig.Pose(
            float(timestamp),
            matrix[:3, 3],
            traj.matrix_to_quaternion_xyzw(matrix[:3, :3]),
        )
        converted = fig.transform_bonn_sensor_pose(pose)
        poses[index, :3, :3] = fig.quaternion_to_matrix(converted.quaternion_xyzw)
        poses[index, :3, 3] = converted.translation
    return traj.Trajectory(source.timestamps.copy(), poses)


def tracking_metrics(sequence: str, run_dir: Path) -> dict[str, Any]:
    dataset = traj.SEQUENCE_DATASETS[sequence]
    estimate_path = run_dir / "CameraTrajectory_TUM.txt"
    estimate = traj.read_tum_trajectory(estimate_path)
    ground_truth = (
        bonn_ground_truth(dataset)
        if sequence.startswith("bonn_")
        else traj.read_tum_trajectory(dataset / "groundtruth.txt", duplicate_policy="first")
    )
    estimate_indices, reference_indices, differences = traj.associate_timestamps(
        estimate.timestamps, ground_truth.timestamps, max_difference=MAX_DELTA
    )
    if len(estimate_indices) < 3:
        raise RuntimeError(f"insufficient trajectory support for {sequence}")
    rotation, translation = traj.umeyama_se3(
        estimate.positions[estimate_indices], ground_truth.positions[reference_indices]
    )
    aligned = (rotation @ estimate.positions[estimate_indices].T).T + translation
    errors = np.linalg.norm(aligned - ground_truth.positions[reference_indices], axis=1)
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "ate_mean_m": float(errors.mean()),
        "ate_median_m": float(np.median(errors)),
        "ate_max_m": float(errors.max()),
        "estimate_pose_count": int(estimate.timestamps.size),
        "association_count": int(estimate_indices.size),
        "association_max_delta_seconds": MAX_DELTA,
        "observed_max_delta_seconds": float(differences.max()),
        "alignment": "fixed-scale Umeyama SE(3)",
        "scale": 1.0,
        "trajectory": record(estimate_path),
        "ground_truth": record(dataset / "groundtruth.txt"),
    }


def alignment_excluding_selected(
    gt_poses: list[Any], estimates: tuple[Any, ...], timestamp: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pairs = []
    for estimated in estimates:
        if abs(estimated.timestamp - timestamp) <= MAX_DELTA:
            continue
        reference = fig.nearest_pose(gt_poses, estimated.timestamp, MAX_DELTA)
        if reference is not None:
            pairs.append((abs(estimated.timestamp - timestamp), reference, estimated))
    pairs.sort(key=lambda item: item[0])
    pairs = pairs[:ALIGNMENT_SUPPORT]
    if len(pairs) < 3:
        raise RuntimeError("fewer than three non-selected alignment poses")
    source = np.asarray([item[1].translation for item in pairs])
    target = np.asarray([item[2].translation for item in pairs])
    relative_rotations = [
        fig.quaternion_to_matrix(item[2].quaternion_xyzw)
        @ fig.quaternion_to_matrix(item[1].quaternion_xyzw).T
        for item in pairs
    ]
    left, _, right_t = np.linalg.svd(np.sum(relative_rotations, axis=0))
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    translation = np.mean(target - (rotation @ source.T).T, axis=0)
    residual = (rotation @ source.T).T + translation - target
    return rotation, translation, {
        "alignment": "fixed-scale SE(3)",
        "scale": 1.0,
        "selected_pose_excluded": True,
        "support_pose_count": len(pairs),
        "support_pose_limit": ALIGNMENT_SUPPORT,
        "translation_rmse_m": float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
        "rotation_map_from_gt": rotation.tolist(),
        "translation_map_from_gt_m": translation.tolist(),
    }


def render_current(
    sequence: str,
    run_dir: Path,
    output_dir: Path,
    frame: int,
    timestamp: float,
    gpu: str,
) -> tuple[Path, dict[str, Any]]:
    dataset = fig.DATASETS[sequence]
    gt_poses = fig.load_tum_poses(dataset / "groundtruth.txt")
    gt_model = "camera"
    if sequence.startswith("bonn_"):
        gt_poses = [fig.transform_bonn_sensor_pose(pose) for pose in gt_poses]
        gt_model = "bonn_rgbd_sensor"
    selected = fig.nearest_pose(gt_poses, timestamp, MAX_DELTA)
    if selected is None:
        raise RuntimeError(f"no GT pose for {sequence} at {timestamp:.9f}")
    map_dir, ply, cameras = final_map(run_dir)
    estimates = fig.load_camera_poses(cameras)
    rotation, translation, alignment = alignment_excluding_selected(
        gt_poses, estimates, timestamp
    )
    mapped_rotation = rotation @ fig.quaternion_to_matrix(selected.quaternion_xyzw)
    mapped = fig.Pose(
        timestamp,
        rotation @ selected.translation + translation,
        fig.matrix_to_quaternion(mapped_rotation),
    )
    output_dir.mkdir(parents=True)
    pose_file = output_dir / "dyn19_full_gt_aligned.csv"
    fig.write_pose_csv(pose_file, frame, timestamp, mapped)
    gaussian_config = REPO / (
        "cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml"
        if sequence.startswith("tum_")
        else "cfg/gaussian_mapper/RGB-D/Bonn/bonn_rgbd.yaml"
    )
    camera_config = REPO / (
        "cfg/view_only/camera_tum_fr3.yaml"
        if sequence.startswith("tum_")
        else "cfg/view_only/camera_bonn.yaml"
    )
    render_dir = output_dir / "render"
    command = [
        str(VIEW_RESULT),
        str(gaussian_config),
        str(camera_config),
        str(ply),
        "--pose-file",
        str(pose_file),
        "--output-dir",
        str(render_dir),
        "--width",
        "640",
        "--height",
        "480",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    completed = subprocess.run(
        command, cwd=REPO, env=environment, text=True, capture_output=True
    )
    (output_dir / "render.stdout.log").write_text(completed.stdout)
    (output_dir / "render.stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"view_result failed for {sequence}: {completed.stderr[-2000:]}"
        )
    render_path = render_dir / f"{frame:06d}.png"
    image = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3) or float(image.std()) < 1.0:
        raise RuntimeError(f"invalid current render: {render_path}")
    alignment.update(
        {
            "ground_truth_pose_model": gt_model,
            "pose_file": record(pose_file),
            "map_dir": str(map_dir),
            "ply": record(ply),
            "cameras": record(cameras),
            "gaussian_config": record(gaussian_config),
            "camera_config": record(camera_config),
            "command": command,
        }
    )
    return render_path, alignment


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean(np.square(reference.astype(np.float64) - estimate.astype(np.float64))))
    return math.inf if mse == 0.0 else 10.0 * math.log10((255.0 * 255.0) / mse)


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    values = []
    for channel in range(3):
        first, second = reference[:, :, channel], estimate[:, :, channel]
        mean_first = cv2.GaussianBlur(first, (11, 11), 1.5)
        mean_second = cv2.GaussianBlur(second, (11, 11), 1.5)
        variance_first = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mean_first**2
        variance_second = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mean_second**2
        covariance = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mean_first * mean_second
        score = ((2 * mean_first * mean_second + c1) * (2 * covariance + c2)) / (
            (mean_first**2 + mean_second**2 + c1)
            * (variance_first + variance_second + c2)
        )
        values.append(float(score[5:-5, 5:-5].mean()))
    return float(np.mean(values))


def elapsed_seconds(path: Path) -> float:
    pattern = re.compile(r"Elapsed \(wall clock\) time .*: ([0-9:.]+)")
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        fields = match.group(1).split(":")
        if len(fields) == 2:
            return int(fields[0]) * 60.0 + float(fields[1])
        if len(fields) == 3:
            return int(fields[0]) * 3600.0 + int(fields[1]) * 60.0 + float(fields[2])
    raise RuntimeError(f"no elapsed wall time in {path}")


def runtime_metrics(root: Path, sequence: str) -> dict[str, Any]:
    status_path = root / "logs" / f"{sequence}.status"
    time_path = root / "logs" / f"{sequence}.time"
    status = key_values(status_path)
    if status.get("exit_code") != "0" or status.get("frame_end_exclusive") != "300":
        raise RuntimeError(f"incomplete status: {status_path}")
    epoch = (int(status["end_epoch_ns"]) - int(status["start_epoch_ns"])) / 1e9
    time_value = elapsed_seconds(time_path)
    if abs(epoch - time_value) > 1.0:
        raise RuntimeError(f"timing disagreement for {sequence}: {epoch} vs {time_value}")
    return {
        "wall_time_seconds": epoch,
        "usr_bin_time_seconds": time_value,
        "throughput_frames_per_second": FRAME_END / epoch,
        "status": record(status_path),
        "usr_bin_time": record(time_path),
    }


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def historical_rows(name: str) -> tuple[list[str], list[list[str]]]:
    with (HISTORICAL_TABLES / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0], [row for row in rows[1:] if not row[0].startswith("Ours")]


def load_font(size: int):
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def contact_sheet(rows: list[dict[str, Any]], output: Path) -> None:
    tile_width, tile_height = 640, 480
    header_height, label_height = 58, 34
    image = Image.new(
        "RGB",
        (tile_width * len(COLUMNS), header_height + len(rows) * (tile_height + label_height)),
        (248, 248, 246),
    )
    draw = ImageDraw.Draw(image)
    header_font, row_font = load_font(23), load_font(18)

    def center(box: tuple[int, int, int, int], text: str, font: Any) -> None:
        bounds = draw.textbbox((0, 0), text, font=font)
        x = box[0] + (box[2] - box[0] - (bounds[2] - bounds[0])) // 2
        y = box[1] + (box[3] - box[1] - (bounds[3] - bounds[1])) // 2
        draw.text((x, y), text, font=font, fill=(25, 27, 30))

    for index, (_, label) in enumerate(COLUMNS):
        center((index * tile_width, 0, (index + 1) * tile_width, header_height), label, header_font)
    for row_index, row in enumerate(rows):
        label_y = header_height + row_index * (tile_height + label_height)
        center(
            (0, label_y, tile_width * len(COLUMNS), label_y + label_height),
            f"{LABELS[row['sequence']]} | frame {row['frame']} | t={row['timestamp']:.6f}",
            row_font,
        )
        image_y = label_y + label_height
        for column_index, (column, _) in enumerate(COLUMNS):
            panel = Image.open(row["panels"][column]["path"]).convert("RGB")
            if panel.size != (tile_width, tile_height):
                raise RuntimeError(f"unexpected panel size: {panel.size}")
            image.paste(panel, (column_index * tile_width, image_y))
    image.save(output, format="PNG", compress_level=6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "comparison"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if sha256(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("frozen upstream protocol hash mismatch")
    campaign = key_values(root / "campaign.status")
    if campaign.get("gpu0_exit_code") != "0" or campaign.get("gpu1_exit_code") != "0":
        raise RuntimeError("DYN-21 campaign is incomplete")
    plan = yaml.safe_load((root / "frozen_plan.yaml").read_text())
    protocol = yaml.safe_load(PROTOCOL.read_text())
    historical = json.loads(HISTORICAL_COMPARISON.read_text())
    historical_rows_by_sequence = {row["sequence"]: row for row in historical["rows"]}
    output.mkdir(parents=True)

    tracking, mapping, timing, figure_rows = {}, {}, {}, []
    for sequence in SEQUENCES:
        run_dir = root / "dyn19_full" / sequence / "seed_0000"
        summary = json.loads((run_dir / "run_summary.json").read_text())
        if summary.get("input_frames") != 300 or summary.get("processed_frames") != 300:
            raise RuntimeError(f"bad frame count for {sequence}")
        tracking[sequence] = tracking_metrics(sequence, run_dir)
        timing[sequence] = runtime_metrics(root, sequence)

        row_protocol = protocol["fig4"]["rows"][sequence]
        frame = int(row_protocol["render_frame"])
        timestamp = float(row_protocol["timestamp"])
        row_dir = output / "fig4_rows" / ROW_IDS[sequence]
        panels = {}
        source_row = historical_rows_by_sequence[sequence]
        for method in ("input", "splatam", "photo_slam", "photo_slam_mask"):
            source = Path(source_row["panels"][method]["path"])
            target = row_dir / method / f"{frame:06d}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            panels[method] = record(target)
            panels[method]["source"] = record(source)

        render_path, alignment = render_current(
            sequence, run_dir, row_dir / "dyn19_full", frame, timestamp, args.gpu
        )
        panels["dyn19_full"] = record(render_path)
        reference = cv2.imread(panels["input"]["path"], cv2.IMREAD_COLOR)
        estimate = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
        mapping[sequence] = {
            "psnr_db": psnr(reference, estimate),
            "ssim": ssim(reference, estimate),
            "frame": frame,
            "timestamp": timestamp,
            "alignment": alignment,
        }
        figure_rows.append(
            {
                "sequence": sequence,
                "frame": frame,
                "timestamp": timestamp,
                "panels": panels,
                "alignment": alignment,
            }
        )

    tables = output / "tables"
    header, rows = historical_rows("table1.csv")
    ate_cm = [tracking[sequence]["ate_rmse_m"] * 100.0 for sequence in SEQUENCES]
    current = ["DYN-19 Full (current)", "300"]
    for sequence, value in zip(SEQUENCES, ate_cm):
        current.extend([f"{value:.3f}", str(tracking[sequence]["association_count"])])
    current.extend(
        [
            f"{np.mean(ate_cm):.3f}",
            "1 (seed 0); no variance estimate",
            "verified current Protocol-300 run; fixed 0.02 s unique association",
        ]
    )
    write_csv(tables / "table1_tracking.csv", header, rows + [current])

    header, rows = historical_rows("table2.csv")
    current = ["DYN-19 Full (current)", "300"]
    for sequence in SEQUENCES:
        current.extend([f"{mapping[sequence]['psnr_db']:.3f}", f"{mapping[sequence]['ssim']:.5f}"])
    current.extend(
        [
            f"{np.mean([mapping[s]['psnr_db'] for s in SEQUENCES]):.3f}",
            f"{np.mean([mapping[s]['ssim'] for s in SEQUENCES]):.5f}",
            "diagnostic only; full frame includes dynamic foreground; not a static-scene superiority claim",
        ]
    )
    write_csv(tables / "table2_mapping.csv", header, rows + [current])

    header, rows = historical_rows("table3.csv")
    times = [timing[sequence]["wall_time_seconds"] for sequence in SEQUENCES]
    current = ["DYN-19 Full (current)", "300", *[f"{value:.3f}" for value in times]]
    current.extend(
        [
            f"{sum(times):.3f}",
            f"{(300.0 * len(SEQUENCES)) / sum(times):.5f}",
            "status epoch delta; /usr/bin/time cross-check",
            "verified current Protocol-300 run",
        ]
    )
    write_csv(tables / "table3_runtime.csv", header, rows + [current])

    sheet = output / "fig4_current_comparison_contact_sheet.png"
    contact_sheet(figure_rows, sheet)
    metrics = {
        "contract": "dyn21-protocol300-current-comparison-v1",
        "status": "complete",
        "method": "dyn19_full",
        "seed": 0,
        "frame_start": 0,
        "frame_end_exclusive": 300,
        "tracking": tracking,
        "diagnostic_full_frame_mapping": mapping,
        "runtime": timing,
    }
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    manifest = {
        "contract": "dyn21-protocol300-current-evidence-v1",
        "status": "complete",
        "frozen_plan": record(root / "frozen_plan.yaml"),
        "upstream_protocol": record(PROTOCOL),
        "historical_comparison": record(HISTORICAL_COMPARISON),
        "historical_tables": {
            name: record(HISTORICAL_TABLES / name)
            for name in ("table1.csv", "table2.csv", "table3.csv")
        },
        "historical_failed_boundary_ours_excluded_from_current_tables": True,
        "comparison_columns": [column for column, _ in COLUMNS],
        "figure_rows": figure_rows,
        "panel_count": len(figure_rows) * len(COLUMNS),
        "metrics": record(metrics_path),
        "tables": {
            path.name: record(path)
            for path in sorted(tables.glob("*.csv"))
        },
        "contact_sheet": record(sheet),
        "claim_boundary": plan["claim_boundary"],
        "implementation": record(Path(__file__)),
    }
    manifest_path = output / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksum_paths = [metrics_path, manifest_path, sheet, *sorted(tables.glob("*.csv"))]
    (output / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in checksum_paths)
    )
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "tables": [str(path) for path in sorted(tables.glob("*.csv"))],
        "contact_sheet": str(sheet),
        "manifest": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
