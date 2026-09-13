#!/usr/bin/env python3
"""Execute the frozen DYN-19 multi-seed common-view map-integrity protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_reproducible_benchmark as benchmark  # noqa: E402
import summarize_dyn19_map_integrity as summary  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
PLAN_CONTRACT = "dyn19-map-integrity-plan-v1"
CORE_SEQUENCES = summary.CORE_SEQUENCES
REQUIRED_SEEDS = summary.REQUIRED_SEEDS
REQUIRED_CONFIGS = summary.REQUIRED_CONFIGS
POSE_MODES = summary.POSE_MODES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sequence_view_camera_config(sequence: str) -> Path:
    if sequence.startswith("tum_"):
        return PROJECT / "cfg/view_only/camera_tum_fr3.yaml"
    if sequence.startswith("bonn_"):
        return PROJECT / "cfg/view_only/camera_bonn.yaml"
    raise ValueError(f"No view camera config registered for {sequence}")


def groundtruth_pose_model(sequence: str) -> str:
    return "bonn_rgbd_sensor" if sequence.startswith("bonn_") else "camera"


def result_index(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    payload = load_json(root / "all_results.json")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("DYN-19 tracking root has no result list")
    index = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("DYN-19 result list contains a non-object")
        identity = (
            result.get("config"),
            result.get("sequence"),
            result.get("seed"),
        )
        if identity in index:
            raise ValueError(f"Duplicate DYN-19 result identity: {identity}")
        index[identity] = result
    return index


def validate_tracking_root(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve()
    audit = benchmark.dyn19_phase_plan_contract(root)
    if not audit.get("valid") or audit.get("phase") != "main":
        raise ValueError(f"Tracking root does not pass DYN-19 main plan audit: {audit}")
    phase_report = load_json(root / "dyn19_phase_report.json")
    if phase_report.get("status") != "complete":
        raise ValueError("DYN-19 main tracking phase is not complete")
    overlap_path = root / "dyn19_recovered_support_overlap_aggregate.json"
    overlap = load_json(overlap_path)
    if (
        overlap.get("safety_status") != "pass" or
        overlap.get("status") not in {"pass", "vacuous"}
    ):
        raise ValueError(
            "DYN-19 Full recovered-support overlap aggregate does not have "
            "a complete zero-leak safety status")

    results = result_index(root)
    tasks = []
    expected = {
        (config, sequence, seed)
        for config in REQUIRED_CONFIGS
        for sequence in CORE_SEQUENCES
        for seed in REQUIRED_SEEDS
    }
    missing = sorted(identity for identity in expected if identity not in results)
    if missing:
        raise ValueError(f"Main result matrix lacks map-integrity inputs: {missing}")
    for sequence_index, sequence in enumerate(CORE_SEQUENCES):
        dataset_dir, orb_config, gaussian_config, association_name = (
            benchmark.SEQUENCES[sequence])
        association = dataset_dir / association_name
        groundtruth = dataset_dir / "groundtruth.txt"
        for seed_index, seed in enumerate(REQUIRED_SEEDS):
            run_dirs: dict[str, str] = {}
            run_manifest_sha256: dict[str, str] = {}
            run_ply_sha256: dict[str, str] = {}
            semantic_manifest: dict[str, Any] | None = None
            for config in REQUIRED_CONFIGS:
                result = results[(config, sequence, seed)]
                if result.get("status") != "complete":
                    raise ValueError(
                        f"Tracking input is not complete: {config}/{sequence}/{seed}")
                run_dir = Path(str(result.get("run_dir", ""))).resolve()
                manifest_path = run_dir / "manifest.json"
                if not manifest_path.is_file():
                    raise FileNotFoundError(
                        f"Tracking input has no manifest: {manifest_path}")
                manifest = load_json(manifest_path)
                if (
                    manifest.get("config") != config or
                    manifest.get("sequence") != sequence or
                    manifest.get("seed") != seed
                ):
                    raise ValueError(
                        f"Tracking manifest identity mismatch: {manifest_path}")
                run_dirs[config] = str(run_dir)
                run_manifest_sha256[config] = sha256(manifest_path)
                final_map_dirs = sorted(
                    path for path in run_dir.glob("*_shutdown")
                    if path.is_dir())
                if not final_map_dirs:
                    raise FileNotFoundError(
                        f"Tracking input has no final map: {run_dir}")
                if config == "dyn19_semantic":
                    semantic_manifest = manifest
                    static_mask_dir = run_dir / "static_masks"
                    if not static_mask_dir.is_dir():
                        raise FileNotFoundError(
                            f"Semantic run did not export static masks: {static_mask_dir}")
            if semantic_manifest is None:
                raise RuntimeError("Semantic run provenance was not collected")
            model_path = Path(
                str(semantic_manifest["paths"]["yolo_engine"])).resolve()
            mask_config = Path(
                str(semantic_manifest["paths"]["mask_config"])).resolve()
            if not model_path.is_file() or not mask_config.is_file():
                raise FileNotFoundError(
                    "Semantic model/config provenance is unavailable for "
                    f"{sequence}/seed_{seed:04d}")
            tasks.append({
                "sequence": sequence,
                "seed": seed,
                "dataset_dir": str(dataset_dir.resolve()),
                "association": str(association.resolve()),
                "groundtruth": str(groundtruth.resolve()),
                "orb_config": str(orb_config.resolve()),
                "gaussian_config": str(gaussian_config.resolve()),
                "view_camera_config": str(
                    sequence_view_camera_config(sequence).resolve()),
                "groundtruth_pose_model": groundtruth_pose_model(sequence),
                "run_dirs": run_dirs,
                "run_manifest_sha256": run_manifest_sha256,
                "semantic_static_mask_dir": str(static_mask_dir.resolve()),
                "semantic_model": str(model_path),
                "semantic_model_sha256": sha256(model_path),
                "semantic_mask_config": str(mask_config),
                "semantic_mask_config_sha256": sha256(mask_config),
                "gpu_pair_index": (
                    sequence_index * len(REQUIRED_SEEDS) + seed_index),
            })
    return {
        "root": str(root),
        "plan_audit": audit,
        "phase_report": str(root / "dyn19_phase_report.json"),
        "phase_report_sha256": sha256(root / "dyn19_phase_report.json"),
        "overlap_certificate": str(overlap_path),
        "overlap_certificate_sha256": sha256(overlap_path),
        "overlap_certificate_payload": overlap,
    }, tasks


def freeze_plan(
    output_root: Path,
    tracking: dict[str, Any],
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    for task in tasks:
        task["output_dir"] = str(
            output_root / task["sequence"] / f"seed_{task['seed']:04d}")
        task["gpu"] = args.gpus[task["gpu_pair_index"] % len(args.gpus)]
    payload = {
        "contract": PLAN_CONTRACT,
        "experiment_id": summary.DYN19_EXPERIMENT_ID,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tracking_root": tracking["root"],
        "tracking_main_plan_audit": tracking["plan_audit"],
        "tracking_main_phase_report": tracking["phase_report"],
        "tracking_main_phase_report_sha256": tracking["phase_report_sha256"],
        "tracking_main_overlap_certificate": {
            "path": tracking["overlap_certificate"],
            "sha256": tracking["overlap_certificate_sha256"],
            "status": tracking["overlap_certificate_payload"].get("status"),
            "safety_status": tracking["overlap_certificate_payload"].get(
                "safety_status"),
            "positive_recovery_evidence": (
                tracking["overlap_certificate_payload"].get(
                    "positive_recovery_evidence")),
        },
        "matrix": {
            "core_sequences": list(CORE_SEQUENCES),
            "seeds": list(REQUIRED_SEEDS),
            "configs": list(REQUIRED_CONFIGS),
            "pose_modes": list(POSE_MODES),
        },
        "parameters": {
            "warmup_frames": args.warmup_frames,
            "heldout_stride": args.heldout_stride,
            "max_frames": args.max_frames,
            "pose_max_delta": args.pose_max_delta,
            "keyframe_tolerance": args.keyframe_tolerance,
            "source_stride": args.source_stride,
            "pixel_stride": args.pixel_stride,
            "minimum_frame_gap": args.minimum_frame_gap,
            "occlusion_margin_m": args.occlusion_margin_m,
            "proxy_tau": args.proxy_tau,
            "proxy_tau_high": args.proxy_tau_high,
            "skip_lpips": args.skip_lpips,
        },
        "tasks": tasks,
    }
    plan_path = output_root / "dyn19_map_integrity_plan.json"
    atomic_json(plan_path, payload)
    digest = sha256(plan_path)
    (output_root / "dyn19_map_integrity_plan.sha256").write_text(
        f"{digest}  dyn19_map_integrity_plan.json\n")
    return payload


def run_checked(
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        handle.write(json.dumps({"command": command}, indent=2) + "\n\n")
        completed = subprocess.run(
            command,
            cwd=PROJECT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit {completed.returncode}; inspect {log_path}")


def execute_task(task: dict[str, Any], parameters: dict[str, Any], root: Path) -> None:
    cell_dir = Path(str(task["output_dir"]))
    if cell_dir.exists():
        raise FileExistsError(f"Refusing to reuse map-integrity cell: {cell_dir}")
    sequence = str(task["sequence"])
    seed = int(task["seed"])
    log_dir = root / "logs" / sequence / f"seed_{seed:04d}"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(task["gpu"])
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    python = sys.executable

    prepare_command = [
        python, str(PROJECT / "scripts/prepare_mapping_heldout.py"),
        "--dataset-dir", task["dataset_dir"],
        "--association", task["association"],
        "--groundtruth", task["groundtruth"],
        "--view-camera-config", task["view_camera_config"],
        "--output-dir", str(cell_dir),
        "--mask-dir", task["semantic_static_mask_dir"],
        "--mask-model", task["semantic_model"],
        "--mask-config", task["semantic_mask_config"],
        "--warmup-frames", str(parameters["warmup_frames"]),
        "--stride", str(parameters["heldout_stride"]),
        "--max-frames", str(parameters["max_frames"]),
        "--pose-max-delta", str(parameters["pose_max_delta"]),
        "--keyframe-tolerance", str(parameters["keyframe_tolerance"]),
        "--groundtruth-pose-model", task["groundtruth_pose_model"],
    ]
    for config in REQUIRED_CONFIGS:
        prepare_command.extend([
            "--run", f"{config}={task['run_dirs'][config]}"])
    run_checked(prepare_command, log_dir / "prepare.log", environment)

    proxy_dir = cell_dir / "occluded_background_proxy"
    proxy_command = [
        python, str(PROJECT / "scripts/evaluate_occluded_background_proxy.py"),
        "build",
        "--dataset-dir", task["dataset_dir"],
        "--association", task["association"],
        "--groundtruth", task["groundtruth"],
        "--camera-config", task["orb_config"],
        "--static-mask-dir", task["semantic_static_mask_dir"],
        "--heldout-dir", str(cell_dir),
        "--manifest", str(cell_dir / "manifest.csv"),
        "--output-dir", str(proxy_dir),
        "--groundtruth-pose-model", task["groundtruth_pose_model"],
        "--pose-max-delta", str(parameters["pose_max_delta"]),
        "--source-stride", str(parameters["source_stride"]),
        "--pixel-stride", str(parameters["pixel_stride"]),
        "--minimum-frame-gap", str(parameters["minimum_frame_gap"]),
        "--occlusion-margin-m", str(parameters["occlusion_margin_m"]),
    ]
    run_checked(proxy_command, log_dir / "proxy_build.log", environment)

    protocol = load_json(cell_dir / "protocol.json")
    view_model = protocol.get("identity", {}).get("view_camera_model", {})
    width = int(view_model.get("width", 0))
    height = int(view_model.get("height", 0))
    if width < 1 or height < 1:
        raise ValueError(f"Protocol lacks view dimensions: {cell_dir / 'protocol.json'}")
    for pose_mode in POSE_MODES:
        for config in REQUIRED_CONFIGS:
            run = protocol["runs"][config]
            render_dir = cell_dir / "renders" / pose_mode / config
            pose_file = cell_dir / "poses" / f"{config}_{pose_mode}.csv"
            render_command = [
                str(PROJECT / "bin/view_result"),
                str(run["gaussian_config"]),
                task["view_camera_config"],
                str(run["ply"]),
                "--pose-file", str(pose_file),
                "--output-dir", str(render_dir),
                "--width", str(width),
                "--height", str(height),
            ]
            run_checked(
                render_command,
                log_dir / f"render_{config}_{pose_mode}.log",
                environment,
            )
            metric_dir = cell_dir / "metrics" / pose_mode / config
            normal_command = [
                python, str(PROJECT / "scripts/evaluate_heldout_rendering.py"),
                str(cell_dir),
                "--manifest", str(cell_dir / "manifest.csv"),
                "--render-dir", str(render_dir),
                "--output-dir", str(metric_dir),
                "--protocol", (
                    "DYN-19 common held-out final-map rendering "
                    f"({config}, {pose_mode})"),
            ]
            if parameters["skip_lpips"]:
                normal_command.append("--skip-lpips")
            run_checked(
                normal_command,
                log_dir / f"metrics_{config}_{pose_mode}.log",
                environment,
            )
            proxy_metric_dir = cell_dir / "proxy_metrics" / pose_mode / config
            proxy_metric_command = [
                python,
                str(PROJECT / "scripts/evaluate_occluded_background_proxy.py"),
                "evaluate",
                "--proxy-dir", str(proxy_dir),
                "--manifest", str(cell_dir / "manifest.csv"),
                "--render-dir", str(render_dir),
                "--output-dir", str(proxy_metric_dir),
                "--tau", str(parameters["proxy_tau"]),
                "--tau-high", str(parameters["proxy_tau_high"]),
                "--variant", config,
                "--pose-mode", pose_mode,
            ]
            run_checked(
                proxy_metric_command,
                log_dir / f"proxy_metrics_{config}_{pose_mode}.log",
                environment,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--heldout-stride", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--pose-max-delta", type=float, default=0.03)
    parser.add_argument("--keyframe-tolerance", type=float, default=0.005)
    parser.add_argument("--source-stride", type=int, default=15)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--minimum-frame-gap", type=int, default=30)
    parser.add_argument("--occlusion-margin-m", type=float, default=0.05)
    parser.add_argument("--proxy-tau", type=float, default=30.0)
    parser.add_argument("--proxy-tau-high", type=float, default=60.0)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if (
        not args.gpus or len(args.gpus) != len(set(args.gpus)) or
        args.warmup_frames < 0 or args.heldout_stride < 1 or
        args.max_frames < 1 or args.pose_max_delta <= 0.0 or
        args.keyframe_tolerance < 0.0 or args.source_stride < 1 or
        args.pixel_stride < 1 or args.minimum_frame_gap < 1 or
        args.occlusion_margin_m <= 0.0 or args.proxy_tau < 0.0 or
        args.proxy_tau_high < args.proxy_tau
    ):
        parser.error("Invalid map-integrity protocol parameters")
    output_root = args.output_root.resolve()
    if output_root.exists():
        parser.error(f"output root already exists: {output_root}")
    tracking, tasks = validate_tracking_root(args.tracking_root)
    output_root.mkdir(parents=True)
    plan = freeze_plan(output_root, tracking, tasks, args)
    if args.dry_run:
        print(output_root)
        print(f"planned_cells={len(tasks)}")
        print(
            f"planned_render_metric_cells="
            f"{len(tasks) * len(REQUIRED_CONFIGS) * len(POSE_MODES)}")
        return 0

    for task in tasks:
        execute_task(task, plan["parameters"], output_root)
    summary_payload = summary.summarize(output_root)
    print(json.dumps({
        "output_root": str(output_root),
        "status": summary_payload["status"],
        "rows": len(summary_payload["rows"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
