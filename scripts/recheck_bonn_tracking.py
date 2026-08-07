#!/usr/bin/env python3
"""Re-evaluate saved Bonn trajectories against published groundtruth.txt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
from pathlib import Path


TRAJECTORIES = {
    "all_input_frames": "CameraTrajectory_AllFrames_TUM.txt",
    "valid_optimized_frames": "CameraTrajectory_TUM.txt",
}
METRICS = {
    "ate_rmse_m": ("evo_ape", "trans_part"),
    "rpe_translation_rmse_m": ("evo_rpe", "trans_part"),
    "rpe_rotation_rmse_deg": ("evo_rpe", "angle_deg"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_rmse(output: str) -> float:
    match = re.search(r"^\s*rmse\s+([-+0-9.eE]+)\s*$", output, re.MULTILINE)
    if not match:
        raise ValueError("evo output did not contain an RMSE row")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError("evo returned a non-finite RMSE")
    return value


def evo_command(
    metric: str,
    ground_truth: Path,
    trajectory: Path,
    archive: Path,
) -> list[str]:
    executable, relation = METRICS[metric]
    command = [
        executable,
        "tum",
        str(ground_truth),
        str(trajectory),
        "-a",
        "-r",
        relation,
    ]
    if executable == "evo_rpe":
        command += ["-d", "1", "-u", "f"]
    command += ["--no_warnings", "--save_results", str(archive)]
    return command


def original_metric(metrics: dict, trajectory_label: str, metric: str) -> float:
    key = (
        metric
        if trajectory_label == "all_input_frames"
        else f"valid_optimized_{metric}"
    )
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing original metric: {key}")
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=18)
    parser.add_argument("--tolerance", type=float, default=5e-7)
    args = parser.parse_args()

    results_path = args.results_json.resolve()
    payload = json.loads(results_path.read_text())
    results = [
        row
        for row in payload.get("results", [])
        if str(row.get("sequence", "")).startswith("bonn_")
    ]
    if len(results) != args.expected_runs:
        raise ValueError(
            f"Expected {args.expected_runs} Bonn runs, found {len(results)}"
        )

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    ground_truth_identities = set()
    for result in sorted(
        results,
        key=lambda row: (str(row["config"]), int(row["seed"])),
    ):
        if result.get("status") != "complete":
            raise ValueError(f"Cannot recheck incomplete run: {result}")
        run_dir = Path(result["run_dir"]).resolve()
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        ground_truth = Path(manifest["paths"]["ground_truth"]).resolve()
        recorded_hash = manifest["hashes"]["ground_truth_sha256"]
        actual_hash = sha256(ground_truth)
        if actual_hash != recorded_hash:
            raise ValueError(f"Ground-truth hash mismatch for {run_dir}")
        ground_truth_identities.add((str(ground_truth), actual_hash))

        for trajectory_label, trajectory_name in TRAJECTORIES.items():
            trajectory = run_dir / trajectory_name
            if not trajectory.is_file():
                raise FileNotFoundError(trajectory)
            artifact_dir = (
                output_root
                / str(result["config"])
                / f"seed_{int(result['seed']):04d}"
                / trajectory_label
            )
            artifact_dir.mkdir(parents=True)
            for metric in METRICS:
                archive = artifact_dir / f"{metric}.zip"
                command = evo_command(metric, ground_truth, trajectory, archive)
                completed = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                log = artifact_dir / f"{metric}.txt"
                log.write_text(completed.stdout)
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{metric} failed for {run_dir}: {completed.stdout}"
                    )
                recomputed = parse_rmse(completed.stdout)
                original = original_metric(
                    result["metrics"], trajectory_label, metric
                )
                difference = recomputed - original
                rows.append({
                    "sequence": result["sequence"],
                    "variant": result["config"],
                    "seed": result["seed"],
                    "trajectory_scope": trajectory_label,
                    "metric": metric,
                    "original": f"{original:.9f}",
                    "recomputed": f"{recomputed:.9f}",
                    "difference": f"{difference:.9f}",
                    "within_tolerance": int(abs(difference) <= args.tolerance),
                    "ground_truth": str(ground_truth),
                    "ground_truth_sha256": actual_hash,
                    "trajectory": str(trajectory),
                    "trajectory_sha256": sha256(trajectory),
                    "evo_archive": str(archive),
                    "evo_archive_sha256": sha256(archive),
                    "evo_log": str(log),
                    "evo_log_sha256": sha256(log),
                })

    if len(ground_truth_identities) != 1:
        raise ValueError("Bonn runs do not share one ground-truth identity")
    passed = all(row["within_tolerance"] for row in rows)
    audit_csv = args.audit_csv.resolve()
    audit_json = args.audit_json.resolve()
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    version = importlib.metadata.version("evo")
    ground_truth, ground_truth_hash = next(iter(ground_truth_identities))
    summary = {
        "protocol": "bonn-saved-trajectory-offline-recheck-v1",
        "status": "pass" if passed else "fail",
        "tracking_jobs_rerun": 0,
        "saved_runs_checked": len(results),
        "trajectory_scopes": list(TRAJECTORIES),
        "metric_comparisons": len(rows),
        "tolerance": args.tolerance,
        "all_within_tolerance": passed,
        "ground_truth_semantics": (
            "Published Bonn groundtruth.txt is evaluated directly under the "
            "official TUM-format trajectory protocol; the rendering-only "
            "T_ROS/T_m conversion is not applied to evo."
        ),
        "ground_truth": ground_truth,
        "ground_truth_sha256": ground_truth_hash,
        "results_json": str(results_path),
        "results_json_sha256": sha256(results_path),
        "evo_version": version,
        "artifact_root": str(output_root),
        "audit_csv": str(audit_csv),
        "audit_csv_sha256": sha256(audit_csv),
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
