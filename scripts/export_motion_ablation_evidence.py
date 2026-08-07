#!/usr/bin/env python3
"""Export auditable per-seed tracking and mapping-seed selection evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path


REQUIRED_VARIANTS = (
    "semantic",
    "semantic_motion_init_rgbd",
    "semantic_motion_ungated",
    "semantic_motion_rgbd",
    "semantic_motion_rgbd_shadow",
    "semantic_motion_rgbd_shuffled",
)
DISPLAY_NAMES = {
    "semantic": "Semantic",
    "semantic_motion_init_rgbd": "Init-only",
    "semantic_motion_ungated": "Ungated",
    "semantic_motion_rgbd": "Full",
    "semantic_motion_rgbd_shadow": "Shadow",
    "semantic_motion_rgbd_shuffled": "Lag-30",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_results(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"{path} must contain a non-empty results array")
    identities = set()
    for result in results:
        identity = (result.get("sequence"), result.get("config"), result.get("seed"))
        if identity in identities:
            raise ValueError(f"Duplicate run identity: {identity}")
        identities.add(identity)
        if result.get("status") != "complete":
            raise ValueError(f"Incomplete run in evidence input: {identity}")
        if result.get("config") not in REQUIRED_VARIANTS:
            raise ValueError(f"Unexpected ablation variant: {result.get('config')}")
    return results


def select_median_seed(rows: list[dict]) -> tuple[int, float, dict[int, int]]:
    if not rows or len(rows) % 2 == 0:
        raise ValueError("Median-seed selection requires an odd non-empty seed set")
    ranked = sorted(
        rows, key=lambda row: (float(row["metrics"]["ate_rmse_m"]), int(row["seed"]))
    )
    for row in ranked:
        ate = float(row["metrics"]["ate_rmse_m"])
        if not math.isfinite(ate):
            raise ValueError("Semantic ATE must be finite")
    selected = ranked[len(ranked) // 2]
    ranks = {int(row["seed"]): index + 1 for index, row in enumerate(ranked)}
    return int(selected["seed"]), float(selected["metrics"]["ate_rmse_m"]), ranks


def final_map_artifacts(run_dir: Path) -> tuple[Path, Path]:
    candidates = []
    for directory in run_dir.glob("*_shutdown"):
        match = re.fullmatch(r"(\d+)_shutdown", directory.name)
        if match:
            candidates.append((int(match.group(1)), directory))
    if not candidates:
        raise FileNotFoundError(f"No final map under {run_dir}")
    map_dir = max(candidates)[1]
    cameras = map_dir / "ply/cameras.json"
    point_clouds = []
    for ply in map_dir.glob("ply/point_cloud/iteration_*/point_cloud.ply"):
        match = re.fullmatch(r"iteration_(\d+)", ply.parent.name)
        if match:
            point_clouds.append((int(match.group(1)), ply))
    if not cameras.is_file() or not point_clouds:
        raise FileNotFoundError(f"Incomplete final map under {map_dir}")
    return cameras, max(point_clouds)[1]


def tracking_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in sorted(
        results,
        key=lambda row: (
            str(row["sequence"]),
            REQUIRED_VARIANTS.index(str(row["config"])),
            int(row["seed"]),
        ),
    ):
        metrics = result["metrics"]
        run_dir = Path(result["run_dir"]).resolve()
        manifest = run_dir / "manifest.json"
        trajectory = run_dir / "CameraTrajectory_AllFrames_TUM.txt"
        if not manifest.is_file() or not trajectory.is_file():
            raise FileNotFoundError(f"Missing run evidence under {run_dir}")
        manifest_data = json.loads(manifest.read_text())
        ground_truth = Path(manifest_data["paths"]["ground_truth"]).resolve()
        rows.append({
            "sequence": result["sequence"],
            "variant": result["config"],
            "display_variant": DISPLAY_NAMES[result["config"]],
            "seed": result["seed"],
            "ate_rmse_m": metrics["ate_rmse_m"],
            "rpe_translation_rmse_m": metrics["rpe_translation_rmse_m"],
            "rpe_rotation_rmse_deg": metrics["rpe_rotation_rmse_deg"],
            "failure_rate": metrics["failure_rate"],
            "end_to_end_seconds": metrics["end_to_end_seconds"],
            "valid_motion_priors": metrics["valid_motion_priors"],
            "candidate_motion_priors": metrics["candidate_motion_priors"],
            "would_use_motion_priors": metrics["would_use_motion_priors"],
            "used_motion_priors": metrics["used_motion_priors"],
            "motion_prior_shuffle_lag_frames": metrics[
                "motion_prior_shuffle_lag_frames"
            ],
            "run_dir": str(run_dir),
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "trajectory": str(trajectory),
            "trajectory_sha256": sha256(trajectory),
            "ground_truth": str(ground_truth),
            "ground_truth_sha256": sha256(ground_truth),
        })
    return rows


def mapping_selection_rows(
    results: list[dict], mapping_root: Path
) -> tuple[list[dict], dict]:
    semantic = [row for row in results if row["config"] == "semantic"]
    sequences = sorted({str(row["sequence"]) for row in semantic})
    output = []
    summary = {}
    for sequence in sequences:
        sequence_rows = [row for row in semantic if row["sequence"] == sequence]
        selected_seed, median_ate, ranks = select_median_seed(sequence_rows)
        selected_count = 0
        for result in sorted(sequence_rows, key=lambda row: int(row["seed"])):
            seed = int(result["seed"])
            is_selected = seed == selected_seed
            run_dir = Path(result["run_dir"]).resolve()
            manifest = run_dir / "manifest.json"
            trajectory = run_dir / "CameraTrajectory_AllFrames_TUM.txt"
            row = {
                "sequence": sequence,
                "seed": seed,
                "semantic_ate_rmse_m": result["metrics"]["ate_rmse_m"],
                "ate_rank": ranks[seed],
                "median_ate_rmse_m": median_ate,
                "selected": int(is_selected),
                "selection_rule": "median Semantic all-input-frame ATE",
                "run_dir": str(run_dir),
                "run_manifest": str(manifest),
                "run_manifest_sha256": sha256(manifest),
                "tracking_trajectory": str(trajectory),
                "tracking_trajectory_sha256": sha256(trajectory),
                "final_cameras": "",
                "final_cameras_sha256": "",
                "final_ply": "",
                "final_ply_sha256": "",
                "common_manifest": "",
                "common_manifest_sha256": "",
                "mapping_protocol": "",
                "mapping_protocol_sha256": "",
            }
            if is_selected:
                selected_count += 1
                cameras, ply = final_map_artifacts(run_dir)
                common_manifest = mapping_root / sequence / "manifest.csv"
                protocol = mapping_root / sequence / "protocol.json"
                for path in (common_manifest, protocol):
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing mapping evidence: {path}")
                protocol_data = json.loads(protocol.read_text())
                if protocol_data.get("identity", {}).get("seed") != seed:
                    raise ValueError(
                        f"Mapping protocol seed does not match medoid for {sequence}"
                    )
                row.update({
                    "final_cameras": str(cameras),
                    "final_cameras_sha256": sha256(cameras),
                    "final_ply": str(ply),
                    "final_ply_sha256": sha256(ply),
                    "common_manifest": str(common_manifest),
                    "common_manifest_sha256": sha256(common_manifest),
                    "mapping_protocol": str(protocol),
                    "mapping_protocol_sha256": sha256(protocol),
                })
            output.append(row)
        if selected_count != 1:
            raise ValueError(f"Expected one selected mapping seed for {sequence}")
        summary[sequence] = {
            "selected_seed": selected_seed,
            "median_semantic_ate_rmse_m": median_ate,
            "candidate_seeds": sorted(int(row["seed"]) for row in sequence_rows),
        }
    return output, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    results_path = args.results_json.resolve()
    mapping_root = args.mapping_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(results_path)
    expected = len(REQUIRED_VARIANTS) * len(
        {row["sequence"] for row in results}
    ) * len({row["seed"] for row in results})
    if len(results) != expected:
        raise ValueError(
            f"Expected a complete variant/sequence/seed matrix, got "
            f"{len(results)} rows instead of {expected}"
        )

    per_seed = tracking_rows(results)
    selection, selection_summary = mapping_selection_rows(results, mapping_root)
    write_csv(output_dir / "tracking_per_seed.csv", per_seed)
    write_csv(output_dir / "mapping_seed_selection.csv", selection)
    payload = {
        "protocol": "motion-ablation-evidence-export-v1",
        "results_json": str(results_path),
        "results_json_sha256": sha256(results_path),
        "mapping_root": str(mapping_root),
        "tracking_rows": len(per_seed),
        "selection_rule": "median Semantic all-input-frame ATE per sequence",
        "mapping_selection": selection_summary,
        "display_name_aliases": {
            "semantic_motion_rgbd_shuffled": "Lag-30",
        },
    }
    (output_dir / "evidence_export.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
