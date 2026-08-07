#!/usr/bin/env python3
"""Validate and summarize common-view mapping ablation metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


PROFILES = {
    "motion": {
        "variants": (
            "semantic",
            "semantic_motion_init_rgbd",
            "semantic_motion_ungated",
            "semantic_motion_rgbd",
            "semantic_motion_rgbd_shadow",
            "semantic_motion_rgbd_shuffled",
        ),
        "display_names": {
            "semantic": "Semantic",
            "semantic_motion_init_rgbd": "Init-only",
            "semantic_motion_ungated": "Ungated",
            "semantic_motion_rgbd": "Full",
            "semantic_motion_rgbd_shadow": "Shadow",
            "semantic_motion_rgbd_shuffled": "Lag-30",
        },
    },
    "dypho": {
        "variants": (
            "semantic",
            "dypho_raw",
            "dypho_temporal",
            "dypho_feature",
            "dypho_compatible",
            "dypho_decoupled",
            "dypho_flow_guarded",
            "dypho_flow_adaptive",
        ),
        "display_names": {
            "semantic": "Semantic",
            "dypho_raw": "Raw",
            "dypho_temporal": "Temporal",
            "dypho_feature": "Adaptive ORB",
            "dypho_compatible": "Full",
            "dypho_decoupled": "Decoupled",
            "dypho_flow_guarded": "Flow-Guarded",
            "dypho_flow_adaptive": "Adaptive Flow-Guarded",
        },
    },
    "dypho-core": {
        "variants": (
            "dypho_raw",
            "dypho_decoupled",
            "dypho_flow_guarded",
            "dypho_flow_adaptive",
        ),
        "display_names": {
            "dypho_raw": "Raw",
            "dypho_decoupled": "Decoupled",
            "dypho_flow_guarded": "Flow-Guarded",
            "dypho_flow_adaptive": "Adaptive Flow-Guarded",
        },
        "metric_directories": {
            "dypho_raw": "Raw",
            "dypho_decoupled": "Decoupled",
            "dypho_flow_guarded": "Flow-Guarded",
            "dypho_flow_adaptive": "Adaptive Flow-Guarded",
        },
    },
}
# Backward-compatible aliases for callers of the original motion-only script.
REQUIRED_VARIANTS = PROFILES["motion"]["variants"]
DISPLAY_NAMES = PROFILES["motion"]["display_names"]
POSE_MODES = ("online", "gt_aligned")
METRIC_FIELDS = (
    "psnr_full_mean",
    "ssim_full_mean",
    "lpips_full_mean",
    "psnr_static_mean",
    "ssim_static_mean",
    "lpips_static_mean",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def manifest_frames(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest does not exist: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "frame" not in rows[0]:
        raise ValueError(f"Manifest is empty or lacks a frame column: {path}")
    frames = [int(row["frame"]) for row in rows]
    if len(frames) != len(set(frames)):
        raise ValueError(f"Manifest contains duplicate frame IDs: {path}")
    return frames


def metric_summary_path(
    sequence_dir: Path,
    profile: str,
    pose_mode: str,
    variant: str,
) -> Path:
    metric_directories = PROFILES[profile].get("metric_directories")
    if metric_directories is not None:
        return (
            sequence_dir
            / "metrics"
            / pose_mode
            / metric_directories[variant]
            / "heldout_metrics_summary.json"
        )
    return (
        sequence_dir
        / "metrics"
        / f"{variant}_{pose_mode}"
        / "heldout_metrics_summary.json"
    )


def collect_rows(
    root: Path, profile: str = "motion"
) -> tuple[list[dict], dict]:
    root = root.resolve()
    if profile not in PROFILES:
        raise ValueError(f"Unknown mapping ablation profile: {profile}")
    required_variants = PROFILES[profile]["variants"]
    display_names = PROFILES[profile]["display_names"]
    sequence_dirs = sorted(
        path.parent for path in root.glob("*/manifest.csv") if path.is_file())
    if not sequence_dirs:
        raise ValueError(f"No sequence manifests found under {root}")

    output_rows = []
    sequences = {}
    for sequence_dir in sequence_dirs:
        sequence = sequence_dir.name
        manifest_path = sequence_dir / "manifest.csv"
        protocol_path = sequence_dir / "protocol.json"
        frames = manifest_frames(manifest_path)
        manifest_digest = sha256(manifest_path)
        protocol = load_json(protocol_path)
        identity = protocol.get("identity", {})
        if identity.get("sequence") != sequence:
            raise ValueError(
                f"Protocol sequence mismatch for {sequence}: "
                f"{identity.get('sequence')!r}")
        seed = identity.get("seed")
        if not isinstance(seed, int):
            raise ValueError(f"Protocol seed is missing for {sequence}")

        sequences[sequence] = {
            "seed": seed,
            "frames": len(frames),
            "frame_ids": frames,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_digest,
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
        }
        for pose_mode in POSE_MODES:
            for variant in required_variants:
                metric_path = metric_summary_path(
                    sequence_dir, profile, pose_mode, variant)
                metrics = load_json(metric_path)
                if metrics.get("manifest_sha256") != manifest_digest:
                    raise ValueError(
                        f"Manifest hash mismatch in {metric_path}")
                if metrics.get("frames") != len(frames):
                    raise ValueError(
                        f"Frame count mismatch in {metric_path}")
                if metrics.get("frame_ids") != frames:
                    raise ValueError(
                        f"Frame IDs do not match the common manifest in "
                        f"{metric_path}")

                row = {
                    "sequence": sequence,
                    "seed": seed,
                    "pose_mode": pose_mode,
                    "variant": variant,
                    "display_variant": display_names[variant],
                    "frames": len(frames),
                    "manifest_sha256": manifest_digest,
                    "metrics_file": str(metric_path),
                    "metrics_sha256": sha256(metric_path),
                }
                for field in METRIC_FIELDS:
                    value = metrics.get(field)
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(
                            f"Missing or non-finite {field} in {metric_path}")
                    row[field] = float(value)
                output_rows.append(row)

    return output_rows, sequences


def write_summary(
    root: Path,
    rows: list[dict],
    sequences: dict,
    profile: str = "motion",
) -> None:
    if profile not in PROFILES:
        raise ValueError(f"Unknown mapping ablation profile: {profile}")
    required_variants = PROFILES[profile]["variants"]
    display_names = PROFILES[profile]["display_names"]
    csv_path = root / "mapping_summary.csv"
    json_path = root / "mapping_summary.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "protocol": "common-view-mapping-ablation-v2",
        "profile": profile,
        "status": "complete",
        "evidence_scope": (
            "Single frozen seed common-view diagnostic; not a multi-seed "
            "mapping superiority claim."
        ),
        "required_variants": list(required_variants),
        "display_names": display_names,
        "pose_modes": list(POSE_MODES),
        "sequences": sequences,
        "rows": rows,
    }
    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root containing one directory per common-view sequence",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="motion",
        help="Required variant matrix to validate",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows, sequences = collect_rows(root, args.profile)
    write_summary(root, rows, sequences, args.profile)
    print(
        f"Wrote {len(rows)} rows for {len(sequences)} sequences to "
        f"{root / 'mapping_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
