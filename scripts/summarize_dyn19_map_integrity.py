#!/usr/bin/env python3
"""Validate and summarize the frozen DYN-19 multi-seed map-integrity matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DYN19_EXPERIMENT_ID = "DYN-19_FULL_CAUSAL_MAP_INTEGRITY_20260913"
PLAN_CONTRACT = "dyn19-map-integrity-plan-v1"
SUMMARY_CONTRACT = "dyn19-map-integrity-summary-v1"
CORE_SEQUENCES = (
    "tum_walking_xyz",
    "tum_sitting_halfsphere",
    "bonn_crowd2",
    "bonn_person_tracking2",
)
REQUIRED_SEEDS = (0, 1, 2)
REQUIRED_CONFIGS = (
    "dyn19_semantic",
    "dyn19_mapmatched",
    "dyn19_full",
)
POSE_MODES = ("online", "gt_aligned")
NORMAL_METRICS = (
    "psnr_full_mean",
    "ssim_full_mean",
    "lpips_full_mean",
    "psnr_static_mean",
    "ssim_static_mean",
    "lpips_static_mean",
)
PROXY_METRICS = (
    "proxy_coverage",
    "background_proxy_color_error_mean",
    "background_completeness_at_tau",
    "ghost_risk_proxy_at_tau_high",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def manifest_frames(path: Path) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing held-out manifest: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "frame" not in rows[0]:
        raise ValueError(f"Invalid held-out manifest: {path}")
    frames = [int(row["frame"]) for row in rows]
    if len(frames) != len(set(frames)):
        raise ValueError(f"Duplicate held-out frame in {path}")
    return frames


def finite_or_none(value: Any, label: str, path: Path) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Invalid {label} in {path}: {value!r}")
    return float(value)


def mean_or_none(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def validate_plan(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_path = root / "dyn19_map_integrity_plan.json"
    digest_path = root / "dyn19_map_integrity_plan.sha256"
    plan = load_json(plan_path)
    if plan.get("contract") != PLAN_CONTRACT:
        raise ValueError(f"Unexpected map-integrity plan contract: {plan_path}")
    digest_fields = digest_path.read_text().strip().split()
    if not digest_fields or digest_fields[0] != sha256(plan_path):
        raise ValueError("DYN-19 map-integrity plan digest mismatch")
    if plan.get("experiment_id") != DYN19_EXPERIMENT_ID:
        raise ValueError("DYN-19 experiment identity mismatch")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("DYN-19 map-integrity plan has no task list")
    expected = {
        (sequence, seed)
        for sequence in CORE_SEQUENCES
        for seed in REQUIRED_SEEDS
    }
    actual = {
        (task.get("sequence"), task.get("seed"))
        for task in tasks
        if isinstance(task, dict)
    }
    if len(tasks) != len(expected) or actual != expected:
        raise ValueError(
            f"DYN-19 map-integrity task matrix mismatch: expected={sorted(expected)}, "
            f"actual={sorted(actual)}")
    return plan, tasks


def validate_cell(task: dict[str, Any]) -> list[dict[str, Any]]:
    sequence = str(task["sequence"])
    seed = int(task["seed"])
    cell_dir = Path(str(task["output_dir"]))
    manifest_path = cell_dir / "manifest.csv"
    protocol_path = cell_dir / "protocol.json"
    proxy_path = cell_dir / "occluded_background_proxy"
    frames = manifest_frames(manifest_path)
    manifest_digest = sha256(manifest_path)
    protocol = load_json(protocol_path)
    identity = protocol.get("identity", {})
    if identity.get("sequence") != sequence or identity.get("seed") != seed:
        raise ValueError(f"Protocol identity mismatch in {protocol_path}")
    runs = protocol.get("runs")
    if not isinstance(runs, dict):
        raise ValueError(f"Protocol lacks run provenance: {protocol_path}")
    static_provenance = protocol.get("dataset", {}).get(
        "static_mask_provenance", {})
    expected_mask_dir = Path(str(task["semantic_static_mask_dir"])).resolve()
    actual_mask_dir = Path(str(static_provenance.get("directory", ""))).resolve()
    if actual_mask_dir != expected_mask_dir:
        raise ValueError(
            f"Frozen semantic mask directory mismatch in {protocol_path}")

    expected_run_hashes = task.get("run_manifest_sha256", {})
    for config in REQUIRED_CONFIGS:
        run = runs.get(config)
        if not isinstance(run, dict):
            raise ValueError(f"Missing {config} run provenance in {protocol_path}")
        if run.get("manifest_sha256") != expected_run_hashes.get(config):
            raise ValueError(
                f"Run manifest hash mismatch for {config} in {protocol_path}")

    proxy = load_json(
        proxy_path / "occluded_background_proxy_manifest.json")
    if proxy.get("contract") != "occluded-background-proxy-v1":
        raise ValueError(f"Invalid proxy contract in {proxy_path}")
    if proxy.get("heldout_manifest_sha256") != manifest_digest:
        raise ValueError(f"Proxy manifest mismatch in {proxy_path}")

    rows = []
    for pose_mode in POSE_MODES:
        for config in REQUIRED_CONFIGS:
            normal_path = (
                cell_dir / "metrics" / pose_mode / config /
                "heldout_metrics_summary.json")
            proxy_metric_path = (
                cell_dir / "proxy_metrics" / pose_mode / config /
                "occluded_background_proxy_metrics_summary.json")
            normal = load_json(normal_path)
            proxy_metrics = load_json(proxy_metric_path)
            if normal.get("manifest_sha256") != manifest_digest:
                raise ValueError(
                    f"Normal rendering manifest mismatch: {normal_path}")
            if normal.get("frames") != len(frames) or normal.get("frame_ids") != frames:
                raise ValueError(
                    f"Normal rendering frame mismatch: {normal_path}")
            if proxy_metrics.get("heldout_manifest_sha256") != manifest_digest:
                raise ValueError(
                    f"Proxy metric manifest mismatch: {proxy_metric_path}")
            if (
                proxy_metrics.get("frames") != len(frames) or
                proxy_metrics.get("frame_ids") != frames
            ):
                raise ValueError(
                    f"Proxy metric frame mismatch: {proxy_metric_path}")
            if proxy_metrics.get("variant") != config:
                raise ValueError(
                    f"Proxy metric config mismatch: {proxy_metric_path}")
            if proxy_metrics.get("pose_mode") != pose_mode:
                raise ValueError(
                    f"Proxy metric pose mode mismatch: {proxy_metric_path}")

            row: dict[str, Any] = {
                "sequence": sequence,
                "seed": seed,
                "config": config,
                "pose_mode": pose_mode,
                "frames": len(frames),
                "manifest_sha256": manifest_digest,
                "normal_metrics_path": str(normal_path),
                "normal_metrics_sha256": sha256(normal_path),
                "proxy_metrics_path": str(proxy_metric_path),
                "proxy_metrics_sha256": sha256(proxy_metric_path),
            }
            for field in NORMAL_METRICS:
                row[field] = finite_or_none(
                    normal.get(field), field, normal_path)
            for field in PROXY_METRICS:
                row[field] = finite_or_none(
                    proxy_metrics.get(field), field, proxy_metric_path)
            rows.append(row)
    return rows


def summarize(root: Path) -> dict[str, Any]:
    root = root.resolve()
    plan, tasks = validate_plan(root)
    rows = []
    for task in sorted(tasks, key=lambda item: (
        str(item["sequence"]), int(item["seed"]))):
        rows.extend(validate_cell(task))
    expected_rows = (
        len(CORE_SEQUENCES) * len(REQUIRED_SEEDS) *
        len(REQUIRED_CONFIGS) * len(POSE_MODES)
    )
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Map-integrity rows incomplete: {len(rows)} != {expected_rows}")

    fieldnames = list(rows[0])
    csv_path = root / "dyn19_map_integrity_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["config"], row["pose_mode"])].append(row)
    aggregates = {}
    for (config, pose_mode), group_rows in sorted(groups.items()):
        aggregate = {
            "cells": len(group_rows),
            "sequences": sorted({row["sequence"] for row in group_rows}),
            "seeds": sorted({row["seed"] for row in group_rows}),
        }
        for field in (*NORMAL_METRICS, *PROXY_METRICS):
            aggregate[field] = mean_or_none([row[field] for row in group_rows])
        aggregates[f"{config}/{pose_mode}"] = aggregate

    payload = {
        "contract": SUMMARY_CONTRACT,
        "experiment_id": DYN19_EXPERIMENT_ID,
        "status": "complete",
        "evidence_scope": (
            "Four core claim-bearing scenes x three seeds x Semantic/MapMatched/"
            "Full, with common non-keyframe final-map views. The "
            "occluded-background figures are proxy/diagnostic evidence, not "
            "true ghost-contamination ground truth."
        ),
        "plan": str(root / "dyn19_map_integrity_plan.json"),
        "plan_sha256": sha256(root / "dyn19_map_integrity_plan.json"),
        "tracking_root": plan.get("tracking_root"),
        "tracking_main_overlap_certificate": plan.get(
            "tracking_main_overlap_certificate"),
        "required_matrix": {
            "sequences": list(CORE_SEQUENCES),
            "seeds": list(REQUIRED_SEEDS),
            "configs": list(REQUIRED_CONFIGS),
            "pose_modes": list(POSE_MODES),
            "expected_cells": expected_rows,
        },
        "rows": rows,
        "aggregates": aggregates,
    }
    json_path = root / "dyn19_map_integrity_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
