#!/usr/bin/env python3
"""Import compact, claim-bearing DYN-19 rows from the public metadata release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tracking_rows(path: Path, expected: int) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload["results"]
    if len(results) != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {len(results)}")

    rows = []
    for result in results:
        metrics = result["metrics"]
        certificate = metrics["recovered_support_overlap_certificate"]
        evidence = certificate["evidence"]
        rows.append(
            {
                "config": result["config"],
                "sequence": result["sequence"],
                "seed": result["seed"],
                "status": result["status"],
                "ate_cm": 100.0 * metrics["ate_rmse_m"],
                "failure_rate_percent": 100.0 * metrics["failure_rate"],
                "end_to_end_seconds": metrics["end_to_end_seconds"],
                "trajectory_coverage": metrics["trajectory_coverage"],
                "recovered_support_frames": evidence["recovered_support_frames"],
                "tracking_recovery_audit_pixels": evidence[
                    "tracking_recovery_audit_pixels"
                ],
                "tracking_recovery_mapping_leak_pixels": evidence[
                    "tracking_recovery_mapping_leak_pixels"
                ],
                "certificate_status": certificate["status"],
                "source_commit": result["source"]["commit"],
                "source_dirty": result["source"]["dirty"],
                "run_summary_sha256": certificate["provenance"][
                    "run_summary_sha256"
                ],
                "frame_metrics_sha256": certificate["provenance"][
                    "frame_metrics_sha256"
                ],
            }
        )
    return sorted(rows, key=lambda row: (row["config"], row["sequence"], row["seed"]))


def mapping_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != 72:
        raise ValueError(f"{path}: expected 72 rows, found {len(source_rows)}")

    fields = [
        "sequence",
        "seed",
        "config",
        "pose_mode",
        "frames",
        "psnr_full_mean",
        "ssim_full_mean",
        "lpips_full_mean",
        "psnr_static_mean",
        "ssim_static_mean",
        "lpips_static_mean",
        "proxy_coverage",
        "background_proxy_color_error_mean",
        "background_completeness_at_tau",
        "ghost_risk_proxy_at_tau_high",
        "manifest_sha256",
        "normal_metrics_sha256",
        "proxy_metrics_sha256",
    ]
    return sorted(
        ({field: row[field] for field in fields} for row in source_rows),
        key=lambda row: (
            row["pose_mode"],
            row["config"],
            row["sequence"],
            int(row["seed"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-root",
        type=Path,
        required=True,
        help="Extracted DYN19_metadata.tar.gz metadata directory.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Downloaded DYN19_metadata.tar.gz used to verify provenance.",
    )
    args = parser.parse_args()

    metadata = args.metadata_root.resolve()
    archive = args.archive.resolve()
    expected_archive_sha256 = (
        "925d3d80ad0b9c586875a267bec8170bc3b1a580b19a1a05ad64780bd5ec2fb2"
    )
    archive_sha256 = sha256(archive)
    if archive_sha256 != expected_archive_sha256:
        raise ValueError(
            f"metadata archive hash mismatch: {archive_sha256} != {expected_archive_sha256}"
        )

    main_rows = tracking_rows(metadata / "main" / "all_results.json", 90)
    mechanism_rows = tracking_rows(metadata / "mechanisms" / "all_results.json", 120)
    map_rows = mapping_rows(metadata / "mapping" / "dyn19_map_integrity_summary.csv")

    tracking_fields = [
        "config",
        "sequence",
        "seed",
        "status",
        "ate_cm",
        "failure_rate_percent",
        "end_to_end_seconds",
        "trajectory_coverage",
        "recovered_support_frames",
        "tracking_recovery_audit_pixels",
        "tracking_recovery_mapping_leak_pixels",
        "certificate_status",
        "source_commit",
        "source_dirty",
        "run_summary_sha256",
        "frame_metrics_sha256",
    ]
    mapping_fields = list(map_rows[0])

    DATA.mkdir(exist_ok=True)
    outputs = {
        DATA / "dyn19_main_90_cells.csv": (tracking_fields, main_rows),
        DATA / "dyn19_mechanism_120_cells.csv": (tracking_fields, mechanism_rows),
        DATA / "dyn19_mapping_72_cells.csv": (mapping_fields, map_rows),
    }
    for path, (fields, rows) in outputs.items():
        write_csv(path, fields, rows)

    phase_main = json.loads(
        (metadata / "main" / "dyn19_phase_report.json").read_text(encoding="utf-8")
    )
    phase_mechanisms = json.loads(
        (metadata / "mechanisms" / "dyn19_phase_report.json").read_text(
            encoding="utf-8"
        )
    )
    overlap = json.loads(
        (metadata / "main" / "dyn19_recovered_support_overlap_aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    mapping = json.loads(
        (metadata / "mapping" / "dyn19_map_integrity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = {
        "release_tag": "dyn19-results-20260914",
        "release_date": "2026-09-14",
        "source_commit": "14c6b2ca696a7eacda8f193e5c8a0a44e55d527e",
        "metadata_archive": {
            "name": archive.name,
            "sha256": archive_sha256,
        },
        "completion": {
            "main": {
                "expected": phase_main["expected_runs"],
                "completed": phase_main["completed_runs"],
                "status": phase_main["status"],
            },
            "mechanisms": {
                "expected": phase_mechanisms["expected_runs"],
                "completed": phase_mechanisms["completed_runs"],
                "status": phase_mechanisms["status"],
            },
            "mapping": {
                "expected": mapping["required_matrix"]["expected_cells"],
                "completed": len(mapping["rows"]),
                "status": mapping["status"],
            },
        },
        "main_overlap_certificate": {
            "status": overlap["status"],
            "safety_status": overlap["safety_status"],
            "positive_recovery_evidence": overlap["positive_recovery_evidence"],
            "certificate_status_counts": overlap["certificate_status_counts"],
            "cells": len(overlap["per_sequence_seed"]),
            "recovered_support_frames": sum(
                item["evidence"]["recovered_support_frames"]
                for item in overlap["per_sequence_seed"].values()
            ),
            "audit_pixels": sum(
                item["evidence"]["tracking_recovery_audit_pixels"]
                for item in overlap["per_sequence_seed"].values()
            ),
            "mapping_leak_pixels": sum(
                item["evidence"]["tracking_recovery_mapping_leak_pixels"]
                for item in overlap["per_sequence_seed"].values()
            ),
        },
        "claim_boundary": {
            "design": "ordered ablation",
            "independent_component_effects": False,
            "interaction_effects": False,
            "proxy_is_independent_ground_truth": False,
            "external_baselines_protocol_matched": False,
        },
        "generated_files": {
            path.name: {"rows": len(rows), "sha256": sha256(path)}
            for path, (_, rows) in outputs.items()
        },
    }
    (DATA / "dyn19_release_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
