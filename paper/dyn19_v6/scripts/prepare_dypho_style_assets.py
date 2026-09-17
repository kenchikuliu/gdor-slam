#!/usr/bin/env python3
"""Prepare source-labeled CSVs and audit metadata for the DyPho-style build."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


DEFAULT_RELEASE = Path(
    "/mnt/nas_datasets/slam-experiments/DynaGS-SLAM/"
    "dyn19_release_20260914_r1"
)
DEFAULT_SKILL = Path(
    "/home/slam/.codex/skills/dypho-slam-figure-automation"
)

CONFIGS = (
    ("dyn19_semantic", "DYN-19 Semantic"),
    ("dyn19_mapmatched", "DYN-19 MapMatched"),
    ("dyn19_full", "DYN-19 Full"),
)
TUM4 = (
    ("tum_walking_xyz", "w/xyz"),
    ("tum_walking_halfsphere", "w/half"),
    ("tum_walking_static", "w/static"),
    ("tum_sitting_halfsphere", "s/half"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def expect(condition: bool, message: str) -> dict[str, object]:
    return {"passed": condition, "message": message}


def tracking_table(
    results: list[dict], dypho_table: Path
) -> tuple[list[list[object]], dict[str, object]]:
    header = ["Method"]
    for _, short_name in TUM4:
        header.extend((f"{short_name} ATE", f"{short_name} Std"))
    header.extend(("Avg. ATE", "Avg. Std"))
    rows: list[list[object]] = [header]
    local_summary: dict[str, object] = {}

    for config, label in CONFIGS:
        row: list[object] = [label]
        sequence_means: list[float] = []
        sequence_summary: dict[str, object] = {}
        for sequence, _ in TUM4:
            values = sorted(
                result["metrics"]["ate_rmse_m"] * 100.0
                for result in results
                if result["config"] == config and result["sequence"] == sequence
            )
            if len(values) != 3:
                raise RuntimeError(
                    f"Expected three tracking seeds for {config}/{sequence}, got {len(values)}"
                )
            mean = statistics.fmean(values)
            std = statistics.pstdev(values)
            sequence_means.append(mean)
            row.extend((f"{mean:.4f}", f"{std:.4f}"))
            sequence_summary[sequence] = {
                "seed_values_ate_cm": values,
                "mean_ate_cm": mean,
                "population_std_ate_cm": std,
                "n": 3,
            }
        average = statistics.fmean(sequence_means)
        cross_sequence_std = statistics.pstdev(sequence_means)
        row.extend((f"{average:.4f}", f"{cross_sequence_std:.4f}"))
        rows.append(row)
        local_summary[config] = {
            "sequences": sequence_summary,
            "average_of_sequence_means_ate_cm": average,
            "population_std_of_sequence_means_ate_cm": cross_sequence_std,
        }

    with dypho_table.open("r", encoding="utf-8-sig", newline="") as stream:
        dypho_rows = list(csv.reader(stream))
    published = next(row for row in dypho_rows if row and row[0] == "Ours")
    if len(published) < 12:
        raise RuntimeError("DyPho Table I Ours row is incomplete")
    rows.append(["DyPho-SLAM [ext]", *published[2:12]])

    return rows, {
        "local": local_summary,
        "external_report": {
            "method": "DyPho-SLAM",
            "source_row": published[:12],
            "source_type": "external-report",
            "protocol_match": False,
        },
        "std_convention": (
            "Local per-sequence Std is population Std across seeds 0/1/2 (ddof=0); "
            "local Avg. Std is population Std across the four sequence means. "
            "DyPho values are copied exactly from its registered Table I row."
        ),
    }


def mapping_table(mapping_rows: list[dict[str, str]]) -> tuple[list[list[object]], dict]:
    header = [
        "Configuration",
        "Online PSNR",
        "Online SSIM",
        "GT-align PSNR",
        "GT-align SSIM",
    ]
    rows: list[list[object]] = [header]
    summary: dict[str, object] = {}
    for config, label in CONFIGS:
        short_label = label[len("DYN-19 ") :] if label.startswith("DYN-19 ") else label
        row: list[object] = [short_label]
        config_summary: dict[str, object] = {}
        for pose_mode in ("online", "gt_aligned"):
            selected = [
                item
                for item in mapping_rows
                if item["config"] == config and item["pose_mode"] == pose_mode
            ]
            if len(selected) != 12:
                raise RuntimeError(
                    f"Expected 12 mapping cells for {config}/{pose_mode}, got {len(selected)}"
                )
            psnr = statistics.fmean(float(item["psnr_static_mean"]) for item in selected)
            ssim = statistics.fmean(float(item["ssim_static_mean"]) for item in selected)
            row.extend((f"{psnr:.3f}", f"{ssim:.3f}"))
            config_summary[pose_mode] = {
                "cells": 12,
                "rendered_views": sum(int(item["frames"]) for item in selected),
                "static_psnr_db": psnr,
                "static_ssim": ssim,
            }
        rows.append(row)
        summary[config] = config_summary
    return rows, summary


def runtime_table(results: list[dict]) -> tuple[list[list[object]], dict]:
    header = [
        "Configuration",
        "E2E [s/run]",
        "E2E [ms/frame]",
        "Mean failure [%]",
        "Relative runtime",
        "Route certificate",
    ]
    aggregates: dict[str, dict] = {}
    for config, _ in CONFIGS:
        selected = [result for result in results if result["config"] == config]
        if len(selected) != 30:
            raise RuntimeError(f"Expected 30 runtime rows for {config}, got {len(selected)}")
        total_seconds = sum(result["metrics"]["end_to_end_seconds"] for result in selected)
        total_frames = sum(result["metrics"]["input_frames"] for result in selected)
        failed_frames = sum(
            round(result["metrics"]["failure_rate"] * result["metrics"]["input_frames"])
            for result in selected
        )
        mean_failure_percent = 100.0 * statistics.fmean(
            result["metrics"]["failure_rate"] for result in selected
        )
        certificates = Counter(
            result["metrics"]["recovered_support_certificate_status"]
            for result in selected
        )
        aggregates[config] = {
            "runs": len(selected),
            "total_seconds": total_seconds,
            "mean_seconds_per_run": total_seconds / len(selected),
            "total_frames": total_frames,
            "milliseconds_per_frame": 1000.0 * total_seconds / total_frames,
            "failed_frames": failed_frames,
            "global_failed_frame_percent": 100.0 * failed_frames / total_frames,
            "mean_failure_percent": mean_failure_percent,
            "certificates": dict(sorted(certificates.items())),
        }

    semantic_runtime = aggregates["dyn19_semantic"]["mean_seconds_per_run"]
    rows: list[list[object]] = [header]
    for config, label in CONFIGS:
        item = aggregates[config]
        certificate = "+".join(
            f"{count}/30 {status}" for status, count in item["certificates"].items()
        )
        relative = item["mean_seconds_per_run"] / semantic_runtime
        item["relative_to_semantic"] = relative
        rows.append(
            [
                label[len("DYN-19 ") :] if label.startswith("DYN-19 ") else label,
                f"{item['mean_seconds_per_run']:.3f}",
                f"{item['milliseconds_per_frame']:.3f}",
                f"{item['mean_failure_percent']:.3f}",
                f"{relative:.2f}x",
                certificate,
            ]
        )
    return rows, aggregates


def figure_availability() -> list[dict[str, str]]:
    return [
        {
            "asset": "Fig. 1",
            "status": "blocked",
            "available": "input RGB-D, local trajectories/maps, DYN-19 novel-view renders",
            "missing": "constructed-feature panels and a label-compatible real baseline row",
            "next_action": "instrument/replay feature exports and freeze real baseline panels",
        },
        {
            "asset": "Fig. 2",
            "status": "blocked",
            "available": "dynamic masks, RGB-D input, Gaussian map/checkpoint",
            "missing": "keypoints, feature maps/bins, matches, and temporal-prior state exports",
            "next_action": "add deterministic intermediate-state export hooks",
        },
        {
            "asset": "Fig. 3",
            "status": "inputs-ready",
            "available": "four TUM DYN-19 trajectories and public ground truth",
            "missing": "explicit associated and SE(3)-aligned plot trajectories",
            "next_action": "export aligned trajectories upstream, then run the locked Fig. 3 build",
        },
        {
            "asset": "Fig. 4",
            "status": "blocked-for-locked-layout",
            "available": "common-view input plus Semantic, MapMatched, and Full renders",
            "missing": "real SplaTAM/Photo-SLAM panels required by the locked five-column layout",
            "next_action": "complete P0 baseline renders or approve a separate DYN-specific layout",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL)
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("paper/dyn19_v6/data/dyn19_mapping_cells.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/dyn19_v6/dypho_style_assets"),
    )
    args = parser.parse_args()

    all_results_path = args.release / "metadata/main/all_results.json"
    release_manifest_path = args.release / "DYN19_RELEASE_MANIFEST.json"
    dypho_table_path = args.skill_root / "reference/tables/table1/data.csv"
    mapping_path = args.mapping_csv.resolve()
    output = args.output.resolve()

    results = json.loads(all_results_path.read_text(encoding="utf-8"))["results"]
    mapping_rows = load_csv(mapping_path)
    tracking_rows, tracking_summary = tracking_table(results, dypho_table_path)
    mapping_table_rows, mapping_summary = mapping_table(mapping_rows)
    runtime_rows, runtime_summary = runtime_table(results)

    table1_path = output / "data/table1_tracking.csv"
    table2_path = output / "data/table2_mapping.csv"
    table3_path = output / "data/table3_runtime.csv"
    write_csv(table1_path, tracking_rows)
    write_csv(table2_path, mapping_table_rows)
    write_csv(table3_path, runtime_rows)

    sources = {
        "dyn19_all_results": {
            "path": str(all_results_path),
            "sha256": sha256(all_results_path),
        },
        "dyn19_release_manifest": {
            "path": str(release_manifest_path),
            "sha256": sha256(release_manifest_path),
        },
        "dyn19_mapping_cells": {
            "path": str(mapping_path),
            "sha256": sha256(mapping_path),
        },
        "dypho_table1": {
            "path": str(dypho_table_path),
            "sha256": sha256(dypho_table_path),
        },
    }
    table_manifest = {
        "generated_date": "2026-09-17",
        "status": "draft-only; human review required",
        "comparison_mode": "presentation-flexible",
        "sources": sources,
        "tables": {
            "table1": {
                "purpose": "tracking accuracy on the four TUM scenes used by DyPho Table I",
                "source_types": {
                    "DYN-19 Semantic": "local-rerun",
                    "DYN-19 MapMatched": "local-rerun",
                    "DYN-19 Full": "local-rerun",
                    "DyPho-SLAM [ext]": "external-report",
                },
                "statistics": tracking_summary,
                "claim_boundary": (
                    "DyPho-SLAM is external positioning context, not a protocol-matched rerun."
                ),
            },
            "table2": {
                "purpose": "local multi-seed static-region mapping diagnostic",
                "source_type": "diagnostic",
                "cells": 72,
                "statistics": mapping_summary,
                "claim_boundary": (
                    "No DyPho numeric mapping row exists; GT-aligned rendering remains an "
                    "end-to-end map diagnostic because online poses built the map."
                ),
            },
            "table3": {
                "purpose": "local end-to-end runtime and tracking-safety summary",
                "source_type": "local-rerun",
                "hardware_consistency": "clean-single-gpu",
                "hardware_note": (
                    "All 90 runs used host 'slam', GPU identifier 1, and a serial per-block "
                    "GPU lock; the GPU model was not recorded in the frozen manifests."
                ),
                "timing_scope": (
                    "Application end_to_end_seconds only; no fabricated tracking/mapping split."
                ),
                "failure_scope": (
                    "Paper-facing value is the equal-weight mean of the 30 per-run failure "
                    "rates, matching the manuscript; the globally frame-weighted rate remains "
                    "in this manifest as global_failed_frame_percent."
                ),
                "statistics": runtime_summary,
            },
        },
        "generated_csvs": {
            "table1": {"path": str(table1_path), "sha256": sha256(table1_path)},
            "table2": {"path": str(table2_path), "sha256": sha256(table2_path)},
            "table3": {"path": str(table3_path), "sha256": sha256(table3_path)},
        },
    }
    write_json(output / "data/table_manifest.json", table_manifest)

    availability = figure_availability()
    write_json(output / "reports/figure_availability.json", availability)
    write_csv(
        output / "reports/figure_availability.csv",
        [["asset", "status", "available", "missing", "next_action"]]
        + [
            [
                item["asset"],
                item["status"],
                item["available"],
                item["missing"],
                item["next_action"],
            ]
            for item in availability
        ],
    )

    checks = [
        expect(len(results) == 90, "main all_results contains 90 runs"),
        expect(all(item["status"] == "complete" for item in results), "all 90 runs complete"),
        expect(len(mapping_rows) == 72, "mapping evidence contains 72 cells"),
        expect(
            {item["config"] for item in results} == {config for config, _ in CONFIGS},
            "main result configs match the three table rows",
        ),
        expect(
            table_manifest["tables"]["table1"]["source_types"]["DyPho-SLAM [ext]"]
            == "external-report",
            "DyPho row is labeled external-report",
        ),
        expect(
            "DyPho" not in json.dumps(mapping_summary),
            "mapping table contains no fabricated DyPho numeric row",
        ),
    ]
    write_json(
        output / "reports/source_validation.json",
        {
            "passed": all(check["passed"] for check in checks),
            "checks": checks,
            "release_allowed": False,
            "required_next_step": "run skill build and complete human review",
        },
    )
    return 0 if all(check["passed"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
