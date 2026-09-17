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
DEFAULT_PROTOCOL300 = Path(
    "/home/slam/experiments/dypho_skill_full_rerun_20260729"
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
ORDERED_ABLATIONS = (
    ("main", "dyn19_semantic", "Semantic"),
    ("main", "dyn19_mapmatched", "MapMatched"),
    ("mechanisms", "dyn19_temporal_map", "T+M"),
    ("mechanisms", "dyn19_temporal_flow_map", "T+F+M"),
    ("mechanisms", "dyn19_temporal_flow_risk_map", "T+F+R+M"),
    ("main", "dyn19_full", "T+F+R+A+M (Full)"),
    ("mechanisms", "dyn19_full_no_mapping", "T+F+R+A+NoM"),
)


def external_method_name(name: str) -> str:
    renamed = {
        "Splatam": "SplaTAM",
        "Ours": "DyPho-SLAM",
    }.get(name, name)
    return f"{renamed} [ext]"


def external_tracking_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    selected: list[list[str]] = []
    for row in rows[1:]:
        if not row or not row[0] or row[0].endswith("methods"):
            continue
        if len(row) < 12:
            raise RuntimeError(f"Incomplete external tracking row: {row}")
        selected.append([external_method_name(row[0]), *row[2:12]])
    return selected


def external_runtime_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    selected: list[list[str]] = []
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        if len(row) < 5:
            raise RuntimeError(f"Incomplete external runtime row: {row}")
        selected.append([external_method_name(row[0]), *row[1:5], "external-report"])
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


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
    external_rows = external_tracking_rows(dypho_table)
    rows: list[list[object]] = [header, *external_rows]
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

    return rows, {
        "local": local_summary,
        "external_reports": {
            row[0]: {
                "values": row[1:],
                "source_type": "external-report",
                "protocol_match": False,
            }
            for row in external_rows
        },
        "std_convention": (
            "Local per-sequence Std is population Std across seeds 0/1/2 (ddof=0); "
            "local Avg. Std is population Std across the four sequence means. "
            "Every [ext] value is copied exactly from the registered DyPho Table I."
        ),
    }


def ordered_ablation_table(
    main_results: list[dict], mechanism_results: list[dict]
) -> tuple[list[list[object]], dict[str, object]]:
    header = [
        "Configuration",
        "w/xyz ATE",
        "w/xyz Std",
        "w/half ATE",
        "w/half Std",
        "Avg. ATE",
    ]
    rows: list[list[object]] = [header]
    summary: dict[str, object] = {}
    result_sets = {"main": main_results, "mechanisms": mechanism_results}
    target_sequences = ("tum_walking_xyz", "tum_walking_halfsphere")

    for phase, config, label in ORDERED_ABLATIONS:
        row: list[object] = [label]
        sequence_means: list[float] = []
        sequence_summary: dict[str, object] = {}
        for sequence in target_sequences:
            values = sorted(
                result["metrics"]["ate_rmse_m"] * 100.0
                for result in result_sets[phase]
                if result["config"] == config and result["sequence"] == sequence
            )
            if len(values) != 3:
                raise RuntimeError(
                    f"Expected three ablation seeds for {config}/{sequence}, got {len(values)}"
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
        row.append(f"{average:.4f}")
        rows.append(row)
        summary[config] = {
            "phase": phase,
            "sequences": sequence_summary,
            "average_of_sequence_means_ate_cm": average,
        }
    return rows, summary


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


def broad_runtime_table(
    dypho_table: Path, runtime_summary: dict[str, dict]
) -> tuple[list[list[object]], dict[str, object]]:
    header = [
        "Method",
        "Tracking [ms]",
        "Mapping [ms]",
        "FPS [hz]",
        "Operate time",
        "Source scope",
    ]
    external_rows = external_runtime_rows(dypho_table)
    full = runtime_summary["dyn19_full"]
    full_fps = full["total_frames"] / full["total_seconds"]
    local_row = [
        "DYN-19 Full [local E2E]",
        "×",
        "×",
        f"{full_fps:.2f}",
        f"{full['mean_seconds_per_run']:.2f} s/run",
        "local E2E; 30 runs",
    ]
    rows: list[list[object]] = [header, *external_rows, local_row]
    return rows, {
        "external_reports": {
            row[0]: {
                "tracking_ms": row[1],
                "mapping_ms": row[2],
                "fps_hz": row[3],
                "operate_time": row[4],
                "source_type": "external-report",
                "protocol_match": False,
            }
            for row in external_rows
        },
        "local_dyn19_full": {
            "source_type": "local-rerun",
            "timing_scope": "application end-to-end",
            "runs": full["runs"],
            "total_frames": full["total_frames"],
            "total_seconds": full["total_seconds"],
            "aggregate_fps": full_fps,
            "mean_seconds_per_run": full["mean_seconds_per_run"],
            "tracking_mapping_split_available": False,
        },
        "claim_boundary": (
            "External rows are source-reported on mixed hardware and scopes. The local "
            "DYN-19 row is application end-to-end throughput; no component split is inferred."
        ),
    }


def comparison_matrix(protocol300: Path) -> list[list[str]]:
    header = [
        "ID",
        "FlowParse minimum surface",
        "Sequences / scope",
        "Methods",
        "Evidence status",
        "Source type",
        "Evidence path",
        "Claim boundary / next action",
    ]
    p300_tables = protocol300 / "metrics/protocol300/tables"
    p300_fig4 = protocol300 / "figure_evidence_protocol300_v2"
    rows = [
        [
            "M1",
            "Broad tracking comparison",
            "TUM fr3/w/xyz, w/half, w/static, s/half",
            "ORB-SLAM3; Dyna-SLAM; NICE-SLAM; ESLAM; RoDyn-SLAM; SplaTAM; GS-SLAM; GassiDy; DGS-SLAM; Photo-SLAM; DyPho-SLAM; DYN-19 variants",
            "complete-source-labeled",
            "external-report + local-rerun",
            "paper/dyn19_v6/dypho_style_assets/data/table1_tracking.csv",
            "External rows are not protocol-matched and are excluded from local aggregate claims.",
        ],
        [
            "M2",
            "Module ablation on w/xyz and w/half",
            "TUM fr3/w/xyz and w/half; seeds 0/1/2",
            "Semantic; MapMatched; T+M; T+F+M; T+F+R+M; Full; Full-NoM",
            "complete-local",
            "local-rerun",
            "paper/dyn19_v6/dypho_supplemental_assets/data/table2_ablation.csv",
            "Ordered configurations, not a factorial component-effect estimate.",
        ],
        [
            "M3",
            "Current-method quantitative mapping",
            "4 scenes x 3 seeds x 3 configs x 2 pose modes; common views",
            "DYN-19 Semantic; MapMatched; Full",
            "complete-local-diagnostic",
            "diagnostic",
            "paper/dyn19_v6/dypho_style_assets/data/table2_mapping.csv",
            "Static-region PSNR/SSIM diagnostic; not universal mapping superiority.",
        ],
        [
            "M4",
            "Local baseline tracking",
            "Protocol-300: fr3/w/xyz, fr3/w/half, Bonn ps_track, Bonn r3",
            "SplaTAM; Photo-SLAM; DynaGS semantic; historical Ours",
            "complete-historical-local",
            "local-rerun + failed-boundary",
            str(p300_tables / "table1.csv"),
            "Seed 0 and 300 frames only; historical Ours is a failed-boundary ablation, not DYN-19 Full.",
        ],
        [
            "M5",
            "Local baseline mapping metrics",
            "Protocol-300 full-frame frozen-pose diagnostic",
            "SplaTAM; Photo-SLAM; DynaGS semantic; historical Ours",
            "complete-historical-diagnostic",
            "diagnostic + failed-boundary",
            str(p300_tables / "table2.csv"),
            "Dynamic foreground is included; do not claim static-map superiority.",
        ],
        [
            "M6",
            "Qualitative mapping comparison",
            "4 rows x 5 columns: input, SplaTAM, Photo-SLAM, mask-only, historical Ours",
            "SplaTAM; Photo-SLAM; Photo-SLAM mask-only; historical Ours",
            "complete-historical-package; current-method-cell-missing",
            "local-render + failed-boundary",
            str(p300_fig4 / "page_manifest.json"),
            "20 real 640x480 panels exist. A co-registered DYN-19 Full row is still required for a current-method Fig.4 claim.",
        ],
        [
            "M7",
            "Broad runtime comparison",
            "Source-reported mixed hardware plus local DYN-19 E2E",
            "11 published rows; DYN-19 Full",
            "complete-source-labeled",
            "external-report + local-rerun",
            "paper/dyn19_v6/dypho_supplemental_assets/data/table3_runtime_broad.csv",
            "Descriptive only; hardware and timing scopes differ, and DYN-19 has no tracking/mapping split.",
        ],
        [
            "M8",
            "Local baseline runtime",
            "Protocol-300 end-to-end wall time",
            "SplaTAM; Photo-SLAM; DynaGS semantic; historical Ours",
            "complete-historical-local",
            "local-rerun + failed-boundary",
            str(p300_tables / "table3.csv"),
            "Comparable within Protocol-300 only; historical Ours is not DYN-19 Full.",
        ],
        [
            "M9",
            "Current DYN-19 Full vs local SplaTAM/Photo-SLAM package",
            "One co-registered tracking/mapping/runtime/Fig.4 package",
            "DYN-19 Full; SplaTAM; Photo-SLAM",
            "missing-current-method-cell",
            "planned local-rerun",
            "not yet registered",
            "Run or render DYN-19 Full under the frozen baseline package; do not relabel the historical failed-boundary Ours row.",
        ],
    ]
    return [header, *rows]


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
            "status": "historical-package-complete-current-method-blocked",
            "available": (
                "Protocol-300 has 20 real input/SplaTAM/Photo-SLAM/mask-only/"
                "failed-boundary panels; DYN-19 has separate local variant renders"
            ),
            "missing": (
                "a co-registered DYN-19 Full panel/package against the Protocol-300 "
                "baselines; historical Ours cannot be relabeled"
            ),
            "next_action": (
                "run or render DYN-19 Full under the frozen baseline package, then build "
                "a current-method Fig. 4"
            ),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--skill-root", type=Path, default=DEFAULT_SKILL)
    parser.add_argument("--protocol300-root", type=Path, default=DEFAULT_PROTOCOL300)
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
    parser.add_argument(
        "--supplemental-output",
        type=Path,
        default=Path("paper/dyn19_v6/dypho_supplemental_assets"),
    )
    parser.add_argument(
        "--comparison-matrix",
        type=Path,
        default=Path("docs/FLOWPARSE_MINIMUM_EXPERIMENT_MATRIX.csv"),
    )
    args = parser.parse_args()

    all_results_path = args.release / "metadata/main/all_results.json"
    mechanism_results_path = args.release / "metadata/mechanisms/all_results.json"
    release_manifest_path = args.release / "DYN19_RELEASE_MANIFEST.json"
    dypho_table_path = args.skill_root / "reference/tables/table1/data.csv"
    dypho_runtime_path = args.skill_root / "reference/tables/table3/data.csv"
    mapping_path = args.mapping_csv.resolve()
    output = args.output.resolve()
    supplemental_output = args.supplemental_output.resolve()
    protocol300 = args.protocol300_root.resolve()
    comparison_matrix_path = args.comparison_matrix.resolve()

    results = json.loads(all_results_path.read_text(encoding="utf-8"))["results"]
    mechanism_results = json.loads(
        mechanism_results_path.read_text(encoding="utf-8")
    )["results"]
    mapping_rows = load_csv(mapping_path)
    tracking_rows, tracking_summary = tracking_table(results, dypho_table_path)
    mapping_table_rows, mapping_summary = mapping_table(mapping_rows)
    runtime_rows, runtime_summary = runtime_table(results)
    ablation_rows, ablation_summary = ordered_ablation_table(
        results, mechanism_results
    )
    broad_runtime_rows, broad_runtime_summary = broad_runtime_table(
        dypho_runtime_path, runtime_summary
    )

    table1_path = output / "data/table1_tracking.csv"
    table2_path = output / "data/table2_mapping.csv"
    table3_path = output / "data/table3_runtime.csv"
    write_csv(table1_path, tracking_rows)
    write_csv(table2_path, mapping_table_rows)
    write_csv(table3_path, runtime_rows)

    supplemental_table2_path = supplemental_output / "data/table2_ablation.csv"
    supplemental_table3_path = supplemental_output / "data/table3_runtime_broad.csv"
    write_csv(supplemental_table2_path, ablation_rows)
    write_csv(supplemental_table3_path, broad_runtime_rows)
    write_csv(comparison_matrix_path, comparison_matrix(protocol300))

    protocol300_sources = {
        "protocol": protocol300 / "protocol.yaml",
        "verification": protocol300 / "baseline_protocol_300/verification.json",
        "table1": protocol300 / "metrics/protocol300/tables/table1.csv",
        "table2": protocol300 / "metrics/protocol300/tables/table2.csv",
        "table3": protocol300 / "metrics/protocol300/tables/table3.csv",
        "table_manifest": protocol300 / "metrics/protocol300/table_manifest.json",
        "fig4_page_manifest": (
            protocol300 / "figure_evidence_protocol300_v2/page_manifest.json"
        ),
        "fig4_comparison_manifest": (
            protocol300 / "figure_evidence_protocol300_v2/comparison_manifest.json"
        ),
    }
    for source_name, source_path in protocol300_sources.items():
        if not source_path.is_file():
            raise RuntimeError(
                f"Missing Protocol-300 source {source_name}: {source_path}"
            )

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
        "dyn19_mechanism_results": {
            "path": str(mechanism_results_path),
            "sha256": sha256(mechanism_results_path),
        },
        "dypho_table3": {
            "path": str(dypho_runtime_path),
            "sha256": sha256(dypho_runtime_path),
        },
        "protocol300": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in protocol300_sources.items()
        },
    }
    tracking_source_types = {
        row[0]: "external-report" for row in tracking_rows[1:] if "[ext]" in row[0]
    }
    tracking_source_types.update(
        {label: "local-rerun" for _, label in CONFIGS}
    )
    table_manifest = {
        "generated_date": "2026-09-17",
        "status": "draft-only; human review required",
        "comparison_mode": "presentation-flexible",
        "sources": sources,
        "tables": {
            "table1": {
                "purpose": "tracking accuracy on the four TUM scenes used by DyPho Table I",
                "source_types": tracking_source_types,
                "statistics": tracking_summary,
                "claim_boundary": (
                    "Every [ext] row is external positioning context, not a "
                    "protocol-matched local rerun. No cross-protocol rank is claimed."
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

    supplemental_manifest = {
        "generated_date": "2026-09-17",
        "status": "draft-only; human review required",
        "comparison_mode": "presentation-flexible",
        "sources": sources,
        "tables": {
            "table2": {
                "purpose": "ordered DYN-19 module ablation on TUM w/xyz and w/half",
                "source_type": "local-rerun",
                "statistics": ablation_summary,
                "claim_boundary": (
                    "Rows are ordered complete configurations, not a factorial estimate "
                    "of independent module effects or interactions."
                ),
            },
            "table3": {
                "purpose": "broad descriptive runtime comparison",
                "source_types": {
                    row[0]: ("external-report" if "[ext]" in row[0] else "local-rerun")
                    for row in broad_runtime_rows[1:]
                },
                "statistics": broad_runtime_summary,
                "claim_boundary": broad_runtime_summary["claim_boundary"],
            },
        },
        "generated_csvs": {
            "table2": {
                "path": str(supplemental_table2_path),
                "sha256": sha256(supplemental_table2_path),
            },
            "table3": {
                "path": str(supplemental_table3_path),
                "sha256": sha256(supplemental_table3_path),
            },
            "comparison_matrix": {
                "path": str(comparison_matrix_path),
                "sha256": sha256(comparison_matrix_path),
            },
        },
    }
    write_json(supplemental_output / "data/table_manifest.json", supplemental_manifest)

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

    protocol300_verification = json.loads(
        protocol300_sources["verification"].read_text(encoding="utf-8")
    )
    protocol300_page = json.loads(
        protocol300_sources["fig4_page_manifest"].read_text(encoding="utf-8")
    )

    checks = [
        expect(len(results) == 90, "main all_results contains 90 runs"),
        expect(all(item["status"] == "complete" for item in results), "all 90 runs complete"),
        expect(len(mapping_rows) == 72, "mapping evidence contains 72 cells"),
        expect(
            len(mechanism_results) == 120,
            "mechanism all_results contains 120 runs",
        ),
        expect(
            all(item["status"] == "complete" for item in mechanism_results),
            "all 120 mechanism runs complete",
        ),
        expect(
            {item["config"] for item in results} == {config for config, _ in CONFIGS},
            "main result configs match the three table rows",
        ),
        expect(
            sum(
                source_type == "external-report"
                for source_type in tracking_source_types.values()
            )
            == 11,
            "tracking table contains 11 visibly labeled external-report rows",
        ),
        expect(
            "DyPho" not in json.dumps(mapping_summary),
            "mapping table contains no fabricated DyPho numeric row",
        ),
        expect(len(ablation_rows) == 8, "ablation table contains seven configurations"),
        expect(
            len(broad_runtime_rows) == 13,
            "broad runtime table contains 11 external rows plus DYN-19 Full",
        ),
        expect(
            protocol300_verification["status"] == "complete"
            and protocol300_verification["expected_records"] == 16
            and len(protocol300_verification["records"]) == 16
            and all(
                record["status"] == "complete"
                for record in protocol300_verification["records"]
            ),
            "Protocol-300 contains 16 complete verified method-sequence cells",
        ),
        expect(
            protocol300_page["status"] == "complete"
            and protocol300_page["image_count"] == 20
            and protocol300_page["pending_occurrence_count"] == 0,
            "Protocol-300 Fig. 4 package contains 20 panels and zero pending entries",
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
    write_json(
        supplemental_output / "reports/source_validation.json",
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
