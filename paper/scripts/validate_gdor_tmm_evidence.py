#!/usr/bin/env python3
"""Validate GDOR-SLAM TMM numbers, asset counts, and claim boundaries."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DYPHO = ROOT / "dypho_style_gdor_dyn19"


def load_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values) -> float:
    return float(np.mean(list(values)))


def median(values) -> float:
    return float(np.median(list(values)))


def wtl(deltas: list[float]) -> tuple[int, int, int]:
    wins = sum(delta > 1e-12 for delta in deltas)
    losses = sum(delta < -1e-12 for delta in deltas)
    return wins, len(deltas) - wins - losses, losses


def require_text(text: str, value: str, failures: list[str]) -> None:
    if value not in text:
        failures.append(f"missing manuscript value/text: {value}")


def require_equal(actual, expected, label: str, failures: list[str]) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected!r}, found {actual!r}")


def grouped_ate(rows: list[dict[str, str]]) -> dict[tuple[str, str, int], float]:
    return {
        (row["config"], row["sequence"], int(row["seed"])): float(row["ate_cm"])
        for row in rows
    }


def config_stats(
    cells: dict[tuple[str, str, int], float], config: str
) -> dict[str, object]:
    values = [value for (name, _, _), value in cells.items() if name == config]
    sequences = sorted({sequence for name, sequence, _ in cells if name == config})
    sequence_means = {
        sequence: mean(
            value
            for (name, seq, _), value in cells.items()
            if name == config and seq == sequence
        )
        for sequence in sequences
    }
    return {
        "mean_cm": mean(values),
        "median_cell_cm": median(values),
        "sequence_means_cm": sequence_means,
    }


def compare_configs(
    cells: dict[tuple[str, str, int], float], baseline: str, method: str
) -> dict[str, object]:
    keys = sorted((sequence, seed) for name, sequence, seed in cells if name == baseline)
    deltas = [cells[(baseline, sequence, seed)] - cells[(method, sequence, seed)] for sequence, seed in keys]
    baseline_stats = config_stats(cells, baseline)
    method_stats = config_stats(cells, method)
    sequence_deltas = [
        baseline_stats["sequence_means_cm"][sequence]
        - method_stats["sequence_means_cm"][sequence]
        for sequence in sorted(baseline_stats["sequence_means_cm"])
    ]
    return {
        "baseline_mean_cm": baseline_stats["mean_cm"],
        "method_mean_cm": method_stats["mean_cm"],
        "mean_reduction_percent": 100.0
        * (baseline_stats["mean_cm"] - method_stats["mean_cm"])
        / baseline_stats["mean_cm"],
        "paired_cell_median_gain_cm": median(deltas),
        "cell_wtl": wtl(deltas),
        "sequence_mean_median_gain_cm": median(sequence_deltas),
        "sequence_wtl": wtl(sequence_deltas),
    }


def main() -> None:
    manuscript_files = [ROOT / "main.tex"] + sorted((ROOT / "sections").glob("*.tex"))
    manuscript = "\n".join(path.read_text(encoding="utf-8") for path in manuscript_files)
    failures: list[str] = []

    dyn15 = load_csv("dyn15_tracking_per_sequence.csv")
    dyn15_by_config: dict[str, list[float]] = defaultdict(list)
    for row in dyn15:
        dyn15_by_config[row["config_label"]].append(float(row["ate_rmse_cm"]))
    dyn15_all7_semantic = mean(dyn15_by_config["Semantic"])
    dyn15_all7_gdor = mean(dyn15_by_config["Guarded/GDOR"])

    dyn18 = load_csv("dyn18_tum4_cells.csv")
    dyn18_by_config: dict[str, list[float]] = defaultdict(list)
    for row in dyn18:
        dyn18_by_config[row["config"]].append(float(row["ate_cm"]))
    dyn18_semantic = mean(dyn18_by_config["semantic"])
    dyn18_gdor = mean(dyn18_by_config["dypho_flow_guarded"])

    dyn17 = load_csv("dyn17_mapmatched_control.csv")
    dyn17_matched = mean(float(row["mapmatched_no_track_reuse_ate_cm"]) for row in dyn17)
    dyn17_gdor = mean(float(row["guarded_gdor_ate_cm"]) for row in dyn17)

    dyn16 = load_csv("dyn16_commonview_mapping.csv")
    dyn16_online = [row for row in dyn16 if row["pose_mode"] == "online"]
    dyn16_mapping: dict[str, dict[str, float]] = {}
    for method in ["Semantic", "Guarded/GDOR", "Strict"]:
        rows = [row for row in dyn16_online if row["method"] == method]
        dyn16_mapping[method] = {
            "psnr": mean(float(row["psnr_static"]) for row in rows),
            "ssim": mean(float(row["ssim_static"]) for row in rows),
            "lpips": mean(float(row["lpips_static"]) for row in rows),
        }

    external_reports = load_csv("external_report_quantitative.csv")
    require_equal(len(external_reports), 6, "external-report row count", failures)
    require_equal(
        sorted({row["method"] for row in external_reports}),
        ["DG-SLAM", "DyPho-SLAM", "DynaSLAM"],
        "external-report methods",
        failures,
    )
    if any(row["local_rerun"] != "false" for row in external_reports):
        failures.append("external-report row incorrectly marked as a local rerun")
    if any(row["protocol_match"] != "false" for row in external_reports):
        failures.append("external-report row incorrectly marked protocol-matched")
    expected_external_values = {
        ("tracking", "DynaSLAM"): "1.5;2.5;0.6;1.7 (ten-run medians)",
        ("tracking", "DG-SLAM"): "1.6;0.6 (paper-reported subset)",
        ("tracking", "DyPho-SLAM"): "1.6;2.6;0.6;1.6 (paper reports Std separately)",
        ("mapping", "DG-SLAM"): "8.06;15.46;43.67% (reported averages)",
        ("mapping", "DynaSLAM"): "no numeric mapping-quality metric reported",
        ("mapping", "DyPho-SLAM"): "no PSNR SSIM or LPIPS mapping table reported",
    }
    require_equal(
        {(row["task"], row["method"]): row["reported_values"] for row in external_reports},
        expected_external_values,
        "external-report registered values",
        failures,
    )

    dyn19_main = load_csv("dyn19_main_90_cells.csv")
    require_equal(len(dyn19_main), 90, "DYN-19 main row count", failures)
    require_equal(
        sorted({row["config"] for row in dyn19_main}),
        ["dyn19_full", "dyn19_mapmatched", "dyn19_semantic"],
        "DYN-19 main configs",
        failures,
    )
    require_equal(sorted({int(row["seed"]) for row in dyn19_main}), [0, 1, 2], "DYN-19 seeds", failures)
    require_equal(len({row["sequence"] for row in dyn19_main}), 10, "DYN-19 sequence count", failures)
    if "tum_sitting_halfsphere" not in {row["sequence"] for row in dyn19_main}:
        failures.append("DYN-19 main denominator omits tum_sitting_halfsphere")
    if any(row["status"] != "complete" for row in dyn19_main):
        failures.append("DYN-19 main contains a non-complete row")
    if any(row["source_commit"] != "14c6b2ca696a7eacda8f193e5c8a0a44e55d527e" for row in dyn19_main):
        failures.append("DYN-19 main contains a mismatched source commit")

    dyn19_cells = grouped_ate(dyn19_main)
    dyn19_semantic = config_stats(dyn19_cells, "dyn19_semantic")
    dyn19_matched = config_stats(dyn19_cells, "dyn19_mapmatched")
    dyn19_full = config_stats(dyn19_cells, "dyn19_full")
    dyn19_full_vs_semantic = compare_configs(dyn19_cells, "dyn19_semantic", "dyn19_full")
    dyn19_full_vs_matched = compare_configs(dyn19_cells, "dyn19_mapmatched", "dyn19_full")

    full_rows = [row for row in dyn19_main if row["config"] == "dyn19_full"]
    require_equal(len(full_rows), 30, "DYN-19 Full row count", failures)
    if any(row["certificate_status"] != "pass" for row in full_rows):
        failures.append("DYN-19 Full contains a non-passing route certificate")
    full_recovered_frames = sum(int(row["recovered_support_frames"]) for row in full_rows)
    full_audit_pixels = sum(int(row["tracking_recovery_audit_pixels"]) for row in full_rows)
    full_leak_pixels = sum(
        int(row["tracking_recovery_mapping_leak_pixels"]) for row in full_rows
    )
    require_equal(full_leak_pixels, 0, "DYN-19 Full overlap count", failures)

    dyn19_mechanism = load_csv("dyn19_mechanism_120_cells.csv")
    require_equal(len(dyn19_mechanism), 120, "DYN-19 mechanism row count", failures)
    mechanism_configs = [
        "dyn19_full_no_mapping",
        "dyn19_temporal_flow_map",
        "dyn19_temporal_flow_risk_map",
        "dyn19_temporal_map",
    ]
    require_equal(
        sorted({row["config"] for row in dyn19_mechanism}),
        mechanism_configs,
        "DYN-19 mechanism configs",
        failures,
    )
    if any(row["status"] != "complete" for row in dyn19_mechanism):
        failures.append("DYN-19 mechanism contains a non-complete row")
    positive_mechanisms = [
        row for row in dyn19_mechanism if row["config"] != "dyn19_full_no_mapping"
    ]
    if any(row["certificate_status"] != "pass" for row in positive_mechanisms):
        failures.append("DYN-19 positive mechanism row has a non-passing certificate")
    no_mapping_rows = [
        row for row in dyn19_mechanism if row["config"] == "dyn19_full_no_mapping"
    ]
    if any(row["certificate_status"] != "fail" for row in no_mapping_rows):
        failures.append("DYN-19 Full-NoM does not contain 30 expected certificate failures")
    no_mapping_leak_pixels = sum(
        int(row["tracking_recovery_mapping_leak_pixels"]) for row in no_mapping_rows
    )

    ordered_rows = dyn19_main + dyn19_mechanism
    ordered_cells = grouped_ate(ordered_rows)
    ordered_configs = [
        "dyn19_semantic",
        "dyn19_mapmatched",
        "dyn19_temporal_map",
        "dyn19_temporal_flow_map",
        "dyn19_temporal_flow_risk_map",
        "dyn19_full",
        "dyn19_full_no_mapping",
    ]
    ordered_summary = {config: config_stats(ordered_cells, config) for config in ordered_configs}
    ordered_adjacent = {
        f"{baseline}->{method}": compare_configs(ordered_cells, baseline, method)
        for baseline, method in zip(ordered_configs, ordered_configs[1:])
    }

    dyn19_mapping = load_csv("dyn19_mapping_72_cells.csv")
    require_equal(len(dyn19_mapping), 72, "DYN-19 mapping row count", failures)
    require_equal(
        sorted({row["pose_mode"] for row in dyn19_mapping}),
        ["gt_aligned", "online"],
        "DYN-19 mapping pose modes",
        failures,
    )
    require_equal(len({row["sequence"] for row in dyn19_mapping}), 4, "DYN-19 mapping scene count", failures)

    mapping_aggregates: dict[str, dict[str, float]] = {}
    for pose_mode in ["online", "gt_aligned"]:
        for config in ["dyn19_semantic", "dyn19_mapmatched", "dyn19_full"]:
            rows = [
                row
                for row in dyn19_mapping
                if row["pose_mode"] == pose_mode and row["config"] == config
            ]
            require_equal(len(rows), 12, f"DYN-19 mapping {pose_mode}/{config} cells", failures)
            mapping_aggregates[f"{config}/{pose_mode}"] = {
                "psnr_static": mean(float(row["psnr_static_mean"]) for row in rows),
                "ssim_static": mean(float(row["ssim_static_mean"]) for row in rows),
                "proxy_completeness": mean(
                    float(row["background_completeness_at_tau"]) for row in rows
                ),
                "proxy_ghost_risk": mean(
                    float(row["ghost_risk_proxy_at_tau_high"]) for row in rows
                ),
            }

    online = [row for row in dyn19_mapping if row["pose_mode"] == "online"]
    online_by_key = {
        (row["config"], row["sequence"], int(row["seed"])): float(row["psnr_static_mean"])
        for row in online
    }
    map_comparisons = {}
    for baseline in ["dyn19_semantic", "dyn19_mapmatched"]:
        keys = sorted((sequence, seed) for config, sequence, seed in online_by_key if config == baseline)
        deltas = [
            online_by_key[("dyn19_full", sequence, seed)]
            - online_by_key[(baseline, sequence, seed)]
            for sequence, seed in keys
        ]
        map_comparisons[baseline] = {
            "mean_gain_db": mean(deltas),
            "median_gain_db": median(deltas),
            "wtl": wtl(deltas),
        }

    expected_strings = [
        f"{dyn15_all7_semantic:.3f}",
        f"{dyn15_all7_gdor:.3f}",
        f"{dyn18_semantic:.3f}",
        f"{dyn18_gdor:.3f}",
        f"{dyn19_semantic['mean_cm']:.3f}",
        f"{dyn19_matched['mean_cm']:.3f}",
        f"{dyn19_full['mean_cm']:.3f}",
        f"{dyn19_full_vs_semantic['paired_cell_median_gain_cm']:.3f} cm",
        "19/0/11",
        f"{dyn19_full_vs_matched['paired_cell_median_gain_cm']:.3f} cm",
        "16/0/14",
        "4/0/6",
        "42.500 cm to 11.080 cm",
        "36.775 cm to 4.864 cm",
        f"{full_audit_pixels:,}",
        f"{no_mapping_leak_pixels:,}",
        f"{mapping_aggregates['dyn19_full/online']['psnr_static']:.3f}",
        f"{map_comparisons['dyn19_semantic']['mean_gain_db']:.3f} dB",
        f"{map_comparisons['dyn19_semantic']['median_gain_db']:.3f} dB",
        "11/0/1",
        "ordered ablation",
        "limited preservation diagnostic",
        "frame that contains recovered tracking support does not receive a persistent-map admission privilege",
        "do not identify independent component effects or interactions",
        "remain pending",
        "DynaSLAM$^\\dagger$",
        "DG-SLAM$^\\dagger$",
        "DyPho-SLAM$^\\dagger$",
        "Acc. 8.06 cm; Comp. 15.46 cm; Comp.$@5$ cm 43.67\\%",
        "no PSNR/SSIM/LPIPS table",
        "not pooled with local results",
    ]
    for value in expected_strings:
        require_text(manuscript, value, failures)

    unsafe_patterns = {
        "zero ghost proof": r"(?i)(prove[sd]?|proof of)\s+zero[- ]ghost",
        "zero contamination proof": r"(?i)(prove[sd]?|proof of)\s+zero[- ]contamination",
        "ghost-free claim": r"(?i)\bghost[- ]free\b",
    }
    for label, pattern in unsafe_patterns.items():
        if re.search(pattern, manuscript):
            failures.append(f"unsafe claim wording found: {label}")

    figure_count = len(re.findall(r"\\begin\{figure\*?\}", manuscript))
    table_count = len(re.findall(r"\\begin\{table\*?\}", manuscript))
    require_equal(figure_count, 5, "figure count", failures)
    require_equal(table_count, 3, "table count", failures)

    figure_files = [
        ROOT / "figures" / "fig1_overview.png",
        ROOT / "figures" / "fig2_method_schematic.png",
        ROOT / "figures" / "fig3_trajectory_examples.png",
        ROOT / "figures" / "fig4_commonview_mapping_audit.png",
        ROOT / "figures" / "fig5_failure_recovery_control.png",
    ]
    for path in figure_files:
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing or empty figure: {path.relative_to(ROOT)}")

    dypho_source_counts = {
        "table1_tracking.csv": 3,
        "table2_mapping.csv": 9,
        "table3_runtime.csv": 7,
        "ordered_ablation.csv": 7,
    }
    for name, expected_count in dypho_source_counts.items():
        path = DYPHO / "source_tables" / name
        if not path.is_file():
            failures.append(f"missing DyPho-style source table: {path.relative_to(ROOT)}")
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            row_count = sum(1 for _ in csv.DictReader(stream))
        require_equal(row_count, expected_count, f"DyPho-style {name} row count", failures)

    dypho_self_check_path = DYPHO / "mvp_package" / "reports" / "self_check.json"
    if not dypho_self_check_path.is_file():
        failures.append("missing DyPho-style MVP self-check")
        dypho_self_check = {}
    else:
        dypho_self_check = json.loads(dypho_self_check_path.read_text(encoding="utf-8"))
        require_equal(dypho_self_check.get("passed"), True, "DyPho-style MVP self-check", failures)
        require_equal(
            dypho_self_check.get("release_allowed"),
            False,
            "DyPho-style MVP release gate",
            failures,
        )
        require_equal(
            dypho_self_check.get("summary", {}).get("missing_or_unverified_artifact_count"),
            4,
            "DyPho-style explicit missing artifact count",
            failures,
        )

    dypho_audits = {}
    for table_id in ("table1", "table2", "table3"):
        draft = DYPHO / "table_drafts" / "draft" / f"{table_id}.png"
        audit_path = DYPHO / "table_drafts" / "reports" / f"{table_id}.audit.json"
        if not draft.is_file() or draft.stat().st_size == 0:
            failures.append(f"missing or empty DyPho-style draft: {draft.relative_to(ROOT)}")
        if not audit_path.is_file():
            failures.append(f"missing DyPho-style audit: {audit_path.relative_to(ROOT)}")
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        dypho_audits[table_id] = audit
        require_equal(audit.get("passed"), True, f"DyPho-style {table_id} audit", failures)
        require_equal(
            audit.get("release_allowed"),
            False,
            f"DyPho-style {table_id} release gate",
            failures,
        )

    missing_artifacts_path = DYPHO / "mvp_package" / "reports" / "missing_artifacts.csv"
    if not missing_artifacts_path.is_file():
        failures.append("missing DyPho-style missing-artifacts report")
        missing_dypho_artifacts = []
    else:
        with missing_artifacts_path.open(newline="", encoding="utf-8") as stream:
            missing_dypho_artifacts = [
                row for row in csv.DictReader(stream) if row["status"] != "present"
            ]
        require_equal(
            sorted(row["requirement"] for row in missing_dypho_artifacts),
            sorted(
                [
                    "artifact_sources.figure_panels.fig4.fr3_w_xyz__ours",
                    "artifact_sources.figure_panels.fig4.fr3_w_half__ours",
                    "artifact_sources.figure_panels.fig4.bonn_ps_track__ours",
                    "artifact_sources.figure_panels.fig4.bonn_r3__ours",
                ]
            ),
            "DyPho-style Fig.4 blockers",
            failures,
        )

    provenance = json.loads((DATA / "dyn19_release_provenance.json").read_text(encoding="utf-8"))
    require_equal(
        provenance["metadata_archive"]["sha256"],
        "925d3d80ad0b9c586875a267bec8170bc3b1a580b19a1a05ad64780bd5ec2fb2",
        "DYN-19 metadata archive hash",
        failures,
    )

    report = {
        "passed": not failures,
        "failures": failures,
        "figure_count": figure_count,
        "table_count": table_count,
        "computed": {
            "dyn15_all7_semantic_cm": dyn15_all7_semantic,
            "dyn15_all7_gdor_cm": dyn15_all7_gdor,
            "dyn18_tum4_semantic_cm": dyn18_semantic,
            "dyn18_tum4_gdor_cm": dyn18_gdor,
            "dyn17_matched_cm": dyn17_matched,
            "dyn17_gdor_cm": dyn17_gdor,
            "dyn16_online_static_region": dyn16_mapping,
            "external_reports": external_reports,
            "dyn19_main": {
                "semantic": dyn19_semantic,
                "mapmatched": dyn19_matched,
                "full": dyn19_full,
                "full_vs_semantic": dyn19_full_vs_semantic,
                "full_vs_mapmatched": dyn19_full_vs_matched,
            },
            "dyn19_route_certificate": {
                "full_cells": len(full_rows),
                "recovered_support_frames": full_recovered_frames,
                "audit_pixels": full_audit_pixels,
                "mapping_leak_pixels": full_leak_pixels,
                "full_no_mapping_failed_cells": len(no_mapping_rows),
                "full_no_mapping_overlap_pixels": no_mapping_leak_pixels,
            },
            "dyn19_ordered_ablation": {
                "configs": ordered_summary,
                "adjacent_contrasts": ordered_adjacent,
            },
            "dyn19_mapping": {
                "aggregates": mapping_aggregates,
                "online_static_psnr_comparisons": map_comparisons,
            },
            "dypho_style_support": {
                "source_table_rows": dypho_source_counts,
                "mvp_self_check": dypho_self_check,
                "table_audits": dypho_audits,
                "fig4_missing_artifacts": missing_dypho_artifacts,
            },
        },
        "claim_boundary": {
            "official_external_superiority": False,
            "external_report_rows_present": True,
            "external_report_protocol_matched": False,
            "factorial_or_interaction_effects": False,
            "multi_seed_mapping_diagnostic": True,
            "independent_ghost_or_completeness_truth": False,
            "zero_ghost_or_contamination_proof": False,
        },
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "evidence_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
