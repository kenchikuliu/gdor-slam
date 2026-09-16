#!/usr/bin/env python3
"""Validate the standalone DyPho-style comparison package."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PARENT_DATA = ROOT.parent / "data"
REPORT = ROOT / "reports" / "dypho_comparison_validation.json"


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def rounded_mean(rows: list[dict[str, str]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def main() -> None:
    failures: list[str] = []
    tracking = read_csv("dypho_style_tracking_comparison.csv")
    tracking_source = read_csv("dyn19_tracking_sequence_means.csv")
    with (PARENT_DATA / "dyn18_tum4_cells.csv").open(newline="", encoding="utf-8") as stream:
        dyn18_source = list(csv.DictReader(stream))
    mapping = read_csv("dypho_style_mapping_comparison.csv")
    mapping_source = read_csv("dyn19_mapping_cells.csv")
    external = read_csv("external_quantitative_reference.csv")

    if len(tracking) != 4:
        failures.append(f"expected 4 tracking rows, found {len(tracking)}")
    if len(mapping) != 6:
        failures.append(f"expected 6 mapping rows, found {len(mapping)}")

    tracking_configs = {
        "semantic_dyn19_ate_cm": "dyn19_semantic",
        "mapmatched_dyn19_ate_cm": "dyn19_mapmatched",
        "full_dyn19_ate_cm": "dyn19_full",
    }
    for row in tracking:
        for output_field, config in tracking_configs.items():
            source_rows = [
                source
                for source in tracking_source
                if source["sequence"] == row["sequence"]
                and source["config"] == config
            ]
            if len(source_rows) != 1:
                failures.append(
                    f"missing unique DYN-19 source row for {row['sequence']} / {config}"
                )
                continue
            if abs(float(row[output_field]) - float(source_rows[0]["ate_rmse_cm"])) > 0.00006:
                failures.append(
                    f"tracking mismatch for {row['sequence']} / {output_field}"
                )

        for output_field, config in (
            ("guarded_dyn18_ate_cm", "dypho_flow_guarded"),
            ("strict_dyn18_ate_cm", "dypho_flow_camera_guided_strict"),
        ):
            source_rows = [
                source
                for source in dyn18_source
                if source["sequence"] == row["sequence"]
                and source["config"] == config
            ]
            if len(source_rows) != 3:
                failures.append(
                    f"expected 3 DYN-18 cells for {row['sequence']} / {config}, "
                    f"found {len(source_rows)}"
                )
                continue
            expected = rounded_mean(source_rows, "ate_cm")
            if abs(float(row[output_field]) - expected) > 0.00006:
                failures.append(
                    f"DYN-18 mismatch for {row['sequence']} / {output_field}"
                )

    mapping_labels = {
        "Semantic": "Semantic",
        "MapMatched": "MapMatched",
        "Full": "T+F+R+A+M (Full)",
    }
    mapping_fields = {
        "psnr_full_db": "psnr_full_mean",
        "psnr_static_db": "psnr_static_mean",
        "ssim_full": "ssim_full_mean",
        "ssim_static": "ssim_static_mean",
    }
    for row in mapping:
        source_rows = [
            source
            for source in mapping_source
            if source["pose_mode"] == row["pose_mode"]
            and source["config_label"] == mapping_labels[row["configuration"]]
        ]
        if len(source_rows) != 12:
            failures.append(
                f"expected 12 mapping cells for {row['pose_mode']} / {row['configuration']}, "
                f"found {len(source_rows)}"
            )
            continue
        for output_field, source_field in mapping_fields.items():
            expected = rounded_mean(source_rows, source_field)
            if abs(float(row[output_field]) - expected) > 0.001:
                failures.append(
                    f"mapping mismatch for {row['pose_mode']} / {row['configuration']} / "
                    f"{output_field}"
                )
        if row["external_dypho_mapping_numeric"] != "not reported":
            failures.append("DyPho mapping status must remain 'not reported'")

    dypho_tracking = [
        row for row in external if row["method"] == "DyPho-SLAM" and row["task"] == "tracking"
    ]
    dypho_mapping = [
        row for row in external if row["method"] == "DyPho-SLAM" and row["task"] == "mapping"
    ]
    if len(dypho_tracking) != 1:
        failures.append("expected one DyPho tracking reference row")
    else:
        published = [
            float(value.strip())
            for value in dypho_tracking[0]["reported_values"].split("(", 1)[0].split(";")
        ]
        local_rows = {
            row["sequence"]: float(row["dypho_published_ate_cm"])
            for row in tracking
        }
        ordered_local = [
            local_rows["tum_walking_xyz"],
            local_rows["tum_walking_halfsphere"],
            local_rows["tum_walking_static"],
            local_rows["tum_sitting_halfsphere"],
        ]
        if published != ordered_local:
            failures.append("DyPho published ATE values do not match the external ledger")
    if len(dypho_mapping) != 1 or dypho_mapping[0]["reported_values"] != "no PSNR SSIM or LPIPS mapping table reported":
        failures.append("DyPho mapping reference must remain qualitative-only")

    report = {
        "passed": not failures,
        "tracking_rows": len(tracking),
        "mapping_rows": len(mapping),
        "external_reference_rows": len(external),
        "failures": failures,
        "claim_boundary": {
            "dypho_tracking": "published external reference; not protocol-matched",
            "dypho_mapping": "numeric mapping values not reported",
            "local_mapping": "common-view PSNR/SSIM diagnostic",
        },
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"validated {len(tracking)} tracking rows and {len(mapping)} mapping rows")


if __name__ == "__main__":
    main()
