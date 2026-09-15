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


def load_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values) -> float:
    return float(np.mean(list(values)))


def require_text(text: str, value: str, failures: list[str]) -> None:
    if value not in text:
        failures.append(f"missing manuscript value/text: {value}")


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
    dyn18_by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in dyn18:
        dyn18_by_config[row["config"]].append(float(row["ate_cm"]))
        dyn18_by_pair[(row["config"], row["sequence"])].append(float(row["ate_cm"]))
    dyn18_semantic = mean(dyn18_by_config["semantic"])
    dyn18_gdor = mean(dyn18_by_config["dypho_flow_guarded"])

    dyn17 = load_csv("dyn17_mapmatched_control.csv")
    dyn17_matched = mean(float(row["mapmatched_no_track_reuse_ate_cm"]) for row in dyn17)
    dyn17_gdor = mean(float(row["guarded_gdor_ate_cm"]) for row in dyn17)

    dyn16 = load_csv("dyn16_commonview_mapping.csv")
    online = [row for row in dyn16 if row["pose_mode"] == "online"]
    mapping: dict[str, dict[str, float]] = {}
    for method in ["Semantic", "Guarded/GDOR", "Strict"]:
        rows = [row for row in online if row["method"] == method]
        mapping[method] = {
            "psnr": mean(float(row["psnr_static"]) for row in rows),
            "ssim": mean(float(row["ssim_static"]) for row in rows),
            "lpips": mean(float(row["lpips_static"]) for row in rows),
        }

    expected_strings = [
        f"{dyn15_all7_semantic:.3f}",
        f"{dyn15_all7_gdor:.3f}",
        f"{dyn18_semantic:.3f}",
        f"{dyn18_gdor:.3f}",
        f"{dyn17_matched:.3f}",
        f"{dyn17_gdor:.3f}",
        f"{mapping['Semantic']['psnr']:.3f}",
        f"{mapping['Guarded/GDOR']['psnr']:.3f}",
        f"{mapping['Guarded/GDOR']['ssim']:.3f}",
        f"{mapping['Guarded/GDOR']['lpips']:.3f}",
        "limited preservation diagnostic",
        "7/0/0",
        "9/0/3",
        "3/0/0",
        "0.864 cm median sequence improvement",
        "frame that contains recovered tracking support does not receive a persistent-map admission privilege",
        "does not yet establish superiority over the official DyPho-SLAM implementation",
    ]
    for value in expected_strings:
        require_text(manuscript, value, failures)

    figure_count = len(re.findall(r"\\begin\{figure\*?\}", manuscript))
    table_count = len(re.findall(r"\\begin\{table\*?\}", manuscript))
    if figure_count != 5:
        failures.append(f"expected 5 figures, found {figure_count}")
    if table_count != 3:
        failures.append(f"expected 3 tables, found {table_count}")

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
            "dyn16_online_static_region": mapping,
        },
        "claim_boundary": {
            "official_dypho_superiority": False,
            "multi_seed_mapping_superiority": False,
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
