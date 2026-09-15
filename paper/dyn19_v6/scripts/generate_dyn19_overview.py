#!/usr/bin/env python3
"""Generate the DYN-19 evidence overview from the public CSV package."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

COLORS = {
    "Semantic": "#6b7280",
    "MapMatched": "#2563eb",
    "T+M": "#0891b2",
    "T+F+M": "#d97706",
    "T+F+R+M": "#7c3aed",
    "Full": "#15803d",
    "Full-NoM": "#b91c1c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    tracking = load_csv("dyn19_tracking_sequence_means.csv")
    mapping = load_csv("dyn19_mapping_cells.csv")
    by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in tracking:
        by_config[row["config"]].append(row)

    sequence_order = [
        "bonn_balloon",
        "bonn_crowd",
        "bonn_crowd2",
        "bonn_crowd3",
        "bonn_person_tracking",
        "bonn_person_tracking2",
        "tum_sitting_halfsphere",
        "tum_walking_halfsphere",
        "tum_walking_static",
        "tum_walking_xyz",
    ]
    sequence_labels = [
        "Balloon",
        "Crowd",
        "Crowd2",
        "Crowd3",
        "Person",
        "Person2",
        "Sit-half",
        "Walk-half",
        "Walk-static",
        "Walk-xyz",
    ]
    lookup = {(row["config"], row["sequence"]): row for row in tracking}

    fig = plt.figure(figsize=(15.8, 7.8), constrained_layout=False)
    grid = fig.add_gridspec(2, 5, height_ratios=[1.08, 1.0], hspace=0.42, wspace=0.48)
    fig.suptitle(
        "DYN-19: guarded recovery is aggregate-positive, route-audited, and sequence-dependent",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )

    ax_seq = fig.add_subplot(grid[0, :])
    x = np.arange(len(sequence_order))
    width = 0.25
    for offset, config, label in [
        (-width, "dyn19_semantic", "Semantic"),
        (0.0, "dyn19_mapmatched", "MapMatched"),
        (width, "dyn19_full", "Full"),
    ]:
        values = [float(lookup[(config, sequence)]["ate_rmse_cm"]) for sequence in sequence_order]
        ax_seq.bar(x + offset, values, width=width, color=COLORS[label], label=label)
    ax_seq.set_yscale("log")
    ax_seq.set_ylim(0.55, 60)
    ax_seq.set_ylabel("ATE RMSE [cm], log scale")
    ax_seq.set_xticks(x, sequence_labels)
    ax_seq.grid(axis="y", which="both", alpha=0.22)
    ax_seq.legend(frameon=False, ncol=3, loc="upper right")
    ax_seq.set_title("Ten predeclared sequences, three seeds each (Full beats Semantic on 7/10 means)", fontsize=12.5)

    ax_ablation = fig.add_subplot(grid[1, :3])
    configs = [
        ("dyn19_semantic", "Semantic"),
        ("dyn19_mapmatched", "MapMatched"),
        ("dyn19_temporal_map", "T+M"),
        ("dyn19_temporal_flow_map", "T+F+M"),
        ("dyn19_temporal_flow_risk_map", "T+F+R+M"),
        ("dyn19_full", "Full"),
        ("dyn19_full_no_mapping", "Full-NoM"),
    ]
    values = [mean(by_config[config], "ate_rmse_cm") for config, _ in configs]
    bars = ax_ablation.bar(
        np.arange(len(configs)),
        values,
        color=[COLORS[label] for _, label in configs],
    )
    for bar, value in zip(bars, values):
        ax_ablation.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.2,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax_ablation.set_ylim(0, 11.5)
    ax_ablation.set_ylabel("Ten-sequence mean ATE [cm]")
    ax_ablation.set_xticks(np.arange(len(configs)), [label for _, label in configs], rotation=22, ha="right")
    ax_ablation.grid(axis="y", alpha=0.22)
    ax_ablation.set_title("Ordered configurations (not factorial component effects)", fontsize=12.5)

    ax_map = fig.add_subplot(grid[1, 3:])
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in mapping:
        grouped[(row["pose_mode"], row["config"])].append(row)
    map_labels = ["Online\nSemantic", "Online\nFull", "GT-aligned\nSemantic", "GT-aligned\nFull"]
    map_keys = [
        ("online", "dyn19_semantic"),
        ("online", "dyn19_full"),
        ("gt_aligned", "dyn19_semantic"),
        ("gt_aligned", "dyn19_full"),
    ]
    map_values = [mean(grouped[key], "psnr_static_mean") for key in map_keys]
    map_colors = [COLORS["Semantic"], COLORS["Full"], COLORS["Semantic"], COLORS["Full"]]
    bars = ax_map.bar(np.arange(4), map_values, color=map_colors)
    for bar, value in zip(bars, map_values):
        ax_map.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.25,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    ax_map.set_ylim(0, 22)
    ax_map.set_ylabel("Static-region PSNR [dB]")
    ax_map.set_xticks(np.arange(4), map_labels)
    ax_map.grid(axis="y", alpha=0.22)
    ax_map.set_title("Four scenes x three seeds x common views", fontsize=12.5)

    fig.text(
        0.5,
        0.008,
        "Full: 30/30 observed-recovery zero-overlap passes. Full-NoM: 30/30 expected overlap failures. T+M remains a consumed-data candidate.",
        ha="center",
        fontsize=10.5,
        color="#374151",
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = [FIGURES / "fig1_dyn19_overview.png", FIGURES / "fig1_dyn19_overview.pdf"]
    fig.savefig(outputs[0], dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)

    provenance = {
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "inputs": {
            name: sha256(DATA / name)
            for name in ("dyn19_tracking_sequence_means.csv", "dyn19_mapping_cells.csv")
        },
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
        "claim_boundary": (
            "DYN-19 is an ordered ablation. T+M is not yet a held-out final method, "
            "and mapping values are common-view diagnostics rather than universal superiority."
        ),
    }
    (ROOT / "reports" / "generated_figure_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
