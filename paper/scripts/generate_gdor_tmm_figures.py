#!/usr/bin/env python3
"""Generate evidence-linked GDOR-SLAM TMM overview and method figures."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
SOURCES = ROOT / "figure_sources" / "fig1"


CONFIG_LABELS = {
    "semantic": "Semantic",
    "dypho_flow_guarded": "GDOR",
    "dypho_flow_camera_guided_strict": "Strict",
}

COLORS = {
    "Semantic": "#6b7280",
    "GDOR": "#15803d",
    "Strict": "#c2410c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dyn18_means() -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    with (DATA / "dyn18_tum4_cells.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            grouped[(row["config"], row["sequence"])].append(float(row["ate_cm"]))

    result: dict[str, dict[str, float]] = defaultdict(dict)
    for (config, sequence), values in grouped.items():
        result[CONFIG_LABELS[config]][sequence] = float(np.mean(values))
    return dict(result)


def load_dyn17_mean() -> tuple[float, float, float]:
    semantic, matched, gdor = [], [], []
    with (DATA / "dyn17_mapmatched_control.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            semantic.append(float(row["semantic_ate_cm"]))
            matched.append(float(row["mapmatched_no_track_reuse_ate_cm"]))
            gdor.append(float(row["guarded_gdor_ate_cm"]))
    return float(np.mean(semantic)), float(np.mean(matched)), float(np.mean(gdor))


def load_dyn15_all7() -> tuple[float, float]:
    values: dict[str, list[float]] = defaultdict(list)
    with (DATA / "dyn15_tracking_per_sequence.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            values[row["config_label"]].append(float(row["ate_rmse_cm"]))
    return float(np.mean(values["Semantic"])), float(np.mean(values["Guarded/GDOR"]))


def add_box(ax, xy, width, height, title, body, edge, face="#ffffff"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=2.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(x + width / 2, y + height - 0.055, title, ha="center", va="top",
            fontsize=15, fontweight="bold", color=edge)
    ax.text(x + width / 2, y + height / 2 - 0.02, body, ha="center", va="center",
            fontsize=11.5, color="#1f2937", linespacing=1.25)
    return box


def add_arrow(ax, start, end, color="#4b5563", width=1.8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                 linewidth=width, color=color))


def generate_overview() -> list[Path]:
    means = load_dyn18_means()
    dyn15_semantic, dyn15_gdor = load_dyn15_all7()
    dyn17_semantic, dyn17_matched, dyn17_gdor = load_dyn17_mean()

    sequence_order = [
        "tum_walking_xyz",
        "tum_walking_halfsphere",
        "tum_walking_static",
        "tum_sitting_halfsphere",
    ]
    sequence_labels = ["Walking xyz", "Walking half", "Walking static", "Sitting half"]

    fig = plt.figure(figsize=(15.8, 8.9), constrained_layout=False)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.25], hspace=0.26, wspace=0.28)
    fig.suptitle(
        "GDOR-SLAM: guarded tracking support without persistent dynamic writes",
        fontsize=23,
        fontweight="bold",
        y=0.985,
    )

    for index, name in enumerate(["input_t1.png", "input_t2.png", "input_t3.png"]):
        ax = fig.add_subplot(grid[0, index * 2:(index + 1) * 2])
        ax.imshow(Image.open(SOURCES / name).convert("RGB"))
        ax.set_title(["Dynamic occlusion", "Partial background exposure", "Large masked region"][index],
                     fontsize=13, fontweight="bold")
        ax.axis("off")

    ax_bar = fig.add_subplot(grid[1, :4])
    x = np.arange(len(sequence_order))
    width = 0.24
    for offset, method in zip([-width, 0.0, width], ["Semantic", "GDOR", "Strict"]):
        values = [means[method][sequence] for sequence in sequence_order]
        bars = ax_bar.bar(x + offset, values, width=width, color=COLORS[method], label=method)
        for bar, value in zip(bars, values):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, value * 1.08, f"{value:.2f}",
                        ha="center", va="bottom", fontsize=8.5, rotation=0)
    ax_bar.set_yscale("log")
    ax_bar.set_ylim(0.55, 55)
    ax_bar.set_ylabel("ATE RMSE [cm], log scale (lower is better)", fontsize=11)
    ax_bar.set_xticks(x, sequence_labels)
    ax_bar.grid(axis="y", which="both", alpha=0.24)
    ax_bar.legend(frameon=False, ncol=3, loc="upper left")
    ax_bar.set_title("Frozen DYN-18 TUM4 validation: three seeds, full sequences", fontsize=14,
                     fontweight="bold")

    ax_text = fig.add_subplot(grid[1, 4:])
    ax_text.axis("off")
    tum4_semantic = float(np.mean([means["Semantic"][s] for s in sequence_order]))
    tum4_gdor = float(np.mean([means["GDOR"][s] for s in sequence_order]))
    dyn15_gain = 100.0 * (dyn15_semantic - dyn15_gdor) / dyn15_semantic
    tum4_gain = 100.0 * (tum4_semantic - tum4_gdor) / tum4_semantic
    matched_gain = 100.0 * (dyn17_matched - dyn17_gdor) / dyn17_matched
    summary = (
        "Evidence summary\n\n"
        f"DYN-15 All7 (63 runs)\n"
        f"{dyn15_semantic:.3f} -> {dyn15_gdor:.3f} cm  (-{dyn15_gain:.1f}%)\n\n"
        f"DYN-18 frozen TUM4 (36 cells)\n"
        f"{tum4_semantic:.3f} -> {tum4_gdor:.3f} cm  (-{tum4_gain:.1f}%)\n\n"
        f"DYN-17 matched persistent weight\n"
        f"Semantic {dyn17_semantic:.3f} | matched {dyn17_matched:.3f}\n"
        f"GDOR {dyn17_gdor:.3f} cm  ({matched_gain:.1f}% below matched)\n\n"
        "Recovered observations may support tracking,\n"
        "but never bypass persistent Gaussian admission."
    )
    ax_text.text(
        0.02,
        0.98,
        summary,
        transform=ax_text.transAxes,
        ha="left",
        va="top",
        fontsize=12.2,
        linespacing=1.28,
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.5),
    )

    fig.text(
        0.5,
        0.012,
        "All values are local same-source comparisons. Published DyPho-SLAM values remain external-report context.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )

    outputs = [FIGURES / "fig1_overview.png", FIGURES / "fig1_overview.pdf"]
    fig.savefig(outputs[0], dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def generate_method() -> list[Path]:
    fig, ax = plt.subplots(figsize=(15.8, 7.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("GDOR-SLAM: guarded recovery with separate tracking support and mapping weight",
                 fontsize=23, fontweight="bold", pad=15)

    add_box(ax, (0.025, 0.60), 0.19, 0.25, "Excluded evidence",
            "RGB-D frame\nSemantic dynamic mask\nConservative flow mask", "#475569", "#f8fafc")
    add_box(ax, (0.265, 0.60), 0.22, 0.25, "Support qualification",
            "Temporal depth consistency\nResidual-flow consistency\nTracking-risk state", "#b45309", "#fff7ed")
    add_box(ax, (0.535, 0.60), 0.20, 0.25, "Tracking support",
            "Recover selected pixels\nFeature replenishment\nTransient pose support", "#15803d", "#f0fdf4")
    add_box(ax, (0.785, 0.60), 0.19, 0.25, "Camera estimate",
            "ORB/RGB-D tracking\nCurrent-frame pose\nNo dynamic state persistence", "#1d4ed8", "#eff6ff")

    add_arrow(ax, (0.215, 0.725), (0.265, 0.725))
    add_arrow(ax, (0.485, 0.725), (0.535, 0.725))
    add_arrow(ax, (0.735, 0.725), (0.785, 0.725))

    add_box(ax, (0.16, 0.17), 0.30, 0.24, "Persistent mapping route",
            "Uses the conservative map mask\nRejects uncertain ORB/RGB-D insertion\nNo recovered dynamic-region write", "#475569", "#f8fafc")
    add_box(ax, (0.56, 0.17), 0.29, 0.24, "Matched causal control",
            "Same persistent mapping weight\nNo temporal recovery\nNo adaptive tracking reuse", "#7c3aed", "#faf5ff")

    add_arrow(ax, (0.12, 0.60), (0.28, 0.41), color="#64748b")
    add_arrow(ax, (0.635, 0.60), (0.70, 0.41), color="#7c3aed")
    add_arrow(ax, (0.46, 0.29), (0.56, 0.29), color="#7c3aed")

    ax.text(0.5, 0.045,
            "Core contract: tracking support does not automatically grant persistent Gaussian map admission.",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#334155")

    outputs = [FIGURES / "fig2_method_schematic.png", FIGURES / "fig2_method_schematic.pdf"]
    fig.savefig(outputs[0], dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = generate_overview() + generate_method()
    inputs = sorted(DATA.glob("*.csv")) + sorted(SOURCES.glob("*.png"))
    manifest = {
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
        "claim_boundary": (
            "Figures summarize local same-source GDOR evidence. They do not establish "
            "superiority over the official DyPho-SLAM implementation."
        ),
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "generated_figure_provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
