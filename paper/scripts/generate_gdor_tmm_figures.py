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
    "Matched": "#7c3aed",
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


def load_dyn19_episode() -> list[dict[str, str]]:
    with (DATA / "dyn19_failure_recovery_episode.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


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


def parse_intervals(value: str) -> list[tuple[int, int]]:
    if not value or value == "none":
        return []
    intervals = []
    for item in value.split(";"):
        start, end = item.split("-")
        intervals.append((int(start), int(end)))
    return intervals


def generate_failure_recovery_control() -> list[Path]:
    rows = load_dyn19_episode()
    labels = [row["label"] for row in rows]
    frames = max(int(row["frames"]) for row in rows)
    y_pos = np.arange(len(labels))[::-1]

    fig = plt.figure(figsize=(15.8, 8.8), constrained_layout=False)
    grid = fig.add_gridspec(3, 1, height_ratios=[0.95, 1.15, 1.0], hspace=0.58)
    fig.suptitle(
        "Failure interval, recovered tracking support, and matched control",
        fontsize=20,
        fontweight="bold",
        y=0.975,
    )

    ax_fail = fig.add_subplot(grid[0, 0])
    for row, y in zip(rows, y_pos):
        label = row["label"]
        intervals = parse_intervals(row["lost_intervals"])
        if intervals:
            for start, end in intervals:
                ax_fail.broken_barh(
                    [(start, end - start + 1)],
                    (y - 0.28, 0.56),
                    facecolors=COLORS[label],
                    alpha=0.92,
                )
        else:
            ax_fail.plot([0, frames], [y, y], color=COLORS[label], linewidth=3.0)
        ax_fail.text(
            frames + 14,
            y,
            f"{int(row['lost_frames'])} lost frames",
            va="center",
            fontsize=10.5,
            color="#334155",
        )
    ax_fail.set_xlim(0, frames + 170)
    ax_fail.set_yticks(y_pos, labels)
    ax_fail.set_xlabel("Frame index")
    ax_fail.set_title("Per-frame failure intervals on TUM sitting_halfsphere, seed 0",
                      fontsize=13.5, fontweight="bold")
    ax_fail.grid(axis="x", alpha=0.2)
    ax_fail.spines[["top", "right"]].set_visible(False)

    ax_support = fig.add_subplot(grid[1, 0])
    x = np.arange(len(rows))
    width = 0.24
    temporal_mpx = [
        float(row["temporal_recovered_static_pixels"]) / 1_000_000.0
        for row in rows
    ]
    audit_mpx = [
        float(row["tracking_recovery_audit_pixels"]) / 1_000_000.0
        for row in rows
    ]
    extracted = [float(row["extracted_features_mean"]) / 100.0 for row in rows]
    ax_support.bar(x - width, temporal_mpx, width, color="#86efac",
                   label="Temporal recovered support [Mpx]")
    ax_support.bar(x, audit_mpx, width, color="#15803d",
                   label="Audited recovered support [Mpx]")
    ax_support.bar(x + width, extracted, width, color="#94a3b8",
                   label="Extracted features mean / 100")
    for index, row in enumerate(rows):
        label_y = max(temporal_mpx[index], audit_mpx[index], extracted[index]) + 1.2
        ax_support.text(
            index,
            label_y,
            f"{int(row['recovered_support_frames'])} recovered frames\n"
            f"{int(row['adaptive_feature_active_frames'])} adaptive frames",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#334155",
        )
    ax_support.set_xticks(x, labels)
    ax_support.set_ylim(0, max(max(temporal_mpx), max(audit_mpx), max(extracted)) + 7.0)
    ax_support.set_ylabel("Count scale")
    ax_support.set_title("Recovered support and feature replenishment are tracking-side signals",
                         fontsize=13.5, fontweight="bold")
    ax_support.grid(axis="y", alpha=0.22)
    ax_support.legend(frameon=False, ncol=3, loc="upper left")
    ax_support.spines[["top", "right"]].set_visible(False)

    ax_control = fig.add_subplot(grid[2, 0])
    ate = [float(row["ate_cm"]) for row in rows]
    bars = ax_control.bar(labels, ate, color=[COLORS[label] for label in labels], alpha=0.92)
    for bar, row in zip(bars, rows):
        height = bar.get_height()
        if height > 8.0:
            y = height - 1.2
            va = "top"
            text_color = "white"
        else:
            y = height + 1.0
            va = "bottom"
            text_color = "#334155"
        ax_control.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{float(row['ate_cm']):.2f} cm\n"
            f"fail {float(row['failure_rate_percent']):.1f}%\n"
            f"cert {row['certificate_status']}",
            ha="center",
            va=va,
            fontsize=10,
            color=text_color,
        )
    ax_control.set_ylim(0, max(ate) * 1.28)
    ax_control.set_ylabel("ATE RMSE [cm]")
    ax_control.set_title("Matched control keeps the persistent mapping route but disables recovery",
                         fontsize=13.5, fontweight="bold")
    ax_control.grid(axis="y", alpha=0.22)
    ax_control.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.006,
        "The zero-overlap certificate is a route-invariant diagnostic, not independent ghost-contamination ground truth.",
        ha="center",
        fontsize=10.5,
        color="#475569",
    )
    outputs = [
        FIGURES / "fig5_failure_recovery_control.png",
        FIGURES / "fig5_failure_recovery_control.pdf",
    ]
    fig.savefig(outputs[0], dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = generate_overview() + generate_method() + generate_failure_recovery_control()
    inputs = sorted(DATA.glob("*.csv")) + sorted(SOURCES.glob("*.png"))
    manifest = {
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
        "claim_boundary": (
            "Figures summarize local same-source GDOR evidence. They do not establish "
            "superiority over the official DyPho-SLAM implementation, zero ghost "
            "contamination, or mapping superiority."
        ),
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "generated_figure_provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
