#!/usr/bin/env python3
"""Build provenance-bounded DyPho-style GDOR/DYN-19 source tables."""

from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "data"
OUTPUT = ROOT / "paper" / "dypho_style_gdor_dyn19" / "source_tables"

SEQUENCES = [
    "tum_walking_xyz",
    "tum_walking_halfsphere",
    "tum_walking_static",
    "tum_sitting_halfsphere",
    "bonn_balloon",
    "bonn_crowd",
    "bonn_crowd2",
    "bonn_crowd3",
    "bonn_person_tracking",
    "bonn_person_tracking2",
]

CONFIG_ORDER = [
    "dyn19_semantic",
    "dyn19_mapmatched",
    "dyn19_temporal_map",
    "dyn19_temporal_flow_map",
    "dyn19_temporal_flow_risk_map",
    "dyn19_full",
    "dyn19_full_no_mapping",
]

LABELS = {
    "dyn19_semantic": "Semantic",
    "dyn19_mapmatched": "MapMatched",
    "dyn19_temporal_map": "T+M",
    "dyn19_temporal_flow_map": "T+F+M",
    "dyn19_temporal_flow_risk_map": "T+F+R+M",
    "dyn19_full": "Full: T+F+R+A+M",
    "dyn19_full_no_mapping": "Full-NoM: T+F+R+A",
}

FACTORS = {
    "dyn19_semantic": (0, 0, 0, 0, 0),
    "dyn19_mapmatched": (0, 0, 0, 0, 1),
    "dyn19_temporal_map": (1, 0, 0, 0, 1),
    "dyn19_temporal_flow_map": (1, 1, 0, 0, 1),
    "dyn19_temporal_flow_risk_map": (1, 1, 1, 0, 1),
    "dyn19_full": (1, 1, 1, 1, 1),
    "dyn19_full_no_mapping": (1, 1, 1, 1, 0),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def keyed(rows: list[dict[str, str]], extra: tuple[str, ...] = ()) -> dict[tuple[str, ...], dict[str, str]]:
    keys = ("sequence", "seed") + extra
    return {tuple(row[key] for key in keys): row for row in rows}


def paired_stats(
    baseline: list[dict[str, str]],
    current: list[dict[str, str]],
    metric: str,
    *,
    higher_is_better: bool = False,
    extra_keys: tuple[str, ...] = (),
) -> dict[str, object]:
    baseline_by_key = keyed(baseline, extra_keys)
    current_by_key = keyed(current, extra_keys)
    keys = sorted(set(baseline_by_key) & set(current_by_key))
    gains: list[float] = []
    wins = ties = losses = 0
    for key in keys:
        base_value = number(baseline_by_key[key], metric)
        current_value = number(current_by_key[key], metric)
        gain = current_value - base_value if higher_is_better else base_value - current_value
        gains.append(gain)
        if gain > 1e-12:
            wins += 1
        elif gain < -1e-12:
            losses += 1
        else:
            ties += 1
    return {
        "count": len(keys),
        "baseline_mean": mean([number(baseline_by_key[key], metric) for key in keys]),
        "current_mean": mean([number(current_by_key[key], metric) for key in keys]),
        "median_gain": statistics.median(gains),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "baseline_by_key": baseline_by_key,
        "current_by_key": current_by_key,
    }


def certificate_summary(rows: list[dict[str, str]]) -> str:
    counts = Counter(row["certificate_status"] for row in rows)
    return "; ".join(f"{status}={counts[status]}" for status in sorted(counts))


def tracking_failure_context(
    baseline: list[dict[str, str]], current: list[dict[str, str]]
) -> str:
    baseline_by_sequence: dict[str, list[float]] = defaultdict(list)
    current_by_sequence: dict[str, list[float]] = defaultdict(list)
    for row in baseline:
        baseline_by_sequence[row["sequence"]].append(number(row, "ate_cm"))
    for row in current:
        current_by_sequence[row["sequence"]].append(number(row, "ate_cm"))
    sequence = max(baseline_by_sequence, key=lambda item: mean(baseline_by_sequence[item]))
    return (
        f"{sequence}: {fmt(mean(baseline_by_sequence[sequence]))} -> "
        f"{fmt(mean(current_by_sequence[sequence]))} cm"
    )


def mapping_failure_context(
    baseline: list[dict[str, str]], current: list[dict[str, str]]
) -> str:
    base_by_key = keyed(baseline, ("pose_mode",))
    current_by_key = keyed(current, ("pose_mode",))
    key = min(base_by_key, key=lambda item: number(base_by_key[item], "psnr_static_mean"))
    sequence, seed, _ = key
    return (
        f"{sequence}/seed{seed}: {fmt(number(base_by_key[key], 'psnr_static_mean'))} -> "
        f"{fmt(number(current_by_key[key], 'psnr_static_mean'))} dB"
    )


def build_tracking_table(main_rows: list[dict[str, str]]) -> None:
    by_config = {
        config: [row for row in main_rows if row["config"] == config]
        for config in ("dyn19_semantic", "dyn19_mapmatched", "dyn19_full")
    }
    fields = ["Method", "Source Type", "Cells", "Seeds"]
    for sequence in SEQUENCES:
        fields.extend([f"{sequence} ATE Mean cm", f"{sequence} ATE Std cm"])
    fields.extend(
        [
            "Overall Mean ATE cm",
            "Overall Median ATE cm",
            "Mean Failure Percent",
            "Route Certificates",
            "Comparison Context",
            "Failure Case Context",
            "Claim Boundary",
        ]
    )

    output_rows: list[dict[str, object]] = []
    for config in ("dyn19_semantic", "dyn19_mapmatched", "dyn19_full"):
        rows = by_config[config]
        output: dict[str, object] = {
            "Method": LABELS[config],
            "Source Type": "local-rerun",
            "Cells": len(rows),
            "Seeds": "0;1;2",
        }
        for sequence in SEQUENCES:
            values = [number(row, "ate_cm") for row in rows if row["sequence"] == sequence]
            output[f"{sequence} ATE Mean cm"] = fmt(mean(values))
            output[f"{sequence} ATE Std cm"] = fmt(sample_std(values))
        ate_values = [number(row, "ate_cm") for row in rows]
        output["Overall Mean ATE cm"] = fmt(mean(ate_values))
        output["Overall Median ATE cm"] = fmt(statistics.median(ate_values))
        output["Mean Failure Percent"] = fmt(
            mean([number(row, "failure_rate_percent") for row in rows])
        )
        output["Route Certificates"] = certificate_summary(rows)
        output["Claim Boundary"] = (
            "Three-seed full-sequence local DYN-19 tracking; no protocol-matched external ranking."
        )
        if config == "dyn19_semantic":
            output["Comparison Context"] = "Reference configuration."
            output["Failure Case Context"] = "Reference configuration."
        else:
            baseline = by_config["dyn19_semantic"]
            stats = paired_stats(baseline, rows, "ate_cm")
            context = (
                f"vs Semantic mean {fmt(stats['baseline_mean'])} -> {fmt(stats['current_mean'])} cm; "
                f"paired median gain {fmt(stats['median_gain'])} cm; "
                f"W/T/L {stats['wins']}/{stats['ties']}/{stats['losses']}"
            )
            failure = tracking_failure_context(baseline, rows)
            if config == "dyn19_full":
                map_stats = paired_stats(by_config["dyn19_mapmatched"], rows, "ate_cm")
                context += (
                    f"; vs MapMatched mean {fmt(map_stats['baseline_mean'])} -> "
                    f"{fmt(map_stats['current_mean'])} cm; paired median gain "
                    f"{fmt(map_stats['median_gain'])} cm; W/T/L "
                    f"{map_stats['wins']}/{map_stats['ties']}/{map_stats['losses']}"
                )
                failure += "; MapMatched failure: " + tracking_failure_context(
                    by_config["dyn19_mapmatched"], rows
                )
            output["Comparison Context"] = context
            output["Failure Case Context"] = failure
        output_rows.append(output)
    write_rows(OUTPUT / "table1_tracking.csv", fields, output_rows)


def build_mapping_table(
    mapping_rows: list[dict[str, str]], external_rows: list[dict[str, str]]
) -> None:
    fields = [
        "Pose Mode",
        "Method",
        "Source Type",
        "Cells",
        "Static PSNR Mean dB",
        "Static SSIM Mean",
        "Proxy Completeness Mean",
        "Proxy Ghost Risk Mean",
        "Native Mapping Evidence",
        "Protocol Match",
        "Paired Comparison Context",
        "Failure Cell Context",
        "Claim Boundary",
    ]
    output_rows: list[dict[str, object]] = []
    for pose_mode in ("online", "gt_aligned"):
        pose_rows = [row for row in mapping_rows if row["pose_mode"] == pose_mode]
        by_config = {
            config: [row for row in pose_rows if row["config"] == config]
            for config in ("dyn19_semantic", "dyn19_mapmatched", "dyn19_full")
        }
        for config in ("dyn19_semantic", "dyn19_mapmatched", "dyn19_full"):
            rows = by_config[config]
            output: dict[str, object] = {
                "Pose Mode": pose_mode,
                "Method": LABELS[config],
                "Source Type": "local-rerun-diagnostic",
                "Cells": len(rows),
                "Static PSNR Mean dB": fmt(mean([number(row, "psnr_static_mean") for row in rows])),
                "Static SSIM Mean": fmt(mean([number(row, "ssim_static_mean") for row in rows])),
                "Proxy Completeness Mean": fmt(
                    mean([number(row, "background_completeness_at_tau") for row in rows])
                ),
                "Proxy Ghost Risk Mean": fmt(
                    mean([number(row, "ghost_risk_proxy_at_tau_high") for row in rows])
                ),
                "Native Mapping Evidence": "DYN-19 held-out static-region render metrics",
                "Protocol Match": "local DYN-19 cell",
                "Claim Boundary": (
                    "Held-out static-region image-fidelity diagnostic. Completeness and ghost risk are proxies, not independent truth."
                ),
            }
            if config == "dyn19_semantic":
                output["Paired Comparison Context"] = "Reference configuration."
                output["Failure Cell Context"] = "Reference configuration."
            else:
                baseline = by_config["dyn19_semantic"]
                stats = paired_stats(
                    baseline,
                    rows,
                    "psnr_static_mean",
                    higher_is_better=True,
                    extra_keys=("pose_mode",),
                )
                context = (
                    f"vs Semantic mean gain {fmt(stats['current_mean'] - stats['baseline_mean'])} dB; "
                    f"paired median gain {fmt(stats['median_gain'])} dB; "
                    f"W/T/L {stats['wins']}/{stats['ties']}/{stats['losses']}"
                )
                failure = mapping_failure_context(baseline, rows)
                if config == "dyn19_full":
                    map_stats = paired_stats(
                        by_config["dyn19_mapmatched"],
                        rows,
                        "psnr_static_mean",
                        higher_is_better=True,
                        extra_keys=("pose_mode",),
                    )
                    context += (
                        f"; vs MapMatched mean gain "
                        f"{fmt(map_stats['current_mean'] - map_stats['baseline_mean'])} dB; "
                        f"paired median gain {fmt(map_stats['median_gain'])} dB; "
                        f"W/T/L {map_stats['wins']}/{map_stats['ties']}/{map_stats['losses']}"
                    )
                    failure += "; MapMatched failure: " + mapping_failure_context(
                        by_config["dyn19_mapmatched"], rows
                    )
                output["Paired Comparison Context"] = context
                output["Failure Cell Context"] = failure
            output_rows.append(output)
    for row in external_rows:
        if row["task"] != "mapping":
            continue
        output_rows.append(
            {
                "Pose Mode": "native external protocol",
                "Method": row["method"],
                "Source Type": "external-report",
                "Cells": "",
                "Static PSNR Mean dB": "",
                "Static SSIM Mean": "",
                "Proxy Completeness Mean": "",
                "Proxy Ghost Risk Mean": "",
                "Native Mapping Evidence": (
                    f"{row['citation']} {row['source_table']}; {row['dataset']}; "
                    f"{row['metric']}: {row['reported_values']}"
                ),
                "Protocol Match": "false",
                "Paired Comparison Context": "Not pooled with DYN-19 image-fidelity cells.",
                "Failure Cell Context": "Not available under a shared DYN-19 view protocol.",
                "Claim Boundary": row["claim_boundary"],
            }
        )
    write_rows(OUTPUT / "table2_mapping.csv", fields, output_rows)


def combined_config_rows(
    main_rows: list[dict[str, str]], mechanism_rows: list[dict[str, str]]
) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for config in CONFIG_ORDER:
        source = main_rows if config in {"dyn19_semantic", "dyn19_mapmatched", "dyn19_full"} else mechanism_rows
        result[config] = [row for row in source if row["config"] == config]
    return result


def build_runtime_table(config_rows: dict[str, list[dict[str, str]]]) -> None:
    fields = [
        "Method",
        "Source Type",
        "Runs",
        "Sequences",
        "Seeds",
        "Mean End-to-End Seconds",
        "Median End-to-End Seconds",
        "Min End-to-End Seconds",
        "Max End-to-End Seconds",
        "Mean Failure Percent",
        "GPU",
        "Hardware Consistency",
        "Claim Boundary",
    ]
    output_rows: list[dict[str, object]] = []
    for config in CONFIG_ORDER:
        rows = config_rows[config]
        runtimes = [number(row, "end_to_end_seconds") for row in rows]
        output_rows.append(
            {
                "Method": LABELS[config],
                "Source Type": "local-rerun",
                "Runs": len(rows),
                "Sequences": len({row["sequence"] for row in rows}),
                "Seeds": "0;1;2",
                "Mean End-to-End Seconds": fmt(mean(runtimes)),
                "Median End-to-End Seconds": fmt(statistics.median(runtimes)),
                "Min End-to-End Seconds": fmt(min(runtimes)),
                "Max End-to-End Seconds": fmt(max(runtimes)),
                "Mean Failure Percent": fmt(
                    mean([number(row, "failure_rate_percent") for row in rows])
                ),
                "GPU": "NVIDIA GeForce RTX 4090 (CUDA device 1)",
                "Hardware Consistency": "clean-single-gpu",
                "Claim Boundary": (
                    "DYN-19 end-to-end wall time on full sequences; not directly comparable to published per-stage runtime rows."
                ),
            }
        )
    write_rows(OUTPUT / "table3_runtime.csv", fields, output_rows)


def build_ordered_ablation(config_rows: dict[str, list[dict[str, str]]]) -> None:
    fields = [
        "Order",
        "Configuration",
        "T",
        "F",
        "R",
        "A",
        "M",
        "Cells",
        "Mean ATE cm",
        "Median ATE cm",
        "Adjacent Paired Median Gain cm",
        "Adjacent W/T/L",
        "Route Certificates",
        "Design Boundary",
    ]
    output_rows: list[dict[str, object]] = []
    previous: list[dict[str, str]] | None = None
    for order, config in enumerate(CONFIG_ORDER, start=1):
        rows = config_rows[config]
        ate_values = [number(row, "ate_cm") for row in rows]
        factors = FACTORS[config]
        if previous is None:
            adjacent_median = ""
            adjacent_wtl = ""
        else:
            stats = paired_stats(previous, rows, "ate_cm")
            adjacent_median = fmt(stats["median_gain"])
            adjacent_wtl = f"{stats['wins']}/{stats['ties']}/{stats['losses']}"
        output_rows.append(
            {
                "Order": order,
                "Configuration": LABELS[config],
                "T": factors[0],
                "F": factors[1],
                "R": factors[2],
                "A": factors[3],
                "M": factors[4],
                "Cells": len(rows),
                "Mean ATE cm": fmt(mean(ate_values)),
                "Median ATE cm": fmt(statistics.median(ate_values)),
                "Adjacent Paired Median Gain cm": adjacent_median,
                "Adjacent W/T/L": adjacent_wtl,
                "Route Certificates": certificate_summary(rows),
                "Design Boundary": (
                    "Ordered ablation only; adjacent bundle contrasts do not identify independent effects or interactions."
                ),
            }
        )
        previous = rows
    write_rows(OUTPUT / "ordered_ablation.csv", fields, output_rows)


def main() -> None:
    main_rows = read_rows(DATA / "dyn19_main_90_cells.csv")
    mechanism_rows = read_rows(DATA / "dyn19_mechanism_120_cells.csv")
    mapping_rows = read_rows(DATA / "dyn19_mapping_72_cells.csv")
    external_rows = read_rows(DATA / "external_report_quantitative.csv")
    if len(main_rows) != 90 or len(mechanism_rows) != 120 or len(mapping_rows) != 72:
        raise RuntimeError("DYN-19 source-cell counts do not match the frozen 90/120/72 contract")
    config_rows = combined_config_rows(main_rows, mechanism_rows)
    build_tracking_table(main_rows)
    build_mapping_table(mapping_rows, external_rows)
    build_runtime_table(config_rows)
    build_ordered_ablation(config_rows)


if __name__ == "__main__":
    main()
