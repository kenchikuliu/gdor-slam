#!/usr/bin/env python3
"""Freeze baseline-only recovery episodes from a Static replay scan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL = "development-only-recovery-v2-20260730"
REQUIRED_COLUMNS = {
    "checkpoint",
    "static_scan_only",
    "static_executed",
    "dynamic_executed",
    "prior_subspace",
    "max_static_information_leverage",
    "static_information_leverage_mode",
    "replay_mode",
    "horizon",
    "start_timestamp",
    "end_timestamp",
    "static_success",
    "static_first_inliers",
    "static_min_inliers",
    "static_final_inliers",
}
MANIFEST_FIELDS = (
    "event_id",
    "checkpoint",
    "horizon",
    "start_timestamp",
    "end_timestamp",
    "static_first_inliers",
    "static_min_inliers",
    "static_final_inliers",
    "selection_rule",
    "static_scan_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_static_scan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Static scan is missing columns: " + ", ".join(sorted(missing))
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Static scan contains no replayable rows")
    for row in rows:
        if row["static_scan_only"] != "1":
            raise ValueError("Episode selection requires --static-scan-only")
        if row["static_executed"] != "1" or row["dynamic_executed"] != "0":
            raise ValueError("Static scan executed a non-Static branch")
        if (
            row["prior_subspace"] != "translation"
            or row["replay_mode"] != "posterior-local-map-v1"
            or row["static_information_leverage_mode"]
            != "normalize-to-target"
            or abs(float(row["max_static_information_leverage"]) - 0.025)
            > 1e-8
        ):
            raise ValueError("Static scan does not match the frozen v2 method")
    return rows


def freeze_episodes(
    rows: list[dict[str, str]],
    sequence: str,
    seed: int,
    horizon: int = 10,
    max_support_inliers: int = 150,
    scan_sha256: str = "",
) -> list[dict[str, Any]]:
    if not sequence or seed < 0 or horizon <= 0 or max_support_inliers < 0:
        raise ValueError("Invalid episode selection arguments")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if int(row["horizon"]) != horizon or row["static_success"] != "1":
            continue
        entry = int(row["static_first_inliers"])
        minimum = int(row["static_min_inliers"])
        final = int(row["static_final_inliers"])
        start = float(row["start_timestamp"])
        end = float(row["end_timestamp"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("Static scan contains an invalid event interval")
        if minimum <= max_support_inliers:
            candidates.append(
                {
                    "checkpoint": row["checkpoint"],
                    "horizon": horizon,
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "static_first_inliers": entry,
                    "static_min_inliers": minimum,
                    "static_final_inliers": final,
                }
            )
    candidates.sort(key=lambda row: (row["start_timestamp"], row["checkpoint"]))

    selected: list[dict[str, Any]] = []
    previous_end = -math.inf
    for candidate in candidates:
        if candidate["start_timestamp"] < previous_end - 1e-9:
            continue
        selected.append(candidate)
        previous_end = candidate["end_timestamp"]

    rule = (
        f"static_horizon_min_inliers_le_{max_support_inliers};"
        f"static_h{horizon}_success;greedy_nonoverlap"
    )
    for index, event in enumerate(selected, start=1):
        event["event_id"] = f"{sequence}_seed{seed}_e{index:03d}"
        event["selection_rule"] = rule
        event["static_scan_sha256"] = scan_sha256
    return selected


def write_manifest(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(events)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("static_scan", type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--max-support-inliers", type=int, default=150)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    try:
        scan_hash = sha256(args.static_scan)
        rows = load_static_scan(args.static_scan)
        events = freeze_episodes(
            rows,
            sequence=args.sequence,
            seed=args.seed,
            horizon=args.horizon,
            max_support_inliers=args.max_support_inliers,
            scan_sha256=scan_hash,
        )
        write_manifest(args.output_csv, events)
        metadata = {
            "protocol": PROTOCOL,
            "sequence": args.sequence,
            "seed": args.seed,
            "horizon": args.horizon,
            "max_support_inliers": args.max_support_inliers,
            "selection_inputs": "Static branch only; Dynamic and GT unavailable",
            "static_scan": str(args.static_scan.resolve()),
            "static_scan_sha256": scan_hash,
            "event_manifest": str(args.output_csv.resolve()),
            "event_manifest_sha256": sha256(args.output_csv),
            "candidate_rows": len(rows),
            "selected_events": len(events),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"recovery episode preparation failed: {error}")
        return 1
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
