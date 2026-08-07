#!/usr/bin/env python3
"""Aggregate the six predeclared development-only recovery-v2 cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL = "development-only-recovery-v2-20260730"
REQUIRED_CELLS = {
    (sequence, seed)
    for sequence in ("tum_walking_xyz", "bonn_crowd")
    for seed in (0, 1, 2)
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(paths: list[Path]) -> dict[str, Any]:
    cells: dict[tuple[str, int], dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for path in paths:
        result = json.loads(path.read_text())
        if result.get("protocol") != PROTOCOL:
            raise ValueError(f"Unexpected protocol in {path}")
        key = (result["sequence"], int(result["seed"]))
        if key in cells:
            raise ValueError(f"Duplicate sequence/seed cell: {key}")
        cells[key] = result
        source_hashes[str(path.resolve())] = sha256(path)
    if set(cells) != REQUIRED_CELLS:
        raise ValueError(
            f"Expected cells {sorted(REQUIRED_CELLS)}, got {sorted(cells)}"
        )

    minimum_events = min(cell["events"] for cell in cells.values())
    sufficient = minimum_events >= 3
    sham_valid = all(
        cell["matched_sham_exact_static"] for cell in cells.values()
    )
    translation_cells = sum(
        bool(cell["oracle_strict_translation_improvement"])
        for cell in cells.values()
    )
    rotation_cells = sum(
        bool(cell["oracle_mean_rotation_non_regression"])
        for cell in cells.values()
    )
    potential_gate_pass = (
        sufficient and sham_valid and translation_cells == 6 and rotation_cells == 6
    )
    if not sham_valid:
        verdict = "invalid_matched_sham"
    elif not sufficient:
        verdict = "insufficient_data"
    elif potential_gate_pass:
        verdict = "oracle_potential_retained"
    else:
        verdict = "scientific_no_go"

    table = []
    for key in sorted(cells):
        cell = cells[key]
        methods = cell["methods"]
        table.append(
            {
                "sequence": key[0],
                "seed": key[1],
                "events": cell["events"],
                "oracle_dynamic_selections": cell[
                    "oracle_dynamic_selections"
                ],
                "static_translation_mean_m": methods["static"][
                    "translation_rpe_m"
                ]["mean"],
                "dynamic_translation_mean_m": methods["dynamic"][
                    "translation_rpe_m"
                ]["mean"],
                "oracle_translation_mean_m": methods["oracle"][
                    "translation_rpe_m"
                ]["mean"],
                "static_rotation_mean_rad": methods["static"][
                    "rotation_rpe_rad"
                ]["mean"],
                "oracle_rotation_mean_rad": methods["oracle"][
                    "rotation_rpe_rad"
                ]["mean"],
                "oracle_translation_improved": cell[
                    "oracle_strict_translation_improvement"
                ],
                "oracle_rotation_non_regression": cell[
                    "oracle_mean_rotation_non_regression"
                ],
            }
        )
    return {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "decision_rule": (
            "At least three frozen non-overlapping recovery events per cell; "
            "matched-sham exactly equals Static; Oracle-Delayed-H strictly "
            "improves mean translation RPE and does not regress mean rotation "
            "in all six sequence-by-seed cells."
        ),
        "minimum_events_per_cell": minimum_events,
        "sufficient_data": sufficient,
        "matched_sham_valid": sham_valid,
        "oracle_translation_pass_cells": translation_cells,
        "oracle_rotation_pass_cells": rotation_cells,
        "oracle_potential_gate_pass": potential_gate_pass,
        "cells": table,
        "source_summary_sha256": source_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.summaries)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"recovery v2 aggregation failed: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
