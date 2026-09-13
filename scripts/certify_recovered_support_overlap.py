#!/usr/bin/env python3
"""Certify the recovered-tracking-support versus mapping-support invariant.

The certificate is deliberately per run. A zero leak without recovered support
is safe but vacuous; it is never promoted to evidence that recovery occurred.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "frame",
    "tracking_recovery_audit_pixels",
    "tracking_recovery_mapping_leak_pixels",
    "mapping_weight_present",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def nonnegative_int(value: str | int | float, label: str) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value!r}") from exc
    if numeric < 0:
        raise ValueError(f"{label} is negative: {numeric}")
    return numeric


def validate_plan_binding(manifest: dict[str, Any]) -> list[str]:
    binding = manifest.get("plan_binding")
    if not isinstance(binding, dict):
        return ["manifest has no plan binding"]
    if binding.get("contract") != "benchmark-plan-binding-v1":
        return ["manifest has an invalid plan-binding contract"]
    plan_path = Path(str(binding.get("path", "")))
    if not plan_path.is_file():
        return [f"frozen plan is missing: {plan_path}"]
    recorded = binding.get("sha256")
    actual = sha256(plan_path)
    if not isinstance(recorded, str) or actual != recorded:
        return ["frozen plan SHA-256 does not match the manifest binding"]
    return []


def certify(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "run_summary.json"
    frame_metrics_path = run_dir / "frame_metrics.csv"
    for path in (manifest_path, summary_path, frame_metrics_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required run artifact is missing: {path}")

    manifest = load_object(manifest_path)
    summary = load_object(summary_path)
    provenance_errors = validate_plan_binding(manifest)

    with frame_metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("frame_metrics.csv has no header")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(
                "frame_metrics.csv lacks certificate columns: "
                + ", ".join(missing))
        rows = list(reader)
    if not rows:
        raise ValueError("frame_metrics.csv has no data rows")

    seen_frames: set[int] = set()
    audit_pixels = 0
    leak_pixels = 0
    recovered_frames = 0
    missing_mapping_weight_frames: list[int] = []
    mapping_weight_frames = 0
    for row in rows:
        frame = nonnegative_int(row["frame"], "frame")
        if frame in seen_frames:
            raise ValueError(f"frame_metrics.csv has duplicate frame {frame}")
        seen_frames.add(frame)
        audit = nonnegative_int(
            row["tracking_recovery_audit_pixels"],
            f"frame {frame} audit pixels")
        leak = nonnegative_int(
            row["tracking_recovery_mapping_leak_pixels"],
            f"frame {frame} leak pixels")
        mapping_weight_present = nonnegative_int(
            row["mapping_weight_present"],
            f"frame {frame} mapping_weight_present")
        if mapping_weight_present not in (0, 1):
            raise ValueError(
                f"frame {frame} mapping_weight_present must be 0 or 1")
        audit_pixels += audit
        leak_pixels += leak
        mapping_weight_frames += mapping_weight_present
        if audit > 0:
            recovered_frames += 1
            if mapping_weight_present == 0:
                missing_mapping_weight_frames.append(frame)

    summary_audit = nonnegative_int(
        summary.get("tracking_recovery_audit_pixels", -1),
        "run summary audit pixels")
    summary_leak = nonnegative_int(
        summary.get("tracking_recovery_mapping_leak_pixels", -1),
        "run summary leak pixels")
    summary_mapping_frames = nonnegative_int(
        summary.get("mapping_weight_frames", -1),
        "run summary mapping-weight frames")
    accounting_errors = []
    if summary_audit != audit_pixels:
        accounting_errors.append(
            "run summary audit pixels do not equal frame_metrics.csv")
    if summary_leak != leak_pixels:
        accounting_errors.append(
            "run summary leak pixels do not equal frame_metrics.csv")
    if summary_mapping_frames != mapping_weight_frames:
        accounting_errors.append(
            "run summary mapping-weight frames do not equal frame_metrics.csv")

    recovered_support_observed = audit_pixels > 0
    zero_overlap = leak_pixels == 0
    mapping_support_present = (
        recovered_support_observed and not missing_mapping_weight_frames)
    failures = [
        *provenance_errors,
        *accounting_errors,
    ]
    if leak_pixels > 0:
        failures.append("recovered tracking support overlaps mapping support")
    if missing_mapping_weight_frames:
        failures.append(
            "recovered tracking support occurred without mapping weight on "
            f"frames {missing_mapping_weight_frames}")

    if failures:
        status = "fail"
    elif recovered_support_observed and mapping_support_present:
        status = "pass"
    else:
        status = "vacuous"

    return {
        "contract": "recovered-support-overlap-certificate-v1",
        "status": status,
        "run_dir": str(run_dir),
        "identity": {
            "sequence": manifest.get("sequence"),
            "config": manifest.get("config"),
            "seed": manifest.get("seed"),
        },
        "evidence": {
            "frames": len(rows),
            "recovered_support_observed": recovered_support_observed,
            "recovered_support_frames": recovered_frames,
            "tracking_recovery_audit_pixels": audit_pixels,
            "tracking_recovery_mapping_leak_pixels": leak_pixels,
            "zero_overlap": zero_overlap,
            "mapping_weight_frames": mapping_weight_frames,
            "mapping_support_present_for_recovered_frames":
                mapping_support_present,
            "recovered_frames_without_mapping_weight":
                missing_mapping_weight_frames,
        },
        "accounting": {
            "summary_audit_pixels": summary_audit,
            "summary_leak_pixels": summary_leak,
            "summary_mapping_weight_frames": summary_mapping_frames,
            "errors": accounting_errors,
        },
        "provenance": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "run_summary": str(summary_path),
            "run_summary_sha256": sha256(summary_path),
            "frame_metrics": str(frame_metrics_path),
            "frame_metrics_sha256": sha256(frame_metrics_path),
            "plan_binding": manifest.get("plan_binding"),
            "errors": provenance_errors,
        },
        "failures": failures,
        "interpretation": {
            "pass": (
                "Recovered tracking support was observed, mapping support was "
                "present, and no recovered support overlapped map admission."),
            "vacuous": (
                "No recovered tracking support was observed; zero overlap is "
                "a safety observation, not positive recovery evidence."),
            "fail": (
                "A leak, missing mapping weight for recovered support, stale "
                "provenance, or counter-accounting mismatch was observed."),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to RUN_DIR/recovered_support_overlap_certificate.json",
    )
    args = parser.parse_args()

    certificate = certify(args.run_dir)
    output = args.output or (
        args.run_dir / "recovered_support_overlap_certificate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
