#!/usr/bin/env python3
"""Export path-sanitized DYN-19 evidence from a verified results release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


REVISION = Path(__file__).resolve().parents[1]
DATA = REVISION / "data"
EXPECTED_RELEASE_ID = "DYN19_RESULTS_20260914_R1"
EXPECTED_EXPERIMENT_ID = "DYN-19_FULL_CAUSAL_MAP_INTEGRITY_20260913"
EXPECTED_SOURCE_COMMIT = "14c6b2ca696a7eacda8f193e5c8a0a44e55d527e"

CONFIG_LABELS = {
    "dyn19_semantic": "Semantic",
    "dyn19_mapmatched": "MapMatched",
    "dyn19_temporal_map": "T+M",
    "dyn19_temporal_flow_map": "T+F+M",
    "dyn19_temporal_flow_risk_map": "T+F+R+M",
    "dyn19_full": "T+F+R+A+M (Full)",
    "dyn19_full_no_mapping": "T+F+R+A+NoM (Full-NoM)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def f(value: str | float, scale: float = 1.0, digits: int = 9) -> str:
    return f"{float(value) * scale:.{digits}f}"


def validate_release(root: Path, manifest: dict) -> None:
    if manifest.get("release_id") != EXPECTED_RELEASE_ID:
        raise ValueError(f"Unexpected release_id: {manifest.get('release_id')}")
    if manifest.get("experiment_id") != EXPECTED_EXPERIMENT_ID:
        raise ValueError(f"Unexpected experiment_id: {manifest.get('experiment_id')}")
    if manifest.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError(f"Unexpected source_commit: {manifest.get('source_commit')}")

    checksum_rows = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        checksum_path = Path(name)
        if checksum_path.is_absolute():
            name = str(checksum_path.relative_to(root))
        elif name.startswith("./"):
            name = name[2:]
        checksum_rows[name] = digest
    for asset in manifest["assets"]:
        name = f"assets/{asset['name']}"
        path = root / name
        if not path.is_file() or path.stat().st_size != asset["bytes"]:
            raise ValueError(f"Missing or size-mismatched release asset: {name}")
        if checksum_rows.get(name) != asset["sha256"]:
            raise ValueError(f"SHA256SUMS disagrees with release manifest for {name}")


def certificate_rows(root: Path) -> tuple[list[dict], dict[tuple[str, str], Counter]]:
    rows = []
    status_by_config_sequence: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for phase in ("main", "mechanisms"):
        results = load_json(root / "metadata" / phase / "all_results.json")["results"]
        for result in results:
            metrics = result["metrics"]
            certificate = metrics["recovered_support_overlap_certificate"]
            evidence = certificate["evidence"]
            provenance = certificate["provenance"]
            status = certificate["status"]
            status_by_config_sequence[(result["config"], result["sequence"])][status] += 1
            rows.append(
                {
                    "phase": phase,
                    "config": result["config"],
                    "config_label": CONFIG_LABELS[result["config"]],
                    "sequence": result["sequence"],
                    "seed": result["seed"],
                    "certificate_status": status,
                    "recovered_support_observed": str(evidence["recovered_support_observed"]).lower(),
                    "mapping_support_present_for_recovered_frames": str(
                        evidence["mapping_support_present_for_recovered_frames"]
                    ).lower(),
                    "recovered_support_frames": evidence["recovered_support_frames"],
                    "tracking_recovery_audit_pixels": evidence[
                        "tracking_recovery_audit_pixels"
                    ],
                    "tracking_recovery_mapping_leak_pixels": evidence[
                        "tracking_recovery_mapping_leak_pixels"
                    ],
                    "benchmark_plan_sha256": provenance["plan_binding"]["sha256"],
                    "manifest_sha256": provenance["manifest_sha256"],
                    "run_summary_sha256": provenance["run_summary_sha256"],
                    "frame_metrics_sha256": provenance["frame_metrics_sha256"],
                    "source_commit": result["source"]["commit"],
                    "release_id": EXPECTED_RELEASE_ID,
                }
            )
    rows.sort(key=lambda row: (row["phase"], row["config"], row["sequence"], row["seed"]))
    return rows, status_by_config_sequence


def tracking_rows(root: Path, status_counts: dict[tuple[str, str], Counter]) -> list[dict]:
    rows = []
    for phase in ("main", "mechanisms"):
        for source in load_csv(root / "metadata" / phase / "aggregate.csv"):
            counts = status_counts[(source["config"], source["sequence"])]
            rows.append(
                {
                    "phase": phase,
                    "config": source["config"],
                    "config_label": CONFIG_LABELS[source["config"]],
                    "sequence": source["sequence"],
                    "successful_runs": source["successful_runs"],
                    "ate_rmse_cm": f(source["ate_mean_m"], 100.0),
                    "rpe_translation_cm": f(source["rpe_translation_mean_m"], 100.0),
                    "failure_rate_percent": f(source["failure_rate_mean"], 100.0),
                    "runtime_seconds": f(source["end_to_end_mean_seconds"]),
                    "recovery_audit_pixels_mean": f(
                        source["tracking_recovery_audit_pixels_mean"], digits=3
                    ),
                    "recovery_mapping_leak_pixels_mean": f(
                        source["tracking_recovery_mapping_leak_pixels_mean"], digits=3
                    ),
                    "certificate_passes": counts["pass"],
                    "certificate_vacuous": counts["vacuous"],
                    "certificate_failures": counts["fail"],
                    "source_commit": EXPECTED_SOURCE_COMMIT,
                    "release_id": EXPECTED_RELEASE_ID,
                }
            )
    rows.sort(key=lambda row: (row["phase"], row["config"], row["sequence"]))
    return rows


def mapping_rows(root: Path) -> list[dict]:
    fields = [
        "sequence",
        "seed",
        "config",
        "pose_mode",
        "frames",
        "manifest_sha256",
        "normal_metrics_sha256",
        "proxy_metrics_sha256",
        "psnr_full_mean",
        "ssim_full_mean",
        "psnr_static_mean",
        "ssim_static_mean",
        "proxy_coverage",
        "background_proxy_color_error_mean",
        "background_completeness_at_tau",
        "ghost_risk_proxy_at_tau_high",
    ]
    rows = []
    for source in load_csv(
        root / "metadata" / "mapping" / "dyn19_map_integrity_summary.csv"
    ):
        row = {key: source[key] for key in fields}
        row["config_label"] = CONFIG_LABELS[source["config"]]
        row["source_commit"] = EXPECTED_SOURCE_COMMIT
        row["release_id"] = EXPECTED_RELEASE_ID
        rows.append(row)
    rows.sort(key=lambda row: (row["pose_mode"], row["config"], row["sequence"], int(row["seed"])))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()

    manifest_path = root / "DYN19_RELEASE_MANIFEST.json"
    manifest = load_json(manifest_path)
    validate_release(root, manifest)
    DATA.mkdir(parents=True, exist_ok=True)

    certificates, status_counts = certificate_rows(root)
    tracking = tracking_rows(root, status_counts)
    mapping = mapping_rows(root)

    outputs = {
        "dyn19_tracking_sequence_means.csv": (
            list(tracking[0]),
            tracking,
        ),
        "dyn19_route_certificates.csv": (
            list(certificates[0]),
            certificates,
        ),
        "dyn19_mapping_cells.csv": (
            list(mapping[0]),
            mapping,
        ),
    }
    for name, (fieldnames, rows) in outputs.items():
        write_csv(DATA / name, fieldnames, rows)

    source_files = [
        manifest_path,
        root / "SHA256SUMS",
        root / "metadata" / "main" / "aggregate.csv",
        root / "metadata" / "main" / "all_results.json",
        root / "metadata" / "main" / "benchmark_plan.json",
        root / "metadata" / "mechanisms" / "aggregate.csv",
        root / "metadata" / "mechanisms" / "all_results.json",
        root / "metadata" / "mechanisms" / "benchmark_plan.json",
        root / "metadata" / "mapping" / "dyn19_map_integrity_summary.csv",
        root / "metadata" / "mapping" / "dyn19_map_integrity_plan.json",
    ]
    provenance = {
        "contract": "gdor-dyn19-public-evidence-v1",
        "generated_on": "2026-09-15",
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "release_id": EXPECTED_RELEASE_ID,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "release_manifest_sha256": sha256(manifest_path),
        "source_files": {
            str(path.relative_to(root)): sha256(path) for path in source_files
        },
        "derived_files": {name: sha256(DATA / name) for name in outputs},
        "counts": {
            "tracking_sequence_means": len(tracking),
            "route_certificate_cells": len(certificates),
            "mapping_metric_cells": len(mapping),
        },
        "claim_boundaries": manifest["claim_boundaries"],
    }
    (DATA / "dyn19_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
