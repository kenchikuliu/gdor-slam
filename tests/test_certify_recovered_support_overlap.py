from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "certify_recovered_support_overlap.py"
)
SPEC = importlib.util.spec_from_file_location("certificate", SCRIPT)
CERTIFICATE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CERTIFICATE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_run(
    root: Path,
    rows: list[dict[str, int]],
) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    plan_path = root / "benchmark_plan.json"
    plan_path.write_text(json.dumps({"tasks": []}) + "\n")
    manifest = {
        "sequence": "tum_sitting_halfsphere",
        "config": "dyn19_full",
        "seed": 1,
        "plan_binding": {
            "contract": "benchmark-plan-binding-v1",
            "path": str(plan_path),
            "sha256": sha256(plan_path),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")
    summary = {
        "tracking_recovery_audit_pixels": sum(
            row["tracking_recovery_audit_pixels"] for row in rows),
        "tracking_recovery_mapping_leak_pixels": sum(
            row["tracking_recovery_mapping_leak_pixels"] for row in rows),
        "mapping_weight_frames": sum(
            row["mapping_weight_present"] for row in rows),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary) + "\n")
    with (run_dir / "frame_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run_dir


class RecoveredSupportOverlapCertificateTest(unittest.TestCase):
    def test_pass_requires_observed_recovery_mapping_weight_and_no_overlap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), [
                {
                    "frame": 0,
                    "tracking_recovery_audit_pixels": 7,
                    "tracking_recovery_mapping_leak_pixels": 0,
                    "mapping_weight_present": 1,
                },
                {
                    "frame": 1,
                    "tracking_recovery_audit_pixels": 0,
                    "tracking_recovery_mapping_leak_pixels": 0,
                    "mapping_weight_present": 1,
                },
            ])
            certificate = CERTIFICATE.certify(run_dir)

        self.assertEqual(certificate["status"], "pass")
        self.assertTrue(
            certificate["evidence"]["recovered_support_observed"])
        self.assertTrue(certificate["evidence"]["zero_overlap"])

    def test_zero_overlap_without_recovery_is_vacuous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), [
                {
                    "frame": 0,
                    "tracking_recovery_audit_pixels": 0,
                    "tracking_recovery_mapping_leak_pixels": 0,
                    "mapping_weight_present": 1,
                },
            ])
            certificate = CERTIFICATE.certify(run_dir)

        self.assertEqual(certificate["status"], "vacuous")
        self.assertFalse(
            certificate["evidence"]["recovered_support_observed"])
        self.assertTrue(certificate["evidence"]["zero_overlap"])

    def test_overlap_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = make_run(Path(directory), [
                {
                    "frame": 0,
                    "tracking_recovery_audit_pixels": 7,
                    "tracking_recovery_mapping_leak_pixels": 1,
                    "mapping_weight_present": 1,
                },
            ])
            certificate = CERTIFICATE.certify(run_dir)

        self.assertEqual(certificate["status"], "fail")
        self.assertIn(
            "recovered tracking support overlaps mapping support",
            certificate["failures"],
        )


if __name__ == "__main__":
    unittest.main()
