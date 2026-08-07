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
    / "summarize_mapping_ablation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_mapping_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MappingAblationSummaryTest(unittest.TestCase):
    def make_fixture(
        self, root: Path, profile: str = "motion"
    ) -> Path:
        sequence = root / "example_sequence"
        sequence.mkdir()
        manifest = sequence / "manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame"])
            writer.writeheader()
            writer.writerow({"frame": 3})
            writer.writerow({"frame": 7})
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        (sequence / "protocol.json").write_text(json.dumps({
            "identity": {"sequence": "example_sequence", "seed": 2},
        }))

        payload = {
            "manifest_sha256": digest,
            "frames": 2,
            "frame_ids": [3, 7],
            "psnr_full_mean": 10.0,
            "ssim_full_mean": 0.5,
            "lpips_full_mean": 0.4,
            "psnr_static_mean": 11.0,
            "ssim_static_mean": 0.6,
            "lpips_static_mean": 0.3,
        }
        for pose_mode in MODULE.POSE_MODES:
            for variant in MODULE.PROFILES[profile]["variants"]:
                metric_path = MODULE.metric_summary_path(
                    sequence, profile, pose_mode, variant)
                metric_dir = metric_path.parent
                metric_dir.mkdir(parents=True)
                metric_path.write_text(json.dumps(payload))
        return sequence

    def test_collects_complete_six_by_two_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root)
            rows, sequences = MODULE.collect_rows(root)
            MODULE.write_summary(root, rows, sequences)

            self.assertEqual(len(rows), 12)
            self.assertEqual(
                next(
                    row["display_variant"]
                    for row in rows
                    if row["variant"] == "semantic_motion_rgbd_shuffled"
                ),
                "Lag-30",
            )
            self.assertEqual(sequences["example_sequence"]["frames"], 2)
            summary = json.loads((root / "mapping_summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertIn("Single frozen seed", summary["evidence_scope"])

    def test_rejects_frame_set_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = self.make_fixture(root)
            metric_path = (
                sequence
                / "metrics"
                / "semantic_online"
                / "heldout_metrics_summary.json"
            )
            payload = json.loads(metric_path.read_text())
            payload["frame_ids"] = [3, 8]
            metric_path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(
                    ValueError, "Frame IDs do not match"):
                MODULE.collect_rows(root)

    def test_collects_dypho_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root, "dypho")
            rows, sequences = MODULE.collect_rows(root, "dypho")
            MODULE.write_summary(root, rows, sequences, "dypho")

            self.assertEqual(len(rows), 16)
            self.assertEqual(
                {
                    row["variant"] for row in rows
                },
                set(MODULE.PROFILES["dypho"]["variants"]),
            )
            summary = json.loads(
                (root / "mapping_summary.json").read_text())
            self.assertEqual(summary["profile"], "dypho")
            self.assertEqual(
                summary["display_names"]["dypho_compatible"], "Full")
            self.assertEqual(
                summary["display_names"]["dypho_decoupled"], "Decoupled")
            self.assertEqual(
                summary["display_names"]["dypho_flow_guarded"],
                "Flow-Guarded")
            self.assertEqual(
                summary["display_names"]["dypho_flow_adaptive"],
                "Adaptive Flow-Guarded")

    def test_collects_dypho_core_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_fixture(root, "dypho-core")
            rows, sequences = MODULE.collect_rows(root, "dypho-core")
            MODULE.write_summary(root, rows, sequences, "dypho-core")

            self.assertEqual(len(rows), 8)
            self.assertEqual(
                {row["variant"] for row in rows},
                {
                    "dypho_raw",
                    "dypho_decoupled",
                    "dypho_flow_guarded",
                    "dypho_flow_adaptive",
                },
            )
            summary = json.loads(
                (root / "mapping_summary.json").read_text())
            self.assertEqual(summary["profile"], "dypho-core")
            self.assertEqual(
                summary["display_names"]["dypho_flow_guarded"],
                "Flow-Guarded")
            self.assertEqual(
                summary["display_names"]["dypho_flow_adaptive"],
                "Adaptive Flow-Guarded")


if __name__ == "__main__":
    unittest.main()
