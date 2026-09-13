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
    / "summarize_dyn19_map_integrity.py"
)
SPEC = importlib.util.spec_from_file_location("map_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Dyn19MapIntegritySummaryTest(unittest.TestCase):
    def write_fixture(self, root: Path) -> None:
        semantic_mask_dir = root / "semantic_masks"
        semantic_mask_dir.mkdir()
        tasks = []
        for sequence in SUMMARY.CORE_SEQUENCES:
            for seed in SUMMARY.REQUIRED_SEEDS:
                cell_dir = root / sequence / f"seed_{seed:04d}"
                cell_dir.mkdir(parents=True)
                manifest = cell_dir / "manifest.csv"
                with manifest.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("frame",))
                    writer.writeheader()
                    writer.writerow({"frame": 42})
                manifest_sha = sha256(manifest)
                run_hashes = {
                    config: f"{config}-{sequence}-{seed}"
                    for config in SUMMARY.REQUIRED_CONFIGS
                }
                protocol = {
                    "identity": {"sequence": sequence, "seed": seed},
                    "dataset": {
                        "static_mask_provenance": {
                            "directory": str(semantic_mask_dir),
                        },
                    },
                    "runs": {
                        config: {"manifest_sha256": run_hashes[config]}
                        for config in SUMMARY.REQUIRED_CONFIGS
                    },
                }
                (cell_dir / "protocol.json").write_text(
                    json.dumps(protocol) + "\n")
                proxy_dir = cell_dir / "occluded_background_proxy"
                proxy_dir.mkdir()
                (proxy_dir / "occluded_background_proxy_manifest.json").write_text(
                    json.dumps({
                        "contract": "occluded-background-proxy-v1",
                        "heldout_manifest_sha256": manifest_sha,
                    }) + "\n")
                for pose_mode in SUMMARY.POSE_MODES:
                    for config in SUMMARY.REQUIRED_CONFIGS:
                        normal_dir = cell_dir / "metrics" / pose_mode / config
                        normal_dir.mkdir(parents=True)
                        normal = {
                            "manifest_sha256": manifest_sha,
                            "frames": 1,
                            "frame_ids": [42],
                            "psnr_full_mean": 20.0,
                            "ssim_full_mean": 0.7,
                            "lpips_full_mean": 0.3,
                            "psnr_static_mean": 21.0,
                            "ssim_static_mean": 0.8,
                            "lpips_static_mean": 0.2,
                        }
                        (normal_dir / "heldout_metrics_summary.json").write_text(
                            json.dumps(normal) + "\n")
                        proxy_metric_dir = (
                            cell_dir / "proxy_metrics" / pose_mode / config)
                        proxy_metric_dir.mkdir(parents=True)
                        proxy_metrics = {
                            "heldout_manifest_sha256": manifest_sha,
                            "frames": 1,
                            "frame_ids": [42],
                            "variant": config,
                            "pose_mode": pose_mode,
                            "proxy_coverage": 0.4,
                            "background_proxy_color_error_mean": 12.0,
                            "background_completeness_at_tau": 0.8,
                            "ghost_risk_proxy_at_tau_high": 0.1,
                        }
                        (
                            proxy_metric_dir /
                            "occluded_background_proxy_metrics_summary.json"
                        ).write_text(json.dumps(proxy_metrics) + "\n")
                tasks.append({
                    "sequence": sequence,
                    "seed": seed,
                    "output_dir": str(cell_dir),
                    "semantic_static_mask_dir": str(semantic_mask_dir),
                    "run_manifest_sha256": run_hashes,
                })
        plan = {
            "contract": SUMMARY.PLAN_CONTRACT,
            "experiment_id": SUMMARY.DYN19_EXPERIMENT_ID,
            "tracking_root": "/nas/dyn19-main",
            "tracking_main_overlap_certificate": {
                "safety_status": "pass",
            },
            "tasks": tasks,
        }
        plan_path = root / "dyn19_map_integrity_plan.json"
        plan_path.write_text(json.dumps(plan) + "\n")
        (root / "dyn19_map_integrity_plan.sha256").write_text(
            f"{sha256(plan_path)}  dyn19_map_integrity_plan.json\n")

    def test_complete_matrix_is_validated_and_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            payload = SUMMARY.summarize(root)
            saved = json.loads(
                (root / "dyn19_map_integrity_summary.json").read_text())

        self.assertEqual(payload["status"], "complete")
        self.assertEqual(len(payload["rows"]), 72)
        self.assertEqual(
            payload["aggregates"]["dyn19_full/gt_aligned"]["cells"], 12)
        self.assertAlmostEqual(
            payload["aggregates"]["dyn19_full/gt_aligned"][
                "background_completeness_at_tau"],
            0.8,
        )
        self.assertIn("not true ghost-contamination", saved["evidence_scope"])


if __name__ == "__main__":
    unittest.main()
