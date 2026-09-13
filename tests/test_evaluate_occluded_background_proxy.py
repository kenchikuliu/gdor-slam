from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_occluded_background_proxy.py"
)
SPEC = importlib.util.spec_from_file_location("proxy", SCRIPT)
PROXY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PROXY
SPEC.loader.exec_module(PROXY)


class OccludedBackgroundProxyTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> dict[str, Path]:
        dataset = root / "dataset"
        rgb_dir = dataset / "rgb"
        depth_dir = dataset / "depth"
        mask_dir = root / "semantic_masks"
        heldout = root / "heldout"
        fixed_mask_dir = heldout / "static_mask"
        rgb_dir.mkdir(parents=True)
        depth_dir.mkdir()
        mask_dir.mkdir()
        fixed_mask_dir.mkdir(parents=True)

        association_rows = []
        source_colors = {
            0: np.array([12, 34, 56], dtype=np.uint8),
            1: np.array([0, 0, 0], dtype=np.uint8),
            2: np.array([180, 150, 120], dtype=np.uint8),
        }
        source_depths = {0: 2000, 1: 1000, 2: 3000}
        for frame in range(3):
            color = np.zeros((3, 3, 3), dtype=np.uint8)
            color[0, 0] = source_colors[frame]
            depth = np.full(
                (3, 3), source_depths[frame], dtype=np.uint16)
            cv2.imwrite(str(rgb_dir / f"{frame}.png"), color)
            cv2.imwrite(str(depth_dir / f"{frame}.png"), depth)
            mask = np.full((3, 3), 255, dtype=np.uint8)
            if frame == 1:
                mask[0, 0] = 0
            cv2.imwrite(str(mask_dir / f"{frame:06d}.png"), mask)
            association_rows.append(
                f"{float(frame):.6f} rgb/{frame}.png "
                f"{float(frame):.6f} depth/{frame}.png")

        association = dataset / "associations.txt"
        association.write_text("\n".join(association_rows) + "\n")
        groundtruth = dataset / "groundtruth.txt"
        groundtruth.write_text(
            "\n".join(
                f"{float(frame):.6f} 0 0 0 0 0 0 1"
                for frame in range(3)
            ) + "\n")
        camera = root / "camera.yaml"
        camera.write_text(
            "%YAML:1.0\n"
            "Camera.width: 3\n"
            "Camera.height: 3\n"
            "Camera1.fx: 1.0\n"
            "Camera1.fy: 1.0\n"
            "Camera1.cx: 0.0\n"
            "Camera1.cy: 0.0\n"
            "RGBD.DepthMapFactor: 1000.0\n"
        )
        heldout_mask = fixed_mask_dir / "000001.png"
        heldout_mask.write_bytes((mask_dir / "000001.png").read_bytes())
        manifest = heldout / "manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "frame",
                    "timestamp",
                    "ground_truth",
                    "static_mask",
                    "static_mask_sha256",
                ),
            )
            writer.writeheader()
            writer.writerow({
                "frame": 1,
                "timestamp": "1.000000000",
                "ground_truth": "ground_truth/000001.png",
                "static_mask": "static_mask/000001.png",
                "static_mask_sha256": PROXY.sha256(heldout_mask),
            })
        return {
            "dataset": dataset,
            "association": association,
            "groundtruth": groundtruth,
            "camera": camera,
            "mask_dir": mask_dir,
            "heldout": heldout,
            "manifest": manifest,
        }

    def test_build_reprojects_background_with_nearest_depth_zbuffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            proxy_dir = root / "proxy"
            summary = PROXY.build_proxy(
                dataset_dir=paths["dataset"],
                association_path=paths["association"],
                groundtruth_path=paths["groundtruth"],
                camera_config=paths["camera"],
                static_mask_dir=paths["mask_dir"],
                heldout_dir=paths["heldout"],
                manifest_path=paths["manifest"],
                output_dir=proxy_dir,
                groundtruth_pose_model="camera",
                pose_max_delta=0.01,
                source_stride=1,
                pixel_stride=1,
                minimum_frame_gap=1,
                occlusion_margin_m=0.05,
            )
            proxy = cv2.imread(
                str(proxy_dir / "proxy" / "000001.png"), cv2.IMREAD_COLOR)
            support = cv2.imread(
                str(proxy_dir / "support" / "000001.png"),
                cv2.IMREAD_GRAYSCALE)

        self.assertEqual(summary["summary"]["target_dynamic_pixels"], 1)
        self.assertEqual(summary["summary"]["proxy_support_pixels"], 1)
        self.assertAlmostEqual(summary["summary"]["proxy_coverage"], 1.0)
        self.assertEqual(support[0, 0], 255)
        self.assertTrue(np.array_equal(
            proxy[0, 0], np.array([12, 34, 56], dtype=np.uint8)))

    def test_metrics_report_completeness_and_ghost_risk_as_proxy_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.make_fixture(root)
            proxy_dir = root / "proxy"
            PROXY.build_proxy(
                dataset_dir=paths["dataset"],
                association_path=paths["association"],
                groundtruth_path=paths["groundtruth"],
                camera_config=paths["camera"],
                static_mask_dir=paths["mask_dir"],
                heldout_dir=paths["heldout"],
                manifest_path=paths["manifest"],
                output_dir=proxy_dir,
                groundtruth_pose_model="camera",
                pose_max_delta=0.01,
                source_stride=1,
                pixel_stride=1,
                minimum_frame_gap=1,
                occlusion_margin_m=0.05,
            )
            perfect_render_dir = root / "render_perfect"
            perfect_render_dir.mkdir()
            (perfect_render_dir / "000001.png").write_bytes(
                (proxy_dir / "proxy" / "000001.png").read_bytes())
            perfect = PROXY.evaluate_proxy(
                proxy_dir=proxy_dir,
                heldout_manifest_path=paths["manifest"],
                render_dir=perfect_render_dir,
                output_dir=root / "metrics_perfect",
                tau=1.0,
                tau_high=10.0,
                variant="dyn19_full",
                pose_mode="gt_aligned",
            )

            bad_render_dir = root / "render_bad"
            bad_render_dir.mkdir()
            cv2.imwrite(
                str(bad_render_dir / "000001.png"),
                np.zeros((3, 3, 3), dtype=np.uint8))
            bad = PROXY.evaluate_proxy(
                proxy_dir=proxy_dir,
                heldout_manifest_path=paths["manifest"],
                render_dir=bad_render_dir,
                output_dir=root / "metrics_bad",
                tau=1.0,
                tau_high=10.0,
                variant="dyn19_semantic",
                pose_mode="online",
            )
            written = json.loads(
                (root / "metrics_bad" /
                 "occluded_background_proxy_metrics_summary.json").read_text())

        self.assertEqual(perfect["contract"], PROXY.METRIC_CONTRACT)
        self.assertAlmostEqual(perfect["proxy_coverage"], 1.0)
        self.assertAlmostEqual(perfect["background_completeness_at_tau"], 1.0)
        self.assertAlmostEqual(perfect["ghost_risk_proxy_at_tau_high"], 0.0)
        self.assertAlmostEqual(bad["background_completeness_at_tau"], 0.0)
        self.assertAlmostEqual(bad["ghost_risk_proxy_at_tau_high"], 1.0)
        self.assertIn("not a true ghost-count metric", written["protocol"])


if __name__ == "__main__":
    unittest.main()
