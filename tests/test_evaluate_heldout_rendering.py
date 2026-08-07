from __future__ import annotations

import importlib.util
import hashlib
import math
import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_heldout_rendering.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_heldout_rendering", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RegionMetricTest(unittest.TestCase):
    def test_static_mask_excludes_dynamic_region_error(self) -> None:
        reference = np.zeros((16, 16, 3), dtype=np.uint8)
        estimate = reference.copy()
        estimate[:, 8:, :] = 255
        full_mask = np.ones((16, 16), dtype=bool)
        static_mask = full_mask.copy()
        static_mask[:, 8:] = False

        full_psnr = MODULE.masked_psnr(reference, estimate, full_mask)
        static_psnr = MODULE.masked_psnr(reference, estimate, static_mask)

        self.assertTrue(math.isfinite(full_psnr))
        self.assertTrue(math.isinf(static_psnr))

    def test_disabled_lpips_reports_all_regions(self) -> None:
        evaluator = MODULE.LpipsEvaluator(enabled=False)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        masks = {
            "full": np.ones((8, 8), dtype=bool),
            "empty": np.zeros((8, 8), dtype=bool),
        }

        results = evaluator.evaluate_regions(image, image, masks)

        self.assertEqual(set(results), set(masks))
        self.assertTrue(all(math.isnan(value) for value in results.values()))

    def write_manifest(self, root: Path, frames: list[int]) -> Path:
        path = root / "manifest.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["frame", "ground_truth", "static_mask"])
            writer.writeheader()
            for frame in frames:
                writer.writerow({
                    "frame": frame,
                    "ground_truth": f"ground_truth/{frame:06d}.png",
                    "static_mask": f"static_mask/{frame:06d}.png",
                })
        return path

    def test_manifest_missing_render_fails_entire_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ground_truth").mkdir()
            (root / "static_mask").mkdir()
            (root / "render").mkdir()
            manifest = self.write_manifest(root, [1, 2])
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            mask = np.ones((8, 8), dtype=np.uint8) * 255
            for frame in (1, 2):
                cv2.imwrite(
                    str(root / "ground_truth" / f"{frame:06d}.png"), image)
                cv2.imwrite(
                    str(root / "static_mask" / f"{frame:06d}.png"), mask)
            cv2.imwrite(str(root / "render/000001.png"), image)

            with self.assertRaisesRegex(
                    RuntimeError, "does not exactly match"):
                MODULE.load_manifest_triplets(
                    manifest, root, root / "render")

    def test_manifest_unreadable_image_is_not_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ground_truth").mkdir()
            (root / "static_mask").mkdir()
            (root / "render").mkdir()
            manifest = self.write_manifest(root, [7])
            (root / "ground_truth/000007.png").write_bytes(b"not an image")
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            mask = np.ones((8, 8), dtype=np.uint8) * 255
            cv2.imwrite(str(root / "render/000007.png"), image)
            cv2.imwrite(str(root / "static_mask/000007.png"), mask)

            with self.assertRaisesRegex(RuntimeError, "unreadable ground_truth"):
                MODULE.load_manifest_triplets(
                    manifest, root, root / "render")

    def test_manifest_rejects_unexpected_stale_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ground_truth").mkdir()
            (root / "static_mask").mkdir()
            (root / "render").mkdir()
            manifest = self.write_manifest(root, [3])
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            mask = np.ones((8, 8), dtype=np.uint8) * 255
            cv2.imwrite(str(root / "ground_truth/000003.png"), image)
            cv2.imwrite(str(root / "static_mask/000003.png"), mask)
            cv2.imwrite(str(root / "render/000003.png"), image)
            cv2.imwrite(str(root / "render/000004.png"), image)

            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                MODULE.load_manifest_triplets(
                    manifest, root, root / "render")

    def test_manifest_hash_mismatch_fails_entire_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ground_truth").mkdir()
            (root / "static_mask").mkdir()
            (root / "render").mkdir()
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            mask = np.ones((8, 8), dtype=np.uint8) * 255
            reference = root / "ground_truth/000009.png"
            mask_path = root / "static_mask/000009.png"
            cv2.imwrite(str(reference), image)
            cv2.imwrite(str(mask_path), mask)
            cv2.imwrite(str(root / "render/000009.png"), image)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "frame", "ground_truth", "ground_truth_sha256",
                    "static_mask", "static_mask_sha256",
                ])
                writer.writeheader()
                writer.writerow({
                    "frame": 9,
                    "ground_truth": "ground_truth/000009.png",
                    "ground_truth_sha256": "0" * 64,
                    "static_mask": "static_mask/000009.png",
                    "static_mask_sha256": hashlib.sha256(
                        mask_path.read_bytes()).hexdigest(),
                })

            with self.assertRaisesRegex(RuntimeError, "hash does not match"):
                MODULE.load_manifest_triplets(
                    manifest, root, root / "render")

    def test_manifest_rejects_empty_static_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ground_truth").mkdir()
            (root / "static_mask").mkdir()
            (root / "render").mkdir()
            manifest = self.write_manifest(root, [11])
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            mask = np.zeros((8, 8), dtype=np.uint8)
            cv2.imwrite(str(root / "ground_truth/000011.png"), image)
            cv2.imwrite(str(root / "render/000011.png"), image)
            cv2.imwrite(str(root / "static_mask/000011.png"), mask)

            with self.assertRaisesRegex(RuntimeError, "no positive pixels"):
                MODULE.load_manifest_triplets(
                    manifest, root, root / "render")


if __name__ == "__main__":
    unittest.main()
