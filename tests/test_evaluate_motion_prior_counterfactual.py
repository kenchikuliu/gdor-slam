from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_motion_prior_counterfactual.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_motion_prior_counterfactual", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CounterfactualEvaluationTest(unittest.TestCase):
    def test_duplicate_ground_truth_timestamps_are_averaged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ground_truth = Path(directory) / "groundtruth.txt"
            ground_truth.write_text(
                "0.0 0 0 0 0 0 0 1\n"
                "1.0 1 0 0 0 0 0 1\n"
                "1.0 3 0 0 0 0 0 -1\n"
                "2.0 4 0 0 0 0 0 1\n")

            timestamps, poses, metadata = MODULE.load_tum_poses(ground_truth)

            self.assertEqual(timestamps, [0.0, 1.0, 2.0])
            self.assertAlmostEqual(poses[1][0], 2.0)
            self.assertAlmostEqual(abs(poses[1][6]), 1.0)
            self.assertEqual(metadata["duplicate_timestamp_groups"], 1)
            self.assertEqual(metadata["duplicate_rows_merged"], 1)

    def test_decreasing_ground_truth_timestamp_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ground_truth = Path(directory) / "groundtruth.txt"
            ground_truth.write_text(
                "1.0 0 0 0 0 0 0 1\n"
                "0.0 0 0 0 0 0 0 1\n")

            with self.assertRaisesRegex(ValueError, "nondecreasing"):
                MODULE.load_tum_poses(ground_truth)

    def test_dynamic_hypothesis_can_improve_relative_translation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth = root / "groundtruth.txt"
            ground_truth.write_text(
                "0.0 0 0 0 0 0 0 1\n"
                "1.0 1 0 0 0 0 0 1\n")
            counterfactual = root / "motion_prior_counterfactual.csv"
            fields = [
                "frame", "timestamp", "previous_timestamp",
                "prior_would_use", "prior_used",
                "static_inliers", "dynamic_inliers",
                "common_support", "static_common_score",
                "dynamic_common_score",
                "direct_common_support",
                "direct_static_depth_score",
                "direct_dynamic_depth_score",
                "direct_static_photometric_score",
                "direct_dynamic_photometric_score",
                "direct_static_combined_score",
                "direct_dynamic_combined_score",
                "previous_tx", "previous_ty", "previous_tz",
                "previous_qx", "previous_qy", "previous_qz", "previous_qw",
                "static_tx", "static_ty", "static_tz",
                "static_qx", "static_qy", "static_qz", "static_qw",
                "dynamic_tx", "dynamic_ty", "dynamic_tz",
                "dynamic_qx", "dynamic_qy", "dynamic_qz", "dynamic_qw",
            ]
            with counterfactual.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                supported_row = {
                    "frame": 1,
                    "timestamp": 1.0,
                    "previous_timestamp": 0.0,
                    "prior_would_use": 1,
                    "prior_used": 0,
                    "static_inliers": 100,
                    "dynamic_inliers": 110,
                    "common_support": 20,
                    "static_common_score": 2.0,
                    "dynamic_common_score": 1.0,
                    "direct_common_support": 100,
                    "direct_static_depth_score": 2.0,
                    "direct_dynamic_depth_score": 1.0,
                    "direct_static_photometric_score": 2.0,
                    "direct_dynamic_photometric_score": 1.0,
                    "direct_static_combined_score": 2.0,
                    "direct_dynamic_combined_score": 1.0,
                    "previous_tx": 0.0,
                    "previous_ty": 0.0,
                    "previous_tz": 0.0,
                    "previous_qx": 0.0,
                    "previous_qy": 0.0,
                    "previous_qz": 0.0,
                    "previous_qw": 1.0,
                    "static_tx": 0.8,
                    "static_ty": 0.0,
                    "static_tz": 0.0,
                    "static_qx": 0.0,
                    "static_qy": 0.0,
                    "static_qz": 0.0,
                    "static_qw": 1.0,
                    "dynamic_tx": 1.0,
                    "dynamic_ty": 0.0,
                    "dynamic_tz": 0.0,
                    "dynamic_qx": 0.0,
                    "dynamic_qy": 0.0,
                    "dynamic_qz": 0.0,
                    "dynamic_qw": 1.0,
                }
                outside_row = dict(supported_row)
                outside_row.update({
                    "frame": 0,
                    "timestamp": -0.5,
                    "previous_timestamp": -1.0,
                })
                writer.writerow(outside_row)
                writer.writerow(supported_row)

            rows, summary = MODULE.evaluate(
                counterfactual, ground_truth, 0.01)

            self.assertAlmostEqual(
                rows[0]["static_translation_error_m"], 0.2)
            self.assertAlmostEqual(
                rows[0]["dynamic_translation_error_m"], 0.0)
            self.assertAlmostEqual(
                rows[0]["hypothesis_translation_delta_m"], 0.2)
            self.assertEqual(rows[0]["dynamic_translation_win"], 1)
            self.assertEqual(
                summary["gate_would_use"]["dynamic_translation_win_rate"], 1.0)
            self.assertAlmostEqual(
                summary["gate_would_use"][
                    "static_translation_sum_squared_error_m2"], 0.04)
            self.assertAlmostEqual(
                summary["gate_would_use"][
                    "dynamic_translation_sum_squared_error_m2"], 0.0)
            self.assertEqual(
                summary["gate_would_use"]["dynamic_translation_wins"], 1)
            self.assertAlmostEqual(
                summary["gate_would_use"][
                    "worst_translation_degradation_m"], -0.2)
            self.assertEqual(summary["source_pairs"], 2)
            self.assertEqual(summary["ground_truth_supported_pairs"], 1)
            self.assertEqual(summary["outside_ground_truth_range_pairs"], 1)
            self.assertEqual(
                summary["outside_ground_truth_range_frames"], [0])
            self.assertEqual(
                summary["ground_truth_duplicate_timestamp_groups"], 0)
            self.assertEqual(
                summary["common_score_prefers_dynamic"]["pairs"], 1)
            self.assertEqual(
                summary["common_score_prefers_dynamic"][
                    "dynamic_translation_win_rate"], 1.0)
            self.assertEqual(
                summary["direct_combined_prefers_dynamic"]["pairs"], 1)

    def test_pose_interpolation_uses_translation_and_slerp(self) -> None:
        timestamps = [0.0, 1.0]
        poses = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        ]
        pose, span = MODULE.interpolate_pose(
            timestamps, poses, 0.5, 1.1)
        self.assertAlmostEqual(pose[0], 1.0)
        self.assertAlmostEqual(abs(pose[5]), 2 ** -0.5)
        self.assertAlmostEqual(abs(pose[6]), 2 ** -0.5)
        self.assertAlmostEqual(span, 1.0)

    def test_outside_ground_truth_range_fails(self) -> None:
        timestamps = [0.0, 1.0]
        poses = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ]
        with self.assertRaisesRegex(ValueError, "outside the ground-truth range"):
            MODULE.interpolate_pose(timestamps, poses, 2.0, 1.1)

    def test_large_ground_truth_gap_fails(self) -> None:
        timestamps = [0.0, 1.0]
        poses = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ]
        with self.assertRaisesRegex(ValueError, "bracket.*exceeding"):
            MODULE.interpolate_pose(timestamps, poses, 0.5, 0.1)


if __name__ == "__main__":
    unittest.main()
