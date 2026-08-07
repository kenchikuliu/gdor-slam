from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_checkpointed_forks.py"
)
SPEC = importlib.util.spec_from_file_location("fork_evaluation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CheckpointedForkEvaluationTest(unittest.TestCase):
    def write_fixture(self, root: Path) -> tuple[Path, Path]:
        ground_truth = root / "groundtruth.txt"
        ground_truth.write_text(
            "1.0 0 0 0 0 0 0 1\n"
            "2.0 1 0 0 0 0 0 1\n"
        )
        fork_csv = root / "forks.csv"
        row = {
            "checkpoint": "frame_000001.yml.gz",
            "selection": "all",
            "gate_policy": "direct-combined",
            "information_scale_multiplier": "0.5",
            "max_static_information_leverage": "0.1",
            "static_information_leverage_mode": "normalize-to-target",
            "horizon": "1",
            "start_timestamp": "1.0",
            "end_timestamp": "2.0",
            "static_success": "1",
            "dynamic_success": "1",
        }
        row["velocity_neutral_success"] = "1"
        for prefix in (
            "start_tcw",
            "static_tcw",
            "dynamic_tcw",
            "velocity_neutral_tcw",
        ):
            row.update(
                {
                    f"{prefix}_tx": "0",
                    f"{prefix}_ty": "0",
                    f"{prefix}_tz": "0",
                    f"{prefix}_qx": "0",
                    f"{prefix}_qy": "0",
                    f"{prefix}_qz": "0",
                    f"{prefix}_qw": "1",
                }
            )
        row["static_tcw_tx"] = "-1"
        row["dynamic_tcw_tx"] = "-1.1"
        row["velocity_neutral_tcw_tx"] = "-0.9"
        with fork_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(row))
            writer.writeheader()
            writer.writerow(row)
        return fork_csv, ground_truth

    def test_evaluates_relative_pose_without_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fork_csv, ground_truth = self.write_fixture(Path(directory))
            summary, rows = MODULE.evaluate(
                fork_csv, ground_truth, max_ground_truth_gap=0.1
            )
            horizon = summary["horizons"]["1"]
            self.assertEqual(horizon["paired_successes"], 1)
            self.assertEqual(summary["prior_subspace"], "full")
            self.assertEqual(summary["gate_policy"], "direct-combined")
            self.assertEqual(rows[0]["gate_policy"], "direct-combined")
            self.assertEqual(rows[0]["prior_subspace"], "full")
            self.assertEqual(summary["intervention_projection"], "full")
            self.assertEqual(rows[0]["intervention_projection"], "full")
            self.assertEqual(summary["information_scale_multiplier"], 0.5)
            self.assertEqual(
                summary["max_static_information_leverage"], 0.1
            )
            self.assertEqual(
                summary["static_information_leverage_mode"],
                "normalize-to-target",
            )
            self.assertEqual(
                rows[0]["static_information_leverage_mode"],
                "normalize-to-target",
            )
            self.assertAlmostEqual(
                rows[0]["static_translation_rpe_m"], 0.0, places=9
            )
            self.assertAlmostEqual(
                rows[0]["dynamic_translation_rpe_m"], 0.1, places=9
            )
            self.assertEqual(
                horizon["dynamic_translation_win_count"], 0
            )
            self.assertAlmostEqual(
                rows[0]["velocity_neutral_translation_rpe_m"],
                0.1,
                places=9,
            )
            self.assertEqual(
                horizon["velocity_neutral_translation_win_count"], 0
            )

    def test_rejects_unmatched_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fork_csv, ground_truth = self.write_fixture(Path(directory))
            ground_truth.write_text(
                "10.0 0 0 0 0 0 0 1\n"
                "20.0 1 0 0 0 0 0 1\n"
            )
            with self.assertRaisesRegex(ValueError, "No fork endpoints"):
                MODULE.evaluate(
                    fork_csv, ground_truth, max_ground_truth_gap=0.1
                )

    def test_zero_selected_checkpoints_are_insufficient_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth = root / "groundtruth.txt"
            ground_truth.write_text("1.0 0 0 0 0 0 0 1\n")
            fork_csv = root / "forks.csv"
            fork_csv.write_text(
                "checkpoint,selection,horizon,start_timestamp,end_timestamp\n"
            )

            summary, rows = MODULE.evaluate(
                fork_csv, ground_truth, max_ground_truth_gap=0.1
            )

            self.assertEqual(summary["status"], "insufficient_data")
            self.assertEqual(summary["input_rows"], 0)
            self.assertEqual(summary["horizons"], {})
            self.assertEqual(rows, [])

    def test_rejects_mixed_information_scales(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fork_csv, ground_truth = self.write_fixture(Path(directory))
            with fork_csv.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
                fieldnames = tuple(rows[0])
            second = dict(rows[0])
            second["checkpoint"] = "frame_000002.yml.gz"
            second["information_scale_multiplier"] = "1.0"
            with fork_csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows((rows[0], second))
            with self.assertRaisesRegex(
                ValueError, "one finite non-negative"
            ):
                MODULE.evaluate(
                    fork_csv,
                    ground_truth,
                    max_ground_truth_gap=0.1,
                )

    def test_rejects_mixed_gate_policies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fork_csv, ground_truth = self.write_fixture(Path(directory))
            with fork_csv.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            second = dict(rows[0])
            second["checkpoint"] = "frame_000002.yml.gz"
            second["gate_policy"] = "legacy-conjunction"
            fieldnames = tuple(rows[0])
            with fork_csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows((rows[0], second))
            with self.assertRaisesRegex(
                ValueError, "one supported gate_policy"
            ):
                MODULE.evaluate(
                    fork_csv,
                    ground_truth,
                    max_ground_truth_gap=0.1,
                )

    def test_rejects_mixed_prior_subspaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fork_csv, ground_truth = self.write_fixture(Path(directory))
            with fork_csv.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["prior_subspace"] = "full"
            second = dict(rows[0])
            second["checkpoint"] = "frame_000002.yml.gz"
            second["prior_subspace"] = "translation"
            fieldnames = tuple(rows[0])
            with fork_csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows((rows[0], second))
            with self.assertRaisesRegex(
                ValueError, "one supported prior_subspace"
            ):
                MODULE.evaluate(
                    fork_csv,
                    ground_truth,
                    max_ground_truth_gap=0.1,
                )

    def test_rejects_mixed_intervention_projections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fork_csv, ground_truth = self.write_fixture(Path(directory))
            with fork_csv.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["intervention_projection"] = "full"
            second = dict(rows[0])
            second["checkpoint"] = "frame_000002.yml.gz"
            second["intervention_projection"] = "common-score"
            fieldnames = tuple(rows[0])
            with fork_csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows((rows[0], second))
            with self.assertRaisesRegex(
                ValueError, "one supported intervention_projection"
            ):
                MODULE.evaluate(
                    fork_csv,
                    ground_truth,
                    max_ground_truth_gap=0.1,
                )

    def test_interpolates_translation_and_rotation(self) -> None:
        timestamps = [1.0, 3.0]
        poses = [
            MODULE.np.asarray((0, 0, 0, 0, 0, 0, 1), dtype=float),
            MODULE.np.asarray((2, 0, 0, 0, 0, 1, 0), dtype=float),
        ]
        interpolated = MODULE.interpolate_pose(
            2.0, timestamps, poses, max_bracket_span=2.0
        )
        self.assertIsNotNone(interpolated)
        assert interpolated is not None
        self.assertAlmostEqual(interpolated[0, 3], 1.0, places=9)
        rotated_x = interpolated[:3, :3] @ MODULE.np.asarray((1, 0, 0))
        self.assertAlmostEqual(rotated_x[0], 0.0, places=9)
        self.assertAlmostEqual(rotated_x[1], 1.0, places=9)

    def test_relative_error_uses_standard_gt_inverse_order(self) -> None:
        identity = MODULE.np.eye(4)
        estimated_end_tcw = MODULE.pose_matrix(
            [1, 0, 0, 0, 0, 0, 1]
        )
        root_half = 2 ** -0.5
        end_twc_gt = MODULE.pose_matrix(
            [0, 1, 0, 0, 0, -root_half, root_half]
        )
        translation, rotation = MODULE.relative_error(
            identity, estimated_end_tcw, identity, end_twc_gt
        )
        self.assertAlmostEqual(translation, 0.0, places=9)
        self.assertAlmostEqual(rotation, MODULE.math.pi / 2.0, places=9)


if __name__ == "__main__":
    unittest.main()
