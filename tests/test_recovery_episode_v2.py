from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREPARE = load_script("prepare_recovery_episodes.py")
EVALUATE = load_script("evaluate_recovery_episode_v2.py")
SUMMARIZE = load_script("summarize_recovery_episode_v2.py")


class RecoveryEpisodeV2Test(unittest.TestCase):
    def test_freeze_uses_static_recovery_and_greedy_nonoverlap(self) -> None:
        rows = []
        for checkpoint, start, end, entry, minimum, final in (
            ("frame_000010.yml.gz", 1.0, 2.0, 240, 140, 180),
            ("frame_000011.yml.gz", 1.5, 2.5, 230, 130, 190),
            ("frame_000020.yml.gz", 2.0, 3.0, 250, 150, 151),
            ("frame_000030.yml.gz", 3.0, 4.0, 151, 151, 200),
        ):
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "horizon": "10",
                    "start_timestamp": str(start),
                    "end_timestamp": str(end),
                    "static_success": "1",
                    "static_first_inliers": str(entry),
                    "static_min_inliers": str(minimum),
                    "static_final_inliers": str(final),
                }
            )
        events = PREPARE.freeze_episodes(
            rows, "tum_walking_xyz", 0, scan_sha256="abc"
        )
        self.assertEqual(
            [event["checkpoint"] for event in events],
            ["frame_000010.yml.gz", "frame_000020.yml.gz"],
        )
        self.assertEqual(events[0]["event_id"], "tum_walking_xyz_seed0_e001")

    def write_evaluation_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        ground_truth = root / "groundtruth.txt"
        ground_truth.write_text(
            "1.0 0 0 0 0 0 0 1\n"
            "2.0 1 0 0 0 0 0 1\n"
        )
        manifest = root / "episodes.csv"
        with manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "event_id",
                    "checkpoint",
                    "horizon",
                    "static_min_inliers",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "event_id": "tum_walking_xyz_seed0_e001",
                    "checkpoint": "frame_000010.yml.gz",
                    "horizon": "10",
                    "static_min_inliers": "120",
                }
            )

        row = {
            "event_id": "tum_walking_xyz_seed0_e001",
            "checkpoint": "frame_000010.yml.gz",
            "horizon": "10",
            "start_timestamp": "1.0",
            "end_timestamp": "2.0",
            "static_scan_only": "0",
            "static_executed": "1",
            "dynamic_executed": "1",
            "matched_sham_static_executed": "1",
            "matched_sham_dynamic_executed": "1",
            "matched_sham_selected_branch": "static",
            "prior_subspace": "translation",
            "replay_mode": "posterior-local-map-v1",
            "max_static_information_leverage": "0.025",
            "static_information_leverage_mode": "normalize-to-target",
            "static_success": "1",
            "velocity_neutral_success": "1",
            "shadow_success": "1",
            "static_first_inliers": "140",
            "static_min_inliers": "120",
            "static_final_inliers": "180",
            "velocity_neutral_first_inliers": "142",
            "velocity_neutral_min_inliers": "121",
            "velocity_neutral_final_inliers": "181",
        }
        for prefix in (
            "start_tcw",
            "static_tcw",
            "velocity_neutral_tcw",
            "shadow_tcw",
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
        row["static_tcw_tx"] = "-1.2"
        row["shadow_tcw_tx"] = "-1.2"
        row["velocity_neutral_tcw_tx"] = "-1.05"
        forks = root / "forks.csv"
        with forks.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(row))
            writer.writeheader()
            writer.writerow(row)
        return forks, manifest, ground_truth

    def test_oracle_selects_one_complete_velocity_neutral_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            forks, manifest, ground_truth = self.write_evaluation_fixture(
                Path(directory)
            )
            summary, rows = EVALUATE.evaluate(
                forks,
                manifest,
                ground_truth,
                "tum_walking_xyz",
                0,
            )
            self.assertEqual(rows[0]["oracle_selected_branch"], "dynamic")
            self.assertAlmostEqual(rows[0]["static_translation_rpe_m"], 0.2)
            self.assertAlmostEqual(rows[0]["dynamic_translation_rpe_m"], 0.05)
            self.assertAlmostEqual(rows[0]["oracle_translation_rpe_m"], 0.05)
            self.assertTrue(summary["matched_sham_exact_static"])
            self.assertTrue(summary["oracle_strict_translation_improvement"])

    def test_rejects_sham_that_does_not_publish_static(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            forks, manifest, ground_truth = self.write_evaluation_fixture(
                Path(directory)
            )
            with forks.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["shadow_tcw_tx"] = "-1.1"
            with forks.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "exactly equivalent"):
                EVALUATE.evaluate(
                    forks,
                    manifest,
                    ground_truth,
                    "tum_walking_xyz",
                    0,
                )

    def test_oracle_ties_and_dynamic_failure_choose_static(self) -> None:
        self.assertEqual(
            EVALUATE.choose_oracle(True, True, 0.1, 0.01, 0.1, 0.01),
            "static",
        )

    def test_aggregate_requires_all_six_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for sequence in ("tum_walking_xyz", "bonn_crowd"):
                for seed in (0, 1, 2):
                    result = {
                        "protocol": SUMMARIZE.PROTOCOL,
                        "sequence": sequence,
                        "seed": seed,
                        "events": 4,
                        "matched_sham_exact_static": True,
                        "oracle_strict_translation_improvement": not (
                            sequence == "bonn_crowd" and seed == 2
                        ),
                        "oracle_mean_rotation_non_regression": True,
                        "oracle_dynamic_selections": 1,
                        "methods": {
                            "static": {
                                "translation_rpe_m": {"mean": 0.2},
                                "rotation_rpe_rad": {"mean": 0.02},
                            },
                            "dynamic": {
                                "translation_rpe_m": {"mean": 0.3},
                            },
                            "oracle": {
                                "translation_rpe_m": {"mean": 0.1},
                                "rotation_rpe_rad": {"mean": 0.02},
                            },
                        },
                    }
                    path = root / f"{sequence}_{seed}.json"
                    path.write_text(json.dumps(result))
                    paths.append(path)
            summary = SUMMARIZE.summarize(paths)
            self.assertEqual(summary["verdict"], "scientific_no_go")
            self.assertEqual(summary["oracle_translation_pass_cells"], 5)
        self.assertEqual(
            EVALUATE.choose_oracle(True, False, 0.1, 0.01, None, None),
            "static",
        )


if __name__ == "__main__":
    unittest.main()
