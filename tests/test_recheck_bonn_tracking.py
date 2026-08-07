from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "recheck_bonn_tracking.py"
)
SPEC = importlib.util.spec_from_file_location("recheck_bonn_tracking", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BonnTrackingRecheckTest(unittest.TestCase):
    def test_parse_rmse(self) -> None:
        self.assertAlmostEqual(MODULE.parse_rmse("rmse  0.012345\n"), 0.012345)

    def test_primary_and_auxiliary_metric_keys(self) -> None:
        metrics = {
            "ate_rmse_m": 0.02,
            "valid_optimized_ate_rmse_m": 0.03,
        }
        self.assertEqual(
            MODULE.original_metric(metrics, "all_input_frames", "ate_rmse_m"),
            0.02,
        )
        self.assertEqual(
            MODULE.original_metric(
                metrics, "valid_optimized_frames", "ate_rmse_m"
            ),
            0.03,
        )

    def test_rpe_command_uses_one_frame_delta(self) -> None:
        command = MODULE.evo_command(
            "rpe_translation_rmse_m",
            Path("/gt.txt"),
            Path("/trajectory.txt"),
            Path("/result.zip"),
        )
        self.assertEqual(command[:2], ["evo_rpe", "tum"])
        self.assertIn("1", command)
        self.assertIn("f", command)


if __name__ == "__main__":
    unittest.main()
