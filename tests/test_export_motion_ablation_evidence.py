from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "export_motion_ablation_evidence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "export_motion_ablation_evidence", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MotionAblationEvidenceTest(unittest.TestCase):
    def test_median_seed_is_selected_by_semantic_ate(self) -> None:
        rows = [
            {"seed": 0, "metrics": {"ate_rmse_m": 0.021}},
            {"seed": 1, "metrics": {"ate_rmse_m": 0.023}},
            {"seed": 2, "metrics": {"ate_rmse_m": 0.020}},
        ]

        seed, ate, ranks = MODULE.select_median_seed(rows)

        self.assertEqual(seed, 0)
        self.assertAlmostEqual(ate, 0.021)
        self.assertEqual(ranks, {2: 1, 0: 2, 1: 3})

    def test_even_seed_set_is_rejected(self) -> None:
        rows = [
            {"seed": 0, "metrics": {"ate_rmse_m": 0.021}},
            {"seed": 1, "metrics": {"ate_rmse_m": 0.023}},
        ]
        with self.assertRaisesRegex(ValueError, "odd"):
            MODULE.select_median_seed(rows)

    def test_public_control_name_is_lag_30(self) -> None:
        self.assertEqual(
            MODULE.DISPLAY_NAMES["semantic_motion_rgbd_shuffled"], "Lag-30"
        )


if __name__ == "__main__":
    unittest.main()
