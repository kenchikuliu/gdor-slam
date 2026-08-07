from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_mapping_heldout.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_mapping_heldout", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PoseMathTest(unittest.TestCase):
    def test_quaternion_matrix_round_trip(self) -> None:
        half_angle = 0.37
        quaternion = np.asarray([
            0.0, math.sin(half_angle), 0.0, math.cos(half_angle)])
        matrix = MODULE.quaternion_to_matrix(quaternion)
        recovered = MODULE.matrix_to_quaternion(matrix)

        np.testing.assert_allclose(recovered, quaternion, atol=1e-9)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0)

    def test_rigid_alignment_recovers_transform(self) -> None:
        source = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        angle = 0.42
        rotation = np.asarray([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation = np.asarray([1.5, -0.4, 0.8])
        target = (rotation @ source.T).T + translation

        recovered_rotation, recovered_translation, rmse = (
            MODULE.rigid_alignment(source, target))

        np.testing.assert_allclose(recovered_rotation, rotation, atol=1e-9)
        np.testing.assert_allclose(
            recovered_translation, translation, atol=1e-9)
        self.assertLess(rmse, 1e-9)

    def test_nearest_pose_respects_maximum_delta(self) -> None:
        poses = [
            MODULE.Pose(1.0, np.zeros(3), np.asarray([0.0, 0.0, 0.0, 1.0])),
            MODULE.Pose(2.0, np.ones(3), np.asarray([0.0, 0.0, 0.0, 1.0])),
        ]

        self.assertIs(poses[1], MODULE.nearest_pose(poses, 1.98, 0.03))
        self.assertIsNone(MODULE.nearest_pose(poses, 1.90, 0.03))

    def test_bonn_rgbd_sensor_pose_uses_official_fixed_calibration(self) -> None:
        pose = MODULE.Pose(
            1.0,
            np.asarray([0.2, -0.3, 1.4]),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
        )
        transformed = MODULE.transform_bonn_rgbd_sensor_pose(pose)
        marker_pose = np.eye(4)
        marker_pose[:3, 3] = pose.translation
        expected = (
            MODULE.BONN_T_ROS_INVERSE
            @ marker_pose
            @ MODULE.BONN_T_ROS
            @ MODULE.BONN_T_M
        )

        np.testing.assert_allclose(
            transformed.translation, expected[:3, 3], atol=1e-12)
        rotation = MODULE.quaternion_to_matrix(
            transformed.quaternion_xyzw)
        np.testing.assert_allclose(
            rotation,
            MODULE.closest_rotation_matrix(expected[:3, :3]),
            atol=1e-9,
        )
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_final_map_cameras_supply_alignment_poses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            path.write_text(json.dumps([
                {
                    "img_name": "rgb/2.000000.png",
                    "position": [1.0, 2.0, 3.0],
                    "rotation": np.eye(3).tolist(),
                },
                {
                    "img_name": "rgb/1.000000.png",
                    "position": [0.0, 0.0, 0.0],
                    "rotation": np.eye(3).tolist(),
                },
            ]))

            poses = MODULE.load_final_map_cameras(path)

            self.assertEqual([pose.timestamp for pose in poses], [1.0, 2.0])
            np.testing.assert_allclose(poses[1].translation, [1.0, 2.0, 3.0])

    def test_alignment_ignores_online_trajectory(self) -> None:
        ground_truth = [
            MODULE.Pose(
                float(index),
                np.asarray([float(index), float(index % 2), 0.0]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            )
            for index in range(4)
        ]
        translation = np.asarray([3.0, -2.0, 0.5])
        final_map_keyframes = [
            MODULE.Pose(
                pose.timestamp,
                pose.translation + translation,
                pose.quaternion_xyzw,
            )
            for pose in ground_truth
        ]
        run = types.SimpleNamespace(
            final_map_keyframes=final_map_keyframes,
            trajectory=[
                MODULE.Pose(
                    pose.timestamp,
                    pose.translation + np.asarray([100.0, 0.0, 0.0]),
                    pose.quaternion_xyzw,
                )
                for pose in ground_truth
            ],
        )

        rotation, recovered_translation, rmse, pairs = (
            MODULE.alignment_for_run(ground_truth, run, 1e-6))

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(
            recovered_translation, translation, atol=1e-9)
        self.assertLess(rmse, 1e-9)
        self.assertEqual(pairs, 4)

    def test_nonzero_distortion_is_explicitly_rejected(self) -> None:
        view = {
            "width": 640.0,
            "height": 480.0,
            "fx": 525.0,
            "fy": 525.0,
            "cx": 319.5,
            "cy": 239.5,
            "k1": 0.1,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0,
        }
        orb = dict(view)
        orb["k1"] = 0.0

        with self.assertRaisesRegex(ValueError, "non-zero distortion"):
            MODULE.validate_camera_compatibility(view, orb)

    def test_nonzero_orb_distortion_is_explicitly_rejected(self) -> None:
        view = {
            "width": 640.0,
            "height": 480.0,
            "fx": 525.0,
            "fy": 525.0,
            "cx": 319.5,
            "cy": 239.5,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "k3": 0.0,
        }
        orb = dict(view)
        orb["p1"] = 0.01

        with self.assertRaisesRegex(ValueError, "non-zero distortion in the ORB"):
            MODULE.validate_camera_compatibility(view, orb)

    def test_bonn_rendering_requires_explicit_sensor_pose_model(self) -> None:
        MODULE.validate_groundtruth_pose_model(
            "bonn_crowd", "bonn_rgbd_sensor")
        MODULE.validate_groundtruth_pose_model("tum_walking_xyz", "camera")
        with self.assertRaisesRegex(
            ValueError, "requires.*bonn_rgbd_sensor"
        ):
            MODULE.validate_groundtruth_pose_model("bonn_crowd", "camera")
        with self.assertRaisesRegex(ValueError, "deprecated"):
            MODULE.validate_groundtruth_pose_model("bonn_crowd", "bonn_mocap")
        with self.assertRaisesRegex(ValueError, "cannot be applied"):
            MODULE.validate_groundtruth_pose_model(
                "tum_walking_xyz", "bonn_rgbd_sensor")

    def test_run_identity_binds_paths_hashes_and_camera_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            association = dataset / "associations.txt"
            ground_truth = dataset / "groundtruth.txt"
            orb_config = root / "orb.yaml"
            gaussian_config = root / "gaussian.yaml"
            view_config = root / "view.yaml"
            association.write_text("1.0 rgb/1.0.png 1.0 depth/1.0.png\n")
            ground_truth.write_text("1.0 0 0 0 0 0 0 1\n")
            gaussian_config.write_text("Model.sh_degree: 3\n")
            orb_config.write_text(
                "%YAML:1.0\n"
                "Camera.width: 640\nCamera.height: 480\n"
                "Camera1.fx: 525.0\nCamera1.fy: 525.0\n"
                "Camera1.cx: 319.5\nCamera1.cy: 239.5\n"
                "Camera1.k1: 0.0\nCamera1.k2: 0.0\n"
                "Camera1.p1: 0.0\nCamera1.p2: 0.0\n"
            )
            view_config.write_text(
                "%YAML:1.0\n"
                "Camera.w: 640\nCamera.h: 480\n"
                "Camera.fx: 525.0\nCamera.fy: 525.0\n"
                "Camera.cx: 319.5\nCamera.cy: 239.5\n"
                "Camera.k1: 0.0\nCamera.k2: 0.0\n"
                "Camera.p1: 0.0\nCamera.p2: 0.0\n"
                "Camera.k3: 0.0\n"
            )
            manifest = {
                "sequence": "tum_walking_xyz",
                "seed": 0,
                "source": {
                    "runtime_source_snapshot_sha256": "source-snapshot",
                },
                "paths": {
                    "dataset": str(dataset),
                    "association": str(association),
                    "ground_truth": str(ground_truth),
                },
                "hashes": {
                    "association_sha256": MODULE.sha256(association),
                    "ground_truth_sha256": MODULE.sha256(ground_truth),
                    "orb_config_sha256": MODULE.sha256(orb_config),
                    "gaussian_config_sha256": MODULE.sha256(gaussian_config),
                },
            }
            run = types.SimpleNamespace(
                name="semantic",
                manifest=manifest,
                orb_config=orb_config,
                gaussian_config=gaussian_config,
            )

            identity = MODULE.validate_run_identities(
                [run], dataset, association, ground_truth, view_config)

            self.assertEqual(identity["sequence"], "tum_walking_xyz")
            self.assertEqual(identity["seed"], 0)
            self.assertEqual(identity["view_camera_model"]["width"], 640.0)

            manifest["hashes"]["association_sha256"] = "stale"
            with self.assertRaisesRegex(ValueError, "stale association_sha256"):
                MODULE.validate_run_identities(
                    [run], dataset, association, ground_truth, view_config)


if __name__ == "__main__":
    unittest.main()
