import json
from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import prepare_matched_oracle_poses as oracle_poses


def test_rotation_round_trip_identity():
    quaternion = oracle_poses.rotation_matrix_to_quaternion(np.eye(3))
    np.testing.assert_allclose(quaternion, [0.0, 0.0, 0.0, 1.0])


def test_prepare_writes_tcw_and_invalid_boundary(tmp_path):
    association = tmp_path / "associations.txt"
    association.write_text(
        "0.0 rgb/0.png 0.0 depth/0.png\n"
        "0.5 rgb/1.png 0.5 depth/1.png\n"
        "1.5 rgb/2.png 1.5 depth/2.png\n")
    ground_truth = tmp_path / "groundtruth.txt"
    ground_truth.write_text(
        "0.0 0 0 0 0 0 0 1\n"
        "1.0 1 0 0 0 0 0 1\n")
    output = tmp_path / "manifest" / "poses.csv"
    metadata_output = tmp_path / "manifest" / "poses.metadata.json"

    metadata = oracle_poses.prepare(
        association, ground_truth, output, metadata_output, 1.1)

    rows = output.read_text().splitlines()
    assert rows[0] == oracle_poses.HEADER
    assert rows[1].split(",")[2:] == [
        "1", "0.000000000", "0.000000000", "0.000000000",
        "0.000000000", "0.000000000", "0.000000000", "1.000000000",
    ]
    assert rows[2].split(",")[2:6] == [
        "1", "-0.500000000", "0.000000000", "0.000000000"]
    assert rows[3].split(",")[2] == "0"
    assert metadata["valid_ground_truth_frames"] == 2
    assert json.loads(metadata_output.read_text())["pose_convention"] == "Tcw"
