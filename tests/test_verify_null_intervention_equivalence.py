import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verify_null_intervention_equivalence.py"
)
SPEC = importlib.util.spec_from_file_location("null_equivalence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NullInterventionEquivalenceTest(unittest.TestCase):
    def make_run(self, root: Path, name: str, fingerprint: str = "map") -> Path:
        run = root / name
        (run / "replay_packets").mkdir(parents=True)
        execution = {
            field: "0" for field in MODULE.EXECUTION_FIELDS
        }
        execution.update(
            {
                "frame": "0",
                "timestamp": "1.0",
                "valid": "1",
                "tracking_state": "2",
                "prior_would_use_id": "-1",
                "prior_selected_id": "-1",
                "map_fingerprint": fingerprint,
                "freeze_protocol": MODULE.FREEZE_PROTOCOL,
                "freeze_requested": "1",
                "local_mapping_idle_ack": "1",
                "local_mapping_stopped_ack": "1",
                "loop_closing_idle_ack": "1",
                "loop_closing_stopped_ack": "1",
                "gba_stopped_ack": "1",
                "freeze_epoch": "1",
            }
        )
        with (run / "replay_execution_state.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=MODULE.EXECUTION_FIELDS)
            writer.writeheader()
            writer.writerow(execution)
        with (run / "frame_metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=("frame", "prior_used")
            )
            writer.writeheader()
            writer.writerow({"frame": "0", "prior_used": "0"})
        packet_fields = ("packet", *MODULE.PACKET_INDEX_FIELDS)
        with (run / "replay_packets" / "index.csv").open(
            "w", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=packet_fields)
            writer.writeheader()
            packet = {field: "0" for field in packet_fields}
            packet.update(
                {
                    "packet": "frame_000001.yml.gz",
                    "frame": "1",
                    "timestamp": "1.1",
                    "freeze_protocol": MODULE.FREEZE_PROTOCOL,
                    "freeze_requested": "1",
                    "local_mapping_idle_ack": "1",
                    "local_mapping_stopped_ack": "1",
                    "loop_closing_idle_ack": "1",
                    "loop_closing_stopped_ack": "1",
                    "gba_stopped_ack": "1",
                    "freeze_epoch": "1",
                }
            )
            writer.writerow(packet)
        (run / "CameraTrajectory_AllFrames_TUM.txt").write_text(
            "1.0 0 0 0 0 0 0 1\n"
        )
        return run

    def test_three_identical_runs_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [self.make_run(root, f"run_{index}") for index in range(3)]
            with mock.patch.object(MODULE, "verify_replay_packets"):
                report = MODULE.evaluate(runs, Path("/unused"))
            self.assertEqual(report["status"], "pass")

    def test_reports_first_field_difference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [
                self.make_run(root, "run_0"),
                self.make_run(root, "run_1", fingerprint="changed"),
                self.make_run(root, "run_2"),
            ]
            with mock.patch.object(MODULE, "verify_replay_packets"):
                report = MODULE.evaluate(runs, Path("/unused"))
            self.assertEqual(report["status"], "fail")
            self.assertEqual(
                report["first_difference"]["field"], "map_fingerprint"
            )

    def test_reports_descriptor_hash_difference_before_pose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [self.make_run(root, f"run_{index}") for index in range(3)]
            rows = MODULE.read_csv(
                runs[1] / "replay_execution_state.csv",
                MODULE.EXECUTION_FIELDS,
            )
            rows[0]["current_descriptors_hash"] = "changed"
            rows[0]["current_pose_hash"] = "changed_pose"
            with (runs[1] / "replay_execution_state.csv").open(
                "w", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=MODULE.EXECUTION_FIELDS
                )
                writer.writeheader()
                writer.writerows(rows)
            with mock.patch.object(MODULE, "verify_replay_packets"):
                report = MODULE.evaluate(runs, Path("/unused"))
            self.assertEqual(report["status"], "fail")
            self.assertEqual(
                report["first_difference"]["field"],
                "current_descriptors_hash",
            )

    def test_rejects_nonzero_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory), "run")
            rows = MODULE.read_csv(
                run / "replay_execution_state.csv",
                MODULE.EXECUTION_FIELDS,
            )
            rows[0]["prior_selected_id"] = "10"
            with (run / "replay_execution_state.csv").open(
                "w", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=MODULE.EXECUTION_FIELDS
                )
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Null intervention"):
                MODULE.verify_run(run, Path("/unused"))

    def test_rejects_incomplete_freeze_ack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(Path(directory), "run")
            index_path = run / "replay_packets" / "index.csv"
            rows = MODULE.read_csv(
                index_path, ("packet", *MODULE.PACKET_INDEX_FIELDS)
            )
            rows[0]["loop_closing_stopped_ack"] = "0"
            with index_path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=("packet", *MODULE.PACKET_INDEX_FIELDS),
                )
                writer.writeheader()
                writer.writerows(rows)
            with mock.patch.object(MODULE, "verify_replay_packets"):
                with self.assertRaisesRegex(
                    ValueError, "Incomplete map-freeze"
                ):
                    MODULE.verify_run(run, Path("/unused"))


if __name__ == "__main__":
    unittest.main()
