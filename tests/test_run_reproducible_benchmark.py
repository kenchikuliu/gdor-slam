from __future__ import annotations

import cv2
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
import zipfile


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_reproducible_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark", SCRIPT)
BENCHMARK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCHMARK)


def complete(config: str, sequence: str, seed: int, ate: float) -> dict:
    has_motion = config in {
        "semantic_motion",
        "semantic_motion_shadow",
        "semantic_motion_consensus",
        "semantic_motion_consensus_shadow",
        "semantic_motion_tempered",
        "semantic_motion_tempered_shadow",
        "semantic_motion_direct",
        "semantic_motion_direct_shadow",
        "semantic_motion_rgbd",
        "semantic_motion_rgbd_shadow",
        "semantic_motion_init_rgbd",
        "semantic_motion_ungated",
        "semantic_motion_rgbd_shuffled",
        "semantic_motion_rgbd_shadow_translation",
        "semantic_motion_rgbd_shadow_translation_shadow",
        "semantic_motion_rgbd_shadow_translation_lag30",
    }
    uses_temporal = config in {
        "dypho_temporal", "dypho_compatible", "dypho_decoupled",
        "dypho_flow_guarded", "dypho_flow_exact", "dypho_flow_adaptive"}
    uses_adaptive = config in {
        "dypho_feature", "dypho_compatible", "dypho_decoupled",
        "dypho_flow_guarded", "dypho_flow_exact", "dypho_flow_adaptive"}
    result = {
        "status": "complete",
        "config": config,
        "sequence": sequence,
        "seed": seed,
        "metrics": {
            "ate_rmse_m": ate,
            "ate_p95_m": ate,
            "rpe_translation_rmse_m": ate,
            "rpe_translation_p95_m": ate,
            "rpe_rotation_rmse_deg": ate,
            "rpe_rotation_p95_deg": ate,
            "failure_rate": 0.0,
            "end_to_end_seconds": 1.0,
            "trajectory_coverage": 1.0,
            "carried_forward_rate": 0.0,
            "valid_motion_priors": 1 if has_motion else 0,
            "candidate_motion_priors": 1 if has_motion else 0,
            "would_use_motion_priors": 1 if has_motion else 0,
            "used_motion_priors": 1 if config in {
                "semantic_motion",
                "semantic_motion_consensus",
                "semantic_motion_tempered",
                "semantic_motion_direct",
                "semantic_motion_rgbd",
                "semantic_motion_init_rgbd",
                "semantic_motion_ungated",
                "semantic_motion_rgbd_shuffled",
                "semantic_motion_rgbd_shadow_translation",
                "semantic_motion_rgbd_shadow_translation_lag30",
            } else 0,
            "shadow_translation_applied_frames":
                1 if config in {
                    "semantic_motion_rgbd_shadow_translation",
                    "semantic_motion_rgbd_shadow_translation_lag30",
                } else 0,
            "shadow_translation_factor_injections":
                1 if config in {
                    "semantic_motion_rgbd_shadow_translation",
                    "semantic_motion_rgbd_shadow_translation_lag30",
                } else 0,
            "velocity_neutralized_frames":
                1 if config in {
                    "semantic_motion_rgbd_shadow_translation",
                    "semantic_motion_rgbd_shadow_translation_lag30",
                } else 0,
            "mask_pose_predicted_frames": 2 if uses_temporal else 0,
            "temporal_refinement_frames": 1 if uses_temporal else 0,
            "temporal_recovered_static_pixels":
                11 if uses_temporal else 0,
            "temporal_added_dynamic_pixels":
                7 if uses_temporal else 0,
            "temporal_flow_guard_valid_frames":
                10 if config in {
                    "dypho_flow_guarded", "dypho_flow_exact",
                    "dypho_flow_adaptive"} else 0,
            "temporal_flow_guard_rejected_pixels":
                100 if config in {
                    "dypho_flow_guarded", "dypho_flow_exact",
                    "dypho_flow_adaptive"} else 0,
            "tracking_recovery_audit_pixels":
                1000 if config in {
                    "dypho_flow_guarded", "dypho_flow_exact",
                    "dypho_flow_adaptive"} else 0,
            "tracking_recovery_mapping_leak_pixels": 0,
            "static_mask_ratio_mean": 0.8,
            "mapping_weight_frames": 100,
            "mapping_static_ratio_mean": 0.75,
            "mapping_keyframes_with_static_weight": 12,
            "mapping_weight_pixels": 1000,
            "mapping_static_pixels": 750,
            "mapping_orb_map_points_considered": 200,
            "mapping_orb_map_points_rejected_static_mask": 20,
            "mapping_rgbd_densification_candidates": 400,
            "mapping_rgbd_densification_rejected_static_mask": 80,
            "adaptive_feature_active_frames":
                25 if uses_adaptive else 0,
            "adaptive_fast_threshold_mean":
                15.0 if uses_adaptive else 20.0,
            "extracted_features_mean":
                450.0 if uses_adaptive else 300.0,
            "synchronize_local_mapping": True,
            "synchronize_loop_closing": True,
            "gaussian_mapper_enabled": False,
        },
    }
    result["source"] = {
        "commit": "synthetic-clean-commit",
        "dirty": False,
        "runtime_source_snapshot_sha256": "synthetic-clean-snapshot",
    }
    return result


def failed(config: str, sequence: str, seed: int) -> dict:
    return {
        "status": "failed",
        "config": config,
        "sequence": sequence,
        "seed": seed,
        "error": "synthetic failure",
    }


def freeze_dyn19_main_plan(root: Path) -> list[dict]:
    tasks, _ = BENCHMARK.build_task_blocks(
        list(BENCHMARK.DYN19_MAIN_CONFIGS),
        list(BENCHMARK.DYN19_CLAIM_SEQUENCES),
        list(BENCHMARK.DYN19_REQUIRED_SEEDS),
        ["0"],
        0,
        {"commit": "synthetic-clean-commit", "dirty": False},
        root,
        export_static_masks=True,
        dyn19_phase="main",
    )
    payload = {
        "contract": "reproducible-benchmark-plan-v3",
        "tasks": tasks,
        "dyn19": {
            "experiment_id": BENCHMARK.DYN19_EXPERIMENT_ID,
            "phase": "main",
            "claim_bearing_sequences": BENCHMARK.DYN19_CLAIM_SEQUENCES,
            "required_seeds": BENCHMARK.DYN19_REQUIRED_SEEDS,
            "phase_configs": BENCHMARK.DYN19_MAIN_CONFIGS,
            "factor_matrix": BENCHMARK.DYN19_FACTOR_MATRIX,
            "config_contract": BENCHMARK.dyn19_config_contract(),
        },
    }
    BENCHMARK.freeze_benchmark_plan(root, payload, tasks)
    return tasks


def freeze_dyn20_heldout_plan(root: Path) -> list[dict]:
    tasks, _ = BENCHMARK.build_task_blocks(
        list(BENCHMARK.DYN20_CONFIGS),
        list(BENCHMARK.DYN20_HELDOUT_SEQUENCES),
        list(BENCHMARK.DYN20_REQUIRED_SEEDS),
        ["0"],
        0,
        {"commit": "synthetic-clean-commit", "dirty": False},
        root,
        dyn20_heldout=True,
    )
    payload = {
        "contract": "reproducible-benchmark-plan-v3",
        "tasks": tasks,
        "dyn20": {
            "experiment_id": BENCHMARK.DYN20_EXPERIMENT_ID,
            "candidate": BENCHMARK.DYN20_CANDIDATE,
            "selection_source_experiment_id": BENCHMARK.DYN19_EXPERIMENT_ID,
            "config_contract": BENCHMARK.dyn19_config_contract(),
        },
    }
    BENCHMARK.freeze_benchmark_plan(root, payload, tasks)
    return tasks


class AggregateGateTest(unittest.TestCase):
    def run_aggregate(self, results: list[dict]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        BENCHMARK.aggregate(root, results)
        return root

    def run_flow_adaptive_gate(self, results: list[dict]) -> dict:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        tasks = [
            {
                "config": result["config"],
                "sequence": result["sequence"],
                "seed": result["seed"],
                "heldout_stride": 0,
                "disable_gaussian_mapper": False,
            }
            for result in results
        ]
        plan = {
            "contract": "reproducible-benchmark-plan-v3",
            "tasks": tasks,
            "flow_adaptive_incumbent_gate": {
                "predeclared": True,
                "config_contract":
                    BENCHMARK.flow_adaptive_config_contract(),
            },
        }
        BENCHMARK.atomic_json(root / "benchmark_plan.json", plan)
        digest = BENCHMARK.sha256(root / "benchmark_plan.json")
        (root / "benchmark_plan.sha256").write_text(
            f"{digest}  benchmark_plan.json\n")
        BENCHMARK.aggregate(root, results)
        return json.loads(
            (root / "flow_adaptive_incumbent_gate.json").read_text())

    def flow_adaptive_results(
        self, candidate_ate: float = 0.095, parent_ate: float = 0.100
    ) -> list[dict]:
        results = []
        for sequence in BENCHMARK.FLOW_ADAPTIVE_DEVELOPMENT_SEQUENCES:
            for seed in BENCHMARK.FLOW_ADAPTIVE_REQUIRED_SEEDS:
                parent = complete(
                    BENCHMARK.FLOW_ADAPTIVE_PARENT_CONFIG,
                    sequence, seed, parent_ate)
                candidate = complete(
                    BENCHMARK.FLOW_ADAPTIVE_CANDIDATE_CONFIG,
                    sequence, seed, candidate_ate)
                parent["metrics"][
                    "temporal_flow_guard_rejected_pixels"] = 120
                parent["metrics"][
                    "tracking_recovery_audit_pixels"] = 900
                candidate["metrics"][
                    "temporal_flow_guard_rejected_pixels"] = 100
                candidate["metrics"][
                    "tracking_recovery_audit_pixels"] = 1000
                results.extend((parent, candidate))
        return results

    def test_flow_adaptive_incumbent_gate_can_pass(self) -> None:
        gate = self.run_flow_adaptive_gate(
            self.flow_adaptive_results())

        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["decision"], "DEVELOPMENT_ADVANCE")
        self.assertTrue(gate["execution_integrity"])
        self.assertEqual(gate["sequence_passes"], 3)
        self.assertEqual(gate["paired_ate_wins"], 9)

    def test_flow_adaptive_gate_rejects_dirty_source_first(self) -> None:
        results = self.flow_adaptive_results()
        results[0]["source"]["dirty"] = True

        gate = self.run_flow_adaptive_gate(results)

        self.assertEqual(gate["status"], "invalid_execution")
        self.assertEqual(gate["decision"], "REPAIR_INFRASTRUCTURE")
        self.assertFalse(gate["source_clean"])

    def test_flow_adaptive_gate_marks_missing_run_incomplete(self) -> None:
        results = self.flow_adaptive_results()[:-1]

        gate = self.run_flow_adaptive_gate(results)

        self.assertEqual(gate["status"], "incomplete")
        self.assertEqual(gate["decision"], "NO_DECISION")
        self.assertEqual(len(gate["missing_identities"]), 1)

    def test_flow_adaptive_gate_fails_closed_on_mapping_leak(self) -> None:
        results = self.flow_adaptive_results()
        results[-1]["metrics"][
            "tracking_recovery_mapping_leak_pixels"] = 1

        gate = self.run_flow_adaptive_gate(results)

        self.assertEqual(gate["status"], "safety_failure")
        self.assertEqual(gate["decision"], "SAFETY_FAILURE")
        self.assertFalse(
            gate["sequences"]["bonn_crowd"]["mapping_leak_free"])

    def test_flow_adaptive_gate_stops_scientific_failure(self) -> None:
        gate = self.run_flow_adaptive_gate(
            self.flow_adaptive_results(candidate_ate=0.100))

        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["decision"], "STOP_ADAPTIVE_RADIUS")

    def test_yolo_engine_contract_hashes_required_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "detector.engine"
            engine.write_bytes(b"engine bytes")
            config = root / "mask.yaml"
            config.write_text(
                "%YAML:1.0\n"
                "mask.use_yolo: 1\n"
                "mask.use_external_mask: 0\n"
                f'mask.yolo_model_path: "{engine}"\n'
            )

            contract = BENCHMARK.yolo_engine_contract(config)

            self.assertTrue(contract["required"])
            self.assertEqual(contract["path"], str(engine.resolve()))
            self.assertEqual(
                contract["sha256"], BENCHMARK.sha256(engine))

    def test_yolo_engine_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "mask.yaml"
            config.write_text(
                "%YAML:1.0\n"
                "mask.use_yolo: 1\n"
                "mask.use_external_mask: 0\n"
                f'mask.yolo_model_path: "{root / "missing.engine"}"\n'
            )

            with self.assertRaisesRegex(
                FileNotFoundError, "YOLO engine not found"
            ):
                BENCHMARK.yolo_engine_contract(config)

    def test_reads_evo_error_percentile_from_result_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.zip"
            payload = io.BytesIO()
            BENCHMARK.np.save(
                payload, BENCHMARK.np.asarray([0.0, 1.0, 2.0, 3.0]))
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("error_array.npy", payload.getvalue())

            value = BENCHMARK.result_error_percentile(path, 95.0)

            self.assertAlmostEqual(value, 2.85)

    def test_shadow_translation_tracking_gate_passes_all_controls(self) -> None:
        results = []
        for sequence in BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES:
            for seed in BENCHMARK.SHADOW_TRANSLATION_REQUIRED_SEEDS:
                for index, config in enumerate(
                    BENCHMARK.SHADOW_TRANSLATION_CONTROLS
                ):
                    results.append(
                        complete(config, sequence, seed, 0.20 + 0.01 * index))
                results.append(
                    complete(
                        BENCHMARK.SHADOW_TRANSLATION_CONFIG,
                        sequence, seed, 0.10))
        by_key = {
            (result["config"], result["sequence"], result["seed"]): result
            for result in results
        }
        planned = {
            (config, sequence):
                set(BENCHMARK.SHADOW_TRANSLATION_REQUIRED_SEEDS)
            for config in BENCHMARK.SHADOW_TRANSLATION_TRACKING_CONFIGS
            for sequence in BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            BENCHMARK.write_shadow_translation_tracking_gate(
                root, by_key, planned,
                BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES,
                "gate.json", "development")
            gate = json.loads((root / "gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        self.assertTrue(gate["shadow_translation_tracking_retained"])

    def test_shadow_translation_tracking_gate_rejects_lag30_loss(self) -> None:
        results = []
        for sequence in BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES:
            for seed in BENCHMARK.SHADOW_TRANSLATION_REQUIRED_SEEDS:
                for config in BENCHMARK.SHADOW_TRANSLATION_CONTROLS:
                    results.append(complete(config, sequence, seed, 0.20))
                results.append(
                    complete(
                        BENCHMARK.SHADOW_TRANSLATION_CONFIG,
                        sequence, seed, 0.30))
        by_key = {
            (result["config"], result["sequence"], result["seed"]): result
            for result in results
        }
        planned = {
            (config, sequence):
                set(BENCHMARK.SHADOW_TRANSLATION_REQUIRED_SEEDS)
            for config in BENCHMARK.SHADOW_TRANSLATION_TRACKING_CONFIGS
            for sequence in BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            BENCHMARK.write_shadow_translation_tracking_gate(
                root, by_key, planned,
                BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES,
                "gate.json", "development")
            gate = json.loads((root / "gate.json").read_text())

        self.assertEqual(gate["status"], "fail")
        self.assertFalse(gate["shadow_translation_tracking_retained"])

    def test_shadow_translation_gate_rejects_velocity_contract_mismatch(
        self,
    ) -> None:
        results = []
        for sequence in BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES:
            for seed in BENCHMARK.SHADOW_TRANSLATION_REQUIRED_SEEDS:
                for index, config in enumerate(
                    BENCHMARK.SHADOW_TRANSLATION_CONTROLS
                ):
                    results.append(
                        complete(config, sequence, seed, 0.20 + 0.01 * index))
                candidate = complete(
                    BENCHMARK.SHADOW_TRANSLATION_CONFIG,
                    sequence, seed, 0.10)
                if sequence == (
                    BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES[0]
                ) and seed == BENCHMARK.SHADOW_TRANSLATION_REQUIRED_SEEDS[0]:
                    candidate["metrics"]["velocity_neutralized_frames"] = 0
                results.append(candidate)
        by_key = {
            (result["config"], result["sequence"], result["seed"]): result
            for result in results
        }
        planned = {
            (config, sequence):
                set(BENCHMARK.SHADOW_TRANSLATION_REQUIRED_SEEDS)
            for config in BENCHMARK.SHADOW_TRANSLATION_TRACKING_CONFIGS
            for sequence in BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            BENCHMARK.write_shadow_translation_tracking_gate(
                root, by_key, planned,
                BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES,
                "gate.json", "development")
            gate = json.loads((root / "gate.json").read_text())

        self.assertEqual(gate["status"], "fail")
        self.assertFalse(gate["shadow_translation_tracking_retained"])
        self.assertFalse(
            gate["sequences"][
                BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES[0]
            ]["velocity_neutral_contract_pass"])

    def test_shadow_translation_gate_allows_rejected_factor_attempt(
        self,
    ) -> None:
        results = []
        for sequence in BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES:
            for seed in BENCHMARK.SHADOW_TRANSLATION_REQUIRED_SEEDS:
                for index, config in enumerate(
                    BENCHMARK.SHADOW_TRANSLATION_CONTROLS
                ):
                    results.append(
                        complete(config, sequence, seed, 0.20 + 0.01 * index))
                candidate = complete(
                    BENCHMARK.SHADOW_TRANSLATION_CONFIG,
                    sequence, seed, 0.10)
                if seed == BENCHMARK.SHADOW_TRANSLATION_REQUIRED_SEEDS[0]:
                    candidate["metrics"][
                        "shadow_translation_factor_injections"
                    ] = 2
                results.append(candidate)
        by_key = {
            (result["config"], result["sequence"], result["seed"]): result
            for result in results
        }
        planned = {
            (config, sequence):
                set(BENCHMARK.SHADOW_TRANSLATION_REQUIRED_SEEDS)
            for config in BENCHMARK.SHADOW_TRANSLATION_TRACKING_CONFIGS
            for sequence in BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            BENCHMARK.write_shadow_translation_tracking_gate(
                root, by_key, planned,
                BENCHMARK.SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES,
                "gate.json", "development")
            gate = json.loads((root / "gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        self.assertTrue(gate["shadow_translation_tracking_retained"])
        self.assertTrue(all(
            sequence["velocity_neutral_contract_pass"]
            for sequence in gate["sequences"].values()))

    def test_frozen_plan_binds_exact_task_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "config": "semantic",
                "sequence": "sequence_a",
                "seed": 0,
                "run_dir": str(root / "run"),
                "asset_contract": {"contract": "synthetic"},
            }
            payload = {
                "contract": "reproducible-benchmark-plan-v3",
                "tasks": [dict(task)],
            }
            BENCHMARK.freeze_benchmark_plan(root, payload, [task])

            BENCHMARK.validate_task_plan_binding(task)
            task["seed"] = 1
            with self.assertRaisesRegex(
                ValueError, "missing from or ambiguous"
            ):
                BENCHMARK.validate_task_plan_binding(task)

    def test_frozen_plan_rejects_plan_file_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "config": "semantic",
                "sequence": "sequence_a",
                "seed": 0,
                "run_dir": str(root / "run"),
            }
            payload = {
                "contract": "reproducible-benchmark-plan-v3",
                "tasks": [dict(task)],
            }
            BENCHMARK.freeze_benchmark_plan(root, payload, [task])
            (root / "benchmark_plan.json").write_text("{}\n")

            with self.assertRaisesRegex(
                ValueError, "SHA-256 changed"
            ):
                BENCHMARK.validate_task_plan_binding(task)

    def test_all_failed_planned_sequence_is_not_silently_dropped(self) -> None:
        results = [
            complete("semantic", "sequence_a", seed, 0.2)
            for seed in range(3)
        ]
        results.extend(
            failed("full", "sequence_a", seed)
            for seed in range(3)
        )
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "full_fusion_claim_gate.json").read_text())

        self.assertEqual(gate["status"], "insufficient_data")
        self.assertIn("sequence_a", gate["sequences"])
        self.assertEqual(
            gate["sequences"]["sequence_a"]["status"], "incomplete")
        self.assertEqual(
            len(gate["sequences"]["sequence_a"]["missing_runs"]), 3)

    def test_complete_full_gate_can_pass(self) -> None:
        results = []
        for seed in range(3):
            results.append(complete("semantic", "sequence_a", seed, 0.2))
            results.append(complete("full", "sequence_a", seed, 0.1))
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "full_fusion_claim_gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        self.assertEqual(
            gate["sequences"]["sequence_a"]["status"], "complete")

    def test_failed_motion_run_blocks_motion_gate(self) -> None:
        results = []
        for seed in range(3):
            results.append(complete("semantic", "sequence_a", seed, 0.2))
            if seed == 2:
                results.append(failed("semantic_motion", "sequence_a", seed))
            else:
                results.append(
                    complete("semantic_motion", "sequence_a", seed, 0.1))
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_prior_claim_gate.json").read_text())

        self.assertEqual(gate["status"], "insufficient_data")
        self.assertEqual(
            gate["sequences"]["sequence_a"]["status"], "incomplete")
        self.assertEqual(
            gate["sequences"]["sequence_a"]["missing_runs"],
            ["semantic_motion/seed_2"])

    def test_five_seed_motion_gate_requires_prior_use_in_four_seeds(self) -> None:
        results = []
        for seed in range(5):
            results.append(complete("semantic", "sequence_a", seed, 0.2))
            motion = complete("semantic_motion", "sequence_a", seed, 0.1)
            motion["metrics"]["used_motion_priors"] = 1 if seed < 2 else 0
            results.append(motion)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_prior_claim_gate.json").read_text())

        self.assertEqual(gate["status"], "fail")
        self.assertFalse(gate["trigger_condition"])
        self.assertEqual(
            gate["sequences"]["sequence_a"]["motion_used_seed_rate"], 0.4)

    def test_duplicate_result_identity_is_rejected(self) -> None:
        result = complete("semantic", "sequence_a", 0, 0.2)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Duplicate benchmark"):
                BENCHMARK.aggregate(Path(directory), [result, dict(result)])

    def test_task_blocks_keep_all_configs_in_one_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tasks, blocks = BENCHMARK.build_task_blocks(
                ["semantic", "full"],
                ["sequence_a"],
                [0, 1],
                ["0", "1"],
                0,
                {"commit": "synthetic"},
                Path(directory),
                synchronize_local_mapping=True,
                synchronize_loop_closing=True,
                disable_gaussian_mapper=True,
            )

        self.assertEqual(len(tasks), 4)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(
            [task["config"] for task in blocks[0]], ["semantic", "full"])
        self.assertEqual(
            [task["config"] for task in blocks[1]], ["full", "semantic"])
        for block in blocks:
            self.assertEqual(len({task["pair_id"] for task in block}), 1)
            self.assertEqual(len({task["gpu"] for task in block}), 1)
            self.assertTrue(all(
                task["synchronize_local_mapping"] for task in block))
            self.assertTrue(all(
                task["synchronize_loop_closing"] for task in block))
            self.assertTrue(all(
                task["disable_gaussian_mapper"] for task in block))

    def test_shadow_translation_horizon_sweep_uses_runtime_overrides(self) -> None:
        base = BENCHMARK.CONFIGS[
            "semantic_motion_rgbd_shadow_translation"]
        for config, horizon in (
            BENCHMARK.SHADOW_TRANSLATION_HORIZON_OVERRIDES.items()
        ):
            self.assertEqual(BENCHMARK.CONFIGS[config], base)
            self.assertEqual(
                [
                    "--shadow-translation-horizon",
                    str(horizon),
                ],
                BENCHMARK.config_runtime_arguments(config),
            )

    def test_post_run_failure_updates_existing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            manifest_path = run_dir / "manifest.json"
            manifest_path.write_text(json.dumps({"status": "running"}))
            task = {
                "config": "semantic",
                "sequence": "sequence_a",
                "seed": 0,
                "gpu": "0",
                "run_dir": str(run_dir),
            }
            BENCHMARK.GPU_LOCKS = {"0": BENCHMARK.threading.Lock()}
            original = BENCHMARK.run_one
            BENCHMARK.run_one = lambda *_args, **_kwargs: (
                (_ for _ in ()).throw(RuntimeError("evaluation failed")))
            try:
                result = BENCHMARK.run_paired_block([task])[0]
            finally:
                BENCHMARK.run_one = original

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("evaluation failed", manifest["error"])

    def test_shadow_config_is_available_and_never_counts_as_used(self) -> None:
        self.assertIn("semantic_motion_shadow", BENCHMARK.CONFIGS)
        result = complete(
            "semantic_motion_shadow", "sequence_a", 0, 0.2)
        self.assertEqual(result["metrics"]["would_use_motion_priors"], 1)
        self.assertEqual(result["metrics"]["used_motion_priors"], 0)

    def test_dypho_ablation_configs_are_available_and_matched(self) -> None:
        self.assertEqual(
            BENCHMARK.DYPHO_ABLATION_CONFIGS,
            (
                "semantic",
                "dypho_raw",
                "dypho_temporal",
                "dypho_feature",
                "dypho_compatible",
                "dypho_decoupled",
                "dypho_flow_guarded",
                "dypho_flow_exact",
                "dypho_flow_adaptive",
            ),
        )
        expected = {
            "dypho_raw": (0, 0),
            "dypho_temporal": (1, 0),
            "dypho_feature": (0, 1),
            "dypho_compatible": (1, 1),
            "dypho_decoupled": (1, 1),
            "dypho_flow_guarded": (1, 1),
            "dypho_flow_exact": (1, 1),
            "dypho_flow_adaptive": (1, 1),
        }
        for config, (temporal, adaptive) in expected.items():
            storage = cv2.FileStorage(
                str(BENCHMARK.CONFIGS[config]), cv2.FILE_STORAGE_READ)
            self.assertTrue(storage.isOpened())
            self.assertEqual(
                storage.getNode(
                    "mask.use_temporal_background_refinement").real(),
                temporal,
            )
            self.assertEqual(
                storage.getNode(
                    "mask.use_adaptive_feature_extraction").real(),
                adaptive,
            )
            self.assertEqual(
                storage.getNode("mask.enable_flow_hard").real(), 1)
            self.assertEqual(
                storage.getNode(
                    "mask.use_hard_mapping_mask").real(), 1)
            self.assertEqual(
                storage.getNode("mask.use_motion_pose_prior").real(), 0)
        decoupled = cv2.FileStorage(
            str(BENCHMARK.CONFIGS["dypho_decoupled"]),
            cv2.FILE_STORAGE_READ)
        self.assertEqual(
            decoupled.getNode("mask.temporal_recovery_only").real(), 1)
        self.assertEqual(
            decoupled.getNode(
                "mask.temporal_conservative_mapping").real(), 1)
        flow_guarded = cv2.FileStorage(
            str(BENCHMARK.CONFIGS["dypho_flow_guarded"]),
            cv2.FILE_STORAGE_READ)
        self.assertEqual(
            flow_guarded.getNode(
                "mask.temporal_recovery_flow_guard").real(), 1)
        self.assertEqual(
            flow_guarded.getNode(
                "mask.adaptive_feature_min_previous_inliers").real(), 150)
        self.assertEqual(
            flow_guarded.getNode(
                "mask.adaptive_feature_hold_frames").real(), 5)
        flow_adaptive = cv2.FileStorage(
            str(BENCHMARK.CONFIGS["dypho_flow_adaptive"]),
            cv2.FILE_STORAGE_READ)
        self.assertEqual(
            flow_adaptive.getNode(
                "mask.temporal_recovery_flow_guard").real(), 1)
        self.assertEqual(
            flow_adaptive.getNode(
                "mask.temporal_flow_guard_safe_radius").real(), 1)
        self.assertEqual(
            flow_adaptive.getNode(
                "mask.temporal_flow_guard_adaptive_radius").real(), 1)
        self.assertEqual(
            flow_adaptive.getNode(
                "mask.temporal_flow_guard_high_confidence_scale").real(), 2.0)
        flow_exact = cv2.FileStorage(
            str(BENCHMARK.CONFIGS["dypho_flow_exact"]),
            cv2.FILE_STORAGE_READ)
        self.assertEqual(
            flow_exact.getNode(
                "mask.temporal_recovery_flow_guard").real(), 1)
        self.assertEqual(
            flow_exact.getNode(
                "mask.temporal_flow_guard_safe_radius").real(), 0)
        self.assertEqual(
            flow_exact.getNode(
                "mask.temporal_flow_guard_adaptive_radius").real(), 0)
        self.assertTrue(
            BENCHMARK.flow_adaptive_config_contract()["valid"])

    def test_aggregate_reports_dypho_intervention_metrics(self) -> None:
        root = self.run_aggregate([
            complete("dypho_compatible", "sequence_a", 0, 0.1),
            complete("dypho_compatible", "sequence_a", 1, 0.2),
        ])
        with (root / "aggregate.csv").open(newline="") as handle:
            rows = list(BENCHMARK.csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(float(row["mask_pose_predicted_frames_mean"]), 2.0)
        self.assertEqual(float(row["temporal_refinement_frames_mean"]), 1.0)
        self.assertEqual(
            float(row["temporal_recovered_static_pixels_mean"]), 11.0)
        self.assertEqual(
            float(row["temporal_added_dynamic_pixels_mean"]), 7.0)
        self.assertEqual(
            float(row["tracking_recovery_audit_pixels_mean"]), 0.0)
        self.assertEqual(
            float(row[
                "tracking_recovery_mapping_leak_pixels_mean"]), 0.0)
        self.assertEqual(float(row["static_mask_ratio_mean"]), 0.8)
        self.assertEqual(float(row["mapping_weight_frames_mean"]), 100.0)
        self.assertEqual(float(row["mapping_static_ratio_mean"]), 0.75)
        self.assertEqual(
            float(row[
                "mapping_orb_map_points_rejected_static_mask_mean"]),
            20.0)
        self.assertEqual(
            float(row[
                "mapping_rgbd_densification_rejected_static_mask_mean"]),
            80.0)
        self.assertEqual(
            float(row["adaptive_feature_active_frames_mean"]), 25.0)
        self.assertEqual(float(row["adaptive_fast_threshold_mean"]), 15.0)
        self.assertEqual(float(row["extracted_features_mean"]), 450.0)

    def test_consensus_configs_are_available(self) -> None:
        self.assertIn("semantic_motion_consensus", BENCHMARK.CONFIGS)
        self.assertIn("semantic_motion_consensus_shadow", BENCHMARK.CONFIGS)
        for config, expected_shadow in (
            ("semantic_motion_consensus", 0),
            ("semantic_motion_consensus_shadow", 1),
        ):
            storage = cv2.FileStorage(
                str(BENCHMARK.CONFIGS[config]), cv2.FILE_STORAGE_READ)
            self.assertTrue(storage.isOpened())
            self.assertEqual(
                storage.getNode("mask.motion_gate_min_inlier_gain").real(), 1)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_gate_min_inlier_gain_ratio").real(), 0)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_gate_max_static_inliers").real(), 150)
            self.assertAlmostEqual(
                storage.getNode(
                    "mask.motion_gate_max_translation_innovation").real(), 0.01)
            self.assertAlmostEqual(
                storage.getNode(
                    "mask.motion_gate_max_rotation_innovation").real(),
                0.00872665)
            self.assertEqual(
                storage.getNode("mask.motion_prior_inject_local_map").real(), 0)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_pose_prior_shadow_only").real(),
                expected_shadow)
            storage.release()

    def test_tempered_configs_freeze_half_information(self) -> None:
        for config, expected_shadow in (
            ("semantic_motion_tempered", 0),
            ("semantic_motion_tempered_shadow", 1),
        ):
            storage = cv2.FileStorage(
                str(BENCHMARK.CONFIGS[config]), cv2.FILE_STORAGE_READ)
            self.assertTrue(storage.isOpened())
            self.assertAlmostEqual(
                storage.getNode(
                    "mask.motion_prior_information_scale").real(), 0.5)
            self.assertEqual(
                storage.getNode("mask.motion_prior_inject_local_map").real(), 0)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_pose_prior_shadow_only").real(),
                    expected_shadow)
            storage.release()

    def test_direct_configs_freeze_independent_validation(self) -> None:
        for config, expected_shadow in (
            ("semantic_motion_direct", 0),
            ("semantic_motion_direct_shadow", 1),
        ):
            storage = cv2.FileStorage(
                str(BENCHMARK.CONFIGS[config]), cv2.FILE_STORAGE_READ)
            self.assertTrue(storage.isOpened())
            self.assertAlmostEqual(
                storage.getNode(
                    "mask.motion_prior_information_scale").real(), 0.5)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_prior_use_direct_validation").real(), 1)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_prior_direct_score_mode").real(), 0)
            self.assertEqual(
                storage.getNode("mask.motion_prior_inject_local_map").real(), 0)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_pose_prior_shadow_only").real(),
                expected_shadow)
            storage.release()

    def test_rgbd_configs_select_combined_validation(self) -> None:
        for config, expected_shadow in (
            ("semantic_motion_rgbd", 0),
            ("semantic_motion_rgbd_shadow", 1),
        ):
            storage = cv2.FileStorage(
                str(BENCHMARK.CONFIGS[config]), cv2.FILE_STORAGE_READ)
            self.assertTrue(storage.isOpened())
            self.assertAlmostEqual(
                storage.getNode(
                    "mask.motion_prior_information_scale").real(), 0.5)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_prior_use_direct_validation").real(), 1)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_prior_direct_score_mode").real(), 2)
            self.assertEqual(
                storage.getNode("mask.motion_prior_inject_local_map").real(), 0)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_pose_prior_shadow_only").real(),
                expected_shadow)
            storage.release()

    def test_init_rgbd_configs_are_initialization_only(self) -> None:
        gate_keys = (
            "mask.motion_gate_min_inlier_gain",
            "mask.motion_gate_min_inlier_gain_ratio",
            "mask.motion_gate_max_static_inliers",
            "mask.motion_gate_min_translation_innovation",
            "mask.motion_gate_max_translation_innovation",
            "mask.motion_gate_max_rotation_innovation",
            "mask.motion_prior_information_scale",
            "mask.motion_prior_use_direct_validation",
            "mask.motion_prior_direct_score_mode",
            "mask.motion_prior_inject_local_map",
            "mask.motion_prior_require_common_support_improvement",
            "mask.motion_prior_bypass_reliability_gate",
            "mask.motion_prior_shuffle_lag_frames",
        )
        full = cv2.FileStorage(
            str(BENCHMARK.CONFIGS["semantic_motion_rgbd"]),
            cv2.FILE_STORAGE_READ,
        )
        self.assertTrue(full.isOpened())
        expected_gate = {
            key: full.getNode(key).real() for key in gate_keys
        }
        full.release()

        for config, expected_shadow in (
            ("semantic_motion_init_rgbd", 0),
            ("semantic_motion_init_rgbd_shadow", 1),
        ):
            storage = cv2.FileStorage(
                str(BENCHMARK.CONFIGS[config]), cv2.FILE_STORAGE_READ)
            self.assertTrue(storage.isOpened())
            self.assertEqual(
                storage.getNode(
                    "mask.motion_prior_use_direct_validation").real(), 1)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_prior_direct_score_mode").real(), 2)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_prior_initialization_only").real(), 1)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_prior_require_common_support_improvement"
                ).real(), 0)
            self.assertAlmostEqual(
                storage.getNode(
                    "mask.motion_gate_min_translation_innovation").real(),
                0.0)
            self.assertEqual(
                storage.getNode("mask.motion_prior_inject_local_map").real(), 0)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_prior_bypass_reliability_gate").real(), 0)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_prior_shuffle_lag_frames").real(), 0)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_pose_prior_shadow_only").real(),
                expected_shadow)
            self.assertEqual(
                {key: storage.getNode(key).real() for key in gate_keys},
                expected_gate)
            storage.release()

    def test_motion_ablation_controls_have_distinct_semantics(self) -> None:
        ungated = cv2.FileStorage(
            str(BENCHMARK.CONFIGS["semantic_motion_ungated"]),
            cv2.FILE_STORAGE_READ,
        )
        shuffled = cv2.FileStorage(
            str(BENCHMARK.CONFIGS["semantic_motion_rgbd_shuffled"]),
            cv2.FILE_STORAGE_READ,
        )
        self.assertTrue(ungated.isOpened())
        self.assertTrue(shuffled.isOpened())
        self.assertEqual(
            ungated.getNode(
                "mask.motion_prior_bypass_reliability_gate").real(), 1)
        self.assertEqual(
            ungated.getNode(
                "mask.motion_prior_initialization_only").real(), 0)
        self.assertEqual(
            shuffled.getNode(
                "mask.motion_prior_bypass_reliability_gate").real(), 0)
        self.assertEqual(
            shuffled.getNode(
                "mask.motion_prior_shuffle_lag_frames").real(), 30)
        self.assertEqual(
            shuffled.getNode(
                "mask.motion_prior_use_direct_validation").real(), 1)
        ungated.release()
        shuffled.release()

    def test_shadow_translation_controls_share_action_contract(self) -> None:
        self.assertEqual(
            BENCHMARK.SHADOW_TRANSLATION_HELDOUT_SEQUENCES,
            ("tum_walking_rpy", "bonn_balloon"))
        for config in (
            "semantic_motion_rgbd_shadow_translation_shadow",
            "semantic_motion_rgbd_shadow_translation_lag30",
            "semantic_motion_rgbd_shadow_translation",
        ):
            storage = cv2.FileStorage(
                str(BENCHMARK.CONFIGS[config]), cv2.FILE_STORAGE_READ)
            self.assertTrue(storage.isOpened())
            self.assertAlmostEqual(
                storage.getNode(
                    "mask.motion_gate_min_translation_innovation").real(),
                0.0015)
            self.assertAlmostEqual(
                storage.getNode(
                    "mask.motion_gate_max_translation_innovation").real(),
                0.014)
            self.assertAlmostEqual(
                storage.getNode("mask.motion_min_information").real(),
                2500.0)
            self.assertAlmostEqual(
                storage.getNode(
                    "mask.motion_prior_information_scale").real(),
                0.5)
            self.assertAlmostEqual(
                storage.getNode(
                    "mask.motion_prior_posterior_min_score_improvement").real(),
                0.0)
            self.assertAlmostEqual(
                storage.getNode(
                    "mask.motion_prior_shadow_translation_blend").real(),
                1.0)
            self.assertEqual(
                storage.getNode(
                    "mask.motion_prior_shadow_translation_horizon").real(),
                1)
            storage.release()
        self.assertEqual(
            BENCHMARK.SHADOW_TRANSLATION_FROZEN_PARAMETERS,
            {
                "minimum_schur_information_eigenvalue": 2500.0,
                "minimum_translation_innovation_m": 0.0015,
                "maximum_translation_innovation_m": 0.014,
                "configured_information_scale": 0.5,
                "replay_information_scale_multiplier": 1.0,
                "effective_information_scale": 0.5,
                "posterior_min_score_improvement": 0.0,
                "translation_blend": 1.0,
                "translation_horizon_frames": 1,
                "prior_subspace": "translation_only_isotropic",
                "factor_scope": "selected_frame_only",
                "velocity_update": "retain_incoming_increment_once",
            })
        shadow = cv2.FileStorage(
            str(BENCHMARK.CONFIGS[
                "semantic_motion_rgbd_shadow_translation_shadow"]),
            cv2.FILE_STORAGE_READ)
        lag30 = cv2.FileStorage(
            str(BENCHMARK.CONFIGS[
                "semantic_motion_rgbd_shadow_translation_lag30"]),
            cv2.FILE_STORAGE_READ)
        self.assertEqual(
            shadow.getNode("mask.motion_pose_prior_shadow_only").real(), 1)
        self.assertEqual(
            lag30.getNode("mask.motion_pose_prior_shadow_only").real(), 0)
        self.assertEqual(
            lag30.getNode("mask.motion_prior_shuffle_lag_frames").real(), 30)
        shadow.release()
        lag30.release()

    def test_complete_motion_ablation_has_matched_three_seed_plan(self) -> None:
        results = []
        for seed in range(3):
            for index, config in enumerate(
                BENCHMARK.MOTION_ABLATION_CONFIGS
            ):
                results.append(
                    complete(
                        config, "sequence_a", seed,
                        0.1 + 0.01 * index))
        root = self.run_aggregate(results)
        report = json.loads(
            (root / "motion_ablation_summary.json").read_text())

        self.assertEqual(report["status"], "complete")
        sequence = report["sequences"]["sequence_a"]
        self.assertTrue(sequence["same_seed_plan"])
        self.assertTrue(sequence["claim_ready_three_seed_minimum"])
        self.assertEqual(
            set(sequence["variants"]),
            set(BENCHMARK.MOTION_ABLATION_CONFIGS))

    def test_missing_motion_ablation_control_is_insufficient(self) -> None:
        results = [
            complete(config, "sequence_a", 0, 0.1)
            for config in BENCHMARK.MOTION_ABLATION_CONFIGS
            if config != "semantic_motion_rgbd_shuffled"
        ]
        root = self.run_aggregate(results)
        report = json.loads(
            (root / "motion_ablation_summary.json").read_text())

        self.assertEqual(report["status"], "insufficient_data")
        self.assertIn(
            "semantic_motion_rgbd_shuffled",
            report["sequences"]["sequence_a"]["missing_configs"])

    def test_complete_direct_counterfactual_gate_can_pass(self) -> None:
        results = []
        for sequence in BENCHMARK.MOTION_DIRECT_HELDOUT_SEQUENCES:
            for seed in range(3):
                result = complete(
                    "semantic_motion_direct_shadow", sequence, seed, 0.2)
                result["metrics"]["motion_prior_counterfactual"] = {
                    "gate_would_use": {
                        "pairs": 2,
                        "static_translation_sum_squared_error_m2": 0.0002,
                        "dynamic_translation_sum_squared_error_m2": 0.0001,
                        "dynamic_translation_wins": 2,
                        "worst_translation_degradation_m": 0.001,
                    }
                }
                results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_direct_counterfactual_gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["prior_information_scale"], 0.5)
        for sequence in BENCHMARK.MOTION_DIRECT_HELDOUT_SEQUENCES:
            self.assertEqual(
                gate["sequences"][sequence]["selected_pairs"], 6)
            self.assertTrue(
                gate["sequences"][sequence]["mechanism_pass"])

    def test_omitted_direct_heldout_sequence_is_insufficient(self) -> None:
        results = []
        for seed in range(3):
            result = complete(
                "semantic_motion_direct_shadow",
                "tum_sitting_xyz", seed, 0.2)
            result["metrics"]["motion_prior_counterfactual"] = {
                "gate_would_use": {
                    "pairs": 2,
                    "static_translation_sum_squared_error_m2": 0.0002,
                    "dynamic_translation_sum_squared_error_m2": 0.0001,
                    "dynamic_translation_wins": 2,
                    "worst_translation_degradation_m": 0.001,
                }
            }
            results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_direct_counterfactual_gate.json").read_text())

        self.assertEqual(gate["status"], "insufficient_data")
        self.assertEqual(
            gate["sequences"]["bonn_person_tracking"]["status"],
            "not_planned")

    def test_direct_gate_rejects_non_preregistered_seed_plan(self) -> None:
        results = []
        for sequence in BENCHMARK.MOTION_DIRECT_HELDOUT_SEQUENCES:
            for seed in (3, 4, 5):
                result = complete(
                    "semantic_motion_direct_shadow", sequence, seed, 0.2)
                result["metrics"]["motion_prior_counterfactual"] = {
                    "gate_would_use": {
                        "pairs": 2,
                        "static_translation_sum_squared_error_m2": 0.0002,
                        "dynamic_translation_sum_squared_error_m2": 0.0001,
                        "dynamic_translation_wins": 2,
                        "worst_translation_degradation_m": 0.001,
                    }
                }
                results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_direct_counterfactual_gate.json").read_text())

        self.assertEqual(gate["status"], "insufficient_data")
        for sequence in BENCHMARK.MOTION_DIRECT_HELDOUT_SEQUENCES:
            self.assertFalse(
                gate["sequences"][sequence]["seed_plan_matches"])

    def test_complete_direct_end_to_end_gate_can_pass(self) -> None:
        results = []
        ate_by_config = {
            "semantic": 0.2,
            "semantic_motion_direct_shadow": 0.15,
            "semantic_motion_direct": 0.1,
        }
        for sequence in BENCHMARK.MOTION_DIRECT_HELDOUT_SEQUENCES:
            for seed in range(3):
                for config, ate in ate_by_config.items():
                    result = complete(config, sequence, seed, ate)
                    if config == "semantic_motion_direct_shadow":
                        result["metrics"]["motion_prior_counterfactual"] = {
                            "gate_would_use": {
                                "pairs": 2,
                                "static_translation_sum_squared_error_m2":
                                    0.0002,
                                "dynamic_translation_sum_squared_error_m2":
                                    0.0001,
                                "dynamic_translation_wins": 2,
                                "worst_translation_degradation_m": 0.001,
                            }
                        }
                    results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_direct_end_to_end_gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["paired_win_rate_over_shadow"], 1.0)
        self.assertTrue(gate["mechanism_gate_condition"])

    def test_failed_direct_mechanism_blocks_end_to_end_gate(self) -> None:
        results = []
        for sequence in BENCHMARK.MOTION_DIRECT_HELDOUT_SEQUENCES:
            for seed in range(3):
                for config, ate in (
                    ("semantic", 0.2),
                    ("semantic_motion_direct_shadow", 0.15),
                    ("semantic_motion_direct", 0.1),
                ):
                    result = complete(config, sequence, seed, ate)
                    if config == "semantic_motion_direct_shadow":
                        result["metrics"]["motion_prior_counterfactual"] = {
                            "gate_would_use": {
                                "pairs": 2,
                                "static_translation_sum_squared_error_m2":
                                    0.0001,
                                "dynamic_translation_sum_squared_error_m2":
                                    0.0002,
                                "dynamic_translation_wins": 0,
                                "worst_translation_degradation_m": 0.002,
                            }
                        }
                    results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_direct_end_to_end_gate.json").read_text())

        self.assertEqual(gate["status"], "fail")
        self.assertFalse(gate["mechanism_gate_condition"])
        self.assertFalse(gate["motion_direct_intervention_retained"])

    def test_complete_rgbd_counterfactual_gate_can_pass(self) -> None:
        results = []
        for sequence in BENCHMARK.MOTION_RGBD_HELDOUT_SEQUENCES:
            for seed in BENCHMARK.MOTION_RGBD_HELDOUT_SEEDS:
                result = complete(
                    "semantic_motion_rgbd_shadow", sequence, seed, 0.2)
                result["metrics"]["motion_prior_counterfactual"] = {
                    "gate_would_use": {
                        "pairs": 2,
                        "static_translation_sum_squared_error_m2": 0.0002,
                        "dynamic_translation_sum_squared_error_m2": 0.0001,
                        "dynamic_translation_wins": 2,
                        "worst_translation_degradation_m": 0.001,
                    }
                }
                results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_rgbd_counterfactual_gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        self.assertTrue(gate["motion_rgbd_gate_retained"])
        for sequence in BENCHMARK.MOTION_RGBD_HELDOUT_SEQUENCES:
            self.assertTrue(
                gate["sequences"][sequence]["seed_plan_matches"])

    def test_complete_rgbd_end_to_end_gate_can_pass(self) -> None:
        results = []
        ate_by_config = {
            "semantic": 0.2,
            "semantic_motion_rgbd_shadow": 0.15,
            "semantic_motion_rgbd": 0.1,
        }
        for sequence in BENCHMARK.MOTION_RGBD_HELDOUT_SEQUENCES:
            for seed in BENCHMARK.MOTION_RGBD_HELDOUT_SEEDS:
                for config, ate in ate_by_config.items():
                    result = complete(config, sequence, seed, ate)
                    if config == "semantic_motion_rgbd_shadow":
                        result["metrics"]["motion_prior_counterfactual"] = {
                            "gate_would_use": {
                                "pairs": 2,
                                "static_translation_sum_squared_error_m2":
                                    0.0002,
                                "dynamic_translation_sum_squared_error_m2":
                                    0.0001,
                                "dynamic_translation_wins": 2,
                                "worst_translation_degradation_m": 0.001,
                            }
                        }
                    results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_rgbd_end_to_end_gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        self.assertTrue(gate["mechanism_gate_condition"])
        self.assertTrue(gate["motion_rgbd_intervention_retained"])

    def test_complete_tempered_counterfactual_gate_can_pass(self) -> None:
        results = []
        for sequence in BENCHMARK.MOTION_TEMPERED_HELDOUT_SEQUENCES:
            for seed in range(3):
                result = complete(
                    "semantic_motion_tempered_shadow", sequence, seed, 0.2)
                result["metrics"]["motion_prior_counterfactual"] = {
                    "gate_would_use": {
                        "pairs": 2,
                        "static_translation_sum_squared_error_m2": 0.0002,
                        "dynamic_translation_sum_squared_error_m2": 0.0001,
                        "dynamic_translation_wins": 2,
                        "worst_translation_degradation_m": 0.001,
                    }
                }
                results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_tempered_counterfactual_gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["prior_information_scale"], 0.5)
        for sequence in BENCHMARK.MOTION_TEMPERED_HELDOUT_SEQUENCES:
            self.assertEqual(
                gate["sequences"][sequence]["selected_pairs"], 6)
            self.assertTrue(
                gate["sequences"][sequence]["mechanism_pass"])

    def test_omitted_tempered_heldout_sequence_is_insufficient(self) -> None:
        results = []
        for seed in range(3):
            result = complete(
                "semantic_motion_tempered_shadow",
                "tum_walking_static", seed, 0.2)
            result["metrics"]["motion_prior_counterfactual"] = {
                "gate_would_use": {
                    "pairs": 2,
                    "static_translation_sum_squared_error_m2": 0.0002,
                    "dynamic_translation_sum_squared_error_m2": 0.0001,
                    "dynamic_translation_wins": 2,
                    "worst_translation_degradation_m": 0.001,
                }
            }
            results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_tempered_counterfactual_gate.json").read_text())

        self.assertEqual(gate["status"], "insufficient_data")
        self.assertEqual(
            gate["sequences"]["bonn_crowd3"]["status"], "not_planned")

    def test_complete_tempered_end_to_end_gate_can_pass(self) -> None:
        results = []
        ate_by_config = {
            "semantic": 0.2,
            "semantic_motion_tempered_shadow": 0.15,
            "semantic_motion_tempered": 0.1,
        }
        for sequence in BENCHMARK.MOTION_TEMPERED_HELDOUT_SEQUENCES:
            for seed in range(3):
                for config, ate in ate_by_config.items():
                    results.append(complete(config, sequence, seed, ate))
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_tempered_end_to_end_gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["paired_win_rate_over_shadow"], 1.0)

    def test_complete_consensus_counterfactual_gate_can_pass(self) -> None:
        results = []
        for sequence in BENCHMARK.MOTION_CONSENSUS_HELDOUT_SEQUENCES:
            for seed in range(3):
                result = complete(
                    "semantic_motion_consensus_shadow", sequence, seed, 0.2)
                result["metrics"]["motion_prior_counterfactual"] = {
                    "gate_would_use": {
                        "pairs": 2,
                        "static_translation_sum_squared_error_m2": 0.0002,
                        "dynamic_translation_sum_squared_error_m2": 0.0001,
                        "dynamic_translation_wins": 2,
                        "worst_translation_degradation_m": 0.001,
                    }
                }
                results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_consensus_counterfactual_gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        for sequence in BENCHMARK.MOTION_CONSENSUS_HELDOUT_SEQUENCES:
            self.assertEqual(
                gate["sequences"][sequence]["selected_pairs"], 6)
            self.assertTrue(
                gate["sequences"][sequence]["mechanism_pass"])

    def test_consensus_gate_rejects_large_worst_degradation(self) -> None:
        results = []
        for sequence in BENCHMARK.MOTION_CONSENSUS_HELDOUT_SEQUENCES:
            for seed in range(3):
                result = complete(
                    "semantic_motion_consensus_shadow", sequence, seed, 0.2)
                result["metrics"]["motion_prior_counterfactual"] = {
                    "gate_would_use": {
                        "pairs": 2,
                        "static_translation_sum_squared_error_m2": 0.0002,
                        "dynamic_translation_sum_squared_error_m2": 0.0001,
                        "dynamic_translation_wins": 2,
                        "worst_translation_degradation_m": (
                            0.011
                            if sequence == "bonn_crowd2" and seed == 2
                            else 0.001),
                    }
                }
                results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_consensus_counterfactual_gate.json").read_text())

        self.assertEqual(gate["status"], "fail")
        self.assertFalse(
            gate["sequences"]["bonn_crowd2"]["mechanism_pass"])

    def test_consensus_gate_accepts_zero_pair_seed_without_crashing(self) -> None:
        results = []
        for sequence in BENCHMARK.MOTION_CONSENSUS_HELDOUT_SEQUENCES:
            for seed in range(3):
                result = complete(
                    "semantic_motion_consensus_shadow", sequence, seed, 0.2)
                pairs = 0 if seed == 0 else 3
                result["metrics"]["motion_prior_counterfactual"] = {
                    "gate_would_use": {
                        "pairs": pairs,
                        "static_translation_sum_squared_error_m2": (
                            0.0 if not pairs else 0.0003),
                        "dynamic_translation_sum_squared_error_m2": (
                            0.0 if not pairs else 0.0001),
                        "dynamic_translation_wins": pairs,
                        "worst_translation_degradation_m": (
                            None if not pairs else 0.001),
                    }
                }
                results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_consensus_counterfactual_gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        for sequence in BENCHMARK.MOTION_CONSENSUS_HELDOUT_SEQUENCES:
            self.assertEqual(
                gate["sequences"][sequence]["selected_pairs"], 6)

    def test_omitted_consensus_heldout_sequence_is_insufficient(self) -> None:
        results = []
        for seed in range(3):
            result = complete(
                "semantic_motion_consensus_shadow",
                "tum_walking_halfsphere", seed, 0.2)
            result["metrics"]["motion_prior_counterfactual"] = {
                "gate_would_use": {
                    "pairs": 2,
                    "static_translation_sum_squared_error_m2": 0.0002,
                    "dynamic_translation_sum_squared_error_m2": 0.0001,
                    "dynamic_translation_wins": 2,
                    "worst_translation_degradation_m": 0.001,
                }
            }
            results.append(result)
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_consensus_counterfactual_gate.json").read_text())

        self.assertEqual(gate["status"], "insufficient_data")
        self.assertEqual(
            gate["sequences"]["bonn_crowd2"]["status"], "not_planned")

    def test_complete_consensus_end_to_end_gate_can_pass(self) -> None:
        results = []
        ate_by_config = {
            "semantic": 0.2,
            "semantic_motion_consensus_shadow": 0.15,
            "semantic_motion_consensus": 0.1,
        }
        for sequence in BENCHMARK.MOTION_CONSENSUS_HELDOUT_SEQUENCES:
            for seed in range(3):
                for config, ate in ate_by_config.items():
                    results.append(complete(config, sequence, seed, ate))
        root = self.run_aggregate(results)
        gate = json.loads(
            (root / "motion_consensus_end_to_end_gate.json").read_text())

        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["paired_win_rate_over_shadow"], 1.0)

    def test_dyn19_claim_union_and_phase_denominator_are_frozen(self) -> None:
        expected = {
            "tum_walking_xyz",
            "tum_walking_halfsphere",
            "tum_walking_static",
            "tum_sitting_halfsphere",
            "bonn_balloon",
            "bonn_crowd",
            "bonn_crowd2",
            "bonn_crowd3",
            "bonn_person_tracking",
            "bonn_person_tracking2",
        }
        self.assertEqual(set(BENCHMARK.DYN19_CLAIM_SEQUENCES), expected)
        self.assertIn(
            "tum_sitting_halfsphere", BENCHMARK.DYN19_CLAIM_SEQUENCES)
        self.assertEqual(
            len(BENCHMARK.DYN19_MAIN_CONFIGS) *
            len(BENCHMARK.DYN19_CLAIM_SEQUENCES) *
            len(BENCHMARK.DYN19_REQUIRED_SEEDS),
            90,
        )
        self.assertEqual(
            len(BENCHMARK.DYN19_MECHANISM_CONFIGS) *
            len(BENCHMARK.DYN19_CLAIM_SEQUENCES) *
            len(BENCHMARK.DYN19_REQUIRED_SEEDS),
            120,
        )

    def test_dyn19_config_contract_allows_only_declared_factor_changes(self) -> None:
        contract = BENCHMARK.dyn19_config_contract()

        self.assertTrue(contract["valid"])
        self.assertTrue(contract["key_sets_match"])
        self.assertTrue(contract["factor_values_match"])
        for config in BENCHMARK.DYN19_TRACKING_CONFIGS:
            self.assertEqual(
                contract["configs"][config]["non_factor_differences"], [])
        self.assertEqual(
            BENCHMARK.DYN19_FACTOR_MATRIX["dyn19_full"][
                "description"],
            "T+F+R+A+M",
        )
        self.assertEqual(
            BENCHMARK.DYN19_FACTOR_MATRIX["dyn19_full_no_mapping"][
                "description"],
            "T+F+R+A+NoM",
        )

    def test_dyn19_main_exports_only_semantic_static_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tasks, _ = BENCHMARK.build_task_blocks(
                list(BENCHMARK.DYN19_MAIN_CONFIGS),
                list(BENCHMARK.DYN19_CLAIM_SEQUENCES),
                list(BENCHMARK.DYN19_REQUIRED_SEEDS),
                ["0"],
                0,
                {"commit": "synthetic-clean-commit", "dirty": False},
                Path(directory),
                export_static_masks=True,
                dyn19_phase="main",
            )

        semantic_exports = {
            (task["config"], task["sequence"], task["seed"])
            for task in tasks
            if task["export_static_masks"]
        }
        expected = {
            ("dyn19_semantic", sequence, seed)
            for sequence in BENCHMARK.DYN19_CLAIM_SEQUENCES
            for seed in BENCHMARK.DYN19_REQUIRED_SEEDS
        }
        self.assertEqual(semantic_exports, expected)
        self.assertTrue(all(
            task["dyn19"]["phase"] == "main" for task in tasks))

    def test_dyn19_main_report_preserves_pass_and_vacuous_certificates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_dyn19_main_plan(root)
            results = []
            for config in BENCHMARK.DYN19_MAIN_CONFIGS:
                for sequence in BENCHMARK.DYN19_CLAIM_SEQUENCES:
                    for seed in BENCHMARK.DYN19_REQUIRED_SEEDS:
                        result = complete(config, sequence, seed, 0.1)
                        if config == "dyn19_full":
                            recovered = (
                                sequence == "tum_sitting_halfsphere" and
                                seed == 1
                            )
                            result["metrics"][
                                "recovered_support_overlap_certificate"
                            ] = {
                                "status": "pass" if recovered else "vacuous",
                                "evidence": {
                                    "tracking_recovery_audit_pixels": (
                                        10 if recovered else 0),
                                    "tracking_recovery_mapping_leak_pixels": 0,
                                },
                            }
                        results.append(result)

            BENCHMARK.write_dyn19_phase_reports(root, results)
            report = json.loads(
                (root / "dyn19_recovered_support_overlap_aggregate.json")
                .read_text())
            phase_report = json.loads(
                (root / "dyn19_phase_report.json").read_text())

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["safety_status"], "pass")
        self.assertTrue(report["all_completed"])
        self.assertTrue(report["all_zero_leak"])
        self.assertEqual(report["certificate_status_counts"]["pass"], 1)
        self.assertEqual(report["certificate_status_counts"]["vacuous"], 29)
        self.assertEqual(report["positive_recovery_evidence"], "observed")
        self.assertEqual(phase_report["status"], "complete")

    def test_dyn19_main_report_marks_all_vacuous_rows_as_safety_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_dyn19_main_plan(root)
            results = []
            for config in BENCHMARK.DYN19_MAIN_CONFIGS:
                for sequence in BENCHMARK.DYN19_CLAIM_SEQUENCES:
                    for seed in BENCHMARK.DYN19_REQUIRED_SEEDS:
                        result = complete(config, sequence, seed, 0.1)
                        if config == "dyn19_full":
                            result["metrics"][
                                "recovered_support_overlap_certificate"
                            ] = {
                                "status": "vacuous",
                                "evidence": {
                                    "tracking_recovery_audit_pixels": 0,
                                    "tracking_recovery_mapping_leak_pixels": 0,
                                },
                            }
                        results.append(result)

            BENCHMARK.write_dyn19_phase_reports(root, results)
            report = json.loads(
                (root / "dyn19_recovered_support_overlap_aggregate.json")
                .read_text())

        self.assertEqual(report["status"], "vacuous")
        self.assertEqual(report["safety_status"], "pass")
        self.assertTrue(report["all_completed"])
        self.assertTrue(report["all_zero_leak"])
        self.assertEqual(report["certificate_status_counts"]["pass"], 0)
        self.assertEqual(report["certificate_status_counts"]["vacuous"], 30)
        self.assertEqual(
            report["positive_recovery_evidence"], "not_observed_all_vacuous")

    def test_dyn20_split_is_disjoint_from_dyn19_and_frozen_to_48_runs(self) -> None:
        self.assertFalse(
            set(BENCHMARK.DYN20_HELDOUT_SEQUENCES) &
            set(BENCHMARK.DYN19_CLAIM_SEQUENCES))
        self.assertIn(
            "tum_sitting_static", BENCHMARK.DYN20_HELDOUT_SEQUENCES)
        self.assertEqual(
            len(BENCHMARK.DYN20_CONFIGS) *
            len(BENCHMARK.DYN20_HELDOUT_SEQUENCES) *
            len(BENCHMARK.DYN20_REQUIRED_SEEDS),
            48,
        )

    def test_dyn20_tasks_carry_separate_experiment_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tasks = freeze_dyn20_heldout_plan(Path(directory))
            audit = BENCHMARK.dyn20_heldout_plan_contract(Path(directory))

        self.assertTrue(audit["valid"])
        self.assertEqual(audit["dyn19_sequence_overlap"], [])
        self.assertTrue(all("dyn20" in task for task in tasks))
        self.assertTrue(all("dyn19" not in task for task in tasks))
        self.assertEqual(
            sum(task["dyn20"]["role"] == "candidate" for task in tasks), 12)

    def test_dyn20_report_applies_predeclared_tplusm_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_dyn20_heldout_plan(root)
            results = []
            ate_by_config = {
                "dyn19_semantic": 0.15,
                "dyn19_mapmatched": 0.12,
                "dyn19_temporal_map": 0.08,
                "dyn19_full": 0.10,
            }
            for config in BENCHMARK.DYN20_CONFIGS:
                for sequence in BENCHMARK.DYN20_HELDOUT_SEQUENCES:
                    for seed in BENCHMARK.DYN20_REQUIRED_SEEDS:
                        result = complete(
                            config, sequence, seed, ate_by_config[config])
                        if config == BENCHMARK.DYN20_CANDIDATE:
                            result["metrics"][
                                "recovered_support_overlap_certificate"
                            ] = {
                                "status": "pass",
                                "evidence": {
                                    "tracking_recovery_audit_pixels": 10,
                                    "tracking_recovery_mapping_leak_pixels": 0,
                                },
                            }
                        results.append(result)
            BENCHMARK.write_dyn20_heldout_report(root, results)
            report = json.loads(
                (root / "dyn20_tplusm_heldout_report.json").read_text())

        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["tplusm_promoted"])
        self.assertEqual(
            report["conditions"]["candidate_route_passes"], 12)
        self.assertEqual(
            report["conditions"]["paired_ate_wins_over_full"], 12)


if __name__ == "__main__":
    unittest.main()
