#!/usr/bin/env python3
"""Non-destructive multi-seed benchmark runner for DynaGS-SLAM."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import threading
from typing import Any
import zipfile

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = Path(
    os.environ.get("DYNAGS_DATASETS_ROOT", str(PROJECT.parent))
).expanduser().resolve()
EXPERIMENTS = DATASETS / "DyGeoFusion-SLAM-experiments" / "reproducible_v2"

CONFIGS = {
    "no_mask": PROJECT / "config/mask_config_no_yolo.yaml",
    "semantic": PROJECT / "config/mask_config.yaml",
    "semantic_motion": PROJECT / "config/mask_config_semantic_motion.yaml",
    "semantic_motion_shadow": (
        PROJECT / "config/mask_config_semantic_motion_shadow.yaml"),
    "semantic_motion_consensus": (
        PROJECT / "config/mask_config_semantic_motion_consensus.yaml"),
    "semantic_motion_consensus_shadow": (
        PROJECT / "config/mask_config_semantic_motion_consensus_shadow.yaml"),
    "semantic_motion_tempered": (
        PROJECT / "config/mask_config_semantic_motion_tempered.yaml"),
    "semantic_motion_tempered_shadow": (
        PROJECT / "config/mask_config_semantic_motion_tempered_shadow.yaml"),
    "semantic_motion_direct": (
        PROJECT / "config/mask_config_semantic_motion_direct.yaml"),
    "semantic_motion_direct_shadow": (
        PROJECT / "config/mask_config_semantic_motion_direct_shadow.yaml"),
    "semantic_motion_rgbd": (
        PROJECT / "config/mask_config_semantic_motion_rgbd.yaml"),
    "semantic_motion_rgbd_shadow": (
        PROJECT / "config/mask_config_semantic_motion_rgbd_shadow.yaml"),
    "semantic_motion_ungated": (
        PROJECT / "config/mask_config_semantic_motion_ungated.yaml"),
    "semantic_motion_rgbd_shuffled": (
        PROJECT /
        "config/mask_config_semantic_motion_rgbd_shuffled.yaml"),
    "semantic_motion_rgbd_shadow_translation": (
        PROJECT /
        "config/mask_config_semantic_motion_rgbd_shadow_translation.yaml"),
    "semantic_motion_rgbd_shadow_translation_shadow": (
        PROJECT /
        "config/"
        "mask_config_semantic_motion_rgbd_shadow_translation_shadow.yaml"),
    "semantic_motion_rgbd_shadow_translation_lag30": (
        PROJECT /
        "config/"
        "mask_config_semantic_motion_rgbd_shadow_translation_lag30.yaml"),
    "semantic_motion_init_rgbd": (
        PROJECT / "config/mask_config_semantic_motion_init_rgbd.yaml"),
    "semantic_motion_init_rgbd_shadow": (
        PROJECT / "config/mask_config_semantic_motion_init_rgbd_shadow.yaml"),
    "geometry": PROJECT / "config/mask_config_geo.yaml",
    "flow": PROJECT / "config/mask_config_flow.yaml",
    "full": PROJECT / "config/mask_config_full.yaml",
    "full_motion": PROJECT / "config/mask_config_full_motion.yaml",
    "dypho_raw": PROJECT / "config/mask_config_dypho_raw.yaml",
    "dypho_temporal": (
        PROJECT / "config/mask_config_dypho_temporal.yaml"),
    "dypho_feature": PROJECT / "config/mask_config_dypho_feature.yaml",
    "dypho_compatible": (
        PROJECT / "config/mask_config_dypho_compatible.yaml"),
    "dypho_decoupled": (
        PROJECT / "config/mask_config_dypho_decoupled.yaml"),
    "dypho_flow_guarded": (
        PROJECT / "config/mask_config_dypho_flow_guarded.yaml"),
    "dypho_flow_exact": (
        PROJECT / "config/mask_config_dypho_flow_exact.yaml"),
    "dypho_flow_adaptive": (
        PROJECT / "config/mask_config_dypho_flow_adaptive.yaml"),
    "dyn19_semantic": (
        PROJECT / "config/mask_config_dyn19_semantic.yaml"),
    "dyn19_mapmatched": (
        PROJECT / "config/mask_config_dyn19_mapmatched.yaml"),
    "dyn19_temporal_map": (
        PROJECT / "config/mask_config_dyn19_temporal_map.yaml"),
    "dyn19_temporal_flow_map": (
        PROJECT / "config/mask_config_dyn19_temporal_flow_map.yaml"),
    "dyn19_temporal_flow_risk_map": (
        PROJECT / "config/mask_config_dyn19_temporal_flow_risk_map.yaml"),
    "dyn19_full": (
        PROJECT / "config/mask_config_dyn19_full.yaml"),
    "dyn19_full_no_mapping": (
        PROJECT / "config/mask_config_dyn19_full_no_mapping.yaml"),
}

DYN19_EXPERIMENT_ID = "DYN-19_FULL_CAUSAL_MAP_INTEGRITY_20260913"
DYN19_CLAIM_SEQUENCES = (
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
)
DYN19_REQUIRED_SEEDS = (0, 1, 2)
DYN19_MAIN_CONFIGS = (
    "dyn19_semantic",
    "dyn19_mapmatched",
    "dyn19_full",
)
DYN19_MECHANISM_CONFIGS = (
    "dyn19_temporal_map",
    "dyn19_temporal_flow_map",
    "dyn19_temporal_flow_risk_map",
    "dyn19_full_no_mapping",
)
DYN19_TRACKING_CONFIGS = (
    *DYN19_MAIN_CONFIGS,
    *DYN19_MECHANISM_CONFIGS,
)
DYN19_PHASE_CONFIGS = {
    "main": DYN19_MAIN_CONFIGS,
    "mechanisms": DYN19_MECHANISM_CONFIGS,
}
DYN19_FACTOR_KEYS = (
    "mask.use_flow",
    "mask.enable_flow_hard",
    "mask.use_temporal_background_refinement",
    "mask.use_adaptive_feature_extraction",
    "mask.temporal_recovery_only",
    "mask.temporal_conservative_mapping",
    "mask.temporal_recovery_flow_guard",
    "mask.temporal_recovery_require_tracking_risk",
)
DYN19_FACTOR_MATRIX = {
    "dyn19_semantic": {
        "temporal_depth": False,
        "recovery_flow_guard": False,
        "tracking_risk_gate": False,
        "adaptive_fast": False,
        "raw_mapping_weight": False,
        "description": "Semantic hard exclusion",
    },
    "dyn19_mapmatched": {
        "temporal_depth": False,
        "recovery_flow_guard": False,
        "tracking_risk_gate": False,
        "adaptive_fast": False,
        "raw_mapping_weight": True,
        "description": "MapMatched-NoTrackReuse",
    },
    "dyn19_temporal_map": {
        "temporal_depth": True,
        "recovery_flow_guard": False,
        "tracking_risk_gate": False,
        "adaptive_fast": False,
        "raw_mapping_weight": True,
        "description": "T+M",
    },
    "dyn19_temporal_flow_map": {
        "temporal_depth": True,
        "recovery_flow_guard": True,
        "tracking_risk_gate": False,
        "adaptive_fast": False,
        "raw_mapping_weight": True,
        "description": "T+F+M",
    },
    "dyn19_temporal_flow_risk_map": {
        "temporal_depth": True,
        "recovery_flow_guard": True,
        "tracking_risk_gate": True,
        "adaptive_fast": False,
        "raw_mapping_weight": True,
        "description": "T+F+R+M",
    },
    "dyn19_full": {
        "temporal_depth": True,
        "recovery_flow_guard": True,
        "tracking_risk_gate": True,
        "adaptive_fast": True,
        "raw_mapping_weight": True,
        "description": "T+F+R+A+M",
    },
    "dyn19_full_no_mapping": {
        "temporal_depth": True,
        "recovery_flow_guard": True,
        "tracking_risk_gate": True,
        "adaptive_fast": True,
        "raw_mapping_weight": False,
        "description": "T+F+R+A+NoM",
    },
}
DYN19_EXPECTED_FACTOR_VALUES = {
    config: {
        "mask.use_flow": "1" if factors["raw_mapping_weight"] or
        factors["temporal_depth"] else "0",
        "mask.enable_flow_hard": "1" if factors["raw_mapping_weight"] or
        factors["temporal_depth"] else "0",
        "mask.use_temporal_background_refinement": (
            "1" if factors["temporal_depth"] else "0"),
        "mask.use_adaptive_feature_extraction": (
            "1" if factors["adaptive_fast"] else "0"),
        "mask.temporal_recovery_only": (
            "0" if config == "dyn19_semantic" else "1"),
        "mask.temporal_conservative_mapping": (
            "1" if factors["raw_mapping_weight"] else "0"),
        "mask.temporal_recovery_flow_guard": (
            "1" if factors["recovery_flow_guard"] else "0"),
        "mask.temporal_recovery_require_tracking_risk": (
            "1" if factors["tracking_risk_gate"] else "0"),
    }
    for config, factors in DYN19_FACTOR_MATRIX.items()
}
DYPHO_ABLATION_CONFIGS = (
    "semantic",
    "dypho_raw",
    "dypho_temporal",
    "dypho_feature",
    "dypho_compatible",
    "dypho_decoupled",
    "dypho_flow_guarded",
    "dypho_flow_exact",
    "dypho_flow_adaptive",
)

FLOW_ADAPTIVE_PARENT_CONFIG = "dypho_flow_exact"
FLOW_ADAPTIVE_CANDIDATE_CONFIG = "dypho_flow_adaptive"
FLOW_ADAPTIVE_DEVELOPMENT_SEQUENCES = (
    "tum_walking_xyz",
    "tum_walking_rpy",
    "bonn_crowd",
)
FLOW_ADAPTIVE_REQUIRED_SEEDS = (0, 1, 2)
FLOW_ADAPTIVE_MIN_ATE_IMPROVEMENT = 0.03
FLOW_ADAPTIVE_MAX_RISK_DEGRADATION = 0.02
FLOW_ADAPTIVE_MAX_FAILURE_RATE_INCREASE = 0.005
FLOW_ADAPTIVE_MIN_SEQUENCE_PASSES = 2
FLOW_ADAPTIVE_MIN_PAIRED_ATE_WINS = 6
FLOW_ADAPTIVE_RISK_METRICS = (
    "ate_p95_m",
    "rpe_translation_rmse_m",
    "rpe_translation_p95_m",
    "rpe_rotation_rmse_deg",
    "rpe_rotation_p95_deg",
)
FLOW_ADAPTIVE_CONFIG_DIFFERENCES = {
    "mask.temporal_flow_guard_safe_radius",
    "mask.temporal_flow_guard_adaptive_radius",
}
MATCHED_ORACLE_MODES = {
    "semantic_motion_rgbd_oracle": "apply",
    "semantic_motion_rgbd_oracle_sham": "sham",
}
CONFIGS.update({
    config: PROJECT / "config/mask_config_semantic_motion_rgbd.yaml"
    for config in MATCHED_ORACLE_MODES
})
SHADOW_TRANSLATION_HORIZON_OVERRIDES = {
    "semantic_motion_rgbd_shadow_translation_h1": 1,
    "semantic_motion_rgbd_shadow_translation_h3": 3,
    "semantic_motion_rgbd_shadow_translation_h5": 5,
}
CONFIGS.update({
    config: CONFIGS["semantic_motion_rgbd_shadow_translation"]
    for config in SHADOW_TRANSLATION_HORIZON_OVERRIDES
})

MOTION_ABLATION_CONFIGS = (
    "semantic",
    "semantic_motion_init_rgbd",
    "semantic_motion_ungated",
    "semantic_motion_rgbd",
    "semantic_motion_rgbd_shadow",
    "semantic_motion_rgbd_shuffled",
)

SHADOW_TRANSLATION_CONFIG = "semantic_motion_rgbd_shadow_translation"
SHADOW_TRANSLATION_CONTROLS = (
    "semantic",
    "semantic_motion_rgbd_shadow_translation_shadow",
    "semantic_motion_rgbd_shadow_translation_lag30",
)
SHADOW_TRANSLATION_TRACKING_CONFIGS = (
    *SHADOW_TRANSLATION_CONTROLS,
    SHADOW_TRANSLATION_CONFIG,
)
SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES = (
    "tum_walking_xyz",
    "bonn_crowd",
)
SHADOW_TRANSLATION_HELDOUT_SEQUENCES = (
    "tum_walking_rpy",
    "bonn_balloon",
)
SHADOW_TRANSLATION_REQUIRED_SEEDS = (0, 1, 2)
SHADOW_TRANSLATION_FROZEN_PARAMETERS = {
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
}

RUNTIME_LIBRARIES = (
    PROJECT / "ORB-SLAM3/lib/libORB_SLAM3.so",
    PROJECT / "lib/libmotion3d.so",
    PROJECT / "lib/libdynamic_mask_refiner.so",
    PROJECT / "lib/libgaussian_mapper.so",
)

TUM_ORB = PROJECT / "cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml"
TUM_GAUSSIAN = PROJECT / "cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml"
BONN_ORB = PROJECT / "cfg/ORB_SLAM3/RGB-D/Bonn/bonn_rgbd.yaml"
BONN_GAUSSIAN = PROJECT / "cfg/gaussian_mapper/RGB-D/Bonn/bonn_rgbd.yaml"

SEQUENCES = {
    "tum_walking_xyz": (
        DATASETS / "rgbd_dataset_freiburg3_walking_xyz", TUM_ORB, TUM_GAUSSIAN,
        "associations.txt"),
    "tum_walking_halfsphere": (
        DATASETS / "rgbd_dataset_freiburg3_walking_halfsphere", TUM_ORB,
        TUM_GAUSSIAN, "associations.txt"),
    "tum_walking_rpy": (
        DATASETS / "rgbd_dataset_freiburg3_walking_rpy", TUM_ORB,
        TUM_GAUSSIAN, "associations.txt"),
    "tum_walking_static": (
        DATASETS / "rgbd_dataset_freiburg3_walking_static", TUM_ORB, TUM_GAUSSIAN,
        "associations.txt"),
    "tum_sitting_xyz": (
        DATASETS / "rgbd_dataset_freiburg3_sitting_xyz", TUM_ORB, TUM_GAUSSIAN,
        "associations.txt"),
    "tum_sitting_rpy": (
        DATASETS / "rgbd_dataset_freiburg3_sitting_rpy", TUM_ORB, TUM_GAUSSIAN,
        "associations.txt"),
    "tum_sitting_halfsphere": (
        DATASETS / "rgbd_dataset_freiburg3_sitting_halfsphere", TUM_ORB,
        TUM_GAUSSIAN, "associations.txt"),
    "bonn_crowd": (
        DATASETS / "rgbd_bonn_dynamic/rgbd_bonn_crowd", BONN_ORB, BONN_GAUSSIAN,
        "association.txt"),
    "bonn_crowd2": (
        DATASETS / "rgbd_bonn_dynamic/rgbd_bonn_crowd2", BONN_ORB, BONN_GAUSSIAN,
        "associations.txt"),
    "bonn_crowd3": (
        DATASETS / "rgbd_bonn_dynamic/rgbd_bonn_crowd3", BONN_ORB, BONN_GAUSSIAN,
        "associations.txt"),
    "bonn_balloon": (
        DATASETS / "rgbd_bonn_dynamic/rgbd_bonn_balloon", BONN_ORB,
        BONN_GAUSSIAN, "associations.txt"),
    "bonn_person_tracking": (
        DATASETS / "rgbd_bonn_dynamic/rgbd_bonn_person_tracking", BONN_ORB,
        BONN_GAUSSIAN, "associations.txt"),
    "bonn_person_tracking2": (
        DATASETS / "rgbd_bonn_dynamic/rgbd_bonn_person_tracking2", BONN_ORB,
        BONN_GAUSSIAN, "associations.txt"),
}

MOTION_CONSENSUS_HELDOUT_SEQUENCES = (
    "tum_walking_halfsphere",
    "bonn_crowd2",
)

MOTION_TEMPERED_HELDOUT_SEQUENCES = (
    "tum_walking_static",
    "bonn_crowd3",
)

MOTION_DIRECT_HELDOUT_SEQUENCES = (
    "tum_sitting_xyz",
    "bonn_person_tracking",
)
MOTION_DIRECT_HELDOUT_SEEDS = (0, 1, 2)

MOTION_RGBD_HELDOUT_SEQUENCES = (
    "tum_sitting_rpy",
    "bonn_person_tracking2",
)
MOTION_RGBD_HELDOUT_SEEDS = (0, 1, 2)

GPU_LOCKS: dict[str, threading.Lock] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opencv_yaml_value(path: Path, key: str) -> str | None:
    prefix = f"{key}:"
    with path.open() as stream:
        for line in stream:
            content = line.split("#", 1)[0].strip()
            if not content.startswith(prefix):
                continue
            value = content[len(prefix):].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value
    return None


def opencv_yaml_bool(path: Path, key: str, default: bool = False) -> bool:
    value = opencv_yaml_value(path, key)
    if value is None:
        return default
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{path}: {key} is not boolean: {value}")


def opencv_yaml_contract(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open() as stream:
        for line in stream:
            content = line.split("#", 1)[0].strip()
            if not content or content.startswith("%") or ":" not in content:
                continue
            key, value = content.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                raise ValueError(f"{path}: invalid YAML entry: {content}")
            if key in values:
                raise ValueError(f"{path}: duplicate YAML key: {key}")
            if (
                len(value) >= 2 and value[0] == value[-1] and
                value[0] in "\"'"
            ):
                value = value[1:-1]
            values[key] = value
    return values


def dyn19_config_contract() -> dict[str, Any]:
    contracts = {
        config: opencv_yaml_contract(CONFIGS[config])
        for config in DYN19_TRACKING_CONFIGS
    }
    reference = contracts["dyn19_full"]
    key_sets_match = all(
        set(values) == set(reference)
        for values in contracts.values())
    factor_values_match = all(
        all(
            contracts[config].get(key) == expected
            for key, expected in expected_values.items())
        for config, expected_values in DYN19_EXPECTED_FACTOR_VALUES.items())
    non_factor_differences = {
        config: sorted(
            key for key in set(reference) & set(values)
            if key not in DYN19_FACTOR_KEYS and
            values[key] != reference[key])
        for config, values in contracts.items()
    }
    return {
        "contract": "dyn19-factor-config-matrix-v1",
        "valid": bool(
            key_sets_match and factor_values_match and
            not any(non_factor_differences.values())),
        "configs": {
            config: {
                "path": str(CONFIGS[config]),
                "sha256": sha256(CONFIGS[config]),
                "factors": DYN19_FACTOR_MATRIX[config],
                "expected_factor_values":
                    DYN19_EXPECTED_FACTOR_VALUES[config],
                "actual_factor_values": {
                    key: contracts[config].get(key)
                    for key in DYN19_FACTOR_KEYS
                },
                "non_factor_differences":
                    non_factor_differences[config],
            }
            for config in DYN19_TRACKING_CONFIGS
        },
        "key_sets_match": key_sets_match,
        "factor_values_match": factor_values_match,
    }


def dyn19_phase_plan_contract(root: Path) -> dict[str, Any]:
    plan_path = root / "benchmark_plan.json"
    digest_path = root / "benchmark_plan.sha256"
    result: dict[str, Any] = {
        "contract": "dyn19-phase-plan-audit-v1",
        "valid": False,
        "plan_path": str(plan_path),
        "phase": None,
        "expected_tasks": 0,
        "actual_tasks": 0,
    }
    if not plan_path.is_file() or not digest_path.is_file():
        result["error"] = "missing frozen benchmark plan or digest"
        return result
    digest_fields = digest_path.read_text().strip().split()
    actual_digest = sha256(plan_path)
    if not digest_fields or digest_fields[0] != actual_digest:
        result["error"] = "benchmark plan digest mismatch"
        return result
    plan = json.loads(plan_path.read_text())
    dyn19 = plan.get("dyn19")
    if not isinstance(dyn19, dict):
        result["error"] = "plan is not a DYN-19 phase plan"
        return result
    phase = dyn19.get("phase")
    result["phase"] = phase
    if phase not in DYN19_PHASE_CONFIGS:
        result["error"] = "unknown DYN-19 phase"
        return result
    expected_configs = DYN19_PHASE_CONFIGS[phase]
    expected = {
        (config, sequence, seed)
        for config in expected_configs
        for sequence in DYN19_CLAIM_SEQUENCES
        for seed in DYN19_REQUIRED_SEEDS
    }
    tasks = plan.get("tasks", [])
    actual = {
        (task.get("config"), task.get("sequence"), task.get("seed"))
        for task in tasks
        if isinstance(task, dict)
    }
    expected_exports = {
        ("dyn19_semantic", sequence, seed)
        for sequence in DYN19_CLAIM_SEQUENCES
        for seed in DYN19_REQUIRED_SEEDS
    } if phase == "main" else set()
    export_tasks = {
        (task.get("config"), task.get("sequence"), task.get("seed"))
        for task in tasks
        if isinstance(task, dict) and task.get("export_static_masks")
    }
    result.update({
        "expected_tasks": len(expected),
        "actual_tasks": len(tasks),
        "task_identity_match": len(tasks) == len(expected) and actual == expected,
        "semantic_mask_exports_match": export_tasks == expected_exports,
        "config_contract": dyn19_config_contract(),
        "recorded_config_contract": dyn19.get("config_contract"),
    })
    result["valid"] = bool(
        dyn19.get("experiment_id") == DYN19_EXPERIMENT_ID and
        result["task_identity_match"] and
        result["semantic_mask_exports_match"] and
        result["config_contract"]["valid"] and
        result["recorded_config_contract"] == result["config_contract"])
    return result


def flow_adaptive_config_contract() -> dict[str, Any]:
    parent_path = CONFIGS[FLOW_ADAPTIVE_PARENT_CONFIG]
    candidate_path = CONFIGS[FLOW_ADAPTIVE_CANDIDATE_CONFIG]
    parent = opencv_yaml_contract(parent_path)
    candidate = opencv_yaml_contract(candidate_path)
    missing_from_parent = sorted(set(candidate) - set(parent))
    missing_from_candidate = sorted(set(parent) - set(candidate))
    differences = sorted(
        key for key in set(parent) & set(candidate)
        if parent[key] != candidate[key]
    )
    expected_values = {
        "mask.temporal_flow_guard_safe_radius": ("0", "1"),
        "mask.temporal_flow_guard_adaptive_radius": ("0", "1"),
    }
    values_match = all(
        parent.get(key) == expected_parent and
        candidate.get(key) == expected_candidate
        for key, (expected_parent, expected_candidate)
        in expected_values.items()
    )
    valid = bool(
        not missing_from_parent and
        not missing_from_candidate and
        set(differences) == FLOW_ADAPTIVE_CONFIG_DIFFERENCES and
        values_match
    )
    return {
        "contract": "flow-adaptive-config-pair-v1",
        "valid": valid,
        "parent": FLOW_ADAPTIVE_PARENT_CONFIG,
        "candidate": FLOW_ADAPTIVE_CANDIDATE_CONFIG,
        "parent_sha256": sha256(parent_path),
        "candidate_sha256": sha256(candidate_path),
        "allowed_differences": sorted(FLOW_ADAPTIVE_CONFIG_DIFFERENCES),
        "actual_differences": differences,
        "missing_from_parent": missing_from_parent,
        "missing_from_candidate": missing_from_candidate,
        "expected_values": {
            key: {"parent": values[0], "candidate": values[1]}
            for key, values in expected_values.items()
        },
        "actual_values": {
            key: {
                "parent": parent.get(key),
                "candidate": candidate.get(key),
            }
            for key in expected_values
        },
    }


def flow_adaptive_plan_contract(root: Path) -> dict[str, Any]:
    plan_path = root / "benchmark_plan.json"
    digest_path = root / "benchmark_plan.sha256"
    result: dict[str, Any] = {
        "contract": "flow-adaptive-benchmark-plan-audit-v1",
        "valid": False,
        "plan_path": str(plan_path),
        "plan_sha256": None,
        "recorded_sha256": None,
        "predeclared": False,
        "task_identity_match": False,
        "task_runtime_match": False,
        "config_contract_match": False,
    }
    if not plan_path.is_file() or not digest_path.is_file():
        result["error"] = "missing frozen benchmark plan or digest"
        return result
    actual_digest = sha256(plan_path)
    digest_fields = digest_path.read_text().strip().split()
    recorded_digest = digest_fields[0] if digest_fields else ""
    result["plan_sha256"] = actual_digest
    result["recorded_sha256"] = recorded_digest
    if actual_digest != recorded_digest:
        result["error"] = "benchmark plan digest mismatch"
        return result
    try:
        plan = json.loads(plan_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        result["error"] = f"invalid benchmark plan: {exc}"
        return result
    gate = plan.get("flow_adaptive_incumbent_gate", {})
    result["predeclared"] = bool(gate.get("predeclared", False))
    expected_identities = {
        (config, sequence, seed)
        for config in (
            FLOW_ADAPTIVE_PARENT_CONFIG,
            FLOW_ADAPTIVE_CANDIDATE_CONFIG,
        )
        for sequence in FLOW_ADAPTIVE_DEVELOPMENT_SEQUENCES
        for seed in FLOW_ADAPTIVE_REQUIRED_SEEDS
    }
    tasks = plan.get("tasks", [])
    actual_identities = {
        (task.get("config"), task.get("sequence"), task.get("seed"))
        for task in tasks
        if isinstance(task, dict)
    }
    result["task_identity_match"] = bool(
        len(tasks) == len(expected_identities) and
        actual_identities == expected_identities
    )
    result["task_runtime_match"] = bool(
        tasks and all(
            task.get("heldout_stride") == 0 and
            not task.get("disable_gaussian_mapper", False)
            for task in tasks
            if isinstance(task, dict)
        )
    )
    result["config_contract_match"] = bool(
        gate.get("config_contract") == flow_adaptive_config_contract()
    )
    result["valid"] = bool(
        result["predeclared"] and
        result["task_identity_match"] and
        result["task_runtime_match"] and
        result["config_contract_match"]
    )
    return result


def yolo_engine_contract(mask_config: Path) -> dict[str, Any]:
    use_yolo = opencv_yaml_bool(mask_config, "mask.use_yolo", True)
    use_external = opencv_yaml_bool(
        mask_config, "mask.use_external_mask", False)
    if not use_yolo or use_external:
        return {
            "required": False,
            "path": None,
            "sha256": None,
        }
    configured_path = opencv_yaml_value(
        mask_config, "mask.yolo_model_path")
    if not configured_path:
        raise ValueError(
            f"{mask_config}: YOLO is enabled but mask.yolo_model_path is empty")
    engine = Path(configured_path)
    if not engine.is_absolute():
        engine = PROJECT / engine
    engine = engine.resolve()
    if not engine.is_file():
        raise FileNotFoundError(f"YOLO engine not found: {engine}")
    if engine.stat().st_size == 0:
        raise ValueError(f"YOLO engine is empty: {engine}")
    return {
        "required": True,
        "path": str(engine),
        "sha256": sha256(engine),
    }


def run_text(command: list[str], cwd: Path = PROJECT) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def source_state() -> dict[str, Any]:
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=PROJECT)
    untracked = run_text(
        ["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
    runtime_roots = (
        "include/", "src/", "ORB-SLAM3/include/", "ORB-SLAM3/src/",
        "examples/", "config/", "cfg/", "scripts/",
    )
    runtime_names = {"CMakeLists.txt", "run_tum_dynamic.sh"}
    untracked_runtime_hashes = {
        relative: sha256(PROJECT / relative)
        for relative in sorted(untracked)
        if relative in runtime_names or relative.startswith(runtime_roots)
    }
    snapshot = hashlib.sha256()
    snapshot.update(run_text(["git", "rev-parse", "HEAD"]).encode())
    snapshot.update(diff)
    for relative, digest in untracked_runtime_hashes.items():
        snapshot.update(relative.encode())
        snapshot.update(digest.encode())
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "dirty": bool(run_text(["git", "status", "--short"])),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "runtime_source_snapshot_sha256": snapshot.hexdigest(),
        "untracked_runtime_file_sha256": untracked_runtime_hashes,
        "status": run_text(["git", "status", "--short"]),
    }


def count_data_lines(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip() and not line.lstrip().startswith("#"))


def parse_rmse(output: str) -> float:
    match = re.search(r"^\s*rmse\s+([-+0-9.eE]+)\s*$", output, re.MULTILINE)
    if not match:
        raise ValueError("evo output did not contain an RMSE row")
    return float(match.group(1))


def result_error_percentile(path: Path, percentile: float) -> float:
    with zipfile.ZipFile(path) as archive:
        try:
            payload = archive.read("error_array.npy")
        except KeyError as exc:
            raise ValueError(
                f"evo result has no error_array.npy: {path}") from exc
    errors = np.load(io.BytesIO(payload), allow_pickle=False)
    if errors.ndim != 1 or errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError(f"evo result has an invalid error array: {path}")
    return float(np.percentile(errors, percentile))


def evaluate_evo(run_dir: Path, ground_truth: Path, trajectory_name: str,
                 artifact_label: str) -> dict[str, float]:
    trajectory = run_dir / trajectory_name
    commands = {
        "ate_rmse_m": [
            "evo_ape", "tum", str(ground_truth), str(trajectory), "-a",
            "-r", "trans_part", "--no_warnings",
            "--save_results", str(run_dir / f"ape_{artifact_label}.zip")],
        "rpe_translation_rmse_m": [
            "evo_rpe", "tum", str(ground_truth), str(trajectory), "-a",
            "-r", "trans_part", "-d", "1", "-u", "f", "--no_warnings",
            "--save_results", str(run_dir / f"rpe_translation_{artifact_label}.zip")],
        "rpe_rotation_rmse_deg": [
            "evo_rpe", "tum", str(ground_truth), str(trajectory), "-a",
            "-r", "angle_deg", "-d", "1", "-u", "f", "--no_warnings",
            "--save_results", str(run_dir / f"rpe_rotation_{artifact_label}.zip")],
    }
    percentile_names = {
        "ate_rmse_m": "ate_p95_m",
        "rpe_translation_rmse_m": "rpe_translation_p95_m",
        "rpe_rotation_rmse_deg": "rpe_rotation_p95_deg",
    }
    metrics = {}
    for name, command in commands.items():
        completed = subprocess.run(command, cwd=PROJECT, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (run_dir / f"{name}_{artifact_label}.txt").write_text(completed.stdout)
        if completed.returncode != 0:
            raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
        metrics[name] = parse_rmse(completed.stdout)
        metrics[percentile_names[name]] = result_error_percentile(
            Path(command[-1]), 95.0)
    return metrics


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def task_without_plan_binding(task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in task.items()
        if key != "plan_binding"
    }


def freeze_benchmark_plan(
    root: Path,
    payload: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> str:
    plan_path = root / "benchmark_plan.json"
    atomic_json(plan_path, payload)
    plan_sha256 = sha256(plan_path)
    (root / "benchmark_plan.sha256").write_text(
        f"{plan_sha256}  benchmark_plan.json\n")
    for task in tasks:
        task["plan_binding"] = {
            "contract": "benchmark-plan-binding-v1",
            "path": str(plan_path.resolve()),
            "sha256": plan_sha256,
        }
    return plan_sha256


def validate_task_plan_binding(task: dict[str, Any]) -> dict[str, Any]:
    binding = task.get("plan_binding")
    if not isinstance(binding, dict) or (
        binding.get("contract") != "benchmark-plan-binding-v1"
    ):
        raise ValueError("Benchmark task has no valid plan binding")
    plan_path = Path(str(binding.get("path", "")))
    if not plan_path.is_file():
        raise FileNotFoundError(f"Benchmark plan not found: {plan_path}")
    actual_sha256 = sha256(plan_path)
    if actual_sha256 != binding.get("sha256"):
        raise ValueError(
            "Benchmark plan SHA-256 changed after task scheduling")
    plan = json.loads(plan_path.read_text())
    planned_task = task_without_plan_binding(task)
    exact_matches = [
        candidate
        for candidate in plan.get("tasks", [])
        if candidate == planned_task
    ]
    if len(exact_matches) != 1:
        raise ValueError(
            "Benchmark task is missing from or ambiguous in the frozen plan")
    return plan


def config_runtime_arguments(config: str) -> list[str]:
    if config not in SHADOW_TRANSLATION_HORIZON_OVERRIDES:
        return []
    return [
        "--shadow-translation-horizon",
        str(SHADOW_TRANSLATION_HORIZON_OVERRIDES[config]),
    ]


def benchmark_task_asset_contract(
    config: str,
    sequence: str,
) -> dict[str, Any]:
    sequence_dir, orb_config, gaussian_config, association_name = (
        SEQUENCES[sequence])
    association = sequence_dir / association_name
    ground_truth = sequence_dir / "groundtruth.txt"
    mask_config = CONFIGS[config]
    binary = PROJECT / "bin/tum_rgbd_dynamic"
    vocabulary = PROJECT / "ORB-SLAM3/Vocabulary/ORBvoc.txt"
    runtime_arguments = config_runtime_arguments(config)
    return {
        "contract": "benchmark-task-assets-v1",
        "paths": {
            "binary": str(binary.resolve()),
            "runtime_libraries": [
                str(path.resolve()) for path in RUNTIME_LIBRARIES
            ],
            "vocabulary": str(vocabulary.resolve()),
            "dataset": str(sequence_dir.resolve()),
            "association": str(association.resolve()),
            "ground_truth": str(ground_truth.resolve()),
            "orb_config": str(orb_config.resolve()),
            "gaussian_config": str(gaussian_config.resolve()),
            "mask_config": str(mask_config.resolve()),
        },
        "hashes": {
            "binary_sha256": sha256(binary),
            "runtime_library_sha256": {
                str(path.relative_to(PROJECT)): sha256(path)
                for path in RUNTIME_LIBRARIES
            },
            "vocabulary_sha256": sha256(vocabulary),
            "association_sha256": sha256(association),
            "ground_truth_sha256": sha256(ground_truth),
            "orb_config_sha256": sha256(orb_config),
            "gaussian_config_sha256": sha256(gaussian_config),
            "mask_config_sha256": sha256(mask_config),
        },
        "yolo_engine": yolo_engine_contract(mask_config),
        "runtime_arguments": runtime_arguments,
    }


def run_one(
    task: dict[str, Any], acquire_gpu_lock: bool = True
) -> dict[str, Any]:
    validate_task_plan_binding(task)
    actual_assets = benchmark_task_asset_contract(
        task["config"], task["sequence"])
    if actual_assets != task.get("asset_contract"):
        raise ValueError(
            "Benchmark assets changed after the plan was frozen")

    run_dir = Path(task["run_dir"])
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)

    sequence_dir, orb_config, gaussian_config, association_name = SEQUENCES[task["sequence"]]
    association = sequence_dir / association_name
    ground_truth = sequence_dir / "groundtruth.txt"
    mask_config = CONFIGS[task["config"]]
    binary = PROJECT / "bin/tum_rgbd_dynamic"
    command = [
        str(binary), str(PROJECT / "ORB-SLAM3/Vocabulary/ORBvoc.txt"),
        str(orb_config), str(gaussian_config), str(sequence_dir), str(association),
        str(run_dir), str(mask_config), "no_viewer", "--no-realtime",
        "--seed", str(task["seed"]),
    ]
    command += actual_assets["runtime_arguments"]
    matched_oracle_mode = MATCHED_ORACLE_MODES.get(task["config"])
    matched_oracle_pose_manifest = None
    if matched_oracle_mode is not None:
        matched_oracle_pose_manifest = Path(
            task["matched_oracle_pose_manifest"])
        if not matched_oracle_pose_manifest.is_file():
            raise FileNotFoundError(
                "Missing matched-oracle pose manifest: "
                f"{matched_oracle_pose_manifest}")
        if sha256(matched_oracle_pose_manifest) != task.get(
                "matched_oracle_pose_sha256"):
            raise ValueError(
                "Matched-oracle pose manifest changed after plan freeze")
        command += [
            "--matched-oracle-mode", matched_oracle_mode,
            "--matched-oracle-poses", str(matched_oracle_pose_manifest),
            "--capture-replay-state",
        ]
    if task.get("synchronize_local_mapping", False):
        command.append("--sync-local-mapping")
    if task.get("synchronize_loop_closing", False):
        command.append("--sync-loop-closing")
    if task.get("disable_gaussian_mapper", False):
        command.append("--disable-gaussian-mapper")
    if task.get("export_static_masks", False):
        command.append("--export-static-masks")
    if task["heldout_stride"] > 0:
        command += ["--heldout-stride", str(task["heldout_stride"])]

    manifest = {
        "status": "running",
        "protocol": (
            "full input sequence; primary ATE/RPE uses one online pose per input frame "
            "with last-valid-pose carry-forward when tracking is invalid; optimized "
            "valid-frame trajectory is auxiliary; SE(3) alignment without scale correction"
        ),
        "sequence": task["sequence"],
        "config": task["config"],
        "seed": task["seed"],
        "gpu": task["gpu"],
        "synchronize_local_mapping":
            task.get("synchronize_local_mapping", False),
        "synchronize_loop_closing":
            task.get("synchronize_loop_closing", False),
        "disable_gaussian_mapper":
            task.get("disable_gaussian_mapper", False),
        "export_static_masks":
            task.get("export_static_masks", False),
        "matched_oracle_mode": matched_oracle_mode,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "command": command,
        "input_frames": count_data_lines(association),
        "paths": {
            "dataset": str(sequence_dir), "association": str(association),
            "ground_truth": str(ground_truth), "orb_config": str(orb_config),
            "gaussian_config": str(gaussian_config), "mask_config": str(mask_config),
            "vocabulary": str(PROJECT / "ORB-SLAM3/Vocabulary/ORBvoc.txt"),
            "yolo_engine": actual_assets["yolo_engine"]["path"],
            "matched_oracle_pose_manifest": (
                str(matched_oracle_pose_manifest)
                if matched_oracle_pose_manifest is not None else ""),
        },
        "hashes": {
            "binary_sha256": sha256(binary),
            "runtime_library_sha256": {
                str(path.relative_to(PROJECT)): sha256(path)
                for path in RUNTIME_LIBRARIES
            },
            "orb_config_sha256": sha256(orb_config),
            "gaussian_config_sha256": sha256(gaussian_config),
            "mask_config_sha256": sha256(mask_config),
            "association_sha256": sha256(association),
            "ground_truth_sha256": sha256(ground_truth),
            "vocabulary_sha256": sha256(
                PROJECT / "ORB-SLAM3/Vocabulary/ORBvoc.txt"),
            "yolo_engine_sha256":
                actual_assets["yolo_engine"]["sha256"],
            "matched_oracle_pose_sha256": (
                task.get("matched_oracle_pose_sha256", "")
                if matched_oracle_pose_manifest is not None else ""),
        },
        "source": task["source"],
        "plan_binding": task["plan_binding"],
        "asset_contract": actual_assets,
    }
    atomic_json(run_dir / "manifest.json", manifest)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(task["gpu"])
    # This repository uses a LibTorch build that predates support for the
    # expandable_segments allocator option. Do not let an unrelated parent
    # shell setting change benchmark behavior or abort CUDA initialization.
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    def execute() -> subprocess.CompletedProcess:
        with (run_dir / "run.log").open("w") as log:
            return subprocess.run(
                command, cwd=PROJECT, env=environment,
                stdout=log, stderr=subprocess.STDOUT)
    if acquire_gpu_lock:
        with GPU_LOCKS[str(task["gpu"])]:
            completed = execute()
    else:
        completed = execute()
    manifest["return_code"] = completed.returncode
    manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    if completed.returncode != 0:
        manifest["status"] = "failed"
        atomic_json(run_dir / "manifest.json", manifest)
        return {"status": "failed", "run_dir": str(run_dir), **task}

    run_summary = json.loads((run_dir / "run_summary.json").read_text())
    metrics = evaluate_evo(
        run_dir, ground_truth, "CameraTrajectory_AllFrames_TUM.txt", "all_input_frames")
    valid_optimized = evaluate_evo(
        run_dir, ground_truth, "CameraTrajectory_TUM.txt", "valid_optimized_frames")
    metrics.update({f"valid_optimized_{name}": value
                    for name, value in valid_optimized.items()})
    trajectory_frames = count_data_lines(run_dir / "CameraTrajectory_AllFrames_TUM.txt")
    valid_optimized_frames = count_data_lines(run_dir / "CameraTrajectory_TUM.txt")
    if trajectory_frames != manifest["input_frames"]:
        raise RuntimeError(
            f"All-frame trajectory has {trajectory_frames} poses for "
            f"{manifest['input_frames']} input frames")
    metrics.update({
        "trajectory_frames": trajectory_frames,
        "input_frames": manifest["input_frames"],
        "trajectory_coverage": trajectory_frames / manifest["input_frames"],
        "valid_optimized_trajectory_frames": valid_optimized_frames,
        "valid_optimized_trajectory_coverage": (
            valid_optimized_frames / manifest["input_frames"]),
        "failure_rate": run_summary["failure_rate"],
        "end_to_end_seconds": run_summary["end_to_end_seconds"],
        "valid_motion_priors": run_summary.get(
            "valid_motion_priors", run_summary["accepted_motion_priors"]),
        "raw_motion_priors": run_summary.get(
            "raw_motion_priors", run_summary["accepted_motion_priors"]),
        # Legacy alias retained for artifacts produced before the stage name was corrected.
        "accepted_motion_priors": run_summary["accepted_motion_priors"],
        "candidate_motion_priors": run_summary.get("candidate_motion_priors", 0),
        "consensus_pass_motion_priors": run_summary.get(
            "consensus_pass_motion_priors", 0),
        "direct_pass_motion_priors": run_summary.get(
            "direct_pass_motion_priors", 0),
        "common_support_pass_motion_priors": run_summary.get(
            "common_support_pass_motion_priors", 0),
        "would_use_motion_priors": run_summary.get(
            "would_use_motion_priors", run_summary.get("used_motion_priors", 0)),
        "used_motion_priors": run_summary.get("used_motion_priors", 0),
        "matched_oracle_mode": run_summary.get(
            "matched_oracle_mode", "disabled"),
        "matched_oracle_evaluations": run_summary.get(
            "matched_oracle_evaluations", 0),
        "matched_oracle_dynamic_preferences": run_summary.get(
            "matched_oracle_dynamic_preferences", 0),
        "matched_oracle_applied": run_summary.get(
            "matched_oracle_applied", 0),
        "motion_prior_bypass_reliability_gate": run_summary.get(
            "motion_prior_bypass_reliability_gate", False),
        "motion_prior_shuffle_lag_frames": run_summary.get(
            "motion_prior_shuffle_lag_frames", 0),
        "motion_prior_shadow_translation_horizon": run_summary.get(
            "motion_prior_shadow_translation_horizon", 0),
        "motion_prior_posterior_min_score_improvement": run_summary.get(
            "motion_prior_posterior_min_score_improvement", 0.0),
        "motion_prior_shadow_translation_blend": run_summary.get(
            "motion_prior_shadow_translation_blend", 1.0),
        "shadow_translation_applied_frames": run_summary.get(
            "shadow_translation_applied_frames", 0),
        "shadow_translation_factor_injections": run_summary.get(
            "shadow_translation_factor_injections", 0),
        "velocity_neutralized_frames": run_summary.get(
            "velocity_neutralized_frames", 0),
        "shadow_translation_stopped_frames": run_summary.get(
            "shadow_translation_stopped_frames", 0),
        "shadow_translation_propagation_rejections": run_summary.get(
            "shadow_translation_propagation_rejections", 0),
        "motion_gate_min_translation_innovation_m": run_summary.get(
            "motion_gate_min_translation_innovation_m", 0.0),
        "motion_gate_max_translation_innovation_m": run_summary.get(
            "motion_gate_max_translation_innovation_m", 0.0),
        "synchronize_local_mapping": run_summary.get(
            "synchronize_local_mapping", False),
        "synchronize_loop_closing": run_summary.get(
            "synchronize_loop_closing", False),
        "gaussian_mapper_enabled": run_summary.get(
            "gaussian_mapper_enabled", True),
        "counterfactual_pose_pairs": run_summary.get(
            "counterfactual_pose_pairs", 0),
        "mask_pose_predicted_frames": run_summary.get(
            "mask_pose_predicted_frames", 0),
        "temporal_refinement_frames": run_summary.get(
            "temporal_refinement_frames", 0),
        "temporal_recovered_static_pixels": run_summary.get(
            "temporal_recovered_static_pixels", 0),
        "temporal_added_dynamic_pixels": run_summary.get(
            "temporal_added_dynamic_pixels", 0),
        "temporal_flow_guard_valid_frames": run_summary.get(
            "temporal_flow_guard_valid_frames", 0),
        "temporal_flow_guard_rejected_pixels": run_summary.get(
            "temporal_flow_guard_rejected_pixels", 0),
        "temporal_recovery_risk_active_frames": run_summary.get(
            "temporal_recovery_risk_active_frames", 0),
        "temporal_recovery_risk_blocked_frames": run_summary.get(
            "temporal_recovery_risk_blocked_frames", 0),
        "tracking_recovery_candidate_pixels": run_summary.get(
            "tracking_recovery_candidate_pixels", 0),
        "tracking_recovery_blocked_pixels": run_summary.get(
            "tracking_recovery_blocked_pixels", 0),
        "tracking_recovery_audit_pixels": run_summary.get(
            "tracking_recovery_audit_pixels", 0),
        "tracking_recovery_mapping_leak_pixels": run_summary.get(
            "tracking_recovery_mapping_leak_pixels", 0),
        "extracted_features_total": run_summary.get(
            "extracted_features_total", 0),
        "static_mask_ratio_mean": run_summary.get(
            "static_mask_ratio_mean", 1.0),
        "mapping_weight_frames": run_summary.get(
            "mapping_weight_frames", 0),
        "mapping_static_ratio_mean": run_summary.get(
            "mapping_static_ratio_mean", 1.0),
        "mapping_keyframes_with_static_weight": run_summary.get(
            "mapping_keyframes_with_static_weight", 0),
        "mapping_weight_pixels": run_summary.get(
            "mapping_weight_pixels", 0),
        "mapping_static_pixels": run_summary.get(
            "mapping_static_pixels", 0),
        "mapping_orb_map_points_considered": run_summary.get(
            "mapping_orb_map_points_considered", 0),
        "mapping_orb_map_points_rejected_static_mask": run_summary.get(
            "mapping_orb_map_points_rejected_static_mask", 0),
        "mapping_rgbd_densification_candidates": run_summary.get(
            "mapping_rgbd_densification_candidates", 0),
        "mapping_rgbd_densification_rejected_static_mask": run_summary.get(
            "mapping_rgbd_densification_rejected_static_mask", 0),
        "adaptive_feature_active_frames": run_summary.get(
            "adaptive_feature_active_frames", 0),
        "adaptive_fast_threshold_mean": run_summary.get(
            "adaptive_fast_threshold_mean", 0.0),
        "extracted_features_mean": run_summary.get(
            "extracted_features_mean", 0.0),
        "carried_forward_rate": run_summary.get(
            "carried_forward_rate",
            1.0 - run_summary["tracked_frames"] / run_summary["processed_frames"]),
    })

    certificate_path = (
        run_dir / "recovered_support_overlap_certificate.json")
    certifier = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts/certify_recovered_support_overlap.py"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(certificate_path),
        ],
        cwd=PROJECT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (run_dir / "recovered_support_overlap_certificate.log").write_text(
        certifier.stdout)
    if certifier.returncode != 0 or not certificate_path.is_file():
        raise RuntimeError(
            "Recovered-support overlap certificate failed with "
            f"exit code {certifier.returncode}")
    certificate = json.loads(certificate_path.read_text())
    metrics["recovered_support_overlap_certificate"] = certificate
    certificate_evidence = certificate.get("evidence", {})
    metrics["recovered_support_certificate_status"] = certificate.get(
        "status", "missing")
    metrics["recovered_support_certificate_audit_pixels"] = (
        certificate_evidence.get("tracking_recovery_audit_pixels", 0))
    metrics["recovered_support_certificate_leak_pixels"] = (
        certificate_evidence.get(
            "tracking_recovery_mapping_leak_pixels", 0))
    metrics["recovered_support_certificate_vacuous"] = (
        certificate.get("status") == "vacuous")

    if metrics["counterfactual_pose_pairs"] > 0:
        evaluator = subprocess.run(
            [
                sys.executable,
                str(PROJECT / "scripts/evaluate_motion_prior_counterfactual.py"),
                str(run_dir),
                str(ground_truth),
            ],
            cwd=PROJECT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (run_dir / "motion_prior_counterfactual_evaluation.log").write_text(
            evaluator.stdout)
        counterfactual_summary = (
            run_dir / "motion_prior_counterfactual_summary.json")
        if evaluator.returncode != 0 or not counterfactual_summary.is_file():
            raise RuntimeError(
                "motion-prior counterfactual evaluation failed with "
                f"exit code {evaluator.returncode}")
        metrics["motion_prior_counterfactual"] = json.loads(
            counterfactual_summary.read_text())

    heldout_summary = run_dir / "heldout_novel_view/heldout_metrics_summary.json"
    if task["heldout_stride"] > 0 and (run_dir / "heldout_novel_view").is_dir():
        evaluator = subprocess.run(
            [sys.executable, str(PROJECT / "scripts/evaluate_heldout_rendering.py"),
             str(run_dir / "heldout_novel_view")],
            cwd=PROJECT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (run_dir / "heldout_evaluation.log").write_text(evaluator.stdout)
        if evaluator.returncode == 0 and heldout_summary.is_file():
            metrics["heldout"] = json.loads(heldout_summary.read_text())
        else:
            metrics["heldout_error"] = f"exit {evaluator.returncode}"

    result = {
        "status": "complete", "sequence": task["sequence"],
        "config": task["config"], "seed": task["seed"],
        "run_dir": str(run_dir), "metrics": metrics,
        "source": task["source"],
    }
    atomic_json(run_dir / "result.json", result)
    manifest["status"] = "complete"
    atomic_json(run_dir / "manifest.json", manifest)
    return result


def run_paired_block(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tasks:
        return []
    gpu = str(tasks[0]["gpu"])
    if any(str(task["gpu"]) != gpu for task in tasks):
        raise ValueError("Every task in a paired block must use the same GPU")
    results = []
    with GPU_LOCKS[gpu]:
        for task in tasks:
            try:
                result = run_one(task, acquire_gpu_lock=False)
            except Exception as exc:
                run_dir = Path(task["run_dir"])
                manifest_path = run_dir / "manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text())
                    manifest["status"] = "failed"
                    manifest["finished_at"] = (
                        dt.datetime.now(dt.timezone.utc).isoformat())
                    manifest["error"] = f"{type(exc).__name__}: {exc}"
                    atomic_json(manifest_path, manifest)
                result = {
                    "status": "failed",
                    **task,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
    return results


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def relative_change(candidate: float, parent: float) -> float | None:
    if not math.isfinite(candidate) or not math.isfinite(parent):
        return None
    if parent == 0.0:
        return 0.0 if candidate == 0.0 else None
    return (candidate - parent) / parent


def write_flow_adaptive_incumbent_gate(
    root: Path,
    results: list[dict[str, Any]],
) -> None:
    relevant = [
        result for result in results
        if result.get("config") in {
            FLOW_ADAPTIVE_PARENT_CONFIG,
            FLOW_ADAPTIVE_CANDIDATE_CONFIG,
        }
    ]
    output = root / "flow_adaptive_incumbent_gate.json"
    if not relevant:
        atomic_json(output, {
            "protocol": "flow-adaptive-vs-exact-incumbent-gate-v1",
            "status": "not_planned",
            "decision": "NO_DECISION",
            "candidate": FLOW_ADAPTIVE_CANDIDATE_CONFIG,
            "parent": FLOW_ADAPTIVE_PARENT_CONFIG,
        })
        return

    expected_identities = {
        (config, sequence, seed)
        for config in (
            FLOW_ADAPTIVE_PARENT_CONFIG,
            FLOW_ADAPTIVE_CANDIDATE_CONFIG,
        )
        for sequence in FLOW_ADAPTIVE_DEVELOPMENT_SEQUENCES
        for seed in FLOW_ADAPTIVE_REQUIRED_SEEDS
    }
    planned_identities = {
        (result.get("config"), result.get("sequence"), result.get("seed"))
        for result in relevant
    }
    duplicate_identities = sorted(
        identity for identity in planned_identities
        if sum(
            (result.get("config"), result.get("sequence"), result.get("seed")) ==
            identity
            for result in relevant
        ) != 1
    )
    missing_identities = sorted(expected_identities - planned_identities)
    unexpected_identities = sorted(planned_identities - expected_identities)
    by_key = {
        (result.get("config"), result.get("sequence"), result.get("seed")):
            result
        for result in relevant
    }
    incomplete_identities = sorted(
        identity for identity in expected_identities
        if identity in by_key and by_key[identity].get("status") != "complete"
    )

    source_states = [result.get("source") for result in relevant]
    source_records_present = all(
        isinstance(source, dict) for source in source_states
    )
    source_clean = bool(
        source_records_present and
        all(not source.get("dirty", True) for source in source_states)
    )
    source_snapshots = {
        (
            str(source.get("commit", "")),
            str(source.get("runtime_source_snapshot_sha256", "")),
        )
        for source in source_states
        if isinstance(source, dict)
    }
    single_source_snapshot = bool(
        source_records_present and len(source_snapshots) == 1
    )
    config_contract = flow_adaptive_config_contract()
    plan_contract = flow_adaptive_plan_contract(root)

    required_metrics = {
        "ate_rmse_m",
        "ate_p95_m",
        "rpe_translation_rmse_m",
        "rpe_translation_p95_m",
        "rpe_rotation_rmse_deg",
        "rpe_rotation_p95_deg",
        "failure_rate",
        "trajectory_coverage",
        "temporal_refinement_frames",
        "temporal_recovered_static_pixels",
        "temporal_flow_guard_valid_frames",
        "temporal_flow_guard_rejected_pixels",
        "tracking_recovery_audit_pixels",
        "tracking_recovery_mapping_leak_pixels",
    }
    invalid_metric_identities: list[tuple[str, str, int]] = []
    for identity in expected_identities:
        result = by_key.get(identity)
        if result is None or result.get("status") != "complete":
            continue
        metrics = result.get("metrics")
        if not isinstance(metrics, dict) or not required_metrics.issubset(metrics):
            invalid_metric_identities.append(identity)
            continue
        if any(
            not isinstance(metrics[name], (int, float)) or
            not math.isfinite(float(metrics[name]))
            for name in required_metrics
        ):
            invalid_metric_identities.append(identity)

    evidence_complete = not (
        duplicate_identities or missing_identities or unexpected_identities or
        incomplete_identities or invalid_metric_identities
    )
    execution_integrity = bool(
        evidence_complete and source_clean and single_source_snapshot and
        config_contract["valid"] and plan_contract["valid"]
    )

    sequence_checks: dict[str, Any] = {}
    paired_ate_wins = 0
    sequence_passes = 0
    all_coverage_complete = True
    all_actions_nonzero = True
    all_actions_separated = True
    all_mapping_leak_free = True
    if evidence_complete:
        for sequence in FLOW_ADAPTIVE_DEVELOPMENT_SEQUENCES:
            parent_runs = [
                by_key[(FLOW_ADAPTIVE_PARENT_CONFIG, sequence, seed)]
                for seed in FLOW_ADAPTIVE_REQUIRED_SEEDS
            ]
            candidate_runs = [
                by_key[(FLOW_ADAPTIVE_CANDIDATE_CONFIG, sequence, seed)]
                for seed in FLOW_ADAPTIVE_REQUIRED_SEEDS
            ]
            parent_metrics = [run["metrics"] for run in parent_runs]
            candidate_metrics = [run["metrics"] for run in candidate_runs]
            parent_ate = statistics.mean(
                metrics["ate_rmse_m"] for metrics in parent_metrics)
            candidate_ate = statistics.mean(
                metrics["ate_rmse_m"] for metrics in candidate_metrics)
            ate_change = relative_change(candidate_ate, parent_ate)
            ate_improvement = (
                -ate_change if ate_change is not None else None
            )
            sequence_paired_wins = sum(
                candidate["ate_rmse_m"] < parent["ate_rmse_m"]
                for candidate, parent in zip(
                    candidate_metrics, parent_metrics)
            )
            paired_ate_wins += sequence_paired_wins

            risk_changes: dict[str, float | None] = {}
            for metric in FLOW_ADAPTIVE_RISK_METRICS:
                parent_mean = statistics.mean(
                    values[metric] for values in parent_metrics)
                candidate_mean = statistics.mean(
                    values[metric] for values in candidate_metrics)
                risk_changes[metric] = relative_change(
                    candidate_mean, parent_mean)
            risk_pass = all(
                change is not None and
                change <= FLOW_ADAPTIVE_MAX_RISK_DEGRADATION
                for change in risk_changes.values()
            )
            parent_failure = statistics.mean(
                metrics["failure_rate"] for metrics in parent_metrics)
            candidate_failure = statistics.mean(
                metrics["failure_rate"] for metrics in candidate_metrics)
            failure_pass = (
                candidate_failure <=
                parent_failure + FLOW_ADAPTIVE_MAX_FAILURE_RATE_INCREASE
            )
            coverage_pass = all(
                metrics["trajectory_coverage"] == 1.0
                for metrics in parent_metrics + candidate_metrics
            )
            action_nonzero = all(
                metrics["temporal_refinement_frames"] > 0 and
                metrics["temporal_recovered_static_pixels"] > 0 and
                metrics["temporal_flow_guard_valid_frames"] > 0 and
                metrics["tracking_recovery_audit_pixels"] > 0
                for metrics in parent_metrics + candidate_metrics
            )
            action_separated = any(
                candidate["temporal_flow_guard_rejected_pixels"] !=
                    parent["temporal_flow_guard_rejected_pixels"] or
                candidate["tracking_recovery_audit_pixels"] !=
                    parent["tracking_recovery_audit_pixels"]
                for candidate, parent in zip(
                    candidate_metrics, parent_metrics)
            )
            mapping_leak_pixels = sum(
                int(metrics["tracking_recovery_mapping_leak_pixels"])
                for metrics in parent_metrics + candidate_metrics
            )
            mapping_leak_free = mapping_leak_pixels == 0
            ate_pass = bool(
                ate_improvement is not None and
                ate_improvement >= FLOW_ADAPTIVE_MIN_ATE_IMPROVEMENT
            )
            sequence_pass = bool(
                ate_pass and risk_pass and failure_pass and coverage_pass and
                action_nonzero and action_separated and mapping_leak_free
            )
            if sequence_pass:
                sequence_passes += 1
            all_coverage_complete &= coverage_pass
            all_actions_nonzero &= action_nonzero
            all_actions_separated &= action_separated
            all_mapping_leak_free &= mapping_leak_free
            sequence_checks[sequence] = {
                "parent_ate_mean_m": parent_ate,
                "candidate_ate_mean_m": candidate_ate,
                "candidate_ate_improvement_fraction": ate_improvement,
                "ate_improvement_at_least_3pct": ate_pass,
                "candidate_paired_ate_wins": sequence_paired_wins,
                "risk_relative_changes": risk_changes,
                "risk_degradation_within_2pct": risk_pass,
                "parent_failure_rate_mean": parent_failure,
                "candidate_failure_rate_mean": candidate_failure,
                "failure_rate_condition": failure_pass,
                "trajectory_coverage_complete": coverage_pass,
                "actions_nonzero": action_nonzero,
                "actions_separated": action_separated,
                "tracking_recovery_mapping_leak_pixels":
                    mapping_leak_pixels,
                "mapping_leak_free": mapping_leak_free,
                "sequence_pass": sequence_pass,
            }

    scientific_pass = bool(
        execution_integrity and
        sequence_passes >= FLOW_ADAPTIVE_MIN_SEQUENCE_PASSES and
        paired_ate_wins >= FLOW_ADAPTIVE_MIN_PAIRED_ATE_WINS and
        all_coverage_complete and all_actions_nonzero and
        all_actions_separated and all_mapping_leak_free
    )
    if not evidence_complete:
        status = "incomplete"
        decision = "NO_DECISION"
    elif not execution_integrity:
        status = "invalid_execution"
        decision = "REPAIR_INFRASTRUCTURE"
    elif not all_mapping_leak_free:
        status = "safety_failure"
        decision = "SAFETY_FAILURE"
    elif scientific_pass:
        status = "pass"
        decision = "DEVELOPMENT_ADVANCE"
    else:
        status = "fail"
        decision = "STOP_ADAPTIVE_RADIUS"

    atomic_json(output, {
        "protocol": "flow-adaptive-vs-exact-incumbent-gate-v1",
        "status": status,
        "decision": decision,
        "candidate": FLOW_ADAPTIVE_CANDIDATE_CONFIG,
        "parent": FLOW_ADAPTIVE_PARENT_CONFIG,
        "required_sequences": list(FLOW_ADAPTIVE_DEVELOPMENT_SEQUENCES),
        "required_seeds": list(FLOW_ADAPTIVE_REQUIRED_SEEDS),
        "source_clean": source_clean,
        "single_source_snapshot": single_source_snapshot,
        "source_snapshots": [list(item) for item in sorted(source_snapshots)],
        "config_contract": config_contract,
        "plan_contract": plan_contract,
        "execution_integrity": execution_integrity,
        "missing_identities": [list(item) for item in missing_identities],
        "unexpected_identities": [list(item) for item in unexpected_identities],
        "duplicate_identities": [list(item) for item in duplicate_identities],
        "incomplete_identities": [list(item) for item in incomplete_identities],
        "invalid_metric_identities": [
            list(item) for item in invalid_metric_identities
        ],
        "sequence_passes": sequence_passes,
        "required_sequence_passes": FLOW_ADAPTIVE_MIN_SEQUENCE_PASSES,
        "paired_ate_wins": paired_ate_wins,
        "required_paired_ate_wins": FLOW_ADAPTIVE_MIN_PAIRED_ATE_WINS,
        "predeclared_rule": (
            "Exactly two cells, three development sequences, and seeds 0, 1, "
            "and 2 must complete from one clean source snapshot. The configs "
            "may differ only in the frozen exact-versus-adaptive radius keys. "
            "Every run must have 100% trajectory coverage, nonzero temporal "
            "recovery and valid flow-guard action, and zero recovered-tracking "
            "pixels admitted as static Gaussian-mapping support. The action "
            "must differ between cells. Adaptive must improve mean ATE by at "
            "least 3% on at least two of three sequences, win at least six of "
            "nine paired-seed ATE comparisons, keep mean ATE p95 and both "
            "translation/rotation RPE RMSE and p95 within 2% of Parent on each "
            "sequence, and increase failure rate by at most 0.5 percentage "
            "points. Thresholds cannot be changed after observing results."
        ),
        "sequences": sequence_checks,
    })


def write_motion_counterfactual_gate(
    root: Path,
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    planned_by_pair: dict[tuple[str, str], set[int]],
    shadow_config: str,
    sequences: tuple[str, ...],
    output_filename: str,
    retained_key: str,
    label: str,
    required_seeds: tuple[int, ...] | None = None,
) -> None:
    required_fields = {
        "pairs",
        "static_translation_sum_squared_error_m2",
        "dynamic_translation_sum_squared_error_m2",
        "dynamic_translation_wins",
        "worst_translation_degradation_m",
    }
    sequence_checks = {}
    complete_evidence = True
    for sequence in sequences:
        planned_seeds = sorted(
            planned_by_pair.get((shadow_config, sequence), set()))
        if not planned_seeds:
            sequence_checks[sequence] = {
                "status": "not_planned",
                "planned_seeds": [],
                "required_seeds": (
                    list(required_seeds) if required_seeds is not None
                    else None),
                "seed_plan_matches": False,
                "missing_runs": [],
                "missing_evidence": [],
                "selected_pairs": 0,
                "mechanism_pass": False,
            }
            complete_evidence = False
            continue

        missing_runs = [
            f"{shadow_config}/seed_{seed}"
            for seed in planned_seeds
            if (shadow_config, sequence, seed) not in by_key
        ]
        summaries = []
        missing_evidence = []
        for seed in planned_seeds:
            run = by_key.get((shadow_config, sequence, seed))
            if run is None:
                continue
            summary = (
                run.get("metrics", {})
                .get("motion_prior_counterfactual", {})
                .get("gate_would_use"))
            if (
                not isinstance(summary, dict) or
                not required_fields.issubset(summary)
            ):
                missing_evidence.append(
                    f"{shadow_config}/seed_{seed}/counterfactual")
            else:
                summaries.append(summary)

        seed_plan_matches = (
            required_seeds is None or
            planned_seeds == list(required_seeds))
        sequence_complete = (
            len(planned_seeds) >= 3 and
            seed_plan_matches and
            not missing_runs and
            not missing_evidence and
            len(summaries) == len(planned_seeds))
        complete_evidence &= sequence_complete
        pairs = sum(item["pairs"] for item in summaries)
        static_sse = sum(
            item["static_translation_sum_squared_error_m2"]
            for item in summaries)
        dynamic_sse = sum(
            item["dynamic_translation_sum_squared_error_m2"]
            for item in summaries)
        wins = sum(item["dynamic_translation_wins"] for item in summaries)
        static_rmse = math.sqrt(static_sse / pairs) if pairs else None
        dynamic_rmse = math.sqrt(dynamic_sse / pairs) if pairs else None
        win_rate = wins / pairs if pairs else None
        per_run_worst = [
            item["worst_translation_degradation_m"]
            for item in summaries
            if item["worst_translation_degradation_m"] is not None
        ]
        worst_degradation = max(per_run_worst) if per_run_worst else None
        mechanism_pass = bool(
            sequence_complete and
            pairs >= 5 and
            win_rate is not None and win_rate >= 0.60 and
            dynamic_rmse is not None and static_rmse is not None and
            dynamic_rmse <= static_rmse and
            worst_degradation is not None and worst_degradation <= 0.01)
        sequence_checks[sequence] = {
            "status": "complete" if sequence_complete else "incomplete",
            "planned_seeds": planned_seeds,
            "required_seeds": (
                list(required_seeds) if required_seeds is not None else None),
            "seed_plan_matches": seed_plan_matches,
            "missing_runs": missing_runs,
            "missing_evidence": missing_evidence,
            "selected_pairs": pairs,
            "dynamic_translation_wins": wins,
            "dynamic_translation_win_rate": win_rate,
            "static_translation_rmse_m": static_rmse,
            "dynamic_translation_rmse_m": dynamic_rmse,
            "worst_translation_degradation_m": worst_degradation,
            "mechanism_pass": mechanism_pass,
        }

    retained = bool(
        sequence_checks and complete_evidence and
        all(item["mechanism_pass"] for item in sequence_checks.values()))
    decision = {
        "status": (
            "pass" if retained else
            "fail" if sequence_checks and complete_evidence else
            "insufficient_data"),
        retained_key: retained,
        "prior_information_scale": 0.5,
        "predeclared_rule": (
            f"On both fresh held-out sequences, three planned {label} shadow "
            "seeds must complete, the frozen consensus gate must select at "
            "least five same-state counterfactual pairs, dynamic translation "
            "must win at least 60% of pairs, pooled dynamic translation RMSE "
            "must not exceed pooled static translation RMSE, and the worst "
            "per-pair translation degradation must not exceed 0.01 m."
        ),
        "sequences": sequence_checks,
    }
    atomic_json(root / output_filename, decision)


def write_tempered_counterfactual_gate(
    root: Path,
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    planned_by_pair: dict[tuple[str, str], set[int]],
) -> None:
    write_motion_counterfactual_gate(
        root, by_key, planned_by_pair,
        "semantic_motion_tempered_shadow",
        MOTION_TEMPERED_HELDOUT_SEQUENCES,
        "motion_tempered_counterfactual_gate.json",
        "motion_tempered_gate_retained",
        "tempered",
    )


def write_direct_counterfactual_gate(
    root: Path,
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    planned_by_pair: dict[tuple[str, str], set[int]],
) -> None:
    write_motion_counterfactual_gate(
        root, by_key, planned_by_pair,
        "semantic_motion_direct_shadow",
        MOTION_DIRECT_HELDOUT_SEQUENCES,
        "motion_direct_counterfactual_gate.json",
        "motion_direct_gate_retained",
        "direct-validated",
        MOTION_DIRECT_HELDOUT_SEEDS,
    )


def write_rgbd_counterfactual_gate(
    root: Path,
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    planned_by_pair: dict[tuple[str, str], set[int]],
) -> None:
    write_motion_counterfactual_gate(
        root, by_key, planned_by_pair,
        "semantic_motion_rgbd_shadow",
        MOTION_RGBD_HELDOUT_SEQUENCES,
        "motion_rgbd_counterfactual_gate.json",
        "motion_rgbd_gate_retained",
        "combined-RGB-D-validated",
        MOTION_RGBD_HELDOUT_SEEDS,
    )


def write_motion_end_to_end_gate(
    root: Path,
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    planned_by_pair: dict[tuple[str, str], set[int]],
    shadow_config: str,
    intervention_config: str,
    sequences: tuple[str, ...],
    output_filename: str,
    retained_key: str,
    prefix: str,
    label: str,
    required_seeds: tuple[int, ...] | None = None,
    mechanism_gate_filename: str | None = None,
    mechanism_retained_key: str | None = None,
) -> None:
    configs = ("semantic", shadow_config, intervention_config)
    sequence_checks = {}
    paired_results = []
    failures = {config: [] for config in configs}
    complete_evidence = True

    for sequence in sequences:
        planned_sets = {
            config: planned_by_pair.get((config, sequence), set())
            for config in configs
        }
        if not all(planned_sets.values()):
            sequence_checks[sequence] = {
                "status": "not_planned",
                "planned_seeds": sorted(set.union(*planned_sets.values())),
                "required_seeds": (
                    list(required_seeds) if required_seeds is not None
                    else None),
                "seed_plan_matches": False,
                "paired_seeds": [],
                "same_seed_plan": False,
                "missing_runs": [],
            }
            complete_evidence = False
            continue

        planned_seeds = sorted(set.union(*planned_sets.values()))
        missing_runs = [
            f"{config}/seed_{seed}"
            for config in configs
            for seed in planned_seeds
            if (config, sequence, seed) not in by_key
        ]
        same_seed_plan = all(
            seeds == planned_sets["semantic"]
            for seeds in planned_sets.values())
        seed_plan_matches = (
            required_seeds is None or
            planned_seeds == list(required_seeds))
        sequence_complete = (
            len(planned_seeds) >= 3 and
            same_seed_plan and
            seed_plan_matches and
            not missing_runs)
        complete_evidence &= sequence_complete
        common_seeds = [
            seed for seed in planned_seeds
            if all((config, sequence, seed) in by_key for config in configs)
        ]
        runs = {
            config: [
                by_key[(config, sequence, seed)]
                for seed in common_seeds]
            for config in configs
        }
        ate = {
            config: [run["metrics"]["ate_rmse_m"] for run in config_runs]
            for config, config_runs in runs.items()
        }
        mean_not_worse_than_semantic = bool(
            common_seeds and
            statistics.mean(ate[intervention_config]) <=
            statistics.mean(ate["semantic"]))
        mean_not_worse_than_shadow = bool(
            common_seeds and
            statistics.mean(ate[intervention_config]) <=
            statistics.mean(ate[shadow_config]))
        selected_seed_count = sum(
            run["metrics"].get("used_motion_priors", 0) > 0
            for run in runs[intervention_config])
        selected_seed_rate = (
            selected_seed_count / len(common_seeds)
            if common_seeds else None)
        sequence_checks[sequence] = {
            "status": "complete" if sequence_complete else "incomplete",
            "planned_seeds": planned_seeds,
            "required_seeds": (
                list(required_seeds) if required_seeds is not None else None),
            "seed_plan_matches": seed_plan_matches,
            "paired_seeds": common_seeds,
            "same_seed_plan": same_seed_plan,
            "missing_runs": missing_runs,
            "semantic_ate_mean_m": (
                statistics.mean(ate["semantic"]) if common_seeds else None),
            f"{prefix}_shadow_ate_mean_m": (
                statistics.mean(ate[shadow_config])
                if common_seeds else None),
            f"{prefix}_ate_mean_m": (
                statistics.mean(ate[intervention_config])
                if common_seeds else None),
            f"{prefix}_mean_not_worse_than_semantic":
                mean_not_worse_than_semantic,
            f"{prefix}_mean_not_worse_than_shadow":
                mean_not_worse_than_shadow,
            f"{prefix}_paired_wins_over_shadow": sum(
                intervention < shadow
                for intervention, shadow in zip(
                    ate[intervention_config], ate[shadow_config])),
            f"{prefix}_selected_seed_count": selected_seed_count,
            f"{prefix}_selected_seed_rate": selected_seed_rate,
            f"{prefix}_selected_in_at_least_two_thirds_seeds": (
                selected_seed_rate is not None and
                selected_seed_rate >= (2.0 / 3.0)),
        }
        paired_results.extend(zip(
            ate[intervention_config], ate[shadow_config]))
        for config in configs:
            failures[config].extend(
                run["metrics"]["failure_rate"] for run in runs[config])

    mechanism_gate_status = None
    mechanism_gate_condition = True
    mechanism_gate_evidence_complete = True
    if mechanism_gate_filename is not None:
        mechanism_gate_path = root / mechanism_gate_filename
        mechanism_gate = (
            json.loads(mechanism_gate_path.read_text())
            if mechanism_gate_path.is_file() else {})
        mechanism_gate_status = mechanism_gate.get("status")
        mechanism_gate_evidence_complete = (
            mechanism_gate_status in {"pass", "fail"})
        mechanism_gate_condition = bool(
            mechanism_gate_status == "pass" and
            mechanism_retained_key is not None and
            mechanism_gate.get(mechanism_retained_key) is True)

    win_rate = (
        sum(intervention < shadow
            for intervention, shadow in paired_results) /
        len(paired_results)
        if paired_results else None)
    failure_condition = bool(
        paired_results and
        statistics.mean(failures[intervention_config]) <=
        statistics.mean(failures["semantic"]) + 0.005 and
        statistics.mean(failures[intervention_config]) <=
        statistics.mean(failures[shadow_config]) + 0.005)
    retained = bool(
        sequence_checks and complete_evidence and
        mechanism_gate_condition and
        all(
            item[f"{prefix}_mean_not_worse_than_semantic"] and
            item[f"{prefix}_mean_not_worse_than_shadow"] and
            item[f"{prefix}_selected_in_at_least_two_thirds_seeds"]
            for item in sequence_checks.values()) and
        win_rate is not None and win_rate >= 0.70 and
        failure_condition)
    decision = {
        "status": (
            "pass" if retained else
            "fail" if (
                sequence_checks and complete_evidence and
                mechanism_gate_evidence_complete) else
            "insufficient_data"),
        retained_key: retained,
        "prior_information_scale": 0.5,
        "mechanism_gate_status": mechanism_gate_status,
        "mechanism_gate_condition": mechanism_gate_condition,
        "predeclared_rule": (
            "All three configurations must complete the same three or more "
            f"seeds on every fresh held-out sequence. {label} mean ATE must "
            f"not exceed Semantic or matched {label} Shadow on any sequence, "
            f"{label} must beat Shadow on at least 70% of all paired seeds, "
            "its mean failure rate must not exceed either control by more "
            "than 0.5 percentage points, and it must be selected in at least "
            "two thirds of seeds on every sequence. When a prerequisite "
            "mechanism gate is named, that gate must also pass."
        ),
        "paired_win_rate_over_shadow": win_rate,
        "failure_rate_condition": failure_condition,
        "failure_rate_means": {
            config: statistics.mean(values) if values else None
            for config, values in failures.items()
        },
        "sequences": sequence_checks,
    }
    atomic_json(root / output_filename, decision)


def write_tempered_end_to_end_gate(
    root: Path,
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    planned_by_pair: dict[tuple[str, str], set[int]],
) -> None:
    write_motion_end_to_end_gate(
        root, by_key, planned_by_pair,
        "semantic_motion_tempered_shadow",
        "semantic_motion_tempered",
        MOTION_TEMPERED_HELDOUT_SEQUENCES,
        "motion_tempered_end_to_end_gate.json",
        "motion_tempered_intervention_retained",
        "tempered",
        "Tempered",
    )


def write_direct_end_to_end_gate(
    root: Path,
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    planned_by_pair: dict[tuple[str, str], set[int]],
) -> None:
    write_motion_end_to_end_gate(
        root, by_key, planned_by_pair,
        "semantic_motion_direct_shadow",
        "semantic_motion_direct",
        MOTION_DIRECT_HELDOUT_SEQUENCES,
        "motion_direct_end_to_end_gate.json",
        "motion_direct_intervention_retained",
        "direct",
        "Direct-validated",
        MOTION_DIRECT_HELDOUT_SEEDS,
        "motion_direct_counterfactual_gate.json",
        "motion_direct_gate_retained",
    )


def write_rgbd_end_to_end_gate(
    root: Path,
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    planned_by_pair: dict[tuple[str, str], set[int]],
) -> None:
    write_motion_end_to_end_gate(
        root, by_key, planned_by_pair,
        "semantic_motion_rgbd_shadow",
        "semantic_motion_rgbd",
        MOTION_RGBD_HELDOUT_SEQUENCES,
        "motion_rgbd_end_to_end_gate.json",
        "motion_rgbd_intervention_retained",
        "rgbd",
        "Combined-RGB-D-validated",
        MOTION_RGBD_HELDOUT_SEEDS,
        "motion_rgbd_counterfactual_gate.json",
        "motion_rgbd_gate_retained",
    )


def write_shadow_translation_tracking_gate(
    root: Path,
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    planned_by_pair: dict[tuple[str, str], set[int]],
    sequences: tuple[str, ...],
    output_filename: str,
    stage: str,
) -> None:
    required_metrics = {
        "ate_rmse_m",
        "ate_p95_m",
        "rpe_translation_rmse_m",
        "rpe_translation_p95_m",
        "rpe_rotation_rmse_deg",
        "failure_rate",
        "end_to_end_seconds",
        "shadow_translation_applied_frames",
        "shadow_translation_factor_injections",
        "velocity_neutralized_frames",
        "synchronize_local_mapping",
        "synchronize_loop_closing",
        "gaussian_mapper_enabled",
    }
    sequence_checks: dict[str, Any] = {}
    complete_evidence = True

    for sequence in sequences:
        planned_sets = {
            config: planned_by_pair.get((config, sequence), set())
            for config in SHADOW_TRANSLATION_TRACKING_CONFIGS
        }
        planned_seeds = sorted(set().union(*planned_sets.values()))
        missing_configs = [
            config for config, seeds in planned_sets.items() if not seeds
        ]
        exact_seed_plan = all(
            seeds == set(SHADOW_TRANSLATION_REQUIRED_SEEDS)
            for seeds in planned_sets.values())
        same_seed_plan = bool(planned_sets) and all(
            seeds == planned_sets[SHADOW_TRANSLATION_TRACKING_CONFIGS[0]]
            for seeds in planned_sets.values())
        missing_runs = [
            f"{config}/seed_{seed}"
            for config in SHADOW_TRANSLATION_TRACKING_CONFIGS
            for seed in SHADOW_TRANSLATION_REQUIRED_SEEDS
            if (config, sequence, seed) not in by_key
        ]
        missing_metrics = [
            f"{config}/seed_{seed}/{metric}"
            for config in SHADOW_TRANSLATION_TRACKING_CONFIGS
            for seed in SHADOW_TRANSLATION_REQUIRED_SEEDS
            if (config, sequence, seed) in by_key
            for metric in sorted(
                required_metrics -
                set(by_key[(config, sequence, seed)].get("metrics", {})))
        ]
        sequence_complete = (
            not missing_configs and exact_seed_plan and same_seed_plan and
            not missing_runs and not missing_metrics)
        synchronized_execution = bool(
            sequence_complete and all(
                by_key[(config, sequence, seed)]["metrics"][
                    "synchronize_local_mapping"] and
                by_key[(config, sequence, seed)]["metrics"][
                    "synchronize_loop_closing"]
                for config in SHADOW_TRANSLATION_TRACKING_CONFIGS
                for seed in SHADOW_TRANSLATION_REQUIRED_SEEDS))
        isolated_tracking_execution = bool(
            sequence_complete and all(
                not by_key[(config, sequence, seed)]["metrics"][
                    "gaussian_mapper_enabled"]
                for config in SHADOW_TRANSLATION_TRACKING_CONFIGS
                for seed in SHADOW_TRANSLATION_REQUIRED_SEEDS))
        sequence_complete = (
            sequence_complete and synchronized_execution and
            isolated_tracking_execution)
        complete_evidence &= sequence_complete

        check: dict[str, Any] = {
            "status": "complete" if sequence_complete else "incomplete",
            "planned_seeds": planned_seeds,
            "required_seeds": list(SHADOW_TRANSLATION_REQUIRED_SEEDS),
            "same_seed_plan": same_seed_plan,
            "exact_seed_plan": exact_seed_plan,
            "missing_configs": missing_configs,
            "missing_runs": missing_runs,
            "missing_metrics": missing_metrics,
            "synchronized_execution": synchronized_execution,
            "isolated_tracking_execution": isolated_tracking_execution,
            "comparisons": {},
            "sequence_pass": False,
        }
        if not sequence_complete:
            sequence_checks[sequence] = check
            continue

        runs = {
            config: [
                by_key[(config, sequence, seed)]
                for seed in SHADOW_TRANSLATION_REQUIRED_SEEDS
            ]
            for config in SHADOW_TRANSLATION_TRACKING_CONFIGS
        }
        metric_values = {
            config: {
                metric: [run["metrics"][metric] for run in config_runs]
                for metric in required_metrics
            }
            for config, config_runs in runs.items()
        }
        candidate = metric_values[SHADOW_TRANSLATION_CONFIG]
        applied_seed_count = sum(
            value > 0
            for value in candidate["shadow_translation_applied_frames"])
        injected_seed_count = sum(
            value > 0
            for value in candidate["shadow_translation_factor_injections"])
        velocity_contract_pass = all(
            applied == neutralized and injected >= applied
            for applied, injected, neutralized in zip(
                candidate["shadow_translation_applied_frames"],
                candidate["shadow_translation_factor_injections"],
                candidate["velocity_neutralized_frames"]))
        intervention_coverage_pass = injected_seed_count >= 2

        comparisons = {}
        for control_name in SHADOW_TRANSLATION_CONTROLS:
            control = metric_values[control_name]
            ate_wins = sum(
                candidate_value < control_value
                for candidate_value, control_value in zip(
                    candidate["ate_rmse_m"], control["ate_rmse_m"]))
            rpe_wins = sum(
                candidate_value < control_value
                for candidate_value, control_value in zip(
                    candidate["rpe_translation_rmse_m"],
                    control["rpe_translation_rmse_m"]))
            comparison = {
                "candidate_ate_mean_m":
                    statistics.mean(candidate["ate_rmse_m"]),
                "control_ate_mean_m":
                    statistics.mean(control["ate_rmse_m"]),
                "candidate_ate_p95_mean_m":
                    statistics.mean(candidate["ate_p95_m"]),
                "control_ate_p95_mean_m":
                    statistics.mean(control["ate_p95_m"]),
                "candidate_rpe_translation_mean_m":
                    statistics.mean(candidate["rpe_translation_rmse_m"]),
                "control_rpe_translation_mean_m":
                    statistics.mean(control["rpe_translation_rmse_m"]),
                "candidate_rpe_translation_p95_mean_m":
                    statistics.mean(candidate["rpe_translation_p95_m"]),
                "control_rpe_translation_p95_mean_m":
                    statistics.mean(control["rpe_translation_p95_m"]),
                "candidate_rpe_rotation_mean_deg":
                    statistics.mean(candidate["rpe_rotation_rmse_deg"]),
                "control_rpe_rotation_mean_deg":
                    statistics.mean(control["rpe_rotation_rmse_deg"]),
                "candidate_failure_rate_mean":
                    statistics.mean(candidate["failure_rate"]),
                "control_failure_rate_mean":
                    statistics.mean(control["failure_rate"]),
                "candidate_worst_seed_ate_m":
                    max(candidate["ate_rmse_m"]),
                "control_worst_seed_ate_m":
                    max(control["ate_rmse_m"]),
                "candidate_worst_seed_rpe_translation_p95_m":
                    max(candidate["rpe_translation_p95_m"]),
                "control_worst_seed_rpe_translation_p95_m":
                    max(control["rpe_translation_p95_m"]),
                "paired_ate_wins": ate_wins,
                "paired_rpe_translation_wins": rpe_wins,
            }
            comparison.update({
                "ate_mean_strictly_better":
                    comparison["candidate_ate_mean_m"] <
                    comparison["control_ate_mean_m"],
                "ate_p95_mean_strictly_better":
                    comparison["candidate_ate_p95_mean_m"] <
                    comparison["control_ate_p95_mean_m"],
                "rpe_translation_mean_strictly_better":
                    comparison["candidate_rpe_translation_mean_m"] <
                    comparison["control_rpe_translation_mean_m"],
                "rpe_translation_p95_mean_strictly_better":
                    comparison["candidate_rpe_translation_p95_mean_m"] <
                    comparison["control_rpe_translation_p95_mean_m"],
                "worst_seed_ate_not_worse":
                    comparison["candidate_worst_seed_ate_m"] <=
                    comparison["control_worst_seed_ate_m"],
                "worst_seed_rpe_translation_p95_not_worse":
                    comparison[
                        "candidate_worst_seed_rpe_translation_p95_m"] <=
                    comparison[
                        "control_worst_seed_rpe_translation_p95_m"],
                "paired_ate_win_rate_at_least_two_thirds": ate_wins >= 2,
                "paired_rpe_translation_win_rate_at_least_two_thirds":
                    rpe_wins >= 2,
                "rotation_not_materially_worse":
                    comparison["candidate_rpe_rotation_mean_deg"] <=
                    comparison["control_rpe_rotation_mean_deg"] + 0.01,
                "failure_rate_not_materially_worse":
                    comparison["candidate_failure_rate_mean"] <=
                    comparison["control_failure_rate_mean"] + 0.005,
            })
            required_checks = (
                "ate_mean_strictly_better",
                "ate_p95_mean_strictly_better",
                "rpe_translation_mean_strictly_better",
                "rpe_translation_p95_mean_strictly_better",
                "worst_seed_ate_not_worse",
                "worst_seed_rpe_translation_p95_not_worse",
                "paired_ate_win_rate_at_least_two_thirds",
                "paired_rpe_translation_win_rate_at_least_two_thirds",
                "rotation_not_materially_worse",
                "failure_rate_not_materially_worse",
            )
            comparison["comparison_pass"] = all(
                comparison[name] for name in required_checks)
            comparisons[control_name] = comparison

        semantic_time = statistics.mean(
            metric_values["semantic"]["end_to_end_seconds"])
        candidate_time = statistics.mean(candidate["end_to_end_seconds"])
        check.update({
            "shadow_translation_applied_seed_count": applied_seed_count,
            "shadow_translation_factor_injected_seed_count":
                injected_seed_count,
            "intervention_factor_injected_in_at_least_two_thirds_seeds":
                intervention_coverage_pass,
            "velocity_neutral_contract_pass": velocity_contract_pass,
            "candidate_end_to_end_mean_seconds": candidate_time,
            "semantic_end_to_end_mean_seconds": semantic_time,
            "runtime_ratio_vs_semantic": (
                candidate_time / semantic_time if semantic_time > 0 else None),
            "comparisons": comparisons,
            "sequence_pass": (
                intervention_coverage_pass and velocity_contract_pass and
                all(item["comparison_pass"] for item in comparisons.values())),
        })
        sequence_checks[sequence] = check

    retained = bool(
        sequence_checks and complete_evidence and
        all(check["sequence_pass"] for check in sequence_checks.values()))
    status = (
        "pass" if retained else
        "fail" if sequence_checks and complete_evidence else
        "insufficient_data")
    atomic_json(root / output_filename, {
        "protocol": "schur-shadow-translation-tracking-gate-v5",
        "stage": stage,
        "status": status,
        "shadow_translation_tracking_retained": retained,
        "candidate": SHADOW_TRANSLATION_CONFIG,
        "controls": list(SHADOW_TRANSLATION_CONTROLS),
        "required_sequences": list(sequences),
        "required_seeds": list(SHADOW_TRANSLATION_REQUIRED_SEEDS),
        "frozen_parameters": SHADOW_TRANSLATION_FROZEN_PARAMETERS,
        "predeclared_rule": (
            "All four variants must complete exactly seeds 0, 1, and 2 on "
            "every required sequence with LocalMapping and LoopClosing "
            "synchronized after each frame and Gaussian mapping disabled. "
            "Against each of Semantic, matched "
            "Shadow, and causal Lag-30, Schur shadow-translation must have "
            "strictly lower mean full-frame ATE, ATE p95, translational RPE, "
            "and translational RPE p95; win paired ATE and translational RPE "
            "on at least two of three seeds; not worsen worst-seed ATE or "
            "translational RPE p95; keep mean rotational RPE within 0.01 "
            "degrees and failure rate within 0.5 percentage points; and "
            "inject the translation-only factor into local-map optimization "
            "in at least two of three seeds. Every applied factor must retain "
            "the incoming constant-velocity increment for exactly one update. "
            "Runtime is reported but is not a pass condition."
        ),
        "sequences": sequence_checks,
    })


def write_motion_ablation_summary(
    root: Path,
    results: list[dict[str, Any]],
) -> None:
    planned_configs = [
        config for config in MOTION_ABLATION_CONFIGS
        if any(result["config"] == config for result in results)
    ]
    if not planned_configs:
        atomic_json(root / "motion_ablation_summary.json", {
            "protocol": "motion-ablation-v1",
            "status": "not_planned",
            "required_configs": list(MOTION_ABLATION_CONFIGS),
            "sequences": {},
        })
        return

    complete = {
        (result["config"], result["sequence"], result["seed"]): result
        for result in results
        if result["status"] == "complete"
    }
    sequences: dict[str, Any] = {}
    for sequence in sorted({result["sequence"] for result in results}):
        planned = {
            config: {
                result["seed"] for result in results
                if result["config"] == config and
                result["sequence"] == sequence
            }
            for config in MOTION_ABLATION_CONFIGS
        }
        missing_configs = [
            config for config, seeds in planned.items() if not seeds
        ]
        seed_union = sorted(set().union(*planned.values()))
        same_seed_plan = (
            not missing_configs and
            all(
                seeds == planned[MOTION_ABLATION_CONFIGS[0]]
                for seeds in planned.values()
            )
        )
        missing_runs = [
            f"{config}/seed_{seed}"
            for config in MOTION_ABLATION_CONFIGS
            for seed in seed_union
            if (config, sequence, seed) not in complete
        ]
        variants: dict[str, Any] = {}
        for config in MOTION_ABLATION_CONFIGS:
            runs = [
                complete[(config, sequence, seed)]
                for seed in seed_union
                if (config, sequence, seed) in complete
            ]
            if not runs:
                variants[config] = {"successful_runs": 0}
                continue
            variants[config] = {
                "successful_runs": len(runs),
                "ate_mean_m": statistics.mean(
                    run["metrics"]["ate_rmse_m"] for run in runs),
                "rpe_translation_mean_m": statistics.mean(
                    run["metrics"]["rpe_translation_rmse_m"]
                    for run in runs),
                "rpe_rotation_mean_deg": statistics.mean(
                    run["metrics"]["rpe_rotation_rmse_deg"]
                    for run in runs),
                "failure_rate_mean": statistics.mean(
                    run["metrics"]["failure_rate"] for run in runs),
                "end_to_end_mean_seconds": statistics.mean(
                    run["metrics"]["end_to_end_seconds"] for run in runs),
                "valid_motion_priors_mean": statistics.mean(
                    run["metrics"]["valid_motion_priors"] for run in runs),
                "would_use_motion_priors_mean": statistics.mean(
                    run["metrics"]["would_use_motion_priors"]
                    for run in runs),
                "used_motion_priors_mean": statistics.mean(
                    run["metrics"]["used_motion_priors"] for run in runs),
            }
        complete_sequence = (
            not missing_configs and same_seed_plan and not missing_runs
        )
        sequences[sequence] = {
            "status": "complete" if complete_sequence else "incomplete",
            "planned_seeds": seed_union,
            "same_seed_plan": same_seed_plan,
            "claim_ready_three_seed_minimum": (
                complete_sequence and len(seed_union) >= 3
            ),
            "missing_configs": missing_configs,
            "missing_runs": missing_runs,
            "variants": variants,
        }

    all_complete = bool(sequences) and all(
        item["status"] == "complete" for item in sequences.values())
    atomic_json(root / "motion_ablation_summary.json", {
        "protocol": "motion-ablation-v1",
        "status": "complete" if all_complete else "insufficient_data",
        "required_configs": list(MOTION_ABLATION_CONFIGS),
        "control_semantics": {
            "semantic": "no motion reuse",
            "semantic_motion_init_rgbd":
                "full-gated dynamic initialization without an SE3 "
                "information factor",
            "semantic_motion_ungated":
                "Schur information factor with reliability gates bypassed",
            "semantic_motion_rgbd":
                "Schur information factor with the full RGB-D gate",
            "semantic_motion_rgbd_shadow":
                "matched full gate evaluated without intervention",
            "semantic_motion_rgbd_shuffled":
                "matched translation-only action with a causal 30-frame "
                "prior lag",
        },
        "sequences": sequences,
    })


def write_dyn19_phase_reports(
    root: Path,
    results: list[dict[str, Any]],
) -> None:
    plan_audit = dyn19_phase_plan_contract(root)
    if plan_audit.get("error") == "plan is not a DYN-19 phase plan":
        return

    phase = plan_audit.get("phase")
    if phase not in DYN19_PHASE_CONFIGS:
        atomic_json(root / "dyn19_phase_report.json", {
            "experiment_id": DYN19_EXPERIMENT_ID,
            "status": "invalid_plan",
            "plan_audit": plan_audit,
        })
        return

    expected_configs = DYN19_PHASE_CONFIGS[phase]
    expected = {
        (config, sequence, seed)
        for config in expected_configs
        for sequence in DYN19_CLAIM_SEQUENCES
        for seed in DYN19_REQUIRED_SEEDS
    }
    result_by_identity: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicate_results = []
    for result in results:
        identity = (
            result.get("config"),
            result.get("sequence"),
            result.get("seed"),
        )
        if identity in result_by_identity:
            duplicate_results.append(identity)
        result_by_identity[identity] = result

    complete = {
        identity: result
        for identity, result in result_by_identity.items()
        if identity in expected and result.get("status") == "complete"
    }
    missing = sorted(
        identity for identity in expected if identity not in complete)
    unexpected = sorted(
        identity for identity in result_by_identity if identity not in expected)

    by_config: dict[str, dict[str, Any]] = {}
    for config in expected_configs:
        expected_config = {
            (config, sequence, seed)
            for sequence in DYN19_CLAIM_SEQUENCES
            for seed in DYN19_REQUIRED_SEEDS
        }
        config_complete = {
            identity: result
            for identity, result in complete.items()
            if identity in expected_config
        }
        certificates = {}
        for identity, result in config_complete.items():
            certificate = result.get("metrics", {}).get(
                "recovered_support_overlap_certificate")
            certificates[f"{identity[1]}/seed_{identity[2]:04d}"] = (
                certificate if isinstance(certificate, dict) else {
                    "status": "missing"})
        by_config[config] = {
            "factors": DYN19_FACTOR_MATRIX[config],
            "expected_runs": len(expected_config),
            "completed_runs": len(config_complete),
            "missing_runs": [
                f"{sequence}/seed_{seed:04d}"
                for _, sequence, seed in sorted(expected_config - set(config_complete))
            ],
            "certificates": certificates,
        }

    full_certificate_report: dict[str, Any] = {
        "status": "not_planned",
        "safety_status": "not_planned",
        "positive_recovery_evidence": "not_planned",
        "per_sequence_seed": {},
    }
    if "dyn19_full" in expected_configs:
        full_runs = by_config["dyn19_full"]
        per_sequence_seed = full_runs["certificates"]
        certificate_statuses = [
            certificate.get("status", "missing")
            for certificate in per_sequence_seed.values()
        ]
        leak_values = [
            certificate.get("evidence", {}).get(
                "tracking_recovery_mapping_leak_pixels", None)
            for certificate in per_sequence_seed.values()
        ]
        complete_full = full_runs["completed_runs"] == full_runs["expected_runs"]
        no_leak = (
            complete_full and
            len(leak_values) == full_runs["expected_runs"] and
            all(isinstance(value, int) and value == 0 for value in leak_values)
        )
        certificate_failures = any(status == "fail" for status in certificate_statuses)
        pass_count = sum(status == "pass" for status in certificate_statuses)
        vacuous_count = sum(
            status == "vacuous" for status in certificate_statuses)
        if not complete_full:
            safety_status = "incomplete"
            aggregate_status = "incomplete"
        elif certificate_failures or not no_leak:
            safety_status = "fail"
            aggregate_status = "fail"
        elif pass_count > 0:
            safety_status = "pass"
            aggregate_status = "pass"
        else:
            # Complete zero-leak runs with no recovered support are safe but
            # cannot substantiate a recovery claim.
            safety_status = "pass"
            aggregate_status = "vacuous"
        full_certificate_report = {
            "contract": "dyn19-main-recovered-support-overlap-aggregate-v1",
            "status": aggregate_status,
            "safety_status": safety_status,
            "zero_leak_required": True,
            "all_completed": complete_full,
            "all_zero_leak": no_leak,
            "certificate_status_counts": {
                "pass": pass_count,
                "vacuous": vacuous_count,
                "fail": sum(status == "fail" for status in certificate_statuses),
                "missing": sum(
                    status not in {"pass", "vacuous", "fail"}
                    for status in certificate_statuses),
            },
            "positive_recovery_evidence": (
                "observed" if pass_count > 0 else "not_observed_all_vacuous"
            ),
            "per_sequence_seed": per_sequence_seed,
            "interpretation": (
                "status=pass requires observed recovered support plus zero "
                "leaks; status=vacuous retains complete zero-leak rows with "
                "no observed recovery as safety-only evidence. safety_status "
                "tracks the independent complete zero-leak invariant.")
        }
        atomic_json(
            root / "dyn19_recovered_support_overlap_aggregate.json",
            full_certificate_report)

    report = {
        "experiment_id": DYN19_EXPERIMENT_ID,
        "contract": "dyn19-phase-report-v1",
        "phase": phase,
        "status": (
            "complete" if plan_audit.get("valid") and
            len(complete) == len(expected) and not duplicate_results and
            not unexpected else "incomplete"),
        "plan_audit": plan_audit,
        "expected_runs": len(expected),
        "completed_runs": len(complete),
        "missing_runs": [
            f"{config}/{sequence}/seed_{seed:04d}"
            for config, sequence, seed in missing
        ],
        "unexpected_results": [
            f"{config}/{sequence}/seed_{seed:04d}"
            for config, sequence, seed in unexpected
        ],
        "duplicate_results": [
            f"{config}/{sequence}/seed_{seed:04d}"
            for config, sequence, seed in duplicate_results
        ],
        "configs": by_config,
        "main_full_overlap_certificate": full_certificate_report,
    }
    atomic_json(root / "dyn19_phase_report.json", report)


def aggregate(root: Path, results: list[dict[str, Any]]) -> None:
    identities = [
        (result["config"], result["sequence"], result["seed"])
        for result in results
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(
            "Duplicate benchmark result identity would overwrite a gate input")
    complete = [result for result in results if result["status"] == "complete"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in complete:
        grouped.setdefault((result["config"], result["sequence"]), []).append(result)

    fields = [
        "config", "sequence", "successful_runs", "ate_mean_m", "ate_std_m",
        "ate_p95_mean_m", "ate_p95_std_m",
        "rpe_translation_mean_m", "rpe_translation_std_m", "rpe_rotation_mean_deg",
        "rpe_translation_p95_mean_m", "rpe_translation_p95_std_m",
        "rpe_rotation_std_deg", "rpe_rotation_p95_mean_deg",
        "rpe_rotation_p95_std_deg", "failure_rate_mean", "end_to_end_mean_seconds",
        "trajectory_coverage_mean", "carried_forward_rate_mean",
        "valid_motion_priors_mean", "raw_motion_priors_mean",
        "candidate_motion_priors_mean",
        "consensus_pass_motion_priors_mean", "direct_pass_motion_priors_mean",
        "common_support_pass_motion_priors_mean",
        "would_use_motion_priors_mean", "used_motion_priors_mean",
        "shadow_translation_applied_frames_mean",
        "shadow_translation_factor_injections_mean",
        "velocity_neutralized_frames_mean",
        "shadow_translation_propagation_rejections_mean",
        "mask_pose_predicted_frames_mean",
        "temporal_refinement_frames_mean",
        "temporal_recovered_static_pixels_mean",
        "temporal_added_dynamic_pixels_mean",
        "temporal_flow_guard_valid_frames_mean",
        "temporal_flow_guard_rejected_pixels_mean",
        "temporal_recovery_risk_active_frames_mean",
        "temporal_recovery_risk_blocked_frames_mean",
        "tracking_recovery_candidate_pixels_mean",
        "tracking_recovery_blocked_pixels_mean",
        "tracking_recovery_audit_pixels_mean",
        "tracking_recovery_mapping_leak_pixels_mean",
        "static_mask_ratio_mean",
        "mapping_weight_frames_mean",
        "mapping_static_ratio_mean",
        "mapping_keyframes_with_static_weight_mean",
        "mapping_weight_pixels_mean",
        "mapping_static_pixels_mean",
        "mapping_orb_map_points_considered_mean",
        "mapping_orb_map_points_rejected_static_mask_mean",
        "mapping_rgbd_densification_candidates_mean",
        "mapping_rgbd_densification_rejected_static_mask_mean",
        "adaptive_feature_active_frames_mean",
        "adaptive_fast_threshold_mean",
        "extracted_features_mean",
        "synchronize_local_mapping_all",
        "synchronize_loop_closing_all",
        "gaussian_mapper_disabled_all",
    ]
    with (root / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (config, sequence), runs in sorted(grouped.items()):
            ate = [run["metrics"]["ate_rmse_m"] for run in runs]
            ate_p95 = [run["metrics"]["ate_p95_m"] for run in runs]
            rpe_t = [run["metrics"]["rpe_translation_rmse_m"] for run in runs]
            rpe_t_p95 = [
                run["metrics"]["rpe_translation_p95_m"] for run in runs]
            rpe_r = [run["metrics"]["rpe_rotation_rmse_deg"] for run in runs]
            rpe_r_p95 = [
                run["metrics"]["rpe_rotation_p95_deg"] for run in runs]
            failure = [run["metrics"]["failure_rate"] for run in runs]
            timing = [run["metrics"]["end_to_end_seconds"] for run in runs]
            coverage = [run["metrics"]["trajectory_coverage"] for run in runs]
            carry_forward = [run["metrics"]["carried_forward_rate"] for run in runs]
            valid_priors = [run["metrics"]["valid_motion_priors"] for run in runs]
            raw_priors = [
                run["metrics"].get(
                    "raw_motion_priors",
                    run["metrics"]["valid_motion_priors"])
                for run in runs]
            candidate_priors = [run["metrics"]["candidate_motion_priors"] for run in runs]
            consensus_pass_priors = [
                run["metrics"].get("consensus_pass_motion_priors", 0)
                for run in runs]
            direct_pass_priors = [
                run["metrics"].get("direct_pass_motion_priors", 0)
                for run in runs]
            common_support_pass_priors = [
                run["metrics"].get(
                    "common_support_pass_motion_priors", 0)
                for run in runs]
            would_use_priors = [
                run["metrics"].get(
                    "would_use_motion_priors",
                    run["metrics"]["used_motion_priors"])
                for run in runs]
            used_priors = [run["metrics"]["used_motion_priors"] for run in runs]
            shadow_translation_applied = [
                run["metrics"].get("shadow_translation_applied_frames", 0)
                for run in runs]
            shadow_translation_factor_injections = [
                run["metrics"].get(
                    "shadow_translation_factor_injections", 0)
                for run in runs]
            velocity_neutralized = [
                run["metrics"].get("velocity_neutralized_frames", 0)
                for run in runs]
            shadow_translation_propagation_rejections = [
                run["metrics"].get(
                    "shadow_translation_propagation_rejections", 0)
                for run in runs]
            mask_pose_predicted = [
                run["metrics"].get("mask_pose_predicted_frames", 0)
                for run in runs]
            temporal_refinement = [
                run["metrics"].get("temporal_refinement_frames", 0)
                for run in runs]
            recovered_static_pixels = [
                run["metrics"].get("temporal_recovered_static_pixels", 0)
                for run in runs]
            added_dynamic_pixels = [
                run["metrics"].get("temporal_added_dynamic_pixels", 0)
                for run in runs]
            flow_guard_valid_frames = [
                run["metrics"].get(
                    "temporal_flow_guard_valid_frames", 0)
                for run in runs]
            flow_guard_rejected_pixels = [
                run["metrics"].get(
                    "temporal_flow_guard_rejected_pixels", 0)
                for run in runs]
            recovery_risk_active_frames = [
                run["metrics"].get(
                    "temporal_recovery_risk_active_frames", 0)
                for run in runs]
            recovery_risk_blocked_frames = [
                run["metrics"].get(
                    "temporal_recovery_risk_blocked_frames", 0)
                for run in runs]
            recovery_candidate_pixels = [
                run["metrics"].get(
                    "tracking_recovery_candidate_pixels", 0)
                for run in runs]
            recovery_blocked_pixels = [
                run["metrics"].get(
                    "tracking_recovery_blocked_pixels", 0)
                for run in runs]
            recovery_audit_pixels = [
                run["metrics"].get(
                    "tracking_recovery_audit_pixels", 0)
                for run in runs]
            recovery_mapping_leak_pixels = [
                run["metrics"].get(
                    "tracking_recovery_mapping_leak_pixels", 0)
                for run in runs]
            static_mask_ratios = [
                run["metrics"].get("static_mask_ratio_mean", 1.0)
                for run in runs]
            mapping_weight_frames = [
                run["metrics"].get("mapping_weight_frames", 0)
                for run in runs]
            mapping_static_ratios = [
                run["metrics"].get("mapping_static_ratio_mean", 1.0)
                for run in runs]
            mapping_keyframes = [
                run["metrics"].get(
                    "mapping_keyframes_with_static_weight", 0)
                for run in runs]
            mapping_weight_pixels = [
                run["metrics"].get("mapping_weight_pixels", 0)
                for run in runs]
            mapping_static_pixels = [
                run["metrics"].get("mapping_static_pixels", 0)
                for run in runs]
            mapping_orb_considered = [
                run["metrics"].get(
                    "mapping_orb_map_points_considered", 0)
                for run in runs]
            mapping_orb_rejected = [
                run["metrics"].get(
                    "mapping_orb_map_points_rejected_static_mask", 0)
                for run in runs]
            mapping_rgbd_candidates = [
                run["metrics"].get(
                    "mapping_rgbd_densification_candidates", 0)
                for run in runs]
            mapping_rgbd_rejected = [
                run["metrics"].get(
                    "mapping_rgbd_densification_rejected_static_mask",
                    0)
                for run in runs]
            adaptive_feature_active_frames = [
                run["metrics"].get("adaptive_feature_active_frames", 0)
                for run in runs]
            adaptive_fast_thresholds = [
                run["metrics"].get("adaptive_fast_threshold_mean", 0.0)
                for run in runs]
            extracted_features = [
                run["metrics"].get("extracted_features_mean", 0.0)
                for run in runs]
            synchronized_local_mapping = [
                run["metrics"].get("synchronize_local_mapping", False)
                for run in runs]
            synchronized_loop_closing = [
                run["metrics"].get("synchronize_loop_closing", False)
                for run in runs]
            gaussian_mapper_disabled = [
                not run["metrics"].get("gaussian_mapper_enabled", True)
                for run in runs]
            ate_mean, ate_std = mean_std(ate)
            ate_p95_mean, ate_p95_std = mean_std(ate_p95)
            rpe_t_mean, rpe_t_std = mean_std(rpe_t)
            rpe_t_p95_mean, rpe_t_p95_std = mean_std(rpe_t_p95)
            rpe_r_mean, rpe_r_std = mean_std(rpe_r)
            rpe_r_p95_mean, rpe_r_p95_std = mean_std(rpe_r_p95)
            writer.writerow({
                "config": config, "sequence": sequence, "successful_runs": len(runs),
                "ate_mean_m": ate_mean, "ate_std_m": ate_std,
                "ate_p95_mean_m": ate_p95_mean,
                "ate_p95_std_m": ate_p95_std,
                "rpe_translation_mean_m": rpe_t_mean,
                "rpe_translation_std_m": rpe_t_std,
                "rpe_translation_p95_mean_m": rpe_t_p95_mean,
                "rpe_translation_p95_std_m": rpe_t_p95_std,
                "rpe_rotation_mean_deg": rpe_r_mean,
                "rpe_rotation_std_deg": rpe_r_std,
                "rpe_rotation_p95_mean_deg": rpe_r_p95_mean,
                "rpe_rotation_p95_std_deg": rpe_r_p95_std,
                "failure_rate_mean": statistics.mean(failure),
                "end_to_end_mean_seconds": statistics.mean(timing),
                "trajectory_coverage_mean": statistics.mean(coverage),
                "carried_forward_rate_mean": statistics.mean(carry_forward),
                "valid_motion_priors_mean": statistics.mean(valid_priors),
                "raw_motion_priors_mean": statistics.mean(raw_priors),
                "candidate_motion_priors_mean": statistics.mean(candidate_priors),
                "consensus_pass_motion_priors_mean":
                    statistics.mean(consensus_pass_priors),
                "direct_pass_motion_priors_mean":
                    statistics.mean(direct_pass_priors),
                "common_support_pass_motion_priors_mean":
                    statistics.mean(common_support_pass_priors),
                "would_use_motion_priors_mean": statistics.mean(would_use_priors),
                "used_motion_priors_mean": statistics.mean(used_priors),
                "shadow_translation_applied_frames_mean":
                    statistics.mean(shadow_translation_applied),
                "shadow_translation_factor_injections_mean":
                    statistics.mean(shadow_translation_factor_injections),
                "velocity_neutralized_frames_mean":
                    statistics.mean(velocity_neutralized),
                "shadow_translation_propagation_rejections_mean":
                    statistics.mean(
                        shadow_translation_propagation_rejections),
                "mask_pose_predicted_frames_mean":
                    statistics.mean(mask_pose_predicted),
                "temporal_refinement_frames_mean":
                    statistics.mean(temporal_refinement),
                "temporal_recovered_static_pixels_mean":
                    statistics.mean(recovered_static_pixels),
                "temporal_added_dynamic_pixels_mean":
                    statistics.mean(added_dynamic_pixels),
                "temporal_flow_guard_valid_frames_mean":
                    statistics.mean(flow_guard_valid_frames),
                "temporal_flow_guard_rejected_pixels_mean":
                    statistics.mean(flow_guard_rejected_pixels),
                "temporal_recovery_risk_active_frames_mean":
                    statistics.mean(recovery_risk_active_frames),
                "temporal_recovery_risk_blocked_frames_mean":
                    statistics.mean(recovery_risk_blocked_frames),
                "tracking_recovery_candidate_pixels_mean":
                    statistics.mean(recovery_candidate_pixels),
                "tracking_recovery_blocked_pixels_mean":
                    statistics.mean(recovery_blocked_pixels),
                "tracking_recovery_audit_pixels_mean":
                    statistics.mean(recovery_audit_pixels),
                "tracking_recovery_mapping_leak_pixels_mean":
                    statistics.mean(recovery_mapping_leak_pixels),
                "static_mask_ratio_mean":
                    statistics.mean(static_mask_ratios),
                "mapping_weight_frames_mean":
                    statistics.mean(mapping_weight_frames),
                "mapping_static_ratio_mean":
                    statistics.mean(mapping_static_ratios),
                "mapping_keyframes_with_static_weight_mean":
                    statistics.mean(mapping_keyframes),
                "mapping_weight_pixels_mean":
                    statistics.mean(mapping_weight_pixels),
                "mapping_static_pixels_mean":
                    statistics.mean(mapping_static_pixels),
                "mapping_orb_map_points_considered_mean":
                    statistics.mean(mapping_orb_considered),
                "mapping_orb_map_points_rejected_static_mask_mean":
                    statistics.mean(mapping_orb_rejected),
                "mapping_rgbd_densification_candidates_mean":
                    statistics.mean(mapping_rgbd_candidates),
                "mapping_rgbd_densification_rejected_static_mask_mean":
                    statistics.mean(mapping_rgbd_rejected),
                "adaptive_feature_active_frames_mean":
                    statistics.mean(adaptive_feature_active_frames),
                "adaptive_fast_threshold_mean":
                    statistics.mean(adaptive_fast_thresholds),
                "extracted_features_mean":
                    statistics.mean(extracted_features),
                "synchronize_local_mapping_all":
                    all(synchronized_local_mapping),
                "synchronize_loop_closing_all":
                    all(synchronized_loop_closing),
                "gaussian_mapper_disabled_all":
                    all(gaussian_mapper_disabled),
            })

    by_key = {(r["config"], r["sequence"], r["seed"]): r for r in complete}
    planned_by_pair: dict[tuple[str, str], set[int]] = {}
    for result in results:
        planned_by_pair.setdefault(
            (result["config"], result["sequence"]), set()).add(result["seed"])
    sequences = sorted({r["sequence"] for r in results})
    paired = []
    full_failure_pairs = []
    sequence_checks = {}
    complete_pairing = False
    for sequence in sequences:
        planned_semantic = planned_by_pair.get(("semantic", sequence), set())
        planned_full = planned_by_pair.get(("full", sequence), set())
        planned_seeds = sorted(planned_semantic & planned_full)
        if not planned_semantic or not planned_full:
            continue
        complete_pairing = True if not sequence_checks else complete_pairing
        semantic = [r for r in complete if r["config"] == "semantic" and r["sequence"] == sequence]
        full = [r for r in complete if r["config"] == "full" and r["sequence"] == sequence]
        common_seeds = sorted({r["seed"] for r in semantic} & {r["seed"] for r in full})
        missing_runs = [
            f"{config}/seed_{seed}"
            for config in ("semantic", "full")
            for seed in planned_seeds
            if (config, sequence, seed) not in by_key
        ]
        sequence_checks[sequence] = {
            "status": "incomplete",
            "planned_paired_seeds": planned_seeds,
            "paired_seeds": common_seeds,
            "missing_runs": missing_runs,
        }
        sequence_complete = (
            len(planned_seeds) >= 3 and
            common_seeds == planned_seeds and
            not missing_runs)
        complete_pairing &= sequence_complete
        if not common_seeds:
            continue
        semantic_ate = [by_key[("semantic", sequence, seed)]["metrics"]["ate_rmse_m"]
                        for seed in common_seeds]
        full_ate = [by_key[("full", sequence, seed)]["metrics"]["ate_rmse_m"]
                    for seed in common_seeds]
        full_wins = sum(f < s for f, s in zip(full_ate, semantic_ate))
        sequence_checks[sequence].update({
            "status": "complete" if sequence_complete else "incomplete",
            "semantic_ate_mean_m": statistics.mean(semantic_ate),
            "full_ate_mean_m": statistics.mean(full_ate),
            "full_mean_better": statistics.mean(full_ate) < statistics.mean(semantic_ate),
            "full_paired_wins": full_wins,
        })
        paired.extend(zip(full_ate, semantic_ate))
        full_failure_pairs.extend(
            (
                by_key[("full", sequence, seed)]["metrics"]["failure_rate"],
                by_key[("semantic", sequence, seed)]["metrics"]["failure_rate"],
            )
            for seed in common_seeds)

    if sequence_checks and paired and complete_pairing:
        paired_win_rate = sum(full < semantic for full, semantic in paired) / len(paired)
        all_sequences_better = all(item["full_mean_better"] for item in sequence_checks.values())
        failure_ok = (
            statistics.mean(pair[0] for pair in full_failure_pairs) <=
            statistics.mean(pair[1] for pair in full_failure_pairs) + 0.005)
        retained = all_sequences_better and paired_win_rate >= 0.7 and failure_ok
        decision = {
            "status": "pass" if retained else "fail",
            "geometry_enhancement_claim_retained": retained,
            "predeclared_rule": (
                "Full mean ATE must beat Semantic on every evaluated sequence, win at least "
                "70% of paired seeds, and increase mean failure rate by no more than 0.5 points."
            ),
            "paired_win_rate": paired_win_rate,
            "failure_rate_condition": failure_ok,
            "sequences": sequence_checks,
        }
    else:
        decision = {
            "status": "insufficient_data",
            "geometry_enhancement_claim_retained": False,
            "reason": (
                "Every planned Semantic/Full sequence requires at least three "
                "successfully completed paired seeds."),
            "sequences": sequence_checks,
        }
    atomic_json(root / "full_fusion_claim_gate.json", decision)

    motion_sequence_checks = {}
    motion_pairs = []
    motion_failure_pairs = []
    complete_motion_pairing = False
    for sequence in sequences:
        planned_semantic = planned_by_pair.get(("semantic", sequence), set())
        planned_motion = planned_by_pair.get(("semantic_motion", sequence), set())
        planned_seeds = sorted(planned_semantic & planned_motion)
        if not planned_semantic or not planned_motion:
            continue
        complete_motion_pairing = (
            True if not motion_sequence_checks else complete_motion_pairing)
        semantic = [r for r in complete
                    if r["config"] == "semantic" and r["sequence"] == sequence]
        motion = [r for r in complete
                  if r["config"] == "semantic_motion" and r["sequence"] == sequence]
        common_seeds = sorted({r["seed"] for r in semantic} & {r["seed"] for r in motion})
        missing_runs = [
            f"{config}/seed_{seed}"
            for config in ("semantic", "semantic_motion")
            for seed in planned_seeds
            if (config, sequence, seed) not in by_key
        ]
        motion_sequence_checks[sequence] = {
            "status": "incomplete",
            "planned_paired_seeds": planned_seeds,
            "paired_seeds": common_seeds,
            "missing_runs": missing_runs,
        }
        sequence_complete = (
            len(planned_seeds) >= 3 and
            common_seeds == planned_seeds and
            not missing_runs)
        complete_motion_pairing &= sequence_complete
        if not common_seeds:
            continue
        semantic_runs = [by_key[("semantic", sequence, seed)] for seed in common_seeds]
        motion_runs = [by_key[("semantic_motion", sequence, seed)] for seed in common_seeds]
        semantic_ate = [run["metrics"]["ate_rmse_m"] for run in semantic_runs]
        motion_ate = [run["metrics"]["ate_rmse_m"] for run in motion_runs]
        used_seed_count = sum(
            run["metrics"].get("used_motion_priors", 0) > 0 for run in motion_runs)
        used_seed_rate = used_seed_count / len(common_seeds)
        motion_sequence_checks[sequence].update({
            "status": "complete" if sequence_complete else "incomplete",
            "semantic_ate_mean_m": statistics.mean(semantic_ate),
            "motion_ate_mean_m": statistics.mean(motion_ate),
            "motion_mean_not_worse": statistics.mean(motion_ate) <= statistics.mean(semantic_ate),
            "motion_paired_wins": sum(m < s for m, s in zip(motion_ate, semantic_ate)),
            "motion_used_seed_count": used_seed_count,
            "motion_used_seed_rate": used_seed_rate,
            "motion_used_in_at_least_70pct_seeds": used_seed_rate >= 0.7,
            "semantic_rpe_translation_mean_m": statistics.mean(
                run["metrics"]["rpe_translation_rmse_m"] for run in semantic_runs),
            "motion_rpe_translation_mean_m": statistics.mean(
                run["metrics"]["rpe_translation_rmse_m"] for run in motion_runs),
            "semantic_rpe_rotation_mean_deg": statistics.mean(
                run["metrics"]["rpe_rotation_rmse_deg"] for run in semantic_runs),
            "motion_rpe_rotation_mean_deg": statistics.mean(
                run["metrics"]["rpe_rotation_rmse_deg"] for run in motion_runs),
            "semantic_end_to_end_mean_seconds": statistics.mean(
                run["metrics"]["end_to_end_seconds"] for run in semantic_runs),
            "motion_end_to_end_mean_seconds": statistics.mean(
                run["metrics"]["end_to_end_seconds"] for run in motion_runs),
        })
        motion_pairs.extend(zip(motion_ate, semantic_ate))
        motion_failure_pairs.extend(
            (motion_run["metrics"]["failure_rate"], semantic_run["metrics"]["failure_rate"])
            for motion_run, semantic_run in zip(motion_runs, semantic_runs))

    if motion_sequence_checks and motion_pairs and complete_motion_pairing:
        motion_win_rate = sum(motion < semantic for motion, semantic in motion_pairs) / len(motion_pairs)
        motion_failure_mean = statistics.mean(pair[0] for pair in motion_failure_pairs)
        semantic_failure_mean = statistics.mean(pair[1] for pair in motion_failure_pairs)
        mean_condition = all(
            item["motion_mean_not_worse"] for item in motion_sequence_checks.values())
        trigger_condition = all(
            item["motion_used_in_at_least_70pct_seeds"]
            for item in motion_sequence_checks.values())
        failure_condition = motion_failure_mean <= semantic_failure_mean + 0.005
        retained = (mean_condition and motion_win_rate >= 0.7 and
                    failure_condition and trigger_condition)
        motion_decision = {
            "status": "pass" if retained else "fail",
            "motion_prior_claim_retained": retained,
            "predeclared_rule": (
                "Semantic+Motion mean ATE must not be worse than Semantic on every "
                "evaluated sequence, win at least 70% of paired seeds, increase mean "
                "failure rate by no more than 0.5 points, and be selected in at least "
                "70% of planned seeds for every sequence. RPE and end-to-end time are "
                "reported but are not binary gate conditions."
            ),
            "paired_win_rate": motion_win_rate,
            "mean_ate_condition": mean_condition,
            "failure_rate_condition": failure_condition,
            "trigger_condition": trigger_condition,
            "semantic_failure_rate_mean": semantic_failure_mean,
            "motion_failure_rate_mean": motion_failure_mean,
            "sequences": motion_sequence_checks,
        }
    else:
        motion_decision = {
            "status": "insufficient_data",
            "motion_prior_claim_retained": False,
            "reason": (
                "Every planned Semantic/Semantic+Motion sequence requires at "
                "least three successfully completed paired seeds."),
            "sequences": motion_sequence_checks,
        }
    atomic_json(root / "motion_prior_claim_gate.json", motion_decision)

    consensus_config = "semantic_motion_consensus_shadow"
    consensus_summary_fields = {
        "pairs",
        "static_translation_sum_squared_error_m2",
        "dynamic_translation_sum_squared_error_m2",
        "dynamic_translation_wins",
        "worst_translation_degradation_m",
    }
    consensus_checks = {}
    complete_consensus_evidence = False
    for sequence in MOTION_CONSENSUS_HELDOUT_SEQUENCES:
        planned_seeds = sorted(
            planned_by_pair.get((consensus_config, sequence), set()))
        if not planned_seeds:
            consensus_checks[sequence] = {
                "status": "not_planned",
                "planned_seeds": [],
                "missing_runs": [],
                "missing_evidence": [],
                "selected_pairs": 0,
                "mechanism_pass": False,
            }
            complete_consensus_evidence = False
            continue
        complete_consensus_evidence = (
            True if not consensus_checks else complete_consensus_evidence)
        missing_runs = [
            f"{consensus_config}/seed_{seed}"
            for seed in planned_seeds
            if (consensus_config, sequence, seed) not in by_key
        ]
        summaries = []
        missing_evidence = []
        for seed in planned_seeds:
            run = by_key.get((consensus_config, sequence, seed))
            if run is None:
                continue
            summary = (
                run.get("metrics", {})
                .get("motion_prior_counterfactual", {})
                .get("gate_would_use"))
            if (
                not isinstance(summary, dict) or
                not consensus_summary_fields.issubset(summary)
            ):
                missing_evidence.append(
                    f"{consensus_config}/seed_{seed}/counterfactual")
            else:
                summaries.append(summary)
        sequence_complete = (
            len(planned_seeds) >= 3 and
            not missing_runs and
            not missing_evidence and
            len(summaries) == len(planned_seeds))
        complete_consensus_evidence &= sequence_complete
        pairs = sum(item["pairs"] for item in summaries)
        static_sse = sum(
            item["static_translation_sum_squared_error_m2"]
            for item in summaries)
        dynamic_sse = sum(
            item["dynamic_translation_sum_squared_error_m2"]
            for item in summaries)
        wins = sum(item["dynamic_translation_wins"] for item in summaries)
        static_rmse = math.sqrt(static_sse / pairs) if pairs else None
        dynamic_rmse = math.sqrt(dynamic_sse / pairs) if pairs else None
        win_rate = wins / pairs if pairs else None
        per_run_worst = [
            item["worst_translation_degradation_m"]
            for item in summaries
            if item["worst_translation_degradation_m"] is not None
        ]
        worst_degradation = max(per_run_worst) if per_run_worst else None
        mechanism_pass = bool(
            pairs >= 5 and
            win_rate is not None and win_rate >= 0.60 and
            dynamic_rmse is not None and static_rmse is not None and
            dynamic_rmse <= static_rmse and
            worst_degradation is not None and worst_degradation <= 0.01)
        consensus_checks[sequence] = {
            "status": "complete" if sequence_complete else "incomplete",
            "planned_seeds": planned_seeds,
            "missing_runs": missing_runs,
            "missing_evidence": missing_evidence,
            "selected_pairs": pairs,
            "dynamic_translation_wins": wins,
            "dynamic_translation_win_rate": win_rate,
            "static_translation_rmse_m": static_rmse,
            "dynamic_translation_rmse_m": dynamic_rmse,
            "worst_translation_degradation_m": worst_degradation,
            "mechanism_pass": mechanism_pass,
        }

    consensus_retained = bool(
        consensus_checks and complete_consensus_evidence and
        all(item["mechanism_pass"] for item in consensus_checks.values()))
    consensus_decision = {
        "status": (
            "pass" if consensus_retained else
            "fail" if consensus_checks and complete_consensus_evidence else
            "insufficient_data"),
        "motion_consensus_gate_retained": consensus_retained,
        "predeclared_rule": (
            "On both required held-out sequences, three planned shadow seeds must "
            "complete, the frozen gate must select at least five same-state "
            "counterfactual pairs, dynamic translation must win at least 60% "
            "of pairs, pooled dynamic translation RMSE must not exceed pooled "
            "static translation RMSE, and the worst per-pair translation "
            "degradation must not exceed 0.01 m."
        ),
        "sequences": consensus_checks,
    }
    atomic_json(
        root / "motion_consensus_counterfactual_gate.json",
        consensus_decision)

    intervention_config = "semantic_motion_consensus"
    end_to_end_configs = (
        "semantic", consensus_config, intervention_config)
    end_to_end_checks = {}
    end_to_end_pairs = []
    end_to_end_failures = {config: [] for config in end_to_end_configs}
    complete_end_to_end_evidence = False
    for sequence in MOTION_CONSENSUS_HELDOUT_SEQUENCES:
        planned_sets = {
            config: planned_by_pair.get((config, sequence), set())
            for config in end_to_end_configs
        }
        if not all(planned_sets.values()):
            end_to_end_checks[sequence] = {
                "status": "not_planned",
                "planned_seeds": sorted(set.union(*planned_sets.values())),
                "paired_seeds": [],
                "same_seed_plan": False,
                "missing_runs": [],
            }
            complete_end_to_end_evidence = False
            continue
        complete_end_to_end_evidence = (
            True if not end_to_end_checks else complete_end_to_end_evidence)
        planned_seeds = sorted(set.union(*planned_sets.values()))
        missing_runs = [
            f"{config}/seed_{seed}"
            for config in end_to_end_configs
            for seed in planned_seeds
            if (config, sequence, seed) not in by_key
        ]
        same_seed_plan = all(
            seeds == planned_sets["semantic"]
            for seeds in planned_sets.values())
        sequence_complete = (
            len(planned_seeds) >= 3 and same_seed_plan and not missing_runs)
        complete_end_to_end_evidence &= sequence_complete
        common_seeds = [
            seed for seed in planned_seeds
            if all(
                (config, sequence, seed) in by_key
                for config in end_to_end_configs)
        ]
        runs = {
            config: [
                by_key[(config, sequence, seed)]
                for seed in common_seeds]
            for config in end_to_end_configs
        }
        ate = {
            config: [run["metrics"]["ate_rmse_m"] for run in config_runs]
            for config, config_runs in runs.items()
        }
        consensus_mean_better_than_semantic = bool(
            common_seeds and
            statistics.mean(ate[intervention_config]) <=
            statistics.mean(ate["semantic"]))
        consensus_mean_better_than_shadow = bool(
            common_seeds and
            statistics.mean(ate[intervention_config]) <=
            statistics.mean(ate[consensus_config]))
        selected_seed_count = sum(
            run["metrics"].get("used_motion_priors", 0) > 0
            for run in runs[intervention_config])
        selected_seed_rate = (
            selected_seed_count / len(common_seeds)
            if common_seeds else None)
        end_to_end_checks[sequence] = {
            "status": "complete" if sequence_complete else "incomplete",
            "planned_seeds": planned_seeds,
            "paired_seeds": common_seeds,
            "same_seed_plan": same_seed_plan,
            "missing_runs": missing_runs,
            "semantic_ate_mean_m": (
                statistics.mean(ate["semantic"]) if common_seeds else None),
            "consensus_shadow_ate_mean_m": (
                statistics.mean(ate[consensus_config])
                if common_seeds else None),
            "consensus_ate_mean_m": (
                statistics.mean(ate[intervention_config])
                if common_seeds else None),
            "consensus_mean_not_worse_than_semantic":
                consensus_mean_better_than_semantic,
            "consensus_mean_not_worse_than_shadow":
                consensus_mean_better_than_shadow,
            "consensus_paired_wins_over_shadow": sum(
                consensus < shadow
                for consensus, shadow in zip(
                    ate[intervention_config], ate[consensus_config])),
            "consensus_selected_seed_count": selected_seed_count,
            "consensus_selected_seed_rate": selected_seed_rate,
            "consensus_selected_in_at_least_two_thirds_seeds": (
                selected_seed_rate is not None and
                selected_seed_rate >= (2.0 / 3.0)),
        }
        end_to_end_pairs.extend(zip(
            ate[intervention_config], ate[consensus_config]))
        for config in end_to_end_configs:
            end_to_end_failures[config].extend(
                run["metrics"]["failure_rate"] for run in runs[config])

    end_to_end_win_rate = (
        sum(consensus < shadow for consensus, shadow in end_to_end_pairs) /
        len(end_to_end_pairs)
        if end_to_end_pairs else None)
    failure_condition = bool(
        end_to_end_pairs and
        statistics.mean(end_to_end_failures[intervention_config]) <=
        statistics.mean(end_to_end_failures["semantic"]) + 0.005 and
        statistics.mean(end_to_end_failures[intervention_config]) <=
        statistics.mean(end_to_end_failures[consensus_config]) + 0.005)
    end_to_end_retained = bool(
        end_to_end_checks and complete_end_to_end_evidence and
        all(
            item["consensus_mean_not_worse_than_semantic"] and
            item["consensus_mean_not_worse_than_shadow"] and
            item["consensus_selected_in_at_least_two_thirds_seeds"]
            for item in end_to_end_checks.values()) and
        end_to_end_win_rate is not None and end_to_end_win_rate >= 0.70 and
        failure_condition)
    end_to_end_decision = {
        "status": (
            "pass" if end_to_end_retained else
            "fail" if end_to_end_checks and complete_end_to_end_evidence else
            "insufficient_data"),
        "motion_consensus_intervention_retained": end_to_end_retained,
        "predeclared_rule": (
            "All three configurations must complete the same three or more "
            "seeds on every sequence. Consensus mean ATE must not exceed "
            "Semantic or matched Consensus Shadow on any sequence, Consensus "
            "must beat Shadow on at least 70% of all paired seeds, its mean "
            "failure rate must not exceed either control by more than 0.5 "
            "percentage points, and it must be selected in at least two thirds "
            "of seeds on every sequence. RPE and end-to-end time are reported "
            "but are not binary gate conditions."
        ),
        "paired_win_rate_over_shadow": end_to_end_win_rate,
        "failure_rate_condition": failure_condition,
        "failure_rate_means": {
            config: (
                statistics.mean(values) if values else None)
            for config, values in end_to_end_failures.items()
        },
        "sequences": end_to_end_checks,
    }
    atomic_json(
        root / "motion_consensus_end_to_end_gate.json",
        end_to_end_decision)
    write_tempered_counterfactual_gate(root, by_key, planned_by_pair)
    write_tempered_end_to_end_gate(root, by_key, planned_by_pair)
    write_direct_counterfactual_gate(root, by_key, planned_by_pair)
    write_direct_end_to_end_gate(root, by_key, planned_by_pair)
    write_rgbd_counterfactual_gate(root, by_key, planned_by_pair)
    write_rgbd_end_to_end_gate(root, by_key, planned_by_pair)
    write_shadow_translation_tracking_gate(
        root, by_key, planned_by_pair,
        SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES,
        "shadow_translation_development_tracking_gate.json",
        "development",
    )
    write_shadow_translation_tracking_gate(
        root, by_key, planned_by_pair,
        SHADOW_TRANSLATION_HELDOUT_SEQUENCES,
        "shadow_translation_heldout_tracking_gate.json",
        "heldout",
    )
    write_motion_ablation_summary(root, results)
    write_flow_adaptive_incumbent_gate(root, results)
    write_dyn19_phase_reports(root, results)


def validate_inputs(configs: list[str], sequences: list[str]) -> None:
    required = [
        PROJECT / "bin/tum_rgbd_dynamic",
        PROJECT / "ORB-SLAM3/Vocabulary/ORBvoc.txt",
        *RUNTIME_LIBRARIES,
    ]
    required += [CONFIGS[name] for name in configs]
    for sequence in sequences:
        directory, orb, gaussian, association = SEQUENCES[sequence]
        required += [directory, orb, gaussian, directory / association, directory / "groundtruth.txt"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark inputs:\n" + "\n".join(missing))
    empty = [str(path) for path in required if path.is_file() and path.stat().st_size == 0]
    if empty:
        raise ValueError("Empty benchmark inputs:\n" + "\n".join(empty))
    for config in configs:
        yolo_engine_contract(CONFIGS[config])
    if {
        FLOW_ADAPTIVE_PARENT_CONFIG,
        FLOW_ADAPTIVE_CANDIDATE_CONFIG,
    } & set(configs):
        contract = flow_adaptive_config_contract()
        if not contract["valid"]:
            raise ValueError(
                "Flow-Adaptive/Exact config pair violates its frozen "
                f"contract: {contract}")
    if set(configs) & set(DYN19_TRACKING_CONFIGS):
        contract = dyn19_config_contract()
        if not contract["valid"]:
            raise ValueError(
                "DYN-19 factor matrix violates its frozen config contract: "
                f"{contract}")


def build_task_blocks(
    configs: list[str],
    sequences: list[str],
    seeds: list[int],
    gpus: list[str],
    heldout_stride: int,
    source: dict[str, Any],
    root: Path,
    synchronize_local_mapping: bool = False,
    synchronize_loop_closing: bool = False,
    disable_gaussian_mapper: bool = False,
    export_static_masks: bool = False,
    dyn19_phase: str | None = None,
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    tasks = []
    blocks = []
    for sequence_index, sequence in enumerate(sequences):
        for seed_index, seed in enumerate(seeds):
            pair_index = sequence_index * len(seeds) + seed_index
            config_order = list(configs)
            if pair_index % 2 == 1:
                config_order.reverse()
            pair_id = f"{sequence}/seed_{seed:04d}"
            gpu = gpus[pair_index % len(gpus)]
            block = []
            for config in config_order:
                task = {
                    "config": config,
                    "sequence": sequence,
                    "seed": seed,
                    "pair_id": pair_id,
                    "gpu": gpu,
                    "heldout_stride": heldout_stride,
                    "synchronize_local_mapping":
                        synchronize_local_mapping,
                    "synchronize_loop_closing":
                        synchronize_loop_closing,
                    "disable_gaussian_mapper":
                        disable_gaussian_mapper,
                    "export_static_masks": (
                        export_static_masks and
                        (dyn19_phase != "main" or
                         config == "dyn19_semantic")),
                    "source": source,
                    "run_dir": str(
                        root / config / sequence / f"seed_{seed:04d}"),
                }
                if dyn19_phase is not None:
                    task["dyn19"] = {
                        "experiment_id": DYN19_EXPERIMENT_ID,
                        "phase": dyn19_phase,
                        "factors": DYN19_FACTOR_MATRIX[config],
                    }
                tasks.append(task)
                block.append(task)
            blocks.append(block)
    return tasks, blocks


def main() -> int:
    global GPU_LOCKS
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIGS),
                        default=None)
    parser.add_argument("--sequences", nargs="+", choices=sorted(SEQUENCES),
                        default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--heldout-stride", type=int, default=None)
    parser.add_argument(
        "--dyn19-phase",
        choices=sorted(DYN19_PHASE_CONFIGS),
        help=(
            "freeze the DYN-19 main or incremental mechanism phase; this "
            "sets the exact 10-sequence, three-seed denominator"),
    )
    parser.add_argument(
        "--export-static-masks",
        action="store_true",
        help="export every final tracking static mask as a frozen PNG snapshot",
    )
    parser.add_argument(
        "--sync-local-mapping", action="store_true",
        help="wait for LocalMapping to become idle after every input frame")
    parser.add_argument(
        "--sync-loop-closing", action="store_true",
        help="wait for LoopClosing to become idle after every input frame")
    parser.add_argument(
        "--disable-gaussian-mapper", action="store_true",
        help=(
            "keep ORB tracking/mapping active but disable the independent "
            "Gaussian mapping thread"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="allow diagnostic runs from a dirty source tree")
    args = parser.parse_args()
    default_configs = ["semantic", "full"]
    default_sequences = [
        "tum_walking_xyz",
        "tum_walking_halfsphere",
        "tum_walking_static",
        "tum_sitting_halfsphere",
        "bonn_crowd",
    ]
    default_seeds = [0, 1, 2, 3, 4]
    if args.dyn19_phase is not None:
        phase_configs = list(DYN19_PHASE_CONFIGS[args.dyn19_phase])
        if args.configs is not None and args.configs != phase_configs:
            parser.error(
                f"--dyn19-phase {args.dyn19_phase} requires configs "
                f"{phase_configs}")
        if (args.sequences is not None and
                args.sequences != list(DYN19_CLAIM_SEQUENCES)):
            parser.error(
                f"--dyn19-phase {args.dyn19_phase} requires the frozen "
                "DYN-19 claim-bearing sequence union")
        if args.seeds is not None and args.seeds != list(DYN19_REQUIRED_SEEDS):
            parser.error(
                f"--dyn19-phase {args.dyn19_phase} requires seeds "
                f"{list(DYN19_REQUIRED_SEEDS)}")
        if args.heldout_stride not in (None, 0):
            parser.error("--dyn19-phase uses heldout_stride=0 for tracking runs")
        args.configs = phase_configs
        args.sequences = list(DYN19_CLAIM_SEQUENCES)
        args.seeds = list(DYN19_REQUIRED_SEEDS)
        args.heldout_stride = 0
        if args.dyn19_phase == "main":
            args.export_static_masks = True
    else:
        args.configs = args.configs or default_configs
        args.sequences = args.sequences or default_sequences
        args.seeds = args.seeds or default_seeds
        args.heldout_stride = (
            20 if args.heldout_stride is None else args.heldout_stride)
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be non-negative")
    for name, values in (
        ("configs", args.configs),
        ("sequences", args.sequences),
        ("seeds", args.seeds),
    ):
        if len(values) != len(set(values)):
            parser.error(f"{name} must not contain duplicates")
    if args.jobs < 1 or args.jobs > len(args.gpus):
        parser.error("jobs must be between 1 and the number of GPU entries")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU entries must be unique")
    if args.sync_loop_closing and not args.sync_local_mapping:
        parser.error("--sync-loop-closing requires --sync-local-mapping")
    validate_inputs(args.configs, args.sequences)
    GPU_LOCKS = {str(gpu): threading.Lock() for gpu in args.gpus}

    state = source_state()
    if state["dirty"] and not args.allow_dirty:
        raise RuntimeError(
            "Refusing a benchmark from a dirty source tree; commit the "
            "runtime changes or pass --allow-dirty for a diagnostic run")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.output_root or (EXPERIMENTS / timestamp)
    if root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {root}")
    root.mkdir(parents=True)
    tasks, blocks = build_task_blocks(
        args.configs, args.sequences, args.seeds, args.gpus,
        args.heldout_stride, state, root,
        synchronize_local_mapping=args.sync_local_mapping,
        synchronize_loop_closing=args.sync_loop_closing,
        disable_gaussian_mapper=args.disable_gaussian_mapper,
        export_static_masks=args.export_static_masks,
        dyn19_phase=args.dyn19_phase)
    asset_contracts = {
        (config, sequence): benchmark_task_asset_contract(config, sequence)
        for config in args.configs
        for sequence in args.sequences
    }
    for task in tasks:
        task["asset_contract"] = asset_contracts[
            (task["config"], task["sequence"])]
    plan_payload = {
        "contract": "reproducible-benchmark-plan-v3",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tasks": tasks, "source": state,
        "full_fusion_gate_is_predeclared": True,
        "motion_consensus_counterfactual_gate_is_predeclared": (
            "semantic_motion_consensus_shadow" in args.configs),
        "motion_consensus_end_to_end_gate_is_predeclared": all(
            config in args.configs
            for config in (
                "semantic",
                "semantic_motion_consensus_shadow",
                "semantic_motion_consensus",
            )),
        "motion_consensus_required_heldout_sequences":
            MOTION_CONSENSUS_HELDOUT_SEQUENCES,
        "motion_tempered_counterfactual_gate_is_predeclared": (
            "semantic_motion_tempered_shadow" in args.configs),
        "motion_tempered_end_to_end_gate_is_predeclared": all(
            config in args.configs
            for config in (
                "semantic",
                "semantic_motion_tempered_shadow",
                "semantic_motion_tempered",
            )),
        "motion_tempered_required_heldout_sequences":
            MOTION_TEMPERED_HELDOUT_SEQUENCES,
        "motion_direct_counterfactual_gate_is_predeclared": (
            "semantic_motion_direct_shadow" in args.configs and
            set(MOTION_DIRECT_HELDOUT_SEQUENCES).issubset(args.sequences) and
            tuple(sorted(args.seeds)) == MOTION_DIRECT_HELDOUT_SEEDS),
        "motion_direct_end_to_end_gate_is_predeclared": (
            all(
                config in args.configs
                for config in (
                    "semantic",
                    "semantic_motion_direct_shadow",
                    "semantic_motion_direct",
                )) and
            set(MOTION_DIRECT_HELDOUT_SEQUENCES).issubset(args.sequences) and
            tuple(sorted(args.seeds)) == MOTION_DIRECT_HELDOUT_SEEDS),
        "motion_direct_required_heldout_sequences":
            MOTION_DIRECT_HELDOUT_SEQUENCES,
        "motion_direct_required_heldout_seeds":
            MOTION_DIRECT_HELDOUT_SEEDS,
        "motion_rgbd_counterfactual_gate_is_predeclared": (
            "semantic_motion_rgbd_shadow" in args.configs and
            set(MOTION_RGBD_HELDOUT_SEQUENCES).issubset(args.sequences) and
            tuple(sorted(args.seeds)) == MOTION_RGBD_HELDOUT_SEEDS),
        "motion_rgbd_end_to_end_gate_is_predeclared": (
            all(
                config in args.configs
                for config in (
                    "semantic",
                    "semantic_motion_rgbd_shadow",
                    "semantic_motion_rgbd",
                )) and
            set(MOTION_RGBD_HELDOUT_SEQUENCES).issubset(args.sequences) and
            tuple(sorted(args.seeds)) == MOTION_RGBD_HELDOUT_SEEDS),
        "motion_rgbd_required_heldout_sequences":
            MOTION_RGBD_HELDOUT_SEQUENCES,
        "motion_rgbd_required_heldout_seeds":
            MOTION_RGBD_HELDOUT_SEEDS,
        "motion_ablation_is_predeclared": all(
            config in args.configs
            for config in MOTION_ABLATION_CONFIGS),
        "motion_ablation_configs": MOTION_ABLATION_CONFIGS,
        "flow_adaptive_incumbent_gate": {
            "protocol": "flow-adaptive-vs-exact-incumbent-gate-v1",
            "predeclared": (
                set(args.configs) == {
                    FLOW_ADAPTIVE_PARENT_CONFIG,
                    FLOW_ADAPTIVE_CANDIDATE_CONFIG,
                } and
                set(args.sequences) ==
                    set(FLOW_ADAPTIVE_DEVELOPMENT_SEQUENCES) and
                tuple(sorted(args.seeds)) ==
                    FLOW_ADAPTIVE_REQUIRED_SEEDS and
                args.heldout_stride == 0 and
                not args.disable_gaussian_mapper
            ),
            "parent": FLOW_ADAPTIVE_PARENT_CONFIG,
            "candidate": FLOW_ADAPTIVE_CANDIDATE_CONFIG,
            "required_sequences": FLOW_ADAPTIVE_DEVELOPMENT_SEQUENCES,
            "required_seeds": FLOW_ADAPTIVE_REQUIRED_SEEDS,
            "minimum_ate_improvement_fraction":
                FLOW_ADAPTIVE_MIN_ATE_IMPROVEMENT,
            "maximum_risk_degradation_fraction":
                FLOW_ADAPTIVE_MAX_RISK_DEGRADATION,
            "maximum_failure_rate_increase":
                FLOW_ADAPTIVE_MAX_FAILURE_RATE_INCREASE,
            "minimum_sequence_passes":
                FLOW_ADAPTIVE_MIN_SEQUENCE_PASSES,
            "minimum_paired_ate_wins":
                FLOW_ADAPTIVE_MIN_PAIRED_ATE_WINS,
            "risk_metrics": FLOW_ADAPTIVE_RISK_METRICS,
            "config_contract": flow_adaptive_config_contract(),
            "runtime_environment": {
                "PYTORCH_CUDA_ALLOC_CONF": "unset",
            },
        },
        "shadow_translation_tracking_gate": {
            "protocol": "schur-shadow-translation-tracking-gate-v5",
            "candidate": SHADOW_TRANSLATION_CONFIG,
            "controls": SHADOW_TRANSLATION_CONTROLS,
            "required_seeds": SHADOW_TRANSLATION_REQUIRED_SEEDS,
            "frozen_parameters": SHADOW_TRANSLATION_FROZEN_PARAMETERS,
            "development_sequences":
                SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES,
            "heldout_sequences": SHADOW_TRANSLATION_HELDOUT_SEQUENCES,
            "development_predeclared": (
                args.sync_local_mapping and
                args.sync_loop_closing and
                args.disable_gaussian_mapper and
                set(SHADOW_TRANSLATION_TRACKING_CONFIGS).issubset(
                    args.configs) and
                set(SHADOW_TRANSLATION_DEVELOPMENT_SEQUENCES).issubset(
                    args.sequences) and
                tuple(sorted(args.seeds)) ==
                    SHADOW_TRANSLATION_REQUIRED_SEEDS),
            "heldout_predeclared": (
                args.sync_local_mapping and
                args.sync_loop_closing and
                args.disable_gaussian_mapper and
                set(SHADOW_TRANSLATION_TRACKING_CONFIGS).issubset(
                    args.configs) and
                set(SHADOW_TRANSLATION_HELDOUT_SEQUENCES).issubset(
                    args.sequences) and
                tuple(sorted(args.seeds)) ==
                    SHADOW_TRANSLATION_REQUIRED_SEEDS),
        },
        "task_order": (
            "each sequence/seed block executes all configurations serially "
            "under one GPU lock; configuration order alternates between blocks"),
    }
    if args.dyn19_phase is not None:
        plan_payload["dyn19"] = {
            "experiment_id": DYN19_EXPERIMENT_ID,
            "phase": args.dyn19_phase,
            "claim_bearing_sequences": DYN19_CLAIM_SEQUENCES,
            "required_seeds": DYN19_REQUIRED_SEEDS,
            "phase_configs": DYN19_PHASE_CONFIGS[args.dyn19_phase],
            "factor_matrix": DYN19_FACTOR_MATRIX,
            "config_contract": dyn19_config_contract(),
            "semantic_mask_snapshot_contract": {
                "required_for_main_phase": True,
                "reference_config": "dyn19_semantic",
                "task_flag": "--export-static-masks",
                "description": (
                    "Semantic static-mask snapshots freeze both common-view "
                    "static regions and occluded-background proxy sources."),
            },
            "recovered_support_overlap_contract": {
                "required_for_main_full": True,
                "full_config": "dyn19_full",
                "zero_leak_required": True,
                "vacuous_rows_are_not_positive_recovery_evidence": True,
            },
        }
    freeze_benchmark_plan(root, plan_payload, tasks)
    if args.dry_run:
        print(root)
        print(f"planned_tasks={len(tasks)}")
        return 0

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_paired_block, block): block
            for block in blocks
        }
        for future in concurrent.futures.as_completed(futures):
            block_results = future.result()
            results.extend(block_results)
            for result in block_results:
                print(json.dumps({key: result.get(key) for key in
                                  ("status", "config", "sequence", "seed",
                                   "run_dir", "error")}))
    atomic_json(root / "all_results.json", {"results": results})
    aggregate(root, results)
    return 0 if all(result["status"] == "complete" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
