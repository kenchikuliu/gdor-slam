from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import evaluate_closed_loop_oracle as evaluator


def row(sequence, seed, oracle_ate, sham_ate, oracle_rpe, sham_rpe):
    return {
        "sequence": sequence,
        "seed": seed,
        "status": "complete",
        "real_fork_pass": True,
        "oracle_actions": 1,
        "oracle": {
            "ate_rmse_m": oracle_ate,
            "one_step_rpe": {
                "translation_rmse_m": oracle_rpe,
                "translation_p95_m": oracle_rpe,
            },
            "failures": {"failure_rate": 0.0},
        },
        "sham": {
            "ate_rmse_m": sham_ate,
            "one_step_rpe": {
                "translation_rmse_m": sham_rpe,
                "translation_p95_m": sham_rpe,
            },
            "failures": {"failure_rate": 0.0},
        },
    }


def test_gate_passes_only_complete_predeclared_plan():
    rows = [
        row(sequence, seed, 0.9, 1.0, 0.9, 1.0)
        for sequence in evaluator.REQUIRED_SEQUENCES
        for seed in evaluator.REQUIRED_SEEDS
    ]
    gate = evaluator.build_gate(
        rows, list(evaluator.REQUIRED_SEQUENCES),
        list(evaluator.REQUIRED_SEEDS))
    assert gate["claim_gate_eligible"]
    assert gate["passed"]


def test_incomplete_smoke_is_diagnostic_only():
    rows = [row("tum_walking_xyz", 0, 0.9, 1.0, 0.9, 1.0)]
    gate = evaluator.build_gate(rows, ["tum_walking_xyz"], [0])
    assert not gate["claim_gate_eligible"]
    assert not gate["passed"]
    assert gate["decision"] == "DIAGNOSTIC_ONLY_INCOMPLETE_PLAN"


def test_replay_capture_requires_both_matched_branches():
    contract = evaluator.verify_replay_capture(
        {"replay_state_capture_enabled": True},
        {"replay_state_capture_enabled": True})
    assert contract == {
        "pass": True,
        "oracle_enabled": True,
        "sham_enabled": True,
    }


def test_replay_capture_fails_closed_on_missing_field():
    contract = evaluator.verify_replay_capture(
        {"replay_state_capture_enabled": True}, {})
    assert not contract["pass"]
    assert contract["oracle_enabled"]
    assert not contract["sham_enabled"]
