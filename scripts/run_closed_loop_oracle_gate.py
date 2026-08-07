#!/usr/bin/env python3
"""Run paired full-sequence matched-sham and GT-oracle continuations."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import threading

import run_reproducible_benchmark as benchmark
from prepare_matched_oracle_poses import prepare


CONFIGS = (
    "semantic_motion_rgbd_oracle_sham",
    "semantic_motion_rgbd_oracle",
)
DEFAULT_SEQUENCES = ("tum_walking_xyz", "bonn_crowd")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_ROOT = (
    benchmark.DATASETS /
    "DyGeoFusion-SLAM-experiments/closed_loop_oracle")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences", nargs="+", choices=sorted(benchmark.SEQUENCES),
        default=list(DEFAULT_SEQUENCES))
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be non-negative")
    if len(args.sequences) != len(set(args.sequences)):
        parser.error("sequences must not contain duplicates")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must not contain duplicates")
    if len(args.gpus) != len(set(args.gpus)):
        parser.error("GPU entries must be unique")
    if args.jobs < 1 or args.jobs > len(args.gpus):
        parser.error("jobs must be between 1 and the number of GPUs")

    benchmark.validate_inputs(list(CONFIGS), args.sequences)
    source = benchmark.source_state()
    if source["dirty"] and not args.allow_dirty:
        raise RuntimeError(
            "Refusing a closed-loop gate from dirty sources; commit first "
            "or use --allow-dirty for a non-claim smoke run")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.output_root or (DEFAULT_ROOT / timestamp)
    if root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {root}")
    root.mkdir(parents=True)

    oracle_assets: dict[str, dict[str, object]] = {}
    for sequence in args.sequences:
        sequence_dir, _, _, association_name = benchmark.SEQUENCES[sequence]
        output = root / "oracle_manifests" / f"{sequence}.csv"
        metadata_output = (
            root / "oracle_manifests" / f"{sequence}.metadata.json")
        oracle_assets[sequence] = prepare(
            sequence_dir / association_name,
            sequence_dir / "groundtruth.txt",
            output, metadata_output, 0.1)

    tasks, blocks = benchmark.build_task_blocks(
        list(CONFIGS), args.sequences, args.seeds, args.gpus,
        0, source, root,
        synchronize_local_mapping=True,
        synchronize_loop_closing=True,
        disable_gaussian_mapper=True)
    asset_contracts = {
        (config, sequence):
            benchmark.benchmark_task_asset_contract(config, sequence)
        for config in CONFIGS
        for sequence in args.sequences
    }
    for task in tasks:
        oracle_asset = oracle_assets[task["sequence"]]
        task["matched_oracle_pose_manifest"] = oracle_asset["manifest"]
        task["matched_oracle_pose_sha256"] = oracle_asset["manifest_sha256"]
        task["asset_contract"] = asset_contracts[
            (task["config"], task["sequence"])]

    plan = {
        "contract": "real-closed-loop-matched-sham-oracle-plan-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": source,
        "tasks": tasks,
        "oracle_assets": oracle_assets,
        "claim_scope": (
            "diagnostic upper bound only; not a deployable method and not "
            "evidence for reversible tracking or delayed mapping"),
        "branch_contract": (
            "independent deterministic prefix replay; exact input-state "
            "identity at the first applied Oracle event; both continuations "
            "then run ORB LocalMapping and LoopClosing to sequence end"),
        "matched_sham_contract": (
            "both branches generate and optimize identical static/dynamic "
            "candidates and evaluate the same GT decision; sham always "
            "commits static"),
        "oracle_rule": (
            "dynamic iff instantaneous camera-center translation error "
            "improves by more than 1e-6 m; ties commit static"),
        "independent_unit": "sequence x seed",
        "predeclared_gate": {
            "complete_pairs": (
                "all planned Oracle/sham sequence-seed runs complete"),
            "real_fork": (
                "exact prefix and first-action input-state identity for every "
                "pair, with at least one Oracle action per sequence"),
            "sequence_utility": (
                "on every sequence, mean full-frame ATE RMSE and one-step "
                "translation RPE are strictly lower; overall paired ATE win "
                "rate >= 2/3; failure rate is not worse; one-step translation "
                "RPE p95 is no more than 2% worse"),
            "stop": (
                "any failed clause is a No-Go; do not launch held-out or "
                "mapping claim runs"),
        },
    }
    benchmark.freeze_benchmark_plan(root, plan, tasks)
    if args.dry_run:
        print(root)
        print(f"planned_tasks={len(tasks)}")
        return 0

    benchmark.GPU_LOCKS = {
        str(gpu): threading.Lock() for gpu in args.gpus}
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs) as executor:
        futures = [
            executor.submit(benchmark.run_paired_block, block)
            for block in blocks
        ]
        for future in concurrent.futures.as_completed(futures):
            block_results = future.result()
            results.extend(block_results)
            for result in block_results:
                print(json.dumps({
                    key: result.get(key)
                    for key in (
                        "status", "config", "sequence", "seed",
                        "run_dir", "error")
                }, sort_keys=True))
    benchmark.atomic_json(root / "all_results.json", {"results": results})

    evaluator = subprocess.run(
        [
            sys.executable,
            str(benchmark.PROJECT /
                "scripts/evaluate_closed_loop_oracle.py"),
            str(root),
        ],
        cwd=benchmark.PROJECT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    (root / "closed_loop_oracle_evaluation.log").write_text(
        evaluator.stdout)
    print(evaluator.stdout, end="")
    runs_complete = all(
        result["status"] == "complete" for result in results)
    return 0 if runs_complete and evaluator.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
