#!/usr/bin/env bash

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-/mnt/nas_datasets/slam-experiments/DynaGS-SLAM/dyn21_protocol300_current_20260917_run01}"
PLAN="$REPO/docs/DYN21_PROTOCOL300_CURRENT_PLAN.yaml"
PROTOCOL="/home/slam/experiments/dypho_skill_full_rerun_20260729/protocol.yaml"
BINARY="$REPO/bin/tum_rgbd_dynamic"
VOCAB="$REPO/ORB-SLAM3/Vocabulary/ORBvoc.txt"
MASK_CONFIG="$REPO/config/mask_config_dyn19_full.yaml"
YOLO_ENGINE="/mnt/nvme_data/datasets/DyGeoFusion-SLAM+/model/yolov8s.engine"

EXPECTED_PROTOCOL_SHA="4ecc0b1dd48d5462324888b1c9d94d8ef18c4d83ce4f057b2ed65a264fbf9bbe"
EXPECTED_BINARY_SHA="42146603f1a43298b6c2925ae97457dd0193ec2aa15086f4c2b72cf5f6501bbe"
EXPECTED_VOCAB_SHA="f8dd027f7a6cb88129821341194d7f2c75b77b3394257ddd0d2229863d1a3570"
EXPECTED_MASK_SHA="860bc5080043397d06f517b65528170adbbd474c416b844f1ab4a2a7d8abecc1"
EXPECTED_YOLO_SHA="edfbc192bba7c95d178baf144d7ac1cec3776339f8f274b85c7e5182a10306a3"

sha256() {
    sha256sum "$1" | awk '{print $1}'
}

require_hash() {
    local path="$1"
    local expected="$2"
    local actual
    if [[ ! -s "$path" ]]; then
        echo "Required file is missing or empty: $path" >&2
        exit 1
    fi
    actual="$(sha256 "$path")"
    if [[ "$actual" != "$expected" ]]; then
        echo "SHA-256 mismatch for $path: $actual != $expected" >&2
        exit 1
    fi
}

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "Refusing to reuse existing result root: $OUTPUT_ROOT" >&2
    exit 1
fi
if [[ -n "$(git -C "$REPO" status --porcelain=v1)" ]]; then
    echo "DYN-21 requires a clean committed source tree." >&2
    exit 1
fi

require_hash "$PROTOCOL" "$EXPECTED_PROTOCOL_SHA"
require_hash "$BINARY" "$EXPECTED_BINARY_SHA"
require_hash "$VOCAB" "$EXPECTED_VOCAB_SHA"
require_hash "$MASK_CONFIG" "$EXPECTED_MASK_SHA"
require_hash "$YOLO_ENGINE" "$EXPECTED_YOLO_SHA"
require_hash "$REPO/ORB-SLAM3/lib/libORB_SLAM3.so" "fbde26919f47feebf2f674043b0279bc7f73666274e4bf576a5e853ef71422a1"
require_hash "$REPO/lib/libdynamic_mask_refiner.so" "3906e8719b976dc54b5af0838f7bcbdd8e60e18823d7630593cda2a31b847429"
require_hash "$REPO/lib/libgaussian_mapper.so" "8eef1ae5629b3992ac25ac69295ae65a4f3190eb7e5a546715423b3f75a9d7b7"
require_hash "$REPO/lib/libmotion3d.so" "6bf19ed279a21be241edaa37e27ec630f323ca2a3f26fb9767998679b56165a3"

mkdir -p "$OUTPUT_ROOT/inputs" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/dyn19_full"
cp "$PLAN" "$OUTPUT_ROOT/frozen_plan.yaml"
cp "$PROTOCOL" "$OUTPUT_ROOT/upstream_protocol.yaml"

SOURCE_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
PLAN_SHA="$(sha256 "$PLAN")"
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
START_NS="$(date +%s%N)"

declare -A DATASETS=(
    [tum_walking_xyz]="/mnt/nvme_data/datasets/rgbd_dataset_freiburg3_walking_xyz"
    [tum_walking_halfsphere]="/mnt/nvme_data/datasets/rgbd_dataset_freiburg3_walking_halfsphere"
    [bonn_person_tracking]="/mnt/nvme_data/datasets/rgbd_bonn_dynamic/rgbd_bonn_person_tracking"
    [bonn_crowd3]="/mnt/nvme_data/datasets/rgbd_bonn_dynamic/rgbd_bonn_crowd3"
)
declare -A ASSOCIATION_SHA=(
    [tum_walking_xyz]="74fef79936373d13bb1ec58210521acb97f6a374ecf7572d380e4392a6740cd9"
    [tum_walking_halfsphere]="3e184c1deb5c9b915470c54c3890e762f8456a61f2df78fdacf69ac6c0334aaa"
    [bonn_person_tracking]="ebf660f1047e63dcd723b703c3985cf6c437a49ca09c060503abd1843fdf3f74"
    [bonn_crowd3]="096c47b134ce90f0804aa65544fd6a1be5c204bf19dc7fe2c7874461d5c23a82"
)

prepare_input() {
    local sequence="$1"
    local input_dir="$OUTPUT_ROOT/inputs/$sequence"
    local association="$input_dir/associations_end_300.txt"
    mkdir -p "$input_dir"
    awk 'BEGIN { count = 0 }
         /^[[:space:]]*#/ { next }
         NF > 0 && count < 300 { print; count += 1 }' \
        "${DATASETS[$sequence]}/associations.txt" > "$association"
    if [[ "$(wc -l < "$association")" -ne 300 ]]; then
        echo "$sequence did not yield exactly 300 association rows" >&2
        exit 1
    fi
    require_hash "$association" "${ASSOCIATION_SHA[$sequence]}"
}

for sequence in tum_walking_xyz tum_walking_halfsphere bonn_person_tracking bonn_crowd3; do
    prepare_input "$sequence"
done

cat > "$OUTPUT_ROOT/campaign.status" <<EOF
contract=dyn21-protocol300-current-campaign-v1
experiment_id=DYN-21_PROTOCOL300_CURRENT_20260917
source_commit=$SOURCE_COMMIT
plan_sha256=$PLAN_SHA
protocol_sha256=$EXPECTED_PROTOCOL_SHA
frame_start=0
frame_end_exclusive=300
seed=0
start_utc=$START_UTC
start_epoch_ns=$START_NS
EOF

run_one() {
    local gpu="$1"
    local sequence="$2"
    local dataset="${DATASETS[$sequence]}"
    local association="$OUTPUT_ROOT/inputs/$sequence/associations_end_300.txt"
    local run_dir="$OUTPUT_ROOT/dyn19_full/$sequence/seed_0000"
    local log="$OUTPUT_ROOT/logs/${sequence}.log"
    local time_file="$OUTPUT_ROOT/logs/${sequence}.time"
    local status="$OUTPUT_ROOT/logs/${sequence}.status"
    local orb_config gaussian_config

    if [[ "$sequence" == tum_* ]]; then
        orb_config="$REPO/cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml"
        gaussian_config="$REPO/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml"
    else
        orb_config="$REPO/cfg/ORB_SLAM3/RGB-D/Bonn/bonn_rgbd.yaml"
        gaussian_config="$REPO/cfg/gaussian_mapper/RGB-D/Bonn/bonn_rgbd.yaml"
    fi

    mkdir -p "$run_dir"
    cat > "$status" <<EOF
contract=dyn21-protocol300-current-run-v1
sequence=$sequence
gpu=$gpu
seed=0
frame_start=0
frame_end_exclusive=300
source_commit=$SOURCE_COMMIT
plan_sha256=$PLAN_SHA
protocol_sha256=$EXPECTED_PROTOCOL_SHA
association=$association
association_sha256=${ASSOCIATION_SHA[$sequence]}
binary_sha256=$EXPECTED_BINARY_SHA
mask_config_sha256=$EXPECTED_MASK_SHA
start_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
start_epoch_ns=$(date +%s%N)
EOF

    local command=(
        "$BINARY" "$VOCAB" "$orb_config" "$gaussian_config"
        "$dataset" "$association" "$run_dir" "$MASK_CONFIG"
        no_viewer --no-realtime --seed 0 --heldout-stride 20
    )

    set +e
    (
        cd "$REPO"
        env CUDA_VISIBLE_DEVICES="$gpu" \
            /usr/bin/time -v -o "$time_file" "${command[@]}"
    ) > "$log" 2>&1
    local rc=$?
    set -e

    cat >> "$status" <<EOF
end_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
end_epoch_ns=$(date +%s%N)
exit_code=$rc
EOF
    if [[ "$rc" -ne 0 ]]; then
        tail -n 100 "$log" >&2
        return "$rc"
    fi

    python3 - "$run_dir" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])
required = [
    "CameraTrajectory_AllFrames_TUM.txt",
    "CameraTrajectory_TUM.txt",
    "frame_metrics.csv",
    "run_summary.json",
    "heldout_novel_view/manifest.csv",
]
for relative in required:
    path = run / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing required artifact: {path}")
summary = json.loads((run / "run_summary.json").read_text())
if summary.get("input_frames") != 300 or summary.get("processed_frames") != 300:
    raise SystemExit(f"unexpected run summary frame count: {summary}")
with (run / "heldout_novel_view/manifest.csv").open(newline="") as handle:
    if not list(csv.DictReader(handle)):
        raise SystemExit("held-out manifest has no rows")
maps = []
for path in run.glob("*_shutdown"):
    match = re.fullmatch(r"(\d+)_shutdown", path.name)
    if match:
        maps.append((int(match.group(1)), path))
if not maps:
    raise SystemExit("final Gaussian map directory is missing")
map_dir = max(maps)[1]
ply = list((map_dir / "ply/point_cloud").glob("iteration_*/point_cloud.ply"))
for path in [map_dir / "ply/cameras.json", *ply]:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing final-map artifact: {path}")
PY
}

set +e
(run_one 0 tum_walking_xyz && run_one 0 bonn_person_tracking) &
PID0=$!
(run_one 1 tum_walking_halfsphere && run_one 1 bonn_crowd3) &
PID1=$!
wait "$PID0"
RC0=$?
wait "$PID1"
RC1=$?
set -e

cat >> "$OUTPUT_ROOT/campaign.status" <<EOF
end_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
end_epoch_ns=$(date +%s%N)
gpu0_exit_code=$RC0
gpu1_exit_code=$RC1
EOF

if [[ "$RC0" -ne 0 || "$RC1" -ne 0 ]]; then
    echo "DYN-21 Protocol-300 campaign failed: gpu0=$RC0 gpu1=$RC1" >&2
    exit 1
fi

echo "Completed DYN-21 Protocol-300 runs: $OUTPUT_ROOT"
