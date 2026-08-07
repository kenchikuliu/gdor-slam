#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: ./run_tum_dynamic.sh [sequence] [options]

Sequences:
  walking_xyz, walking_halfsphere, walking_static
  sitting_xyz, sitting_halfsphere

Options:
  --config NAME          semantic, geometry, flow, full, semantic_motion,
                         dypho_compatible, dypho_decoupled, dypho_flow_guarded,
                         dypho_flow_adaptive, or full_motion (default: semantic)
  --mask-config PATH     Use an explicit YAML config; sets config name to custom
  --seed N               Non-negative experiment seed (default: 0)
  --output-dir PATH      New output directory; existing paths are rejected
  --heldout-stride N     Save eligible non-keyframes every N frames (default: 0)
  --max-frames N         Limit input for smoke testing only (default: full sequence)
  --gpu ID               Physical GPU exposed through CUDA_VISIBLE_DEVICES
  --no-viewer            Disable the GUI viewer
  --no-realtime          Disable playback sleeping (recommended for benchmarks)
  --dry-run              Validate and print the command without running it
  --help                 Show this message

Legacy aliases:
  --semantic-only, --full-fusion, --flow-only
USAGE
}

PROJECT_DIR="${DYNA_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATASETS_DIR="${DYNA_DATASETS_ROOT:-$(dirname "${PROJECT_DIR}")}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_DIR}/output}"

SEQUENCE="walking_halfsphere"
if [[ $# -gt 0 && "$1" != --* ]]; then
    SEQUENCE="$1"
    shift
fi

CONFIG_NAME="semantic"
MASK_CONFIG_OVERRIDE=""
SEED=0
HELDOUT_STRIDE=0
MAX_FRAMES=0
GPU=""
USE_VIEWER=true
REALTIME=true
DRY_RUN=false
OUTPUT_DIR_OVERRIDE="${DYNA_OUTPUT_DIR:-}"

require_value() {
    if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            require_value "$@"
            CONFIG_NAME="$2"
            shift 2
            ;;
        --mask-config)
            require_value "$@"
            MASK_CONFIG_OVERRIDE="$2"
            CONFIG_NAME="custom"
            shift 2
            ;;
        --seed)
            require_value "$@"
            SEED="$2"
            shift 2
            ;;
        --output-dir)
            require_value "$@"
            OUTPUT_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --heldout-stride)
            require_value "$@"
            HELDOUT_STRIDE="$2"
            shift 2
            ;;
        --max-frames)
            require_value "$@"
            MAX_FRAMES="$2"
            shift 2
            ;;
        --gpu)
            require_value "$@"
            GPU="$2"
            shift 2
            ;;
        --no-viewer)
            USE_VIEWER=false
            shift
            ;;
        --no-realtime)
            REALTIME=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --semantic-only)
            CONFIG_NAME="semantic"
            shift
            ;;
        --full-fusion)
            CONFIG_NAME="full"
            shift
            ;;
        --flow-only)
            CONFIG_NAME="flow"
            echo "[notice] --flow-only now selects the Semantic+Residual-Flow ablation." >&2
            shift
            ;;
        --debug)
            echo "--debug no longer rewrites YAML with sed. Use --mask-config with an explicit debug-enabled config." >&2
            exit 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for numeric in "$SEED" "$HELDOUT_STRIDE" "$MAX_FRAMES"; do
    if [[ ! "$numeric" =~ ^[0-9]+$ ]]; then
        echo "Seed, held-out stride, and max frames must be non-negative integers." >&2
        exit 2
    fi
done

case "$SEQUENCE" in
    walking_xyz|fr3_walking_xyz)
        DATASET_SLUG="rgbd_dataset_freiburg3_walking_xyz"
        SEQUENCE_NAME="fr3_walking_xyz"
        ;;
    walking_halfsphere|fr3_walking_halfsphere)
        DATASET_SLUG="rgbd_dataset_freiburg3_walking_halfsphere"
        SEQUENCE_NAME="fr3_walking_halfsphere"
        ;;
    walking_static|fr3_walking_static)
        DATASET_SLUG="rgbd_dataset_freiburg3_walking_static"
        SEQUENCE_NAME="fr3_walking_static"
        ;;
    sitting_xyz|fr3_sitting_xyz)
        DATASET_SLUG="rgbd_dataset_freiburg3_sitting_xyz"
        SEQUENCE_NAME="fr3_sitting_xyz"
        ;;
    sitting_halfsphere|fr3_sitting_halfsphere)
        DATASET_SLUG="rgbd_dataset_freiburg3_sitting_halfsphere"
        SEQUENCE_NAME="fr3_sitting_halfsphere"
        ;;
    *)
        echo "Unknown sequence: $SEQUENCE" >&2
        usage >&2
        exit 2
        ;;
esac

declare -A CONFIG_FILES=(
    [semantic]="mask_config.yaml"
    [geometry]="mask_config_geo.yaml"
    [flow]="mask_config_flow.yaml"
    [full]="mask_config_full.yaml"
    [dypho_compatible]="mask_config_dypho_compatible.yaml"
    [dypho_decoupled]="mask_config_dypho_decoupled.yaml"
    [dypho_flow_guarded]="mask_config_dypho_flow_guarded.yaml"
    [dypho_flow_adaptive]="mask_config_dypho_flow_adaptive.yaml"
    [semantic_motion]="mask_config_semantic_motion.yaml"
    [full_motion]="mask_config_full_motion.yaml"
)

if [[ -n "$MASK_CONFIG_OVERRIDE" ]]; then
    MASK_CFG="$MASK_CONFIG_OVERRIDE"
elif [[ -n "${CONFIG_FILES[$CONFIG_NAME]:-}" ]]; then
    MASK_CFG="${PROJECT_DIR}/config/${CONFIG_FILES[$CONFIG_NAME]}"
else
    echo "Unknown config: $CONFIG_NAME" >&2
    exit 2
fi

DATASET_PATH="${DATASETS_DIR}/${DATASET_SLUG}"
ASSOCIATION_FILE="${ASSOCIATION_FILE:-${DATASET_PATH}/associations.txt}"
VOCAB="${PROJECT_DIR}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_SLAM_CFG="${PROJECT_DIR}/cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml"
GAUSSIAN_CFG="${PROJECT_DIR}/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml"
BINARY="${PROJECT_DIR}/bin/tum_rgbd_dynamic"

for file in "$BINARY" "$VOCAB" "$ORB_SLAM_CFG" "$GAUSSIAN_CFG" "$MASK_CFG" "$ASSOCIATION_FILE"; do
    if [[ ! -s "$file" ]]; then
        echo "Required file is missing or empty: $file" >&2
        exit 1
    fi
done
if [[ ! -d "$DATASET_PATH" ]]; then
    echo "Dataset directory not found: $DATASET_PATH" >&2
    exit 1
fi

if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_OVERRIDE"
else
    printf -v SEED_LABEL '%04d' "$SEED"
    RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${OUTPUT_BASE}/${SEQUENCE_NAME}/${CONFIG_NAME}/seed_${SEED_LABEL}_${RUN_STAMP}"
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to reuse existing output path: $OUTPUT_DIR" >&2
    exit 1
fi

ARGS=(
    "$VOCAB"
    "$ORB_SLAM_CFG"
    "$GAUSSIAN_CFG"
    "$DATASET_PATH"
    "$ASSOCIATION_FILE"
    "$OUTPUT_DIR"
    "$MASK_CFG"
    --seed "$SEED"
    --heldout-stride "$HELDOUT_STRIDE"
)
if [[ "$USE_VIEWER" == false ]]; then
    ARGS+=(no_viewer)
fi
if [[ "$REALTIME" == false ]]; then
    ARGS+=(--no-realtime)
fi
if (( MAX_FRAMES > 0 )); then
    ARGS+=(--max-frames "$MAX_FRAMES")
fi

echo "Sequence: $SEQUENCE_NAME"
echo "Config:   $CONFIG_NAME ($MASK_CFG)"
echo "Seed:     $SEED"
echo "Output:   $OUTPUT_DIR"
printf 'Command:'
if [[ -n "$GPU" ]]; then
    printf ' CUDA_VISIBLE_DEVICES=%q' "$GPU"
fi
printf ' %q' "$BINARY" "${ARGS[@]}"
printf '\n'

if [[ "$DRY_RUN" == true ]]; then
    exit 0
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"
if [[ -n "$GPU" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$BINARY" "${ARGS[@]}"
else
    "$BINARY" "${ARGS[@]}"
fi

echo "Completed: $OUTPUT_DIR"
echo "ATE: evo_ape tum $DATASET_PATH/groundtruth.txt $OUTPUT_DIR/CameraTrajectory_TUM.txt -a"
