#!/bin/bash

set -euo pipefail

PYTHON_CMD=(conda run --no-capture-output -n gaitlab python)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/source/LEAP_Isaaclab${PYTHONPATH:+:$PYTHONPATH}"

SEED=42
NUM_ENVS=8192
DEVICE="cuda:0"
MAX_ITERATIONS=""
CHECKPOINT=""
SIGMA=""
ENABLE_VIDEO=True
HEADLESS=True
VIDEO_LENGTH=600
VIDEO_INTERVAL=8000
CUDA_VISIBLE_DEVICES_ARG=""
RUN_TAG=""
EXTERNAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
RUN_LEAPHAND_GPU_ID="${RUN_LEAPHAND_GPU_ID:-}"
EXTRA_OVERRIDES=()

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift ;;
        --num_envs) NUM_ENVS="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --max_iterations) MAX_ITERATIONS="$2"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift ;;
        --sigma) SIGMA="$2"; shift ;;
        --visible_device) CUDA_VISIBLE_DEVICES_ARG="$2"; shift ;;
        --tag) RUN_TAG="$2"; shift ;;
        --video) ENABLE_VIDEO=True ;;
        --no_video|--disable_video) ENABLE_VIDEO=False ;;
        --headless) HEADLESS=True ;;
        --no_headless|--no-headless) HEADLESS=False ;;
        --video_length) VIDEO_LENGTH="$2"; shift ;;
        --video_interval) VIDEO_INTERVAL="$2"; shift ;;
        --override) EXTRA_OVERRIDES+=("$2"); shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

RUN_NAME="leap-reorientation-z"
if [ -n "$RUN_TAG" ]; then
    RUN_NAME="${RUN_NAME}-${RUN_TAG}"
fi
RUN_NAME="${RUN_NAME}-${SEED}"

echo "=========================================================="
echo "Starting LEAP Hand Reorientation Training:"
echo "Task: Isaac-Reorient-Cube-Leap"
echo "Num Envs: $NUM_ENVS"
echo "Device: $DEVICE"
echo "Seed: $SEED"
if [ -n "$MAX_ITERATIONS" ]; then
    echo "Max Iterations: $MAX_ITERATIONS"
fi
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
fi
echo "Video: $ENABLE_VIDEO"
echo "Headless: $HEADLESS"
if [[ "$ENABLE_VIDEO" == "True" ]]; then
    echo "Video Length: $VIDEO_LENGTH"
    echo "Video Interval: $VIDEO_INTERVAL"
fi
if [ -n "$CUDA_VISIBLE_DEVICES_ARG" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES_ARG"
fi
echo "Run Name: $RUN_NAME"
echo "=========================================================="

if [ -n "$CUDA_VISIBLE_DEVICES_ARG" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_ARG"
fi

DEVICE_INDEX="${DEVICE#cuda:}"
if [[ "$DEVICE" == cuda:* ]] && [ -n "$CUDA_VISIBLE_DEVICES_ARG" ]; then
    DEVICE="cuda:0"
elif [[ "$DEVICE" == cuda:* && "$DEVICE_INDEX" == "0" && "$RUN_LEAPHAND_GPU_ID" =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES="$RUN_LEAPHAND_GPU_ID"
    DEVICE="cuda:0"
elif [[ "$DEVICE" == cuda:* && "$DEVICE_INDEX" == "0" && "$EXTERNAL_CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
    export CUDA_VISIBLE_DEVICES="$EXTERNAL_CUDA_VISIBLE_DEVICES"
    DEVICE="cuda:0"
fi
echo "Effective CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Effective Device: $DEVICE"

VIDEO_ARGS=()
if [[ "$ENABLE_VIDEO" == "True" ]]; then
    shopt -s nullglob
    NVIDIA_VULKAN_ICDS=(/usr/share/vulkan/icd.d/nvidia*icd*.json /etc/vulkan/icd.d/nvidia*icd*.json)
    shopt -u nullglob
    if [[ "${RUN_LEAPHAND_SKIP_VULKAN_CHECK:-0}" != "1" && ${#NVIDIA_VULKAN_ICDS[@]} -eq 0 ]]; then
        echo "Video requested, but no NVIDIA Vulkan ICD was found under /usr/share/vulkan/icd.d or /etc/vulkan/icd.d." >&2
        echo "Isaac Sim headless rendering will fail before recording video on this machine." >&2
        echo "Rerun with --no_video or fix the NVIDIA Vulkan driver/ICD setup." >&2
        exit 1
    fi
    VIDEO_ARGS=(--video --video_length "$VIDEO_LENGTH" --video_interval "$VIDEO_INTERVAL" --enable_cameras)
fi

HEADLESS_ARGS=()
if [[ "$HEADLESS" == "True" ]]; then
    HEADLESS_ARGS=(--headless)
else
    export HEADLESS=0
fi

TRAIN_ARGS=(
    scripts/rl_games/train.py
    --task Isaac-Reorient-Cube-Leap
    --num_envs "$NUM_ENVS"
    --seed "$SEED"
    --device "$DEVICE"
)

if [ -n "$MAX_ITERATIONS" ]; then
    TRAIN_ARGS+=(--max_iterations "$MAX_ITERATIONS")
fi
if [ -n "$CHECKPOINT" ]; then
    TRAIN_ARGS+=(--checkpoint "$CHECKPOINT")
fi
if [ -n "$SIGMA" ]; then
    TRAIN_ARGS+=(--sigma "$SIGMA")
fi

"${PYTHON_CMD[@]}" "${TRAIN_ARGS[@]}" \
    +agent.params.config.full_experiment_name="$RUN_NAME" \
    "${EXTRA_OVERRIDES[@]}" \
    "${VIDEO_ARGS[@]}" \
    "${HEADLESS_ARGS[@]}"
