#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WWW_CXR_MIMIC_CXR_ROOT="${WWW_CXR_MIMIC_CXR_ROOT:-/data/mimic_cxr/mimic_cxr}"
export CXR_POLICY_MODEL="${CXR_POLICY_MODEL:-/models/Qwen3-14B-Instruct}"
export CXR_CHEXBERT_DIR="${CXR_CHEXBERT_DIR:-$ROOT/cache/chexbert}"
export CXR_LABELS14_PICKLE="${CXR_LABELS14_PICKLE:-$WWW_CXR_MIMIC_CXR_ROOT/labels_14_official_study.pickle}"
export CXR_VQA_COUNT=2 CXR_VQA_GPUS="5,6"
export CXR_CLF_COUNT=1 CXR_CLF_GPU=5 CXR_SEG_GPU=6
export CXR_JUDGE_COUNT=1 CXR_JUDGE_GPUS=7 CXR_JUDGE_GPU=7
export CXR_START_DRAFT="${CXR_START_DRAFT:-0}"
export CXR_LOG_DIR="${CXR_LOG_DIR:-$ROOT/logs/8gpu_internal}"
mkdir -p "$CXR_LOG_DIR"
"$ROOT/launch/start_all_tools.sh" restart
"$ROOT/launch/start_all_tools.sh" status
exec python -u scripts/train_grpo.py \
  --config configs/train_8gpu_internal.yaml --phase full@30 \
  --rollout-devices 1,2,3,4 --rollout-concurrency 4 --judge-concurrency 1 \
  --save-interval 1 --log-interval 1 "$@"
