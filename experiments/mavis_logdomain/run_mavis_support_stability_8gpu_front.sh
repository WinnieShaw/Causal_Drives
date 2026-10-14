#!/usr/bin/env bash
set -euo pipefail
project_root="${1:-$PWD}"
cd "${project_root}"
project_root="$PWD"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_size="${MAVIS_MODEL_SIZE:-8b}"
case "${model_size}" in
  8b)
    default_model="${MODEL_PATH:?Set MODEL_PATH to the local model checkpoint}"
    source_tag="qwen3vl8b_mavis_k20_500_cacheisolated_directparquet"
    ;;
  32b)
    default_model="${MODEL_PATH:?Set MODEL_PATH to the local model checkpoint}"
    source_tag="qwen3vl32b_mavis_k20_500_directparquet"
    ;;
  *) echo "MAVIS_MODEL_SIZE must be 8b or 32b" >&2; exit 1 ;;
esac
source="${MAVIS_FIXED_SOURCE:-${project_root}/outputs/mavis_model_scaling/${source_tag}/mavis_drive_scores_k20_0-499_merged.jsonl}"
n="${MAVIS_STABILITY_N:-100}"
run_tag="${MAVIS_RUN_TAG:-qwen3vl${model_size}_mavis_support_k5_10_20_40_draw3_n${n}}"
[[ "${run_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || { echo "Invalid MAVIS_RUN_TAG" >&2; exit 1; }
mode="--launch"
[[ "${MAVIS_PREFLIGHT_ONLY:-0}" != "1" ]] || mode="--preflight"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_CUDA_ALLOC_CONF
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
exec "${MAVIS_PYTHON:-python3}" "${script_dir}/run_mavis_support_stability.py" "${mode}" \
  --project_root "${project_root}" --model_path "${MAVIS_MODEL_PATH:-${default_model}}" \
  --model_size "${model_size}" --dataset_path "${project_root}/MAVIS/data/test-00000-of-00001.parquet" \
  --output_dir "${project_root}/outputs/answer_source_validation/${run_tag}" \
  --fixed_source "${source}" --n "${n}" --gpu_max_memory "${MAVIS_GPU_MAX_MEMORY:-110GiB}"
