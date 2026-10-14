#!/usr/bin/env bash
set -euo pipefail
project_root="${1:-$PWD}"
cd "${project_root}"
project_root="$PWD"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_size="${MAVIS_MODEL_SIZE:-32b}"
case "${model_size}" in
  8b|32b) default_model="${MODEL_PATH:?Set MODEL_PATH to the local model checkpoint}" ;;
  *) echo "MAVIS_MODEL_SIZE must be 8b or 32b" >&2; exit 1 ;;
esac
cap="${MAVIS_COMPLETE_MAX_NEW_TOKENS:-256}"
run_tag="${MAVIS_RUN_TAG:-qwen3vl${model_size}_mavis_complete_k20_cap${cap}_500}"
[[ "${run_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || { echo "Invalid MAVIS_RUN_TAG" >&2; exit 1; }
mode="--launch"
[[ "${MAVIS_PREFLIGHT_ONLY:-0}" != "1" ]] || mode="--preflight"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_CUDA_ALLOC_CONF
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
exec "${MAVIS_PYTHON:-python3}" "${script_dir}/run_mavis_complete_answers.py" "${mode}" \
  --project_root "${project_root}" \
  --model_path "${MAVIS_MODEL_PATH:-${default_model}}" --model_size "${model_size}" \
  --dataset_path "${project_root}/MAVIS/data/test-00000-of-00001.parquet" \
  --output_dir "${project_root}/outputs/answer_source_validation/${run_tag}" \
  --max_new_tokens "${cap}" --gpu_max_memory "${MAVIS_GPU_MAX_MEMORY:-110GiB}"
