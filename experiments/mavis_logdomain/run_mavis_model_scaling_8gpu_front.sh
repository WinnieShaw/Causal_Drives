#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-$PWD}"
cd "${project_root}"
project_root="$PWD"
experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wrapper="${experiment_root}/run_mavis_model_scaling.py"
dataset_path="${project_root}/MAVIS/data/test-00000-of-00001.parquet"
model_size="${MAVIS_MODEL_SIZE:-32b}"
case "${model_size}" in
  32b)
    default_model="${MODEL_PATH:?Set MODEL_PATH to the local model checkpoint}"
    run_name="qwen3vl32b_mavis_k20_500_directparquet"
    fixed_source=""
    ;;
  8b)
    default_model="${MODEL_PATH:?Set MODEL_PATH to the local model checkpoint}"
    run_name="qwen3vl8b_mavis_k20_500_cacheisolated_directparquet"
    fixed_source="${MAVIS_FIXED_SOURCE:-${project_root}/outputs/mavis_vcd_aligned_k20_0-499/mavis_drive_scores_k20_0-499_merged.jsonl}"
    ;;
  *)
    echo "MAVIS_MODEL_SIZE must be 8b or 32b." >&2
    exit 1
    ;;
esac
run_name="${MAVIS_RUN_TAG:-${run_name}}"
if [[ ! "${run_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]]; then
  echo "MAVIS_RUN_TAG must be a safe ASCII basename (1-96 characters), not a path." >&2
  exit 1
fi
model_path="${MAVIS_MODEL_PATH:-${default_model}}"
output_dir="${project_root}/outputs/mavis_model_scaling/${run_name}"
python_bin="${MAVIS_PYTHON:-python3}"
gpu_max_memory="${MAVIS_GPU_MAX_MEMORY:-110GiB}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
unset PYTORCH_CUDA_ALLOC_CONF
fixed_args=()
if [[ -n "${fixed_source}" ]]; then
  fixed_args=(--fixed_source "${fixed_source}")
fi

if [[ ! -f "${wrapper}" ]]; then
  echo "Missing experiment wrapper: ${wrapper}" >&2
  exit 1
fi

# The optional CPU-only path validates local checkpoint shards, dataset,
# dependencies and output identity without loading model weights or CUDA.
mode="--launch"
if [[ "${MAVIS_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  mode="--preflight"
fi

PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
exec "${python_bin}" "${wrapper}" "${mode}" \
  --project_root "${project_root}" \
  --model_path "${model_path}" \
  --dataset_path "${dataset_path}" \
  --output_dir "${output_dir}" \
  --model_size "${model_size}" \
  "${fixed_args[@]}" \
  --gpu_max_memory "${gpu_max_memory}"
