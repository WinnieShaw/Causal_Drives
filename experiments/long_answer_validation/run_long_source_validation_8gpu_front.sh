#!/usr/bin/env bash
set -euo pipefail
project_root="${1:-$PWD}"
cd "${project_root}"
project_root="$PWD"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_id="${SOURCE_MODEL:-qwen3vl8b}"
task="${SOURCE_TASK:-long}"
case "${model_id}" in
  qwen3vl8b|qwen3vl32b) family=qwen3_vl; model="${SOURCE_MODEL_PATH:?Set SOURCE_MODEL_PATH to the local model checkpoint}" ;;
  internvl3_8b) family=internvl; model="${project_root}/models/OpenGVLab/InternVL3-8B-hf" ;;
  *) echo "SOURCE_MODEL must be qwen3vl8b/qwen3vl32b/internvl3_8b" >&2; exit 1 ;;
esac
case "${task}" in
  long) default_tag="${model_id}_long_explanation_qp_k20_cap256_210" ;;
  mavis) default_tag="${model_id}_mavis_source_visual_k20_cap256_200" ;;
  *) echo "SOURCE_TASK must be long or mavis" >&2; exit 1 ;;
esac
run_tag="${RUN_TAG:-${default_tag}}"
[[ "${run_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || { echo "Invalid RUN_TAG" >&2; exit 1; }
mode=--launch
[[ "${SOURCE_PREFLIGHT_ONLY:-0}" != "1" ]] || mode=--preflight
log_root="${project_root}/outputs/long_answer_source_validation/foreground_logs"
mkdir -p "${log_root}"
log_suffix="foreground"
[[ "${mode}" != "--preflight" ]] || log_suffix="cpu_preflight"
exec > >(tee -a "${log_root}/${run_tag}_${log_suffix}.log") 2>&1
echo "task=${task} model=${model_id} mode=${mode}; foreground log: ${log_root}/${run_tag}_${log_suffix}.log"
manifest="${SOURCE_MANIFEST:-${project_root}/outputs/source_response_validation/source_response_8b_210_serialload/manifest.json}"
extra=()
[[ "${task}" != "long" ]] || extra=(--existing_manifest "${manifest}")
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_CUDA_ALLOC_CONF
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
exec "${SOURCE_PYTHON:-python3}" "${script_dir}/run_long_source_validation.py" "${mode}" \
  --project_root "${project_root}" --model_path "${SOURCE_MODEL_PATH:-${model}}" \
  --family "${family}" --task "${task}" --cap 256 --support_size 20 \
  --output_dir "${project_root}/outputs/long_answer_source_validation/${run_tag}" "${extra[@]}"
