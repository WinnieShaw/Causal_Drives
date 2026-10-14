#!/usr/bin/env bash
set -euo pipefail

project_root="${1:-$PWD}"
cd "${project_root}"
project_root="$PWD"
model_path="${MODEL_PATH:?Set MODEL_PATH to the local model checkpoint}"
source_jsonl="${SOURCE_JSONL:-${project_root}/outputs/mavis_vcd_aligned_k20_0-499/mavis_drive_scores_k20_0-499_merged.jsonl}"
run_tag="${RUN_TAG:-mavis_fixed_answer_visual_probe_500_directparquet}"
if [[ ! "${run_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]]; then
  echo "RUN_TAG must be a safe ASCII basename (1-96 characters), not a path." >&2
  exit 1
fi
output_dir="${project_root}/outputs/mavis_visual_evidence_probe/${run_tag}"
script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mavis_visual_evidence_probe.py"
dataset="${MAVIS_DATASET_PATH:-${project_root}/MAVIS/data/test-00000-of-00001.parquet}"

if [[ -e "${output_dir}" ]]; then
  echo "Output exists: ${output_dir}. Use a distinct RUN_TAG; no results overwritten." >&2
  exit 1
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
unset PYTORCH_CUDA_ALLOC_CONF
mode="launch"
if [[ "${PROBE_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  mode="preflight"
fi
PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
exec python3 "${script}" "${mode}" --project_root "${project_root}" \
  --input "${source_jsonl}" --dataset_path "${dataset}" --model_path "${model_path}" \
  --output_dir "${output_dir}"
