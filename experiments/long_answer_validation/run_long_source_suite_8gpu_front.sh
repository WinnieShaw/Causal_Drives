#!/usr/bin/env bash
# Four independent experiments, sequentially on the same allocated eight GPUs.
set -euo pipefail
project_root="${1:-$PWD}"
cd "${project_root}"
project_root="$PWD"
entry="${project_root}/experiments/long_answer_source_validation/run_long_source_validation_8gpu_front.sh"
mode="${SOURCE_PREFLIGHT_ONLY:-0}"
for model in qwen3vl8b qwen3vl32b internvl3_8b; do
  SOURCE_PREFLIGHT_ONLY="${mode}" SOURCE_MODEL="${model}" SOURCE_TASK=long \
  RUN_TAG="${model}_long_explanation_qp_k20_cap256_210" \
  bash "${entry}" "${project_root}"
done
SOURCE_PREFLIGHT_ONLY="${mode}" SOURCE_MODEL=internvl3_8b SOURCE_TASK=mavis \
RUN_TAG=internvl3_8b_mavis_source_visual_k20_cap256_200 \
bash "${entry}" "${project_root}"
