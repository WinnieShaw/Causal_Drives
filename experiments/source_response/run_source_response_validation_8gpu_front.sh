#!/usr/bin/env bash
set -euo pipefail

project_root="${1:?Usage: bash run_source_response_validation_8gpu_front.sh PROJECT_ROOT}"
experiment_dir="${project_root}/experiments/source_response_validation"
core_dir="${project_root}/experiments/causal_drive_validation"
model_path="${MODEL_PATH:?Set MODEL_PATH to an existing checkpoint directory}"
dataset_dir="${DATASET_DIR:-${project_root}/vlms-are-biased}"
input_run="${INPUT_RUN:-${project_root}/outputs/paired_source_validation/paired_source_validation_8b_700}"
run_tag="${RUN_TAG:-source_response_8b_210}"
output_dir="${project_root}/outputs/source_response_validation/${run_tag}"
per_topic="${PER_TOPIC:-30}"
mode="${MODE:-all}"
python_bin="${PYTHON_BIN:-python3}"
min_free_kb="${MIN_FREE_KB:-0}"
probe_mb="${PROBE_MB:-256}"
init_timeout="${MODEL_INIT_TIMEOUT:-1800}"

[[ "${run_tag}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || { echo "RUN_TAG must be a safe basename." >&2; exit 1; }
[[ ! -e "${output_dir}" ]] || { echo "Output exists; set a new RUN_TAG: ${output_dir}" >&2; exit 1; }
mkdir -p "${output_dir}/logs"
exec > >(tee "${output_dir}/logs/foreground.log") 2>&1
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONFAULTHANDLER=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
unset PYTORCH_CUDA_ALLOC_CONF
pids=()
probe=""
record_exit() {
  "${python_bin}" -c 'import sys; sys.path.insert(0, sys.argv[1]); from source_response_runtime import write_state; code=int(sys.argv[3]); write_state(sys.argv[2], "worker_exit", exit_code=code, signal_number=(code-128 if code>128 else None))' \
    "${experiment_dir}" "${output_dir}/logs/worker_${1}_exit.json" "${2}"
}
finish() {
  local code="$?"
  trap - EXIT INT TERM
  if [[ -n "${probe}" && -e "${probe}" ]]; then unlink "${probe}"; fi
  if [[ "${code}" -ne 0 ]]; then
    for pid in "${pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  fi
  "${python_bin}" -c 'import sys; sys.path.insert(0, sys.argv[1]); from source_response_runtime import write_state; write_state(sys.argv[2], "foreground_exit", exit_code=int(sys.argv[3]))' \
    "${experiment_dir}" "${output_dir}/logs/foreground_exit.json" "${code}" || true
  echo "Foreground exit_code=${code}; logs retained: ${output_dir}/logs"
  exit "${code}"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
echo "Runtime safeguards: direct Parquet, checkpoint-native 16-bit dtype, complete single-GPU SDPA, sequential model initialization."
available_kb="$(df -Pk "${project_root}" | awk 'NR==2 {print $4}')"
echo "Project filesystem reported free KB: ${available_kb}; actual fsync write test is authoritative."
if [[ "${min_free_kb}" -gt 0 && "${available_kb}" -lt "${min_free_kb}" ]]; then
  echo "Insufficient space on the project filesystem: ${available_kb} KB free; require ${min_free_kb} KB." >&2
  echo "No model started. Free space on this filesystem, then retry; old outputs are not deleted." >&2
  exit 1
fi
[[ "${probe_mb}" -ge 64 ]] || { echo "PROBE_MB must be at least 64." >&2; exit 1; }
probe="$(mktemp "${project_root}/.source_response_probe.XXXXXX")"
dd if=/dev/urandom of="${probe}" bs=1M count="${probe_mb}" conv=fsync status=none
unlink "${probe}"
probe=""
echo "Project filesystem ${probe_mb} MiB actual fsync write test passed."

"${python_bin}" "${experiment_dir}/run_source_response_validation.py" \
  --manifest unused --preflight_only --model_path "${model_path}" \
  --dataset_dir "${dataset_dir}" --core_dir "${core_dir}"
visible_gpus="$("${python_bin}" -c 'import torch; print(torch.cuda.device_count())')"
[[ "${visible_gpus}" -eq 8 ]] || { echo "Requires exactly 8 visible GPUs; got ${visible_gpus}." >&2; exit 1; }
[[ "${per_topic}" -ge 2 ]] || { echo "PER_TOPIC must be at least 2 for the 8-worker launcher." >&2; exit 1; }
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true
fi

"${python_bin}" "${experiment_dir}/run_source_response_validation.py" \
  --prepare_only --input "${input_run}"/shard_*.jsonl \
  --manifest "${output_dir}/manifest.json" --per_topic "${per_topic}" --seed 43 --mode "${mode}"

for worker_id in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="${worker_id}" \
    "${python_bin}" "${experiment_dir}/run_source_response_validation.py" \
      --manifest "${output_dir}/manifest.json" --model_path "${model_path}" \
      --dataset_dir "${dataset_dir}" --core_dir "${core_dir}" --mode "${mode}" \
      --worker_id "${worker_id}" --num_workers 8 \
      --load_lock "${output_dir}/model_loading.lock" \
      --max_memory_per_gpu "${MAX_MEMORY_PER_GPU:-120GiB}" --max_new_tokens 24 \
      --output "${output_dir}/shard_${worker_id}.jsonl" \
      > "${output_dir}/logs/worker_${worker_id}.log" 2>&1 &
  pids+=("$!")
  echo "worker=${worker_id} pid=${pids[worker_id]} initializing; see logs/worker_${worker_id}.log"
  began="${SECONDS}"
  # Start the next model only after the complete preceding model is ready.
  # Already-ready workers compute concurrently; this does not serialize inference.
  while [[ ! -f "${output_dir}/shard_${worker_id}.metadata.json" ]]; do
    if ! kill -0 "${pids[worker_id]}" 2>/dev/null; then
      code=0
      wait "${pids[worker_id]}" || code="$?"
      record_exit "${worker_id}" "${code}"
      tail -n 80 "${output_dir}/logs/worker_${worker_id}.log"
      echo "Model initialization failed: worker=${worker_id}, exit_code=${code}. No further models started."
      exit 1
    fi
    if [[ "$((SECONDS-began))" -ge "${init_timeout}" ]]; then
      echo "Model initialization timeout: worker=${worker_id}; stopping this run's workers."
      tail -n 80 "${output_dir}/logs/worker_${worker_id}.log"
      exit 1
    fi
    sleep 1
  done
  echo "worker=${worker_id} complete model loaded; next worker may initialize."
done
failed=0
for worker_id in 0 1 2 3 4 5 6 7; do
  code=0
  wait "${pids[worker_id]}" || code="$?"
  record_exit "${worker_id}" "${code}"
  if [[ "${code}" -ne 0 ]]; then
    failed=1
    tail -n 60 "${output_dir}/logs/worker_${worker_id}.log"
  fi
done
[[ "${failed}" -eq 0 ]] || { echo "At least one worker failed; raw results and exit diagnostics retained. Analysis not run." >&2; exit 1; }
"${python_bin}" "${experiment_dir}/analyze_source_response_validation.py" \
  --input "${output_dir}"/shard_*.jsonl --manifest "${output_dir}/manifest.json" \
  --output_dir "${output_dir}/analysis" --bootstrap 5000
echo "Completed: ${output_dir}"
