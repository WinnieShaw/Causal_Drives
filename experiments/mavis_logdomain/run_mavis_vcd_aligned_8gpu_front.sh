#!/usr/bin/env bash
set -euo pipefail

project_root="${1:?Usage: $0 PROJECT_ROOT}"
output_root="${2:-${project_root}/outputs}"
experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
script_path="${experiment_root}/compute_mavis_drive_scores_logdomain.py"
summary_script="${experiment_root}/summarize_mavis_drive_scores.py"
model_path="${MODEL_PATH:?Set MODEL_PATH to the local model checkpoint}"
dataset_path="${MAVIS_DATASET_PATH:-${project_root}/MAVIS/data/test-00000-of-00001.parquet}"
output_dir="${output_root}/mavis_vcd_aligned_k20_0-499"

for required in "${script_path}" "${summary_script}" "${model_path}/config.json" "${dataset_path}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required asset: ${required}" >&2
    exit 1
  fi
done

if [[ "$(python3 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)" -lt 8 ]]; then
  echo "This foreground job requires eight visible GPUs." >&2
  exit 1
fi

mkdir -p "${output_dir}/logs"
cd "${project_root}"

python3 "${script_path}" --self_test

pids=()
for worker_id in 0 1 2 3 4 5 6 7; do
  (
    CUDA_VISIBLE_DEVICES="${worker_id}" \
    SKIP_BERTSCORE=1 \
    PYTHONUNBUFFERED=1 \
    python3 "${script_path}" \
      --model_path "${model_path}" \
      --dataset_path "${dataset_path}" \
      --start 0 \
      --end 499 \
      --batch_size 20 \
      --worker_id "${worker_id}" \
      --num_workers 8 \
      --output_dir "${output_dir}" \
      2>&1 | tee "${output_dir}/logs/worker_${worker_id}.log"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one worker failed. Inspect ${output_dir}/logs/." >&2
  exit 1
fi

python3 - "${output_dir}" <<'PY'
import json
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
pattern = re.compile(r"mavis_drive_scores_(\d+)-(\d+)\.jsonl$")
files = []
for path in output_dir.glob("mavis_drive_scores_*.jsonl"):
    match = pattern.fullmatch(path.name)
    if match:
        files.append((int(match.group(1)), int(match.group(2)), path))
files.sort()

expected = [(start, min(start + 19, 499)) for start in range(0, 500, 20)]
observed = [(start, end) for start, end, _ in files]
if observed != expected:
    raise SystemExit(f"Incomplete shards: observed={observed}, expected={expected}")

merged = output_dir / "mavis_drive_scores_k20_0-499_merged.jsonl"
rows = []
for _, _, path in files:
    with path.open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())

rows.sort(key=lambda row: row["global_index"])
indices = [row["global_index"] for row in rows]
if indices != list(range(500)):
    raise SystemExit(f"Expected global indices 0..499, got {len(indices)} rows")

for row in rows:
    schema = row.get("schema", {})
    if schema.get("vcd_intervention") != "target_image_with_null_question":
        raise SystemExit(f"VCD schema check failed at global_index={row['global_index']}")
    steps = row.get("drive_metrics", {}).get("per_step", [])
    if not steps or not all({"QCD_t", "VCD_t", "PCD_t"} <= set(step) for step in steps):
        raise SystemExit(f"Canonical drive fields missing at global_index={row['global_index']}")

with merged.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Validated and merged {len(rows)} rows into {merged}")
PY

python3 "${summary_script}" \
  --input "${output_dir}/mavis_drive_scores_k20_0-499_merged.jsonl" \
  --output_dir "${output_dir}/analysis" \
  --bins 10 \
  --bootstrap 5000 \
  --seed 42

echo "Eight-GPU MAVIS drive run completed: ${output_dir}"
