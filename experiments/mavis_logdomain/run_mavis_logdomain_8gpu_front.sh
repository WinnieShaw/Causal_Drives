#!/usr/bin/env bash
set -euo pipefail

project_root="${1:?Usage: $0 PROJECT_ROOT}"
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/compute_mavis_drive_scores_logdomain.py"
output_dir="${OUTPUT_DIR:-${project_root}/outputs/mavis_logdomain_k20_0-199}"
model_path="${MODEL_PATH:?Set MODEL_PATH to the local model checkpoint}"
dataset_path="${MAVIS_DATASET_PATH:-${project_root}/MAVIS/data/test-00000-of-00001.parquet}"

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
      --end 199 \
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
pattern = re.compile(r"_(\d+)-(\d+)\.jsonl$")
files = []
for path in output_dir.glob("mavis_drive_scores_*.jsonl"):
    match = pattern.search(path.name)
    if match:
        files.append((int(match.group(1)), int(match.group(2)), path))
files.sort()

expected = [(start, min(start + 19, 199)) for start in range(0, 200, 20)]
observed = [(start, end) for start, end, _ in files]
if observed != expected:
    raise SystemExit(f"Incomplete shards: observed={observed}, expected={expected}")

merged = output_dir / "mavis_causal_pmi_logdomain_k20_0-199_merged.jsonl"
rows = 0
with merged.open("w", encoding="utf-8") as fout:
    for _, _, path in files:
        with path.open(encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    json.loads(line)
                    fout.write(line)
                    rows += 1
if rows != 200:
    raise SystemExit(f"Merged row count is {rows}, expected 200")
print(f"Merged {rows} rows into {merged}")
PY

echo "Eight-GPU log-domain MAVIS run completed: ${output_dir}"
