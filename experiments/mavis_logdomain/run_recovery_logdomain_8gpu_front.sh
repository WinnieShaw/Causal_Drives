#!/usr/bin/env bash
set -euo pipefail

project_root="${1:?Usage: $0 PROJECT_ROOT}"
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/randomized_intervention_recovery_mavis_logdomain.py"
main_results="${MAVIS_MERGED_SCORES:-${project_root}/outputs/mavis_logdomain_k20_0-199/mavis_causal_pmi_logdomain_k20_0-199_merged.jsonl}"
output_dir="${OUTPUT_DIR:-${project_root}/outputs/randomized_recovery_logdomain_m100}"
model_path="${MODEL_PATH:?Set MODEL_PATH to the local model checkpoint}"
dataset_path="${MAVIS_DATASET_PATH:-${project_root}/MAVIS/data/test-00000-of-00001.parquet}"

if [[ ! -s "${main_results}" ]]; then
  echo "Missing main log-domain results: ${main_results}" >&2
  exit 1
fi

mkdir -p "${output_dir}/logs"
cd "${project_root}"

pids=()
for worker_id in 0 1 2 3 4 5 6 7; do
  (
    CUDA_VISIBLE_DEVICES="${worker_id}" \
    PYTHONUNBUFFERED=1 \
    python3 "${script_path}" \
      --model_path "${model_path}" \
      --dataset_path "${dataset_path}" \
      --merged_jsonl "${main_results}" \
      --heldout_start 200 \
      --heldout_end -1 \
      --m 100 \
      --block_size 20 \
      --num_blocks 10 \
      --targets_per_block 5 \
      --seed 20260810 \
      --worker_id "${worker_id}" \
      --num_workers 8 \
      --output "${output_dir}/worker_${worker_id}.jsonl" \
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
  echo "At least one recovery worker failed. Inspect ${output_dir}/logs/." >&2
  exit 1
fi

python3 - "${output_dir}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
files = [output_dir / f"worker_{worker_id}.jsonl" for worker_id in range(8)]
missing = [str(path) for path in files if not path.is_file()]
if missing:
    raise SystemExit(f"Missing worker outputs: {missing}")

rows = []
for path in files:
    with path.open(encoding="utf-8") as fin:
        rows.extend(json.loads(line) for line in fin if line.strip())
rows.sort(key=lambda row: row["global_index"])

indices = [row["global_index"] for row in rows]
if len(rows) != 50 or len(set(indices)) != 50:
    raise SystemExit(f"Expected 50 unique targets, got rows={len(rows)}, unique={len(set(indices))}")
errors = [row for row in rows if "error" in row]
if errors:
    raise SystemExit(f"Recovery contains {len(errors)} failed targets; first={errors[0]}")

merged = output_dir / "randomized_recovery_logdomain_m100_merged.jsonl"
with merged.open("w", encoding="utf-8") as fout:
    for row in rows:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

summary = {
    "num_targets": len(rows),
    "question_qcd_mae": sum(row["errors"]["QCD_MAE_to_randomized"] for row in rows) / len(rows),
    "question_q_obs_mae": sum(row["errors"]["Q_Obs_MAE_to_randomized"] for row in rows) / len(rows),
    "prefix_pcd_mae": sum(row["errors"]["PCD_MAE_to_randomized"] for row in rows) / len(rows),
    "prefix_p_obs_mae": sum(row["errors"]["P_Obs_MAE_to_randomized"] for row in rows) / len(rows),
    "question_qcd_win_fraction": sum(row["errors"]["QCD_wins"] for row in rows) / len(rows),
    "prefix_pcd_win_fraction": sum(row["errors"]["PCD_wins"] for row in rows) / len(rows),
}
summary["question_error_reduction_fraction"] = (
    summary["question_q_obs_mae"] - summary["question_qcd_mae"]
) / summary["question_q_obs_mae"]
summary["prefix_error_reduction_fraction"] = (
    summary["prefix_p_obs_mae"] - summary["prefix_pcd_mae"]
) / summary["prefix_p_obs_mae"]

summary_path = output_dir / "summary.json"
with summary_path.open("w", encoding="utf-8") as fout:
    json.dump(summary, fout, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"Merged recovery results: {merged}")
print(f"Summary: {summary_path}")
PY

echo "Eight-GPU log-domain recovery completed: ${output_dir}"
