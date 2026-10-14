#!/usr/bin/env bash
set -euo pipefail

project_root="${1:?Usage: $0 PROJECT_ROOT}"

bash "$(dirname "${BASH_SOURCE[0]}")/run_mavis_logdomain_8gpu_front.sh" "${project_root}"
bash "$(dirname "${BASH_SOURCE[0]}")/run_recovery_logdomain_8gpu_front.sh" "${project_root}"

echo "Causal-drive log-domain recheck completed."
