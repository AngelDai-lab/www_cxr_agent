#!/usr/bin/env bash
set -euo pipefail

# Create a clean migration archive. Run this on a machine with enough free
# space, or replace the output file with a mounted destination. The default
# excludes old runs/cache/logs; use INCLUDE_CHECKPOINT=1 to include Step 40.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="$(cd "$ROOT/.." && pwd)"
OUT="${1:-$BASE/www_cxr_agent_8gpu_internal.tar.zst}"
INCLUDE_CHECKPOINT="${INCLUDE_CHECKPOINT:-1}"

items=(
  "www-cxr-agent/configs"
  "www-cxr-agent/launch"
  "www-cxr-agent/scripts"
  "www-cxr-agent/tests"
  "www-cxr-agent/www_cxr_agent"
  "www-cxr-agent/third_party"
  "www-cxr-agent/pyproject.toml"
  "www-cxr-agent/requirements_www_cxr_agent.txt"
  "www-cxr-agent/INTERNAL_8GPU_README.md"
  "datasets/mimic_cxr"
  "models/Qwen3-14B-Instruct"
  "models/CheXagent-2-3b"
  "models/XraySigLIP__vit-l-16-siglip-384__webli"
  "models/chexpert-mimic-cxr-findings-baseline"
  "models/chexpert-mimic-cxr-impression-baseline"
  "venv_chexagent"
)

if [[ "$INCLUDE_CHECKPOINT" == 1 ]]; then
  items+=(
    "www-cxr-agent/runs/formal_default_labels14_20261002/checkpoints/trainer_state_step40.pt"
    "www-cxr-agent/runs/formal_default_labels14_20261002/config.yaml"
    "www-cxr-agent/runs/formal_default_labels14_20261002/step_stats.csv"
    "www-cxr-agent/runs/formal_default_labels14_20261002/metrics_summary.csv"
    "www-cxr-agent/runs/formal_default_labels14_20261002/validation_metrics.csv"
    "www-cxr-agent/runs/formal_default_labels14_20261002/CODEX_HANDOFF.md"
  )
fi

printf '%s\n' "Writing $OUT"
printf '%s\n' "Included paths:" >&2
printf '  %s\n' "${items[@]}" >&2

tar --zstd -cf "$OUT" -C "$BASE" "${items[@]}"
sha256sum "$OUT" > "$OUT.sha256"
du -h "$OUT" "$OUT.sha256"
