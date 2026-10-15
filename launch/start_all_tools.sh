#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
LOG_DIR="${CXR_LOG_DIR:-$ROOT/logs}"; PID_DIR="$LOG_DIR/pids"; mkdir -p "$PID_DIR"
PYTHON_MAIN="${CXR_PYTHON_MAIN:-$(command -v python)}"
PYTHON_VQA="${CXR_PYTHON_VQA:-$ROOT/../venv_chexagent/bin/python}"
PROFILE="${CXR_TOOL_PROFILE:-tools}"
case "$PROFILE" in tools|train) ;; *) echo "Unknown profile: $PROFILE" >&2; exit 2 ;; esac
# Both profiles preserve GPU0 for the policy; judge has separate weights on GPU3.
VQA_COUNT="${CXR_VQA_COUNT:-3}"
VQA_GPUS="${CXR_VQA_GPUS:-1,2,3}"
CLF_GPU="${CXR_CLF_GPU:-1}"
SEG_GPU="${CXR_SEG_GPU:-1}"
JUDGE_GPU="${CXR_JUDGE_GPU:-3}"
JUDGE_COUNT="${CXR_JUDGE_COUNT:-3}"
JUDGE_GPUS="${CXR_JUDGE_GPUS:-3,1,2}"
JUDGE_BASE_PORT=7007
DEFAULT_JUDGE=1
# The draft model is part of the declared MCP tool set and the policy may
# select it during verification.  Keep it available by default; callers can
# explicitly disable it with CXR_START_DRAFT=0.
DEFAULT_DRAFT=1
CLF_COUNT="${CXR_CLF_COUNT:-1}"
export CXR_VQA_COUNT="$VQA_COUNT" CXR_CLF_COUNT="$CLF_COUNT"
export CXR_JUDGE_COUNT="$JUDGE_COUNT" CXR_JUDGE_URLS="$(for ((i=0;i<JUDGE_COUNT;i++)); do [[ "$i" -gt 0 ]] && printf ','; printf 'http://127.0.0.1:%d/mcp' $((JUDGE_BASE_PORT+i)); done)"
exec 9>"$PID_DIR/.launcher.lock"
flock -x 9
pid_file() { echo "$PID_DIR/$1.pid"; }
log_file() { echo "$LOG_DIR/$1.log"; }
gpu_at() { tr ',' '\n' <<< "$1" | sed -n "$(( $2 + 1 ))p"; }
# Resolve the real process, including services started outside this script.
service_pid() {
  "$PYTHON_MAIN" - "$1" <<'PYCODE'
import sys, psutil
from www_cxr_agent.tools.tool_configs import SERVERS
name=sys.argv[1]
if name.startswith("llm_judge_tool"):
    port = 7007 + (int(name.rsplit("_", 1)[1]) if "_" in name and name.rsplit("_", 1)[1].isdigit() else 0)
    module = "www_cxr_agent.tools.llm_judge_tool"
else:
    spec=SERVERS[name]; port=spec['port']; module=spec['module']
for p in psutil.process_iter(['pid','cmdline','status']):
    args=p.info['cmdline'] or []
    if p.info['status']==psutil.STATUS_ZOMBIE: continue
    if '-m' not in args or '--port' not in args: continue
    if args[args.index('-m')+1]==module and args[args.index('--port')+1]==str(port):
        print(p.pid); break
PYCODE
}
start_one() {
  local name="$1" port="$2" module="$3" py="$4" gpu="$5" pf pid
  pf="$(pid_file "$name")"; pid="$(service_pid "$name")"
  if [[ -n "$pid" ]]; then echo "$pid" >"$pf"; echo "[running] $name pid=$pid"; return; fi
  echo "[start] $name :$port GPU=$gpu"
  CXR_TOOL_DEVICE="$([[ "$gpu" == cpu ]] && echo cpu || echo cuda:0)" CUDA_VISIBLE_DEVICES="$([[ "$gpu" == cpu ]] && echo "" || echo "$gpu")" setsid nohup "$py" -m "$module" --host 127.0.0.1 --port "$port" 9>&- </dev/null >>"$(log_file "$name")" 2>&1 &
  pid=$!; echo "$pid" >"$pf"
  for ((attempt=0;attempt<60;attempt++)); do
    if ! kill -0 "$pid" 2>/dev/null; then echo "[failed] $name; see $(log_file "$name")" >&2; return 1; fi
    if curl -s --max-time 1 "http://127.0.0.1:$port/mcp" >/dev/null; then echo "[ready] $name pid=$pid"; return; fi
    sleep 1
  done
  echo "[not-ready] $name; see $(log_file "$name")" >&2; return 1
}
stop_one() {
  local pid; pid="$(service_pid "$1")"
  if [[ -n "$pid" ]]; then
    kill "$pid"
    for ((attempt=0;attempt<30;attempt++)); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then echo "[stop-timeout] $1 pid=$pid" >&2; return 1; fi
  fi
  rm -f "$(pid_file "$1")"
}
status_one() {
  local pid; pid="$(service_pid "$1")"
  if [[ -n "$pid" ]]; then
    echo "$pid" >"$(pid_file "$1")"
    local port
    port="$("$PYTHON_MAIN" - "$1" <<'PYCODE'
import sys
from www_cxr_agent.tools.tool_configs import SERVERS
name=sys.argv[1]
if name.startswith('llm_judge_tool'):
    suffix=name.rsplit('_',1)[1] if '_' in name else ''
    print(7007 + (int(suffix) if suffix.isdigit() else 0))
else:
    print(SERVERS[name]['port'])
PYCODE
)"
    if curl -s --max-time 2 "http://127.0.0.1:$port/mcp" >/dev/null; then
      echo "[running] $1 pid=$pid port=$port reachable"
    else echo "[unreachable] $1 pid=$pid port=$port"; fi
  else echo "[stopped] $1"; fi
}
start_all() {
  start_one cxr_preprocess 7001 www_cxr_agent.tools.cxr_preprocess "$PYTHON_MAIN" cpu
  if [[ "${CXR_START_DRAFT:-$DEFAULT_DRAFT}" == 1 ]]; then start_one cxr_report_draft 7002 www_cxr_agent.tools.cxr_report_draft "$PYTHON_MAIN" "$CLF_GPU"; fi
  for ((i=0;i<VQA_COUNT;i++)); do start_one "cxr_vqa_$i" $((7004+i*10)) www_cxr_agent.tools.cxr_vqa "$PYTHON_VQA" "$(gpu_at "$VQA_GPUS" "$i")"; done
  for ((i=0;i<CLF_COUNT;i++)); do start_one "cxr_patho_classifier_$i" $((7003+i*100)) www_cxr_agent.tools.cxr_patho_classifier "$PYTHON_MAIN" "$CLF_GPU"; done
  start_one segment_2d 7006 www_cxr_agent.tools.segment_2d "$PYTHON_MAIN" "$SEG_GPU"
  if [[ "${CXR_START_JUDGE:-$DEFAULT_JUDGE}" == 1 ]]; then
    for ((i=0;i<JUDGE_COUNT;i++)); do
      name="llm_judge_tool"; [[ "$i" -gt 0 ]] && name="llm_judge_tool_$i"
      start_one "$name" $((JUDGE_BASE_PORT+i)) www_cxr_agent.tools.llm_judge_tool "$PYTHON_MAIN" "$(gpu_at "$JUDGE_GPUS" "$i")"
    done
  fi
}
stop_all() { for n in cxr_preprocess cxr_report_draft segment_2d; do stop_one "$n"; done; for ((i=0;i<JUDGE_COUNT;i++)); do name="llm_judge_tool"; [[ "$i" -gt 0 ]] && name="llm_judge_tool_$i"; stop_one "$name"; done; for ((i=0;i<VQA_COUNT;i++)); do stop_one "cxr_vqa_$i"; done; for ((i=0;i<CLF_COUNT;i++)); do stop_one "cxr_patho_classifier_$i"; done; }
status_all() {
  for n in cxr_preprocess cxr_report_draft segment_2d; do status_one "$n"; done
  for ((i=0;i<JUDGE_COUNT;i++)); do name="llm_judge_tool"; [[ "$i" -gt 0 ]] && name="llm_judge_tool_$i"; status_one "$name"; done
  for ((i=0;i<VQA_COUNT;i++)); do status_one "cxr_vqa_$i"; done
  for ((i=0;i<CLF_COUNT;i++)); do status_one "cxr_patho_classifier_$i"; done
}
case "${1:-start}" in start) echo "profile=$PROFILE vqa=$VQA_COUNT gpus=$VQA_GPUS judge_gpu=$JUDGE_GPU"; start_all ;; stop) stop_all ;; restart) stop_all; sleep 1; start_all ;; status) status_all ;; *) echo "usage: $0 {start|stop|restart|status}"; exit 2 ;; esac
