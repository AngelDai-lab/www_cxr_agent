#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CXR_TOOL_NAME="${1:-cxr_preprocess}"
export CXR_TOOL_HOST="${2:-127.0.0.1}"

python - <<'PY'
import os
import subprocess
import sys
from www_cxr_agent.tools.tool_configs import SERVERS, unique_launch_specs

tool = os.environ["CXR_TOOL_NAME"]
host = os.environ["CXR_TOOL_HOST"]

def launch(spec):
    cmd = [
        sys.executable,
        "-m",
        spec["module"],
        "--host",
        host,
        "--port",
        str(spec["port"]),
    ]
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)

if tool == "all":
    for spec in unique_launch_specs():
        print(f"# start: {spec['name']} → {spec['module']} :{spec['port']}", flush=True)
    print("Pass a single tool name to run one MCP process in foreground.", flush=True)
    raise SystemExit(0)

raise SystemExit(launch(SERVERS[tool]))
PY
