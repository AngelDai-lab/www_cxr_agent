#!/usr/bin/env python3
"""Validate installation and hardware without loading model weights."""
from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED = {
    "qwen3_14b": ROOT.parent / "models/Qwen3-14B-Instruct",
    "chexagent": ROOT.parent / "models/CheXagent-2-3b",
    "siglip": ROOT.parent / "models/XraySigLIP__vit-l-16-siglip-384__webli",
    "mimic": ROOT.parent / "datasets/mimic_cxr/mimic_cxr",
    "iu": ROOT.parent / "datasets/iu_xray",
    "chexagent_python": ROOT.parent / "venv_chexagent/bin/python",
}
MODULES = ("torch", "transformers", "fastmcp", "www_cxr_agent.tools.tool_configs", "www_cxr_agent.agents.server_manager")

def main() -> int:
    result = {"root": str(ROOT), "paths": {}, "modules": {}}
    ok = True
    for name, path in EXPECTED.items():
        exists = path.exists()
        result["paths"][name] = {"path": str(path), "exists": exists}
        ok = ok and exists
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            result["modules"][name] = {"ok": True, "version": getattr(mod, "__version__", None)}
        except Exception as exc:
            result["modules"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            ok = False
    try:
        import torch
        result["cuda"] = {"available": torch.cuda.is_available(), "count": torch.cuda.device_count(), "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        ok = ok and torch.cuda.is_available() and torch.cuda.device_count() >= 4
    except Exception as exc:
        result["cuda"] = {"error": f"{type(exc).__name__}: {exc}"}
        ok = False
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
