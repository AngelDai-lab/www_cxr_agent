"""Shared path / device helpers for CXR MCP tool backends (RadAgent-style)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MEDRAX_ROOT = _PROJECT_ROOT / "third_party" / "MedRAX"


def project_root() -> Path:
    return _PROJECT_ROOT


def resolve_path(rel_or_abs: str | Path) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (_PROJECT_ROOT / p).resolve()


def default_device() -> str:
    env = os.environ.get("CXR_TOOL_DEVICE")
    if env:
        return env
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def configure_process_device() -> str:
    """Return the selected device and expose it for startup diagnostics."""
    device = default_device()
    os.environ.setdefault("CXR_TOOL_DEVICE_RESOLVED", device)
    return device


def ensure_medrax_on_path() -> Path | None:
    """Add ``third_party/MedRAX`` to ``sys.path`` if present. Returns root or None."""
    if not _MEDRAX_ROOT.is_dir():
        return None
    root = str(_MEDRAX_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _MEDRAX_ROOT


def model_dir(key: str, default_rel: str) -> Path:
    """Resolve local model dir; prefer env ``CXR_MODEL_<KEY>`` then project-relative path."""
    env_key = f"CXR_MODEL_{key.upper()}"
    if os.environ.get(env_key):
        return Path(os.environ[env_key]).resolve()
    return resolve_path(default_rel)
