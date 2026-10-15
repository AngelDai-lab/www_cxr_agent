"""Seven-item diagnostic checklist loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_checklist(config_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load checklist entries from ``configs/checklist_7.yaml``."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "configs" / "checklist_7.yaml"
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    items = raw.get("checklist") or []
    if not isinstance(items, list):
        raise TypeError("checklist must be a list")
    return items
