"""OmegaConf config loader with YAML extends chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def _resolve_extends(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular extends detected: {path}")
    seen.add(path)
    cfg = OmegaConf.load(path)
    raw = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(raw, dict):
        raise TypeError(f"Config root must be a mapping: {path}")
    extends = raw.pop("extends", None)
    merged: dict[str, Any] = {}
    if extends:
        if isinstance(extends, str):
            extends_list = [extends]
        elif isinstance(extends, list):
            extends_list = list(extends)
        else:
            raise TypeError(f"extends must be str or list, got {type(extends)}")
        for ext in extends_list:
            ext_path = (path.parent / ext).resolve()
            base = _resolve_extends(ext_path, seen)
            merged = OmegaConf.merge(OmegaConf.create(merged), OmegaConf.create(base))
    merged = OmegaConf.merge(OmegaConf.create(merged), OmegaConf.create(raw))
    return OmegaConf.to_container(merged, resolve=False)  # type: ignore[return-value]


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> Any:
    """Load YAML config; follow ``extends`` relative to the config file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    base = _resolve_extends(path, set())
    cfg = OmegaConf.create(base)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_container(cfg, resolve=True)
