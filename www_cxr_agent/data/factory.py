"""Build datasets from ``configs/paths.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from www_cxr_agent.config.loader import load_config
from www_cxr_agent.data.iu_xray import IUXRayDataset
from www_cxr_agent.data.mimic_cxr import MimicCXRDataset
from www_cxr_agent.data.schema import SampleRecord

DatasetName = Literal["mimic_cxr", "iu_xray"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_paths() -> dict:
    cfg = load_config(_project_root() / "configs/paths.yaml")
    return cfg.get("paths") or {}


def build_dataset(
    name: DatasetName,
    *,
    split: str | None = None,
    limit: int | None = None,
    max_seq_length: int | None = None,
) -> list[SampleRecord]:
    paths = load_paths()
    data_paths = paths.get("data") or {}
    if max_seq_length is None:
        pp_cfg = load_config(_project_root() / "configs/data_preprocess.yaml")
        max_seq_length = int(pp_cfg.get("max_seq_length", 60))
    if name == "mimic_cxr":
        root = data_paths.get("mimic_cxr")
        if not root:
            raise ValueError("paths.data.mimic_cxr is not set in configs/paths.yaml")
        return MimicCXRDataset(
            root, split=split, limit=limit, max_seq_length=max_seq_length
        ).samples
    if name == "iu_xray":
        root = data_paths.get("iu_xray")
        if not root:
            raise ValueError("paths.data.iu_xray is not set in configs/paths.yaml")
        split_cache = paths.get("runs", "runs")
        return IUXRayDataset(
            root,
            split=split,
            limit=limit,
            split_cache=Path(split_cache) / "iu_patient_split.json",
            max_seq_length=max_seq_length,
        ).samples
    raise ValueError(f"unknown dataset: {name}")
