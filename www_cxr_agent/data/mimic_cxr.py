"""MIMIC-CXR loader for ``/root/autodl-tmp/datasets/mimic_cxr/mimic_cxr/``.

Expected layout::
    {root}/annotation.json
    {root}/labels_14.pickle
    {root}/images/{image_path[i]}
"""

from __future__ import annotations

import json
import logging
import pickle
import os
from pathlib import Path
from typing import Any, Iterable

from www_cxr_agent.data.schema import SampleRecord
from www_cxr_agent.preprocess.text import (
    clean_report_mimic_cxr,
    truncate_report_words,
)

logger = logging.getLogger(__name__)

_SPLIT_ALIASES = {"valid": "val", "validation": "val"}


def _normalize_split(name: str) -> str:
    key = name.strip().lower()
    return _SPLIT_ALIASES.get(key, key)


def _label_lookup_keys(record: dict[str, Any]) -> list[str]:
    """返回按优先级排列的候选标签键.

    labels_14.pickle 以「图片文件名 stem」(如 02aa804e-...) 作键，
    所以必须优先用 image_path 的文件名；再退回 annotation 的 id/study_id。
    """
    keys: list[str] = []
    img_paths = record.get("image_path") or []
    if not isinstance(img_paths, list):
        img_paths = [img_paths]
    # 1) 图片文件名 stem（最高优先级，pickle 键就是它）
    for rel in img_paths:
        rel = str(rel).strip()
        if not rel:
            continue
        stem = Path(rel).stem   # 去扩展名
        if stem and stem not in keys:
            keys.append(stem)
    # 2) annotation 字段
    for key in ("id", "study_id", "subject_id"):
        val = record.get(key)
        if val is not None and str(val).strip():
            cand = str(val)
            if cand not in keys:
                keys.append(cand)
    return keys


def _resolve_image_paths(root: Path, rel_paths: Iterable[str]) -> list[str]:
    images_root = root / "images"
    out: list[str] = []
    for rel in rel_paths:
        rel = str(rel).strip().lstrip("/\\")
        if not rel:
            continue
        cand = images_root / rel
        if cand.is_file():
            out.append(str(cand.resolve()))
            continue
        cand2 = root / rel
        if cand2.is_file():
            out.append(str(cand2.resolve()))
        else:
            logger.warning("missing image: %s", cand)
    return out


def build_mimic_cxr_samples(
    root: str | Path,
    *,
    split: str | None = None,
    limit: int | None = None,
    max_seq_length: int = 60,
) -> tuple[list[SampleRecord], dict[str, int]]:
    """Load MIMIC-CXR samples from ``annotation.json`` + ``labels_14.pickle``."""
    root = Path(root)
    ann_path = root / "annotation.json"
    # Prefer the verified study-level CheXpert export. The legacy image-level
    # pickle remains available only for comparison and is never silently used.
    label_path = Path(os.environ.get("CXR_LABELS14_PICKLE", root / "labels_14_official_study.pickle"))
    # The official study-level export is mandatory for MIMIC training.  Do
    # not silently fall back to the legacy image-level pickle: its keying and
    # label provenance are different and would invalidate the reward.
    if not label_path.is_file():
        raise FileNotFoundError(
            f"Official study-level labels14 pickle required: {label_path}"
        )
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)

    with ann_path.open(encoding="utf-8") as f:
        annotations = json.load(f)
    labels_map: dict[str, Any] = {}
    if not label_path.is_file():
        raise FileNotFoundError(f"Official study-level MIMIC labels14 required: {label_path}")
    with label_path.open("rb") as f:
        labels_map = pickle.load(f)

    # A bare pickle contains no class names. Never infer its column order from
    # a project default or from prediction correlations.
    import hashlib
    from www_cxr_agent.reward.rewards import LABELS_14
    metadata_path = label_path.with_suffix(".metadata.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        order = metadata.get("labels", [])
        if len(order) != 14 or len(set(order)) != 14 or set(order) != set(LABELS_14) or not metadata.get("source"):
            raise ValueError("Invalid labels14 provenance or class order")
        if hashlib.sha256(label_path.read_bytes()).hexdigest() != metadata.get("sha256"):
            raise ValueError("labels14 metadata checksum mismatch")
        order_basis = "producer_metadata"
    else:
        raise FileNotFoundError(f"Verified labels14 metadata required: {metadata_path}")
    reorder = [order.index(name) for name in LABELS_14]
    study_level = metadata.get("key_type") == "study_id_string"
    if not study_level:
        raise ValueError("MIMIC labels14 must use verified study_id_string keys")

    stats = {
        "total_raw": 0,
        "dropped_empty_report": 0,
        "dropped_missing_label": 0,
        "dropped_missing_image": 0,
        "kept": 0,
    }
    samples: list[SampleRecord] = []
    want_split = _normalize_split(split) if split else None

    for split_name, records in annotations.items():
        norm_split = _normalize_split(split_name)
        if want_split and norm_split != want_split:
            continue
        if not isinstance(records, list):
            continue
        for rec in records:
            stats["total_raw"] += 1
            if not isinstance(rec, dict):
                continue
            findings_raw = str(rec.get("report") or rec.get("findings") or "").strip()
            if not findings_raw:
                stats["dropped_empty_report"] += 1
                continue
            findings = truncate_report_words(
                clean_report_mimic_cxr(findings_raw),
                max_seq_length,
            )

            image_paths = _resolve_image_paths(root, rec.get("image_path") or [])
            if not image_paths:
                stats["dropped_missing_image"] += 1
                continue

            labels_14: list[float] | None = None
            if labels_map:
                found = None
                lookup = [str(int(rec.get("study_id")))] + _label_lookup_keys(rec) if rec.get("study_id") is not None else _label_lookup_keys(rec)
                for cand in lookup:
                    if cand in labels_map:
                        vec = labels_map[cand]
                        if vec is not None:
                            found = vec
                            break
                if found is None:
                    raise KeyError(f"labels_14.pickle has no label for MIMIC sample {rec.get('id') or rec.get('image_path')}")
                labels_14 = [float(x) for x in list(found)]
                if len(labels_14) != 14:
                    raise ValueError(f"labels_14 entry for {rec.get('id')} has {len(labels_14)} dimensions")

                if any(x not in (0, 1) for x in labels_14):
                    raise ValueError("labels14 must contain binary values")
                labels_14 = [labels_14[i] for i in reorder]

            sample: SampleRecord = {
                "id": str(rec.get("id") or f"{norm_split}_{stats['kept']}"),
                "dataset": "mimic_cxr",
                "split": norm_split,
                "findings": findings,
                "findings_raw": findings_raw,
                "image_paths": image_paths,
                "study_id": str(rec.get("study_id") or ""),
                "subject_id": str(rec.get("subject_id") or ""),
            }
            if labels_14 is not None:
                sample["labels_14"] = labels_14
                sample["label_order"] = list(LABELS_14)
                sample["gt_source"] = str(label_path.name)
                sample["label_order_basis"] = order_basis
            samples.append(sample)
            stats["kept"] += 1
            if limit is not None and len(samples) >= limit:
                return samples, stats

    return samples, stats


class MimicCXRDataset:
    """Lightweight index over MIMIC-CXR annotations."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str | None = None,
        limit: int | None = None,
        max_seq_length: int = 60,
    ):
        self.root = Path(root)
        self.split = split
        self.samples, self.stats = build_mimic_cxr_samples(
            self.root, split=split, limit=limit, max_seq_length=max_seq_length
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SampleRecord:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)
