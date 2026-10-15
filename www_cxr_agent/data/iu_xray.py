"""IU X-Ray loader for ``/root/autodl-tmp/datasets/iu_xray/``.

Expected layout::
    {root}/reports_ori.json
    {root}/CXR****_IM-****-*.png
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from www_cxr_agent.data.schema import SampleRecord
from www_cxr_agent.preprocess.text import (
    clean_report_iu_xray,
    truncate_report_words,
)

logger = logging.getLogger(__name__)

_PATIENT_RE = re.compile(r"^(CXR\d+)", re.IGNORECASE)
_DEFAULT_SEED = 42
_SPLIT_RATIO = (0.7, 0.1, 0.2)


def extract_iu_patient_id(image_stem: str) -> str:
    """Extract patient/case id from IU image stem, e.g. ``CXR3809_IM-1919-1001`` → ``CXR3809``."""
    stem = Path(image_stem).stem if "." in image_stem else image_stem
    m = _PATIENT_RE.match(stem.strip())
    if m:
        return m.group(1).upper()
    return stem.split("_")[0].upper()


def _assign_patient_splits(patient_ids: list[str], seed: int) -> dict[str, str]:
    import random

    uniq = sorted(set(patient_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(n * _SPLIT_RATIO[0])
    n_val = int(n * _SPLIT_RATIO[1])
    n_test = n - n_train - n_val
    train_ids = set(uniq[:n_train])
    val_ids = set(uniq[n_train : n_train + n_val])
    test_ids = set(uniq[n_train + n_val :])
    assert len(train_ids) + len(val_ids) + len(test_ids) == n
    mapping: dict[str, str] = {}
    for pid in uniq:
        if pid in train_ids:
            mapping[pid] = "train"
        elif pid in val_ids:
            mapping[pid] = "val"
        else:
            mapping[pid] = "test"
    return mapping


def _resolve_iu_images(root: Path, stems: list[str]) -> list[str]:
    out: list[str] = []
    for stem in stems:
        stem = str(stem).strip()
        if not stem:
            continue
        for suffix in (".png", ".jpg", ".jpeg"):
            cand = root / f"{stem}{suffix}"
            if cand.is_file():
                out.append(str(cand.resolve()))
                break
        else:
            direct = root / stem
            if direct.is_file():
                out.append(str(direct.resolve()))
            else:
                logger.warning("missing IU image for stem: %s", stem)
    return out


def build_iu_xray_samples(
    root: str | Path,
    *,
    split: str | None = None,
    seed: int = _DEFAULT_SEED,
    split_cache: str | Path | None = None,
    limit: int | None = None,
    max_seq_length: int = 60,
) -> tuple[list[SampleRecord], dict[str, int]]:
    """Load IU X-Ray; patient-level 7:1:2 train/val/test."""
    root = Path(root)
    report_path = root / "reports_ori.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)

    with report_path.open(encoding="utf-8") as f:
        reports: dict[str, Any] = json.load(f)

    cache_path = Path(split_cache) if split_cache else root / "iu_patient_split.json"
    patient_ids: list[str] = []
    for _rid, payload in reports.items():
        if not isinstance(payload, dict):
            continue
        images = payload.get("image") or []
        if not images:
            continue
        patient_ids.append(extract_iu_patient_id(str(images[0])))

    if cache_path.is_file():
        with cache_path.open(encoding="utf-8") as f:
            patient_split = json.load(f)
    else:
        patient_split = _assign_patient_splits(patient_ids, seed)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(patient_split, f, indent=2, sort_keys=True)

    stats = {
        "total_raw": 0,
        "dropped_empty_findings": 0,
        "dropped_missing_image": 0,
        "kept": 0,
        "patients": len(set(patient_ids)),
    }
    samples: list[SampleRecord] = []
    want_split = split.strip().lower() if split else None
    if want_split == "valid":
        want_split = "val"

    for report_id, payload in reports.items():
        stats["total_raw"] += 1
        if not isinstance(payload, dict):
            continue
        report = payload.get("report") or {}
        if isinstance(report, str):
            findings_raw = report.strip()
        else:
            findings_raw = str(
                report.get("FINDINGS") or report.get("findings") or ""
            ).strip()
        if not findings_raw:
            stats["dropped_empty_findings"] += 1
            continue
        findings = truncate_report_words(
            clean_report_iu_xray(findings_raw),
            max_seq_length,
        )

        image_stems = payload.get("image") or []
        if not image_stems:
            stats["dropped_missing_image"] += 1
            continue
        patient_id = extract_iu_patient_id(str(image_stems[0]))
        sample_split = patient_split.get(patient_id, "train")
        if want_split and sample_split != want_split:
            continue

        image_paths = _resolve_iu_images(root, [str(x) for x in image_stems])
        if not image_paths:
            stats["dropped_missing_image"] += 1
            continue

        sample: SampleRecord = {
            "id": str(report_id),
            "report_id": str(report_id),
            "dataset": "iu_xray",
            "split": sample_split,
            "findings": findings,
            "findings_raw": findings_raw,
            "image_paths": image_paths,
            "patient_id": patient_id,
        }
        samples.append(sample)
        stats["kept"] += 1
        if limit is not None and len(samples) >= limit:
            break

    return samples, stats


class IUXRayDataset:
    """Lightweight index over IU X-Ray with cached patient split."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str | None = None,
        seed: int = _DEFAULT_SEED,
        split_cache: str | Path | None = None,
        limit: int | None = None,
        max_seq_length: int = 60,
    ):
        self.root = Path(root)
        self.samples, self.stats = build_iu_xray_samples(
            self.root,
            split=split,
            seed=seed,
            split_cache=split_cache,
            limit=limit,
            max_seq_length=max_seq_length,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SampleRecord:
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)
