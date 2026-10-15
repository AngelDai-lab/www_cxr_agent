"""Shared sample record schema for MIMIC-CXR and IU X-Ray."""

from __future__ import annotations

from typing import Any, TypedDict


class SampleRecord(TypedDict, total=False):
    id: str
    dataset: str
    split: str
    findings: str
    findings_raw: str
    image_paths: list[str]
    labels_14: list[float]
    study_id: str
    subject_id: str
    patient_id: str
    report_id: str
    meta: dict[str, Any]
