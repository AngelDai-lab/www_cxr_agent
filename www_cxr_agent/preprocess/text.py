"""R2Gen-style report cleaning (cuhksz-nlp/R2Gen ``modules/tokenizers.py``)."""

from __future__ import annotations

import re

# R2Gen default max report length (tokens); used for word-level truncate in loader.
DEFAULT_MAX_SEQ_LENGTH = 60


def clean_report_mimic_cxr(report: str) -> str:
    """MIMIC-CXR report cleaning as in R2Gen."""
    report_cleaner = (
        lambda t: t.replace("\n", " ")
        .replace("__", "_")
        .replace("  ", " ")
        .replace("..", ".")
        .replace("1. ", "")
        .replace(". 2. ", ". ")
        .replace(". 3. ", ". ")
        .replace(". 4. ", ". ")
        .replace(". 5. ", ". ")
        .replace(" 2. ", ". ")
        .replace(" 3. ", ". ")
        .replace(" 4. ", ". ")
        .replace(" 5. ", ". ")
        .strip()
        .lower()
        .split(". ")
    )
    sent_cleaner = lambda t: re.sub(
        r"[.,?;*!%^&_+():-\[\]{}]",
        "",
        t.replace('"', "")
        .replace("/", "")
        .replace("\\", "")
        .replace("'", "")
        .strip()
        .lower(),
    )
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent)]
    return " . ".join(tokens) + " ."


def clean_report_iu_xray(report: str) -> str:
    """IU X-Ray report cleaning as in R2Gen."""
    report_cleaner = (
        lambda t: t.replace("..", ".")
        .replace("1. ", "")
        .replace(". 2. ", ". ")
        .replace(". 3. ", ". ")
        .replace(". 4. ", ". ")
        .replace(". 5. ", ". ")
        .replace(" 2. ", ". ")
        .replace(" 3. ", ". ")
        .replace(" 4. ", ". ")
        .replace(" 5. ", ". ")
        .strip()
        .lower()
        .split(". ")
    )
    sent_cleaner = lambda t: re.sub(
        r"[.,?;*!%^&_+():-\[\]{}]",
        "",
        t.replace('"', "")
        .replace("/", "")
        .replace("\\", "")
        .replace("'", "")
        .strip()
        .lower(),
    )
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent)]
    return " . ".join(tokens) + " ."


def truncate_report_words(report: str, max_words: int = DEFAULT_MAX_SEQ_LENGTH) -> str:
    """Approximate R2Gen ``max_seq_length`` cap before tokenization."""
    words = report.split()
    if len(words) <= max_words:
        return report
    return " ".join(words[:max_words])
