"""R2Gen-style CXR image preprocessing (eval: Resize 224×224 RGB PNG).

Shared by:
  - MCP tool ``cxr_preprocess`` (Agent 调用)
  - ``data/`` loaders 可选预生成（逻辑等价，只是调用方式不同）

Reference: cuhksz-nlp/R2Gen ``modules/dataloaders.py`` eval transform
  Resize((224, 224)) → ToTensor → ImageNet normalize (normalize 在训练 collate 做；
  MCP 工具输出 224 RGB PNG，下游 frozen 工具各自再做模型内归一化).
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

WindowPreset = Literal["lung", "auto", "none"]
_LUNG_WINDOW_CENTER = -600.0
_LUNG_WINDOW_WIDTH = 1500.0
_DEFAULT_SIZE = 224
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = _PROJECT_ROOT / "runs" / "preprocessed"
_MEDRAX_ROOT = _PROJECT_ROOT / "third_party" / "MedRAX"


def _output_path(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(input_path.resolve()).encode()).hexdigest()[:10]
    return output_dir / f"{input_path.stem}_{digest}_224.png"


def _apply_window(arr: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    return np.clip(arr, lo, hi)


def _load_dicom_array(
    path: Path,
    window_center: Optional[float],
    window_width: Optional[float],
) -> np.ndarray:
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    wc, ww = window_center, window_width
    if wc is None and hasattr(ds, "WindowCenter"):
        wc = ds.WindowCenter
        if isinstance(wc, list):
            wc = wc[0]
    if ww is None and hasattr(ds, "WindowWidth"):
        ww = ds.WindowWidth
        if isinstance(ww, list):
            ww = ww[0]

    if wc is not None and ww is not None:
        arr = _apply_window(arr, float(wc), float(ww))
    else:
        arr = _apply_window(arr, _LUNG_WINDOW_CENTER, _LUNG_WINDOW_WIDTH)

    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    return arr.astype(np.uint8)


def _medrax_dicom_to_pil(
    src: Path,
    out_dir: Path,
    window_center: Optional[float],
    window_width: Optional[float],
) -> Image.Image | None:
    if not _MEDRAX_ROOT.is_dir():
        return None
    root = str(_MEDRAX_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from medrax.tools.dicom import DicomProcessorTool

        tool = DicomProcessorTool(temp_dir=str(out_dir))
        payload, _ = tool._run(
            str(src),
            window_center=window_center,
            window_width=window_width,
        )
        if isinstance(payload, dict) and payload.get("image_path"):
            return Image.open(payload["image_path"]).convert("RGB")
    except ImportError:
        return None
    return None


def _load_rgb_pil(path: Path) -> Image.Image:
    suffix = path.suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        medrax_img = _medrax_dicom_to_pil(path, _DEFAULT_OUT, None, None)
        if medrax_img is not None:
            return medrax_img
        arr = _load_dicom_array(path, None, None)
        return Image.fromarray(arr).convert("RGB")

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def preprocess_cxr_image(
    image: str | Path,
    *,
    output_dir: str | Path | None = None,
    size: int = _DEFAULT_SIZE,
    training: bool = False,
) -> dict[str, str]:
    """R2Gen eval-style: RGB + Resize(size×size) → PNG path.

    ``training=True`` 时额外 RandomCrop/HorizontalFlip（与 R2Gen train transform 一致，
    仅 GRPO 训练 dataloader 使用；MCP 工具始终 ``training=False``）。
    """
    src = Path(image)
    if not src.is_file():
        raise FileNotFoundError(src)

    out_dir = Path(output_dir) if output_dir else _DEFAULT_OUT
    pil = _load_rgb_pil(src)

    if training:
        # R2Gen train: Resize(256) → RandomCrop(224) → RandomHorizontalFlip
        pil = pil.resize((256, 256), Image.Resampling.BILINEAR)
        import random

        w, h = pil.size
        if w > size and h > size:
            left = random.randint(0, w - size)
            top = random.randint(0, h - size)
            pil = pil.crop((left, top, left + size, top + size))
        else:
            pil = pil.resize((size, size), Image.Resampling.BILINEAR)
        if random.random() > 0.5:
            pil = pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    else:
        # R2Gen val/test: Resize((224, 224))
        pil = pil.resize((size, size), Image.Resampling.BILINEAR)

    out_path = _output_path(src, out_dir)
    # Concurrent rollouts can preprocess the same image. Publish complete PNGs
    # atomically so another tool never reads a partially written file.
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".png", delete=False) as stream:
        temporary = stream.name
    try:
        pil.save(temporary, format="PNG")
        os.replace(temporary, out_path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return {"image": str(out_path.resolve())}
