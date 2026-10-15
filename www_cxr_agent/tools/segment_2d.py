"""MCP ``segment_2d`` — RadAgent pattern; backend = MedRAX / TorchXRayVision PSPNet."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Any

from www_cxr_agent.tools.backends import default_device, ensure_medrax_on_path, project_root
from www_cxr_agent.tools.mcp_common import mcp_success, parse_mcp_args

# 本地 PSPNet cache 根目录（写死，禁止走 xrv 联网下载）
# PSPNet 只接受 cache_dir，会自动从 /models_data/pspnet_chestxray_best_model_4.pth 加载
_XRV_CACHE = "/root/autodl-tmp/www-cxr-agent/cache/torchxrayvision"


class Segment2dTool:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or default_device()
        self.temp_dir = project_root() / "runs" / "segmentation"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._impl: Any = None

    def _load(self) -> None:
        if self._impl is not None:
            return
        import matplotlib.pyplot as plt
        import numpy as np
        import skimage.io
        import skimage.measure
        import skimage.transform
        import torch
        import torchvision
        import torchxrayvision as xrv

        if not (Path(_XRV_CACHE) / "pspnet_chestxray_best_model_4.pth").is_file():
            raise FileNotFoundError(
                f"未找到本地 PSPNet 权重: {Path(_XRV_CACHE) / 'pspnet_chestxray_best_model_4.pth'}\n"
                "请将 pspnet_chestxray_best_model_4.pth 放到该路径：\n"
                f" mv pspnet_chestxray_best_model_4.pth {_XRV_CACHE}/\n"
            )
        model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=_XRV_CACHE)
        model = model.to(self.device)
        model.eval()
        transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)]
        )
        organ_map = {
            "Left Clavicle": 0,
            "Right Clavicle": 1,
            "Left Scapula": 2,
            "Right Scapula": 3,
            "Left Lung": 4,
            "Right Lung": 5,
            "Left Hilus Pulmonis": 6,
            "Right Hilus Pulmonis": 7,
            "Heart": 8,
            "Aorta": 9,
            "Facies Diaphragmatica": 10,
            "Mediastinum": 11,
            "Weasand": 12,
            "Spine": 13,
        }
        self._impl = ("xrv", model, transform, organ_map, torch, np, skimage, plt, torchvision, xrv)

    def run(self, image: str, organs: list[str] | str | None = None) -> dict[str, Any]:
        if isinstance(organs, str):
            organs = json.loads(organs) if organs.strip().startswith("[") else [
                o.strip() for o in organs.split(",") if o.strip()
            ]
        self._load()
        if self._impl[0] == "medrax":
            out, _meta = self._impl[1]._run(image, organs)
            if "error" in out:
                raise RuntimeError(out["error"])
            return {
                "mask": out["segmentation_image_path"],
                "metrics": out.get("metrics", {}),
            }

        (
            _k,
            model,
            transform,
            organ_map,
            torch,
            np,
            skimage,
            plt,
            _tv,
            xrv,
        ) = self._impl

        if organs:
            # 大小写不敏感 + 别名归一化
            ci = {k.lower(): k for k in organ_map}  # heart -> Heart
            alias = {
                "lung_fields": ["Left Lung", "Right Lung"],
                "lung": ["Left Lung", "Right Lung"],
                "lungs": ["Left Lung", "Right Lung"],
                "heart": ["Heart"],
                # Agent/checklist terminology -> PSPNet organ names.
                "cardiomediastinum": ["Heart", "Mediastinum"],
                "cardio_mediastinum": ["Heart", "Mediastinum"],
                "cardiac silhouette": ["Heart"],
                "mediastinum": ["Mediastinum"],
                "diaphragm": ["Facies Diaphragmatica"],
                "bones": ["Left Clavicle", "Right Clavicle", "Left Scapula", "Right Scapula", "Spine"],
                "bone": ["Left Clavicle", "Right Clavicle", "Left Scapula", "Right Scapula", "Spine"],
                "spine": ["Spine"],
                "thoracic_spine": ["Spine"],
                "thoracic spine": ["Spine"],
            }
            norm = []
            for o in organs:
                s = o.strip().lower()
                if s in alias:
                    norm += alias[s]
                elif s in ci:
                    norm.append(ci[s])
                else:
                    norm.append(o)  # 保留,交给下方 invalid 检查捕获
            organs = norm
            invalid = [o for o in organs if o not in organ_map]
            if invalid:
                raise ValueError(f"Invalid organs: {invalid}")
            organ_indices = [organ_map[o] for o in organs]
            organ_names = list(organs)
        else:
            organ_indices = list(organ_map.values())
            organ_names = list(organ_map.keys())

        original_img = skimage.io.imread(image)
        if len(original_img.shape) > 2:
            original_img = original_img[:, :, 0]
        img = xrv.datasets.normalize(original_img, 255)[None, ...]
        img = transform(img)
        img = torch.from_numpy(img).to(self.device)
        with torch.no_grad():
            pred = model(img)
        pred_probs = torch.sigmoid(pred)
        pred_masks = (pred_probs > 0.5).float()

        import uuid
        plt.figure(figsize=(10, 10))
        plt.imshow(original_img, cmap="gray")
        colors = plt.cm.rainbow(np.linspace(0, 1, len(organ_indices)))
        metrics: dict[str, Any] = {}
        for organ_idx, color, name in zip(organ_indices, colors, organ_names):
            mask = pred_masks[0, organ_idx].cpu().numpy()
            if mask.sum() <= 0:
                continue
            if mask.shape != original_img.shape:
                orig_h, orig_w = original_img.shape
                crop = min(orig_h, orig_w)
                top, left = (orig_h - crop) // 2, (orig_w - crop) // 2
                resized = skimage.transform.resize(
                    mask, (crop, crop), order=0, preserve_range=True, anti_aliasing=False
                )
                full = np.zeros(original_img.shape)
                full[top : top + crop, left : left + crop] = resized
                mask = full
            colored = np.zeros((*original_img.shape, 4))
            colored[mask > 0] = (*color[:3], 0.3)
            plt.imshow(colored)
            props = skimage.measure.regionprops(mask.astype(int))
            if props:
                p0 = props[0]
                metrics[name] = {
                    "area_pixels": int(mask.sum()),
                    "centroid": (float(p0.centroid[0]), float(p0.centroid[1])),
                    "bbox": tuple(map(int, p0.bbox)),
                    "confidence_score": float(pred_probs[0, organ_idx].mean().cpu()),
                }
        plt.axis("off")
        out_path = self.temp_dir / f"segmentation_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
        return {"mask": str(out_path.resolve()), "metrics": metrics}


if __name__ == "__main__":
    from fastmcp.server.server import FastMCP

    args = parse_mcp_args()
    mcp = FastMCP("segment_2d")
    tool = Segment2dTool()
    semaphore = asyncio.Semaphore(2)

    @mcp.tool(name="segment_2d")
    async def segment_2d(image: str, organs: Any = "") -> dict:
        """2D CXR anatomical segmentation (optional organs list or JSON/CSV)."""
        async with semaphore:
            loop = asyncio.get_event_loop()
            org = organs or None
            result = await loop.run_in_executor(None, lambda: tool.run(image, org))
            return mcp_success(**result)

    print(f"segment_2d MCP → http://{args.host}:{args.port}/mcp", flush=True)
    mcp.run(transport="http", host=args.host, port=args.port, stateless_http=True)
