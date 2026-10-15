"""Frozen 14-label CheXpert DenseNet screening with documented preprocessing."""

from __future__ import annotations

import asyncio
from typing import Any

from www_cxr_agent.tools.backends import default_device, ensure_medrax_on_path
from www_cxr_agent.tools.mcp_common import mcp_success, parse_mcp_args


class CxrPathoClassifierTool:
    """14-pathology scores → Agent hypothesis set H."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or default_device()
        self._impl: Any = None

    def _load(self) -> None:
        if self._impl is not None:
            return
        import json, os, torch
        from pathlib import Path
        from torchvision import models, transforms
        from safetensors.torch import load_file
        root=Path(os.environ.get("CXR_CLASSIFIER_DIR", "cache/chexpert14"))
        labels=json.loads((root/"config.json").read_text())["labels"]
        if len(labels)!=14 or len(set(labels))!=14:
            raise ValueError("Expected fourteen distinct CheXpert labels")
        model=models.densenet121(weights=None)
        model.classifier=torch.nn.Linear(model.classifier.in_features,14)
        state=load_file(str(root/"pytorch_model.safetensors"))
        model.load_state_dict({k.removeprefix("densenet."):v for k,v in state.items()},strict=True)
        model.to(self.device).eval().requires_grad_(False)
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self._impl=(model,transform,labels)

    def run(self, image: str) -> dict[str, Any]:
        self._load()
        import torch
        from PIL import Image
        model,transform,labels=self._impl
        with Image.open(image) as im:
            x=transform(im.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = torch.relu(model.features(x))
            pooled = features.mean(dim=(2, 3))
            probabilities = model.classifier(pooled).sigmoid()[0].cpu().tolist()
            # DenseNet linear-head CAM is a screening cue, not a lesion mask.
            cams = torch.einsum("kc,bchw->bkhw", model.classifier.weight, features)[0].relu()
            localization = {}
            for index, label in enumerate(labels):
                cam = cams[index]
                peak = float(cam.max())
                if label == "No Finding" or peak <= 0:
                    localization[label] = {"status": "unavailable"}
                    continue
                ys, xs = torch.where(cam >= .5 * peak)
                height, width = cam.shape
                localization[label] = {
                    "status": "suspected", "method": "linear_head_cam",
                    "bbox_xyxy_normalized": [float(xs.min()) / width, float(ys.min()) / height,
                                             float(xs.max() + 1) / width, float(ys.max() + 1) / height],
                    "peak_xy_normalized": [float(cam.argmax() % width + .5) / width,
                                           float(cam.argmax() // width + .5) / height],
                    "coordinate_space": "input_image", "verified": False,
                    "limitations": "Weak discriminative activation, not a lesion segmentation or anatomical diagnosis"}

        scores=dict(zip(labels,map(float,probabilities)))
        return {"scores":scores,"scores_14":scores,"unavailable_labels":[],
                "locations":localization,
                "backend":"itsomk/chexpert-densenet121", "localization":"weak_cam_requires_visual_verification"}


if __name__ == "__main__":
    from fastmcp.server.server import FastMCP

    args = parse_mcp_args()
    mcp = FastMCP("cxr_patho_classifier")
    tool = CxrPathoClassifierTool()
    semaphore = asyncio.Semaphore(4)

    @mcp.tool(name="cxr_patho_classifier")
    async def cxr_patho_classifier(image: str) -> dict:
        """Pathology screening: CheXpert 14-class probabilities."""
        async with semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool.run, image)
            return mcp_success(**result)

    print(f"cxr_patho_classifier MCP → http://{args.host}:{args.port}/mcp", flush=True)
    mcp.run(transport="http", host=args.host, port=args.port, stateless_http=True)  
