"""MCP ``cxr_report_draft`` — RadAgent pattern; backend = MedRAX / ViT-BERT baselines."""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Any
from www_cxr_agent.tools.backends import default_device, ensure_medrax_on_path
from www_cxr_agent.tools.mcp_common import mcp_success, parse_mcp_args

# 本地模型根目录
_LOCAL_MODELS = Path("/root/autodl-tmp/models")
FINDINGS_MODEL = _LOCAL_MODELS / "chexpert-mimic-cxr-findings-baseline"
IMPRESSION_MODEL = _LOCAL_MODELS / "chexpert-mimic-cxr-impression-baseline"


class CxrReportDraftTool:
    """Optional draft only — must NOT be used as final Findings."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or default_device()
        self._impl: Any = None
        self.findings_dir = FINDINGS_MODEL
        self.impression_dir = IMPRESSION_MODEL

    def _load(self) -> None:
        if self._impl is not None:
            return

        import torch
        from PIL import Image
        from transformers import (
            BertTokenizer,
            GenerationConfig,
            VisionEncoderDecoderModel,
            ViTImageProcessor,
        )

        def _load_pair(local_or_id: Path, hf_id: str):
            # 只在本地目录确实存在时才用它；否则仍回退 hf_id（保持离线安全的兜底可打印）
            src = str(local_or_id) if local_or_id.is_dir() else hf_id
            model = VisionEncoderDecoderModel.from_pretrained(src).eval().to(self.device)
            tok = BertTokenizer.from_pretrained(src)
            proc = ViTImageProcessor.from_pretrained(src)
            return model, tok, proc

        f_model, f_tok, f_proc = _load_pair(
            self.findings_dir, "/root/autodl-tmp/models/chexpert-mimic-cxr-findings-baseline"
        )
        i_model, i_tok, i_proc = _load_pair(
            self.impression_dir, "/root/autodl-tmp/models/chexpert-mimic-cxr-impression-baseline"
        )
        self._impl = ("hf", torch, Image, GenerationConfig, f_model, f_tok, f_proc, i_model, i_tok, i_proc)

    def run(self, image: str) -> dict[str, str]:
        self._load()
        if self._impl[0] == "medrax":
            text, _meta = self._impl[1]._run(image)
            if isinstance(text, str) and text.startswith("Error"):
                raise RuntimeError(text)
            return {"text": str(text)}

        torch, Image, GenerationConfig, f_model, f_tok, f_proc, i_model, i_tok, i_proc = self._impl[1:]

        def _gen(model, tok, proc) -> str:
            pil = Image.open(image).convert("RGB")
            pixels = proc(pil, return_tensors="pt").pixel_values
            expected = model.config.encoder.image_size
            if pixels.shape[-1] != expected:
                pixels = torch.nn.functional.interpolate(
                    pixels, size=(expected, expected), mode="bilinear", align_corners=False
                )
            pixels = pixels.to(self.device)
            cfg = GenerationConfig(
                num_return_sequences=1,
                max_length=128,
                use_cache=True,
                bos_token_id=model.config.bos_token_id,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
                decoder_start_token_id=tok.cls_token_id,
            )
            with torch.inference_mode():
                ids = model.generate(pixels, generation_config=cfg)
            return tok.batch_decode(ids, skip_special_tokens=True)[0]

        findings = _gen(f_model, f_tok, f_proc)
        impression = _gen(i_model, i_tok, i_proc)
        text = (
            "CHEST X-RAY REPORT\n\n"
            f"FINDINGS:\n{findings}\n\n"
            f"IMPRESSION:\n{impression}"
        )
        return {"text": text}


if __name__ == "__main__":
    from fastmcp.server.server import FastMCP

    args = parse_mcp_args()
    mcp = FastMCP("cxr_report_draft")
    tool = CxrReportDraftTool()
    semaphore = asyncio.Semaphore(1)

    @mcp.tool(name="cxr_report_draft")
    async def cxr_report_draft(image: str) -> dict:
        """Draft Findings+Impression (reference only; not final report)."""
        async with semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool.run, image)
        return mcp_success(**result)

    print(f"cxr_report_draft MCP → http://{args.host}:{args.port}/mcp", flush=True)
    mcp.run(transport="http", host=args.host, port=args.port, stateless_http=True)