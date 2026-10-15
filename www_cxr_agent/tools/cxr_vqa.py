"""MCP ``cxr_vqa`` + ``cxr_multiview_vqa`` — one process, two tools (shared CheXagent)."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from www_cxr_agent.tools.backends import default_device, ensure_medrax_on_path
from www_cxr_agent.tools.mcp_common import mcp_success, parse_mcp_args

# 本地模型路径（写死，禁止走 HF 联网）
_CHEXAGENT_MODEL = "/root/autodl-tmp/models/CheXagent-2-3b"
_XRAYSIGLIP_MODEL = "/root/autodl-tmp/models/XraySigLIP__vit-l-16-siglip-384__webli"


def _patch_chexagent_runtime_to_local():
    """将 CheXagent config 里指向 HF 仓库的视觉塔路径改成本地路径（幂等）。"""
    cfg_path = Path(_CHEXAGENT_MODEL) / "config.json"
    if not cfg_path.is_file():
        return
    try:
        text = cfg_path.read_text(encoding="utf-8")
        replaced = text.replace(
            "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli", _XRAYSIGLIP_MODEL
        )
        if replaced != text:
            shutil.copy(cfg_path, str(cfg_path) + ".bak")
            cfg_path.write_text(replaced, encoding="utf-8")
            print(
                f"[cxr_vqa] 已将 config.json 中视觉塔路径改写为本地: {_XRAYSIGLIP_MODEL}",
                flush=True,
            )
    except Exception:  # noqa: BLE001
        pass


class CxrVqaTool:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or default_device()
        self._impl: Any = None

    def _load(self) -> None:
        if self._impl is not None:
            return
        local = _CHEXAGENT_MODEL
        _patch_chexagent_runtime_to_local()

        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        original_ver = transformers.__version__
        transformers.__version__ = "4.40.0"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                local,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
        finally:
            transformers.__version__ = original_ver

        model.eval()
        model.to(self.device)

        # 关键修复：确保整个模型（含视觉塔）都是 bf16
        # CheXagent 的视觉塔可能单独加载为 float32，导致 forward 时 dtype 不匹配
        model = model.to(dtype=torch.bfloat16)

        self._impl = ("hf", torch, tokenizer, model)

    def run(
        self, image_paths: list[str], query: str, max_new_tokens: int = 512, prior_report: str | None = None
    ) -> dict[str, str]:
        for p in image_paths:
            if not Path(p).is_file():
                raise FileNotFoundError(p)
        self._load()

        if self._impl[0] == "medrax":
            out, _meta = self._impl[1]._run(image_paths, query, max_new_tokens)
            if "error" in out:
                raise RuntimeError(out["error"])
            return {"answer": str(out.get("response", ""))}

        torch, tokenizer, model = self._impl[1:]

        query_fmt = tokenizer.from_list_format(
            [*[{"image": p} for p in image_paths], {"text": query}]
        )
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query_fmt},
        ]
        if prior_report is not None:
            conv += [{"from":"gpt","value":prior_report},
                     {"from":"human","value":"Re-examine the chest X-rays and revise your previous report. Return a complete Findings paragraph, including normal findings and any uncertainty."}]
        input_ids = tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt"
        ).to(device=self.device)

        with torch.inference_mode():
            output = model.generate(
                input_ids,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=True,
                max_new_tokens=max_new_tokens,
            )[0]

        answer = tokenizer.decode(output[input_ids.size(1) :], skip_special_tokens=True)
        return {"answer": answer}


if __name__ == "__main__":
    from fastmcp.server.server import FastMCP

    args = parse_mcp_args()

    mcp = FastMCP("cxr_vqa")
    tool = CxrVqaTool()
    semaphore = asyncio.Semaphore(2)

    @mcp.tool(name="cxr_vqa")
    async def cxr_vqa(image: str, query: str) -> dict:
        """Single-view CXR VQA for hypothesis verification."""
        async with semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: tool.run([image], query)
            )
        return mcp_success(**result)

    @mcp.tool(name="cxr_multiview_vqa")
    async def cxr_multiview_vqa(image_pa: str, image_lat: str, query: str) -> dict:
        """PA+LAT dual-view CXR VQA."""
        async with semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: tool.run([image_pa, image_lat], query)
            )
        return mcp_success(**result)

    @mcp.tool(name="cxr_reflect_report")
    async def cxr_reflect_report(images: list[str], prior_report: str) -> dict:
        async with semaphore:
            result=await asyncio.to_thread(tool.run, images,
                "Write the Findings section of a chest radiograph report.",512,prior_report)
        return mcp_success(**result)

    print(
        f"cxr_vqa(+multiview) MCP → http://{args.host}:{args.port}/mcp",
        flush=True,
    )
    mcp.run(transport="http", host=args.host, port=args.port, stateless_http=True)
