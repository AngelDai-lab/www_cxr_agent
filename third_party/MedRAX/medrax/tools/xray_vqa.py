"""CheXagent 4bit 量化版 XRayVQATool —— 替换 /medrax/tools/xray_vqa.py 的同名文件即可。

关键改造（相对原版 bf16 版）：
  1. from_pretrained 时加 BitsAndBytesConfig(load_in_4bit=True, nf4, bf16)，把整模型压到 4bit
  2. 本服务器 bitsandbytes **旧版不支持 skip_modules**（categorical），所以加载完后
     手动把「视觉塔」里所有 Linear4bit 反量化回普通 bf16 nn.Linear，恢复形状、统一 dtype
  3. 注入「输入跟随权重」对齐补丁（conv2d/linear/layer_norm/matmul/bmm/einsum），根治视觉塔 dtype 混战
  4. 语言骨架保持 4bit，最终显存 ≈3.9G，可与 Qwen3(10G) 在 24G 单卡共存

用法不变：XRayVQATool(model_name=..., device="cuda", dtype=torch.bfloat16)
7004 MCP 服务照常初始化该工具即可。
"""
from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
from pydantic import BaseModel, Field
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


def _dequant_visual_back_to_bf16(model):
    """加载完整 4bit 后，把视觉塔所有 Linear4bit 反量化回普通 bf16 nn.Linear。

    本服务器 bnb 旧版不支持 BitsAndBytesConfig 的 skip_modules（会被静默忽略，
    视觉塔会被一并量化并把 Linear 权重压成 (1, N) 外形，导致 forward 报
    'mat1 and mat2 shapes cannot be multiplied'）。这条手动反量化路径绕开该缺陷：
    语言骨架保持 4bit，只有视觉塔恢复为全精度，足以保证形状正确 + dtype 统一。
    """
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit as BnbLinear4bit
    import torch.nn as nn

    # 1) 定位视觉塔根：完整模块路径是 model.visual（即 model.model.visual）
    vis_root, vis_prefix = None, None
    for n, mod in model.named_modules():
        if n == "model.visual":
            vis_root, vis_prefix = mod, n
            break
    if vis_root is None:
        for n, mod in model.named_modules():
            if "visual" in n.split(".") and any(
                isinstance(c, BnbLinear4bit) for c in mod.modules()
            ):
                vis_root, vis_prefix = mod, n
                break
    if vis_root is None:
        print("[xray_vqa] warn: 未找到视觉塔根，跳过反量化（可能非4bit加载）", flush=True)
        return 0

    # 2) 逐层替换 Linear4bit -> nn.Linear(bf16)
    replaced = 0

    def walk(parent, prefix):
        nonlocal replaced
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}"
            if isinstance(child, BnbLinear4bit):
                w = child.weight
                qs = getattr(w, "quant_state", None)
                dq = (
                    child.weight.data
                    if qs is None
                    else bnb.functional.dequantize_4bit(child.weight.data, qs)
                )
                if dq.numel() != child.in_features * child.out_features:
                    print(f"[xray_vqa] warn: {path} 反量化尺寸异常跳过", flush=True)
                    continue
                new = nn.Linear(
                    child.in_features, child.out_features,
                    bias=(child.bias is not None),
                )
                new.weight.data = dq.reshape(new.weight.shape).to(torch.bfloat16)
                new = new.to("cuda")
                if child.bias is not None:
                    new.bias.data = child.bias.detach().to(torch.bfloat16)
                setattr(parent, name, new)
                replaced += 1
            else:
                walk(child, path)

    walk(vis_root, vis_prefix)
    vis_root.to(dtype=torch.bfloat16)
    print(f"[xray_vqa] 视觉塔反量化完成：{replaced} 个 Linear4bit -> bf16 nn.Linear", flush=True)
    return replaced


def _install_dtype_alignment_patch():
    """全局注入「输入跟随权重」补丁，根治视觉塔(fp32原生)与量化层(bf16)在 conv/linear/norm/matmul 上的 dtype 混战。

    与探针脚本保持一致的实现，避免 medrax 内部自行运算时再次踩 dtype 坑。
    """
    import torch.nn.functional as F
    _oc = F.conv2d; _ol = F.linear; _om = torch.matmul
    _ob = torch.bmm; _oe = torch.einsum
    _olr = getattr(torch, "layer_norm", None)

    def _conv2d(i, w, b=None, stride=1, padding=0, dilation=1, groups=1):
        if i.dtype != w.dtype:
            i = i.to(w.dtype)
        if b is not None and b.dtype != i.dtype:
            b = b.to(i.dtype)
        return _oc(i, w, b, stride, padding, dilation, groups)

    def _linear(i, w, b=None):
        if i.dtype != w.dtype:
            i = i.to(w.dtype)
        if b is not None and b.dtype != i.dtype:
            b = b.to(i.dtype)
        return _ol(i, w, b)

    def _ln(inp, norm_shape, weight=None, bias=None, eps=1e-5, cudnn_enabled=True):
        target = None
        if weight is not None and weight.dtype in (torch.float16, torch.bfloat16):
            target = weight.dtype
        if target is not None and inp.dtype != target:
            inp = inp.to(target)
        return _olr(inp, norm_shape, weight, bias, eps, cudnn_enabled)

    def _matmul(a, b):
        if a.dtype != b.dtype:
            dt = a.dtype if a.dtype in (torch.float16, torch.bfloat16) else b.dtype
            a, b = a.to(dt), b.to(dt)
        return _om(a, b)

    def _bmm(a, b):
        if a.dtype != b.dtype:
            dt = a.dtype if a.dtype in (torch.float16, torch.bfloat16) else b.dtype
            a, b = a.to(dt), b.to(dt)
        return _ob(a, b)

    def _einsum(eq, *ops):
        ops = list(ops)
        fp = [t.dtype for t in ops if t.dtype in (torch.float16, torch.bfloat16, torch.float32)]
        if fp and len({str(t.dtype) for t in ops}) > 1:
            dt = fp[0]
            ops = [t.to(dt) if t.dtype != dt else t for t in ops]
        return _oe(eq, *ops)

    F.conv2d = _conv2d
    F.linear = _linear
    torch.matmul = _matmul
    torch.bmm = _bmm
    torch.einsum = _einsum
    if _olr is not None:
        torch.layer_norm = _ln
    print("[xray_vqa] 已注入 输入跟随权重 对齐补丁", flush=True)


class XRayVQAToolInput(BaseModel):
    """Input schema for the CheXagent Tool."""
    image_paths: List[str] = Field(
        ...,
        description="List of paths to chest X-ray images to analyze",
    )
    prompt: str = Field(
        ..., description="Question or instruction about the chest X-ray images"
    )
    max_new_tokens: int = Field(
        512, description="Maximum number of tokens to generate in the response"
    )


class XRayVQATool(BaseTool):
    """Tool that leverages CheXagent for comprehensive chest X-ray analysis (4bit quantized)."""

    name: str = "chest_xray_expert"
    description: str = (
        "A versatile tool for analyzing chest X-rays. "
        "Can perform multiple tasks including: visual question answering, report generation, "
        "abnormality detection, comparative analysis, anatomical description, "
        "and clinical interpretation. Input should be paths to X-ray images "
        "and a natural language prompt describing the analysis needed."
    )
    args_schema: Type[BaseModel] = XRayVQAToolInput
    return_direct: bool = True
    cache_dir: Optional[str] = None
    device: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16
    tokenizer: Optional[AutoTokenizer] = None
    model: Optional[AutoModelForCausalLM] = None

    def __init__(
        self,
        model_name: str = "StanfordAIMI/CheXagent-2-3b",
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the XRayVQATool (4bit quantization + visual tower keep bf16)."""
        super().__init__(**kwargs)
        import transformers

        original_transformers_version = transformers.__version__
        transformers.__version__ = "4.40.0"

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir,
        )

        # ---- 4bit 量化加载 ----
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # 与 .to(bf16) 对齐
            bnb_4bit_use_double_quant=True,
            # 注意：本服务器 bnb 旧版不支持 skip_modules(会被忽略)，故不加，
            # 改为加载后手动反量化视觉塔
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=self.device,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            quantization_config=quant_config,
        )
        # ---- 视觉塔反量化回 bf16（关键修复旧 bnb 无法 skip 视觉塔）----
        _dequant_visual_back_to_bf16(self.model)
        _install_dtype_alignment_patch()
        # 注意：不做整体 self.model.to(dtype)，否则会把 bnb 量化语言层也 cast 破坏 4bit。
        # 视觉塔已在反量化函数内统一为 bf16，语言骨架保持 4bit，dtype 由对齐补丁兜底。
        self.model.eval()

        transformers.__version__ = original_transformers_version

    def _generate_response(
        self, image_paths: List[str], prompt: str, max_new_tokens: int
    ) -> str:
        """Generate response using CheXagent model."""
        query = self.tokenizer.from_list_format(
            [*[{"image": path} for path in image_paths], {"text": prompt}]
        )
        conv = [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": query},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, return_tensors="pt",
        ).to(device=self.device)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                do_sample=False, num_beams=1, temperature=1.0, top_p=1.0,
                use_cache=True, max_new_tokens=max_new_tokens,
            )[0]
        response = self.tokenizer.decode(output[input_ids.size(1): -1])
        return response

    def _run(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Execute the chest X-ray analysis."""
        try:
            for path in image_paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Image file not found: {path}")
            response = self._generate_response(image_paths, prompt, max_new_tokens)
            output = {"response": response}
            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "completed",
            }
            return output, metadata
        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "image_paths": image_paths,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "analysis_status": "failed",
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_paths, prompt, max_new_tokens)