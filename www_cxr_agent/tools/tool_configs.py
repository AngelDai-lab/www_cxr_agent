"""MCP registry: three VQA instances on GPUs 1/2/3, policy on GPU0.
Independent frozen judge shares GPU3; CPU/lightweight services are launch-configured.
Use launch/start_all_tools.sh for the current four-GPU deployment.
"""

from __future__ import annotations

import os
from typing import Any, TypedDict


class ToolServerSpec(TypedDict):
    name: str
    port: int
    host: str
    module: str
    description: str
    inputs: list[str]
    outputs: list[str]
    model_paths: dict[str, str]


HOST = "127.0.0.1"

# ---------- VQA 池配置 ----------
VQA_POOL_SIZE = int(os.environ.get("CXR_VQA_COUNT", "3"))
VQA_BASE_PORT = 7004
VQA_PORT_STEP = 10


def vqa_ports() -> list[int]:
    return [VQA_BASE_PORT + i * VQA_PORT_STEP for i in range(VQA_POOL_SIZE)]


def vqa_pool_instance_names() -> list[str]:
    return [f"cxr_vqa_{i}" for i in range(VQA_POOL_SIZE)]


# ---------- Patho Classifier 池配置 ----------
CLF_POOL_SIZE = int(os.environ.get("CXR_CLF_COUNT", "1"))
CLF_BASE_PORT = 7003
CLF_PORT_STEP = 100


def clf_ports() -> list[int]:
    return [CLF_BASE_PORT + i * CLF_PORT_STEP for i in range(CLF_POOL_SIZE)]


def clf_pool_instance_names() -> list[str]:
    return [f"cxr_patho_classifier_{i}" for i in range(CLF_POOL_SIZE)]


# ---------- 服务器注册 ----------
SERVERS: dict[str, ToolServerSpec] = {
    "cxr_preprocess": {
        "name": "cxr_preprocess",
        "port": 7001,
        "host": HOST,
        "module": "www_cxr_agent.tools.cxr_preprocess",
        "description": (
            "Preprocess CXR (DICOM/JPG/PNG) to a standard RGB 224x224 PNG path for downstream tools."
        ),
        "inputs": ["image"],
        "outputs": ["image"],
        "model_paths": {},
    },
    "cxr_report_draft": {
        "name": "cxr_report_draft",
        "port": 7002,
        "host": HOST,
        "module": "www_cxr_agent.tools.cxr_report_draft",
        "description": (
            "Generate a draft Findings+Impression report as REFERENCE ONLY; "
            "never use directly as the final Agent report."
        ),
        "inputs": ["image"],
        "outputs": ["text"],
        "model_paths": {
            "findings": "models/chexpert-mimic-cxr-findings-baseline",
            "impression": "models/chexpert-mimic-cxr-impression-baseline",
        },
    },
    # ---- Patho Classifier 池：2 实例 ----
    **{
        f"cxr_patho_classifier_{i}": {
            "name": f"cxr_patho_classifier_{i}",
            "port": CLF_BASE_PORT + i * CLF_PORT_STEP,
            "host": HOST,
            "module": "www_cxr_agent.tools.cxr_patho_classifier",
            "description": f"Pathology classifier pool instance #{i} (round-robin, internal)",
            "inputs": ["image"],
            "outputs": ["scores"],
            "model_paths": {"torchxrayvision": "cache/torchxrayvision/densenet121-res224-all"},
        }
        for i in range(CLF_POOL_SIZE)
    },
    # ---- VQA 池：4 实例 ----
    **{
        f"cxr_vqa_{i}": {
            "name": f"cxr_vqa_{i}",
            "port": VQA_BASE_PORT + i * VQA_PORT_STEP,
            "host": HOST,
            "module": "www_cxr_agent.tools.cxr_vqa",
            "description": f"CXR VQA pool instance #{i} (round-robin, internal)",
            "inputs": ["image", "query"],
            "outputs": ["answer"],
            "model_paths": {"chexagent": "models/CheXagent-2-3b"},
        }
        for i in range(VQA_POOL_SIZE)
    },
    "segment_2d": {
        "name": "segment_2d",
        "port": 7006,
        "host": HOST,
        "module": "www_cxr_agent.tools.segment_2d",
        "description": (
            "2D chest anatomical segmentation; optional organs list; returns mask PNG path and metrics."
        ),
        "inputs": ["image"],
        "outputs": ["mask", "metrics"],
        "model_paths": {"torchxrayvision": "cache/torchxrayvision/chestx_det"},
    },
    # Independent frozen judge service, launched on GPU3 by default.
    "cxr_negation_checker": {
        "name": "cxr_negation_checker",
        "port": 7007,
        "host": HOST,
        "module": "www_cxr_agent.tools.llm_judge_tool",
        "description": (
            "Check whether a candidate sentence is consistent with the evidence set "
            "(negation / contradiction judge). Returns score 0|1 and rationale. "
            "Started by `launch/start_all_tools.sh start` on GPU3 by default."
        ),
        "inputs": ["candidate_sentence", "evidence_set"],
        "outputs": ["score", "rationale"],
        "model_paths": {"llm": "models/Qwen3-14B-Instruct"},
    },
}

# ---------- 对策略模型暴露的工具列表（只暴露逻辑名） ----------
ALL_TOOL_CONFIGS: list[dict[str, Any]] = [
    {
        "name": spec["name"],
        "description": spec["description"],
        "parameter": {
            k: {"type": "string", "description": k} for k in spec["inputs"]
        },
    }
    for spec in SERVERS.values()
    if not spec["name"].startswith("cxr_vqa_")
    and not spec["name"].startswith("cxr_patho_classifier_")
]
ALL_TOOL_CONFIGS += [
    {
        "name": "cxr_patho_classifier",
        "description": "Pathology screening classifier: returns probabilities for standard CXR pathologies to build the hypothesis set H.",
        "parameter": {"image": {"type": "string", "description": "image"}},
    },
    {
        "name": "cxr_vqa",
        "description": "Single-view CXR visual question answering for local hypothesis verification.",
        "parameter": {
            "image": {"type": "string", "description": "image"},
            "query": {"type": "string", "description": "query"},
        },
    },
    {
        "name": "cxr_multiview_vqa",
        "description": "PA+LAT dual-view CXR VQA for joint hypothesis verification.",
        "parameter": {
            "image_pa": {"type": "string", "description": "image_pa"},
            "image_lat": {"type": "string", "description": "image_lat"},
            "query": {"type": "string", "description": "query"},
        },
    },
]


def server_url(tool_name: str) -> str:
    spec = SERVERS[tool_name]
    return f"http://{spec['host']}:{spec['port']}"


def unique_launch_specs() -> list[ToolServerSpec]:
    seen: set[tuple[str, int]] = set()
    out: list[ToolServerSpec] = []
    for spec in SERVERS.values():
        key = (spec["module"], spec["port"])
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out
