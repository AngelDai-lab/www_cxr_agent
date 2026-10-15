"""FastMCP HTTP client — 多工具池版（VQA 12 + Classifier 4 + round-robin）

池化工具（对上层透明）:
- cxr_vqa / cxr_multiview_vqa → VQA 池 (12 实例)
- cxr_patho_classifier → Classifier 池 (4 实例)

替换: www_cxr_agent/agents/server_manager.py
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from www_cxr_agent.tools.tool_configs import (
    SERVERS,
    VQA_POOL_SIZE,
    CLF_POOL_SIZE,
    vqa_pool_instance_names,
    clf_pool_instance_names,
    server_url,
)
from fastmcp.client.transports import StreamableHttpTransport as HTTPTransport

logger = logging.getLogger(__name__)

# 池化工具的逻辑名 → (池大小, 实例名前缀, 原始工具名映射)
_POOLED_TOOLS = {
    "cxr_vqa": {
        "size": VQA_POOL_SIZE,
        "prefix": "cxr_vqa_",
        "tool_name_on_server": "cxr_vqa",  # 每个实例进程上的实际 tool 名
    },
    "cxr_multiview_vqa": {
        "size": VQA_POOL_SIZE,
        "prefix": "cxr_vqa_",
        "tool_name_on_server": "cxr_multiview_vqa",
    },
    "cxr_patho_classifier": {
        "size": CLF_POOL_SIZE,
        "prefix": "cxr_patho_classifier_",
        "tool_name_on_server": "cxr_patho_classifier",
    },
}


def mcp_endpoint(tool_name: str) -> str:
    return f"{server_url(tool_name)}/mcp"


class ToolServerManager:
    """Connect to locally running FastMCP tool processes.

    池化工具自动 round-robin 分发到多个实例，对上层完全透明。
    """

    _shared_cycles = {}

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}
        # 每个池化工具族一个 round-robin 计数器
        self._rr_cycles: dict[str, itertools.cycle] = {}
        for logic_name, cfg in _POOLED_TOOLS.items():
            self._rr_cycles[logic_name] = self._shared_cycles.setdefault(cfg["prefix"], itertools.cycle(range(cfg["size"])))

    def _is_pooled(self, tool_name: str) -> bool:
        return tool_name in _POOLED_TOOLS

    def _pool_instance_name(self, tool_name: str) -> str:
        """取池化工具的下一个实例名（round-robin）。"""
        cfg = _POOLED_TOOLS[tool_name]
        idx = next(self._rr_cycles[tool_name])
        return f"{cfg['prefix']}{idx}"

    def _tool_name_on_server(self, tool_name: str) -> str:
        """逻辑名对应的实例进程上的实际 tool 名。"""
        if tool_name in _POOLED_TOOLS:
            return _POOLED_TOOLS[tool_name]["tool_name_on_server"]
        return tool_name

    async def connect(self, tool_name: str) -> None:
        """连接工具服务器。池化工具：连接所有池实例。"""
        from fastmcp import Client

        if self._is_pooled(tool_name):
            cfg = _POOLED_TOOLS[tool_name]
            pool_size = cfg["size"]
            prefix = cfg["prefix"]
            for i in range(pool_size):
                inst_name = f"{prefix}{i}"
                if inst_name in self._clients:
                    continue
                spec = SERVERS[inst_name]
                client = Client(
                    HTTPTransport(url=f"http://{spec['host']}:{spec['port']}/mcp"),
                    timeout=float(os.environ.get("CXR_TOOL_TIMEOUT", "180")),
                )
                await client._connect()
                self._clients[inst_name] = client
                logger.info("Connected MCP tool server: %s (port %d)",
                            inst_name, spec["port"])
            logger.info("%s 池全部就绪 (%d 实例)", tool_name, pool_size)
        else:
            if tool_name in self._clients:
                return
            client = Client(
                HTTPTransport(url=mcp_endpoint(tool_name)),
                timeout=float(os.environ.get("CXR_TOOL_TIMEOUT", "180")),
            )
            await client._connect()
            self._clients[tool_name] = client
            logger.info("Connected MCP tool server: %s", tool_name)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """调用工具。池化工具自动选实例 round-robin。"""
        if self._is_pooled(tool_name):
            inst_name = self._pool_instance_name(tool_name)
            if inst_name not in self._clients:
                await self.connect(tool_name)
            client = self._clients[inst_name]
            actual_tool = self._tool_name_on_server(tool_name)
            result = await client.call_tool(actual_tool, arguments)
        else:
            if tool_name not in self._clients:
                await self.connect(tool_name)
            client = self._clients[tool_name]
            result = await client.call_tool(tool_name, arguments)

        content = getattr(result, "structured_content", None) or {}
        if isinstance(content, dict) and "outputs" in content:
            return content["outputs"]
        return content

    async def close(self) -> None:
        for name, client in list(self._clients.items()):
            try:
                await client._disconnect()
            except Exception as exc:
                logger.warning("disconnect %s: %s", name, exc)
        self._clients.clear()

    def call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        return asyncio.run(self.call_tool(tool_name, arguments))


def start_tool_process(tool_name: str, *, host: str = "127.0.0.1") -> subprocess.Popen[str]:
    spec = SERVERS[tool_name]
    port = spec["port"]
    cmd = [sys.executable, "-m", spec["module"], "--host", host, "--port", str(port)]
    logger.info("Starting tool process: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def write_tool_registry(path: str | Path) -> None:
    registry = {
        name: {"url": mcp_endpoint(name), "port": spec["port"], "module": spec["module"]}
        for name, spec in SERVERS.items()
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=2), encoding="utf-8")


async def wait_for_tool(tool_name: str, *, timeout: float = 120.0, interval: float = 1.0) -> bool:
    """等待工具就绪。池化工具需要等待所有实例就绪。"""
    import aiohttp

    async def _check_port(port: int) -> bool:
        url = f"http://127.0.0.1:{port}/mcp"
        headers = {"Accept": "application/json, text/event-stream"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status in (200, 400, 406):
                        return True
        except Exception:
            pass
        return False

    if tool_name in _POOLED_TOOLS:
        cfg = _POOLED_TOOLS[tool_name]
        prefix = cfg["prefix"]
        pool_size = cfg["size"]
        ports = [SERVERS[f"{prefix}{i}"]["port"] for i in range(pool_size)]
        deadline = time.time() + timeout
        while time.time() < deadline:
            results = await asyncio.gather(*[_check_port(p) for p in ports])
            if all(results):
                return True
            await asyncio.sleep(interval)
        ready_count = sum(
            1 for i in range(pool_size)
            if SERVERS[f"{prefix}{i}"]["port"] in [
                p for p, r in zip(
                    ports,
                    await asyncio.gather(*[_check_port(p) for p in ports])
                ) if r
            ]
        )
        logger.warning("%s 池就绪超时: %d/%d 实例就绪", tool_name, ready_count, pool_size)
        return False
    else:
        spec = SERVERS[tool_name]
        return await _check_port(spec["port"])
