"""Shared helpers for CXR FastMCP tool servers."""

from __future__ import annotations

import argparse
from typing import Any


def parse_mcp_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CXR FastMCP tool server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7001)
    return parser.parse_args()


def mcp_success(**outputs: Any) -> dict[str, Any]:
    """RadAgent-compatible MCP payload."""
    return {"outputs": outputs}
