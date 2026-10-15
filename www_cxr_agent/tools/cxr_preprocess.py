"""MCP ``cxr_preprocess`` — RadAgent-style: tool body + FastMCP shell.

Body: JPG/PNG EXIF→RGB→224 PNG; DICOM prefers MedRAX ``DicomProcessorTool`` if present.
"""

from __future__ import annotations

import asyncio

from www_cxr_agent.preprocess.image import preprocess_cxr_image
from www_cxr_agent.tools.mcp_common import mcp_success, parse_mcp_args


class CxrPreprocessTool:
    def run(self, image: str) -> dict[str, str]:
        return preprocess_cxr_image(image, training=False)


if __name__ == "__main__":
    from fastmcp.server.server import FastMCP

    args = parse_mcp_args()
    mcp = FastMCP("cxr_preprocess")
    tool = CxrPreprocessTool()
    semaphore = asyncio.Semaphore(8)

    @mcp.tool(name="cxr_preprocess")
    async def cxr_preprocess(image: str) -> dict:
        """CXR preprocess: DICOM/JPG/PNG → RGB 224×224 PNG path."""
        async with semaphore:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tool.run, image)
            return mcp_success(**result)

    print(f"cxr_preprocess MCP → http://{args.host}:{args.port}/mcp", flush=True)
    mcp.run(transport="http", host=args.host, port=args.port, stateless_http=True)
