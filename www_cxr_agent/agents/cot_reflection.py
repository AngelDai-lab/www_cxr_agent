"""Visual CoT/reflection baseline: frozen CheXagent, no diagnostic tool planning.

Transport uses the existing VQA service. This is a VLM baseline, NOT text-only Qwen3
receiving an inaccessible image path; its backbone difference must be reported.
"""
from .server_manager import ToolServerManager

class CoTReflectionAgent:
    async def run(self, sample):
        import time
        started=time.monotonic()
        manager=ToolServerManager()
        image=sample["abs_images"][0]
        name="cxr_multiview_vqa" if sample.get("image_lat") else "cxr_vqa"
        args={"image_pa":image,"image_lat":sample["image_lat"]} if sample.get("image_lat") else {"image":image}
        try:
            first=await manager.call_tool(name,dict(args,query="Inspect the radiograph systematically. Describe supported findings and uncertainties, then draft a Findings paragraph."))
            from fastmcp import Client
            images=[image]+([sample["image_lat"]] if sample.get("image_lat") else [])
            async with Client("http://127.0.0.1:7014/mcp",timeout=180) as client:
                response=await client.call_tool("cxr_reflect_report",{"images":images,"prior_report":first["answer"]})
                second=response.structured_content["outputs"]
        finally:
            await manager.close()
        return {"id":sample["id"],"findings":second["answer"],"report_ref":sample["report"],
                "method":"chexagent_cot_reflection","trajectory":{"tool_calls":[],"evidence_cache":[],"checklist_checked":[]},
                "efficiency":{"elapsed_seconds":time.monotonic()-started},
                "baseline_trace":{"draft":first["answer"],"visual_forward_calls":2}}
