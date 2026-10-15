"""Reward evaluation by an independent frozen MCP model, never policy weights."""
import asyncio
import concurrent.futures
import hashlib
import json
import logging
import os
import itertools
from pathlib import Path
import threading
import uuid

logger = logging.getLogger(__name__)


def _judge_urls() -> list[str]:
    raw = os.environ.get("CXR_JUDGE_URLS")
    if not raw:
        count = max(1, int(os.environ.get("CXR_JUDGE_COUNT", "1")))
        raw = ",".join(f"http://127.0.0.1:{7007 + i}/mcp" for i in range(count))
    return [u.strip() for u in raw.split(",") if u.strip()]


_judge_cycle = itertools.cycle(_judge_urls())
# Evidence interpretation is invoked from one asyncio event loop per rollout
# worker thread.  An asyncio.Semaphore cannot safely be shared across those
# loops, so use a process-wide threading gate.  This keeps the number of
# in-flight evidence calls aligned with the frozen MCP judge pool even when
# multiple policy replicas collect trajectories concurrently.
_judge_gate = threading.BoundedSemaphore(
    max(1, int(os.environ.get("CXR_EVIDENCE_JUDGE_CONCURRENCY", os.environ.get("CXR_JUDGE_COUNT", "3"))))
)


def _next_judge_url() -> str:
    return next(_judge_cycle)

async def _score(trajectory):
    from fastmcp import Client
    url = os.environ.get("CXR_JUDGE_URL") or _next_judge_url()
    async with Client(url, timeout=120) as client:
        result = await client.call_tool("cxr_tool_judge", {"trajectory": trajectory})
        content = result.structured_content
        score = float(content.get("outputs", content)["score"])
        if not 0 <= score <= 1:
            raise ValueError("Independent judge returned invalid score")
        return score

def run_tool_judge(trajectory):
    # Reward calculation may run inside the training asyncio loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_score(trajectory))).result(timeout=150)

async def evaluate_report_async(findings, reference, trajectory):
    from fastmcp import Client
    url = os.environ.get("CXR_JUDGE_URL") or _next_judge_url()
    async with Client(url, timeout=180) as client:
        selected=[]; index_map={}
        for original, tc in enumerate(trajectory.get("tool_calls",[])):
            if tc.get("status")=="ok" and tc.get("name") in ("cxr_vqa","cxr_multiview_vqa","segment_2d"):
                index_map[original]=len(selected); selected.append(tc)
        def remap(items):
            output=[]
            for item in items:
                copy=dict(item)
                ids=copy.get("evidence_indices",[])
                copy["evidence_indices"]=[index_map[i] for i in ids if i in index_map]
                copy["evidence"]=[dict(e,tool_call_index=index_map[e["tool_call_index"]])
                                  for e in copy.get("evidence",[]) if e.get("tool_call_index") in index_map]
                output.append(copy)
            return output
        evidence_only={"tool_calls":selected,
                       "evidence_cache":remap(trajectory.get("evidence_cache",[])),
                       "checklist_checked":remap(trajectory.get("checklist_checked",[]))}
        result = await client.call_tool("cxr_report_judge", {"findings": findings, "reference": reference, "trajectory": evidence_only})
        data = result.structured_content
        assessment = data.get("outputs", data)
        # Persist citations in the original full trajectory coordinate system.
        reverse_map = {selected: original for original, selected in index_map.items()}
        for item in assessment.get("support_details", []):
            item["evidence_indices"] = [reverse_map[i] for i in item.get("evidence_indices", [])]
        assessment["citation_space"] = "trajectory.tool_calls"
    if not evidence_only["tool_calls"]:
        assessment["unsupported_count"]=assessment["candidate_count"]
        assessment["supported_finding_count"]=0
        assessment["misalign"]=1.0 if assessment["candidate_count"] else 0.0
    from www_cxr_agent.eval.metrics_clinical import report_quality
    assessment.update(await asyncio.to_thread(report_quality, findings, reference))
    return assessment


async def evaluate_report_resilient(findings, reference, trajectory):
    """Retry report-judge schema/transport errors without aborting a rollout batch."""
    last = None
    for attempt in range(3):
        try:
            return await evaluate_report_async(findings, reference, trajectory)
        except Exception as exc:
            last = exc
            logger.warning("report judge attempt %d/3 failed: %s", attempt + 1, exc)
            if attempt < 2:
                await asyncio.sleep(min(2.0 * (attempt + 1), 4.0))
    raise RuntimeError("Report judge failed after 3 attempts; refusing a fabricated reward. "
                       "Completed rollout files are retained for diagnosis.") from last

def _validate_evidence(output, tool_calls):
    from www_cxr_agent.agents.layered_agent import CHECKLIST_ITEMS
    if not isinstance(output, dict):
        raise ValueError("Evidence response must be an object")
    allowed = {i for i, tc in enumerate(tool_calls) if tc.get("status") == "ok"
               and tc.get("name") in ("cxr_vqa", "cxr_multiview_vqa", "segment_2d")}
    for key, statuses in (("findings", ("present", "absent", "normal", "uncertain", "conflict", "unverified")),
                          ("checklist", ("checked", "uncertain", "unchecked"))):
        if not isinstance(output.get(key), list):
            raise ValueError("Evidence response missing " + key)
        for item in output[key]:
            if not isinstance(item, dict) or item.get("status") not in statuses:
                raise ValueError(f"Invalid evidence status in {key}: {item!r}")
            ids = item.get("evidence_indices", [])
            if not isinstance(ids, list) or any(type(i) is not int or i not in allowed for i in ids):
                raise ValueError("Invalid evidence citations")
            if item["status"] in ("present", "absent", "normal", "conflict", "checked") and not ids:
                raise ValueError("Unsupported evidence status")
    names = [item.get("name") for item in output["checklist"]]
    if len(names) != len(CHECKLIST_ITEMS) or set(names) != set(CHECKLIST_ITEMS):
        raise ValueError("Evidence checklist must contain all seven unique items")


async def interpret_evidence(tool_calls, hypotheses):
    from fastmcp import Client
    # Retry schema and transport failures. Never accept invented indices or fabricate
    # checked status; the service receives explicit validation feedback.
    # Evidence is a pure function of successful tool observations and the
    # hypotheses. Persisting it avoids re-running the frozen judge when a
    # rollout is resumed or a repeated observation is encountered.
    cache_root = Path(os.environ.get("CXR_EVIDENCE_CACHE_DIR", os.environ.get("CXR_ROLLOUT_CACHE_DIR", "runs/rollout_cache"))) / "evidence"
    payload = json.dumps({"tool_calls": tool_calls, "hypotheses": hypotheses}, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    cache_path = cache_root / ("evidence_" + hashlib.sha256(payload).hexdigest() + ".json")
    try:
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            _validate_evidence(cached, tool_calls)
            logger.info("evidence judge cache hit key=%s", cache_path.stem[-12:])
            return cached
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("evidence judge cache read failed: %s", exc)
    fixed_url = os.environ.get("CXR_JUDGE_URL")
    # Acquire without blocking an event loop thread.  The semaphore is
    # process-wide because each rollout owns a separate asyncio loop.
    await asyncio.to_thread(_judge_gate.acquire)
    try:
        feedback = ""
        for attempt in range(3):
            url = fixed_url or _next_judge_url()
            try:
                # Reconnect for every attempt. Connection/setup/teardown
                # failures can happen outside call_tool's await and must be
                # retried just like MCP tool errors.
                async with Client(url, timeout=900) as client:
                    result = await client.call_tool("cxr_evidence_judge", {
                        "tool_calls": tool_calls, "hypotheses": hypotheses,
                        "validation_feedback": feedback})
                    data = result.structured_content
                    output = data.get("outputs", data)
                    _validate_evidence(output, tool_calls)
                try:
                    cache_root.mkdir(parents=True, exist_ok=True)
                    tmp = cache_path.with_suffix("." + uuid.uuid4().hex + ".tmp")
                    tmp.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
                    tmp.replace(cache_path)
                except OSError as cache_exc:
                    logger.warning("evidence judge cache write failed: %s", cache_exc)
                return output
            except Exception as exc:
                # Includes FastMCP/MCP errors and transport-level httpx
                # ConnectError/RemoteProtocolError raised by async context
                # setup or teardown. The final attempt remains conservative.
                logger.warning("evidence judge attempt %d/3 failed url=%s: %s", attempt + 1, url, exc)
                if attempt == 2:
                    return {
                        "findings": [{"finding": str(h.get("pathology", "unknown")),
                                      "status": "unverified", "location": "unknown",
                                      "evidence_indices": [],
                                      "rationale": f"evidence judge unavailable: {exc}"}
                                     for h in hypotheses],
                        "checklist": [{"name": name, "status": "unchecked",
                                       "evidence": [], "notes": "judge unavailable"}
                                      for name in ("trachea_mediastinum", "lung_fields", "pleura", "heart", "diaphragm", "bones", "devices")],
                        "assessment_version": "timeout_conservative_v1",
                    }
                feedback = str(exc)[-1000:]
                await asyncio.sleep(min(2.0 * (attempt + 1), 4.0))
    finally:
        _judge_gate.release()
