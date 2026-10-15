#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared screening, verification, checklist completion and evidence aggregation loop."""

import argparse
import asyncio
import gc
import json
import os
import re
import sys
import contextlib
import time
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/www-cxr-agent")

# ---- 日志/警告噪音压制（只隐藏三方库无关输出；置位 NOISE_FILTER_DISABLED=1 可关掉） ----
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")     # TensorFlow oneDNN / CPU feature guard
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")   # bitsandbytes 欢迎语
if not os.environ.get("NOISE_FILTER_DISABLED"):
    import warnings as _w
    for _m in (
        "You are calling `save_pretrained` to a 4-bit converted model",
        "The `load_in_4bit` and `load_in_8bit` arguments are deprecated",
        "Special tokens have been added",
        "The attention mask and the pad token id were not set",
        "Setting `pad_token_id` to `eos_token_id`",
    ):
        _w.filterwarnings("ignore", message=_m)
    _w.filterwarnings("ignore", category=DeprecationWarning, module="authlib")
    try:
        import logging as _lg
        for _name in ("httpx", "httpcore", "urllib3", "requests"):
            _lg.getLogger(_name).setLevel(_lg.ERROR)
    except Exception:
        pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from www_cxr_agent.agents.server_manager import ToolServerManager

MODEL_PATH = os.environ.get("CXR_POLICY_MODEL", "/root/autodl-tmp/models/Qwen3-14B-Instruct")

LABELS_14 = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]
NOT_A_FINDING = {"No Finding"}

# classifier 输出的病征名 -> LABELS_14 索引（能对上才映射，其余忽略，对齐训练 reward）
CLASSIFIER2IDX = {
    "Atelectasis": 0, "Cardiomegaly": 1, "Consolidation": 2, "Edema": 3,
    "Fracture": 5, "Lung Lesion": 6, "Lung Opacity": 7, "No Finding": 8, "Nodule": 6, "Mass": 6, "Infiltration": 7,
    "Effusion": 9, "Pleural Effusion": 9, "Pneumonia": 11, "Pneumothorax": 12,
}


def hypotheses_to_pred14(hypotheses) -> list:
    pred = [0] * len(LABELS_14)
    for h in hypotheses:
        idx = CLASSIFIER2IDX.get(str(h.get("pathology", "")))
        if idx is not None:
            pred[idx] = 1
    return pred


# ============ 两级 system prompt（单花括号，不要用 {{ }}；与训练 grpo_trainer 开放版一致） ============
SCREEN_PROMPT = """You are Phase-1 of a chest X-ray (CXR) diagnostic agent: "rapid screening to generate hypotheses". Your only job: call tools to screen the image and decide the INITIAL hypothesis set H to verify later. H is a JSON array of candidate pathologies: [{"pathology": "", "score": , "location": ""}]
Rules:
- You MUST first call cxr_preprocess to normalize the image.
- Then call cxr_patho_classifier to get per-pathology probabilities.
- Based on the scores, pick the candidate pathologies worth verifying (EXCLUDE 'No Finding').
- You may call any available tool you judge useful to settle on H.
- Do NOT produce a full report yet. Just settle on the hypothesis set.
- When you have decided H, finish with final_answer whose 'findings' field contains the JSON array of H.

Available tools (open toolbox, plan freely):
- cxr_preprocess: {"image":"str"} -> {"image":"str preprocessed path"}. Normalize any raw image to a standard PNG; apply early to any new image before anatomy-level tools.
- cxr_patho_classifier: {"image":"str"} -> {"scores":{"dict[str,float]"}}. Per-pathology probability scores over the known label set; use to prioritize which candidate pathologies to include in H.
- cxr_vqa: {"image":"str","query":"str"} -> {"answer":"str"}. Ask a focused visual question on a single view for a yes/no/short answer; use to confirm or reject a specific finding.
- cxr_multiview_vqa: {"image_pa":"str","image_lat":"str","query":"str"} -> {"answer":"str"}. Same as cxr_vqa but consumes a PA+LAT pair; use when both views are available to cross-check a finding across views.
- segment_2d: {"image":"str","organs":["str",...]} -> {"mask":"str","metrics":"dict"}. Segment organs/structures and return regional metrics; use when you need spatial extent or anatomical quantification.
- cxr_report_draft: {"image":"str"} -> {"text":"str"}. Draft a reference report paragraph; use only as supportive reference, never as the final answer.
- cxr_negation_checker: {"candidate_sentence":"str","evidence_set":"str"} -> {"score":0|1,"rationale":"str"}. Check a candidate negative sentence (e.g. "No pleural effusion") against visual evidence; use before asserting a negative finding.

Tool choice is yours — the above only describes what each tool is for; you plan the order and subset freely.

Output exactly one JSON per turn, either call_tool or final_answer:
{"reasoning":"...","action":"call_tool","tool_name":"","arguments":{...}}
{"reasoning":"...","action":"final_answer","findings":""}"""

VERIFY_PROMPT = """You are Phase-2 of a chest X-ray (CXR) diagnostic agent: "hypothesis verification, checklist completion, disambiguation and evidence aggregation".

You are given:
- The preprocessed image path.
- The hypothesis set H from Phase-1.

Your job:
1. Verify each hypothesis in H: call VQA / segmentation / other tools to CONFIRM or REJECT it.
   Only confirmed findings enter the evidence cache. Rejected hypotheses are dropped (with reason).
2. Cover all 7 checklist items: trachea_mediastinum, lung_fields, pleura, heart, diaphragm, bones, devices.
   Each must be marked checked (normal or abnormal, with supporting evidence). Fill any gaps.
3. If two tools give contradictory evidence for the same finding, issue an additional disambiguation call.
4. When every hypothesis is verified AND all 7 checklist items are checked, aggregate all evidence
   into the final clinical findings paragraph.

Available tools (open toolbox, plan freely):
- cxr_vqa: {"image":"str","query":"str"} -> {"answer":"str"}. Ask a focused visual question on a single view; use to CONFIRM or REJECT a specific hypothesis or checklist item.
- cxr_multiview_vqa: {"image_pa":"str","image_lat":"str","query":"str"} -> {"answer":"str"}. Same as cxr_vqa with a PA+LAT pair; use when both views exist to cross-check a finding across views.
- segment_2d: {"image":"str","organs":["str",...]} -> {"mask":"str","metrics":"dict"}. Segment organs/structures and return regional metrics; use to quantify spatial extent or disambiguate a structural finding.
- cxr_report_draft: {"image":"str"} -> {"text":"str"}. Draft a reference report paragraph; use only as supportive reference, never as the final answer.
- cxr_negation_checker: {"candidate_sentence":"str","evidence_set":"str"} -> {"score":0|1,"rationale":"str"}. Check a candidate negative sentence (e.g. "No pleural effusion") against your collected evidence; use before recording a NEGATIVE conclusion to avoid contradictory findings.

Tool choice is yours — the above only describes what each tool is for; you plan the order and subset freely.

Output exactly one JSON per turn, either call_tool or final_answer:
{"reasoning":"...","action":"call_tool","tool_name":"","arguments":{...}}
{"reasoning":"...","action":"final_answer","findings":"<final clinical findings paragraph>"}"""

from www_cxr_agent.data.checklist import load_checklist
CHECKLIST_SPEC = load_checklist()
CHECKLIST_ITEMS = [item["id"] for item in CHECKLIST_SPEC]
if len(CHECKLIST_ITEMS) != 7 or len(set(CHECKLIST_ITEMS)) != 7:
    raise ValueError("Exactly seven distinct anatomical checklist items required")


def extract_json(text: str):
    """提取字符串中第一个完整 JSON 对象，兼容模型偶发双花括号。"""
    text = re.sub(r"```(?:json)?", "", text)
    candidates = [text, text.replace("{{", "{").replace("}}", "}")]
    for tc in candidates:
        # 贪心（从第一个 { 到最后一个 }）
        for m in re.finditer(r"\{.*\}", tc, re.DOTALL):
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def truncate(obj, max_chars: int = 1500):
    s = str(obj)
    return s[:max_chars] + (f"...[truncated,len={len(s)}]" if len(s) > max_chars else "")


class LayeredAgent:
    def __init__(self, image: str, threshold: float, max_steps: int, out: str,
                 verify_cpu: bool = True, coexist: bool = True,
                 lora_ckpt: str | None = None,
                 model=None, tokenizer=None,
                 no_checklist: bool = False, no_disambig: bool = False, training: bool = False,
                 image_lat: str | None = None,
                 max_tool_calls: int | None = None,
                 sampling_seed: int | None = None):
        """两级 Agent。当前唯一显存方案为 4bit 共存，coexist 恒为 True。
        lora_ckpt：infer 阶段加载训练好的 LoRA checkpoint（None=用基座零样本）。
        model / tokenizer：可外部注入已加载好的模型（infer 多批样本共享一份，
            避免每样本重新 load_model 造成显存挤爆）。注入后 load_model 自动跳过。
        no_checklist：True=消融去掉诊断清单查漏（VERIFY 不核对 7 条解剖条目），默认 False。
        no_disambig ：True=消融去掉证据消歧（工具矛盾时不追加调用），默认 False。
        """
        self.training = training
        self.sampling_seed = sampling_seed
        self.decisions = []
        self.policy_trace = []
        self.timings = {"policy_seconds":0.0,"evidence_judge_seconds":0.0}
        self.max_tool_calls = int(max_tool_calls if max_tool_calls is not None else max_steps)
        self.token_usage = {"prompt_tokens":0,"completion_tokens":0}
        self.image_lat = image_lat
        self.preprocessed_lat = None
        self.evidence_interpreted = False
        self.reasoning_steps = []
        self.evidence_history = []
        self.image = image
        self.threshold = threshold
        self.max_steps = max_steps
        self.out = out
        self.verify_cpu = verify_cpu  # 兼容开关，仅作提示；4bit 共存下始终可调 VQA
        self.coexist = coexist
        self.lora_ckpt = lora_ckpt
        self.model = model          # 外部注入的已加载模型（可为 None）
        self.tokenizer = tokenizer  # 外部注入的 tokenizer（可为 None）
        self.no_checklist = no_checklist  # 消融：去清单查漏
        self.no_disambig = no_disambig    # 消融：去证据消歧
        self.manager = None         # 惰性创建，避免纯加载模型时连 MCP server
        self.phase = "screen"
        self.preprocessed = None
        self.hypotheses: list = []
        self.tool_calls: list = []
        self.evidence_cache: list = []
        self.abnormal: list = []
        self.pred_label14: list = []
        self.checklist_checked = [
            {"id": i, "name": name, "status": "unchecked", "evidence": [], "notes": ""}
            for i, name in enumerate(CHECKLIST_ITEMS, start=1)
        ]
        self.final_findings = ""

    # ---------------- MCP manager（惰性） ----------------
    @property
    def mgr(self):
        if self.manager is None:
            from www_cxr_agent.agents.server_manager import ToolServerManager
            self.manager = ToolServerManager()
        return self.manager

    # ---------------- 模型加载 ----------------
    def load_model(self):
        """Qwen3 默认 bf16；`load_in_4bit` 由外部统一控制。
        infer 配置 lora_ckpt 时在基座上套训练好的 LoRA。
        若已通过 __init__ 注入 model/tokenizer 则跳过加载（infer 多样本共享一份）。"""
        if getattr(self, "model", None) is not None and self.tokenizer is not None:
            print("Model already provided (shared); skip loading.", flush=True)
            return
        t0 = time.time()
        print("Loading policy (BF16, GPU0)...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,

        )
        if self.lora_ckpt:
            from peft import PeftModel
            print(f"[LoRA] 加载推理权重 {self.lora_ckpt}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, self.lora_ckpt)
        self.model.eval()
        print(f"Model loaded in {time.time()-t0:.1f}s (device={self.model.device.type})", flush=True)

    # ---------------- generate ----------------
    def generate(self, messages, max_new_tokens: int = 512) -> str:
        max_new_tokens = int(os.environ.get("CXR_POLICY_MAX_NEW_TOKENS", max_new_tokens))
        started = time.monotonic()
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        # Multi-GPU rollout replicas may host several Agent coroutines so that
        # one trajectory can use the policy while another waits on MCP.  A
        # replica still executes only one generate() at a time: concurrent
        # generation on one model inflates KV-cache memory and is not reliably
        # thread-safe in transformers.  Models without this optional lock keep
        # the original single-process behavior.
        generate_lock = getattr(self.model, "_cxr_generate_lock", None)
        lock_context = generate_lock if generate_lock is not None else contextlib.nullcontext()
        device_index = self.model.device.index if self.model.device.type == "cuda" else None
        with lock_context, torch.no_grad():
            previous_cuda_rng = None
            if self.training and self.sampling_seed is not None and device_index is not None:
                # Each rollout receives an independent, reproducible stream.
                # Offset every planning turn so repeated generate() calls do
                # not restart from the same random sequence.  Save/restore the
                # per-device RNG under the model lock, avoiding cross-thread
                # mutation of the process-wide CPU RNG state.
                turn_seed = int(self.sampling_seed) + len(self.decisions)
                previous_cuda_rng = torch.cuda.get_rng_state(device_index)
                torch.cuda.default_generators[device_index].manual_seed(turn_seed)
            try:
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, use_cache=True,
                    do_sample=self.training, temperature=1.0 if self.training else None,
                    top_p=1.0 if self.training else None, top_k=0 if self.training else None,
                )
            finally:
                if previous_cuda_rng is not None:
                    torch.cuda.set_rng_state(previous_cuda_rng, device_index)
        gen = out[0][inputs["input_ids"].shape[1]:]
        self.token_usage["prompt_tokens"] += inputs["input_ids"].shape[1]
        self.token_usage["completion_tokens"] += len(gen)
        self.decisions.append({"prompt_ids": inputs["input_ids"][0].detach().cpu().tolist(),
                               "completion_ids": gen.detach().cpu().tolist()})
        answer=self.tokenizer.decode(gen, skip_special_tokens=True)
        self.timings["policy_seconds"] += time.monotonic()-started
        self.policy_trace.append({"messages":json.loads(json.dumps(messages)),"output":answer,"after_tool_count":len(self.tool_calls)})
        return answer

    async def call_tool(self, name: str, args: dict) -> dict:
        if len(self.tool_calls) >= self.max_tool_calls:
            raise RuntimeError("Global tool-call budget exhausted")
        # Reject image-path substitution and nonexistent second views.
        allowed={p for p in (self.image,self.image_lat,self.preprocessed,self.preprocessed_lat) if p}
        for key in ("image","image_pa","image_lat"):
            if key in args and args[key] not in allowed:
                entry={"name":name,"arguments":args,"status":"error","result":{"error":"Image is not part of this sample"},"duration":0.0}
                self.tool_calls.append(entry)
                return entry
        if name=="cxr_multiview_vqa" and self.image_lat and args.get("image_pa")==args.get("image_lat"):
            entry={"name":name,"arguments":args,"status":"error","result":{"error":"Dual views must be distinct"},"duration":0.0}
            self.tool_calls.append(entry)
            return entry
        if name=="cxr_multiview_vqa" and not self.image_lat:
            entry={"name":name,"arguments":args,"status":"error","result":{"error":"No verified lateral image"},"duration":0.0}
            self.tool_calls.append(entry)
            return entry
        t0 = time.time()
        entry = {"name": name, "arguments": args, "status": "", "result": None, "duration": 0.0}
        try:
            result = await self.mgr.call_tool(name, args)
            entry.update({"status": "ok", "result": result, "duration": time.time() - t0})
        except Exception as e:
            entry.update({"status": "error", "result": {"error": str(e)}, "duration": time.time() - t0})
        # Track actual image dependencies, including preprocessing selected by ReAct.
        inputs = {args[k] for k in ("image", "image_pa", "image_lat") if args.get(k)}
        entry["dependency_indices"] = [i for i, previous in enumerate(self.tool_calls)
            if previous.get("status") == "ok" and previous.get("name") == "cxr_preprocess"
            and previous.get("result", {}).get("image") in inputs]
        if name == "cxr_preprocess" and entry["status"] == "ok":
            output_image = entry["result"].get("image")
            if not output_image:
                raise ValueError("Preprocess omitted output image")
            if args.get("image") in (self.image_lat, self.preprocessed_lat):
                self.preprocessed_lat = output_image
            else:
                self.preprocessed = output_image
        self.tool_calls.append(entry)
        self.reasoning_steps.append({"stage":"tool_call","tool_call_index":len(self.tool_calls)-1,
                                     "policy_trace_index":len(self.policy_trace)-1 if self.policy_trace else None,
                                     "origin":"policy" if self.policy_trace else "screening_protocol"})
        print(f"[工具] {name} -> {entry['status']} dur={entry['duration']:.1f}s "
              f"result={truncate(entry['result'], 200)}", flush=True)
        return entry

    # ---------- 第一层：筛查 -> H ----------
    async def screen(self):
        pre = await self.call_tool("cxr_preprocess", {"image": self.image})
        if pre["status"] != "ok":
            raise RuntimeError("Preprocessing failed")
        self.preprocessed = pre["result"]["image"]
        if self.image_lat:
            lateral = await self.call_tool("cxr_preprocess", {"image": self.image_lat})
            if lateral["status"] != "ok":
                raise RuntimeError("Lateral preprocessing failed")
            self.preprocessed_lat = lateral["result"]["image"]
        clf = await self.call_tool("cxr_patho_classifier", {"image": self.preprocessed})
        if clf["status"] != "ok":
            raise RuntimeError("Screening classifier failed")
        scores = clf["result"]["scores"]
        raw = self.generate([
            {"role": "system", "content": "Select candidate pathologies from the supplied screening scores. Do not write a report or invent locations. Return JSON {\"hypotheses\": [{\"pathology\": str, \"score\": number, \"location\": \"unknown\"}]}. An empty list is valid."},
            {"role": "user", "content": json.dumps(scores)}])
        parsed = extract_json(raw)
        hyps = self._parse_hypotheses(parsed.get("hypotheses", [])) if parsed else []
        if not parsed or "hypotheses" not in parsed:
            hyps = [{"pathology": key, "score": value, "location": "unknown"}
                    for key, value in scores.items() if key != "No Finding" and value >= self.threshold]
        self.hypotheses = [{"pathology": h["pathology"], "score": scores[h["pathology"]], "location": "unknown"}
                           for h in hyps if h["pathology"] in scores and h["pathology"] != "No Finding"]
        for hypothesis in self.hypotheses:
            hypothesis["screening_location"] = clf["result"].get("locations", {}).get(hypothesis["pathology"], {"status": "unavailable"})
        self.initial_hypotheses = json.loads(json.dumps(self.hypotheses))
        self.reasoning_steps.append({"stage":"hypothesis_selection","used_tool_indices":[i for i,t in enumerate(self.tool_calls) if t["name"]=="cxr_patho_classifier"]})
        print(f"[screen] hypotheses={len(self.hypotheses)}", flush=True)

    def _parse_hypotheses(self, findings_val) -> list:
        # findings 可能是"已经是 Python 对象"（extract_json 解析后直接是 list），
        # 也可能是"字符串化的 JSON 数组"。都兼容。
        arr = findings_val
        if isinstance(arr, str):
            try:
                arr = json.loads(arr)
            except Exception:
                m = re.search(r"\[.*\]", arr, re.DOTALL)
                if not m:
                    return []
                try:
                    arr = json.loads(m.group(0))
                except Exception:
                    return []
        if not isinstance(arr, list):
            return []
        out = []
        for h in arr:
            if not isinstance(h, dict):
                continue
            out.append({
                "pathology": str(h.get("pathology", h.get("p", "unknown"))),
                "score": float(h.get("score", h.get("s", 0.0))),
                "location": str(h.get("location", h.get("loc", "unknown"))),
            })
        return out

    async def _fallback_screen(self):
        pre = await self.call_tool("cxr_preprocess", {"image": self.image})
        pre_img = (pre["result"] or {}).get("image", self.image)
        self.preprocessed = pre_img
        clf = await self.call_tool("cxr_patho_classifier", {"image": pre_img})

    def _record_hypotheses(self):
        print(f"[记录H] 共 {len(self.hypotheses)} 条候选", flush=True)

    # ---------- 第二层：验证 -> findings ----------
    async def verify(self):
        print("\n===== PHASE 2: verify H + checklist + disambiguate + aggregate =====", flush=True)
        if self.preprocessed is None:
            self.preprocessed = self.image

        context_parts = [
            f"Lateral image: {self.preprocessed_lat or 'unavailable'}. Use multiview VQA when available.",
            f"Preprocessed image: {self.preprocessed}",
            f"Hypotheses H: {json.dumps(self.hypotheses, ensure_ascii=False)}",
        ]
        if not self.no_checklist:
            context_parts.append("For checklist calls, request a broad anatomical assessment supported by actual answers.")
            context_parts.append(
                f"Checklist assessment criteria (must cover all): {json.dumps(CHECKLIST_SPEC, ensure_ascii=False)}; "
                "cover every checklist item, then give final findings.")
        else:
            context_parts.append("(No checklist requirement in this variant; just verify findings.)")
        context = "\n".join(context_parts)
        verify_prompt = getattr(self,"verification_prompt",VERIFY_PROMPT)
        if self.no_checklist:
            verify_prompt = verify_prompt.replace("2. Cover all 7 checklist items: trachea_mediastinum, lung_fields, pleura, heart, diaphragm, bones, devices.\n   Each must be marked checked (normal or abnormal, with supporting evidence). Fill any gaps.",
                                                  "2. Do not run anatomical checklist completion in this variant.")
            verify_prompt = verify_prompt.replace("AND all 7 checklist items are checked", "")
        if self.no_disambig:
            verify_prompt = verify_prompt.replace("3. If two tools give contradictory evidence for the same finding, issue an additional disambiguation call.",
                                                  "3. Preserve conflicting evidence as uncertain; do not add disambiguation calls in this variant.")
        messages = [
            {"role": "system", "content": verify_prompt},
            {"role": "user", "content": context},
        ]

        step = 0
        final_rejections = 0
        last_rejection_signature = None
        while step < self.max_steps and len(self.tool_calls) < self.max_tool_calls:
            step += 1
            # Only the latest authoritative state belongs in the prompt.
            # Full snapshots remain in evidence_history and policy_trace.
            messages = [m for m in messages if not m.get("content", "").startswith("CURRENT_VERIFICATION_STATE\n")]
            messages.append({"role":"user","content":"CURRENT_VERIFICATION_STATE\n" + json.dumps({
                "remaining_tool_calls":self.max_tool_calls-len(self.tool_calls),
                "remaining_planning_turns":self.max_steps-step+1,
                "evidence_cache":self.evidence_cache,
                "checklist":[] if self.no_checklist else self.checklist_checked,
                "instruction":("Verify findings; no anatomical checklist completion in this variant." if self.no_checklist else "Plan verification and anatomical coverage jointly; one meaningful assessment can address multiple hypotheses. Preserve uncertainty.")})})
            raw = self.generate(messages)
            print(f"[Phase2 step{step}] {truncate(raw, 300)}", flush=True)
            parsed = extract_json(raw)
            if parsed is None:
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Output exactly one JSON (call_tool or final_answer)."},
                ]
                continue
            action = parsed.get("action")
            if action == "final_answer":
                await self._refresh_evidence()
                # ``uncertain`` is an assessed, reportable limitation (for
                # example an indeterminate lesion), not an unfinished query.
                # Only unsupported findings and unresolved conflicts require
                # another visual call; otherwise final_answer can loop until
                # the planning budget is exhausted.
                unresolved=[e["finding"] for e in self.evidence_cache if e["status"] in (("unverified",) if self.no_disambig else ("unverified","conflict"))]
                missing=[] if self.no_checklist else [it["name"] for it in self.checklist_checked if it["status"]!="checked"]
                rejection_signature = (tuple(sorted(unresolved)), tuple(sorted(missing)))
                if rejection_signature == last_rejection_signature:
                    final_rejections += 1
                else:
                    final_rejections = 1
                    last_rejection_signature = rejection_signature
                # Conflicts/indeterminate findings are valid limitations after
                # disambiguation has been attempted. Do not loop forever when
                # the policy keeps emitting the same final_answer.
                if (unresolved or missing) and step < self.max_steps and final_rejections <= 2:
                    messages += [{"role":"assistant","content":raw},
                                 {"role":"user","content":json.dumps({"unresolved":unresolved,"unchecked":missing,"instruction":("Preserve conflicts as uncertain; do not issue conflict-resolution calls." if self.no_disambig else "Final answer rejected: explicitly ask a visual tool about the listed unresolved findings and unchecked regions. You must obtain new evidence before attempting final_answer again. Choose the tool and order yourself; preserve genuinely unassessable findings as uncertain.")})}]
                    continue
                break
            if action == "call_tool":
                obs = await self.call_tool(parsed.get("tool_name"), parsed.get("arguments", {}))
                await self._refresh_evidence()
                final_rejections = 0
                last_rejection_signature = None
                if self.no_disambig:
                    follow = ("Continue gathering evidence; when tools conflict, "
                              "do NOT issue extra disambiguation calls — trust the "
                              "accumulated evidence and finally give findings.")
                else:
                    follow = "Continue verifying hypotheses and resolving conflicts, then give findings."
                if not self.no_checklist:
                    follow += " Complete outstanding anatomical checklist assessments."
                if obs.get("name") == "cxr_negation_checker":
                    follow += (" This text-only consistency check supplies no new visual evidence and cannot "
                               "mark checklist items checked. For remaining unchecked/uncertain regions, "
                               "ask a visual tool a targeted anatomical question using the actual image. "
                               "Repeating the same negation check will not resolve missing visual coverage.")
                previous = self.tool_calls[:-1]
                if any(t.get("name") == obs.get("name") and t.get("arguments") == obs.get("arguments") for t in previous):
                    follow += (" You have repeated an identical tool request. Review its existing result; "
                               "if evidence is insufficient, change the visual question or view to address the gap. "
                               "Preserve uncertainty when the image cannot resolve it.")
                self.reasoning_steps.append({"stage":"verification_context","observed_tool_indices":[len(self.tool_calls)-1]})
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Observation:\n{truncate(obs['result'])}\n\n{follow}"},
                ]
            else:
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Invalid action. Use call_tool or final_answer."},
                ]

        # Interpret answer content (not question keywords) with a frozen judge.
        await self._refresh_evidence()
        unresolved = [e["finding"] for e in self.evidence_cache if e["status"] in (("unverified",) if self.no_disambig else ("unverified","conflict"))]
        self.termination = {
            "reason": "complete" if not unresolved and (self.no_checklist or all(x["status"]=="checked" for x in self.checklist_checked)) else "budget_exhausted",
            "unresolved_findings": unresolved,
            "preserved_conflicts": [e["finding"] for e in self.evidence_cache if e["status"] == "conflict"],
            "checklist_enabled": not self.no_checklist, "disambiguation_enabled": not self.no_disambig,
            "unchecked_items": [] if self.no_checklist else [x["name"] for x in self.checklist_checked if x["status"]!="checked"],
            "planning_turns":step,"tool_calls":len(self.tool_calls),
            "planning_budget":self.max_steps,"tool_budget":self.max_tool_calls,
        }
        evidence = {"findings": self.evidence_cache, "checklist": self.checklist_checked,
                    "limitations": self.termination}
        synthesis_indices = sorted({i for e in self.evidence_cache for i in e["evidence_indices"]}
                                   | {e["tool_call_index"] for it in self.checklist_checked for e in it["evidence"]})
        self.reasoning_steps.append({"stage":"report_synthesis","used_tool_indices":synthesis_indices,"policy_generated":True})
        referenced = {i for step in self.reasoning_steps for i in step.get("used_tool_indices", [])}
        pending = list(referenced)
        while pending:
            for dependency in self.tool_calls[pending.pop()].get("dependency_indices", []):
                if dependency not in referenced:
                    referenced.add(dependency)
                    pending.append(dependency)
        self.used_tool_indices = sorted(referenced)
        self.final_findings = self.generate([
            {"role": "system", "content": "Write a concise chest X-ray Findings paragraph using ONLY supplied evidence. Do not promote unverified hypotheses to findings. Preserve uncertainty, unresolved conflicts and assessment limitations. Return the paragraph only."},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}])
        if not self.final_findings.strip():
            raise RuntimeError("Empty evidence-based report")

    async def visual_query(self, query):
        if self.preprocessed_lat:
            return await self.call_tool("cxr_multiview_vqa", {"image_pa":self.preprocessed,
                                         "image_lat":self.preprocessed_lat,"query":query})
        return await self.call_tool("cxr_vqa", {"image":self.preprocessed,"query":query})

    async def _refresh_evidence(self):
        from www_cxr_agent.reward.reward_judge import interpret_evidence
        # Reinterpreting identical observations wastes judge time and can change
        # termination without any new evidence.
        if getattr(self, "_interpreted_tool_count", -1) == len(self.tool_calls):
            return
        # The judge only consumes successful visual answers. Failures,
        # preprocessing and text-only tools cannot add diagnostic evidence.
        # Retain original indices so cached citations remain valid.
        visual = [(i, tc["name"], tc.get("arguments", {}), tc.get("result", {}))
                  for i, tc in enumerate(self.tool_calls)
                  if tc.get("status") == "ok" and tc.get("name") in
                  ("cxr_vqa", "cxr_multiview_vqa", "segment_2d")]
        key = json.dumps([visual, [h["pathology"] for h in self.hypotheses]], sort_keys=True)
        if getattr(self, "_visual_evidence_key", None) == key:
            self._interpreted_tool_count = len(self.tool_calls)
            self.timings["evidence_cache_hits"] = self.timings.get("evidence_cache_hits", 0) + 1
            return
        started=time.monotonic()
        if visual:
            result = await interpret_evidence(self.tool_calls, self.hypotheses)
            if result.get("assessment_version") == "timeout_conservative_v1":
                raise RuntimeError("Evidence judge unavailable; refusing to train on infrastructure failure")
        else:
            result = {"assessment_version": "no_visual_evidence",
                      "findings": [{"finding": h["pathology"], "status": "unverified",
                                    "location": "unknown", "evidence_indices": [],
                                    "rationale": "No successful visual evidence yet"}
                                   for h in self.hypotheses],
                      "checklist": [{"name": name, "status": "unchecked", "evidence_indices": []}
                                    for name in CHECKLIST_ITEMS]}

        self.timings["evidence_judge_seconds"]+=time.monotonic()-started
        self.evidence_assessment_version = result.get("assessment_version", "unknown")
        entries = result["findings"]
        if any(e.get("status") not in ("present","absent","normal","uncertain","conflict","unverified") for e in entries):
            raise ValueError("Invalid evidence status")
        for h in self.hypotheses:
            if not any(e["finding"].strip().casefold()==h["pathology"].strip().casefold() for e in entries):
                entries.append({"finding":h["pathology"],"status":"unverified","location":"unknown","evidence_indices":[],"rationale":"Missing interpretation"})
        for entry in entries:
            cited=[str(self.tool_calls[i]["result"].get("answer","")).strip().lower().rstrip('.')
                   for i in entry.get("evidence_indices",[])]
            if cited and all(a in ("matched","not matched","unmatched") for a in cited):
                entry["status"]="uncertain"
                entry["rationale"]="Matching-task output does not establish a clinical finding"
            if entry["finding"].strip().casefold() in ("other lesion","lesion","abnormality","abnormalities","finding"):
                entry["status"]="uncertain"
                entry["rationale"]="Non-specific tool category is not a confirmed clinical diagnosis"
        for h in self.hypotheses:
            match=next((e for e in entries if e["finding"].casefold()==h["pathology"].casefold() and e.get("evidence_indices")),None)
            if match:
                h["location"]=match.get("location","unknown")
        self.evidence_cache = []
        for e in entries:
            evidence = [{"tool_call_index":i,"query":self.tool_calls[i]["arguments"].get("query"),
                         "answer":self.tool_calls[i]["result"].get("answer",json.dumps(self.tool_calls[i]["result"]))} for i in e["evidence_indices"]]
            self.evidence_cache.append(dict(e, confirmed=e["status"]=="present", evidence=evidence,
                                            notes=e.get("rationale","")))
        by_name={it["name"]:it for it in result["checklist"]}
        for item in self.checklist_checked:
            if self.no_checklist:
                item.update(status="disabled",evidence=[])
                continue
            parsed=by_name.get(item["name"],{})
            status=parsed.get("status","unchecked")
            if status not in ("checked","uncertain","unchecked"):
                raise ValueError("Invalid checklist status")
            # Semantic judge validates breadth and answer adequacy; query
            # keywords alone neither establish nor invalidate coverage.
            item.update(status=status,notes=parsed.get("rationale",""),evidence=[
                {"tool_call_index":i,"answer":self.tool_calls[i]["result"].get("answer",json.dumps(self.tool_calls[i]["result"]))}
                for i in parsed.get("evidence_indices",[]) if isinstance(i,int) and 0 <= i < len(self.tool_calls)])
        self.abnormal=[e["finding"] for e in self.evidence_cache if e["confirmed"]]
        self.pred_label14=[]  # CE labels must come from the final report classifier.
        self.evidence_interpreted=True
        self._visual_evidence_key = key
        self._interpreted_tool_count=len(self.tool_calls)
        self.evidence_history.append({"after_tool_count":len(self.tool_calls),"findings":json.loads(json.dumps(self.evidence_cache)),"checklist":json.loads(json.dumps(self.checklist_checked))})

    def _auto_checklist(self, parsed, obs):
        # Semantic coverage is assessed by _refresh_evidence; a query alone proves nothing.
        return None

    def _aggregate_evidence(self):
        # Do not overwrite semantic evidence (including new checklist findings) with hypotheses.
        return self.evidence_cache

    def build_trajectory(self):
        covered = sum(1 for it in self.checklist_checked if it["status"] == "checked")
        self._aggregate_evidence()
        status_map = {"present": "confirmed", "absent": "rejected",
                      "normal": "rejected", "uncertain": "uncertain",
                      "conflict": "uncertain", "unverified": "uncertain"}
        verification_statuses = [{"finding": e.get("finding"),
                                  "status": status_map.get(e.get("status"), "uncertain"),
                                  "raw_status": e.get("status"),
                                  "evidence_indices": e.get("evidence_indices", [])}
                                 for e in self.evidence_cache]
        return {
            "tool_calls": [t for t in self.tool_calls if t["name"] != "__phase__"],
            "evidence_cache": self.evidence_cache,
            "verification_statuses": verification_statuses,
            "evidence_assessment_version": getattr(self, "evidence_assessment_version", "unknown"),
            "checklist_checked": self.checklist_checked,
            "abnormal": self.abnormal,
            "pred_label14": self.pred_label14,
            "_coverage": f"{covered}/{len(self.checklist_checked)}",
            "findings": self.final_findings,
            "used_tool_indices": getattr(self, "used_tool_indices", []),
            "reasoning_steps": self.reasoning_steps,
            "policy_trace": self.policy_trace,
            "tool_budget": self.max_tool_calls,
            "hypotheses": self.hypotheses,
            "initial_hypotheses": getattr(self, "initial_hypotheses", []),
            "evidence_history": self.evidence_history,
            "termination": getattr(self, "termination", {}),
            "images": {"frontal": self.image, "lateral": self.image_lat},
        }

    async def run(self, sample_id: str = "demo", phase: str = "full"):
        started = time.monotonic()
        # phase: "screen"（仅 Phase1，快、不出 VQA 显存） 或 "full"（完整两阶段）
        if self.model is None:
            self.load_model()          # infer 走 run() 时不手动 load，这里懒加载
        try:
            await self.screen()
            if phase == "full":
                await self.verify()
        finally:
            if self.manager is not None:
                await self.manager.close()

        # screen 模式：findings 用 H 的 JSON 概括；full 模式：findings 用最终临床段落
        findings = self.final_findings or ""
        if phase == "screen" and not findings:
            findings = json.dumps(
                [{"pathology": h["pathology"], "score": h["score"], "location": h["location"]}
                 for h in self.hypotheses],
                ensure_ascii=False,
            )

        record = {
            "id": sample_id,
            "phase": phase,
            "findings": findings,
            "trajectory": self.build_trajectory(),
            "efficiency": dict(self.token_usage,**self.timings,elapsed_seconds=time.monotonic()-started),
        }
        if self.out:
            Path(self.out).parent.mkdir(parents=True, exist_ok=True)
            with open(self.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if not self.training:
            print("\n========== 推理产物 (JSONL) ==========")
            print(json.dumps(record, ensure_ascii=False, indent=2))
        else:
            print(f"[trajectory] sample={sample_id} termination={getattr(self, 'termination', {})}", flush=True)
        return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--id", default="demo")
    ap.add_argument("--out", default="runs/layered_run.jsonl")
    ap.add_argument("--phase", default="full",
                    help="screen=仅Phase1(不调VQA,单卡可跑); full=完整两阶段")
    ap.add_argument("--coexist", action="store_true",
                    help="4bit 共存模式（当前唯一方式）。CheXagent~3.9G + Qwen3~10G 单卡共存")
    args = ap.parse_args()

    agent = LayeredAgent(image=args.image, threshold=args.threshold,
                         max_steps=args.max_steps, out=args.out,
                         coexist=True)  # 4bit 共存为当前唯一方案
    agent.load_model()
    result = asyncio.run(agent.run(sample_id=args.id, phase=args.phase))
    print("\n轨迹: ", json.dumps(result["trajectory"], ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
