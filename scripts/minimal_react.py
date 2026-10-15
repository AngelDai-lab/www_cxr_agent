#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimal_react.py — 最小链路 ReAct 验证脚本（WWW CXR Agent）

目标：验证"模型自主决策 → 调 MCP 工具 → 观察回填 → 继续推理 / 结束"这条一步 react 链路真正work。

用法：
    cd /root/autodl-tmp/www-cxr-agent
    python scripts/minimal_react.py \
        --image /root/autodl-tmp/datasets/mimic_cxr/mimic_cxr/images/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg \
        --max-steps 6

默认工具集（args.tools 逗号分隔）：
    cxr_preprocess, cxr_report_draft, cxr_patho_classifier, cxr_vqa, segment_2d, cxr_negation_checker
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="fastmcp")
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/www-cxr-agent")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from www_cxr_agent.agents.server_manager import ToolServerManager

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-14B-Instruct"

# 工具 schema（与 tool_configs 注册名一致，供 system prompt）
TOOL_SCHEMA = {
    "cxr_preprocess": '{"image": "str: 原始胸片路径"}   -> {"image": "str 预处理后PNG路径"}',
    "cxr_report_draft": '{"image": "str"}   -> {"text": "str 报告草稿(不作终稿)"}',
    "cxr_patho_classifier": '{"image": "str"}   -> {"scores": "dict[str,float] 18类病理概率"}',
    "cxr_vqa": '{"image": "str", "query": "str"}   -> {"answer": "str"}',
    "segment_2d": '{"image": "str", "organs": "list[str] 可选"}   -> {"mask": "str", "metrics": "dict"}',
    "cxr_negation_checker": '{"candidate_sentence": "str", "evidence_set": "list[str]"}   -> {"score": 0|1, "rationale": "str"}',
}

SYSTEM_PROMPT = """You are a chest X-ray (CXR) analysis agent. You must decide, step by step, which diagnostic tool to call to gather evidence, and finally produce a clinical findings paragraph.

You have the following tools available:

{TOOLS}

## Output protocol
Every turn you MUST respond with EXACTLY ONE JSON object (no other text, no markdown fences). Choose one of two forms:

Call a tool:
{{"reasoning": "why you call this tool now", "action": "call_tool", "tool_name": "<tool name>", "arguments": {{<args>}}}}

Finish:
{{"reasoning": "why you now have enough evidence", "action": "final_answer", "findings": "<final clinical findings text>"}}

## Rules
- Do NOT invent findings without tool evidence. You are a text-only model: you cannot "see" the image, you must call tools to obtain evidence.
- Do NOT say "Okay, let me try..." or describe your intent in prose. Output the JSON directly.
- Call cxr_preprocess first to normalize the image, then use other tools on the preprocessed image path.
- When you have enough evidence, output final_answer.
- Keep final findings concise and clinical."""

REASONING_POST = """Tool observation returned. Based strictly on the observation above, decide your next action or produce the final answer. Output exactly one JSON object (call_tool or final_answer)."""


def build_system_prompt(tools: list[str]) -> str:
    lines = []
    for t in tools:
        if t in TOOL_SCHEMA:
            lines.append(f"- {t}: {TOOL_SCHEMA[t]}")
    return SYSTEM_PROMPT.format(TOOLS="\n".join(lines))


def extract_json(text: str):
    """提取模型输出中第一个完整 JSON 对象。"""
    # 去掉可能的 markdown 围栏
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def truncate(obj, max_chars: int = 2000):
    s = str(obj)
    return s[:max_chars] + (f"...[truncated, len={len(s)}]" if len(s) > max_chars else "")


class MinimalReAct:
    def __init__(self, tools: list[str], max_steps: int, image: str):
        self.tools = tools
        self.max_steps = max_steps
        self.image = image
        self.manager = ToolServerManager()
        self.trajectory = []       # 每步记录
        self.timings = {}          # 各阶段耗时

    def load_model(self):
        t0 = time.time()
        print("Loading model...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NF4
            bnb_4bit_use_double_quant=True,     # 双量化
            bnb_4bit_compute_dtype=torch.bfloat16,  # 计算仍用 bf16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.timings["model_load"] = time.time() - t0
        print(f"Model loaded in {self.timings['model_load']:.1f}s", flush=True)

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # 最小链路用贪心，稳定
                temperature=None,
                top_p=None,      
                top_k=None,     
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    async def call_tool(self, name: str, args: dict) -> dict:
        t0 = time.time()
        try:
            result = await self.manager.call_tool(name, args)
            duration = time.time() - t0
            return {"ok": True, "result": result, "duration": duration}
        except Exception as e:
            duration = time.time() - t0
            return {"ok": False, "error": str(e), "duration": duration}

    async def run(self):
        sys_prompt = build_system_prompt(self.tools)
        messages = [{"role": "system", "content": sys_prompt}]
        # 首轮注入图片任务
        messages.append({"role": "user", "content": f"Analyze the chest X-ray at path: {self.image}"})

        overall_start = time.time()
        step = 0
        final = None
        while step < self.max_steps:
            step += 1
            step_start = time.time()
            print(f"\n========== Step {step} ==========", flush=True)

            # 模型推理
            t_gen = time.time()
            raw = self.generate(messages)
            gen_time = time.time() - t_gen
            print(f"[模型输出] {raw}", flush=True)

            parsed = extract_json(raw)
            if parsed is None:
                # 反例：模型没输出 JSON，记录并重试
                print("[解析失败] 未检测到 JSON，注入 '请只输出JSON'")
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Your previous reply was not a valid JSON. Respond with EXACTLY one JSON object: call_tool or final_answer."})
                continue

            action = parsed.get("action")
            if action == "final_answer":
                final = parsed.get("findings", "")
                print(f"\n[最终答案]\n{final}", flush=True)
                self.trajectory.append({"step": step, "action": "final_answer", "findings": final, "time": gen_time})
                break

            if action == "call_tool":
                tool_name = parsed.get("tool_name")
                args = parsed.get("arguments", {})
                print(f"[调用工具] {tool_name} args={truncate(args, 300)}", flush=True)
                obs = await self.call_tool(tool_name, args)
                print(f"[观察] ok={obs['ok']} dur={obs['duration']:.1f}s", flush=True)
                self.trajectory.append({"step": step, "action": "call_tool", "tool_name": tool_name, "args": args, "obs_ok": obs["ok"], "obs_time": obs["duration"]})

                # 回填：assistant 的 call_tool JSON + tool observation
                messages.append({"role": "assistant", "content": raw})
                obs_content = truncate(obs.get("result") if obs["ok"] else {"error": obs.get("error")})
                messages.append({"role": "user", "content": f"Observation:\n{obs_content}\n\n{REASONING_POST}"})
            else:
                print("[未知action] ", parsed, flush=True)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": "Invalid action. Must be 'call_tool' or 'final_answer'. Output one JSON."})

            self.timings[f"step_{step}_total"] = time.time() - step_start

        self.timings["overall"] = time.time() - overall_start

        await self.manager.close()

        # 汇总
        print("\n========== 时间统计 ==========")
        for k, v in self.timings.items():
            print(f"  {k}: {v:.2f}s", flush=True)
        n_call = sum(1 for t in self.trajectory if t["action"] == "call_tool")
        n_ok = sum(1 for t in self.trajectory if t["action"] == "call_tool" and t.get("obs_ok"))
        print(f"\n工具调用: {n_call} 次, 成功 {n_ok} 次")
        return {"final": final, "trajectory": self.trajectory, "timings": self.timings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--tools", default="cxr_preprocess,cxr_report_draft,cxr_patho_classifier")
    args = ap.parse_args()

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    agent = MinimalReAct(tools=tools, max_steps=args.max_steps, image=args.image)
    agent.load_model()
    result = asyncio.run(agent.run())
    print("\n========== 轨迹 JSON ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()