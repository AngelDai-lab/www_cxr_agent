# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SmokeGate:
    def __init__(self, out_log: str):
        self.out_log = out_log
        self.results: dict[str, dict] = {}
        self._lines: list[str] = []

    def step(self, name: str, ok: bool, detail: str = ""):
        rec = {"ok": ok, "detail": detail,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.results[name] = rec
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}: {detail}"
        self._lines.append(line)
        print(line, flush=True)
        self._flush()

    def run(self, gate2_n=1, selected="all", skip_gate3=False):
        try:
            if selected in ("all", "1"):
                asyncio.run(self._gate1())
            if selected in ("all", "2"):
                asyncio.run(self._gate2(gate2_n))
            if selected in ("all", "3") and not skip_gate3:
                self._gate3()
        except Exception as exc:
            self.step("exception", False, f"{type(exc).__name__}: {exc}")
        finally:
            self._flush()
        return bool(self.results) and all(r["ok"] for r in self.results.values())

    # ---------------- gates ----------------
    # ---- 全工具栈（§1.3.2 工具 MCP：7001-7007）----
    # 工具名 -> 正确输入（每工具各用合法参数调一次，全过才算 gate1 PASS）
    # 参数名严格对齐 layered_agent 的 tool prompt（server 端 pydantic 校验）
    TOOL_PROBES = {
        "cxr_preprocess":      {"image": "__SAMPLE__"},
        "cxr_patho_classifier": {"image": "__SAMPLE__"},
        "cxr_vqa":             {"image": "__SAMPLE__",
                               "query": "Is there any abnormality in this chest X-ray?"},
        "cxr_multiview_vqa":   {"image_pa": "__SAMPLE__",
                               "image_lat": "__SAMPLE__",
                               "query": "Compare the two views and describe findings."},
        "segment_2d":          {"image": "__SAMPLE__"},
        "cxr_report_draft":    {"image": "__SAMPLE__"},   # 只收 image，返回 {"text"}
        # 7007 negation/一致性判官：evidence_set 是单个字符串（server 端 pydantic 校验为 str）
        # 注：layered_agent prompt 文档写数组，但 server schema 实为 str，以实际 pydantic 为准
        "cxr_negation_checker": {"candidate_sentence": "There is no pneumothorax.",
                                "evidence_set": "cxr_vqa: No pneumothorax"},
    }

    async def _gate1(self):
        """第一关(全栈)：遍历 7001-7007 每个 MCP 工具各调一次，全返回正常即 PASS。"""
        try:
            from www_cxr_agent.agents.server_manager import ToolServerManager
            m = ToolServerManager()
            img = _sample_img()
            per_tool = []
            ok_all = True
            for name, args in self.TOOL_PROBES.items():
                if name == "cxr_report_draft" and os.environ.get("SMOKE_REQUIRE_DRAFT", "0") != "1":
                    per_tool.append("cxr_report_draft:SKIP(optional)")
                    continue
                print(f"[gate1] calling {name}", flush=True)
                self._lines.append(f"[START] {name}")
                self._flush()
                started = time.monotonic()
                a = {k: (img if v == "__SAMPLE__" else v) for k, v in args.items()}
                try:
                    r = await asyncio.wait_for(m.call_tool(name, a), timeout=120)
                    tool_ok = bool(r)
                    if name == "cxr_patho_classifier":
                        tool_ok = tool_ok and "scores" in (r or {})
                    if name == "cxr_vqa" or name == "cxr_multiview_vqa":
                        ans = (r or {}).get("answer", "")
                        tool_ok = tool_ok and bool(str(ans).strip())
                    per_tool.append(f"{name}:{'OK' if tool_ok else 'EMPTY/err'}")
                    ok_all = ok_all and tool_ok
                except Exception as e:  # noqa: BLE001
                    per_tool.append(f"{name}:EXC={type(e).__name__}: {e}")
                    ok_all = False
                self._lines.append(f"[DONE] {per_tool[-1]} seconds={time.monotonic()-started:.2f}")
                self._flush()
            await m.close()
            self.step("gate1_tool_stack", ok_all, " | ".join(per_tool))
        except Exception as e:  # noqa: BLE001
            self.step("gate1_tool_stack", False, str(e))

    async def _gate2(self, n: int):
        """第二关(一体式)：单图出 findings + 轨迹 + 7/7 清单。
        优先校验已跑通的真实 JSONL 产物（含 --coexist 一体式）；
        缺失时才现场跑一张。判据：findings 非空 && trajectory 非空 && coverage 7/7。"""
        try:
            from www_cxr_agent.agents.layered_agent import LayeredAgent
            agent = LayeredAgent(image=_sample_img(), threshold=0.30, max_steps=8,
                                 out=self.out_log + ".prediction.jsonl", coexist=True)
            rec = await agent.run(sample_id="smoke", phase="full")
            findings = bool((rec or {}).get("findings"))
            traj = bool((rec or {}).get("trajectory"))
            cov = ((rec or {}).get("trajectory") or {}).get("_coverage", "0/0")
            trajectory=rec.get("trajectory",{})
            items=trajectory.get("checklist_checked",[])
            termination=trajectory.get("termination",{})
            from www_cxr_agent.agents.layered_agent import CHECKLIST_ITEMS
            calls=trajectory.get("tool_calls",[])
            assessed=len(items)==7 and {it.get("name") for it in items}==set(CHECKLIST_ITEMS)
            for item in items:
                for evidence in item.get("evidence",[]):
                    idx=evidence.get("tool_call_index")
                    assessed=assessed and isinstance(idx,int) and 0<=idx<len(calls) and calls[idx].get("status")=="ok"
                if item.get("status")=="checked" and not item.get("evidence"):
                    assessed=False
            honest_completion=termination.get("reason")=="budget_exhausted" or (termination.get("reason")=="complete" and cov=="7/7" and not termination.get("unresolved_findings"))
            self.step("gate2_infer", findings and traj and assessed and honest_completion,
                      f"findings={findings}, traj={traj}, coverage={cov}, termination={termination}")
        except Exception as e:  # noqa: BLE001
            self.step("gate2_infer", False, str(e))

    def _gate3(self):
        """第三关(GRPO full 短训)：trial 规模，rollout 含完整两级链路，
        量化 7004 做逐假设 VQA 验证奖励；判据 n>0 且 avg_R 未退化。"""
        try:
            print("[gate3] loading GRPO trainer", flush=True)
            from www_cxr_agent.train import GRPOTrainer
            from www_cxr_agent.reward import RewardComposer
            reward = RewardComposer(phase="explore")
            reward.set_phase(os.environ.get("CXR_SMOKE_REWARD_PHASE", "explore"))
            trainer = GRPOTrainer(
                model_path="/root/autodl-tmp/models/Qwen3-14B-Instruct",
                lora_rank=16, lora_alpha=32, load_in_4bit=False,
                rollout=4, batch_size=1, max_steps=3, lr=1e-5,
                reward=reward, max_agent_steps=32, max_tool_calls=32,
                phase="full",   # full：Phase1 + 7004 VQA 逐假设验证
            )
            trainer.load_model()
            before = {name: p.detach().cpu().clone() for name, p in trainer.model.named_parameters() if p.requires_grad}
            print("[gate3] model loaded; starting one short rollout", flush=True)
            from train_grpo import load_dataset
            sample = load_dataset("mimic_cxr", "train", n=1)[0]
            if sample.get("label14") is None:
                raise RuntimeError("Smoke training requires real labels")
            rolls = trainer.fit([sample], trial=False, max_steps=1,
                                ckpt_dir=self.out_log + ".checkpoints",
                                traj_out=self.out_log + ".trajectories.jsonl")
            n = len(rolls)
            avg_r = sum(r.reward for r in rolls) / n if n else 0.0
            # full 链路判据：奖赏来自 7004 验证（VQA 被调用则 n_tools 会含 cxr_vqa）
            n_vqa_tools = sum(
                1 for r in rolls for t in r.tool_calls if t.get("name") in ("cxr_vqa", "cxr_multiview_vqa")
            )
            import torch
            from pathlib import Path
            delta = sum(float((p.detach().cpu()-before[name]).abs().sum())
                        for name, p in trainer.model.named_parameters() if name in before)
            checkpoint = Path(self.out_log + ".checkpoints/lora_step1/adapter_config.json").is_file()
            grad = getattr(trainer, "last_gradient_norm", 0.0)
            self.step("gate3_grpo_short_train", n >= 2 and n_vqa_tools > 0 and delta > 0 and grad > 0 and checkpoint and all(torch.isfinite(torch.tensor(r.reward)) for r in rolls),
                      f"n_rolls={n} avg_R={avg_r:.4f} vqa_calls={n_vqa_tools} parameter_delta={delta} gradient_norm={grad} checkpoint={checkpoint}")
        except Exception as e:  # noqa: BLE001
            self.step("gate3_grpo_short_train", False, str(e))

    def _flush(self):
        os.makedirs(os.path.dirname(self.out_log) or ".", exist_ok=True)
        with open(self.out_log, "w", encoding="utf-8") as f:
            f.write("\n".join(self._lines) + "\n")
            f.write("\n=== summary ===\n")
            f.write(json.dumps(self.results, ensure_ascii=False, indent=2) + "\n")


def _sample_img() -> str:
    """定位一张预处理的样例图。"""
    import glob
    cands = glob.glob("/root/autodl-tmp/www-cxr-agent/runs/preprocessed/*.png")
    return cands[0] if cands else "/root/autodl-tmp/www-cxr-agent/runs/preprocessed/sample.png"


def _mock_screen(sample) -> dict:
    """框架替身：正式接入 layered_agent.run(sample_id, phase='screen')。
    SampleRecord 是 dict；兼容 dataclass。"""
    sid = sample.get("id") if isinstance(sample, dict) else getattr(sample, "id", None)
    return {"id": sid, "phase": "screen",
            "findings": "mock", "trajectory": {"tool_calls": []}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/smoke_gate.log")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--gate", choices=["all", "1", "2", "3"], default="all")
    ap.add_argument("--skip-gate3", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--stage-timeout", type=int, default=1800)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.worker:
        import faulthandler
        faulthandler.enable()
        # Parent process owns the hard timeout; avoid asynchronous stack walks
        # through native tensor dispatch while GPU kernels are executing.
        ok = SmokeGate(args.out).run(args.n, args.gate, args.skip_gate3)
        raise SystemExit(0 if ok else 1)
    import subprocess
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stages = ["1", "2", "3"] if args.gate == "all" else [args.gate]
    if args.skip_gate3:
        stages = [g for g in stages if g != "3"]
    results = {}
    for stage in stages:
        log = args.out + f".gate{stage}.console.log"
        env = dict(os.environ)
        # Policy/LoRA never spread onto the tool GPUs.
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CXR_POLICY_GPU", "0")
        cmd = [sys.executable, "-u", __file__, "--worker", "--gate", stage,
               "--n", str(args.n), "--out", args.out + f".gate{stage}"]
        print(f"[START] gate{stage}; live log: {log}", flush=True)
        started = time.monotonic()
        with open(log, "w") as stream:
            proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env)
            try:
                code = proc.wait(timeout=args.stage_timeout)
                results[stage] = {"status": "PASS" if code == 0 else "FAIL", "exit_code": code}
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                results[stage] = {"status": "TIMEOUT"}
        results[stage].update(seconds=round(time.monotonic()-started, 2), log=log)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"[END] gate{stage}: {results[stage]}", flush=True)
        if results[stage]["status"] != "PASS":
            break
    raise SystemExit(0 if results and all(r["status"] == "PASS" for r in results.values()) else 1)

if __name__ == "__main__":
    main()
