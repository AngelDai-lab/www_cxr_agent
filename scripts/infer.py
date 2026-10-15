# -*- coding: utf-8 -*-
"""推理：加载训练好的 LoRA 权重，跑两级 Agent，产出统一 JSONL（§1.3.2）。

用法（服务器 base 环境）：
    python scripts/infer.py --ckpt runs/checkpoints/lora_step50 \
        --phase full --out runs/infer_full.jsonl --n 20

--ckpt 为空时用基座 Qwen3 零样本（等价 Tool-use-without-RL / 消融去掉 GRPO 的推理侧基座）。
phase=screen 仅第一层出 H；phase=full 完整两阶段出 final findings + 证据轨迹。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 复用 train_grpo.py 内已跑通的 load_dataset（读 MimicCXRDataset + 统一字段），
# 避免依赖不存在的独立 data_loaders 模块，确保 infer 与训练使用同一套数据结构。
from train_grpo import load_dataset
from www_cxr_agent.agents.layered_agent import LayeredAgent  # noqa: F401
from www_cxr_agent.agents.react_single_layer import SingleReactAgent  # noqa: F401


def load_shared_model(ckpt: str | None):
    """只加载一次 Qwen3（4bit）+ 可选 LoRA，所有样本共享，避免每样本重载挤爆显存。
    复用 LayeredAgent.load_model 的加载逻辑，但通过一个临时空 agent 触发。"""
    tmp = LayeredAgent(image=None, threshold=0.5, max_steps=20, out="",
                       coexist=True, lora_ckpt=ckpt)
    tmp.load_model()
    return tmp.model, tmp.tokenizer


def infer_one(sample: dict, phase: str, threshold: int, max_steps: int,
              model, tokenizer, backend: str = "layered",
              no_checklist: bool = False, no_disambig: bool = False, max_tool_calls: int = 32) -> dict:
    """对单个样本建 Agent（注入共享模型），run 后返回推理产物记录。
    所有样本共享同一份 Qwen3+LoRA（model/tokenizer），仅 image/阈值不同。
    backend: layered=两层(LayeredAgent) | react_single=单层(SingleReactAgent, 消融一)。"""
    if backend == "cot_reflection":
        from www_cxr_agent.agents.cot_reflection import CoTReflectionAgent
        return asyncio.run(CoTReflectionAgent().run(sample))
    if backend in ("vlm_direct", "encoder_decoder"):
        from www_cxr_agent.agents.server_manager import ToolServerManager
        async def direct():
            import time
            started=time.monotonic()
            manager=ToolServerManager()
            try:
                if backend=="encoder_decoder":
                    result=await manager.call_tool("cxr_report_draft",{"image":sample["abs_images"][0]})
                    text=result["text"].split("FINDINGS:",1)[-1].split("IMPRESSION:",1)[0].strip()
                    return {"id":sample["id"],"findings":text,"report_ref":sample["report"],"method":"local_encoder_decoder","trajectory":{},"efficiency":{"elapsed_seconds":time.monotonic()-started}}
                args={"image_pa":sample["abs_images"][0],"image_lat":sample["image_lat"]} if sample.get("image_lat") else {"image":sample["abs_images"][0]}
                result=await manager.call_tool("cxr_multiview_vqa" if sample.get("image_lat") else "cxr_vqa",
                                              dict(args,query="Write the Findings section of a chest radiograph report."))
            finally:
                await manager.close()
            return {"id":sample["id"],"findings":result["answer"],"report_ref":sample["report"],"method":"chexagent_direct","trajectory":{},"efficiency":{"elapsed_seconds":time.monotonic()-started}}
        return asyncio.run(direct())
    image = (sample.get("abs_images") or [None])[0]
    if not image:
        logging.warning("sample %s has no abs_images, skip", sample.get("id"))
        return {"id": sample.get("id"), "phase": phase, "findings": "",
                "trajectory": {}}
    if backend == "react_single":
        # 消融一：单层 ReAct。与两层共用同一份共享 model/tokenizer（注入后 SingleReactAgent 跳过加载）。
        agent = SingleReactAgent(
            image=image, image_lat=sample.get("image_lat"), threshold=threshold, max_steps=max_steps, max_tool_calls=max_tool_calls, out="",
            coexist=True, lora_ckpt=None, model=model, tokenizer=tokenizer,
            no_checklist=no_checklist, no_disambig=no_disambig,
        )
    else:
        agent = LayeredAgent(
            image=image,
            image_lat=sample.get("image_lat"),
            threshold=threshold,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            out="",                # 禁止 agent 自写，统一由 infer.main 写入，避免 JSONL 重复行
            coexist=True,          # CheXagent 4bit 与 Qwen3 单卡共存，不启停子进程
            lora_ckpt=None,        # LoRA 已在共享模型加载时套好
            model=model,
            tokenizer=tokenizer,
            no_checklist=no_checklist,
            no_disambig=no_disambig,
        )
    record=asyncio.run(agent.run(sample_id=str(sample.get("id")), phase=phase))
    record.update(report_ref=sample["report"],split=sample.get("split"),subject_id=sample.get("subject_id"),
                                  dataset=sample.get("dataset"), label14=sample.get("label14"),
                                  label_order=sample.get("label_order"), gt_source=sample.get("gt_source"))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="LoRA 权重目录；为空则用基座 Qwen3 零样本")
    ap.add_argument("--phase", default="full", choices=["screen", "full"])
    ap.add_argument("--backend", default="layered",
                    choices=["layered", "react_single", "cot_reflection", "vlm_direct", "encoder_decoder"],
                    help="layered=两层Agent(默认) | react_single=单层ReAct(消融一去分层)")
    ap.add_argument("--out", default="runs/infer_screen.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default="mimic_cxr",choices=["mimic_cxr","iu_xray"])
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="H 生成/筛选用阈值（传给 LayeredAgent）")
    ap.add_argument("--max-tool-calls", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=32,
                    help="单层 ReAct 决策步数上限（传给 LayeredAgent）")
    ap.add_argument("--single-view",action="store_true",help="Use the same paired cohort, discard lateral input only")
    ap.add_argument("--paired-only",action="store_true")
    ap.add_argument("--ids",help="JSON array of fixed evaluation sample IDs")
    ap.add_argument("--no-checklist", action="store_true",
                    help="消融：去掉诊断清单查漏（VERIFY 不核对 7 条解剖条目）")
    ap.add_argument("--no-disambig", action="store_true",
                    help="消融：去掉证据消歧（工具矛盾时不追加调用）")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    samples = load_dataset(args.dataset, args.split, n=None if args.ids else args.n)
    if args.ids:
        requested=json.loads(Path(args.ids).read_text())
        by_id={s["id"]:s for s in samples}
        samples=[by_id[key] for key in requested]
    if args.paired_only:
        samples=[s for s in samples if s.get("image_lat")]
    if args.single_view:
        samples=[dict(s,image_lat=None) for s in samples]
    if not samples:
        logging.error("no samples loaded")
        sys.exit(1)

    # 只加载一次 Qwen3+LoRA，所有样本共享（方向2：避免每样本重载挤爆显存）
    logging.info("loading shared model (ckpt=%s) ...", args.ckpt)
    model, tokenizer = load_shared_model(args.ckpt) if args.backend in ("layered","react_single") else (None,None)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            rec = infer_one(
                s, args.phase, args.threshold, args.max_steps,
                model, tokenizer, backend=args.backend,
                no_checklist=args.no_checklist,
                no_disambig=args.no_disambig,
                max_tool_calls=args.max_tool_calls,
            )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logging.info("inferred %s", rec.get("id"))

    logging.info("infer done -> %s (%d)", args.out, len(samples))


if __name__ == "__main__":
    main()