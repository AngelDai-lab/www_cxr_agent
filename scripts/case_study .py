# -*- coding: utf-8 -*-
"""Case Study 案例抽取（超参实验 #3）：从轨迹 jsonl 挑典型轨迹导出给人看。

背景（论文四/case study）要挑 3-4 个典型案例可视化完整推理轨迹：
  (a) 初筛假阳性假设 → 验证阶段被驳回（证明控假阳）
  (b) 初筛漏检 → 清单查漏阶段补回（证明防漏诊）
  (c) 工具矛盾 → 消歧过程
  (d) 失败案例 1 个（让步展示模型特点）

本脚本做半自动筛选：按"轨迹是否包含驳回/消歧/补检行为"给候选排序，
导出 human-readable 的 md 供你挑最终案例。无法完全自动化判别的，
给出候选清单 + 自动标注的"特征线索"，由你人工定稿。

用法（服务器 base 环境）：
    python scripts/case_study.py --jsonl runs/trajectories.jsonl --out runs/case_candidates.md
输出：runs/case_candidates.md（含候选案例的轨迹摘要，按特征分类）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

KEYWORDS_REJECT = ["rule out", "ruleout", "reject", "drop", "not evidence",
                   "no evidence", "exclude", "false positive", "normal"]
KEYWORDS_RESCUE = ["checklist", "gap", "missed", "re-check", "recheck",
                   "covered", "add to findings", "overlook", "catch"]
KEYWORDS_DISAMBIG = ["disambig", "contradict", "conflict", "inconsistent",
                     "resolve", "uncertain", "clarif", "segment_2d", "multiview"]


def classify(text: str) -> list[str]:
    t = text.lower()
    tags = []
    if any(k in t for k in KEYWORDS_REJECT):
        tags.append("驳回_假阳")
    if any(k in t for k in KEYWORDS_RESCUE):
        tags.append("查漏_补回")
    if any(k in t for k in KEYWORDS_DISAMBIG):
        tags.append("消歧_矛盾")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", default="runs/case_candidates.md")
    ap.add_argument("--top", type=int, default=6, help="每个特征取前几个候选")
    args = ap.parse_args()

    records = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    buckets: dict[str, list] = {"驳回_假阳": [], "查漏_补回": [], "消歧_矛盾": [], "消歧_已解决": [], "失败候选": []}
    unlabelled = []
    for rec in records:
        traj = rec.get("trajectory") or rec
        hypotheses={h["pathology"].casefold() for h in traj.get("initial_hypotheses", traj.get("hypotheses",[]))}
        entries=traj.get("evidence_cache",[])
        tags=[]
        if any(e.get("status")=="absent" and e["finding"].casefold() in hypotheses for e in entries):
            tags.append("驳回_假阳")
        if any(e.get("status")=="present" and e["finding"].casefold() not in hypotheses for e in entries):
            tags.append("查漏_补回")
        if any(e.get("status")=="conflict" for snapshot in traj.get("evidence_history",[]) for e in snapshot.get("findings",[])):
            tags.append("消歧_矛盾")
        historical_conflicts = {e["finding"].casefold() for snapshot in traj.get("evidence_history", [])
                                for e in snapshot.get("findings", []) if e.get("status") == "conflict"}
        if any(e["finding"].casefold() in historical_conflicts and e.get("status") in ("present", "absent") for e in entries):
            tags.append("消歧_已解决")
        if traj.get("termination", {}).get("reason") != "complete" or traj.get("report_assessment", {}).get("misalign", 0) > 0:
            tags.append("失败候选")
        entry = {"id": rec.get("id") or rec.get("sample_id"),
                 "tags": tags,
                 "n_tools": len(traj.get("tool_calls", []) or []),
                 "findings": (traj.get("findings") or "")[:300],
                 "step": rec.get("step"), "reference": rec.get("report_ref", traj.get("report_ref", "")),
                 "termination": traj.get("termination"), "evidence_history": traj.get("evidence_history", []),
                 "tool_calls": traj.get("tool_calls", []), "report_assessment": traj.get("report_assessment")}
        if tags:
            for t in tags:
                buckets[t].append(entry)
        else:
            unlabelled.append(entry)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Case Study 候选案例\n\n")
        f.write(f"共 {len(records)} 条轨迹，按行为特征分类。\n\n")
        for tag in buckets:
            f.write(f"## 特征：{tag}\n")
            for e in buckets[tag][: args.top]:
                f.write(f"- **id={e['id']}** | tools={e['n_tools']} | tags={','.join(e['tags'])}\n")
                f.write(f"  findings: {e['findings']}\n")
                f.write("\n```json\n" + json.dumps(e, ensure_ascii=False, indent=2) + "\n```\n")
            f.write("\n")
        f.write("## 未标注（需人工看是否算失败案例）\n")
        for e in unlabelled[:10]:
            f.write(f"- id={e['id']} | tools={e['n_tools']} | findings: {e['findings']}\n")
        f.write("\n> 提示：以上为半自动候选。定稿前请结合完整 trajectory 复核，"
                "确保对应论文 (a)(b)(c)(d) 四类。\n")
    logger.info("case candidates -> %s", args.out)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()