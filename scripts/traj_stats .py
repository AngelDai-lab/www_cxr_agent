# -*- coding: utf-8 -*-
"""轨迹审查统计（超参实验 #4）：从轨迹落盘 jsonl 统计轨迹长度 / 工具调用序列分布。

输入：轨迹落盘补丁产出的 runs/trajectories.jsonl（或 infer 产出的 *_infer.jsonl，若能截取 trajectory）。
输出：
  统计直方图数据（json）:
    - 轨迹长度分布（每条 rollout 的工具调用数 tool_calls 长度）
    - 工具名使用频次（工具调用序列分布 → 画柱状图/直方图）
    - 工具共现/转置（可选，用于桑基图的前后阶段对比）

用法（服务器 base 环境）：
    python scripts/traj_stats.py --jsonl runs/trajectories.jsonl --out runs/traj_stats.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True,
                    help="轨迹 jsonl（trajectories.jsonl 或 *_infer.jsonl）")
    ap.add_argument("--out", default="runs/traj_stats.json")
    args = ap.parse_args()

    records = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        logger.error("empty jsonl: %s", args.jsonl)
        sys.exit(1)

    traj_lens: Counter = Counter()      # tool_calls 长度分布
    tool_usage: Counter = Counter()     # 各工具被调用次数
    tool_order_1st: Counter = Counter()  # 第一个工具（序列起点）
    transitions: Counter = Counter()
    tool_last: Counter = Counter()      # 最后一个工具（序列终点）

    for rec in records:
        traj = rec.get("trajectory") or rec
        calls = traj.get("tool_calls", []) or []
        # 过滤掉非工具控制标记（如 __phase__）
        real = [t for t in calls if t.get("name") and t.get("name") != "__phase__"]
        traj_lens[len(real)] += 1
        for i, t in enumerate(real):
            tool_usage[t["name"]] += 1
        for left,right in zip(real,real[1:]):
            transitions[left["name"]+" -> "+right["name"]]+=1
        if real:
            tool_order_1st[real[0]["name"]] += 1
            tool_last[real[-1]["name"]] += 1

    parse = lambda c: {f"{k}": v for k, v in sorted(c.items())}  # noqa: E731
    stats = {
        "n_trajectories": len(records),
        "traj_length_hist": parse(traj_lens),
        "tool_usage_count": parse(tool_usage),
        "tool_first_usage": parse(tool_order_1st),
        "tool_last_usage": parse(tool_last),
        "tool_transitions": parse(transitions),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("stats done -> %s", args.out)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()