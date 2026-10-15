# -*- coding: utf-8 -*-
"""消融：双视角 -> 单视角（仅 PA 输入）数据侧脚本。

背景（§消融清单"双视角→单视角，仅 PA 输入"）：
  MimicCXRDataset.samples 里 image_paths 保留全部视图（PA + lateral 等）。
  本脚本把每个样本裁剪为只保留第一张（PA）视图，导出新的单视角 JSONL，
  供 infer.py / eval.py / train 用统一 data loader 读取。

核心不碰训练模型，纯数据预处理，可在 GPU 被 base 占用时安全运行。

用法（服务器 base 环境，必须用 local 镜像同步后的脚本）：
    python scripts/make_single_view.py \
        --ann /数据/mimic_cxr/annotation.json \
        --out runs/single_view.jsonl

说明：
  - 保留每条样本其余字段（id / labels_14 / report_ref / split 等）不动；
  - image_paths 只留列表第一个元素（假定为 PA 前位视图，实际按你的注释格式；
    若 PA 不是首位，请改 PA_INDEX 或用 path 含 'PA' 匹配）；
  - 输出每行一个 dict：{**原样本字段, "image_paths": [pa_path]}，
    数据 loader 读到它就只喂一张 PA 图 -> 单视角。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)


def load_annotation(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # 若是 jsonl 请改成逐行读


def keep_pa(image_paths: list[str], pa_index: int = 0) -> list[str]:
    """默认取首位视图为 PA。若注释里 PA 不是第一位，传 pa_index；若按文件名含'PA'，
    可自行改未注释逻辑按关键字过滤。"""
    if not image_paths:
        return []
    return [image_paths[pa_index]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="MIMIC 注释 json（含 image_paths）")
    ap.add_argument("--out", default="runs/single_view.jsonl")
    ap.add_argument("--pa-index", type=int, default=0,
                    help="PA 视图在 image_paths 中的下标（默认0）")
    args = ap.parse_args()

    samples = load_annotation(args.ann)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    kept = dropped = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            paths = s.get("image_paths") or []
            pa = keep_pa(paths, args.pa_index)
            if not pa:
                dropped += 1
                continue
            rec = dict(s)
            rec["image_paths"] = pa
            rec["n_views"] = 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    logger.info("single-view done -> %s (kept=%d dropped=%d)",
                args.out, kept, dropped)


if __name__ == "__main__":
    main()