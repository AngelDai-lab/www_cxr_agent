#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_sweep.py — 显存压力测试（WWW CXR Agent）

在指定的 batch_size 序列下，对 Qwen3-14B（4bit NF4）跑一次固定长度(步)的前向，
逐档记录：
  - torch.cuda.max_memory_allocated()   当前分配峰值
  - torch.cuda.memory_reserved()        保留显存
  - torch.cuda.max_memory_reserved()    保留峰值
  - 是否 OOM，OOM 时的信息

用途：定位 24GB 显卡在 GRPO rollout（batch * rollout）下的显存边界，
     为训练配置（batch、rollout、max steps）提供依据。

用法：
    cd /root/autodl-tmp/www-cxr-agent
    python scripts/batch_sweep.py --batches 1,2,4,6,8 --max-new-tokens 512
"""
import argparse
import gc
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 压掉 TensorFlow oneDNN 日志

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "/root/autodl-tmp/models/Qwen3-14B-Instruct"
PROMPT = (
    "You are a chest X-ray analysis agent. Respond with a JSON object. "
    "Give a short clinical findings paragraph about the imaged chest X-ray.\n"
    "Image: /root/autodl-tmp/datasets/mimic_cxr/mimic_cxr/images/p10/p10000032/s50414267/"
    "02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
)


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # 避免 pad==eos 导致的 attention mask 警告
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tok, model


def format_batch(tokenizer, texts):
    msgs = [tokenizer.apply_chat_template(
        [{"role": "user", "content": t}],
        tokenize=False, add_generation_prompt=True,
    ) for t in texts]
    enc = tokenizer(msgs, return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"]  # 显式返回 attention_mask


def run_batch(tok, model, batch_size: int, seq_len: int, max_new_tokens: int):
    """对 batch 个真实 prompt 跑一次前向生成，返回 (batch, 峰值分配, OOM为None则还有错误)。"""
    texts = [PROMPT] * batch_size
    input_ids, attention_mask = format_batch(tok, texts)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    before = torch.cuda.max_memory_allocated(device=model.device)
    torch.cuda.reset_peak_memory_stats(device=model.device)

    t0 = time.time()
    try:
        with torch.no_grad():
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        peak = torch.cuda.max_memory_allocated(device=model.device)
        reserved = torch.cuda.memory_reserved(device=model.device)
        max_reserved = torch.cuda.max_memory_reserved(device=model.device)
        return {
            "batch": batch_size,
            "peak_allocated_MB": round(peak / 1024**2, 1),
            "reserved_MB": round(reserved / 1024**2, 1),
            "max_reserved_MB": round(max_reserved / 1024**2, 1),
            "profile_increase_MB": round((peak - before) / 1024**2, 1),
            "sec": round(time.time() - t0, 2),
            "oom": False,
            "total_MB": round(torch.cuda.get_device_properties(model.device).total_memory / 1024**2, 1),
        }
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated(device=model.device)
        return {
            "batch": batch_size,
            "peak_allocated_MB": round(peak / 1024**2, 1),
            "oom": True,
            "note": "torch.cuda.OutOfMemoryError",
            "total_MB": round(torch.cuda.get_device_properties(model.device).total_memory / 1024**2, 1),
        }
    except Exception as e:
        return {
            "batch": batch_size,
            "peak_allocated_MB": round(torch.cuda.max_memory_allocated(device=model.device) / 1024**2, 1),
            "oom": True,
            "note": f"{type(e).__name__}: {e}",
        }
    finally:
        # 清空该 batch 的缓存（huggingface 会在三次后抱怨；这里主动 gc）
        del input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,4,6,8")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    tok, model = load_model()
    total = torch.cuda.get_device_properties(model.device).total_memory / 1024**2
    print(f"模型加载完成。设备可用显存: {total:.0f} MB")
    print(f"4bit 模型权重占用(初始): {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    print("=" * 70)
    print(f"{'batch':<6}{'生成/s':<10}{'峰值分配MB':<14}{'峰值保留MB':<14}{'增量MB':<10}{'结果':<10}")
    print("-" * 70)

    for b in batches:
        r = run_batch(tok, model, b, seq_len=0, max_new_tokens=args.max_new_tokens)
        if r["oom"]:
            print(f"{b:<6}{'-':<10}{r['peak_allocated_MB']:<14}{'-':<14}{'-':<10}{'OOM ' + r.get('note','')}")
            print(f"\n>>> batch={b} 触发 OOM，24GB 边界在 batch < {b} 附近。UP 到 {int(total)}MB。")
            break
        sec_s = f"{r['sec']:.1f}"
        inc_s = f"+{r['profile_increase_MB']}"
        print(f"{b:<6}{sec_s:<10}{r['peak_allocated_MB']:<14}{r['max_reserved_MB']:<14}{inc_s:<10}{'OK'}")
        print(f"      (保留 {r['reserved_MB']:.0f} MB / 总量 {r['total_MB']:.0f} MB，占比 {100*r['reserved_MB']/r['total_MB']:.0f}%)")

    print("=" * 70)


if __name__ == "__main__":
    main()