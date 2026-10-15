#!/usr/bin/env bash
# =============================================================
# compare_ckpt.sh —— 对比多个 LoRA checkpoint 在同一批测试图上的推理+评估
#
# 背景：评估 lora_step1 / lora_step3 / lora_step6 等不同训练步数的
#       权重在"同一批测试图"上的诊断效果，为正式训练的 checkpoint 选择提供依据。
#
# load_dataset 用 MimicCXRDataset.samples + limit=n 顺序取样（无 shuffle），
# 因此对同一 split + 同一 --n，各 checkpoint 推理用的是同一批图，对比公平。
#
# 用法（服务器 base 环境）：
#   bash scripts/compare_ckpt.sh \
#       --ckpts "lora_step1 lora_step3 lora_step6" \
#       --n 10 --phase full --split test
#
# 产出（在 runs/ 下）：
#   runs/cmp_<ckpt>_infer.jsonl   每个 checkpoint 的推理 JSONL
#   runs/cmp_<ckpt>_eval.json     每个 checkpoint 的评估结果
#   runs/compare_summary.json     汇总对比表（一个 JSON 文件，3个均放）
#   runs/compare_summary.txt      同一内容的人类可读版
# =============================================================
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp/www-cxr-agent}"
CDIR="$ROOT/runs"
CKPTS=""
N=10
PHASE="full"
SPLIT="test"
THRESH=0.30
MAXSTEPS=20
INFER_PY="$ROOT/scripts/infer.py"
EVAL_PY="$ROOT/scripts/eval.py"

usage() {
    echo "用法:"
    echo "  bash $0 --ckpts \"lora_step1 lora_step3 lora_step6\" [--n 10] [--phase full] [--split test] [--threshold 0.30] [--max-steps 20] [--root /root/autodl-tmp/www-cxr-agent]"
    exit 1
}

# ----- 解析参数 -----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpts) CKPTS="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --threshold) THRESH="$2"; shift 2 ;;
        --max-steps) MAXSTEPS="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

if [[ -z "$CKPTS" ]]; then
    echo "缺少 --ckpts（至少一个 checkpoint 目录名）"; usage
fi
[[ -d "$ROOT" ]] || { echo "目录不存在: $ROOT"; exit 1; }
mkdir -p "$CDIR"

echo "== 工作目录: $ROOT =="
echo "== 对比的 checkpoint: $CKPTS =="
echo "== 每批推理样本数: $N, phase: $PHASE, split: $SPLIT =="
echo

SUMMARY="{}"

for ck in $CKPTS; do
    CKPT_DIR="$ROOT/runs/checkpoints/$ck"
    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "[SKIP] checkpoint 目录不存在: $CKPT_DIR"
        continue
    fi
    echo "########## [$ck] 开始推理 (n=$N) ##########"
    INFER_OUT="$CDIR/cmp_${ck}_infer.jsonl"
    EVAL_OUT="$CDIR/cmp_${ck}_eval.json"

    python "$INFER_PY" \
        --ckpt "$CKPT_DIR" \
        --phase "$PHASE" \
        --n "$N" \
        --split "$SPLIT" \
        --threshold "$THRESH" \
        --max-steps "$MAXSTEPS" \
        --out "$INFER_OUT"
    echo "[$ck] 推理完成 -> $INFER_OUT"

    echo "########## [$ck] 开始评估 ##########"
    python "$EVAL_PY" --jsonl "$INFER_OUT" --out "$EVAL_OUT"
    echo "[$ck] 评估完成 -> $EVAL_OUT"
    echo

    # 把每个 ckpt 的 eval 结果并入汇总 JSON（用 python 合并，避免 shell 手拼 JSON）
    SUMMARY=$(python - "$SUMMARY" "$EVAL_OUT" "$ck" <<'PY'
import json, sys
summary = json.loads(sys.argv[1]) if sys.argv[1] not in ("{}","") else {}
try:
    with open(sys.argv[2]) as f:
        ev = json.load(f)
except Exception as e:
    print(f"[warn] 读取 {sys.argv[2]} 失败: {e}", file=sys.stderr); ev = {}
# 只保留评估里便于对比的核心指标
keys_subset = ["report_ce_f1","FP_rate","Miss_rate","P_misalign","checklist_coverage"]
sub = {k: ev.get(k) for k in keys_subset if k in ev}
if "tool" in ev:
    sub["R_succ"] = ev["tool"].get("R_succ")
    sub["R_div"]   = ev["tool"].get("R_div")
    sub["R_coh"]   = ev["tool"].get("R_coh")
summary[sys.argv[3]] = sub
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
)
done

# ----- 写汇总文件 -----
echo "$SUMMARY" > "$CDIR/compare_summary.json"
# 生成人类可读文本表
python - "$CDIR/compare_summary.json" "$CDIR/compare_summary.txt" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
lines = []
lines.append("Checkpoint 对比表（同一批测试图）")
lines.append("=" * 60)
metrics = ["report_ce_f1","FP_rate","Miss_rate","P_misalign","checklist_coverage","R_succ","R_div","R_coh"]
hdr = "{:<18}".format("metric")
for ck in data.keys():
    hdr += "{:>14}".format(ck)
lines.append(hdr)
for m in metrics:
    row = "{:<18}".format(m)
    for ck in data.keys():
        v = data[ck].get(m)
        row += "{:>14}".format("-" if v is None else f"{v:.3f}")
    lines.append(row)
text = "\n".join(lines)
print(text)
with open(sys.argv[2], "w") as f:
    f.write(text + "\n")
PY

echo
echo "========== 全部完成 =========="
echo "汇总文件: "
echo "  $CDIR/compare_summary.json"
echo "  $CDIR/compare_summary.txt"
echo
echo "读取方式: cat $CDIR/compare_summary.txt"