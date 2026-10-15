# 内网 8×A100 部署包

迁移内容：**同一份代码 + `configs/train_8gpu_internal.yaml` + `launch/start_8gpu_internal.sh` + 本说明 + 本地模型/数据/标签**。不依赖外网，MCP 工具绑定 `127.0.0.1`。

## 设备布局

- GPU0：Qwen3-14B BF16 LoRA 训练与 backward
- GPU1–4：4 个冻结 Qwen3 rollout 副本
- GPU5–6：2 个 CheXagent VQA；classifier 在 5，segment 在 6
- GPU7：1 个冻结 Qwen3 judge

A100 40GB 先做容量探针；80GB 才考虑提高 VQA/judge 并发，均须以 `nvidia-smi` 为准。

## 必须迁移

项目代码、Qwen3-14B、CheXagent-2-3b/XraySigLIP、classifier/segment/CheXbert/RadGraph 本地模型、MIMIC-CXR 图像/报告、`labels_14_official_study.pickle`。不要只复制 checkpoint；checkpoint 依赖相同基座、LoRA 配置、奖励实现和标签。

## 设置路径

```bash
export WWW_CXR_MIMIC_CXR_ROOT=/data/mimic_cxr/mimic_cxr
export CXR_POLICY_MODEL=/models/Qwen3-14B-Instruct
export CXR_LABELS14_PICKLE=$WWW_CXR_MIMIC_CXR_ROOT/labels_14_official_study.pickle
export CXR_CHEXBERT_DIR=/data/models/chexbert
```

## 启动

先做 3 条容量探针：

```bash
bash launch/start_8gpu_internal.sh --trial --n 3 --rollout 2 --batch 1 --max-steps 1
```

再做短训：

```bash
bash launch/start_8gpu_internal.sh --n 10 --rollout 4 --batch 2 --max-steps 5
```

正式训练（脱离 SSH）：

```bash
nohup bash launch/start_8gpu_internal.sh > logs/formal_8gpu_internal.console.log 2>&1 &
echo $! > logs/formal_8gpu_internal.pid
```

恢复：

```bash
bash launch/start_8gpu_internal.sh --resume-state runs/formal_8gpu_internal/checkpoints/trainer_state_stepN.pt
```

正式前必须完成：模型/标签存在性检查、`start_all_tools.sh status`、全工具 gate1、3–8 条 schema/reward 探针、GPU 峰值与无 OOM。trial 不能当正式结果。8 卡不会自动带来 8 倍加速。
