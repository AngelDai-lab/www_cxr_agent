# WWW 胸片 Agent 实验执行安排

## 1. 实验准备

### 1.1 数据集路径

路径统一写在 `configs/paths.yaml`，脚本从配置读取。单张后前位或后前位+侧位双视角均可。

#### 1.1.1 MIMIC-CXR

| 项 | 路径 |
|----|------|
| 数据根目录 | `/data1t/mimic_cxr/mimic_cxr` |
| 影像 | `/data1t/mimic_cxr/mimic_cxr/images` |
| 划分 + 报告 | `/data1t/mimic_cxr/mimic_cxr/annotation.json` |
| 14 类标签 | `/data1t/mimic_cxr/mimic_cxr/labels_14.pickle` |

```text
/data1t/mimic_cxr/mimic_cxr/
├── images/
├── annotation.json
├── labels_14.pickle
├── mimic-cxr-label-LL…
├── init_prototypes.pt
└── test_label_from_ve.pth
```

**划分：** `annotation.json` 顶层 `train` / `val`（或 `valid`）/ `test`；剔除无报告样本。

**加载：** 影像绝对路径 = `images/` + `image_path[i]`（保留全部视图）。

**`annotation.json` 格式：**

```json
{
  "train": [
    {
      "id": "174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962",
      "study_id": "50414267",
      "subject_id": "10000032",
      "report": "The lungs are clear. ...",
      "image_path": [
        "p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"
      ],
      "split": "train"
    }
  ],
  "val": [],
  "test": []
}
```

- **Findings：** 字段 `report` → 下游 `findings`。
- **标签（CheXpert-14）：** `labels_14.pickle` 为「样本键 → 14 维 0/1 向量」；用 `id`（否则 `study_id`）查表，查不到则剔除。类名顺序见 §1.1.3。

#### 1.1.2 IU_xray

| 项 | 路径 |
|----|------|
| 数据根目录 | `/data1t/cxr/iu_xray` |
| 报告 | `/data1t/cxr/iu_xray/reports_ori.json` |

```text
/data1t/cxr/iu_xray/
├── reports_ori.json
├── CXR****_IM-****-*.png
└── …
```

**划分：** 患者级 **7:1:2**（train/val/test）；剔除无报告样本。跨集评测。

**加载：** `reports_ori.json` + 同目录 png；绝对路径 = 根目录 + `image[i]` + `.png`（按列表加载全部视图）。

**`reports_ori.json` 格式：**

```json
{
  "<report_id>": {
    "image": ["CXR3809_IM-1919-1001", "CXR3809_IM-1919-2001"],
    "report": {
      "FINDINGS": "...",
      "IMPRESSION": "...",
      "INDICATION": "...",
      "COMPARISON": "..."
    }
  }
}
```

- **Findings：** `report.FINDINGS` → 下游 `findings`。

```yaml
# configs/paths.yaml → paths.data（§1.1）
mimic_cxr: /data1t/mimic_cxr/mimic_cxr
mimic_annotation: /data1t/mimic_cxr/mimic_cxr/annotation.json
mimic_labels_14: /data1t/mimic_cxr/mimic_cxr/labels_14.pickle
mimic_images: /data1t/mimic_cxr/mimic_cxr/images
iu_xray: /data1t/cxr/iu_xray
iu_reports: /data1t/cxr/iu_xray/reports_ori.json
```

#### 1.1.3 CheXpert-14 类名顺序

`configs/label_map_14.json` 与 MIMIC `labels_14.pickle`、训练奖励 $F1_{14}$ 共用同一顺序：

```json
["Atelectasis","Cardiomegaly","Consolidation","Edema","Enlarged Cardiomediastinum","Fracture","Lung Lesion","Lung Opacity","No Finding","Pleural Effusion","Pleural Other","Pneumonia","Pneumothorax","Support Devices"]
```

- **训练奖励 $F1_{14}$（MIMIC）：** GT 用样本 `id`（没有则用 `study_id`）从 `labels_14.pickle` 取 14 维向量；生成侧 CheXbert 抽 14 类再算 F1。  
- 字典无该键：样本剔除并记日志。

---

### 1.2 GPU

| 项 | 约定 |
|----|------|
| 资源 | 复旦服务器 24G；不足则更大显存 |
| 策略模型 | Qwen3-14B-Instruct |
| 加载 | 工具与策略模型错峰；避免多工具与策略模型同时占满显存 |

**显存验证（正式训练前）**

环境就绪后，完成一次最小链路验证：加载策略模型、按需加载 1～2 个 CXR 工具权重，MIMIC / IU 各取少量样本完成一次前向或短 rollout。确认模型、数据与工具同时驻留时是否 OOM，并区分瓶颈在策略模型、工具，或两者叠加。

**若 OOM，优先调参，再考虑换机器。** 依次尝试并记录每次改动的显存峰值与是否通过：

- 降低 **batch_size** / **rollout**（试跑可用 batch=1、rollout=2，见 §1.6 试跑规模）
- **工具与策略错峰加载**（先启动工具 MCP，策略推理时再占用 GPU；或单卡只跑策略、工具放 CPU/另一卡）
- 策略侧 **LoRA** 推理；冻结模型必要时 **4bit/8bit** 加载（实现支持时）
- 缩短序列长度、减少同时启用的工具数，先跑最小工具集

上述仍无法满足 §1.6 正式训练 batch/rollout 时，再换更大显存服务器。

在其他自备服务器（可选 AutoDL）有网环境创建运行环境：

依赖：`requirements_www_cxr_agent.txt`（Python 3.11）。

```bash
conda create -n www_cxr_agent python=3.11 -y
conda activate www_cxr_agent

# 按机器 CUDA 改 cu121；CUDA 11.8 用 cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cd /path/to/www-cxr-agent
pip install -r requirements_www_cxr_agent.txt
```

复旦侧仍用离线包，见 §1.5.1。

### 1.3 模型

下列为跑通本方法需准备的权重与工具；**下载与装环境在有网机完成**（§1.2 / §1.5.1），复旦只接拷贝包。

#### 1.3.1 策略模型（可训练，LoRA）

| 模型 | 用途 | HuggingFace | 本地目录 |
|------|------|-------------|----------|
| **Qwen3-14B-Instruct** | 策略 / Agent 规划 | [Qwen/Qwen3-14B-Instruct](https://huggingface.co/Qwen/Qwen3-14B-Instruct) | `models/Qwen3-14B-Instruct` |

同一权重目录另供 `cxr_negation_checker`、奖励 LLM judge 冻结推理（不挂 LoRA、不参与 GRPO）。

#### 1.3.2 工具

工具权重冻结。

**Agent 工具箱**

| 工具 | 功能 | 实现 | 权重下载 | MCP 注册名 |
|------|------|------|----------|------------|
| 影像预处理 | Step 1：原始图 → 标准 PNG | `tools/cxr_preprocess.py` | — | `cxr_preprocess` |
| 病理筛查 | 概率 → $\mathcal{H}$ | `tools/cxr_patho_classifier.py` | [densenet121-res224-all](https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt) | `cxr_patho_classifier` |
| 报告草稿（可选） | 参考草稿，不作终稿 | `tools/cxr_report_draft.py` | [findings](https://huggingface.co/IAMJB/chexpert-mimic-cxr-findings-baseline) · [impression](https://huggingface.co/IAMJB/chexpert-mimic-cxr-impression-baseline) | `cxr_report_draft` |
| 单视图 VQA | 单张核验 | `tools/cxr_vqa.py`（与双视图同进程 7004） | [CheXagent-2-3b](https://huggingface.co/StanfordAIMI/CheXagent-2-3b) | `cxr_vqa` |
| 双视图 VQA | PA+LAT | 同上 | 同上 | `cxr_multiview_vqa` |
| 解剖分割 | mask + metrics | `tools/segment_2d.py` | [chestx-det](https://github.com/mlmed/torchxrayvision/releases/download/v1/pspnet_chestxray_best_model_4.pth) | `segment_2d` |
| 否定句校验 | 句↔证据 | `tools/llm_judge_tool.py` | [Qwen3-14B-Instruct](https://huggingface.co/Qwen/Qwen3-14B-Instruct) | `cxr_negation_checker` |

权重落盘：HF → `models/`；xrv → `cache/torchxrayvision/`（§1.5.1）。

**CXR 诊断清单（7 项）**

第二层须逐条覆盖；结果写入证据缓存与 `checklist_checked`。

| # | 清单条目 | 检查要点 |
|---|----------|----------|
| 1 | 气道与纵隔轮廓 | 气管居中与管壁形态；纵隔轮廓、移位、增宽 |
| 2 | 肺野 | 双肺透亮度；实变 / 浸润 / GGO；结节 / 肿块；气胸；间质纹理 |
| 3 | 胸膜 | 胸腔积液、肋膈角钝化；胸膜增厚 / 钙化 |
| 4 | 心脏 | 心脏轮廓；心胸比；肥大相关征象 |
| 5 | 横膈 | 位置、形态、两侧高度对比；膈下游离气体；膈面与肋膈角 |
| 6 | 骨骼系统 | 肋骨、锁骨、胸椎；骨折、骨质破坏、增生 |
| 7 | 植入设备与体外装置 | 导管、CVC、起搏器 / ICD 电极、手术夹；位置与并发症 |

**策略决策格式**

工具列表由 `ALL_TOOL_CONFIGS` 写入 system prompt。策略模型每轮输出一个 JSON 对象（纯文本）。`action` 为 `call_tool` 时由 `server_manager` 调用对应 MCP，返回写入轨迹后再进入下一轮；`action` 为 `final_answer` 时结束。工具注册见 `www_cxr_agent/tools/tool_configs.py`。

调用工具：

```json
{
  "reasoning": "…",
  "action": "call_tool",
  "tool_name": "<MCP 注册名>",
  "arguments": { "<param>": "<value>" }
}
```

结束推理：

```json
{
  "reasoning": "…",
  "action": "final_answer",
  "findings": "<最终 Findings 文本>"
}
```

**入参 / 出参**

| MCP 注册名 | 入参 `arguments` | 出参（写入 `tool_calls[].result`） |
|------------|------------------|-------------------------------------|
| `cxr_preprocess` | `image`：str | `image`：str，预处理后 PNG 路径 |
| `cxr_patho_classifier` | `image`：str | `scores`：`dict[str, float]` |
| `cxr_report_draft` | `image`：str | `text`：str（不作终稿） |
| `cxr_vqa` | `image`：str；`query`：str | `answer`：str |
| `cxr_multiview_vqa` | `image_pa`：str；`image_lat`：str；`query`：str | `answer`：str |
| `segment_2d` | `image`：str；`organs`：str[]（可选） | `mask`：str；`metrics`：dict |
| `cxr_negation_checker` | `candidate_sentence`：str；`evidence_set`：str[] 或 JSON 串 | `score`：0 \| 1；`rationale`：str |

**注册与启动**

| 项 | 位置 |
|----|------|
| `SERVERS` + `ALL_TOOL_CONFIGS` | `www_cxr_agent/tools/tool_configs.py`（端口 7001–7007） |
| 启动脚本 | `launch/start_tool_node.sh` → `runs/tool_server_registry.json` |
| HTTP 调用 | `www_cxr_agent/agents/server_manager.py` |
| Agent 主循环 | `www_cxr_agent/agents/layered_agent.py` |

**两层分工（方法稿模块一）**

| 层 | 目标 |
|----|------|
| 第一层 | 调用外部病理筛查类工具做初筛，得到假设集 $\mathcal{H}$；本层不生成完整报告 |
| 第二层 | 对 $\mathcal{H}$ 调度外部视觉工具确认 / 驳回；对照诊断清单查漏；矛盾时追加调用消歧；证据写入缓存后聚合为 Findings |

工具箱全体对策略开放；调用由策略模型规划。策略经 `server_manager` 调工具，不在策略权重内嵌视觉模型。

#### 1.3.3 评价指标

**（1）报告临床质量指标（主指标）**

| 指标 | 定义 | 计算 |
|------|------|------|
| **Report CE-F1** | 报告临床有效性；CheXpert 十四类标签一致性 | 参考 / 生成 Findings 各抽 CheXpert-14 标签，**macro F1**；test 平均 |
| **RadGraph F1** | RadGraph 实体–关系匹配 | RadGraph 对参考 / 生成 Findings 匹配 F1；test 平均 |

**（2）NLG 指标（辅助）**

| 指标 | 计算 |
|------|------|
| **BLEU-1** | 参考 / 生成 Findings，test 平均 |
| **BLEU-4** | 同上 |
| **ROUGE-L** | 同上 |
| **METEOR** | 同上 |

**（3）可追溯性 / 证据对齐**

| 指标 | 定义 | 计算 |
|------|------|------|
| **证据失配率** $P_{\mathrm{misalign}}$ | 报告中无工具证据支撑的异常发现占比（幻觉率） | 生成 Findings 异常发现集合；逐条查轨迹证据缓存；无支撑条数 / 异常发现总数；test 平均 ×100 |
| **证据覆盖率** | 有证据支撑的发现占报告全部发现的比例 | 有证据条数 / 异常发现总数；或 $1 - P_{\mathrm{misalign}}$ |

**（4）假阳性率与漏诊率**

| 指标 | 计算 |
|------|------|
| **假阳性率（FP）** | CheXpert-14：生成=1 且参考=0；14 类 macro 平均；×100 |
| **漏诊率（Miss）** | CheXpert-14：参考=1 且生成=0；14 类 macro 平均；×100 |

**（5）清单覆盖率**

| 指标 | 定义 | 计算 |
|------|------|------|
| **清单覆盖率** | 7 项解剖条目已检查比例 | 已检查数 / 7；test 平均 ×100 |

七项：气道/纵隔、肺野、胸膜、心脏、横膈、骨骼、植入物/体外装置。

**（6）工具行为**

由轨迹 `tool_calls` 统计；test 集对各例取平均后汇报。

| 量 | 含义 | 轨迹统计 |
|----|------|----------|
| $N_{\mathrm{call}}$ | 工具调用总次数 | `len(tool_calls)` |
| $N_{\mathrm{succ}}$ | 成功执行的工具调用次数 | `status=="ok"` 计数 |
| $N_{\mathrm{used}}$ | 实际使用的不同工具种类数目 | `tool_name` 去重计数 |
| $N_{\mathrm{avail}}$ | 全部可用工具种类总数 | MCP 工具箱 = 7 |
| $N_{\mathrm{coh}}$ | 输出结果被后续推理步骤实际引用的有效工具调用数目 | 轨迹标注 / 引用关系计数 |

| 汇报量 | 计算 |
|--------|------|
| $R_{\mathrm{succ}}=N_{\mathrm{succ}}/N_{\mathrm{call}}$ | 调用成功率 |
| $R_{\mathrm{div}}=N_{\mathrm{used}}/N_{\mathrm{avail}}$ | 工具多样性 |
| $R_{\mathrm{coh}}=N_{\mathrm{coh}}/N_{\mathrm{call}}$ | 有效调用率 |

另记：单例墙钟耗时（秒）、策略侧 token 消耗。

---

### 1.4 对比 Baseline

三类对比，对应三个维度优势（《WWW实验.pdf》§二）。

| 类别 | 方法 | 论文 | 代码 / 权重 |
|------|------|------|-------------|
| **Ours** | 两级 Agent + GRPO | 方法稿 | 本工程；策略 `models/Qwen3-14B-Instruct` + LoRA（`runs/`） |
| **① 传统端到端** | R2Gen（EMNLP 2020） | [ACL](https://aclanthology.org/2020.emnlp-main.112/) | [cuhksz-nlp/R2Gen](https://github.com/cuhksz-nlp/R2Gen)；MIMIC 权重见仓库 |
| | R2GenCMN（ACL 2021） | [ACL](https://aclanthology.org/2021.acl-long.459/) | [cuhksz-nlp/R2GenCMN](https://github.com/cuhksz-nlp/R2GenCMN) |
| | PPKED（CVPR 2021） | [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Exploring_and_Distilling_Posterior_and_Prior_Knowledge_for_Radiology_Report_CVPR_2021_paper.html) | 无官方公开代码 / 权重 |
| | METransformer（CVPR 2023） | [arXiv](https://arxiv.org/abs/2304.02211) | 无官方公开代码 / 权重 |
| | ORGAN（ACL 2023） | [ACL](https://aclanthology.org/2023.acl-long.451/) | [wjhou/ORGan](https://github.com/wjhou/ORGan) |
| | PromptMRG（AAAI 2024） | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28038) | [jhb86253817/PromptMRG](https://github.com/jhb86253817/PromptMRG) |
| | MLRG（CVPR 2025） | [arXiv](https://arxiv.org/abs/2502.20056) | [mk-runner/MLRG](https://github.com/mk-runner/MLRG) |
| | PriorRG（AAAI 2026） | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37657) | [mk-runner/PriorRG](https://github.com/mk-runner/PriorRG)；HF checkpoint |
| **② 大模型 / VLM** | CheXagent（CheX-VLM 类） | [arXiv](https://arxiv.org/abs/2401.12208) | [Stanford-AIMI/CheXagent](https://github.com/Stanford-AIMI/CheXagent)；HF `StanfordAIMI/CheXagent-2-3b` |
| | Med-Gemini | [Nature](https://www.nature.com/articles/s41586-024-08452-9) 等 | 无公开可下载权重 |
| | MAIRA-2（Microsoft, 2024） | [arXiv](https://arxiv.org/abs/2406.04449) | HF `microsoft/maira-2` |
| | RaDialog | [arXiv](https://arxiv.org/abs/2311.18681) | [ChantalMP/RaDialog_v2](https://github.com/ChantalMP/RaDialog_v2)；HF 权重 |
| | MedRAT | [arXiv](https://arxiv.org/abs/2407.03919) | [Eladhi/MedRAT](https://github.com/Eladhi/MedRAT) |
| | LLM-CXR（ICLR 2024） | [arXiv](https://arxiv.org/abs/2305.11490) | [hyn2028/llm-cxr](https://github.com/hyn2028/llm-cxr) |
| | MIMIC-Diff-VQA | [KDD 2023](https://dl.acm.org/doi/10.1145/3580305.3599819) | [Holipori/EKAID](https://github.com/Holipori/EKAID) |
| | KG4RM / R2GenKG | [arXiv](https://arxiv.org/pdf/2508.03426) | [Event-AHU/Medical_Image_Analysis](https://github.com/Event-AHU/Medical_Image_Analysis) |
| **③ Agent + 推理** | ReAct | 同策略模型 + §1.3.2 工具箱；单层 ReAct；无 GRPO | 本工程 |
| | CoT / Reflection | 同策略模型；无 MCP 工具 | 本工程 |
| | Tool-use without RL | 完整两级 Agent；仅 prompting，无 GRPO | 本工程 |

#### 1.4.4 消融（《WWW实验.pdf》§三）

| 消融项 | 设置 | 验证目标 |
|--------|------|----------|
| 去掉分层结构 | 单层 ReAct 直接规划调用 | 「假设生成→验证→查漏」分层设计的必要性（**核心消融**） |
| 去掉 GRPO 训练 | 仅 prompting 策略 | RL 训练的作用（**核心消融**） |
| 去掉 $P_{\mathrm{misalign}}$ 惩罚项 | $\lambda_{\mathrm{mis}}=0$ | 证据失配惩罚对幻觉率的抑制效果 |
| （补充）去掉诊断清单查漏 | 第二层不做解剖条目核对 | 清单约束对覆盖率的贡献 |
| （补充）去掉证据消歧 | 工具矛盾时不追加调用 | 消歧机制的作用 |
| （补充）去掉工具行为奖励 $R_{\mathrm{tool}}$ | 仅用 $R_{\mathrm{quality}}$ 训练 | 工具行为奖励的必要性 |
| （补充）逐项去掉 $R_{\mathrm{tool}}$ 子奖励 | 分别去掉 $R_{\mathrm{succ}}$ / $R_{\mathrm{div}}$ / $R_{\mathrm{coh}}$ / $R_{\mathrm{toolJudge}}$ | 各子奖励的贡献 |
| （补充）双视角→单视角 | 仅 PA 输入 | 多视角工具的价值 |

标「（补充）」的项放入论文补充材料。

---

### 1.5 代码基础

#### 1.5.1 运行环境

复旦服务器**不能联网**；运行环境直接用离线包（**配置与安装包**可询问 @刘亦宸-大二-13788953154）。

**有网机**只准备代码、模型、工具；评测依赖（CheXbert、RadGraph 等）若安装有问题，可先在有网机 **CPU 本地跑通**。

| 产物 | 内容 |
|------|------|
| `www_cxr_agent.tar.gz` | 离线 conda 环境包 |
| `models/` | Qwen3-14B-Instruct、CheXagent-2-3b、report draft 等 |
| `cache/torchxrayvision/` | 分类 / 分割权重 |
| `third_party/MedRAX/` | 工具源码 |
| 本仓库 | 代码 + `configs/` |

#### 1.5.2 项目架构

目录、configs、MCP 注册、CLI 入口与轨迹/评测接口已搭好；各模块按 §1.3 / §1.6 继续实现。

链路：**数据 → 冻结工具箱 → 两级 Agent → GRPO 训 LoRA → 统一评测**。

```text
www-cxr-agent/
├── WWW实验执行安排.md          # 本执行文档
├── configs/                    # 路径、训练、Agent、评测、消融 yaml（默认已填）
├── www_cxr_agent/
│   ├── config/                 # 读 configs
│   ├── data/                   # MIMIC / IU loader（仅原始路径 + 标签）
│   ├── tools/                  # §1.3.2 七 MCP；tool_configs 注册（已就绪）
│   ├── agents/                 # layered_agent、server_manager、trajectory
│   ├── reward/                 # R_align 分项
│   ├── train/                  # GRPO + LoRA
│   ├── eval/                   # §1.3.3 六类 metrics + runner
│   └── baselines/              # 外部预测 → 统一 JSONL
├── scripts/                    # infer · train_grpo · eval · smoke_gate
├── launch/                     # 按 SERVERS 启 MCP 进程
├── third_party/MedRAX/
├── models/   cache/   runs/
```

| 模块 | 已有 | 待补 |
|------|------|------|
| `configs/` | 默认 yaml/json、消融模板 | 路径：`paths.yaml`；prompt：`reward_judge.yaml` |
| `data/` | `mimic_cxr.py` / `iu_xray.py` / `checklist.py` | — |
| `tools/` | MCP 端口/schema、`tool_configs.py` | 各工具后端（§1.3.2） |
| `agents/` | `trajectory`、factory、layered / ReAct / CoT 模块 | 两级主循环 + JSON 调 MCP |
| `reward/` + `train/` | 模块划分、trainer 入口 | $R_{\mathrm{align}}$ 与 GRPO（§1.6） |
| `eval/` + `scripts/eval.py` | metrics 分文件、runner 入口 | 六类指标聚合（§1.3.3） |
| `scripts/` | infer / train_grpo / eval / smoke_gate 入口 | 串 data → agent → train/eval |
| `baselines/` | adapter 入口 | §1.4 外部方法输出转 JSONL |

**主方法 Agent 循环**

- **第一层**：病理筛查得 $\mathcal{H}$；不生成完整报告  
- **第二层**：视觉工具核验 / 清单查漏 / 消歧；证据缓存对齐后输出 Findings  
- **策略**：每轮 JSON `call_tool` / `final_answer` → rollout → GRPO 更新 LoRA  

**推理产物（eval 统一输入）**

```json
{"id": "...", "findings": "...", "trajectory": {"tool_calls": [], "evidence_cache": [], "checklist_checked": []}}
```

---

### 1.6 参数配置

写入 `configs/train_default.yaml`（名称可自定）。

| 配置项 | 默认值 |
|--------|--------|
| 策略模型 | Qwen3-14B-Instruct |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| rollout | 8 |
| batch size | 6 |
| 学习率 | $1\times 10^{-5}$ |
| 最大步数 | 150 |
| 早停 | 开 |
| 工具权重 | 冻结 |
| $\lambda_{\mathrm{mis}}$（探索期 / 收敛期） | 0.2 / 0.3 |
| $(w_s,w_d,w_c,w_j)$（探索期） | $(0.1,\ 0.5,\ 0.5,\ 0)$ |
| $(w_s,w_d,w_c,w_j)$（收敛期） | $(0.1,\ 0.2,\ 0.2,\ 0.2)$ |
| $F1_{\mathrm{abnorm}}$ / $R_{\mathrm{toolJudge}}$ 评判器 | `models/Qwen3-14B-Instruct`（冻结）；prompt → `configs/reward_judge.yaml` |
| $F1_{14}$ GT | `labels_14.pickle`；生成侧 CheXbert |

**试跑规模：**

| 项 | 值 |
|----|----|
| 样本 | 约 10～50 |
| rollout | 2 |
| batch | 1～2 |
| 步数 | 数步～10 步 |

---

## 2. 实验安排（哪几天做什么）

| 天数 | 阶段 | 做什么 | 完成标准 |
|------|------|--------|----------|
| 1～5 | 环境 · 显存验证 | 配置 `configs/paths.yaml`（§1.1）；启动 MCP + Qwen3；MIMIC / IU 少量样本最小链路（§1.2） | 最小链路通过；显存峰值有记录 |
| 6～10 | 代码 · 试跑 | §1.3.2 工具 MCP + 两级 Agent + 奖励/GRPO + 训推评脚本；单工具 → 少量推理 → §1.6 短训门禁 | 单张图出 Findings+轨迹；试跑三关通过 |
| 11～16 | 主方法 | 默认超参正式训练；MIMIC test + IU test；冻检查点 | 检查点冻结；MIMIC / IU 结果表各一行 Ours |
| 17～22 | Baseline | §1.4 三类 baseline 各跑 test；同一 §1.3.3 评测脚本 | 结果表补全 baseline 行 |
| 23～27 | 消融 · 超参 | §1.4.4（含补充）；PDF §四两阶段课表 + $\lambda_{\mathrm{mis}}$ 敏感性 | 消融表 + 超参曲线 |
| 28～30 | 效率 · 案例 · 定稿 | §1.3.3（6）工具行为；案例图 3～4 例；数字对表定稿 | 效率表、案例图、结果/消融表冻结 |

根据实验时间灵活调整。

---

## 3. 交付与验收（摘要）

| 交付物 | 说明 |
|--------|------|
| 配置文件 | 路径、超参 |
| 数据统计 | MIMIC / IU 加载口径与剔除日志 |
| 试跑日志 | 门禁三关 |
| 默认检查点 | 主方法 |
| 结果表 | MIMIC / IU test；§1.4 各类 + Ours |
| 消融 + 超参 | §1.4.4 + PDF §四 |
| 效率表 + 案例图 | $R_{\mathrm{succ}}$ / $R_{\mathrm{div}}$ / $R_{\mathrm{coh}}$；轨迹案例 3～4 例 |

| 编号 | 验收 | 标准 |
|------|------|------|
| V-01 | 门禁 | 试跑三关有记录 |
| V-02 | 可追溯 | 结果表每行对应唯一配置与预测产物 |
| V-03 | 公平 | §1.3.3 |
| V-04 | 资源 | 24G；不足则更大显存 |

---
