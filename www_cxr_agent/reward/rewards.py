# -*- coding: utf-8 -*-
"""Report-aligned reward: labels14 GT + CheXbert report labels.

Clinical labels come from frozen CheXbert over final reports. Independent report
assessment and actual cited tool outputs are required; hypotheses are not labels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)

# §1.3.3 与 MIMIC labels_14 共用顺序
REWARD_IMPL_VERSION = "labels14_gt_v5_official_study_reportmatch"
# Authoritative MIMIC labels_14 / configs/label_map_14.json order.
LABELS_14 = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
CHEXBERT_LABELS_14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


def chexbert_to_mimic14(labels: list[int | float], source_order=None) -> list[int]:
    """Convert CheXbert multi-label output into authoritative MIMIC order."""
    source = list(source_order or CHEXBERT_LABELS_14)
    if len(labels) != len(source):
        raise ValueError(f"CheXbert labels must have 14 dimensions, got {len(labels)}")
    if len(source) != 14 or len(set(source)) != 14 or any(v not in (0, 1) for v in labels):
        raise ValueError("Expected 14 unique label names and binary predictions")
    values = dict(zip(source, labels))
    if set(values) != set(LABELS_14):
        raise ValueError("CheXbert/MIMIC label names do not match")
    return [int(bool(values[name])) for name in LABELS_14]


# ---------- F1 ----------
def macro_f1(y_true: list[list[int]], y_pred: list[list[int]], n_classes: int = 14) -> float:
    """Macro-F1 over 14 independent binary labels (sample x class)."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("aligned nonempty multilabel arrays required")
    if any(len(x) != n_classes for x in y_true + y_pred):
        raise ValueError(f"expected {n_classes}-dimensional multilabel vectors")
    scores=[]
    for i in range(n_classes):
        tp=fp=fn=0
        for truth, pred in zip(y_true, y_pred):
            t=bool(truth[i]); p=bool(pred[i]); tp += t and p; fp += (not t) and p; fn += t and (not p)
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return float(sum(scores)/n_classes)


def label_metrics(gt_rows, pred_rows):
    """Shared binary-GT clinical metrics; cohort macro is not mean sample macro."""
    import numpy as np
    g, p = np.asarray(gt_rows), np.asarray(pred_rows)
    if g.ndim != 2 or g.shape != p.shape or g.shape[1] != 14 or not len(g):
        raise ValueError("Nonempty aligned N x 14 labels required")
    if not np.isin(g, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("Labels must be finite binary values; no silent threshold conversion")
    tp = ((g == 1) & (p == 1)).sum(0)
    fp = ((g == 0) & (p == 1)).sum(0)
    fn = ((g == 1) & (p == 0)).sum(0)
    tn = ((g == 0) & (p == 0)).sum(0)
    def divide(a, b):
        return np.divide(a, b, out=np.zeros(14, dtype=float), where=b > 0)
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    return {"F1_14": float(f1.mean()),
            "F1_14_sample_mean": float(np.mean([macro_f1([a.tolist()], [b.tolist()]) for a,b in zip(g,p)])),
            "F1_abnorm": float(np.mean([make_f1_abnorm_from_labels({"label14": a.tolist()}, {"pred_label14": b.tolist()}) for a,b in zip(g,p)])),
            "FP_rate": float(divide(fp, fp + tn).mean()),
            "Miss_rate": float(divide(fn, fn + tp).mean()),
            "per_label_f1": f1.tolist(), "tp": tp.tolist(), "fp": fp.tolist(),
            "fn": fn.tolist(), "tn": tn.tolist(), "label_order": LABELS_14,
            "gt_source": "labels_14_official_study.pickle", "zero_division": 0,
            "f1_abnorm_definition": "mean_sample_active_abnormal_label_f1_excluding_no_finding_and_devices"}


# ---------- 证据失配（幻觉率）P_misalign ----------
def misalign_rate(findings_abnormal: list[str], evidence_cache: list[dict]) -> float:
    """报告异常发现中，无工具证据支撑的比例（§1.3.3 证据失配率）。

    findings_abnormal : 报告中被识别为异常的发现列表（异常+疾病关键词）
    evidence_cache     : 轨迹 evidence_cache（已有 finding 字段）
    """
    if not findings_abnormal:
        return 0.0
    supported = {ev.get("finding", "").lower() for ev in evidence_cache if ev.get("confirmed")}
    unsupported = [f for f in findings_abnormal if f.lower() not in supported]
    return len(unsupported) / len(findings_abnormal)


# ---------- 工具行为奖励 R_tool ----------
@dataclass
class ToolReward:
    """R_tool：成功|多样性|一致性|工具judge 子奖励加权（§1.6 w 切换）。

    judge_fn（可选注入）：由训练器注入的 R_toolJudge 评判可调用，
    输入 trajectory -> 返回 [0,1] 归一化得分，衡量 Agent 对 7 项诊断
    清单的遵从程度（§1.6）。探索阶段 w_judge=0 时天然不生效
    （得分 x 0）；收敛阶段 w_judge=0.2 才真正调用。用法见 score()。
    """
    w_succ: float = 0.1
    w_div: float = 0.5
    w_coh: float = 0.5
    w_judge: float = 0.0
    judge_fn: Optional[Callable] = field(default=None, repr=False)  # 由 trainer 注入

    def call_success_rate(self, tool_calls: list[dict]) -> float:
        if not tool_calls:
            return 0.0
        ok = sum(1 for t in tool_calls if t.get("status") == "ok")
        return ok / len(tool_calls)

    def diversity(self, tool_calls: list[dict], n_avail: int = 7) -> float:
        used = {t.get("name") for t in tool_calls}
        return len(used) / max(n_avail, 1)

    def coherence(self, tool_calls: list[dict]) -> float:
        """Fraction of successful calls referenced by downstream reasoning."""
        if not tool_calls:
            return 0.0
        # Coherence requires an explicit downstream reference, not nonempty output.
        return sum(bool(t.get("used_by")) and t.get("status") == "ok" for t in tool_calls) / len(tool_calls)

    def score(self, trajectory: dict) -> float:
        from .reward_judge import run_tool_judge  # 惰性导入：训练期基座权重 Judge
        referenced=set(trajectory.get("used_tool_indices", []))
        for item in trajectory.get("evidence_cache",[]):referenced.update(item.get("evidence_indices",[]))
        for item in trajectory.get("checklist_checked",[]):
            referenced.update(e.get("tool_call_index") for e in item.get("evidence",[]) if isinstance(e.get("tool_call_index"),int))
        raw_calls = trajectory.get("tool_calls", [])
        pending = list(referenced)
        while pending:
            i = pending.pop()
            if type(i) is not int or not 0 <= i < len(raw_calls):
                raise ValueError("Invalid downstream tool reference")
            for dependency in raw_calls[i].get("dependency_indices", []):
                if type(dependency) is not int or not 0 <= dependency < i:
                    raise ValueError("Tool dependencies must refer to preceding calls")
                if dependency not in referenced:
                    referenced.add(dependency)
                    pending.append(dependency)
        trajectory["used_tool_indices"] = sorted(referenced)
        calls = [dict(tc, used_by=(i in referenced))
                 for i, tc in enumerate(trajectory.get("tool_calls", []))]
        rs = self.call_success_rate(calls)
        rd = self.diversity(calls)
        rc = self.coherence(calls)
        # R_toolJudge：探索期 w_judge=0 -> 该项恒为 0；收敛期 w_judge=0.2 才真正调用
        rj = 0.0
        if self.w_judge > 0:
            rj = float(run_tool_judge(trajectory))
        trajectory["tool_reward_parts"] = dict(r_succ=rs,r_div=rd,r_coh=rc,r_judge=rj,
                                               weights=[self.w_succ,self.w_div,self.w_coh,self.w_judge])
        return (self.w_succ * rs + self.w_div * rd
                + self.w_coh * rc + self.w_judge * rj)


# ---------- 组合奖励 ----------
@dataclass
class RewardComposer:
    """按阶段（探索/收敛）切换权重，构成最终组合奖励。"""
    phase: str = "explore"           # "explore" | "converge"
    lamb_mis: float = 0.2
    f1_abnorm: Optional[Callable] = None   # 生成侧 F1_abnorm（接 CheXbert，留空）
    tool_reward: ToolReward = field(default_factory=ToolReward)
    phase_weights: dict | None = None
    phase_penalties: dict | None = None

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        if phase == "converge":
            self.lamb_mis = 0.3
            self.tool_reward.w_div = 0.2
            self.tool_reward.w_coh = 0.2
            self.tool_reward.w_judge = 0.2
        else:
            self.lamb_mis = 0.2
            self.tool_reward.w_div = 0.5
            self.tool_reward.w_coh = 0.5
            self.tool_reward.w_judge = 0.0

        if self.phase_weights and phase in self.phase_weights:
            weights = self.phase_weights[phase]
            self.tool_reward.w_succ, self.tool_reward.w_div, self.tool_reward.w_coh, self.tool_reward.w_judge = map(float, weights)
        if self.phase_penalties and phase in self.phase_penalties:
            self.lamb_mis = float(self.phase_penalties[phase])

    def compute(self, sample: dict, trajectory: dict) -> float:
        """组合奖励 = F1_abnorm - lamb*P_misalign + R_tool。

        sample  : 含 label14
        trajectory : {tool_calls, evidence_cache, findings, abnormal}
        """
        return self.compute_parts(sample, trajectory)["total"]

    def compute_parts(self, sample: dict, trajectory: dict) -> dict:
        """返回组合奖励各分量（供训练逐步记录 P_misalign / R_quality 用）。

        返回 dict：{f1_abnorm, r_misalign, r_tool, total}
        这是超参实验"两阶段调度曲线 / λ_mis 敏感性"的数据来源。
        """
        assessment = trajectory.get("report_assessment")
        if assessment is None:
            raise RuntimeError("Report-based clinical reward missing; refusing hypothesis-derived proxy")
        if "fallback" in str(assessment.get("assessment_version", "")) or "fallback" in assessment.get("quality_details", {}):
            raise RuntimeError("Failed report assessment cannot be used as a training reward")
        gt = sample.get("label14")
        if gt is None or len(gt) != 14:
            raise ValueError("MIMIC sample missing authoritative 14-dimensional label14 GT")
        # The report judge is an independent evidence/misalignment assessor;
        # clinical F1 must come from the frozen CheXbert classifier on the
        # generated report text, never from judge-vs-reference matching.
        from www_cxr_agent.eval.metrics_clinical import report_classifier
        pred, _ = report_classifier().predict([str(trajectory.get("findings", ""))])
        pred = pred[0]
        if len(pred) != 14:
            raise ValueError("CheXbert generated-report labels are not 14-dimensional")
        if any(x not in (0, 1) for x in gt):
            raise ValueError("label14 GT must be binary")
        gt_mimic = [int(x) for x in gt]
        pred_mimic = chexbert_to_mimic14(pred)
        metrics = label_metrics([gt_mimic], [pred_mimic])
        f1_14 = metrics["F1_14"]
        # The paper's F1_abnorm is the independent report-judge abnormality
        # matching score (exact/partial matches), not a label-derived proxy.
        if "f1_abnorm" not in assessment:
            raise RuntimeError("Report judge missing required f1_abnorm")
        f1_abnorm = float(assessment["f1_abnorm"])
        if not 0.0 <= f1_abnorm <= 1.0:
            raise ValueError("Report judge f1_abnorm must be in [0,1]")
        trajectory.update(label14=gt_mimic, pred_label14=pred_mimic,
                          label_order=LABELS_14, gt_source=str(sample.get("gt_source", "labels_14_official_study.pickle")))
        r_misalign = float(assessment["misalign"])
        r_tool = self.tool_reward.score(trajectory)

        total = f1_14 + f1_abnorm - self.lamb_mis * r_misalign + r_tool
        logger.debug("R=%.4f (F1a=%.4f mis=%.3f*%.2f tool=%.4f)",
                     total, f1_abnorm, r_misalign, self.lamb_mis, r_tool)
        return {
            "f1_abnorm": f1_abnorm,
            "f1_14": f1_14,
            "quality_complete": "f1_14" in assessment,
            "r_misalign": r_misalign,
            "lambda_mis":self.lamb_mis,"misalign_penalty":self.lamb_mis*r_misalign,
            "quality":f1_14+f1_abnorm,"reward_phase":self.phase,
            "r_tool": r_tool,
            "total": total,
            "reward_impl_version": REWARD_IMPL_VERSION,
        }


def make_f1_abnorm_from_labels(sample: dict, trajectory: dict) -> float:
    """F1_abnorm：异常类（排除 No Finding / Support Devices）上生成 vs GT 的 macro F1。
    GT 用 sample.label14；生成侧由 trajectory['pred_label14'] 提供。
    按多标签 macro-F1 计算：每个异常类独立算二分类 F1，再取平均。
    全负类（GT=0 且 Pred=0）跳过不参与平均，避免 zero_division 拉低均值。
    纯 Python 实现，不依赖 sklearn。"""
    gt = sample.get("label14", [])
    pred = trajectory.get("pred_label14", [])
    if not gt or not pred or len(gt) != len(pred):
        return 0.0
    idx = [i for i, name in enumerate(LABELS_14)
           if name not in ("No Finding", "Support Devices")]
    f1s = []
    eps = 1e-9
    for i in idx:
        gt_pos = gt[i] > 0.5
        pred_pos = int(pred[i]) == 1
        tp = 1 if (gt_pos and pred_pos) else 0
        fp = 1 if (not gt_pos and pred_pos) else 0
        fn = 1 if (gt_pos and not pred_pos) else 0
        if tp == 0 and fp == 0 and fn == 0:
            continue  # 该类全负，跳过不参与平均
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1)
    if not f1s:
        return 0.0
    return sum(f1s) / len(f1s)
