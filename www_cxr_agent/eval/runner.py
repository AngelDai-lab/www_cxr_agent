"""One evaluator for every baseline, ablation and validation split."""
import asyncio
import json
from pathlib import Path
from .metrics_clinical import compute_clinical
from .metrics_nlg import compute_nlg
from .metrics_alignment import compute_alignment
from .metrics_efficiency import compute_efficiency

async def evaluate_records_async(records, include_radgraph=True):
    from www_cxr_agent.reward.reward_judge import evaluate_report_resilient
    if not records or len({r["id"] for r in records})!=len(records):
        raise ValueError("Nonempty records with unique IDs required")
    has_gt = [r.get("label14") is not None for r in records]
    if (any(has_gt) and not all(has_gt)) or any(r.get("dataset") == "mimic_cxr" and r.get("label14") is None for r in records):
        raise ValueError("MIMIC evaluation requires label14 for every record; no reference-label fallback")
    gt_metrics = None
    if all(has_gt):
        from www_cxr_agent.reward.rewards import label_metrics, chexbert_to_mimic14, LABELS_14
        from .metrics_clinical import report_classifier
        if any(r.get("label_order") != LABELS_14 for r in records):
            raise ValueError("Explicit canonical label_order required for GT evaluation")
        predictions14, _ = await asyncio.to_thread(report_classifier().predict, [r["findings"] for r in records])
        pred_gt_order = [chexbert_to_mimic14(p) for p in predictions14]
        gt_metrics = label_metrics([r["label14"] for r in records], pred_gt_order)
        for r, pred in zip(records, pred_gt_order):
            r["pred_label14"] = pred
    for r in records:
        if not r.get("findings","").strip() or not r.get("report_ref","").strip():
            raise ValueError(f"Missing prediction/reference for {r['id']}")
        trajectory=r.setdefault("trajectory",{})
        trajectory["report_assessment"]=await evaluate_report_resilient(r["findings"],r["report_ref"],trajectory)
    invalid = [r["id"] for r in records if r["findings"].replace("<|endoftext|>","").strip().lower().rstrip('.') in ("matched","yes","no","unmatched")]
    predictions=[r["findings"].replace("<|endoftext|>","").strip() for r in records];references=[r["report_ref"] for r in records]
    clinical=await asyncio.to_thread(compute_clinical,predictions,references,include_radgraph)
    alignment=compute_alignment(records)
    efficiency=compute_efficiency(records)
    # Training trajectories already persist ``tool_reward_parts``.  Inference
    # and validation trajectories do not run RewardComposer, so compute the
    # three trace-local components here instead of silently reporting zeros.
    # The independent judge component is intentionally excluded from these
    # aliases; it is only available when the training reward phase requests it.
    from www_cxr_agent.reward.rewards import ToolReward
    fallback_tool_reward = ToolReward(w_succ=0.1, w_div=0.5, w_coh=0.5, w_judge=0.0)
    tool_parts=[]
    for r in records:
        trajectory = r.get("trajectory", {})
        parts = trajectory.get("tool_reward_parts")
        if not isinstance(parts, dict) or not all(k in parts for k in ("r_succ", "r_div", "r_coh")):
            fallback_tool_reward.score(trajectory)
            parts = trajectory.get("tool_reward_parts", {})
        tool_parts.append(parts)
    tool_metrics={k: sum(float(p.get(k, 0.0)) for p in tool_parts)/len(tool_parts)
                  for k in ("r_succ", "r_div", "r_coh")}
    # Stable paper-table aliases are kept at the top level while the detailed
    # nested metric objects remain available for auditability.
    return {"n":len(records),"invalid_report_ids":invalid,"invalid_report_rate":len(invalid)/len(records),"clinical":clinical,
            "nlg":await asyncio.to_thread(compute_nlg,predictions,references),
            "alignment":alignment,"efficiency":efficiency,
            "gt_clinical": gt_metrics,
            "F1_14": gt_metrics["F1_14"] if gt_metrics else None,
            "F1_14_sample_mean": gt_metrics["F1_14_sample_mean"] if gt_metrics else None,
            "F1_abnorm_label_diagnostic": gt_metrics["F1_abnorm"] if gt_metrics else None,
            "F1_abnorm": (sum(float(r.get("trajectory", {}).get("report_assessment", {}).get("f1_abnorm", 0.0)) for r in records) / len(records)) if records else None,
            "gt_source": "labels_14_official_study.pickle" if gt_metrics else "reference_report_chexbert",
            "reference_report_ce_f1": clinical["ce_f1_macro"],
            "reference_report_FP_rate": clinical["fp_rate_macro14"],
            "reference_report_Miss_rate": clinical["miss_rate_macro14"],
            "report_ce_f1": gt_metrics["F1_14"] if gt_metrics else clinical["ce_f1_macro"],
            "P_misalign": alignment["misalign_rate"],
            "FP_rate": gt_metrics["FP_rate"] if gt_metrics else clinical["fp_rate_macro14"],
            "Miss_rate": gt_metrics["Miss_rate"] if gt_metrics else clinical["miss_rate_macro14"],
            "R_succ": tool_metrics["r_succ"], "R_div": tool_metrics["r_div"],
            "R_coh": tool_metrics["r_coh"],
            "checklist_coverage": alignment["checklist_coverage"],
            "evidence_coverage": alignment["evidence_coverage"],
            "tool_redundancy_rate": efficiency["exact_duplicate_call_rate"],
            "verification_statuses": [r.get("trajectory",{}).get("verification_statuses", []) for r in records],
            "complete":include_radgraph, "valid_reports":not invalid}

def run_eval(pred_path, ref_path=None, config=None):
    records=[json.loads(line) for line in Path(pred_path).read_text().splitlines() if line.strip()]
    if ref_path:
        refs={r["id"]:r for r in (json.loads(line) for line in Path(ref_path).read_text().splitlines() if line.strip())}
        if set(refs)!={r["id"] for r in records}:
            raise ValueError("Prediction/reference ID sets differ")
        for r in records:r["report_ref"]=refs[r["id"]]["report_ref"]
    result=asyncio.run(evaluate_records_async(records,include_radgraph=(config or {}).get("include_radgraph",True)))
    return result
