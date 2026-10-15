# -*- coding: utf-8 -*-
"""GRPO+LoRA 训练主入口（§1.6）。

用法（服务器 base 环境，先解 Phase1 子集，规避显存卡点）：
    python scripts/train_grpo.py --config configs/train_default.yaml [--trial]
    python scripts/train_grpo.py --config configs/train_default.yaml \
        --trial --phase screen --n 10
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import yaml

# ---- 压掉三方库噪音日志（TF/httpx/MCP/numexpr/4bit保存警告），只留关键 INFO ----
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # 抑制 TF oneDNN 启动打印
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from www_cxr_agent.reward import RewardComposer
from www_cxr_agent.train import GRPOTrainer


def find_project_root() -> str:
    """定位服务器项目根（含 www_cxr_agent/ 与 configs/ 的目录）。"""
    cur = os.path.dirname(os.path.abspath(__file__))  # .../scripts
    root = os.path.dirname(cur)                        # 项目根
    if os.path.isdir(os.path.join(root, "www_cxr_agent")):
        return root
    return os.getcwd()


MIMIC_ROOT_DEFAULT = "/root/autodl-tmp/datasets/mimic_cxr/mimic_cxr"  # 服务器真实数据根


def load_dataset(dataset: str, split: str, n: int | None = None) -> list[dict]:
    """直接调 MimicCXRDataset 传绝对路径，绕过 build_dataset 的 OmegaConf env 解析。

    dataset: "mimic_cxr" | "iu_xray"
    """
    root = find_project_root()
    if root != os.getcwd():
        sys.path.insert(0, root)
    from www_cxr_agent.data.mimic_cxr import MimicCXRDataset

    roots = {
        "mimic_cxr": MIMIC_ROOT_DEFAULT,
        "iu_xray": "/root/autodl-tmp/datasets/iu_xray",
    }
    # 允许用环境变量覆盖（如路径不同时）
    import os as _os
    env_root = _os.environ.get(f"WWW_CXR_{dataset}_ROOT")
    data_root = env_root or roots.get(dataset, MIMIC_ROOT_DEFAULT)

    if dataset == "mimic_cxr":
        ds = MimicCXRDataset(data_root, split=split, limit=n)
    elif dataset == "iu_xray":
        from www_cxr_agent.data.iu_xray import IUXRayDataset
        ds = IUXRayDataset(data_root, split=split, limit=n,
                           split_cache=os.path.join(root, "runs/iu_patient_split.json"))
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    out = []
    for r in ds.samples:
        label = r.get("labels_14")
        out.append({
            "id": r["id"], "dataset": dataset, "split": r["split"],
            "study_id": r.get("study_id"),
            "subject_id": r.get("subject_id") or r.get("patient_id"),
            "report": r["findings_raw"],
            "label14": list(label) if label is not None else None,
            "label_order": r.get("label_order"), "gt_source": r.get("gt_source"),
            "abs_images": list(r["image_paths"]),
            "view_positions": r.get("view_positions", []),
        })
    # Optional authoritative view map: image basename -> PA/AP/LATERAL/LL.
    import json
    from pathlib import Path
    view_map_path = os.environ.get("CXR_VIEW_METADATA")
    view_map = json.loads(Path(view_map_path).read_text()) if view_map_path else {}
    if dataset == "iu_xray" and not view_map_path:
        import csv
        projection_file=Path(root)/"cache/iu_metadata/indiana_projections.csv"
        if projection_file.is_file():
            with projection_file.open() as f:
                view_map={"CXR"+row["filename"].replace(".dcm.png", ".png"):row["projection"].upper() for row in csv.DictReader(f)}
            view_map_path=str(projection_file)
    for sample in out:
        views = {str(path):str(view_map.get(Path(path).name, view_map.get(Path(path).stem,"UNKNOWN"))).upper()
                 for path in sample["abs_images"]}
        sample["view_positions"] = views
        pa = next((p for p,v in views.items() if v in ("PA","FRONTAL")),None)
        lat = next((p for p,v in views.items() if v in ("LAT","LATERAL","LL")),None)
        if pa:
            sample["abs_images"] = [pa]+[p for p in sample["abs_images"] if p!=pa]
        sample["image_lat"] = lat if pa and lat else None
        sample["view_protocol"] = "PA_LAT" if pa and lat and views[pa]=="PA" else "FRONTAL_LAT" if pa and lat else "single_or_unknown"
        sample["view_metadata_source"] = view_map_path
    return out


def load_cfg(path: str) -> dict:
    from pathlib import Path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    def expand_env(value):
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, dict):
            return {key: expand_env(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand_env(item) for item in value]
        return value
    cfg = expand_env(cfg)
    parent = cfg.pop("extends", None)
    if not parent:
        return cfg
    base = load_cfg(str(Path(path).parent / parent))
    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = value
    merge(base, cfg)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_default.yaml")
    ap.add_argument("--trial", action="store_true", help="试跑规模（§1.6）")
    ap.add_argument("--phase", default="full",
                    help="rollout 阶段: screen(仅Phase1) | full(需更大显存)。"
                         "课程调度用 full@<k>（如 full@30 = 前30步explore权重、之后自动切converge）")
    ap.add_argument("--n", type=int, default=None, help="样本数（覆盖 yaml）")
    ap.add_argument("--trace", action="store_true",
                    help="打印每次策略决策的原始 JSON（调试决策链用）")
    ap.add_argument("--rollout", type=int, default=None,
                    help="覆盖 training.rollout（如中间规模 4），不带 --trial 才生效")
    ap.add_argument("--batch", type=int, default=None,
                    help="覆盖 training.batch_size（如中间规模 2）")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="覆盖 opt.max_steps（如中间规模 40）")
    ap.add_argument("--log-interval", type=int, default=None,
                    help="覆盖 log_interval（如 --log-interval 1 每步都打 loss/avg_R）")
    ap.add_argument("--resume-state", default=None, help="trainer_state_stepN.pt")
    ap.add_argument("--save-interval", type=int, default=None,
                    help="覆盖 save_interval（如 --save-interval 1 每步都存 checkpoint，防中断白跑）")
    ap.add_argument("--rollout-concurrency", type=int, default=None,
                    help="覆盖 rollout 并发；仅容量测试使用")
    ap.add_argument("--judge-concurrency", type=int, default=None,
                    help="覆盖 report judge 并发；仅容量测试使用")
    ap.add_argument("--policy-max-new-tokens", type=int, default=None,
                    help="覆盖策略生成长度；改变生成预算，正式主实验保持默认")
    ap.add_argument("--rollout-devices", default=None,
                    help="逗号分隔的策略 rollout CUDA 设备，如 0,1,2")
    ap.add_argument("--rollout-workers-per-device", type=int, default=None,
                    help="每个策略副本的 Agent 槽位数；同卡 generate 仍串行")
    ap.add_argument("--backward-chunk", type=int, default=None,
                    help="反向传播微批大小（基础设施参数，不改变 GRPO 目标）")
    args = ap.parse_args()

    if args.backward_chunk is not None:
        if args.backward_chunk < 1:
            raise ValueError("--backward-chunk must be positive")
        os.environ["CXR_BACKWARD_CHUNK"] = str(args.backward_chunk)

    logging.basicConfig(level=logging.INFO)

    cfg = load_cfg(args.config)
    pol = cfg["policy"]
    opt = cfg["opt"]
    tr = cfg["training"]
    if args.policy_max_new_tokens is not None:
        os.environ["CXR_POLICY_MAX_NEW_TOKENS"] = str(args.policy_max_new_tokens)
    elif tr.get("policy_max_new_tokens"):
        os.environ.setdefault("CXR_POLICY_MAX_NEW_TOKENS", str(tr["policy_max_new_tokens"]))
    # The launcher exports this for its child processes, but setting it from
    # the run manifest also makes standalone/resume commands deterministic.
    if not os.environ.get("CXR_JUDGE_URLS") and tr.get("judge_count"):
        count = max(1, int(tr["judge_count"]))
        os.environ["CXR_JUDGE_COUNT"] = str(count)
        os.environ["CXR_JUDGE_URLS"] = ",".join(
            f"http://127.0.0.1:{7007 + i}/mcp" for i in range(count))
    rw = cfg["reward"]

    # 阶段：screen(仅Phase1/R_tool+F1) | full(Phase1+7004 VQA验证，证据奖励生效)
    reward = RewardComposer(phase="explore", lamb_mis=rw["lamb_mis"])
    reward.phase_weights = rw.get("w")
    reward.phase_penalties = rw.get("phase_penalties", {"explore": rw["lamb_mis"], "converge": 0.0 if rw["lamb_mis"] == 0 else 0.3})
    # Report-based quality is computed by the independent evaluator, not hypotheses.
    # 从 config 读取 R_tool 子奖励权重（w_s, w_d, w_c, w_j），使 yaml 可配（消融用）。
    # 保留向后兼容：yaml 没给 w 或给 None 时用 ToolReward 默认。
    _w = (rw.get("w") or {}).get("explore")
    if _w and len(_w) == 4:
        reward.tool_reward.w_succ = float(_w[0])
        reward.tool_reward.w_div = float(_w[1])
        reward.tool_reward.w_coh = float(_w[2])
        reward.tool_reward.w_judge = float(_w[3])
    if args.phase == "converge":
        reward.set_phase("converge")

    # 解析课程调度：--phase full@<k> → explore 前 k 步, 之后自动切 converge
    _phase_rl, _phase_k = args.phase, tr.get("curriculum_step",30)
    if "@" in args.phase:
        _phase_rl, _part = args.phase.split("@", 1)
        if _phase_rl not in ("screen", "full"):
            raise SystemExit(f"--phase 课程调度需前缀 screen|full，收到: {args.phase}")
        if _part.isdigit():
            _phase_k = int(_part)
        else:
            raise SystemExit(f"--phase 切点需数字：full@<k>，收到: {args.phase}")
        logging.info("[curriculum] 前 %d 步 explore，之后自动切 converge (rollout=%s)",
                     _phase_k, _phase_rl)

    if opt["max_steps"] <= 0:
        raise ValueError("No-GRPO is an inference baseline; use scripts/infer.py without --ckpt")
    from www_cxr_agent.eval.metrics_clinical import report_classifier
    report_classifier()  # fail before loading the policy if clinical weights are missing
    trainer = GRPOTrainer(
        model_path=pol["model"],
        lora_rank=pol["lora_rank"],
        lora_alpha=pol["lora_alpha"],
        load_in_4bit=pol["load_in_4bit"],
        clip_epsilon=tr.get("clip_epsilon", 0.2),
        kl_beta=tr.get("kl_beta", 0.04),
        rollout=args.rollout if args.rollout is not None else tr["rollout"],
        batch_size=args.batch if args.batch is not None else tr["batch_size"],
        lr=opt["lr"],
        max_steps=args.max_steps if args.max_steps is not None else opt["max_steps"],
        warmup_steps=opt["warmup_steps"],
        reward=reward,
        phase=_phase_rl,
        trace=args.trace,
        seed=cfg["data"].get("seed",42),
        agent_options=cfg.get("agent_options", {}),
        agent_backend=cfg.get("agent_backend", "layered"),
        max_agent_steps=tr.get("max_agent_steps",16),
        max_tool_calls=tr.get("max_tool_calls", tr.get("max_agent_steps",16)),
        rollout_concurrency=args.rollout_concurrency if args.rollout_concurrency is not None else tr.get("rollout_concurrency", 2),
        judge_concurrency=args.judge_concurrency if args.judge_concurrency is not None else tr.get("judge_concurrency", 4),
        rollout_devices=[int(value) for value in (
            args.rollout_devices.split(",") if args.rollout_devices is not None
            else tr.get("rollout_devices", [0])
        )],
        rollout_workers_per_device=(
            args.rollout_workers_per_device if args.rollout_workers_per_device is not None
            else tr.get("rollout_workers_per_device")
        ),
        curriculum_k=_phase_k,          # None=不做课程调度（沿用原行为）
    )
    trainer.load_model()
    if args.resume_state:
        trainer.load_training_state(args.resume_state)

    # 数据：走服务器 factory.build_dataset
    dataset = load_dataset(
        dataset=cfg["data"]["dataset"],          # 应设 mimic_cxr
        split=cfg["data"]["split"],
        n=args.n or (cfg["data"]["n_samples_trial"] if args.trial else None),
    )
    if cfg.get("paired_only"):
        dataset=[s for s in dataset if s.get("image_lat")]
    validation = load_dataset(cfg["data"]["dataset"],"val",n=cfg.get("validation",{}).get("n",2)) if cfg.get("validation",{}).get("enabled") else None
    if cfg.get("paired_only") and validation:
        validation=[s for s in validation if s.get("image_lat")]
    if cfg.get("single_view"):
        dataset=[dict(s,image_lat=None) for s in dataset]
        if validation:validation=[dict(s,image_lat=None) for s in validation]
    if not dataset:raise ValueError("No eligible training samples")
    # Essential experiment provenance, kept with checkpoints and trajectories.
    import hashlib, json
    from pathlib import Path
    run_root=Path(cfg["paths"]["runs_dir"]);run_root.mkdir(parents=True,exist_ok=True)
    manifest={"config":cfg,"arguments":vars(args),
              "train_ids":[x["id"] for x in dataset],
              "validation_ids":[x["id"] for x in validation or []],
              "view_protocols":sorted({x.get("view_protocol","unknown") for x in dataset}),
              "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for folder in ("scripts","www_cxr_agent","configs")
                               for p in Path(folder).rglob("*") if p.is_file() and p.suffix in (".py",".yaml",".json")}}
    import zipfile
    archive_path = run_root / ("source_resume.zip" if args.resume_state else "source.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in manifest["source_sha256"]:
            archive.write(source, source)
    manifest["source_archive"] = archive_path.name
    manifest_path=run_root/("manifest_resume.json" if args.resume_state else "manifest.json")
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    rolls = trainer.fit(
        dataset,
        trial=args.trial,
        log_interval=args.log_interval or cfg["log"]["log_interval"],
        save_interval=args.save_interval or cfg["log"]["save_interval"],
        ckpt_dir=cfg["paths"]["checkpoint_dir"],
        early_stop=opt["early_stop"],
        traj_out=os.path.join(cfg["paths"]["runs_dir"],"trajectories.jsonl"),
        validation_dataset=validation,
        validation_interval=cfg.get("validation",{}).get("interval",1),
        patience=cfg.get("validation",{}).get("patience",3),
    )
    n = len(rolls)
    avg_r = (sum(r.reward for r in rolls) / n) if n else 0.0
    logging.info("training done: %d rolls, avg_R=%.4f", n, avg_r)


if __name__ == "__main__":
    main()
