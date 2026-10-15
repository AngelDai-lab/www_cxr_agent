# -*- coding: utf-8 -*-
"""Shared-agent, completion-masked GRPO with grouped advantages and frozen-base KL."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

@dataclass
class Rollout:
    sample_id: str
    text: str
    messages: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)
    evidence_cache: list = field(default_factory=list)
    abnormal: list = field(default_factory=list)
    pred_label14: list = field(default_factory=list)
    reward: float = 0.0
    phase: str = "screen"
    f1_abnorm: float = 0.0
    r_misalign: float = 0.0
    r_tool: float = 0.0
    decisions: list = field(default_factory=list)
    trajectory: dict = field(default_factory=dict)


# ---------- token-masked clipped GRPO objective ----------
def completion_logps(model, decision, device):
    ids = decision["prompt_ids"] + decision["completion_ids"]
    x = torch.tensor([ids], device=device)
    start = len(decision["prompt_ids"]) - 1
    logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=len(decision["completion_ids"])+1).logits[0, :-1].float()
    targets = x[0, start+1:]
    return -torch.nn.functional.cross_entropy(logits, targets, reduction="none")


def clipped_objective(logps, old_logps, advantages, reference_logps=None, epsilon=0.2, beta=0.04):
    ratio = (logps - old_logps).exp()
    objective = torch.minimum(ratio * advantages, ratio.clamp(1-epsilon, 1+epsilon) * advantages)
    if reference_logps is not None and beta:
        delta = reference_logps - logps
        objective = objective - beta * (delta.exp() - delta - 1)
    return -objective.mean()


def policy_loss(model, tokenizer, rolls, advs, device="cuda", epsilon=0.2, beta=0.04):
    total = 0.0
    model._backward_seconds = 0.0
    # No dropout: likelihood evaluation matches the generating policy.
    model.train()  # activation checkpointing on; dropout is disabled at load time
    completion_jobs = []
    for roll, advantage in zip(rolls, advs):
        count = sum(len(d["completion_ids"]) for d in roll.decisions)
        if not count:
            raise ValueError("Rollout lacks assistant completion tokens")
        for decision in roll.decisions:
            if not decision["completion_ids"]:
                continue
            completion_jobs.append((decision, advantage, count))
    if not completion_jobs:
        raise ValueError("No differentiable completion losses")
    # Stream forward/backward in true micro-batches.  Building every forward
    # graph before the first backward retained all 48-rollout activations and
    # could consume the whole GPU even when CXR_BACKWARD_CHUNK=1.  Each chunk
    # below is now constructed, reduced and backpropagated before the next
    # chunk starts.  Gradients still accumulate the identical summed GRPO
    # objective in the same optimizer step.
    chunk_size = max(1, int(os.environ.get("CXR_BACKWARD_CHUNK", "1")))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    backward_started = time.monotonic()
    for start in range(0, len(completion_jobs), chunk_size):
        chunk_losses = []
        for decision, advantage, count in completion_jobs[start:start + chunk_size]:
            logps = completion_logps(model, decision, device)
            old = decision["old_logps"].to(device)
            ref = decision["reference_logps"].to(device)
            loss = clipped_objective(logps, old, advantage, ref, epsilon, beta)
            loss = loss * (len(logps) / count) / len(rolls)
            chunk_losses.append(loss)
            total += float(loss.detach())
        torch.stack(chunk_losses).sum().backward()
        del chunk_losses, logps, old, ref, loss
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model._backward_seconds = time.monotonic() - backward_started
    return total


# =========================================================================
#                           GRPOTrainer
# =========================================================================
class GRPOTrainer:
    def __init__(
        self,
        model_path: str,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        load_in_4bit: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        rollout: int = 8,
        batch_size: int = 6,
        lr: float = 1e-5,
        max_steps: int = 150,
        warmup_steps: int = 10,
        max_agent_steps: int = 16,
        max_tool_calls: int | None = None,
        max_agent_steps_converge: int = 10,
        reward=None,
        device: str = "cuda",
        seed: int = 42,
        phase: str = "full",
        trace: bool = False,
        curriculum_k: int | None = None,
        clip_epsilon: float = 0.2,
        kl_beta: float = 0.04,
        agent_options: dict | None = None,
        agent_backend: str = "layered",
        gen_batch_size: int = 8,  # batch generate 每批大小，控制 KV cache 显存
        judge_concurrency: int = 4,
        rollout_concurrency: int = 2,
        rollout_devices: list[int] | None = None,
        rollout_workers_per_device: int | None = None,
    ):
        if phase != "full":
            raise ValueError("Report-aligned GRPO requires full phase; screening-only is not the paper objective")
        self.agent_options = agent_options or {}
        self.agent_backend = agent_backend
        self.clip_epsilon = clip_epsilon
        self.kl_beta = kl_beta
        self.phase = phase
        self.trace = trace

        self.curriculum_k = curriculum_k
        self.gen_batch_size = gen_batch_size
        self.rollout_concurrency = max(1, int(rollout_concurrency))
        self.rollout_devices = [int(device) for device in (rollout_devices or [0])]
        if not self.rollout_devices:
            raise ValueError("rollout_devices must contain at least one CUDA device")
        if len(set(self.rollout_devices)) != len(self.rollout_devices):
            raise ValueError("rollout_devices must not contain duplicates")
        self.rollout_workers_per_device = max(
            1,
            int(rollout_workers_per_device)
            if rollout_workers_per_device is not None
            else self.rollout_concurrency,
        )
        self.best_score, self.stale = float("-inf"), 0
        self.validation_history = []
        self.judge_concurrency = max(1, int(judge_concurrency))
        self.model_path = model_path
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.rollout = rollout
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_agent_steps = max_agent_steps
        self.max_tool_calls = int(max_tool_calls if max_tool_calls is not None else max_agent_steps)
        self.max_agent_steps_converge = max_agent_steps_converge
        self.device = device
        self.seed = seed
        from ..reward import RewardComposer
        from ..reward.rewards import REWARD_IMPL_VERSION
        self.reward = reward or RewardComposer(phase="explore")
        self.reward_impl_version = REWARD_IMPL_VERSION
        # Never replace an explicitly configured report-quality scorer.
        torch.manual_seed(seed)

        self.optimizer_updates = 0
        self.skipped_zero_variance = 0
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self._tool_mgr = None
        self.completed_steps = 0
        self._rollout_replicas: dict[int, torch.nn.Module] = {}
        self._rollout_snapshot_checksum: str | None = None

    @property
    def tool_mgr(self):
        if self._tool_mgr is None:
            from www_cxr_agent.agents.server_manager import ToolServerManager
            self._tool_mgr = ToolServerManager()
        return self._tool_mgr

    # ---------- 模型加载 ----------
    def load_model(self):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = dict(
            torch_dtype=self.torch_dtype,
            device_map={"": 0},
            trust_remote_code=True,
        )
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs.update(dict(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ))
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # pad token 必须有（batch generate 需要 attention mask）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        lora = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        if self.load_in_4bit:
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora)
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.model.enable_input_require_grads()
        try:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
            logger.info("gradient checkpointing enabled")
        except Exception as e:  # noqa: BLE001
            logger.warning("gradient checkpointing failed: %s", e)
        self.model.print_trainable_parameters()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: min(1.0, (s + 1) / max(self.warmup_steps, 1)),
        )
        logger.info("R_toolJudge: independent MCP service, no policy weight sharing")

    def _lora_config(self):
        from peft import LoraConfig
        return LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

    @staticmethod
    def _adapter_checksum(state: dict[str, torch.Tensor]) -> str:
        digest = hashlib.sha256()
        for name in sorted(state):
            tensor = state[name].detach().contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _main_adapter_state_cpu(self) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict
        return {name: value.detach().cpu().clone()
                for name, value in get_peft_model_state_dict(self.model).items()}

    def _load_rollout_replica(self, device_index: int):
        from peft import get_peft_model, set_peft_model_state_dict
        from transformers import AutoModelForCausalLM

        logger.info("loading frozen rollout replica on cuda:%d", device_index)
        base = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map={"": device_index},
            trust_remote_code=True,
        )
        replica = get_peft_model(base, self._lora_config())
        for module in replica.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        replica.config.use_cache = True
        state = self._main_adapter_state_cpu()
        result = set_peft_model_state_dict(replica, state)
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(
                f"Unexpected LoRA keys for rollout replica cuda:{device_index}: "
                f"{result.unexpected_keys[:3]}"
            )
        replica.requires_grad_(False)
        replica.eval()
        replica._cxr_generate_lock = threading.Lock()
        replica._cxr_rollout_device = device_index
        self._rollout_replicas[device_index] = replica
        logger.info("frozen rollout replica ready on cuda:%d", device_index)
        return replica

    def _sync_rollout_replicas(self) -> str:
        """Freeze one policy snapshot across all rollout devices for this step."""
        from peft import set_peft_model_state_dict

        state = self._main_adapter_state_cpu()
        expected = self._adapter_checksum(state)
        self.model.eval()
        # Gradient checkpointing requires use_cache=False for training, but
        # autoregressive rollout should use the KV cache.  Restore False
        # before likelihood/backward evaluation below.
        self.model.config.use_cache = True
        self.model._cxr_generate_lock = getattr(self.model, "_cxr_generate_lock", threading.Lock())
        self.model._cxr_rollout_device = 0
        for device_index in self.rollout_devices:
            if device_index == 0:
                continue
            replica = self._rollout_replicas.get(device_index)
            if replica is None:
                replica = self._load_rollout_replica(device_index)
            result = set_peft_model_state_dict(replica, state)
            if getattr(result, "unexpected_keys", None):
                raise RuntimeError(
                    f"Unexpected LoRA keys while syncing cuda:{device_index}: "
                    f"{result.unexpected_keys[:3]}"
                )
            replica.config.use_cache = True
            replica.eval()
            actual = self._adapter_checksum(self._replica_adapter_state_cpu(replica))
            if actual != expected:
                raise RuntimeError(
                    f"Rollout replica checksum mismatch cuda:{device_index}: "
                    f"expected={expected[:12]} actual={actual[:12]}"
                )
        self._rollout_snapshot_checksum = expected
        logger.info(
            "rollout policy snapshot checksum=%s devices=%s workers_per_device=%d",
            expected[:16], self.rollout_devices, self.rollout_workers_per_device,
        )
        return expected

    @staticmethod
    def _replica_adapter_state_cpu(model) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict
        return {name: value.detach().cpu()
                for name, value in get_peft_model_state_dict(model).items()}

    def _rollout_model(self, device_index: int):
        return self.model if device_index == 0 else self._rollout_replicas[device_index]

    async def _rollout_async(self, prompts):
        """Collect bounded concurrent rollouts, then judge reports concurrently.

        Agent generation/log-prob evaluation stays serialized on the shared CUDA
        policy. Independent frozen report judges run with a bounded semaphore,
        which removes avoidable judge round-trip idle time without changing the
        policy snapshot used by a GRPO group.
        """
        from ..reward.reward_judge import evaluate_report_async
        snapshot_checksum = self._sync_rollout_replicas()
        cache_root = Path(os.environ.get("CXR_ROLLOUT_CACHE_DIR", "runs/rollout_cache"))
        cache_root.mkdir(parents=True, exist_ok=True)

        def json_safe(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {str(k): json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(v) for v in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        def cache_key(trajectory):
            payload = json.dumps(
                {"findings": trajectory.get("findings", ""),
                 "report_ref": trajectory.get("report_ref", ""),
                 "tool_calls": trajectory.get("tool_calls", []),
                 "evidence_cache": trajectory.get("evidence_cache", []),
                 "checklist_checked": trajectory.get("checklist_checked", [])},
                ensure_ascii=False, sort_keys=True, default=str,
            ).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()

        def persist_raw(trajectory, sample_id, ordinal):
            path = cache_root / f"raw_{sample_id}_{ordinal}.json"
            if not path.exists():
                path.write_text(json.dumps(json_safe(trajectory), ensure_ascii=False), encoding="utf-8")
            return path
        worker_queue: asyncio.Queue = asyncio.Queue()
        # Round-robin device order keeps the first wave spread across all
        # policy replicas. With two slots per device, the queue is
        # cuda0-slot0, cuda1-slot0, cuda2-slot0, cuda0-slot1, ... rather than
        # filling both slots on cuda0 before touching the other cards.
        for slot in range(self.rollout_workers_per_device):
            for device_index in self.rollout_devices:
                worker_queue.put_nowait((f"cuda{device_index}-slot{slot}", device_index))

        async def collect(sample, ordinal):
            worker_id, device_index = await worker_queue.get()
            try:
                # Each rollout owns a thread/event loop because Agent.run is
                # async while transformers generate() is synchronous.  The
                # per-model generation lock serializes CUDA generation on one
                # replica but lets another Agent use MCP while it is waiting.
                return await asyncio.to_thread(
                    lambda: asyncio.run(collect_one(sample, ordinal, worker_id, device_index))
                )
            finally:
                worker_queue.put_nowait((worker_id, device_index))

        async def collect_one(sample, ordinal: int, worker_id: str, device_index: int):
            from ..agents.factory import create_agent
            rollout_model = self._rollout_model(device_index)
            seed_material = (
                f"{self.seed}:{self.completed_steps + 1}:{sample['id']}:{ordinal}"
            ).encode("utf-8")
            sampling_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8], "big"
            ) % (2**63 - 1)
            agent = create_agent(self.agent_backend, image=sample["abs_images"][0], threshold=0.30,
                                 max_steps=self.max_agent_steps, out="", model=rollout_model,
                                 tokenizer=self.tokenizer, training=True,
                                 image_lat=sample.get("image_lat"),
                                 max_tool_calls=self.max_tool_calls,
                                 sampling_seed=sampling_seed, **self.agent_options)
            try:
                record = await agent.run(sample_id=str(sample["id"]), phase=self.phase)
            finally:
                if agent.manager is not None:
                    await agent.manager.close()
            trajectory = record["trajectory"]
            trajectory["findings"] = record["findings"]
            trajectory["efficiency"] = record.get("efficiency",{})
            trajectory["view_protocol"] = sample.get("view_protocol")
            trajectory["report_ref"] = sample.get("report", "")
            trajectory["input_images"] = sample.get("abs_images", [])
            trajectory["image_lat"] = sample.get("image_lat")
            trajectory["subject_id"] = sample.get("subject_id")
            trajectory["policy_worker_id"] = worker_id
            trajectory["policy_device"] = device_index
            trajectory["policy_snapshot_checksum"] = snapshot_checksum
            trajectory["policy_sampling_seed"] = sampling_seed
            trajectory.setdefault("timings", {}).update({
                "policy_seconds": record.get("efficiency", {}).get("policy_seconds", 0.0),
                "evidence_judge_seconds": record.get("efficiency", {}).get("evidence_judge_seconds", 0.0),
                "mcp_seconds": sum(float(t.get("duration", 0.0)) for t in trajectory.get("tool_calls", [])),
            })
            return sample, agent, record, trajectory
        total = len(prompts) * self.rollout
        completed = 0
        async def collect_with_progress(sample, ordinal):
            nonlocal completed
            result = await collect(sample, ordinal)
            completed += 1
            persist_raw(result[3], str(sample["id"]), ordinal)
            logger.info("Rollout progress: %d/%d complete; sample=%s", completed, total, sample["id"])
            return result
        semaphore = asyncio.Semaphore(self.judge_concurrency)
        async def collect_and_judge(sample, ordinal):
            item = await collect_with_progress(sample, ordinal)
            sample, agent, record, trajectory = item
            started = asyncio.get_running_loop().time()
            key = cache_key(trajectory)
            result_path = cache_root / f"judge_{key}.json"
            assessment = None
            if result_path.exists():
                assessment = json.loads(result_path.read_text(encoding="utf-8"))
                logger.info("report judge cache hit key=%s sample=%s", key[:12], sample["id"])
            if assessment is None or assessment.get("assessment_version") != "separated_v4_fixed_support_ids":
                async with semaphore:
                    from ..reward.reward_judge import evaluate_report_resilient
                    assessment = await evaluate_report_resilient(record["findings"], sample.get("report", ""), trajectory)
                result_path.write_text(json.dumps(json_safe(assessment), ensure_ascii=False), encoding="utf-8")
            trajectory["timings"]["report_judge_seconds"] = asyncio.get_running_loop().time() - started
            return item, assessment
        work = [(sample, ordinal)
                for sample in prompts
                for ordinal in range(self.rollout)]
        tasks = [asyncio.create_task(collect_and_judge(sample, ordinal))
                 for sample, ordinal in work]
        try:
            judged = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        rolls = []
        for (sample, agent, record, trajectory), assessment in judged:
            if "f1_14" not in assessment and os.environ.get("CXR_ALLOW_INCOMPLETE_REWARD") != "1":
                raise RuntimeError("Missing CheXbert F1_14: full-design training blocked; CXR_ALLOW_INCOMPLETE_REWARD=1 is diagnostic only.")
            trajectory["report_assessment"] = assessment
            parts = self.reward.compute_parts(sample, trajectory)
            trajectory["reward_parts"] = parts
            lp_started = asyncio.get_running_loop().time()
            # Compute old/reference likelihoods on the frozen replica that
            # generated this trajectory. This keeps the GRPO objective exact
            # while removing the serialized log-prob tail from GPU0.
            lp_model = self._rollout_model(int(trajectory["policy_device"]))
            lp_device = f"cuda:{int(trajectory['policy_device'])}"
            lp_model.config.use_cache = False
            for decision in agent.decisions:
                with torch.no_grad():
                    decision["old_logps"] = completion_logps(lp_model, decision, lp_device).cpu()
                    with lp_model.disable_adapter():
                        decision["reference_logps"] = completion_logps(lp_model, decision, lp_device).cpu()
            trajectory["timings"]["logprob_seconds"] = asyncio.get_running_loop().time() - lp_started
            rolls.append(Rollout(sample_id=str(sample["id"]), text=record["findings"],
                                 messages=trajectory.get("policy_trace",[]), tool_calls=trajectory["tool_calls"], evidence_cache=trajectory["evidence_cache"],
                                 decisions=agent.decisions, trajectory=trajectory, reward=parts["total"], phase=self.phase,
                                 f1_abnorm=parts["f1_abnorm"], r_misalign=parts["r_misalign"], r_tool=parts["r_tool"]))
            logger.info("shared-agent rollout sample=%s reward=%s", sample["id"], parts)
        return rolls

    # ---------- GRPO 训练 ----------
    def compute_advantages(self, rewards, group_ids=None):
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        groups = group_ids if group_ids is not None else [0] * len(rewards)
        result = torch.zeros_like(r)
        for key in dict.fromkeys(groups):
            indices = [i for i, group in enumerate(groups) if group == key]
            values = r[indices]
            result[indices] = (values - values.mean()) / (values.std(unbiased=False) + 1e-6)
        return result

    def train_step(self, rolls: list[Rollout]) -> torch.Tensor:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.train()
        advs = self.compute_advantages([r.reward for r in rolls], [r.sample_id for r in rolls])
        self.optimizer.zero_grad()
        loss = policy_loss(self.model, self.tokenizer, rolls, advs, self.device, self.clip_epsilon, self.kl_beta)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        grads = [p.grad for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError("Missing or non-finite LoRA gradients")
        self.last_gradient_norm = float(torch.stack([g.float().norm() for g in grads]).norm())
        self.optimizer.step()
        self.optimizer_updates += 1
        self.scheduler.step()
        return torch.tensor(loss, device=self.device)

    def fit(self, dataset, max_steps: int | None = None, log_interval: int = 5,
            save_interval: int = 50, ckpt_dir: str = "runs/checkpoints",
            early_stop: bool = False, trial: bool = False,
            traj_out: str | None = "runs/trajectories.jsonl",
            validation_dataset=None, validation_interval: int = 1, patience: int = 3,
            min_delta: float = 0.0):
        import time
        if early_stop and not validation_dataset:
            raise ValueError("Early stopping requires an independent validation dataset")
        if validation_interval < 1 or patience < 1:
            raise ValueError("Validation interval and patience must be positive")
        if validation_dataset:
            train_ids={s["id"] for s in dataset}
            train_subjects={s.get("subject_id") for s in dataset if s.get("subject_id")}
            if train_ids & {s["id"] for s in validation_dataset} or train_subjects & {s.get("subject_id") for s in validation_dataset if s.get("subject_id")}:
                raise ValueError("Train/validation sample or patient leakage")
        import threading, subprocess
        resource_stop = threading.Event()
        gpu_memory_peaks = {}
        def sample_gpu_memory():
            while not resource_stop.is_set():
                try:
                    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True, timeout=5)
                    for line in output.splitlines():
                        index, used = [int(value.strip()) for value in line.split(",")]
                        gpu_memory_peaks[str(index)] = max(gpu_memory_peaks.get(str(index), 0), used)
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass
                resource_stop.wait(1)
        monitor = threading.Thread(target=sample_gpu_memory, daemon=True)
        initial_lora = {name: p.detach().cpu().clone() for name, p in self.model.named_parameters() if p.requires_grad}
        step_diagnostics = []
        best_score, stale = self.best_score, self.stale
        from collections import Counter
        run_steps = max_steps or self.max_steps
        if trial:
            self.rollout = 2
            self.batch_size = 1
            run_steps = min(run_steps, 3)
            logger.info("[trial] rollout=%d batch=%d steps<=%d",
                        self.rollout, self.batch_size, run_steps)

        os.makedirs(ckpt_dir, exist_ok=True)
        # 每步轨迹独立保存（用于前后训练期对比 / case study / 轨迹分析）
        traj_per_step_dir = os.path.join(os.path.dirname(traj_out or "") or "runs", "trajectories_per_step")
        stats_csv_path = os.path.join(os.path.dirname(traj_out or "") or "runs", "step_stats.csv")
        if traj_out:
            os.makedirs(os.path.dirname(traj_out) or ".", exist_ok=True)
            os.makedirs(traj_per_step_dir, exist_ok=True)
            # 初始化统计 CSV
            if not os.path.exists(stats_csv_path):
                with open(stats_csv_path, "w", encoding="utf-8") as _sf:
                    _sf.write("step,avg_reward,avg_n_tools,avg_len_chars,n_rolls,"
                              "rollout_time_s,train_time_s,step_time_s,policy_s,mcp_s,evidence_judge_s,report_judge_s,logprob_s,backward_s,"
                              "top1_tool,top1_count,top2_tool,top2_count\n")
        samples = list(dataset)

        async def _fit_all() -> list[Rollout]:
            nonlocal best_score, stale
            all_rolls: list[Rollout] = []
            t_step_start = time.time()
            async def evaluate_checkpoint(step):
                nonlocal best_score, stale
                metrics = await self.validate(validation_dataset, ckpt_dir, step)
                self.validation_history.append({"step": step, "metrics": metrics})
                score = float(metrics["F1_14"])
                if score > best_score + min_delta:
                    best_score, stale = score, 0
                    self.best_score, self.stale = best_score, stale
                    self.save(os.path.join(ckpt_dir, "best"), step)
                else:
                    stale += 1
                self.best_score, self.stale = best_score, stale
                self.save(ckpt_dir, step)
                logger.info("Validation official-GT macro-F1=%.4f best=%.4f stale=%d", score, best_score, stale)

            # A checkpoint saved immediately before evaluation must resume that
            # evaluation before another update (including an already final step).
            previous_validation = {v["step"] for v in self.validation_history}
            if (validation_dataset and self.completed_steps > 0
                    and (self.completed_steps % validation_interval == 0 or self.completed_steps == run_steps)
                    and self.completed_steps not in previous_validation):
                logger.info("Resuming pending validation at step %d", self.completed_steps)
                await evaluate_checkpoint(self.completed_steps)
            for step in range(self.completed_steps + 1, run_steps + 1):
                if early_stop and stale >= patience:
                    break
                t_step_start = time.time()
                # 课程调度
                if (self.curriculum_k is not None
                        and step == self.curriculum_k + 1
                        and self.reward.phase != "converge"):
                    self.reward.set_phase("converge")
                    # Curriculum changes reward weights only; action budget is fixed.
                    logger.info("[curriculum] step %d: switch to converge weights, "
                                "max_agent_steps=%d", step, self.max_agent_steps)
                batch = self._next_batch(samples, step)
                if not batch:
                    break

                # ---- Rollout ----
                t_rollout_start = time.time()
                rolls = await self._rollout_async(batch)
                # Zero-variance groups carry no policy-gradient signal. Resample once,
                # preserving the reward definition; never add noise to rewards.
                if not self.compute_advantages([r.reward for r in rolls], [r.sample_id for r in rolls]).abs().any():
                    logger.warning("Zero group variance: one bounded fresh rollout attempt")
                    rolls = await self._rollout_async(batch)
                rollout_time = time.time() - t_rollout_start
                # Release stored token likelihoods after each update; retain only
                # light trajectory records across training steps.
                all_rolls.extend(rolls)

                # ---- 轨迹落盘 ----
                if traj_out:
                    # 追加总文件
                    with open(traj_out, "a", encoding="utf-8") as _tf:
                        for _r in rolls:
                            _tf.write(json.dumps({
                                "step": step,
                                "sample_id": _r.sample_id,
                                "phase": _r.phase,
                                "reward": _r.reward,
                                "text": str(_r.text),
                                "messages": _r.messages,
                                "tool_calls": _r.tool_calls,
                                "evidence_cache": _r.evidence_cache,
                                "abnormal": _r.abnormal,
                                "pred_label14": _r.pred_label14,
                                "trajectory": _r.trajectory,
                            }, ensure_ascii=False) + "\n")
                    # 每步独立文件（用于前后对比 & case study）
                    step_traj_path = os.path.join(traj_per_step_dir, f"step{step}.jsonl")
                    with open(step_traj_path, "w", encoding="utf-8") as _sf:
                        for _r in rolls:
                            _sf.write(json.dumps({
                                "step": step,
                                "sample_id": _r.sample_id,
                                "phase": _r.phase,
                                "reward": _r.reward,
                                "text": str(_r.text),
                                "messages": _r.messages,
                                "tool_calls": _r.tool_calls,
                                "evidence_cache": _r.evidence_cache,
                                "abnormal": _r.abnormal,
                                "pred_label14": _r.pred_label14,
                                "trajectory": _r.trajectory,
                            }, ensure_ascii=False) + "\n")
                    logger.info("[step %d/%d] wrote %d trajectories -> %s (step file: %s)",
                                step, run_steps, len(rolls), traj_out, step_traj_path)

                # ---- Train ----
                t_train_start = time.time()
                adv = self.compute_advantages([r.reward for r in rolls], [r.sample_id for r in rolls])
                if not adv.abs().any():
                    self.skipped_zero_variance += 1
                    logger.warning("Skipping optimizer update: all groups still have zero variance")
                    loss = torch.tensor(0.0, device=self.device)
                else:
                    loss = self.train_step(rolls)
                self.last_update_applied = bool(adv.abs().any())
                groups = {}
                for roll in rolls:
                    groups.setdefault(roll.sample_id, []).append(roll.reward)
                step_diagnostics.append({"step": step, "reward_phase": self.reward.phase,
                    "reward_impl_version": self.reward_impl_version,
                    "group_reward_variances": {key: float(torch.tensor(values).var(unbiased=False)) for key, values in groups.items()},
                    "optimizer_update_applied": self.last_update_applied,
                    "optimizer_updates": self.optimizer_updates})
                for roll in rolls:
                    roll.decisions.clear()
                train_time = time.time() - t_train_start
                step_time = time.time() - t_step_start
                t_step_start = time.time()

                avg_r = sum(r.reward for r in rolls) / len(rolls)
                avg_tools = sum(len(r.tool_calls) for r in rolls) / len(rolls)
                avg_len = sum(len(str(r.text)) for r in rolls) / len(rolls)
                stage_keys = ("policy_seconds", "mcp_seconds", "evidence_judge_seconds", "report_judge_seconds", "logprob_seconds")
                stage_times = {k: sum(float(r.trajectory.get("timings", {}).get(k, 0.0)) for r in rolls) / max(len(rolls), 1) for k in stage_keys}

                stage_times["backward_seconds"] = getattr(self.model,"_backward_seconds",0.0) if self.last_update_applied else 0.0
                logger.info("stage timings (mean per rollout; backward per step): %s",stage_times)

                # ---- 工具调用分布统计（用于轨迹分析） ----
                tool_counter: Counter[str] = Counter()
                for _r in rolls:
                    for _tc in _r.tool_calls:
                        tool_counter[_tc.get("name", "unknown")] += 1
                top_tools = tool_counter.most_common(5)

                # ---- 统计日志 ----
                if step % log_interval == 0 or trial:
                    _f1 = sum(r.f1_abnorm for r in rolls) / len(rolls)
                    _mis = sum(r.r_misalign for r in rolls) / len(rolls)
                    _tool = sum(r.r_tool for r in rolls) / len(rolls)
                    logger.info(
                        "[step %d/%d] loss=%.4f avg_R=%.4f F1a=%.4f "
                        "P_mis=%.4f R_tool=%.4f n_tools=%.1f | "
                        "rollout=%.1fs train=%.1fs total=%.1fs | "
                        "top_tools: %s",
                        step, run_steps, loss.item(), avg_r, _f1,
                        _mis, _tool, avg_tools,
                        rollout_time, train_time, step_time,
                        ", ".join(f"{n}={c}" for n, c in top_tools[:3])
                    )
                else:
                    logger.info(
                        "[step %d/%d] loss=%.4f avg_R=%.4f n_tools=%.1f | "
                        "rollout=%.1fs train=%.1fs total=%.1fs",
                        step, run_steps, loss.item(), avg_r, avg_tools,
                        rollout_time, train_time, step_time
                    )

                # ---- CSV 统计落盘（用于后续效率/轨迹分析画图） ----
                if traj_out:
                    with open(stats_csv_path, "a", encoding="utf-8") as _sf:
                        t1 = top_tools[0] if len(top_tools) > 0 else ("", 0)
                        t2 = top_tools[1] if len(top_tools) > 1 else ("", 0)
                        _sf.write(
                            f"{step},{avg_r:.4f},{avg_tools:.1f},{avg_len:.0f},"
                            f"{len(rolls)},{rollout_time:.1f},{train_time:.1f},{step_time:.1f},"
                            f"{stage_times['policy_seconds']:.1f},{stage_times['mcp_seconds']:.1f},{stage_times['evidence_judge_seconds']:.1f},"
                            f"{stage_times['report_judge_seconds']:.1f},{stage_times['logprob_seconds']:.1f},"
                            f"{stage_times['backward_seconds']:.1f},{t1[0]},{t1[1]},{t2[0]},{t2[1]}\n"
                        )

                self.completed_steps = step
                if validation_dataset and (step % validation_interval == 0 or step == run_steps):
                    # Preserve the completed optimizer update before external evaluation.
                    self.save(ckpt_dir, step)
                    await evaluate_checkpoint(step)
                    if early_stop and stale >= patience:
                        logger.info("Validation early stopping at step %d",step)
                        break
                # Always keep a resumable step-1 checkpoint. This protects a
                # long formal run from losing its first update when a later
                # rollout/judge failure occurs, while retaining the configured
                # sparse checkpoint cadence for subsequent steps.
                if step == 1 or step % save_interval == 0 or step == run_steps:
                    self.save(ckpt_dir, step)
            from pathlib import Path
            summary={"optimizer_updates":self.optimizer_updates,"skipped_zero_variance":self.skipped_zero_variance,
                     "validation":self.validation_history,"last_gradient_norm":getattr(self,"last_gradient_norm",None),
                     "steps_completed":self.completed_steps,"curriculum_k":self.curriculum_k,
                     "action_budget":self.max_tool_calls,"max_agent_steps":self.max_agent_steps,
                     "assessment_version":"separated_v3_assignment",
                     "reward_impl_version": self.reward_impl_version,
                     "step_diagnostics": step_diagnostics,
                     "gpu_memory_peak_mib_sampled": dict(gpu_memory_peaks),
                     "gpu_memory_sampling_seconds": 1,
                     "gpu_memory_scope": "whole server, including frozen tools and judge",
                     "lora_delta_l1_this_run": sum(float((p.detach().cpu()-initial_lora[name]).abs().sum()) for name,p in self.model.named_parameters() if name in initial_lora),
                     "early_stopped": bool(early_stop and stale >= patience),
                     "best_score": self.best_score, "stale": self.stale,
                     "stage_timings_seconds": stage_times if 'stage_times' in locals() else {}}
            Path(ckpt_dir,"training_summary.json").write_text(json.dumps(summary,indent=2))
            return all_rolls

        monitor.start()
        try:
            return asyncio.run(_fit_all())
        finally:
            resource_stop.set()
            monitor.join(timeout=6)

    def _next_batch(self, samples, step) -> list[dict]:
        import random
        n = min(self.batch_size, len(samples))
        return random.Random(self.seed + step).sample(samples, n)

    async def validate(self, samples, output_dir, step):
        from ..agents.factory import create_agent
        from ..eval.runner import evaluate_records_async
        import time
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        predictions_path = root / f"validation_step{step}.jsonl"
        # Reuse only a complete batch for this exact checkpoint and sample set.
        if predictions_path.exists():
            saved = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
            expected = {str(sample["id"]) for sample in samples}
            if (len(saved) == len(samples) and {str(r["id"]) for r in saved} == expected
                    and all(r.get("validation_checkpoint_step") == step for r in saved)):
                logger.info("Reusing %d persisted validation predictions at step %d", len(saved), step)
                metrics = await evaluate_records_async(saved)
                target = root / f"validation_step{step}.json"
                temporary = target.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(metrics, indent=2))
                temporary.replace(target)
                return metrics
        records=[]
        self._sync_rollout_replicas()
        queue: asyncio.Queue = asyncio.Queue()
        for i, sample in enumerate(samples):
            queue.put_nowait((i, sample))
        workers = min(len(self.rollout_devices), max(1, len(samples)))
        completed = 0
        completed_lock = asyncio.Lock()

        async def validate_one(sample, device_index):
            def run_sync():
                async def run_async():
                    model = self._rollout_model(device_index)
                    model.eval(); model.config.use_cache = True
                    agent=create_agent(self.agent_backend,image=sample["abs_images"][0], image_lat=sample.get("image_lat"),
                                       threshold=.30,max_steps=self.max_agent_steps,max_tool_calls=self.max_tool_calls,out="",model=model,
                                       tokenizer=self.tokenizer,training=False,**self.agent_options)
                    started=time.monotonic()
                    try:
                        record=await agent.run(sample_id=sample["id"],phase="full")
                    finally:
                        if agent.manager is not None:
                            await agent.manager.close()
                    record.update(report_ref=sample["report"],split=sample.get("split"),subject_id=sample.get("subject_id"),
                                  dataset=sample.get("dataset"), label14=sample.get("label14"),
                                  label_order=sample.get("label_order"), gt_source=sample.get("gt_source"))
                    record.setdefault("efficiency",{})["elapsed_seconds"]=time.monotonic()-started
                    record.setdefault("trajectory",{})["validation_policy_device"] = device_index
                    return record
                return asyncio.run(run_async())
            return await asyncio.to_thread(run_sync)

        async def worker(worker_index):
            nonlocal completed
            while True:
                try:
                    _, sample = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                record = await validate_one(sample, self.rollout_devices[worker_index % len(self.rollout_devices)])
                records.append(record)
                async with completed_lock:
                    completed += 1
                    logger.info("Validation step %d: %d/%d cases complete; sample=%s device=cuda:%d",
                                step, completed, len(samples), sample["id"],
                                record.get("trajectory",{}).get("validation_policy_device", -1))

        await asyncio.gather(*(worker(i) for i in range(workers)))
        records.sort(key=lambda r: str(r["id"]))
        logger.info("Validation step %d: computing clinical, graph and language metrics", step)
        # Commit reports before any judge/metric call can fail.
        for record in records:
            record["validation_checkpoint_step"] = step
        temporary = predictions_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(r)+"\n" for r in records))
        temporary.replace(predictions_path)
        metrics=await evaluate_records_async(records)
        (root/f"validation_step{step}.json").write_text(json.dumps(metrics,indent=2))
        (root/f"validation_step{step}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        return metrics

    def save(self, ckpt_dir: str, step: int):
        os.makedirs(ckpt_dir, exist_ok=True)
        path=os.path.join(ckpt_dir, f"lora_step{step}")
        self.model.save_pretrained(path)
        torch.save({"optimizer":self.optimizer.state_dict(),"scheduler":self.scheduler.state_dict(),
                    "optimizer_updates":self.optimizer_updates,"skipped_zero_variance":self.skipped_zero_variance,
                    "best_score":self.best_score,"stale":self.stale,"validation_history":self.validation_history,
                    "validation_selection_metric":"official_gt_F1_14",
                    "step":step,"seed":self.seed,"curriculum_k":self.curriculum_k,
                    "reward_phase":self.reward.phase,"reward_impl_version":self.reward_impl_version,
                    "torch_rng":torch.get_rng_state(),
                    "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   os.path.join(ckpt_dir,f"trainer_state_step{step}.pt"))
        self.completed_steps=step
        logger.info("saved checkpoint -> %s/lora_step%d", ckpt_dir, step)

    def load_training_state(self, state_path: str):
        state_path = os.path.abspath(state_path)
        state=torch.load(state_path,map_location="cpu",weights_only=False)
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("load_model must run before load_training_state")
        step = int(state.get("step", 0))
        adapter_dir = os.path.join(os.path.dirname(state_path), f"lora_step{step}")
        adapter_file = os.path.join(adapter_dir, "adapter_model.safetensors")
        if os.path.isfile(adapter_file):
            # PEFT owns adapter-name/key normalization; loading the raw
            # safetensors dict through torch would mark valid LoRA keys as
            # unexpected because of the base_model wrapper prefixes.
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            adapter_state = load_file(adapter_file, device="cpu")
            result = set_peft_model_state_dict(self.model, adapter_state)
            if getattr(result, "unexpected_keys", None):
                raise RuntimeError(f"Unexpected LoRA keys while resuming: {result.unexpected_keys[:3]}")
        else:
            raise FileNotFoundError(f"LoRA adapter weights missing in {adapter_dir}")
        self.optimizer.load_state_dict(state["optimizer"]); self.scheduler.load_state_dict(state["scheduler"])
        # Optimizer checkpoints are serialized on CPU. Move tensor moments to
        # the current parameter device before the first resumed AdamW update.
        for param, slot in self.optimizer.state.items():
            for key, value in list(slot.items()):
                if torch.is_tensor(value) and key != "step":
                    slot[key] = value.to(param.device)
        self.best_score = state.get("best_score", float("-inf"))
        self.stale = state.get("stale", 0)
        self.validation_history = state.get("validation_history", [])
        self.validation_selection_metric = "official_gt_F1_14"
        if self.validation_history:
            self.best_score, self.stale = float("-inf"), 0
            for item in self.validation_history:
                value = float(item["metrics"]["F1_14"])
                if value > self.best_score:
                    self.best_score, self.stale = value, 0
                else:
                    self.stale += 1
            logger.info("Recomputed historical selection using official GT F1_14: best=%s stale=%s; historical best adapters remain unchanged", self.best_score, self.stale)
        self.optimizer_updates = state.get("optimizer_updates", 0)
        self.skipped_zero_variance = state.get("skipped_zero_variance", 0)
        self.curriculum_k = state.get("curriculum_k", self.curriculum_k)
        stored_reward_version = state.get("reward_impl_version", "legacy_unknown")
        self.loaded_reward_impl_version = stored_reward_version
        self.reward.set_phase(state.get("reward_phase", "explore"))
        self.completed_steps=step; self.seed=int(state.get("seed",self.seed))
        if state.get("torch_rng") is not None:torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:torch.cuda.set_rng_state_all(state["cuda_rng"])
        logger.info("resumed trainer state at step %d (reward_impl_version=%s -> %s)",
                    self.completed_steps, stored_reward_version, self.reward_impl_version)
