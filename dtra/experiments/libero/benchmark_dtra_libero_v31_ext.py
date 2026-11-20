"""dtra/experiments/libero/benchmark_dtra_libero_v31_ext.py

DTRA v3.1 arXiv Final Manuscript Extensions on LIBERO-100 (K=100, Strict Encoder Parity):

  Mode 1 (`--mode external_cl`):
    Evaluates three established, published continual learning baselines under the exact same
    Frozen ResNet-18 DualCam + SpatialContextAdapter + Cubic B-Spline (H=16, num_cp=8) protocol
    on sequential LIBERO-100 (`libero_10` + `libero_90`, K=100, milestones 10,30,50,70,100):
      - `arm_derpp`: DER++ (Buzzega et al., NeurIPS 2020) — 5% replay buffer + dark teacher
                     control-point logit distillation (alpha=0.5, beta=0.5).
      - `arm_l2p`  : L2P (Wang et al., CVPR 2022) — Replay-Free, Task-ID-free cosine-keyed
                     Prompt Pool (M=32, top-k=4) over the shared 64-d multimodal query space.
      - `arm_olora`: O-LoRA (Wang et al., EMNLP 2023) — Replay-Free, Task-ID-free cumulative
                     orthogonal low-rank adapters (rank=8) over frozen Task-0 base layers.

  Mode 2 (`--mode extended_eval`):
    Zero-retraining high-coverage evaluation (`n_eval=20` initial states/task = 2,000 rollouts/arm)
    directly from saved finalized K=100 `.pt` checkpoints across Seeds 42, 100, and 2026.
    Records exact per-episode binary outcomes `[0/1] * 20` so `init[0:5]` (reproducibility check),
    `init[5:20]` (15 unseen initial states/task), and `init[0:20]` (full 20-state 95% Wilson CI,
    Dup-Lang vs Unique-Lang, L10 vs L90) are all reported rigorously.

Existing frozen benchmark scripts and `_locked.json` artifacts are never modified.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if os.name == "nt":
    try:
        ctypes.CDLL("ucrtbase")._setmaxstdio(8192)
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import dtra.experiments.libero.benchmark_dtra_libero as base_libero
from dtra.core.bspline_core import BSplineDecoder
from dtra.experiments.libero.benchmark_dtra_libero import (
    ExperienceReplayPolicy,
    FrozenResNet18DualCam,
    LiberoTaskDataset,
    LightweightTextEncoder,
    MultimodalMagneticKeyExtractor,
    SingleActiveEnvManager,
    SpatialContextAdapter,
    TaskSpec,
    _calc_dup_lang_route_acc,
    _calc_policy_params_mb,
    _parse_int_csv,
    build_task_specs,
    calc_cl_metrics,
    eval_policy_on_task,
    extract_obs_tensors,
    forward_fill_R_for_milestone,
    load_task_init_states,
    save_json_atomic,
    set_seed,
)


# ==============================================================================
# 1. External Published CL Policy Implementations (Strict ResNet-18 DualCam Parity)
# ==============================================================================


class DERPPLiberoPolicy(ExperienceReplayPolicy):
    """DER++ (Buzzega et al., NeurIPS 2020) over identical ExperienceReplayPolicy."""

    pass


class L2PLiberoPolicy(nn.Module):
    """L2P (Wang et al., CVPR 2022) adapted to 7-DoF LIBERO-100 visuomotor control.

    - Replay-Free, Task-ID-free.
    - Uses identical FrozenResNet18DualCam + MultimodalMagneticKeyExtractor (64-d query).
    - Learnable Prompt Pool (M=32 prompts, top_k=4, prompt_dim=64) with learnable unit keys.
    - Selected prompt embeddings concatenate with SpatialContextAdapter(256) into B-Spline head.
    - Includes L2P pull-key cosine alignment loss (pull_weight=0.10).
    """

    def __init__(
        self,
        backbone: FrozenResNet18DualCam,
        key_extractor: MultimodalMagneticKeyExtractor,
        bspline_decoder: BSplineDecoder,
        ctx_dim: int = 256,
        pool_size: int = 32,
        top_k: int = 4,
        prompt_dim: int = 64,
        num_cp: int = 8,
        action_dim: int = 7,
        pull_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.key_extractor = key_extractor
        self.bspline_decoder = bspline_decoder
        self.ctx_dim = ctx_dim
        self.pool_size = pool_size
        self.top_k = top_k
        self.prompt_dim = prompt_dim
        self.num_cp = num_cp
        self.action_dim = action_dim
        self.pull_weight = pull_weight

        self.ctx_encoder = SpatialContextAdapter(ctx_dim=ctx_dim)
        self.prompt_keys = nn.Parameter(F.normalize(torch.randn(pool_size, key_extractor.d_topo), dim=-1))
        self.prompt_values = nn.Parameter(torch.randn(pool_size, prompt_dim) * 0.02)

        in_dim = ctx_dim + top_k * prompt_dim  # 256 + 256 = 512
        self.head = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Linear(384, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Linear(384, num_cp * action_dim),
        )

    @property
    def active_basis_count(self) -> int:
        return self.pool_size

    def _select_prompts(self, fa, fe, proprio, lang_emb) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_q = self.key_extractor(fa, fe, proprio, lang_emb)  # (B, 64) unit norm
        k_norm = F.normalize(self.prompt_keys, p=2, dim=-1)  # (M, 64)
        sims = x_q @ k_norm.t()  # (B, M)
        top_vals, top_idx = torch.topk(sims, k=self.top_k, dim=-1)  # (B, K)
        prompts = self.prompt_values[top_idx].reshape(x_q.shape[0], -1)  # (B, K*P)
        pull_loss = (1.0 - top_vals).mean()
        return prompts, pull_loss, top_idx[:, 0]

    def forward_train(
        self, fa, fe, proprio, lang_emb
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self.ctx_encoder(fa, fe, proprio, lang_emb)
        prompts, pull_loss, top1 = self._select_prompts(fa, fe, proprio, lang_emb)
        cp = self.head(torch.cat([ctx, prompts], dim=-1)).view(-1, self.num_cp, self.action_dim)
        traj = self.bspline_decoder(cp)
        return cp, traj, pull_loss, top1

    def reset_episode(self) -> None:
        pass

    def chunk_snap_back(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def episode_snap_back(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def snap_back(self) -> Tuple[float, float]:
        return 0.0, 0.0

    @torch.no_grad()
    def predict_action_chunk(self, agent_u8, eih_u8, proprio, lang_emb) -> Tuple[torch.Tensor, int, float, float]:
        fa, fe = self.backbone(agent_u8, eih_u8, augment=False)
        _, traj, _, top1 = self.forward_train(fa, fe, proprio, lang_emb)
        return traj, int(top1[0].item()), 0.0, 0.0


class LiberoOrthoLoRALinear(nn.Module):
    """Cumulative Task-ID-free Orthogonal LoRA layer (O-LoRA, Wang et al., EMNLP 2023).

    Maintains a merged frozen weight buffer `W_frozen` + frozen normalized `A_bank` so
    forward cost stays fast and O(1) even as K grows to 100 tasks!
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 8.0) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / float(rank)
        self.register_buffer("merged_lora_weight", torch.zeros(out_features, in_features))
        self.register_buffer("frozen_A_rows", torch.empty(0, in_features))
        self.active_A: Optional[nn.Parameter] = None
        self.active_B: Optional[nn.Parameter] = None
        self.num_adapters: int = 0

    def consolidate_active_adapter(self) -> None:
        """Fold finished task's LoRA delta into frozen merged weight and archive its A subspace."""
        if self.active_A is not None and self.active_B is not None:
            with torch.no_grad():
                delta_w = self.scale * (self.active_B.detach() @ self.active_A.detach())
                self.merged_lora_weight.add_(delta_w)
                a_norm = F.normalize(self.active_A.detach().float(), p=2, dim=-1).to(self.frozen_A_rows.dtype)
                if self.frozen_A_rows.numel() == 0:
                    self.frozen_A_rows = a_norm
                else:
                    self.frozen_A_rows = torch.cat([self.frozen_A_rows, a_norm], dim=0)
            self.active_A = None
            self.active_B = None

    def start_new_adapter(self, device: torch.device) -> List[nn.Parameter]:
        for p in self.base.parameters():
            p.requires_grad_(False)
            p.grad = None
        self.consolidate_active_adapter()
        A = nn.Parameter(torch.empty(self.rank, self.in_features, device=device))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device))
        self.active_A = A
        self.active_B = B
        self.num_adapters += 1
        return [A, B]

    def orthogonal_loss(self) -> torch.Tensor:
        if self.active_A is None or self.frozen_A_rows.numel() == 0:
            return torch.zeros((), device=self.base.weight.device)
        a_cur = F.normalize(self.active_A.float(), p=2, dim=-1)  # (r, d_in)
        a_prev = self.frozen_A_rows.float()  # (M*r, d_in)
        gram = a_prev @ a_cur.t()  # (M*r, r)
        return (gram ** 2).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.base.weight + self.merged_lora_weight
        out = F.linear(x, w_eff, self.base.bias)
        if self.active_A is not None and self.active_B is not None:
            out = out + self.scale * F.linear(F.linear(x, self.active_A), self.active_B)
        return out


class OLoRALiberoPolicy(nn.Module):
    """O-LoRA (Wang et al., EMNLP 2023) adapted to 7-DoF LIBERO-100 continual control.

    - Replay-Free, Task-ID-free (folded cumulative orthogonal LoRA weights).
    - Task 0 trains base SpatialContextAdapter + B-Spline MLP head, then freezes base permanently.
    - Tasks 1..99 each add rank-8 orthogonal LoRA adapters to all MLP layers, excluding frozen
      weights from AdamW and penalizing ||A_prev @ A_cur^T||_F^2.
    """

    def __init__(
        self,
        backbone: FrozenResNet18DualCam,
        bspline_decoder: BSplineDecoder,
        ctx_dim: int = 256,
        rank: int = 8,
        ortho_lambda: float = 0.50,
        num_cp: int = 8,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.bspline_decoder = bspline_decoder
        self.ctx_dim = ctx_dim
        self.rank = rank
        self.ortho_lambda = ortho_lambda
        self.num_cp = num_cp
        self.action_dim = action_dim

        self.ctx_encoder = SpatialContextAdapter(ctx_dim=ctx_dim)
        self.fc1 = LiberoOrthoLoRALinear(ctx_dim, 384, rank=rank)
        self.ln1 = nn.LayerNorm(384)
        self.fc2 = LiberoOrthoLoRALinear(384, 384, rank=rank)
        self.ln2 = nn.LayerNorm(384)
        self.fc_out = LiberoOrthoLoRALinear(384, num_cp * action_dim, rank=rank)
        self.lora_layers = [self.fc1, self.fc2, self.fc_out]

    @property
    def active_basis_count(self) -> int:
        return max(1, self.fc1.num_adapters + 1)

    def prepare_task(self, task_idx: int, device: torch.device) -> List[nn.Parameter]:
        if task_idx == 0:
            return [p for p in self.parameters() if p.requires_grad]
        for mod in (self.ctx_encoder, self.ln1, self.ln2):
            for p in mod.parameters():
                p.requires_grad_(False)
                p.grad = None
        active: List[nn.Parameter] = []
        for layer in self.lora_layers:
            active.extend(layer.start_new_adapter(device))
        return active

    def finalize_task(self) -> None:
        for layer in self.lora_layers:
            layer.consolidate_active_adapter()

    def orthogonal_loss(self) -> torch.Tensor:
        return sum(layer.orthogonal_loss() for layer in self.lora_layers)

    def forward_train(self, fa, fe, proprio, lang_emb) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = self.ctx_encoder(fa, fe, proprio, lang_emb)
        h = F.silu(self.ln1(self.fc1(ctx)))
        h = F.silu(self.ln2(self.fc2(h)))
        cp = self.fc_out(h).view(-1, self.num_cp, self.action_dim)
        traj = self.bspline_decoder(cp)
        return cp, traj

    def reset_episode(self) -> None:
        pass

    def chunk_snap_back(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def episode_snap_back(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def snap_back(self) -> Tuple[float, float]:
        return 0.0, 0.0

    @torch.no_grad()
    def predict_action_chunk(self, agent_u8, eih_u8, proprio, lang_emb) -> Tuple[torch.Tensor, int, float, float]:
        fa, fe = self.backbone(agent_u8, eih_u8, augment=False)
        _, traj = self.forward_train(fa, fe, proprio, lang_emb)
        return traj, 0, 0.0, 0.0


# ==============================================================================
# 2. DER++ Replay Helpers & Unified External CL Training Loop
# ==============================================================================


@torch.no_grad()
def extract_derpp_replay_subset(
    policy: DERPPLiberoPolicy,
    dataset: LiberoTaskDataset,
    text_enc: LightweightTextEncoder,
    device: torch.device,
    ratio: float = 0.05,
    batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """Store 5% replay samples + teacher control-point logits (DER++ dark knowledge) on CPU."""
    keep_n = max(16, int(round(ratio * dataset.num_samples)))
    pick = torch.randperm(dataset.num_samples)[:keep_n]
    lang_vec_dev = text_enc.encode(dataset.language, device)
    policy.eval()
    teacher_cps = []
    for s in range(0, keep_n, batch_size):
        idx = pick[s : s + batch_size]
        av, eih, pr, _, _ = dataset.sample_batch(idx, device)
        lg = lang_vec_dev.unsqueeze(0).expand(av.shape[0], -1)
        fa, fe = policy.backbone(av, eih, augment=False)
        cp_t, _ = policy.forward_train(fa, fe, pr, lg)
        teacher_cps.append(cp_t.detach().cpu())
    return {
        "agent_u8": dataset.agent_u8[pick].detach().cpu(),
        "eih_u8": dataset.eih_u8[pick].detach().cpu(),
        "proprio": dataset.proprio[pick].detach().cpu(),
        "actions_gt": dataset.actions_gt[pick].detach().cpu(),
        "ctrl_gt": dataset.ctrl_gt[pick].detach().cpu(),
        "teacher_cp": torch.cat(teacher_cps, dim=0),
        "lang_vec": text_enc.encode(dataset.language, torch.device("cpu")).detach().cpu(),
    }


def _sample_derpp_minibatch(
    replay_buffers: List[Dict[str, torch.Tensor]],
    rep_bs: int,
    device: torch.device,
    max_tasks_per_step: int = 8,
):
    n_buf = len(replay_buffers)
    chosen = (
        replay_buffers
        if n_buf <= max_tasks_per_step
        else [replay_buffers[i] for i in torch.randperm(n_buf)[:max_tasks_per_step].tolist()]
    )
    per_old = max(4, rep_bs // max(1, len(chosen)))
    avs, eihs, prs, acts, ctrls, tcps, lgs = [], [], [], [], [], [], []
    for buf in chosen:
        total_b = int(buf["proprio"].shape[0])
        take = min(per_old, total_b)
        pick = torch.randint(0, total_b, (take,))
        avs.append(buf["agent_u8"][pick])
        eihs.append(buf["eih_u8"][pick])
        prs.append(buf["proprio"][pick])
        acts.append(buf["actions_gt"][pick])
        ctrls.append(buf["ctrl_gt"][pick])
        tcps.append(buf["teacher_cp"][pick])
        lgs.append(buf["lang_vec"].unsqueeze(0).expand(take, -1))
    return (
        torch.cat(avs, dim=0).to(device, non_blocking=True),
        torch.cat(eihs, dim=0).to(device, non_blocking=True),
        torch.cat(prs, dim=0).to(device, non_blocking=True),
        torch.cat(acts, dim=0).to(device, non_blocking=True),
        torch.cat(ctrls, dim=0).to(device, non_blocking=True),
        torch.cat(tcps, dim=0).to(device, non_blocking=True),
        torch.cat(lgs, dim=0).to(device, non_blocking=True),
    )


@torch.no_grad()
def measure_ext_latency_ms(
    policy: nn.Module, device: torch.device, n_warmup: int = 8, n_iters: int = 40
) -> Dict[str, float]:
    policy.eval()
    av = torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.uint8, device=device)
    eih = torch.randint(0, 256, (1, 3, 128, 128), dtype=torch.uint8, device=device)
    pr = torch.zeros(1, 12, dtype=torch.float32, device=device)
    lg = torch.zeros(1, 128, dtype=torch.float32, device=device)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(n_warmup):
        policy.predict_action_chunk(av, eih, pr, lg)
    _sync()
    e2e = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        policy.predict_action_chunk(av, eih, pr, lg)
        _sync()
        e2e.append((time.perf_counter() - t0) * 1000.0)
    return {"active_latency_ms": round(float(np.median(e2e)), 3), "routing_head_ms": 0.0}


def train_ext_arm_on_task(
    arm_name: str,
    policy: nn.Module,
    dataset: LiberoTaskDataset,
    task_idx: int,
    text_enc: LightweightTextEncoder,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    replay_buffers: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    lang_vec = text_enc.encode(dataset.language, device)
    use_amp = device.type == "cuda"
    scaler_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    if arm_name == "arm_derpp":
        params = [p for p in policy.parameters() if p.requires_grad]
    elif arm_name == "arm_l2p":
        params = [p for p in policy.parameters() if p.requires_grad]
    elif arm_name == "arm_olora":
        params = policy.prepare_task(task_idx, device)
    else:
        raise ValueError(f"Unknown external arm: {arm_name}")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    total_samples = 0
    last_loss = 0.0
    rep_bs = max(16, batch_size // 4)

    policy.train()
    for _ in range(epochs):
        ep_loss_sum = 0.0
        ep_samples = 0
        perm = torch.randperm(dataset.num_samples)
        for start in range(0, dataset.num_samples, batch_size):
            b_idx = perm[start : start + batch_size]
            av, eih, pr, act_gt, ctrl_gt = dataset.sample_batch(b_idx, device)
            b_len = av.shape[0]
            lg = lang_vec.unsqueeze(0).expand(b_len, -1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=scaler_dtype, enabled=use_amp):
                fa, fe = policy.backbone(av, eih, augment=True)

                if arm_name == "arm_derpp":
                    cp_cur, traj_cur = policy.forward_train(fa, fe, pr, lg)
                    loss_cur = F.smooth_l1_loss(traj_cur, act_gt, beta=0.05) + 0.25 * F.mse_loss(cp_cur, ctrl_gt)
                    if replay_buffers:
                        o_av, o_eih, o_pr, o_act, o_ctrl, o_tcp, o_lg = _sample_derpp_minibatch(
                            replay_buffers, rep_bs=rep_bs, device=device
                        )
                        o_fa, o_fe = policy.backbone(o_av, o_eih, augment=True)
                        o_cp, o_traj = policy.forward_train(o_fa, o_fe, o_pr, o_lg)
                        loss_rep_gt = F.smooth_l1_loss(o_traj, o_act, beta=0.05) + 0.25 * F.mse_loss(o_cp, o_ctrl)
                        loss_dark = F.mse_loss(o_cp, o_tcp)
                        loss = loss_cur + 0.5 * loss_rep_gt + 0.5 * loss_dark
                    else:
                        loss = loss_cur

                elif arm_name == "arm_l2p":
                    cp_cur, traj_cur, pull_loss, _ = policy.forward_train(fa, fe, pr, lg)
                    loss = (
                        F.smooth_l1_loss(traj_cur, act_gt, beta=0.05)
                        + 0.25 * F.mse_loss(cp_cur, ctrl_gt)
                        + policy.pull_weight * pull_loss
                    )

                elif arm_name == "arm_olora":
                    cp_cur, traj_cur = policy.forward_train(fa, fe, pr, lg)
                    loss = (
                        F.smooth_l1_loss(traj_cur, act_gt, beta=0.05)
                        + 0.25 * F.mse_loss(cp_cur, ctrl_gt)
                        + policy.ortho_lambda * policy.orthogonal_loss()
                    )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            optimizer.step()
            ep_loss_sum += float(loss.item()) * b_len
            ep_samples += b_len
            total_samples += b_len
        last_loss = ep_loss_sum / max(1, ep_samples)
        scheduler.step()

    if arm_name == "arm_derpp" and replay_buffers is not None:
        replay_buffers.append(extract_derpp_replay_subset(policy, dataset, text_enc, device, ratio=0.05))
    elif arm_name == "arm_olora":
        policy.finalize_task()

    dt = max(1e-4, time.perf_counter() - t0)
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == "cuda" else 0.0
    )
    return {
        "train_time_s": round(dt, 2),
        "samples_per_sec": round(total_samples / dt, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "final_loss": round(last_loss, 5),
        "trainable_params": int(sum(p.numel() for p in params)),
        "active_basis_count": int(getattr(policy, "active_basis_count", 1)),
    }


def run_external_cl_mode(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite_list, task_specs = build_task_specs(args)
    num_tasks = len(task_specs)
    lang_counts = Counter(sp.language.strip().lower() for sp in task_specs)
    dup_lang_indices = {sp.slot_idx for sp in task_specs if lang_counts[sp.language.strip().lower()] > 1}

    out_path = args.out_json or str(
        PROJECT_ROOT / "outputs" / "libero" / f"benchmark_dtra_libero_v31_ext_cl_k{num_tasks}_seed{args.seed}.json"
    )
    milestone_list = [m for m in _parse_int_csv(args.eval_milestones) if 1 <= m <= num_tasks]
    if args.eval_milestones and num_tasks not in milestone_list and num_tasks > 0:
        milestone_list.append(num_tasks)
    milestone_set = set(milestone_list)

    print(
        f"=== LIBERO v3.1 External CL Baselines | suites={suite_list} | K={num_tasks} | "
        f"arms={args.arms} | milestones={milestone_list} | epochs={args.epochs} | n_eval={args.n_eval} ===",
        flush=True,
    )

    backbone = FrozenResNet18DualCam().to(device)
    key_extractor = MultimodalMagneticKeyExtractor(d_topo=64, d_lang=128, seed=args.seed).to(device)
    bspline_decoder = BSplineDecoder(num_control_points=args.num_cp, horizon=args.horizon, degree=3).to(device)
    text_enc = LightweightTextEncoder(dim=128)

    task_meta_rows = [
        {
            "idx": sp.slot_idx,
            "suite": sp.suite_name,
            "bm_task_idx": sp.bm_task_idx,
            "name": sp.task_name,
            "language": sp.language,
            "max_steps": sp.max_steps,
            "is_dup_lang": (sp.slot_idx in dup_lang_indices),
        }
        for sp in task_specs
    ]

    results: Dict[str, Any] = {
        "config": vars(args),
        "suites": suite_list,
        "milestones_configured": milestone_list,
        "out_json": out_path,
        "tasks": task_meta_rows,
        "arms": {},
    }
    if args.resume and os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as fp:
                prev_res = json.load(fp)
            if isinstance(prev_res.get("arms"), dict):
                results["arms"] = prev_res["arms"]
            print(f"  [Resume] Loaded {out_path} (arms: {list(results['arms'].keys())})", flush=True)
        except Exception as exc:
            print(f"  [Resume Warning] {exc}", flush=True)

    env_mgr = SingleActiveEnvManager()
    try:
        for arm_name in args.arms:
            prev_arm = results["arms"].get(arm_name)
            if args.resume and isinstance(prev_arm, dict) and prev_arm.get("status") == "completed":
                print(f"\n--- Skipping Completed Arm: {arm_name} ---", flush=True)
                continue

            set_seed(args.seed)
            arm_t0 = time.perf_counter()
            arm_eval_time_s = 0.0
            print(f"\n--- Running External Arm: {arm_name} ---", flush=True)

            if arm_name == "arm_derpp":
                policy = DERPPLiberoPolicy(
                    backbone, bspline_decoder, ctx_dim=256, num_cp=args.num_cp, action_dim=7
                ).to(device)
            elif arm_name == "arm_l2p":
                policy = L2PLiberoPolicy(
                    backbone, key_extractor, bspline_decoder, ctx_dim=256, num_cp=args.num_cp, action_dim=7
                ).to(device)
            elif arm_name == "arm_olora":
                policy = OLoRALiberoPolicy(
                    backbone, bspline_decoder, ctx_dim=256, rank=8, ortho_lambda=0.50, num_cp=args.num_cp, action_dim=7
                ).to(device)
            else:
                raise ValueError(f"Unsupported external arm: {arm_name}")

            R_sr = np.zeros((num_tasks, num_tasks), dtype=np.float32)
            R_route = np.zeros((num_tasks, num_tasks), dtype=np.float32)
            measured_mask = np.zeros((num_tasks, num_tasks), dtype=bool)
            replay_buffers: Optional[List[Dict[str, torch.Tensor]]] = [] if arm_name == "arm_derpp" else None
            train_logs: List[Dict[str, Any]] = []
            eval_details: Dict[str, dict] = {}
            arm_milestones: Dict[str, dict] = {}
            ckpt_pt = f"{out_path}.{arm_name}.pt"
            saved_t_cur = -1

            if args.resume and isinstance(prev_arm, dict) and prev_arm.get("status") == "running":
                eval_details = dict(prev_arm.get("eval_details", {}))
                arm_milestones = dict(prev_arm.get("milestones", {}))
                if os.path.exists(ckpt_pt):
                    try:
                        ckpt_obj = torch.load(ckpt_pt, map_location=device, weights_only=False)
                        saved_t_cur = int(ckpt_obj.get("t_cur", -1))
                        policy = ckpt_obj["policy"].to(device)
                        train_logs = list(ckpt_obj.get("train_logs", []))
                        replay_buffers = ckpt_obj.get("replay_buffers", replay_buffers)
                        print(f"  [Model Resume] Restored {ckpt_pt} at t_cur={saved_t_cur}", flush=True)
                    except Exception as exc:
                        print(f"  [Model Resume Warning] {exc}", flush=True)
                        saved_t_cur = -1

            results["arms"][arm_name] = {
                "status": "running",
                "completed_tasks": max(0, saved_t_cur + 1),
                "milestones": arm_milestones,
                "eval_details": eval_details,
            }

            for t_cur in range(num_tasks):
                sp = task_specs[t_cur]
                if t_cur <= saved_t_cur and len(train_logs) > t_cur:
                    tr_stat = train_logs[t_cur]
                else:
                    cur_ds = LiberoTaskDataset(
                        task_idx=sp.slot_idx,
                        task_name=sp.task_name,
                        language=sp.language,
                        hdf5_path=sp.hdf5_path,
                        horizon=args.horizon,
                        num_cp=args.num_cp,
                        stride=args.stride,
                        max_demos=args.max_demos,
                        cache_on_gpu=True,
                        device=device,
                    )
                    ds_mb = cur_ds.data_bytes / (1024.0 * 1024.0)
                    if ds_mb > 2600.0 and cur_ds.storage_device.type == "cuda":
                        cur_ds.agent_u8 = cur_ds.agent_u8.cpu()
                        cur_ds.eih_u8 = cur_ds.eih_u8.cpu()
                        cur_ds.proprio = cur_ds.proprio.cpu()
                        cur_ds.actions_gt = cur_ds.actions_gt.cpu()
                        cur_ds.ctrl_gt = cur_ds.ctrl_gt.cpu()
                        cur_ds.storage_device = torch.device("cpu")
                        torch.cuda.empty_cache()

                    tr_stat = train_ext_arm_on_task(
                        arm_name=arm_name,
                        policy=policy,
                        dataset=cur_ds,
                        task_idx=t_cur,
                        text_enc=text_enc,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        device=device,
                        replay_buffers=replay_buffers,
                    )
                    train_logs.append(tr_stat)
                    del cur_ds
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    try:
                        torch.save(
                            {
                                "t_cur": t_cur,
                                "policy": policy,
                                "train_logs": train_logs,
                                "replay_buffers": replay_buffers,
                            },
                            ckpt_pt,
                        )
                    except Exception:
                        pass

                print(
                    f"  [Train {arm_name} T{t_cur}] time={tr_stat['train_time_s']}s | "
                    f"vram={tr_stat['peak_vram_mb']}MB | loss={tr_stat['final_loss']:.4f} | "
                    f"params={tr_stat['trainable_params']:,}",
                    flush=True,
                )

                k_cur = t_cur + 1
                is_milestone = (k_cur in milestone_set) if milestone_set else (t_cur == num_tasks - 1)
                eval_indices = list(range(k_cur)) if is_milestone else [t_cur]

                for ev_i in eval_indices:
                    ev_key = f"t{t_cur}_eval{ev_i}"
                    if args.resume and ev_key in eval_details:
                        ev = eval_details[ev_key]
                    else:
                        ev = eval_policy_on_task(
                            policy=policy,
                            spec=task_specs[ev_i],
                            text_enc=text_enc,
                            n_eval=args.n_eval,
                            exec_steps=args.exec_steps,
                            device=device,
                            env_mgr=env_mgr,
                        )
                        eval_details[ev_key] = ev
                        results["arms"][arm_name]["eval_details"] = eval_details
                        save_json_atomic(out_path, results)
                    R_sr[ev_i, t_cur] = ev["sr"]
                    R_route[ev_i, t_cur] = ev["route_acc"]
                    measured_mask[ev_i, t_cur] = True
                    arm_eval_time_s += float(ev.get("eval_time_s", 0.0))

                if is_milestone:
                    R_sub = forward_fill_R_for_milestone(R_sr, measured_mask, k_cur)
                    m_metrics = calc_cl_metrics(R_sub)
                    lat = measure_ext_latency_ms(policy, device)
                    m_params_mb = _calc_policy_params_mb(policy)
                    m_peak_vram = float(max(x["peak_vram_mb"] for x in train_logs[:k_cur]))
                    m_entry = {
                        "milestone_k": k_cur,
                        "mean_sr": m_metrics["mean_final_sr"],
                        "forgetting_rate": m_metrics["forgetting_rate"],
                        "bwt": m_metrics["bwt"],
                        "fwt": m_metrics["fwt"],
                        "total_params_mb": m_params_mb,
                        "active_latency_ms": lat["active_latency_ms"],
                        "peak_vram_mb": m_peak_vram,
                        "eval_sr_vec": [float(R_sr[i, t_cur]) for i in range(k_cur)],
                    }
                    arm_milestones[str(k_cur)] = m_entry
                    results["arms"][arm_name].update(
                        {
                            "completed_tasks": k_cur,
                            "total_params_mb": m_params_mb,
                            "milestones": arm_milestones,
                        }
                    )
                    save_json_atomic(out_path, results)
                    print(
                        f"  *** [Milestone K={k_cur} | {arm_name}] MeanSR={m_entry['mean_sr']*100:.2f}% | "
                        f"FR={m_entry['forgetting_rate']*100:.2f}% | BWT={m_entry['bwt']*100:+.2f}% | "
                        f"Params={m_params_mb:.2f}MB | Lat={m_entry['active_latency_ms']:.2f}ms ***",
                        flush=True,
                    )

            R_filled = forward_fill_R_for_milestone(R_sr, measured_mask, num_tasks)
            metrics = calc_cl_metrics(R_filled)
            final_lat = measure_ext_latency_ms(policy, device)
            final_params_mb = _calc_policy_params_mb(policy)
            dup_sr = float(np.mean([R_sr[i, num_tasks - 1] for i in range(num_tasks) if i in dup_lang_indices])) if dup_lang_indices else 0.0
            uniq_sr = float(np.mean([R_sr[i, num_tasks - 1] for i in range(num_tasks) if i not in dup_lang_indices]))
            results["arms"][arm_name] = {
                "status": "completed",
                "completed_tasks": num_tasks,
                "mean_final_sr": metrics["mean_final_sr"],
                "forgetting_rate": metrics["forgetting_rate"],
                "bwt": metrics["bwt"],
                "fwt": metrics["fwt"],
                "dup_lang_sr": dup_sr,
                "unique_lang_sr": uniq_sr,
                "total_params_mb": final_params_mb,
                "active_latency_ms": final_lat["active_latency_ms"],
                "max_vram_mb": float(max(x["peak_vram_mb"] for x in train_logs)),
                "wall_elapsed_s": round(time.perf_counter() - arm_t0, 2),
                "total_eval_time_s": round(arm_eval_time_s, 2),
                "milestones": arm_milestones,
                "R_sr": R_filled.tolist(),
                "final_eval": {f"T{i}": eval_details[f"t{num_tasks-1}_eval{i}"] for i in range(num_tasks)},
            }
            save_json_atomic(out_path, results)
            print(
                f"  ==> [DONE {arm_name}] FinalSR={metrics['mean_final_sr']*100:.2f}% | "
                f"FR={metrics['forgetting_rate']*100:.2f}% | BWT={metrics['bwt']*100:+.2f}% | "
                f"DupSR={dup_sr*100:.2f}% | UniqSR={uniq_sr*100:.2f}%",
                flush=True,
            )
            del policy, replay_buffers
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        env_mgr.close()

    print(f"\nSaved LIBERO v3.1 external CL results to: {out_path}", flush=True)


# ==============================================================================
# 3. Mode 2: Extended n_eval=20 Evaluation of Saved K=100 Checkpoints (Zero Retrain)
# ==============================================================================


def _install_main_pickle_aliases() -> None:
    """Bind benchmark_dtra_libero classes onto __main__ so saved .pt checkpoints load cleanly."""
    main_mod = sys.modules["__main__"]
    for attr_name in dir(base_libero):
        obj = getattr(base_libero, attr_name)
        if isinstance(obj, type) and not hasattr(main_mod, attr_name):
            setattr(main_mod, attr_name, obj)


def _wilson_ci_95(k: int, n: int) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / float(n)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


@torch.no_grad()
def eval_policy_per_episode(
    policy: nn.Module,
    spec: TaskSpec,
    text_enc: LightweightTextEncoder,
    n_eval: int,
    exec_steps: int,
    device: torch.device,
    env_mgr: SingleActiveEnvManager,
) -> Dict[str, Any]:
    """Evaluate n_eval distinct initial states [0..n_eval-1] and record per-episode outcomes."""
    t0 = time.perf_counter()
    init_states = load_task_init_states(spec)
    lang_vec = text_enc.encode(spec.language, device)
    env = env_mgr.get(spec.bddl_file)
    max_steps = spec.max_steps

    ep_successes: List[int] = []
    ep_routes: List[float] = []
    ep_post_snaps: List[float] = []

    policy.eval()
    for ep in range(n_eval):
        policy.reset_episode()
        env.reset()
        obs = env.set_init_state(init_states[ep % len(init_states)])
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7))

        step_idx = 0
        ep_ok = False
        r_hits = 0
        r_tot = 0
        while step_idx < max_steps:
            av, eih, pr, lg = extract_obs_tensors(obs, lang_vec, device)
            chunk, sel, _, _ = policy.predict_action_chunk(av, eih, pr, lg)
            chunk_np = np.clip(chunk[0].cpu().numpy(), -1.0, 1.0)
            r_tot += 1
            if sel == spec.slot_idx:
                r_hits += 1

            for k in range(min(exec_steps, chunk_np.shape[0])):
                obs, _, _, _ = env.step(chunk_np[k])
                step_idx += 1
                if env.check_success():
                    ep_ok = True
                    break
                if step_idx >= max_steps:
                    break
            if hasattr(policy, "chunk_snap_back"):
                policy.chunk_snap_back()
            if ep_ok:
                break

        if hasattr(policy, "episode_snap_back"):
            _, post_norm = policy.episode_snap_back()
        else:
            _, post_norm = policy.snap_back()
        ep_successes.append(1 if ep_ok else 0)
        ep_routes.append(float(r_hits / max(1, r_tot)))
        ep_post_snaps.append(float(post_norm))

    return {
        "n_eval": n_eval,
        "ep_success": ep_successes,
        "ep_route_acc": [round(x, 4) for x in ep_routes],
        "sr_all": float(np.mean(ep_successes)),
        "sr_first5": float(np.mean(ep_successes[:5])),
        "sr_heldout": float(np.mean(ep_successes[5:])) if len(ep_successes) > 5 else float(np.mean(ep_successes)),
        "route_all": float(np.mean(ep_routes)),
        "post_snap_max": float(max(ep_post_snaps)) if ep_post_snaps else 0.0,
        "eval_time_s": round(time.perf_counter() - t0, 2),
    }


def _summarize_extended_arm(per_task: Dict[str, dict], task_specs: Sequence[TaskSpec], dup_indices: set) -> Dict[str, Any]:
    all_eps: List[int] = []
    first5_eps: List[int] = []
    held_eps: List[int] = []
    dup_eps: List[int] = []
    uniq_eps: List[int] = []
    l10_eps: List[int] = []
    l90_eps: List[int] = []
    routes_all: List[float] = []
    dup_routes: List[float] = []

    for sp in task_specs:
        tk = f"T{sp.slot_idx}"
        if tk not in per_task:
            continue
        rec = per_task[tk]
        eps = rec["ep_success"]
        all_eps.extend(eps)
        first5_eps.extend(eps[:5])
        if len(eps) > 5:
            held_eps.extend(eps[5:])
        routes_all.append(rec["route_all"])
        if sp.slot_idx in dup_indices:
            dup_eps.extend(eps)
            dup_routes.append(rec["route_all"])
        else:
            uniq_eps.extend(eps)
        if sp.suite_name == "libero_10":
            l10_eps.extend(eps)
        else:
            l90_eps.extend(eps)

    lo_all, hi_all = _wilson_ci_95(sum(all_eps), len(all_eps))
    lo_held, hi_held = _wilson_ci_95(sum(held_eps), len(held_eps)) if held_eps else (0.0, 0.0)
    return {
        "evaluated_tasks": len(per_task),
        "total_episodes": len(all_eps),
        "sr_all": round(float(np.mean(all_eps)), 5) if all_eps else 0.0,
        "sr_all_ci95": [round(lo_all, 5), round(hi_all, 5)],
        "sr_first5": round(float(np.mean(first5_eps)), 5) if first5_eps else 0.0,
        "sr_heldout": round(float(np.mean(held_eps)), 5) if held_eps else 0.0,
        "sr_heldout_ci95": [round(lo_held, 5), round(hi_held, 5)],
        "dup_lang_sr": round(float(np.mean(dup_eps)), 5) if dup_eps else 0.0,
        "unique_lang_sr": round(float(np.mean(uniq_eps)), 5) if uniq_eps else 0.0,
        "l10_sr": round(float(np.mean(l10_eps)), 5) if l10_eps else 0.0,
        "l90_sr": round(float(np.mean(l90_eps)), 5) if l90_eps else 0.0,
        "route_all": round(float(np.mean(routes_all)), 5) if routes_all else 0.0,
        "dup_lang_route": round(float(np.mean(dup_routes)), 5) if dup_routes else 0.0,
    }


def run_extended_eval_mode(args: argparse.Namespace) -> None:
    _install_main_pickle_aliases()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite_list, task_specs = build_task_specs(args)
    num_tasks = len(task_specs)
    lang_counts = Counter(sp.language.strip().lower() for sp in task_specs)
    dup_lang_indices = {sp.slot_idx for sp in task_specs if lang_counts[sp.language.strip().lower()] > 1}
    text_enc = LightweightTextEncoder(dim=128)

    out_path = args.out_json or str(
        PROJECT_ROOT / "outputs" / "libero" / f"eval_libero100_extended_neval{args.n_eval}.json"
    )

    ckpt_dir = Path(args.ckpt_dir)
    seed_targets = [int(x.strip()) for x in args.eval_seeds.split(",") if x.strip()]
    arm_targets = list(args.arms)

    results: Dict[str, Any] = {
        "config": {
            "suites": suite_list,
            "num_tasks": num_tasks,
            "n_eval": args.n_eval,
            "exec_steps": args.exec_steps,
            "eval_seeds": seed_targets,
            "arms": arm_targets,
        },
        "runs": {},
    }
    if args.resume and os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as fp:
                prev = json.load(fp)
            if isinstance(prev.get("runs"), dict):
                results["runs"] = prev["runs"]
            print(f"  [Resume] Loaded {out_path} (runs: {list(results['runs'].keys())})", flush=True)
        except Exception as exc:
            print(f"  [Resume Warning] {exc}", flush=True)

    env_mgr = SingleActiveEnvManager()
    try:
        for sd in seed_targets:
            prefix = (
                "benchmark_dtra_libero_10_libero_90_k100.json"
                if sd == 42
                else f"benchmark_dtra_libero_10_libero_90_k100_seed{sd}.json"
            )
            for arm_name in arm_targets:
                run_key = f"seed{sd}_{arm_name}"
                ckpt_path = ckpt_dir / f"{prefix}.{arm_name}.pt"
                if not ckpt_path.exists():
                    print(f"  [Skip] Missing checkpoint: {ckpt_path}", flush=True)
                    continue

                prev_run = results["runs"].get(run_key, {})
                if args.resume and prev_run.get("status") == "completed":
                    print(f"--- Skipping completed extended eval: {run_key} ---", flush=True)
                    continue

                print(f"\n=== Extended Eval (n_eval={args.n_eval}) | {run_key} | {ckpt_path.name} ===", flush=True)
                ckpt_obj = torch.load(str(ckpt_path), map_location=device, weights_only=False)
                policy = ckpt_obj["policy"].to(device)
                if "key_extractor" in ckpt_obj and hasattr(policy, "key_extractor"):
                    policy.key_extractor = ckpt_obj["key_extractor"].to(device)
                policy.eval()

                per_task = dict(prev_run.get("per_task", {}))
                results["runs"][run_key] = {
                    "status": "running",
                    "seed": sd,
                    "arm": arm_name,
                    "ckpt_path": str(ckpt_path),
                    "ckpt_t_cur": int(ckpt_obj.get("t_cur", -1)),
                    "per_task": per_task,
                }

                t_run0 = time.perf_counter()
                for sp in task_specs:
                    tk = f"T{sp.slot_idx}"
                    if args.resume and tk in per_task and len(per_task[tk].get("ep_success", [])) >= args.n_eval:
                        continue
                    rec = eval_policy_per_episode(
                        policy=policy,
                        spec=sp,
                        text_enc=text_enc,
                        n_eval=args.n_eval,
                        exec_steps=args.exec_steps,
                        device=device,
                        env_mgr=env_mgr,
                    )
                    per_task[tk] = rec
                    results["runs"][run_key]["per_task"] = per_task
                    results["runs"][run_key]["summary"] = _summarize_extended_arm(per_task, task_specs, dup_lang_indices)
                    save_json_atomic(out_path, results)
                    if (sp.slot_idx + 1) % 10 == 0 or sp.slot_idx == num_tasks - 1:
                        sm = results["runs"][run_key]["summary"]
                        print(
                            f"  [{run_key} {sp.slot_idx+1}/{num_tasks}] all20={sm['sr_all']*100:.2f}% "
                            f"(first5={sm['sr_first5']*100:.2f}%, held15={sm['sr_heldout']*100:.2f}%, "
                            f"dup={sm['dup_lang_sr']*100:.2f}%, route={sm['route_all']*100:.2f}%) "
                            f"elapsed={(time.perf_counter()-t_run0)/60:.1f}m",
                            flush=True,
                        )

                results["runs"][run_key]["status"] = "completed"
                results["runs"][run_key]["wall_elapsed_s"] = round(time.perf_counter() - t_run0, 2)
                results["runs"][run_key]["summary"] = _summarize_extended_arm(per_task, task_specs, dup_lang_indices)
                save_json_atomic(out_path, results)
                sm = results["runs"][run_key]["summary"]
                print(
                    f"  ==> [DONE {run_key}] SR(n={args.n_eval})={sm['sr_all']*100:.2f}% "
                    f"CI95=[{sm['sr_all_ci95'][0]*100:.2f}%, {sm['sr_all_ci95'][1]*100:.2f}%] | "
                    f"first5={sm['sr_first5']*100:.2f}% | heldout={sm['sr_heldout']*100:.2f}% | "
                    f"DupSR={sm['dup_lang_sr']*100:.2f}% (DupRoute={sm['dup_lang_route']*100:.2f}%)",
                    flush=True,
                )
                del policy, ckpt_obj
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        env_mgr.close()

    print(f"\nSaved LIBERO extended evaluation results to: {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DTRA v3.1 LIBERO-100 External CL & Extended Evaluation")
    p.add_argument("--mode", choices=["external_cl", "extended_eval"], required=True)
    p.add_argument("--suite", type=str, default="libero_10")
    p.add_argument("--suites", type=str, default="libero_10,libero_90")
    p.add_argument("--max-tasks", type=int, default=100)
    p.add_argument("--task-indices", nargs="+", type=int, default=None)
    p.add_argument("--max-demos", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--num-cp", type=int, default=8)
    p.add_argument("--exec-steps", type=int, default=8)
    p.add_argument("--n-eval", type=int, default=5)
    p.add_argument("--max-rollout-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", nargs="+", default=["arm_derpp", "arm_l2p", "arm_olora"])
    p.add_argument("--eval-milestones", type=str, default="10,30,50,70,100")
    p.add_argument("--eval-seeds", type=str, default="42,100,2026")
    p.add_argument("--ckpt-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "libero"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out-json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "external_cl":
        run_external_cl_mode(args)
    else:
        run_extended_eval_mode(args)


if __name__ == "__main__":
    main()
