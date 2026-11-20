"""dtra/experiments/metaworld/ablate_dtra_v31_final.py

DTRA v3.1 arXiv Final Manuscript Supplement on 3D Meta-World (8 Tasks, Strict Parity):
  Part A — Clean-vs-OOD Causal Decomposition:
    Isolates the exact causes of the observed Clean-vs-OOD trade-off between
    Matched Frozen MLP (85.6% Clean / 23.6% OOD) and DTRA Full (72.2% Clean / 51.4% OOD):
      1. causal_frozen_mlp_featreg   : Matched Frozen MLP + DTRA's exact x_conv L2-norm scaling (1.3),
                                       training Dropout (p=0.30), and Gaussian feature noise (sigma=0.05).
      2. causal_dtra_no_featreg      : DTRA Full (alpha learned) with conv_dropout=0, conv_noise=0
                                       (isolates pure manifold skeleton + res_scale=0.38 cap + elasticity).
      3. causal_dtra_rigid_no_featreg: DTRA Rigid (alpha=0) with conv_dropout=0, conv_noise=0
                                       (isolates pure manifold skeleton + res_scale=0.38 cap).
      4. causal_dtra_res075          : DTRA Rigid (alpha=0) with relaxed residual cap (res_scale=0.75)
                                       and mild regularization (p=0.15, sigma=0.025) to map Pareto knee.

  Part B — External Published Continual Learning Baselines (~5.25M Matched Capacity, Task-ID-Free):
      5. ext_derpp : DER++ (Buzzega et al., NeurIPS 2020) — 5% replay buffer + dark logit distillation.
      6. ext_l2p   : L2P (Wang et al., CVPR 2022) — Replay-Free, Task-ID-free cosine-keyed Prompt Pool.
      7. ext_olora : O-LoRA (Wang et al., EMNLP 2023) — Replay-Free, Task-ID-free orthogonal subspace LoRA.

Frozen Zenodo scripts (`benchmark_dtra_3d_metaworld.py`, `ablate_dtra_scientific.py`) are left
completely untouched; this module imports and reuses their exact datasets, encoders, and evaluators.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V1_SRC_DIR = PROJECT_ROOT / "v1_drawing_vla" / "src"
for _p in (str(PROJECT_ROOT), str(V1_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from dtra.core.bspline_core import BSplineDecoder, build_difference_matrices
from dtra.experiments.metaworld.ablate_dtra_scientific import (
    ControlledDTRAPolicy,
    HybridVisualKeyExtractor,
    MatchedFrozenMLPPolicy,
    MetaWorldEvalRegistry,
    eval_arm_task,
    train_arm_task_fast,
)
from dtra.experiments.metaworld.ablate_dtra_supplemental import SharedMultimodalPolicy
from dtra.experiments.metaworld.benchmark_dtra_3d_metaworld import (
    LightweightTextEncoder,
    OUT_DIR,
    TASK_INSTRUCTIONS,
    TASK_SPECS_8,
    pick_device,
    sync,
)
from v1_drawing_vla.src.experiments.benchmark_autonomous_cl import calc_cl_metrics, set_seed
from v1_drawing_vla.src.experiments.benchmark_metaworld_cl import (
    build_rgb_spatial_encoder,
    collect_or_load_task_data,
)


def save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ==============================================================================
# Part A: Feature-Regularized Matched Frozen MLP (Exact Causal Control)
# ==============================================================================


class FeatureRegFrozenMLPPolicy(MatchedFrozenMLPPolicy):
    """Matched Frozen MLP (Multimodal) with DTRA's exact x_conv normalization,
    training dropout (p=0.30), and Gaussian feature noise (sigma=0.05).

    Directly tests whether DTRA's severe-OOD resilience stems solely from Conv4
    feature noise regularization or from the manifold-anchored residual structure.
    """

    def __init__(
        self,
        conv_norm_scale: float = 1.3,
        conv_dropout_p: float = 0.30,
        conv_noise_sigma: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(vision_only=False, **kwargs)
        self.conv_norm_scale = conv_norm_scale
        self.conv_dropout_p = conv_dropout_p
        self.conv_noise_sigma = conv_noise_sigma

    def _transform_cond(self, cond: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        x_conv = cond[:, : self.feat_dim]
        f_st = cond[:, self.feat_dim :]
        x_scaled = F.normalize(x_conv, p=2, dim=-1) * self.conv_norm_scale
        if self.training and not deterministic:
            if self.conv_dropout_p > 0.0:
                x_scaled = F.dropout(x_scaled, p=self.conv_dropout_p, training=True)
            if self.conv_noise_sigma > 0.0:
                x_scaled = x_scaled + self.conv_noise_sigma * torch.randn_like(x_scaled)
        return torch.cat([x_scaled, f_st], dim=-1)

    def loss_from_features(
        self,
        cond: torch.Tensor,
        x_mag: torch.Tensor,
        tgt_cp: torch.Tensor,
        tgt_act: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        cond_reg = self._transform_cond(cond, deterministic=deterministic)
        return super().loss_from_features(cond_reg, x_mag, tgt_cp, tgt_act)

    def loss(
        self,
        img: torch.Tensor,
        st: torch.Tensor,
        tgt_cp: torch.Tensor,
        tgt_act: torch.Tensor,
        instruction: str,
        deterministic: bool = False,
    ) -> torch.Tensor:
        cond, x_mag = self._forward_trunk_with_grad(img, st, [instruction] * img.shape[0])
        return self.loss_from_features(cond, x_mag, tgt_cp, tgt_act, deterministic=deterministic)

    @torch.no_grad()
    def predict(self, img: torch.Tensor, st: torch.Tensor, instruction: str) -> Tuple[np.ndarray, Dict[str, int]]:
        B = img.shape[0]
        cond, x_mag = self._forward_trunk_with_grad(img, st, [instruction] * B)
        cond_reg = self._transform_cond(cond, deterministic=True)
        x_hat = F.normalize(x_mag, p=2, dim=-1)
        sims = torch.matmul(x_hat, self.slot_keys.t())
        best_idx = int(sims[0].argmax().item())
        feat = torch.cat([cond_reg, x_mag], dim=-1)
        cand_cps, sel = self.slots[best_idx](feat)
        best_mode = sel.argmax(dim=1)
        p_star = cand_cps[torch.arange(B, device=img.device), best_mode].clamp(0.0, 1.0)
        act = self.decoder(p_star) * 2.0 - 1.0
        return act[0].detach().cpu().numpy(), {"slot_idx": best_idx, "fine_idx": best_idx}


def _train_slot_or_dtra_arm(
    model: nn.Module,
    tasks: Sequence[str],
    datasets: Dict[str, Dict],
    registry: MetaWorldEvalRegistry,
    device: torch.device,
    seed: int,
    epochs: int,
    n_eval: int,
    n_ood_eval: int,
) -> Dict:
    k_total = len(tasks)
    r_mat = np.zeros((k_total, k_total), dtype=np.float32)
    r_route = np.zeros((k_total, k_total), dtype=np.float32)
    train_s_total = 0.0

    for s_idx, t_name in enumerate(tasks):
        ds = datasets[t_name]
        instr = TASK_INSTRUCTIONS[t_name]
        init_t = torch.from_numpy(ds["init_frames"][:8]).permute(0, 3, 1, 2).float() / 255.0
        model.register_or_expand_task(init_t, instr)
        # FeatureRegFrozenMLPPolicy has stochastic dropout/noise -> disable non-det equivalence check
        verify_eq = not isinstance(model, FeatureRegFrozenMLPPolicy)
        _, dt = train_arm_task_fast(
            model,
            ds,
            instr,
            device,
            epochs=epochs,
            lr=1e-3,
            task_idx=s_idx,
            verify_equivalence=verify_eq,
        )
        train_s_total += dt
        for e_idx in range(s_idx + 1):
            e_name = tasks[e_idx]
            ev = eval_arm_task(
                model, registry, e_name, device, n_eval=n_eval, task_id=e_idx, split="train_heldout", seed=seed
            )
            r_mat[e_idx, s_idx] = ev["success_rate"]
            r_route[e_idx, s_idx] = float(ev["routing_top1_acc"] or 0.0)
            print(
                f"    [T{s_idx}->Eval T{e_idx}] {e_name:17s} | SR={ev['success_rate']*100:5.1f}% "
                f"| Route={r_route[e_idx, s_idx]*100:5.1f}%",
                flush=True,
            )

    cl_m = calc_cl_metrics(r_mat)
    ood_srs, ood_routes = [], []
    for e_idx, e_name in enumerate(tasks):
        ev_o = eval_arm_task(
            model,
            registry,
            e_name,
            device,
            n_eval=n_ood_eval,
            task_id=e_idx,
            split="train_heldout",
            noise_std=0.20,
            seed=seed + 2000,
        )
        ood_srs.append(float(ev_o["success_rate"]))
        ood_routes.append(float(ev_o["routing_top1_acc"] or 0.0))

    total_params = int(sum(p.numel() for p in model.parameters()))
    return {
        "total_params": total_params,
        "train_seconds_total": round(train_s_total, 2),
        "R": r_mat.tolist(),
        "R_route": r_route.tolist(),
        "clean_avg_sr": float(cl_m["avg_final_f1"]),
        "fr": float(cl_m["avg_forgetting"]),
        "bwt": float(cl_m["avg_bwt"]),
        "clean_route_acc": float(np.mean(r_route[:, k_total - 1])),
        "clean_per_task": [float(r_mat[i, k_total - 1]) for i in range(k_total)],
        "ood_severe_sr": float(np.mean(ood_srs)),
        "ood_route_acc": float(np.mean(ood_routes)),
        "ood_per_task": ood_srs,
    }


# ==============================================================================
# Part B: External Published CL Baselines (~5.25M Parameter Parity, Task-ID-Free)
# ==============================================================================


class DERPPSharedPolicy(SharedMultimodalPolicy):
    """DER++ (Buzzega et al., NeurIPS 2020) over identical ~5.25M SharedMultimodalPolicy."""

    def forward_logits(
        self, imgs: torch.Tensor, st: torch.Tensor, instructions: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat, _ = self.forward_features(imgs, st, instructions)
        h = self.shared_mlp(feat)
        pts_logits = self.pts_head(h)
        sel_logits = self.gate_head(h)
        return pts_logits, sel_logits

    def loss_from_logits(
        self,
        pts_logits: torch.Tensor,
        sel_logits: torch.Tensor,
        tgt_cp: torch.Tensor,
        tgt_act: torch.Tensor,
    ) -> torch.Tensor:
        bsz = pts_logits.shape[0]
        cand_cps = torch.sigmoid(pts_logits).view(bsz, self.num_modes, self.num_cp, self.action_dim)
        acts = (
            self.decoder(cand_cps.view(bsz * self.num_modes, self.num_cp, self.action_dim)).view(
                bsz, self.num_modes, self.horizon, self.action_dim
            )
            * 2.0
            - 1.0
        )
        l_act = torch.abs(acts - tgt_act[:, None]).mean(dim=(2, 3))
        l_cp = torch.abs(cand_cps - tgt_cp[:, None]).mean(dim=(2, 3))
        d2 = torch.matmul(self.d2_mat, cand_cps.view(bsz * self.num_modes, self.num_cp, self.action_dim)).view(
            bsz, self.num_modes, -1, self.action_dim
        )
        l_sm = (d2 ** 2).sum(dim=-1).mean(dim=-1)
        tot = l_act + 0.5 * l_cp + 0.01 * l_sm
        best = tot.argmin(dim=1)
        b_idx = torch.arange(bsz, device=pts_logits.device)
        return 0.85 * tot[b_idx, best].mean() + 0.05 * tot.mean() + 0.10 * F.cross_entropy(sel_logits, best)


class L2PPromptPoolPolicy(nn.Module):
    """L2P (Wang et al., CVPR 2022) adapted to continuous visuomotor control (~5.25M).

    - Replay-Free, Task-ID-free.
    - Trains Conv4+state_net on Task 0 then freezes the perception trunk (strict parity with
      MatchedFrozenMLP & DTRA).
    - Maintains a learnable Key-Value Prompt Pool (M=24 prompts, top-k=4 selection via cosine
      similarity in the 128-d multimodal query space) with L2P pull-key regularization.
    - Concatenates selected prompt embeddings with observation features into a shared policy MLP.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        d_key: int = 128,
        pool_size: int = 24,
        top_k: int = 4,
        prompt_dim: int = 128,
        hidden_dim: int = 1480,
        num_cp: int = 6,
        horizon: int = 16,
        action_dim: int = 4,
        num_modes: int = 6,
        pull_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.d_key = d_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.prompt_dim = prompt_dim
        self.num_cp = num_cp
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.pull_weight = pull_weight
        self._frozen_trunk = False

        self.encoder = build_rgb_spatial_encoder(feat_dim=feat_dim)
        self.state_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.text_encoder = LightweightTextEncoder(d_lang=64)
        self.visual_key_extractor = HybridVisualKeyExtractor()

        # Learnable L2P prompt keys and values
        self.prompt_keys = nn.Parameter(F.normalize(torch.randn(pool_size, d_key), dim=-1))
        self.prompt_values = nn.Parameter(torch.randn(pool_size, prompt_dim) * 0.02)

        in_dim = feat_dim + 64 + d_key + top_k * prompt_dim  # 448 + 512 = 960
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.pts_head = nn.Linear(hidden_dim, num_modes * num_cp * action_dim)
        self.gate_head = nn.Linear(hidden_dim, num_modes)

        self.decoder = BSplineDecoder(num_control_points=num_cp, horizon=horizon, degree=3)
        _, d2_np = build_difference_matrices(num_cp)
        self.register_buffer("d2_mat", torch.tensor(d2_np, dtype=torch.float32))

    def reset(self) -> None:
        return None

    def freeze_trunk(self) -> None:
        self._frozen_trunk = True
        for mod in (self.encoder, self.state_net):
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._frozen_trunk:
            self.encoder.eval()
            self.state_net.eval()
        return self

    def _compose_key(self, vis_64: torch.Tensor, instructions: Sequence[str], device: torch.device) -> torch.Tensor:
        lang_64 = self.text_encoder(list(instructions), device=device)
        return F.normalize(torch.cat([0.55 * vis_64, 0.835 * lang_64], dim=-1), p=2, dim=-1)

    def _select_prompts(self, x_mag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(x_mag, p=2, dim=-1)  # (B, 128)
        k_norm = F.normalize(self.prompt_keys, p=2, dim=-1)  # (M, 128)
        sims = q @ k_norm.t()  # (B, M)
        top_vals, top_idx = torch.topk(sims, k=self.top_k, dim=-1)  # (B, K)
        sel_prompts = self.prompt_values[top_idx].reshape(x_mag.shape[0], -1)  # (B, K*P)
        pull_loss = (1.0 - top_vals).mean()
        return sel_prompts, pull_loss

    def forward_head(
        self, cond: torch.Tensor, x_mag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sel_prompts, pull_loss = self._select_prompts(x_mag)
        feat = torch.cat([cond, x_mag, sel_prompts], dim=-1)
        h = self.shared_mlp(feat)
        bsz = cond.shape[0]
        cand_cps = torch.sigmoid(self.pts_head(h)).view(bsz, self.num_modes, self.num_cp, self.action_dim)
        sel = self.gate_head(h)
        return cand_cps, sel, pull_loss

    def loss(
        self,
        imgs: torch.Tensor,
        st: torch.Tensor,
        tgt_cp: torch.Tensor,
        tgt_act: torch.Tensor,
        instructions: Sequence[str],
    ) -> torch.Tensor:
        bsz = imgs.shape[0]
        cond = torch.cat([self.encoder(imgs), self.state_net(st)], dim=-1)
        vis_64 = self.visual_key_extractor(imgs)
        x_mag = self._compose_key(vis_64, instructions, imgs.device)
        cand_cps, sel, pull_loss = self.forward_head(cond, x_mag)

        acts = (
            self.decoder(cand_cps.view(bsz * self.num_modes, self.num_cp, self.action_dim)).view(
                bsz, self.num_modes, self.horizon, self.action_dim
            )
            * 2.0
            - 1.0
        )
        l_act = torch.abs(acts - tgt_act[:, None]).mean(dim=(2, 3))
        l_cp = torch.abs(cand_cps - tgt_cp[:, None]).mean(dim=(2, 3))
        d2 = torch.matmul(self.d2_mat, cand_cps.view(bsz * self.num_modes, self.num_cp, self.action_dim)).view(
            bsz, self.num_modes, -1, self.action_dim
        )
        l_sm = (d2 ** 2).sum(dim=-1).mean(dim=-1)
        tot = l_act + 0.5 * l_cp + 0.01 * l_sm
        best = tot.argmin(dim=1)
        b_idx = torch.arange(bsz, device=imgs.device)
        task_loss = 0.85 * tot[b_idx, best].mean() + 0.05 * tot.mean() + 0.10 * F.cross_entropy(sel, best)
        return task_loss + self.pull_weight * pull_loss

    @torch.no_grad()
    def predict(self, img: torch.Tensor, st: torch.Tensor, instruction: str) -> Tuple[np.ndarray, Dict[str, int]]:
        bsz = img.shape[0]
        cond = torch.cat([self.encoder(img), self.state_net(st)], dim=-1)
        vis_64 = self.visual_key_extractor(img)
        x_mag = self._compose_key(vis_64, [instruction] * bsz, img.device)
        cand_cps, sel, _ = self.forward_head(cond, x_mag)
        best_mode = sel.argmax(dim=1)
        p_star = cand_cps[torch.arange(bsz, device=img.device), best_mode].clamp(0.0, 1.0)
        act = self.decoder(p_star) * 2.0 - 1.0
        return act[0].detach().cpu().numpy(), {"slot_idx": 0}


class OrthoLoRALinear(nn.Module):
    """Cumulative Task-ID-free Orthogonal LoRA linear layer (O-LoRA, Wang et al., EMNLP 2023)."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / float(rank)
        self.lora_As = nn.ParameterList()
        self.lora_Bs = nn.ParameterList()
        self.base_frozen = False

    def freeze_base_and_prior_adapters(self) -> None:
        self.base_frozen = True
        for p in self.base.parameters():
            p.requires_grad_(False)
            p.grad = None
        for p in list(self.lora_As) + list(self.lora_Bs):
            p.requires_grad_(False)
            p.grad = None

    def add_task_adapter(self, device: torch.device) -> Tuple[nn.Parameter, nn.Parameter]:
        self.freeze_base_and_prior_adapters()
        A = nn.Parameter(torch.empty(self.rank, self.in_features, device=device))
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device))
        self.lora_As.append(A)
        self.lora_Bs.append(B)
        return A, B

    def orthogonal_loss(self) -> torch.Tensor:
        if len(self.lora_As) < 2:
            return torch.zeros((), device=self.base.weight.device)
        A_cur = F.normalize(self.lora_As[-1], p=2, dim=-1)  # (r, d_in)
        pen = torch.zeros((), device=A_cur.device)
        for A_prev in self.lora_As[:-1]:
            A_p = F.normalize(A_prev.detach(), p=2, dim=-1)
            gram = A_p @ A_cur.t()  # (r, r)
            pen = pen + (gram ** 2).mean()
        return pen / max(1, len(self.lora_As) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        for A, B in zip(self.lora_As, self.lora_Bs):
            out = out + self.scale * F.linear(F.linear(x, A), B)
        return out


class OLoRAMultimodalPolicy(nn.Module):
    """O-LoRA (Wang et al., EMNLP 2023) adapted to Meta-World (~5.25M base + ortho LoRA).

    - Replay-Free, Task-ID-free (all cumulative adapters sum during inference).
    - Task 0 trains base weights (Conv4 + state_net + base MLP), then freezes base permanently.
    - Tasks 1..7 each allocate rank-16 LoRA adapters on MLP & output layers with orthogonal
      subspace constraint ||A_prev @ A_cur^T||_F^2, excluding frozen weights from AdamW.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        d_key: int = 128,
        hidden_dim: int = 1536,
        rank: int = 16,
        ortho_lambda: float = 0.50,
        num_cp: int = 6,
        horizon: int = 16,
        action_dim: int = 4,
        num_modes: int = 6,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.d_key = d_key
        self.ortho_lambda = ortho_lambda
        self.num_cp = num_cp
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_modes = num_modes
        self._frozen_trunk = False

        self.encoder = build_rgb_spatial_encoder(feat_dim=feat_dim)
        self.state_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.text_encoder = LightweightTextEncoder(d_lang=64)
        self.visual_key_extractor = HybridVisualKeyExtractor()

        in_dim = feat_dim + 64 + d_key  # 448
        self.fc1 = OrthoLoRALinear(in_dim, hidden_dim, rank=rank)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = OrthoLoRALinear(hidden_dim, hidden_dim, rank=rank)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.pts_head = OrthoLoRALinear(hidden_dim, num_modes * num_cp * action_dim, rank=rank)
        self.gate_head = OrthoLoRALinear(hidden_dim, num_modes, rank=max(4, rank // 2))
        self.lora_layers = [self.fc1, self.fc2, self.pts_head, self.gate_head]

        self.decoder = BSplineDecoder(num_control_points=num_cp, horizon=horizon, degree=3)
        _, d2_np = build_difference_matrices(num_cp)
        self.register_buffer("d2_mat", torch.tensor(d2_np, dtype=torch.float32))

    def reset(self) -> None:
        return None

    def prepare_task(self, task_idx: int, device: torch.device) -> List[nn.Parameter]:
        if task_idx == 0:
            return [p for p in self.parameters() if p.requires_grad]
        self._frozen_trunk = True
        for mod in (self.encoder, self.state_net, self.ln1, self.ln2):
            for p in mod.parameters():
                p.requires_grad_(False)
                p.grad = None
            mod.eval()
        active_params: List[nn.Parameter] = []
        for layer in self.lora_layers:
            A, B = layer.add_task_adapter(device)
            active_params.extend([A, B])
        return active_params

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._frozen_trunk:
            self.encoder.eval()
            self.state_net.eval()
            self.ln1.eval()
            self.ln2.eval()
        return self

    def _compose_key(self, vis_64: torch.Tensor, instructions: Sequence[str], device: torch.device) -> torch.Tensor:
        lang_64 = self.text_encoder(list(instructions), device=device)
        return F.normalize(torch.cat([0.55 * vis_64, 0.835 * lang_64], dim=-1), p=2, dim=-1)

    def _forward_mlp(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.silu(self.ln1(self.fc1(feat)))
        h = F.silu(self.ln2(self.fc2(h)))
        bsz = feat.shape[0]
        cand_cps = torch.sigmoid(self.pts_head(h)).view(bsz, self.num_modes, self.num_cp, self.action_dim)
        sel = self.gate_head(h)
        return cand_cps, sel

    def loss(
        self,
        imgs: torch.Tensor,
        st: torch.Tensor,
        tgt_cp: torch.Tensor,
        tgt_act: torch.Tensor,
        instructions: Sequence[str],
    ) -> torch.Tensor:
        bsz = imgs.shape[0]
        cond = torch.cat([self.encoder(imgs), self.state_net(st)], dim=-1)
        vis_64 = self.visual_key_extractor(imgs)
        x_mag = self._compose_key(vis_64, instructions, imgs.device)
        feat = torch.cat([cond, x_mag], dim=-1)
        cand_cps, sel = self._forward_mlp(feat)

        acts = (
            self.decoder(cand_cps.view(bsz * self.num_modes, self.num_cp, self.action_dim)).view(
                bsz, self.num_modes, self.horizon, self.action_dim
            )
            * 2.0
            - 1.0
        )
        l_act = torch.abs(acts - tgt_act[:, None]).mean(dim=(2, 3))
        l_cp = torch.abs(cand_cps - tgt_cp[:, None]).mean(dim=(2, 3))
        d2 = torch.matmul(self.d2_mat, cand_cps.view(bsz * self.num_modes, self.num_cp, self.action_dim)).view(
            bsz, self.num_modes, -1, self.action_dim
        )
        l_sm = (d2 ** 2).sum(dim=-1).mean(dim=-1)
        tot = l_act + 0.5 * l_cp + 0.01 * l_sm
        best = tot.argmin(dim=1)
        b_idx = torch.arange(bsz, device=imgs.device)
        task_loss = 0.85 * tot[b_idx, best].mean() + 0.05 * tot.mean() + 0.10 * F.cross_entropy(sel, best)

        ortho_pen = sum(layer.orthogonal_loss() for layer in self.lora_layers)
        return task_loss + self.ortho_lambda * ortho_pen

    @torch.no_grad()
    def predict(self, img: torch.Tensor, st: torch.Tensor, instruction: str) -> Tuple[np.ndarray, Dict[str, int]]:
        bsz = img.shape[0]
        cond = torch.cat([self.encoder(img), self.state_net(st)], dim=-1)
        vis_64 = self.visual_key_extractor(img)
        x_mag = self._compose_key(vis_64, [instruction] * bsz, img.device)
        feat = torch.cat([cond, x_mag], dim=-1)
        cand_cps, sel = self._forward_mlp(feat)
        best_mode = sel.argmax(dim=1)
        p_star = cand_cps[torch.arange(bsz, device=img.device), best_mode].clamp(0.0, 1.0)
        act = self.decoder(p_star) * 2.0 - 1.0
        return act[0].detach().cpu().numpy(), {"slot_idx": 0}


def _run_derpp(
    tasks: Sequence[str],
    datasets: Dict[str, Dict],
    registry: MetaWorldEvalRegistry,
    device: torch.device,
    seed: int,
    epochs: int,
    n_eval: int,
    n_ood_eval: int,
    er_ratio: float = 0.05,
    alpha_der: float = 0.5,
    beta_der: float = 0.5,
) -> Dict:
    set_seed(seed)
    model = DERPPSharedPolicy().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    k_total = len(tasks)
    r_mat = np.zeros((k_total, k_total), dtype=np.float32)

    buf_imgs, buf_states, buf_cps, buf_acts = [], [], [], []
    buf_pts_logits, buf_sel_logits = [], []
    buf_instrs: List[str] = []
    rng = np.random.default_rng(seed + 888)
    train_s_total = 0.0

    for s_idx, t_name in enumerate(tasks):
        ds = datasets[t_name]
        instr = TASK_INSTRUCTIONS[t_name]
        x_i = (torch.from_numpy(ds["imgs"]).permute(0, 3, 1, 2).float() / 255.0).to(device)
        x_s = torch.from_numpy(ds["states"]).to(device)
        y_c = torch.from_numpy(ds["cps"]).to(device)
        y_a = torch.from_numpy(ds["acts"]).to(device)
        n_samples = x_i.shape[0]
        batch_size = 64

        if len(buf_imgs) > 0:
            bi_all = torch.cat(buf_imgs, dim=0).to(device)
            bs_all = torch.cat(buf_states, dim=0).to(device)
            bc_all = torch.cat(buf_cps, dim=0).to(device)
            ba_all = torch.cat(buf_acts, dim=0).to(device)
            bp_all = torch.cat(buf_pts_logits, dim=0).to(device)
            bg_all = torch.cat(buf_sel_logits, dim=0).to(device)
            buf_n = bi_all.shape[0]
        else:
            bi_all = bs_all = bc_all = ba_all = bp_all = bg_all = None
            buf_n = 0

        opt = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        t0 = time.perf_counter()
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n_samples, device="cpu").to(device)
            for s in range(0, n_samples, batch_size):
                idx = perm[s : s + batch_size]
                pl, sl = model.forward_logits(x_i[idx], x_s[idx], [instr] * len(idx))
                loss = model.loss_from_logits(pl, sl, y_c[idx], y_a[idx])

                if buf_n > 0:
                    rb_n = min(batch_size, buf_n)
                    rb_cpu = torch.randint(0, buf_n, (rb_n,))
                    rb = rb_cpu.to(device)
                    rb_instr = [buf_instrs[int(k)] for k in rb_cpu.tolist()]
                    r_pl, r_sl = model.forward_logits(bi_all[rb], bs_all[rb], rb_instr)
                    loss_distill = F.mse_loss(r_pl, bp_all[rb]) + 0.1 * F.mse_loss(r_sl, bg_all[rb])
                    loss_rep_gt = model.loss_from_logits(r_pl, r_sl, bc_all[rb], ba_all[rb])
                    loss = loss + alpha_der * loss_distill + beta_der * loss_rep_gt

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()
            sched.step()
        sync(device)
        train_s_total += time.perf_counter() - t0

        if s_idx < k_total - 1:
            keep_k = max(1, int(round(er_ratio * n_samples)))
            chosen = rng.choice(n_samples, size=keep_k, replace=False)
            ch_t = torch.from_numpy(chosen).long().to(device)
            model.eval()
            with torch.no_grad():
                t_pl, t_sl = model.forward_logits(x_i[ch_t], x_s[ch_t], [instr] * keep_k)
            buf_imgs.append(x_i[ch_t].cpu())
            buf_states.append(x_s[ch_t].cpu())
            buf_cps.append(y_c[ch_t].cpu())
            buf_acts.append(y_a[ch_t].cpu())
            buf_pts_logits.append(t_pl.cpu())
            buf_sel_logits.append(t_sl.cpu())
            buf_instrs.extend([instr] * keep_k)

        del bi_all, bs_all, bc_all, ba_all, bp_all, bg_all
        for e_idx in range(s_idx + 1):
            e_name = tasks[e_idx]
            ev = eval_arm_task(
                model, registry, e_name, device, n_eval=n_eval, task_id=None, split="train_heldout", seed=seed
            )
            r_mat[e_idx, s_idx] = ev["success_rate"]
            print(f"    [DER++ T{s_idx}->Eval T{e_idx}] {e_name:17s} | SR={ev['success_rate']*100:5.1f}%", flush=True)

    cl_m = calc_cl_metrics(r_mat)
    ood_srs = [
        float(
            eval_arm_task(
                model,
                registry,
                e_name,
                device,
                n_eval=n_ood_eval,
                task_id=None,
                split="train_heldout",
                noise_std=0.20,
                seed=seed + 2000,
            )["success_rate"]
        )
        for e_name in tasks
    ]
    return {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "train_seconds_total": round(train_s_total, 2),
        "R": r_mat.tolist(),
        "clean_avg_sr": float(cl_m["avg_final_f1"]),
        "fr": float(cl_m["avg_forgetting"]),
        "bwt": float(cl_m["avg_bwt"]),
        "clean_per_task": [float(r_mat[i, k_total - 1]) for i in range(k_total)],
        "ood_severe_sr": float(np.mean(ood_srs)),
        "ood_per_task": ood_srs,
    }


def _run_l2p(
    tasks: Sequence[str],
    datasets: Dict[str, Dict],
    registry: MetaWorldEvalRegistry,
    device: torch.device,
    seed: int,
    epochs: int,
    n_eval: int,
    n_ood_eval: int,
) -> Dict:
    set_seed(seed)
    model = L2PPromptPoolPolicy().to(device)
    k_total = len(tasks)
    r_mat = np.zeros((k_total, k_total), dtype=np.float32)
    train_s_total = 0.0

    for s_idx, t_name in enumerate(tasks):
        ds = datasets[t_name]
        instr = TASK_INSTRUCTIONS[t_name]
        x_i = (torch.from_numpy(ds["imgs"]).permute(0, 3, 1, 2).float() / 255.0).to(device)
        x_s = torch.from_numpy(ds["states"]).to(device)
        y_c = torch.from_numpy(ds["cps"]).to(device)
        y_a = torch.from_numpy(ds["acts"]).to(device)
        n_samples = x_i.shape[0]
        batch_size = 64

        if s_idx > 0:
            model.freeze_trunk()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        t0 = time.perf_counter()
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n_samples, device="cpu").to(device)
            for s in range(0, n_samples, batch_size):
                idx = perm[s : s + batch_size]
                loss = model.loss(x_i[idx], x_s[idx], y_c[idx], y_a[idx], [instr] * len(idx))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            sched.step()
        sync(device)
        train_s_total += time.perf_counter() - t0

        for e_idx in range(s_idx + 1):
            e_name = tasks[e_idx]
            ev = eval_arm_task(
                model, registry, e_name, device, n_eval=n_eval, task_id=None, split="train_heldout", seed=seed
            )
            r_mat[e_idx, s_idx] = ev["success_rate"]
            print(f"    [L2P T{s_idx}->Eval T{e_idx}] {e_name:17s} | SR={ev['success_rate']*100:5.1f}%", flush=True)

    cl_m = calc_cl_metrics(r_mat)
    ood_srs = [
        float(
            eval_arm_task(
                model,
                registry,
                e_name,
                device,
                n_eval=n_ood_eval,
                task_id=None,
                split="train_heldout",
                noise_std=0.20,
                seed=seed + 2000,
            )["success_rate"]
        )
        for e_name in tasks
    ]
    return {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "train_seconds_total": round(train_s_total, 2),
        "R": r_mat.tolist(),
        "clean_avg_sr": float(cl_m["avg_final_f1"]),
        "fr": float(cl_m["avg_forgetting"]),
        "bwt": float(cl_m["avg_bwt"]),
        "clean_per_task": [float(r_mat[i, k_total - 1]) for i in range(k_total)],
        "ood_severe_sr": float(np.mean(ood_srs)),
        "ood_per_task": ood_srs,
    }


def _run_olora(
    tasks: Sequence[str],
    datasets: Dict[str, Dict],
    registry: MetaWorldEvalRegistry,
    device: torch.device,
    seed: int,
    epochs: int,
    n_eval: int,
    n_ood_eval: int,
) -> Dict:
    set_seed(seed)
    model = OLoRAMultimodalPolicy().to(device)
    k_total = len(tasks)
    r_mat = np.zeros((k_total, k_total), dtype=np.float32)
    train_s_total = 0.0

    for s_idx, t_name in enumerate(tasks):
        ds = datasets[t_name]
        instr = TASK_INSTRUCTIONS[t_name]
        x_i = (torch.from_numpy(ds["imgs"]).permute(0, 3, 1, 2).float() / 255.0).to(device)
        x_s = torch.from_numpy(ds["states"]).to(device)
        y_c = torch.from_numpy(ds["cps"]).to(device)
        y_a = torch.from_numpy(ds["acts"]).to(device)
        n_samples = x_i.shape[0]
        batch_size = 64

        active_params = model.prepare_task(s_idx, device)
        opt = torch.optim.AdamW(active_params, lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

        t0 = time.perf_counter()
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n_samples, device="cpu").to(device)
            for s in range(0, n_samples, batch_size):
                idx = perm[s : s + batch_size]
                loss = model.loss(x_i[idx], x_s[idx], y_c[idx], y_a[idx], [instr] * len(idx))
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(active_params, 1.0)
                opt.step()
            sched.step()
        sync(device)
        train_s_total += time.perf_counter() - t0

        for e_idx in range(s_idx + 1):
            e_name = tasks[e_idx]
            ev = eval_arm_task(
                model, registry, e_name, device, n_eval=n_eval, task_id=None, split="train_heldout", seed=seed
            )
            r_mat[e_idx, s_idx] = ev["success_rate"]
            print(f"    [O-LoRA T{s_idx}->Eval T{e_idx}] {e_name:17s} | SR={ev['success_rate']*100:5.1f}%", flush=True)

    cl_m = calc_cl_metrics(r_mat)
    ood_srs = [
        float(
            eval_arm_task(
                model,
                registry,
                e_name,
                device,
                n_eval=n_ood_eval,
                task_id=None,
                split="train_heldout",
                noise_std=0.20,
                seed=seed + 2000,
            )["success_rate"]
        )
        for e_name in tasks
    ]
    return {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "train_seconds_total": round(train_s_total, 2),
        "R": r_mat.tolist(),
        "clean_avg_sr": float(cl_m["avg_final_f1"]),
        "fr": float(cl_m["avg_forgetting"]),
        "bwt": float(cl_m["avg_bwt"]),
        "clean_per_task": [float(r_mat[i, k_total - 1]) for i in range(k_total)],
        "ood_severe_sr": float(np.mean(ood_srs)),
        "ood_per_task": ood_srs,
    }


ALL_ARMS = [
    "causal_frozen_mlp_featreg",
    "causal_dtra_no_featreg",
    "causal_dtra_rigid_no_featreg",
    "causal_dtra_res075",
    "ext_derpp",
    "ext_l2p",
    "ext_olora",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meta-World v3.1 Causal Ablation & External CL Baselines")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=55)
    p.add_argument("--n-demos", type=int, default=30)
    p.add_argument("--n-eval", type=int, default=15)
    p.add_argument("--n-ood-eval", type=int, default=15)
    p.add_argument("--arms", nargs="+", default=ALL_ARMS)
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--out-json",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "metaworld" / "ablate_dtra_v31_final_results.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    tasks = list(args.tasks) if args.tasks else [t for t, _ in TASK_SPECS_8]
    out_path = Path(args.out_json)

    print(
        f"=== Meta-World v3.1 Final Supplement | device={device} | seed={args.seed} | "
        f"K={len(tasks)} | epochs={args.epochs} | n_eval={args.n_eval} ===",
        flush=True,
    )

    spec_map = dict(TASK_SPECS_8)
    datasets: Dict[str, Dict] = {}
    for t_name in tasks:
        datasets[t_name] = collect_or_load_task_data(t_name, spec_map[t_name], n_episodes=args.n_demos, H=16, Nk=6)

    registry = MetaWorldEvalRegistry(tasks, ml1_seed=42)
    registry.prepare_all()

    payload: Dict = {
        "config": {
            "seed": args.seed,
            "epochs": args.epochs,
            "n_demos": args.n_demos,
            "n_eval": args.n_eval,
            "n_ood_eval": args.n_ood_eval,
            "tasks": tasks,
            "device": str(device),
        },
        "arms": {},
    }
    if args.resume and out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if isinstance(prev.get("arms"), dict):
                payload["arms"] = prev["arms"]
            print(f"  [Resume] Loaded completed arms: {list(payload['arms'].keys())}", flush=True)
        except Exception as exc:
            print(f"  [Resume Warning] {exc}", flush=True)

    try:
        for arm in args.arms:
            if args.resume and arm in payload["arms"]:
                print(f"--- Skipping completed arm: {arm} ---", flush=True)
                continue
            print(f"\n=== Running Arm: {arm} ===", flush=True)
            set_seed(args.seed)

            if arm == "causal_frozen_mlp_featreg":
                model = FeatureRegFrozenMLPPolicy(
                    conv_norm_scale=1.3, conv_dropout_p=0.30, conv_noise_sigma=0.05
                ).to(device)
                res = _train_slot_or_dtra_arm(
                    model, tasks, datasets, registry, device, args.seed, args.epochs, args.n_eval, args.n_ood_eval
                )
            elif arm == "causal_dtra_no_featreg":
                model = ControlledDTRAPolicy(
                    vision_only=False, rigid_alpha_zero=False, conv_dropout_p=0.0, conv_noise_sigma=0.0, res_scale=0.38
                ).to(device)
                res = _train_slot_or_dtra_arm(
                    model, tasks, datasets, registry, device, args.seed, args.epochs, args.n_eval, args.n_ood_eval
                )
            elif arm == "causal_dtra_rigid_no_featreg":
                model = ControlledDTRAPolicy(
                    vision_only=False, rigid_alpha_zero=True, conv_dropout_p=0.0, conv_noise_sigma=0.0, res_scale=0.38
                ).to(device)
                res = _train_slot_or_dtra_arm(
                    model, tasks, datasets, registry, device, args.seed, args.epochs, args.n_eval, args.n_ood_eval
                )
            elif arm == "causal_dtra_res075":
                model = ControlledDTRAPolicy(
                    vision_only=False, rigid_alpha_zero=True, conv_dropout_p=0.15, conv_noise_sigma=0.025, res_scale=0.75
                ).to(device)
                res = _train_slot_or_dtra_arm(
                    model, tasks, datasets, registry, device, args.seed, args.epochs, args.n_eval, args.n_ood_eval
                )
            elif arm == "ext_derpp":
                res = _run_derpp(
                    tasks, datasets, registry, device, args.seed, args.epochs, args.n_eval, args.n_ood_eval
                )
            elif arm == "ext_l2p":
                res = _run_l2p(
                    tasks, datasets, registry, device, args.seed, args.epochs, args.n_eval, args.n_ood_eval
                )
            elif arm == "ext_olora":
                res = _run_olora(
                    tasks, datasets, registry, device, args.seed, args.epochs, args.n_eval, args.n_ood_eval
                )
            else:
                raise ValueError(f"Unknown arm: {arm}")

            payload["arms"][arm] = res
            save_json_atomic(out_path, payload)
            print(
                f"  ==> [{arm}] Params={res['total_params']:,} | Clean SR={res['clean_avg_sr']*100:5.1f}% | "
                f"FR={res['fr']*100:4.1f}% | BWT={res['bwt']*100:+5.1f}% | OOD={res['ood_severe_sr']*100:5.1f}% | "
                f"time={res['train_seconds_total']:.1f}s",
                flush=True,
            )
            gc.collect()
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
    finally:
        registry.close()

    print(f"\nSaved Meta-World v3.1 results to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
