"""experiments/benchmark_metaworld_cl.py

[Experiment 4] Standard 3D Robotics Continual Learning Benchmark on Meta-World (MuJoCo).
4개 표준 조작 태스크(button-press-v3 -> door-open-v3 -> window-open-v3 -> faucet-open-v3)에 대해
순수 RGB 카메라(128x128) + 4-DoF 로봇 고유수용감각(EEF XYZ + Gripper)만으로
Cubic B-Spline Action Chunking(H=16 -> Nk=6 제어점, 4-DoF) + 잔차 WTA 슬롯 + 자율 시각 프로토타입 라우팅을 이식·검증합니다.

비교 모델 (동일 데이터 분할: train_tasks[0:35] 수집 / held-out train_tasks[35:50] 15개 평가):
  1) Model 1: Naive FT (단일 헤드 B-Spline CNN+EEF 순차 미세조정)
  2) Model 2: Experience Replay (단일 헤드 + 균형 리플레이 256 청크/태스크)
  3) Model 3: Oracle Task-ID Slots (백본 동결 + 6-mode WTA 슬롯, 테스트 시 Task-ID 제공)
  4) Model 4 (Ours): Autonomous 3D ABK-VLA (무-리플레이, 무-Task-ID, Residual WTA + 시각 프로토타입 라우팅)
     - Select-then-Blend 및 Hard Argmax 추론 동시 평가

사용법:
  python experiments/benchmark_metaworld_cl.py --epochs 55 --n-eval 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import metaworld
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from metaworld.policies import (
    SawyerButtonPressV3Policy,
    SawyerDoorOpenV3Policy,
    SawyerFaucetOpenV3Policy,
    SawyerWindowOpenV3Policy,
)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from experiments.benchmark_autonomous_cl import calc_cl_metrics, set_seed
from experiments.benchmark_three_models import pick_device, safe_savefig, sync
from tools.bspline_core import BSplineDecoder, build_difference_matrices, fit_bspline_chunk_bvls

OUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUT_DIR / "mw4_cache"

TASK_SPECS = [
    ("button-press-v3", SawyerButtonPressV3Policy),
    ("door-open-v3", SawyerDoorOpenV3Policy),
    ("window-open-v3", SawyerWindowOpenV3Policy),
    ("faucet-open-v3", SawyerFaucetOpenV3Policy),
]


def build_rgb_spatial_encoder(feat_dim: int = 256) -> nn.Sequential:
    """2D ABK-VLA와 동일한 4-Stage ConvNet + 4x4 공간 풀링(좌표 정보 보존)."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, 2, 1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, 2, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 3, 2, 1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, 3, 2, 1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, feat_dim),
        nn.ReLU(inplace=True),
    )


class MetaWorldBSplineSlotHead(nn.Module):
    """4-DoF 3D 조작 제어점 예측용 경량 WTA / Residual WTA 슬롯 헤드."""

    def __init__(
        self,
        in_dim: int,
        num_cp: int = 6,
        action_dim: int = 4,
        num_modes: int = 6,
        hidden_dim: int = 256,
        is_residual: bool = False,
        zero_init_std: float = 1e-3,
    ):
        super().__init__()
        self.num_cp = num_cp
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.is_residual = is_residual
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pts_head = nn.Linear(hidden_dim, num_modes * num_cp * action_dim)
        self.sel_head = nn.Linear(hidden_dim, num_modes)
        if is_residual:
            self.prior_weight = nn.Parameter(torch.tensor(1.0))
            nn.init.normal_(self.pts_head.weight, mean=0.0, std=zero_init_std)
            nn.init.zeros_(self.pts_head.bias)
        else:
            self.register_parameter("prior_weight", None)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = cond.shape[0]
        h = self.trunk(cond)
        pts = self.pts_head(h).view(B, self.num_modes, self.num_cp, self.action_dim)
        if not self.is_residual:
            pts = torch.sigmoid(pts)
        sel = self.sel_head(h)
        return pts, sel

    def compose_residual(self, base_cp: torch.Tensor, delta_cp: torch.Tensor) -> torch.Tensor:
        if self.prior_weight is None:
            return torch.sigmoid(delta_cp)
        base_logit = torch.logit(base_cp.clamp(0.05, 0.95))
        if delta_cp.dim() == 4 and base_logit.dim() == 3:
            base_logit = base_logit.unsqueeze(1)
        return torch.sigmoid(self.prior_weight * base_logit + delta_cp)


class MetaWorldSingleHeadPolicy(nn.Module):
    """Naive FT / Experience Replay용 단일 헤드 3D B-Spline 정책."""

    def __init__(self, num_cp: int = 6, horizon: int = 16, action_dim: int = 4, feat_dim: int = 256):
        super().__init__()
        self.num_cp = num_cp
        self.horizon = horizon
        self.action_dim = action_dim
        self.encoder = build_rgb_spatial_encoder(feat_dim=feat_dim)
        self.state_net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_cp * action_dim),
        )
        self.decoder = BSplineDecoder(num_control_points=num_cp, horizon=horizon, degree=3)
        _, d2_np = build_difference_matrices(num_cp)
        self.register_buffer("d2_mat", torch.tensor(d2_np, dtype=torch.float32))

    def forward(self, img: torch.Tensor, st: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = torch.cat([self.encoder(img), self.state_net(st)], dim=1)
        cp = torch.sigmoid(self.head(cond)).view(-1, self.num_cp, self.action_dim)
        act = self.decoder(cp) * 2.0 - 1.0
        return cp, act

    def loss(self, img: torch.Tensor, st: torch.Tensor, tgt_cp: torch.Tensor, tgt_act: torch.Tensor) -> torch.Tensor:
        cp, act = self.forward(img, st)
        l_act = F.l1_loss(act, tgt_act)
        l_cp = F.l1_loss(cp, tgt_cp)
        d2 = torch.matmul(self.d2_mat, cp)
        l_sm = (d2 ** 2).sum(dim=-1).mean()
        return l_act + 0.5 * l_cp + 0.01 * l_sm

    @torch.no_grad()
    def predict(self, img: torch.Tensor, st: torch.Tensor, task_id: Optional[int] = None) -> np.ndarray:
        _, act = self.forward(img, st)
        return act[0].detach().cpu().numpy()


class MetaWorldOracleSlotPolicy(nn.Module):
    """Oracle Task-ID Multi-Slot WTA 3D B-Spline 정책."""

    def __init__(
        self, num_cp: int = 6, horizon: int = 16, action_dim: int = 4, feat_dim: int = 256, num_modes: int = 6
    ):
        super().__init__()
        self.num_cp = num_cp
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.encoder = build_rgb_spatial_encoder(feat_dim=feat_dim)
        self.state_net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True)
        )
        self.slots = nn.ModuleList()
        self.decoder = BSplineDecoder(num_control_points=num_cp, horizon=horizon, degree=3)
        _, d2_np = build_difference_matrices(num_cp)
        self.register_buffer("d2_mat", torch.tensor(d2_np, dtype=torch.float32))
        self._frozen = False
        self._active_idx = -1

    def add_slot_and_freeze_prior(self) -> int:
        dev = next(self.encoder.parameters()).device
        if len(self.slots) > 0:
            self._frozen = True
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.state_net.parameters():
                p.requires_grad = False
            self.encoder.eval()
            self.state_net.eval()
            for sl in self.slots:
                for p in sl.parameters():
                    p.requires_grad = False
                sl.eval()
        idx = len(self.slots)
        self.slots.append(
            MetaWorldBSplineSlotHead(256 + 64, self.num_cp, self.action_dim, self.num_modes, is_residual=False).to(dev)
        )
        self._active_idx = idx
        return idx

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._frozen:
            self.encoder.eval()
            self.state_net.eval()
            for idx, sl in enumerate(self.slots):
                if idx != self._active_idx:
                    sl.eval()
        return self

    def loss(
        self, img: torch.Tensor, st: torch.Tensor, tgt_cp: torch.Tensor, tgt_act: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        B = img.shape[0]
        cond = torch.cat([self.encoder(img), self.state_net(st)], dim=1)
        cps, sel = self.slots[task_id](cond)
        acts = self.decoder(cps.view(B * self.num_modes, self.num_cp, self.action_dim)).view(
            B, self.num_modes, self.horizon, self.action_dim
        ) * 2.0 - 1.0
        l_act = torch.abs(acts - tgt_act[:, None]).mean(dim=(2, 3))
        l_cp = torch.abs(cps - tgt_cp[:, None]).mean(dim=(2, 3))
        d2 = torch.matmul(self.d2_mat, cps.view(B * self.num_modes, self.num_cp, self.action_dim)).view(
            B, self.num_modes, -1, self.action_dim
        )
        l_sm = (d2 ** 2).sum(dim=-1).mean(dim=-1)
        tot = l_act + 0.5 * l_cp + 0.01 * l_sm
        best = tot.argmin(dim=1)
        b_idx = torch.arange(B, device=img.device)
        loss_win = tot[b_idx, best].mean()
        loss_all = tot.mean()
        loss_gate = F.cross_entropy(sel, best)
        return 0.85 * loss_win + 0.05 * loss_all + 0.10 * loss_gate

    @torch.no_grad()
    def predict(self, img: torch.Tensor, st: torch.Tensor, task_id: int) -> np.ndarray:
        cond = torch.cat([self.encoder(img), self.state_net(st)], dim=1)
        cps, sel = self.slots[task_id](cond)
        best = sel.argmax(dim=1)
        cp = cps[torch.arange(img.shape[0], device=img.device), best]
        act = self.decoder(cp) * 2.0 - 1.0
        return act[0].detach().cpu().numpy()


class Autonomous3DABKPolicy(nn.Module):
    """Task-ID-Free Autonomous 3D ABK-VLA 정책 (Hybrid DoG+Conv4 Exemplar Routing + Residual WTA Slots)."""

    def __init__(
        self,
        num_cp: int = 6,
        horizon: int = 16,
        action_dim: int = 4,
        feat_dim: int = 256,
        num_modes: int = 6,
        temperature: float = 0.003,
        routing_mode: str = "blend",
        n_exemplars: int = 8,
    ):
        super().__init__()
        self.num_cp = num_cp
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.temperature = temperature
        self.routing_mode = routing_mode
        self.n_exemplars = n_exemplars

        self.encoder = build_rgb_spatial_encoder(feat_dim=feat_dim)
        self.state_net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True)
        )
        self.slots = nn.ModuleList()
        # Conv4 (4096) + High-Pass Spatial (32x32x3 + 16x16x3 = 3840) = 7936-d key
        self.key_dim = 256 * 4 * 4 + 32 * 32 * 3 + 16 * 16 * 3
        self.register_buffer("slot_exemplar_keys", torch.empty(0, n_exemplars, self.key_dim))
        self.decoder = BSplineDecoder(num_control_points=num_cp, horizon=horizon, degree=3)
        _, d2_np = build_difference_matrices(num_cp)
        self.register_buffer("d2_mat", torch.tensor(d2_np, dtype=torch.float32))
        self._frozen = False
        self._active_idx = -1
        self._cached_w_all: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        self._cached_w_all = None

    @torch.no_grad()
    def _compute_key(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        3D Meta-World RGB 장면용 7,936차원 전경 강조 시각 키 생성.
        - conv_f: (B, 4096) — 4-stage ConvNet 출력 (4x4x256) L2 정규화 특징
        - hp    : |I - AvgPool_15x15(I)| — 공유 배경(나무 테이블/벽)을 억제하고 전경 물체·로봇 에지를 추출(DoG 유사)
        - sp_f  : (B, 3840) — hp의 멀티스케일 풀링(32x32x3 + 16x16x3) L2 정규화 벡터
        - 최종키: normalize([0.5 * conv_f; 1.0 * sp_f]) -> (B, 7936)
        """
        was_tr = self.encoder.training
        self.encoder.eval()
        x = imgs
        for layer in list(self.encoder.children())[:-3]:
            x = layer(x)
        conv_f = F.normalize(F.adaptive_avg_pool2d(x, (4, 4)).flatten(1), p=2, dim=-1)
        blur = F.avg_pool2d(imgs, kernel_size=15, stride=1, padding=7)
        hp = torch.abs(imgs - blur)
        p32 = F.avg_pool2d(hp, 4, 4).flatten(1)
        p16 = F.avg_pool2d(hp, 8, 8).flatten(1)
        sp_f = F.normalize(torch.cat([p32, p16], dim=1), p=2, dim=-1)
        if was_tr and not self._frozen:
            self.encoder.train()
        return F.normalize(torch.cat([conv_f * 0.5, sp_f * 1.0], dim=1), p=2, dim=-1)

    @torch.no_grad()
    def _slot_similarities(self, q: torch.Tensor, max_slots: Optional[int] = None) -> torch.Tensor:
        """Multi-exemplar 코사인 유사도: s_k(q) = max_{e in 1..M_ex} <q, K_{k,e}> -> (B, S)."""
        bank = self.slot_exemplar_keys if max_slots is None else self.slot_exemplar_keys[:max_slots]
        # q: (B, D), bank: (S, M_ex, D) -> max cosine sim over exemplars per slot -> (B, S)
        sims_all = torch.einsum("bd,smd->bsm", q, bank)
        return sims_all.max(dim=-1).values

    @torch.no_grad()
    def register_or_expand_slot(
        self, init_imgs: torch.Tensor, novelty_threshold: float = 0.996
    ) -> Tuple[int, bool, float]:
        dev = self.slot_exemplar_keys.device
        ex_imgs = init_imgs[: self.n_exemplars].to(dev)
        if len(ex_imgs) < self.n_exemplars:
            reps = (self.n_exemplars + len(ex_imgs) - 1) // len(ex_imgs)
            ex_imgs = ex_imgs.repeat(reps, 1, 1, 1)[: self.n_exemplars]
        ex_keys = self._compute_key(ex_imgs)  # (M_ex, D)

        if len(self.slots) == 0:
            head = MetaWorldBSplineSlotHead(256 + 64, self.num_cp, self.action_dim, self.num_modes, is_residual=False).to(dev)
            self.slots.append(head)
            self.slot_exemplar_keys = ex_keys.unsqueeze(0)
            self._active_idx = 0
            return 0, True, 0.0

        sims = self._slot_similarities(ex_keys)  # (M_ex, S)
        max_sim = float(sims.max(dim=-1).values.mean().item())
        best_idx = int(sims.mean(dim=0).argmax().item())
        if max_sim < novelty_threshold:
            self.freeze_prior()
            new_idx = len(self.slots)
            head = MetaWorldBSplineSlotHead(
                256 + 64, self.num_cp, self.action_dim, self.num_modes, is_residual=True
            ).to(dev)
            self.slots.append(head)
            self.slot_exemplar_keys = torch.cat([self.slot_exemplar_keys, ex_keys.unsqueeze(0)], dim=0)
            self._active_idx = new_idx
            return new_idx, True, max_sim
        self._active_idx = best_idx
        return best_idx, False, max_sim

    @torch.no_grad()
    def refresh_slot_0_keys(self, init_imgs: torch.Tensor) -> None:
        dev = self.slot_exemplar_keys.device
        ex_imgs = init_imgs[: self.n_exemplars].to(dev)
        self.slot_exemplar_keys[0] = self._compute_key(ex_imgs)

    def freeze_prior(self) -> None:
        self._frozen = True
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.state_net.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.state_net.eval()
        for sl in self.slots:
            for p in sl.parameters():
                p.requires_grad = False
            sl.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._frozen:
            self.encoder.eval()
            self.state_net.eval()
            for idx, sl in enumerate(self.slots):
                if idx != self._active_idx:
                    sl.eval()
        return self

    def _run_prefix(self, q: torch.Tensor, cond: torch.Tensor, upto: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        슬롯 0..upto-1 순차 잔차 prefix 실행 -> stacked_cp (B, upto, Nk, 4), w_all (B, upto).
        - 슬롯 0은 절대 제어점 p0 예측, 슬롯 s>=1은 직전 기저 조합(base_cp) 위에 logit-잔차 누적.
        """
        slot_cps = []
        p0, s0 = self.slots[0](cond)
        idx0 = s0.argmax(dim=1)
        b_idx = torch.arange(cond.shape[0], device=cond.device)
        slot_cps.append(p0[b_idx, idx0])

        for s in range(1, upto):
            sims_prior = self._slot_similarities(q, max_slots=s)
            w_prior = F.softmax(sims_prior / max(1e-6, self.temperature), dim=-1)
            base_cp = torch.sum(w_prior.unsqueeze(-1).unsqueeze(-1) * torch.stack(slot_cps, dim=1), dim=1)
            dp, ss = self.slots[s](cond)
            idx_s = ss.argmax(dim=1)
            cp_s = self.slots[s].compose_residual(base_cp, dp[b_idx, idx_s])
            slot_cps.append(cp_s)

        sims_all = self._slot_similarities(q, max_slots=upto)
        w_all = F.softmax(sims_all / max(1e-6, self.temperature), dim=-1)
        return torch.stack(slot_cps, dim=1), w_all

    def loss(self, img: torch.Tensor, st: torch.Tensor, tgt_cp: torch.Tensor, tgt_act: torch.Tensor) -> torch.Tensor:
        s_idx = self._active_idx
        B = img.shape[0]
        cond = torch.cat([self.encoder(img), self.state_net(st)], dim=1)
        head = self.slots[s_idx]

        if s_idx == 0:
            cand_cps, sel = head(cond)
        else:
            with torch.no_grad():
                q = self._compute_key(img)
                stacked_prior, w_prior = self._run_prefix(q, cond.detach(), upto=s_idx)
                base_cp = torch.sum(w_prior.unsqueeze(-1).unsqueeze(-1) * stacked_prior, dim=1)
            delta_cps, sel = head(cond)
            cand_cps = head.compose_residual(base_cp, delta_cps)

        acts = self.decoder(cand_cps.view(B * self.num_modes, self.num_cp, self.action_dim)).view(
            B, self.num_modes, self.horizon, self.action_dim
        ) * 2.0 - 1.0
        l_act = torch.abs(acts - tgt_act[:, None]).mean(dim=(2, 3))
        l_cp = torch.abs(cand_cps - tgt_cp[:, None]).mean(dim=(2, 3))
        d2 = torch.matmul(self.d2_mat, cand_cps.view(B * self.num_modes, self.num_cp, self.action_dim)).view(
            B, self.num_modes, -1, self.action_dim
        )
        l_sm = (d2 ** 2).sum(dim=-1).mean(dim=-1)
        tot = l_act + 0.5 * l_cp + 0.01 * l_sm
        best = tot.argmin(dim=1)
        b_idx = torch.arange(B, device=img.device)
        loss_win = tot[b_idx, best].mean()
        loss_all = tot.mean()
        loss_gate = F.cross_entropy(sel, best)
        return 0.85 * loss_win + 0.05 * loss_all + 0.10 * loss_gate

    @torch.no_grad()
    def set_episode_init_frame(self, init_img: torch.Tensor) -> torch.Tensor:
        q = self._compute_key(init_img)
        sims = self._slot_similarities(q)
        w = F.softmax(sims / max(1e-6, self.temperature), dim=-1)
        self._cached_w_all = w
        return w

    @torch.no_grad()
    def predict(
        self, img: torch.Tensor, st: torch.Tensor, task_id: Optional[int] = None, routing_mode: Optional[str] = None
    ) -> np.ndarray:
        rmode = self.routing_mode if routing_mode is None else routing_mode
        w_all = self._cached_w_all
        if w_all is None:
            w_all = self.set_episode_init_frame(img)

        q = self._compute_key(img)
        cond = torch.cat([self.encoder(img), self.state_net(st)], dim=1)
        if rmode == "argmax":
            top_slot = w_all.argmax(dim=-1)
            limit = int(top_slot.max().item()) + 1
        else:
            top_slot = None
            limit = len(self.slots)

        stacked_cp, _ = self._run_prefix(q, cond, upto=limit)
        if rmode == "argmax":
            final_cp = stacked_cp[torch.arange(img.shape[0], device=img.device), top_slot]
        else:
            final_cp = torch.sum(w_all.unsqueeze(-1).unsqueeze(-1) * stacked_cp, dim=1)
        act = self.decoder(final_cp.clamp(0.0, 1.0)) * 2.0 - 1.0
        return act[0].detach().cpu().numpy()


def collect_or_load_task_data(task_name: str, policy_cls, n_episodes: int = 30, H: int = 16, Nk: int = 6) -> Dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{task_name}_ep{n_episodes}_h{H}_k{Nk}.npz"
    if cache_file.exists():
        with np.load(cache_file) as d:
            return {k: d[k] for k in d.files}

    print(f"[Collect] Meta-World '{task_name}' ({n_episodes} demos)...", flush=True)
    ml1 = metaworld.ML1(task_name, seed=42)
    env = ml1.train_classes[task_name](render_mode="rgb_array", camera_name="corner", width=128, height=128)
    pol = policy_cls()

    imgs, states, cps, acts, ep_ids, init_frames = [], [], [], [], [], []
    ep_idx = 0
    for t in ml1.train_tasks[:35]:
        env.set_task(t)
        obs, _ = env.reset()
        ep_f, ep_s, ep_a = [], [], []
        succ = False
        for _ in range(150):
            fr = np.flipud(env.render()).copy()
            a = np.clip(pol.get_action(obs), -1.0, 1.0)
            ep_f.append(fr)
            ep_s.append(obs[:4].astype(np.float32))
            ep_a.append(a.astype(np.float32))
            obs, _, _, _, info = env.step(a)
            if info.get("success", 0) > 0:
                succ = True
        if not succ or len(ep_a) < H:
            continue
        init_frames.append(ep_f[0])
        a_arr = np.array(ep_a)
        for i in range(0, len(a_arr) - H + 1, 2):
            ch01 = (a_arr[i : i + H] + 1.0) * 0.5
            cp01, _, _ = fit_bspline_chunk_bvls(ch01[:, :2], Nk, 3, "uniform", 0.01, (0.0, 1.0))
            cp23, _, _ = fit_bspline_chunk_bvls(ch01[:, 2:], Nk, 3, "uniform", 0.01, (0.0, 1.0))
            imgs.append(ep_f[i])
            states.append(ep_s[i])
            cps.append(np.concatenate([cp01, cp23], axis=1).astype(np.float32))
            acts.append(a_arr[i : i + H])
            ep_ids.append(ep_idx)
        ep_idx += 1
        if ep_idx >= n_episodes:
            break
    env.close()

    res = {
        "imgs": np.array(imgs, dtype=np.uint8),
        "states": np.array(states, dtype=np.float32),
        "cps": np.array(cps, dtype=np.float32),
        "acts": np.array(acts, dtype=np.float32),
        "ep_ids": np.array(ep_ids, dtype=np.int32),
        "init_frames": np.array(init_frames, dtype=np.uint8),
    }
    np.savez_compressed(cache_file, **res)
    print(f"  -> Saved {cache_file.name}: {ep_idx} eps, {len(imgs)} chunks", flush=True)
    return res


def eval_metaworld_task(
    model: nn.Module,
    task_name: str,
    device: torch.device,
    n_eval: int = 15,
    exec_steps: int = 8,
    task_id: Optional[int] = None,
    routing_mode: Optional[str] = None,
) -> Dict:
    ml1 = metaworld.ML1(task_name, seed=42)
    env = ml1.train_classes[task_name](render_mode="rgb_array", camera_name="corner", width=128, height=128)
    eval_tasks = ml1.train_tasks[35 : 35 + n_eval]
    model.eval()

    succs = 0
    step_lats = []
    matched_slots = 0
    diag_confs = []

    for t in eval_tasks:
        env.set_task(t)
        obs, _ = env.reset()
        f0 = np.flipud(env.render()).copy()
        if isinstance(model, Autonomous3DABKPolicy):
            model.reset_cache()
            t_f0 = torch.from_numpy(f0).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            w = model.set_episode_init_frame(t_f0)[0].detach().cpu().numpy()
            if task_id is not None:
                matched_slots += int(np.argmax(w) == task_id)
                diag_confs.append(float(w[task_id]))

        done = False
        for _ in range(0, 150, exec_steps):
            fr = np.flipud(env.render()).copy()
            tx = torch.from_numpy(fr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            ts = torch.from_numpy(obs[:4].astype(np.float32)).unsqueeze(0).to(device)
            sync(device)
            t0 = time.perf_counter()
            if isinstance(model, Autonomous3DABKPolicy):
                chunk = model.predict(tx, ts, routing_mode=routing_mode)
            else:
                chunk = model.predict(tx, ts, task_id=task_id)
            sync(device)
            step_lats.append((time.perf_counter() - t0) * 1000.0)
            for a in chunk[:exec_steps]:
                obs, _, _, _, info = env.step(np.clip(a, -1.0, 1.0))
                if info.get("success", 0) > 0:
                    done = True
                    break
            if done:
                break
        succs += int(done)
    env.close()
    return {
        "success_rate": float(succs / len(eval_tasks)),
        "avg_step_ms": float(np.mean(step_lats)) if step_lats else 0.0,
        "routing_top1_acc": float(matched_slots / len(eval_tasks)) if diag_confs else None,
        "routing_diag_conf": float(np.mean(diag_confs)) if diag_confs else None,
    }


def train_mw_task(
    model: nn.Module,
    dataset: Dict,
    device: torch.device,
    epochs: int,
    lr: float,
    task_id: int,
    replay_bank: Optional[List[Tuple[torch.Tensor, ...]]] = None,
) -> int:
    X_i = torch.from_numpy(dataset["imgs"]).permute(0, 3, 1, 2).float() / 255.0
    X_s = torch.from_numpy(dataset["states"])
    Y_c = torch.from_numpy(dataset["cps"])
    Y_a = torch.from_numpy(dataset["acts"])
    loader = DataLoader(TensorDataset(X_i, X_s, Y_c, Y_a), batch_size=64, shuffle=True)

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    for ep in range(1, epochs + 1):
        model.train()
        for bi, bs, bc, ba in loader:
            bi, bs, bc, ba = bi.to(device), bs.to(device), bc.to(device), ba.to(device)
            if isinstance(model, MetaWorldOracleSlotPolicy):
                loss = model.loss(bi, bs, bc, ba, task_id=task_id)
            else:
                loss = model.loss(bi, bs, bc, ba)

            if replay_bank:
                rep_losses = []
                for ri, rs, rc, ra in replay_bank:
                    idx = torch.randint(0, len(ri), (min(32, len(ri)),))
                    rep_losses.append(model.loss(ri[idx].to(device), rs[idx].to(device), rc[idx].to(device), ra[idx].to(device)))
                loss = 0.5 * loss + 0.5 * torch.stack(rep_losses).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()
        sync(device)
    return n_trainable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--n-demos", type=int, default=30)
    parser.add_argument("--n-eval", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device()

    datasets = {name: collect_or_load_task_data(name, pcls, n_episodes=args.n_demos) for name, pcls in TASK_SPECS}
    tasks = [name for name, _ in TASK_SPECS]
    K = len(tasks)

    model_defs = [
        ("mw_naive", "Naive FT (3D B-Spline)", "naive"),
        ("mw_replay", "Exp. Replay (256/task)", "replay"),
        ("mw_oracle", "Oracle Slots (Task-ID)", "oracle"),
        ("mw_auto_abk", "Autonomous 3D ABK-VLA (Ours)", "auto"),
    ]

    results = {"tasks": tasks, "n_demos": args.n_demos, "n_eval": args.n_eval, "epochs": args.epochs, "models": {}}

    for mkey, mlabel, mkind in model_defs:
        print(f"\n=== [{mlabel}] ===", flush=True)
        set_seed(args.seed)
        if mkind in ("naive", "replay"):
            model = MetaWorldSingleHeadPolicy(6, 16, 4).to(device)
        elif mkind == "oracle":
            model = MetaWorldOracleSlotPolicy(6, 16, 4, num_modes=6).to(device)
        else:
            model = Autonomous3DABKPolicy(6, 16, 4, num_modes=6, temperature=0.003).to(device)

        replay_bank = [] if mkind == "replay" else None
        R_blend = np.zeros((K, K), dtype=np.float32)
        R_argmax = np.zeros((K, K), dtype=np.float32)
        expansion_logs = []
        trainable_list = []

        for s_idx, t_name in enumerate(tasks):
            ds = datasets[t_name]
            init_t = torch.from_numpy(ds["init_frames"][:8]).permute(0, 3, 1, 2).float() / 255.0

            if mkind == "oracle":
                model.add_slot_and_freeze_prior()
            elif mkind == "auto":
                idx, exp, msim = model.register_or_expand_slot(init_t, novelty_threshold=0.996)
                expansion_logs.append({"step": s_idx, "task": t_name, "slot": idx, "expanded": exp, "max_prior_sim": msim})
                print(f"  [Novelty] {t_name}: max_sim={msim:.5f} -> expanded={exp}, slot={idx}", flush=True)

            lr_step = 1e-3
            n_tr = train_mw_task(model, ds, device, epochs=args.epochs, lr=lr_step, task_id=s_idx, replay_bank=replay_bank)
            trainable_list.append(n_tr)

            if mkind == "auto" and s_idx == 0:
                model.refresh_slot_0_keys(init_t)
                model.freeze_prior()

            if replay_bank is not None:
                rng = np.random.default_rng(args.seed + s_idx)
                pick = rng.choice(len(ds["imgs"]), size=min(256, len(ds["imgs"])), replace=False)
                replay_bank.append((
                    torch.from_numpy(ds["imgs"][pick]).permute(0, 3, 1, 2).float() / 255.0,
                    torch.from_numpy(ds["states"][pick]),
                    torch.from_numpy(ds["cps"][pick]),
                    torch.from_numpy(ds["acts"][pick]),
                ))

            # Evaluate seen tasks on held-out initializations
            for e_idx in range(s_idx + 1):
                e_name = tasks[e_idx]
                if mkind == "auto":
                    ev_b = eval_metaworld_task(
                        model, e_name, device, n_eval=args.n_eval, task_id=e_idx, routing_mode="blend"
                    )
                    R_blend[e_idx, s_idx] = ev_b["success_rate"]
                    if e_idx == s_idx or s_idx == K - 1:
                        ev_a = eval_metaworld_task(
                            model, e_name, device, n_eval=args.n_eval, task_id=e_idx, routing_mode="argmax"
                        )
                        R_argmax[e_idx, s_idx] = ev_a["success_rate"]
                    print(
                        f"  Eval {e_name:16s} | Blend SR={ev_b['success_rate']*100:5.1f}% | "
                        f"RouteAcc={ev_b['routing_top1_acc']*100:5.1f}% (Conf {ev_b['routing_diag_conf']*100:5.1f}%)",
                        flush=True,
                    )
                else:
                    ev = eval_metaworld_task(
                        model, e_name, device, n_eval=args.n_eval, task_id=(e_idx if mkind == "oracle" else None)
                    )
                    R_blend[e_idx, s_idx] = ev["success_rate"]
                    print(f"  Eval {e_name:16s} | SR={ev['success_rate']*100:5.1f}%", flush=True)

        total_p = sum(p.numel() for p in model.parameters())
        if mkind == "auto":
            for s_idx in range(1, K - 1):
                for e_idx in range(s_idx):
                    R_argmax[e_idx, s_idx] = R_argmax[e_idx, e_idx]
            # Final routing stats on held-out episodes
            final_route_acc, final_route_conf = [], []
            for e_idx, e_name in enumerate(tasks):
                ev_b = eval_metaworld_task(model, e_name, device, n_eval=args.n_eval, task_id=e_idx, routing_mode="blend")
                R_blend[e_idx, K - 1] = ev_b["success_rate"]
                final_route_acc.append(ev_b["routing_top1_acc"])
                final_route_conf.append(ev_b["routing_diag_conf"])
            results["models"][mkey + "_blend"] = {
                "label": mlabel + " (Blend)",
                "R": R_blend.tolist(),
                "summary": calc_cl_metrics(R_blend),
                "total_params": total_p,
                "trainable_T3": trainable_list[-1],
                "heldout_routing_acc": float(np.mean(final_route_acc)),
                "heldout_routing_conf": float(np.mean(final_route_conf)),
                "expansion_logs": expansion_logs,
            }
            results["models"][mkey + "_argmax"] = {
                "label": mlabel + " (Argmax)",
                "R": R_argmax.tolist(),
                "summary": calc_cl_metrics(R_argmax),
                "total_params": total_p,
                "trainable_T3": trainable_list[-1],
                "heldout_routing_acc": float(np.mean(final_route_acc)),
                "heldout_routing_conf": float(np.mean(final_route_conf)),
            }
        else:
            results["models"][mkey] = {
                "label": mlabel,
                "R": R_blend.tolist(),
                "summary": calc_cl_metrics(R_blend),
                "total_params": total_p,
                "trainable_T3": trainable_list[-1],
            }

    out_json = OUT_DIR / "exp4_metaworld_cl.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary print & plot
    print("\n=== Experiment 4: Meta-World 4-Task Continual Learning Summary ===")
    order_keys = ["mw_naive", "mw_replay", "mw_oracle", "mw_auto_abk_blend", "mw_auto_abk_argmax"]
    for mk in order_keys:
        mv = results["models"][mk]
        sm = mv["summary"]
        sr_str = [round(mv["R"][i][K - 1] * 100, 1) for i in range(K)]
        print(
            f"  {mv['label']:34s} | Avg SR={sm['avg_final_f1']*100:5.1f}% | "
            f"FR={sm['avg_forgetting']*100:5.1f}% | BWT={sm['avg_bwt']*100:+6.1f}% | Tasks={sr_str}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), dpi=160)
    fig.patch.set_facecolor("#F8FAFC")
    labels = [results["models"][k]["label"].replace("Autonomous 3D ABK-VLA (Ours) ", "Auto ABK\n") for k in order_keys]
    srs = [results["models"][k]["summary"]["avg_final_f1"] * 100 for k in order_keys]
    frs = [results["models"][k]["summary"]["avg_forgetting"] * 100 for k in order_keys]
    cols = ["#DC2626", "#F59E0B", "#0EA5E9", "#2563EB", "#16A34A"]
    x = np.arange(len(order_keys))

    axes[0].bar(x, srs, color=cols)
    axes[0].set_title("Meta-World Final Avg Success Rate (%) ↑", fontweight="bold")
    axes[0].set_ylim(0, 108)
    for i, v in enumerate(srs):
        axes[0].text(i, v + 2.0, f"{v:.1f}%", ha="center", fontsize=9)

    axes[1].bar(x, frs, color=cols)
    axes[1].set_title("Meta-World Forgetting Rate (%) ↓", fontweight="bold")
    axes[1].set_ylim(0, max(frs) + 12)
    for i, v in enumerate(frs):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Experiment 4: 3D Meta-World Continual Manipulation (RGB + 4-DoF B-Spline, 15 Held-Out Episodes/Task)", fontweight="bold")
    plt.tight_layout()
    saved_png = safe_savefig(OUT_DIR / "exp4_metaworld_cl.png")
    print(f"\n[Done] Saved {out_json} and {saved_png}")


if __name__ == "__main__":
    main()
