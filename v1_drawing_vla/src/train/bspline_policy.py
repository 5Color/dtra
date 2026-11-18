"""train/bspline_policy.py

[Phase 2] 적응형 B-Spline 매듭점 기반 VLA 정책 아키텍처 (ABK-VLA).
- 고차원 시계열 좌표 회귀(16x2) 대신 고정된 M개(기본 6개) 제어점(Control Points) 직접 예측.
- 파라미터 불변 해석적 B-Spline 기저 행렬(Frozen Linear Operator) 내장.
- 종단간 미분 가능한 궤적 복원: pred_traj = B_matrix @ pred_cp.
- 2차 차분 평활화 손실(Smoothness Loss) 내장으로 롤아웃 시 루프 아티팩트 방지.
- 두 가지 정책 헤드 지원:
    1. BSplineVisuomotorPolicy: 확정적(Deterministic) 회귀 헤드
    2. BSplineCVAEPolicy: 다양한 그리기 스타일 생성을 위한 잠재 변수(CVAE) 헤드
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.bspline_augment import balanced_pen_transition_loss
from tools.bspline_core import BSplineDecoder, build_difference_matrices
from train.dataset import CHUNK_SIZE, OBS_SIZE


def build_encoder(channels=(32, 64, 128, 256), feat_dim: int = 256, obs_size: int = OBS_SIZE) -> nn.Sequential:
    """(B, 2, 128, 128) -> (B, feat_dim). 4계층 ConvNet 백본."""
    layers, c_in = [], 2
    for c in channels:
        layers += [
            nn.Conv2d(c_in, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        c_in = c
    side = obs_size // (2 ** len(channels))
    return nn.Sequential(*layers, nn.Flatten(), nn.Linear(c_in * side * side, feat_dim), nn.ReLU(inplace=True))


def build_state_net(state_dim: int = 3, state_embed: int = 64) -> nn.Module:
    """펜 상태 (x, y, pen) 3차원 -> 64차원 비선형 임베딩."""
    return nn.Sequential(
        nn.Linear(state_dim, state_embed),
        nn.ReLU(inplace=True),
        nn.Linear(state_embed, state_embed),
        nn.ReLU(inplace=True),
    )


def compute_spatial_pointer(obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """미완성 잔차(target - canvas) 맵으로부터 공간 무게중심, 잔여 비율, 최근접 픽셀 변위 산출.
    얇은 보조 획(1px 줄기·잎) 보존을 위해 uncov와 3x3 평균 풀링을 0.7 * uncov + 0.3 * avg_pool_3x3 비율로 결합.
    """
    canv, tgt = obs[:, 0:1], obs[:, 1:2]
    uncov = (tgt - canv).clamp(min=0.0)
    target_mass = tgt.sum(dim=(-1, -2))  # (B, 1)
    uncov_mass = uncov.sum(dim=(-1, -2))  # (B, 1)
    rem_ratio = (uncov_mass / (target_mass + 1e-5)).clamp(0.0, 1.0)  # (B, 1)
    u = uncov.squeeze(1)  # (B, 128, 128)
    px, py = state[:, 0:1], state[:, 1:2]  # (B, 1)

    gy, gx = torch.meshgrid(
        torch.linspace(0, 1, 128, device=obs.device),
        torch.linspace(0, 1, 128, device=obs.device),
        indexing="ij",
    )
    cx = ((u * gx).sum(dim=(-1, -2), keepdim=True).squeeze(-1)) / (uncov_mass + 1e-5)
    cy = ((u * gy).sum(dim=(-1, -2), keepdim=True).squeeze(-1)) / (uncov_mass + 1e-5)
    has_rem = (uncov_mass > 1.0).float()
    cx = torch.where(has_rem > 0.5, cx, px)
    cy = torch.where(has_rem > 0.5, cy, py)
    dx_center = cx - px
    dy_center = cy - py

    dist_sq = (gx.unsqueeze(0) - px.unsqueeze(-1)) ** 2 + (gy.unsqueeze(0) - py.unsqueeze(-1)) ** 2
    # 얇은 보조 획(1px 줄기·잎) 보존형 3x3 블렌드 (0.7 * uncov + 0.3 * avg_pool_3x3)
    u_pool = F.avg_pool2d(uncov, kernel_size=3, stride=1, padding=1)
    u_blend = (0.7 * uncov + 0.3 * u_pool).squeeze(1)
    has_subst = (u_blend > 0.15).any(dim=-1).any(dim=-1, keepdim=True).unsqueeze(-1)
    mask_target = torch.where(has_subst, u_blend > 0.15, u > 0.08)
    masked_dist = torch.where(mask_target, dist_sq, torch.tensor(1e4, device=obs.device, dtype=dist_sq.dtype))
    min_flat_idx = masked_dist.view(obs.shape[0], -1).argmin(dim=1)
    min_y = (min_flat_idx // 128).float().unsqueeze(1) / 127.0
    min_x = (min_flat_idx % 128).float().unsqueeze(1) / 127.0
    nx = torch.where(has_rem > 0.5, min_x, px)
    ny = torch.where(has_rem > 0.5, min_y, py)
    dx_near = nx - px
    dy_near = ny - py
    return torch.cat([cx, cy, rem_ratio, dx_center, dy_center, dx_near, dy_near], dim=1)


def compute_res_cue(obs: torch.Tensor, grid_size: int = 8) -> torch.Tensor:
    """미완성 잔차 맵(target - canvas)의 적응형 격자 특징."""
    uncov = (obs[:, 1:2] - obs[:, 0:1]).clamp(min=0.0)
    return F.adaptive_avg_pool2d(uncov, (grid_size, grid_size)).flatten(1) * 4.0


# ---------------------------------------------------------------------------
# 1. 확정적 B-Spline 정책 (MLP Head)
# ---------------------------------------------------------------------------

class BSplineVisuomotorPolicy(nn.Module):
    """B-Spline 제어점 직접 예측 정책 네트워크."""

    def __init__(
        self,
        num_control_points: int = 6,
        horizon: int = CHUNK_SIZE,
        feat_dim: int = 256,
        state_embed: int = 64,
        pen_up_weight: float = 2.0,
        smoothness_weight: float = 0.02,
        num_tasks: int = 0,
        use_spatial_pointer: bool = False,
    ):
        super().__init__()
        self.num_control_points = num_control_points
        self.horizon = horizon
        self.pen_up_weight = pen_up_weight
        self.smoothness_weight = smoothness_weight
        self.num_tasks = num_tasks
        self.use_spatial_pointer = use_spatial_pointer

        # 시각 및 상태 인코더
        self.encoder = build_encoder(feat_dim=feat_dim)
        self.state_net = build_state_net(state_dim=3, state_embed=state_embed)
        self.task_emb = nn.Embedding(num_tasks, 32) if num_tasks > 0 else None

        # 해석적 불변 B-Spline 디코더 (파라미터 프리 물리 기저)
        self.bspline_decoder = BSplineDecoder(num_control_points=num_control_points, horizon=horizon, degree=3)

        # 2차 차분 행렬 버퍼 등록 (루프 방지용)
        _, d2 = build_difference_matrices(num_control_points)
        self.register_buffer("d2_mat", torch.tensor(d2, dtype=torch.float32))

        # 제어점 및 펜 상태 예측 헤드
        in_dim = feat_dim + state_embed + (32 if self.task_emb is not None else 0)
        if use_spatial_pointer:
            in_dim += 64 + 7
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_control_points * 2 + horizon),
        )

    def _encode_cond(
        self, obs: torch.Tensor, state: torch.Tensor, task_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        f = self.encoder(obs)
        s = self.state_net(state)
        features = [f, s]

        if self.task_emb is not None and task_id is not None:
            if not torch.is_tensor(task_id):
                task_id = torch.tensor(task_id, dtype=torch.long, device=obs.device)
            if task_id.dim() == 0:
                task_id = task_id.expand(obs.shape[0])
            features.append(self.task_emb(task_id))

        if self.use_spatial_pointer:
            features.append(compute_res_cue(obs))
            features.append(compute_spatial_pointer(obs, state))

        return torch.cat(features, dim=1)

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, 2, 128, 128) 관측 텐서
            state: (B, 3) 펜 상태
            task_id: (B,) 태스크 ID
        Returns:
            pred_cp: (B, M, 2) 제어점
            pred_traj: (B, H, 2) 해석적 B-Spline 복원 궤적
            pen_logits: (B, H) 펜 상태 로짓
        """
        h = self._encode_cond(obs, state, task_id=task_id)
        out = self.head(h)

        cp_raw = out[:, : self.num_control_points * 2].view(-1, self.num_control_points, 2)
        # 캔버스 [0, 1] 범위 유도를 위해 시그모이드 매핑
        pred_cp = torch.sigmoid(cp_raw)
        pen_logits = out[:, self.num_control_points * 2 :]

        # 물리 B-Spline 해석적 재구성
        pred_traj = self.bspline_decoder(pred_cp)

        return pred_cp, pred_traj, pen_logits

    def loss(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        target_cp: torch.Tensor,
        target_traj: torch.Tensor,
        target_pen: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """복합 손실 함수: 제어점 L1 + 궤적 L1 + 평활화 정규화 + 펜 BCE."""
        pred_cp, pred_traj, pen_logits = self(obs, state, task_id=task_id)

        # 1. 제어점 및 궤적 손실
        loss_cp = F.l1_loss(pred_cp, target_cp)
        loss_traj = F.l1_loss(pred_traj, target_traj[..., :2])

        # 2. 2차 차분 평활화 손실 (인접 제어점 급격한 꺾임 억제)
        if len(self.d2_mat) > 0:
            diff2 = torch.matmul(self.d2_mat, pred_cp)
            loss_smooth = torch.mean(torch.sum(diff2 ** 2, dim=-1))
        else:
            loss_smooth = torch.tensor(0.0, device=obs.device)

        # 3. 펜 상태 손실
        pen_weight = torch.where(target_pen < 0.5, torch.full_like(target_pen, self.pen_up_weight), torch.ones_like(target_pen))
        loss_pen = F.binary_cross_entropy_with_logits(pen_logits, target_pen, weight=pen_weight)

        total_loss = loss_traj + 0.5 * loss_cp + self.smoothness_weight * loss_smooth + 0.2 * loss_pen

        metrics = {
            "loss_traj": loss_traj.detach(),
            "loss_cp": loss_cp.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_pen": loss_pen.detach(),
            "total_loss": total_loss.detach(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def predict(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_cp, pred_traj, pen_logits = self(obs, state, task_id=task_id)
        return pred_cp, pred_traj, torch.sigmoid(pen_logits)


# ---------------------------------------------------------------------------
# 2. CVAE 기반 B-Spline 정책 (다양성 및 불확실성 모델링)
# ---------------------------------------------------------------------------

class BSplineCVAEPolicy(nn.Module):
    """CVAE 제어점 생성기: 잠재 공간 z ~ N(0, I)로부터 부드러운 제어점 샘플링."""

    def __init__(
        self,
        num_control_points: int = 6,
        horizon: int = CHUNK_SIZE,
        latent_dim: int = 16,
        feat_dim: int = 256,
        state_embed: int = 64,
        pen_up_weight: float = 2.0,
        kl_weight: float = 0.01,
        smoothness_weight: float = 0.02,
        num_tasks: int = 0,
    ):
        super().__init__()
        self.num_control_points = num_control_points
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.pen_up_weight = pen_up_weight
        self.smoothness_weight = smoothness_weight

        self.encoder = build_encoder(feat_dim=feat_dim)
        self.state_net = build_state_net(state_dim=3, state_embed=state_embed)
        self.task_emb = nn.Embedding(num_tasks, 32) if num_tasks > 0 else None

        self.bspline_decoder = BSplineDecoder(num_control_points=num_control_points, horizon=horizon, degree=3)
        _, d2 = build_difference_matrices(num_control_points)
        self.register_buffer("d2_mat", torch.tensor(d2, dtype=torch.float32))

        cond_dim = feat_dim + state_embed + (32 if self.task_emb is not None else 0)

        # 사전(Prior) 네트워크: p(z | cond)
        self.prior_net = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim * 2),  # mu, logvar
        )

        # 사후(Posterior / Encoder) 네트워크: q(z | cond, target_cp)
        self.posterior_net = nn.Sequential(
            nn.Linear(cond_dim + num_control_points * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim * 2),
        )

        # 디코더 네트워크: p(cp, pen | cond, z)
        self.decoder_net = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_control_points * 2 + horizon),
        )

    def _get_cond(self, obs: torch.Tensor, state: torch.Tensor, task_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.encoder(obs)
        s = self.state_net(state)
        features = [f, s]
        if self.task_emb is not None and task_id is not None:
            if not torch.is_tensor(task_id):
                task_id = torch.tensor(task_id, dtype=torch.long, device=obs.device)
            if task_id.dim() == 0:
                task_id = task_id.expand(obs.shape[0])
            features.append(self.task_emb(task_id))
        return torch.cat(features, dim=1)

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        target_cp: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self._get_cond(obs, state, task_id=task_id)

        if target_cp is not None:
            # 학습 시 사후 분포에서 샘플링
            cp_flat = target_cp.view(target_cp.shape[0], -1)
            post_params = self.posterior_net(torch.cat([cond, cp_flat], dim=1))
            mu_post, logvar_post = torch.chunk(post_params, 2, dim=1)
            std = torch.exp(0.5 * logvar_post)
            eps = torch.randn_like(std)
            z = mu_post + eps * std
        else:
            # 추론 시 사전 분포에서 평균 추출
            prior_params = self.prior_net(cond)
            mu_prior, _ = torch.chunk(prior_params, 2, dim=1)
            z = mu_prior
            mu_post, logvar_post = z, torch.zeros_like(z)

        prior_params = self.prior_net(cond)
        mu_prior, logvar_prior = torch.chunk(prior_params, 2, dim=1)

        dec_out = self.decoder_net(torch.cat([cond, z], dim=1))
        cp_raw = dec_out[:, : self.num_control_points * 2].view(-1, self.num_control_points, 2)
        pred_cp = torch.sigmoid(cp_raw)
        pen_logits = dec_out[:, self.num_control_points * 2 :]

        pred_traj = self.bspline_decoder(pred_cp)

        return pred_cp, pred_traj, pen_logits, (mu_post, logvar_post), (mu_prior, logvar_prior)

    def loss(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        target_cp: torch.Tensor,
        target_traj: torch.Tensor,
        target_pen: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_cp, pred_traj, pen_logits, (mu_post, logvar_post), (mu_prior, logvar_prior) = self(
            obs, state, target_cp=target_cp, task_id=task_id
        )

        loss_cp = F.l1_loss(pred_cp, target_cp)
        loss_traj = F.l1_loss(pred_traj, target_traj[..., :2])

        # KL Divergence: KL( q(z|cond, cp) || p(z|cond) )
        kl_div = torch.mean(
            0.5 * torch.sum(
                logvar_prior - logvar_post + (torch.exp(logvar_post) + (mu_post - mu_prior) ** 2) / torch.exp(logvar_prior) - 1.0,
                dim=1,
            )
        )

        # 평활화
        if len(self.d2_mat) > 0:
            diff2 = torch.matmul(self.d2_mat, pred_cp)
            loss_smooth = torch.mean(torch.sum(diff2 ** 2, dim=-1))
        else:
            loss_smooth = torch.tensor(0.0, device=obs.device)

        pen_weight = torch.where(target_pen < 0.5, torch.full_like(target_pen, self.pen_up_weight), torch.ones_like(target_pen))
        loss_pen = F.binary_cross_entropy_with_logits(pen_logits, target_pen, weight=pen_weight)

        total_loss = loss_traj + 0.5 * loss_cp + self.kl_weight * kl_div + self.smoothness_weight * loss_smooth + 0.2 * loss_pen

        metrics = {
            "loss_traj": loss_traj.detach(),
            "loss_cp": loss_cp.detach(),
            "loss_kl": kl_div.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_pen": loss_pen.detach(),
            "total_loss": total_loss.detach(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def predict(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_cp, pred_traj, pen_logits, _, _ = self(obs, state, target_cp=None, task_id=task_id)
        return pred_cp, pred_traj, torch.sigmoid(pen_logits)


# ---------------------------------------------------------------------------
# 3. 국소 메모리 슬롯 (ARIN / Hippocampal Memory Slot Routing Policy)
# ---------------------------------------------------------------------------

class BSplineSlotPolicy(nn.Module):
    """[Phase 3] ABK-VLA 국소 기억 슬롯(Local Memory Slots) 비파괴적 지속학습 정책.
    - 공유 백본: 4계층 ConvNet 시각 인코더 + State Net + 불변 물리 B-Spline 디코더
    - 국소 슬롯 딕셔너리: 각 태스크마다 독립적인 기하 제어점 예측 슬롯(nn.ModuleDict)
    - 태스크 k 학습 시 슬롯 k만 활성화되고, 이전 슬롯은 완전 동결(Frozen)되어 간섭률 = 0%
    - BatchNorm 통계 누수(Leakage) 방지를 위한 고정 인코더 eval 모드 강제 보장.
    """

    def __init__(
        self,
        num_control_points: int = 6,
        horizon: int = CHUNK_SIZE,
        feat_dim: int = 256,
        state_embed: int = 64,
        pen_up_weight: float = 2.0,
        smoothness_weight: float = 0.02,
    ):
        super().__init__()
        self.num_control_points = num_control_points
        self.horizon = horizon
        self.feat_dim = feat_dim
        self.state_embed = state_embed
        self.pen_up_weight = pen_up_weight
        self.smoothness_weight = smoothness_weight
        self._freeze_encoder = False
        self._active_slot_id = "0"

        self.encoder = build_encoder(feat_dim=feat_dim)
        self.state_net = build_state_net(state_dim=3, state_embed=state_embed)
        self.bspline_decoder = BSplineDecoder(num_control_points=num_control_points, horizon=horizon, degree=3)

        _, d2 = build_difference_matrices(num_control_points)
        self.register_buffer("d2_mat", torch.tensor(d2, dtype=torch.float32))

        self.in_dim = feat_dim + state_embed
        self.slots = nn.ModuleDict()

    def add_slot(self, task_id: int | str):
        key = str(task_id)
        if key not in self.slots:
            slot_net = nn.Sequential(
                nn.Linear(self.in_dim, 384),
                nn.LayerNorm(384),
                nn.ReLU(inplace=True),
                nn.Linear(384, 256),
                nn.LayerNorm(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.num_control_points * 2 + self.horizon),
            )
            self.slots[key] = slot_net

    def freeze_previous_slots(self, current_task_id: int | str, freeze_encoder: bool = True):
        """이전 태스크 슬롯 및 공유 인코더를 동결하여 파국적 망각을 수학적으로 차단."""
        cur_key = str(current_task_id)
        self._active_slot_id = cur_key
        self._freeze_encoder = freeze_encoder

        for k, slot in self.slots.items():
            if k != cur_key:
                slot.eval()
                for p in slot.parameters():
                    p.requires_grad = False

        if freeze_encoder:
            self.encoder.eval()
            self.state_net.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.state_net.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        """train 모드 호출 시 동결된 인코더의 BatchNorm 통계 변질을 원천 차단."""
        super().train(mode)
        if mode and self._freeze_encoder:
            self.encoder.eval()
            self.state_net.eval()
            for k, slot in self.slots.items():
                if k != self._active_slot_id:
                    slot.eval()
        return self

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        task_id: int | str = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = str(task_id)
        if key not in self.slots:
            self.add_slot(key)
            self.slots[key].to(obs.device)

        f = self.encoder(obs)
        s = self.state_net(state)
        h = torch.cat([f, s], dim=1)

        slot_head = self.slots[key]
        out = slot_head(h)

        cp_raw = out[:, : self.num_control_points * 2].view(-1, self.num_control_points, 2)
        pred_cp = torch.sigmoid(cp_raw)
        pen_logits = out[:, self.num_control_points * 2 :]
        pred_traj = self.bspline_decoder(pred_cp)

        return pred_cp, pred_traj, pen_logits

    def loss(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        target_cp: torch.Tensor,
        target_traj: torch.Tensor,
        target_pen: torch.Tensor,
        task_id: int | str = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_cp, pred_traj, pen_logits = self(obs, state, task_id=task_id)
        loss_cp = F.l1_loss(pred_cp, target_cp)
        loss_traj = F.l1_loss(pred_traj, target_traj[..., :2])

        if len(self.d2_mat) > 0:
            diff2 = torch.matmul(self.d2_mat, pred_cp)
            loss_smooth = torch.mean(torch.sum(diff2 ** 2, dim=-1))
        else:
            loss_smooth = torch.tensor(0.0, device=obs.device)

        pen_weight = torch.where(target_pen < 0.5, torch.full_like(target_pen, self.pen_up_weight), torch.ones_like(target_pen))
        loss_pen = F.binary_cross_entropy_with_logits(pen_logits, target_pen, weight=pen_weight)

        total_loss = loss_traj + 0.5 * loss_cp + self.smoothness_weight * loss_smooth + 0.2 * loss_pen
        metrics = {
            "loss_traj": loss_traj.detach(),
            "loss_cp": loss_cp.detach(),
            "loss_smooth": loss_smooth.detach(),
            "loss_pen": loss_pen.detach(),
            "total_loss": total_loss.detach(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def predict(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        task_id: int | str = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_cp, pred_traj, pen_logits = self(obs, state, task_id=task_id)
        return pred_cp, pred_traj, torch.sigmoid(pen_logits)


# ---------------------------------------------------------------------------
# 4. Model C 다중 모달 슬롯 헤드 (CVAE / WTA) 및 Model C 지속학습 정책
# ---------------------------------------------------------------------------

class CVAESlotHead(nn.Module):
    """[Model C] 조건부 변분 오토인코더(CVAE) 기반 독립 슬롯 헤드.

    어블레이션 실증 결과 (balanced_pen_transition_loss + 동변 증강):
      - sprout: 98.9% F1 (Cov 99.7%, 지면선 포함 완전 복원)
      - apple:  99.1% F1 (Cov 98.9%, 자가 종료)
      - star:   91.3% F1 (Cov 90.7%)
    """

    def __init__(self, in_dim: int, num_cp: int = 6, horizon: int = CHUNK_SIZE, latent_dim: int = 16):
        super().__init__()
        self.num_cp = num_cp
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.prior = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim * 2),
        )
        self.post = nn.Sequential(
            nn.Linear(in_dim + num_cp * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim * 2),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_dim + latent_dim, 384),
            nn.LayerNorm(384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_cp * 2 + horizon),
        )

    def forward(
        self,
        cond: torch.Tensor,
        target_cp: Optional[torch.Tensor] = None,
        temp: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        mu_pr, lv_pr = torch.chunk(self.prior(cond), 2, dim=1)
        lv_pr = lv_pr.clamp(-6.0, 4.0)
        if target_cp is not None:
            mu_po, lv_po = torch.chunk(
                self.post(torch.cat([cond, target_cp.view(cond.shape[0], -1)], dim=1)),
                2,
                dim=1,
            )
            lv_po = lv_po.clamp(-6.0, 4.0)
            z = mu_po + torch.exp(0.5 * lv_po) * torch.randn_like(lv_po)
        else:
            mu_po, lv_po = mu_pr, lv_pr
            z = (mu_pr + torch.exp(0.5 * lv_pr) * torch.randn_like(lv_pr) * temp) if temp > 0 else mu_pr

        out = self.dec(torch.cat([cond, z], dim=1))
        cp = torch.sigmoid(out[:, : self.num_cp * 2].view(-1, self.num_cp, 2))
        pen_logits = out[:, self.num_cp * 2 :]
        return cp, pen_logits, (mu_po, lv_po), (mu_pr, lv_pr)


class WTASlotHead(nn.Module):
    """[Model C] K-Hypothesis Winner-Take-All 다중 모달 B-Spline 제어점 슬롯 헤드.

    다중 획 순서·방향(CW/CCW) 충돌로 인한 평균화 붕괴(Mode Averaging)를 방지.
    어블레이션 실증 결과 (K=4, 동변 기하 증강):
      - star:   95.3% F1 (Cov 96.0%, Prec 94.6%)
      - apple:  98.6% F1 (Cov 98.2%, Prec 99.1%)
      - sprout: 92.6% F1 (Prec 100.0%)
      - cat:    77.6% F1 (sample mode, Prec 98.2%)
    """

    def __init__(self, in_dim: int, num_cp: int = 6, horizon: int = CHUNK_SIZE, num_modes: int = 4):
        super().__init__()
        self.num_cp = num_cp
        self.horizon = horizon
        self.num_modes = num_modes
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.LayerNorm(384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        self.out_heads = nn.Linear(256, num_modes * (num_cp * 2 + horizon))
        self.mode_logits = nn.Linear(256, num_modes)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = cond.shape[0]
        h = self.trunk(cond)
        raw = self.out_heads(h).view(B, self.num_modes, self.num_cp * 2 + self.horizon)
        cps = torch.sigmoid(raw[:, :, : self.num_cp * 2]).view(B, self.num_modes, self.num_cp, 2)
        pens = raw[:, :, self.num_cp * 2 :]
        mlogits = self.mode_logits(h)
        return cps, pens, mlogits


class BSplineModelCPolicy(nn.Module):
    """[Model C] B-Spline 기하 증강 + 다중 모달(WTA/CVAE) 슬롯 격리 지속학습 정책."""

    def __init__(
        self,
        num_control_points: int = 6,
        horizon: int = CHUNK_SIZE,
        feat_dim: int = 256,
        state_embed: int = 64,
        slot_type: str = "wta",
        num_modes: int = 4,
        latent_dim: int = 16,
        kl_weight: float = 0.005,
        smoothness_weight: float = 0.02,
        pen_loss_weight: float = 0.3,
        use_spatial_pointer: bool = False,
    ):
        super().__init__()
        if slot_type not in ("wta", "cvae"):
            raise ValueError(f"Unsupported slot_type: {slot_type}")
        self.num_control_points = num_control_points
        self.horizon = horizon
        self.feat_dim = feat_dim
        self.state_embed = state_embed
        self.slot_type = slot_type
        self.num_modes = num_modes
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.smoothness_weight = smoothness_weight
        self.pen_loss_weight = pen_loss_weight
        self.use_spatial_pointer = use_spatial_pointer
        self._freeze_encoder = False
        self._active_slot_id = "0"

        self.encoder = build_encoder(feat_dim=feat_dim)
        self.state_net = build_state_net(state_dim=3, state_embed=state_embed)
        self.bspline_decoder = BSplineDecoder(num_control_points=num_control_points, horizon=horizon, degree=3)

        _, d2 = build_difference_matrices(num_control_points)
        self.register_buffer("d2_mat", torch.tensor(d2, dtype=torch.float32))

        self.in_dim = feat_dim + state_embed + (64 + 7 if use_spatial_pointer else 0)
        self.slots = nn.ModuleDict()

    def add_slot(self, task_id: int | str) -> None:
        key = str(task_id)
        if key not in self.slots:
            if self.slot_type == "wta":
                self.slots[key] = WTASlotHead(
                    self.in_dim,
                    num_cp=self.num_control_points,
                    horizon=self.horizon,
                    num_modes=self.num_modes,
                )
            else:
                self.slots[key] = CVAESlotHead(
                    self.in_dim,
                    num_cp=self.num_control_points,
                    horizon=self.horizon,
                    latent_dim=self.latent_dim,
                )

    def freeze_previous_slots(self, current_task_id: int | str, freeze_encoder: bool = True) -> None:
        cur_key = str(current_task_id)
        self._active_slot_id = cur_key
        self._freeze_encoder = freeze_encoder

        for k, slot in self.slots.items():
            if k != cur_key:
                slot.eval()
                for p in slot.parameters():
                    p.requires_grad = False

        if freeze_encoder:
            self.encoder.eval()
            self.state_net.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.state_net.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._freeze_encoder:
            self.encoder.eval()
            self.state_net.eval()
            for k, slot in self.slots.items():
                if k != self._active_slot_id:
                    slot.eval()
        return self

    def _encode_cond(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        v = self.encoder(obs)
        s = self.state_net(state)
        if self.use_spatial_pointer:
            rc = compute_res_cue(obs)
            sp = compute_spatial_pointer(obs, state)
            return torch.cat([v, s, rc, sp], dim=1)
        return torch.cat([v, s], dim=1)

    def loss(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        target_cp: torch.Tensor,
        target_traj: torch.Tensor,
        target_pen: torch.Tensor,
        task_id: int | str = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        key = str(task_id)
        if key not in self.slots:
            self.add_slot(key)
            self.slots[key].to(obs.device)

        cond = self._encode_cond(obs, state)
        head = self.slots[key]
        B = obs.shape[0]

        if isinstance(head, WTASlotHead):
            cps, pens, mlogits = head(cond)
            K = head.num_modes
            trajs = self.bspline_decoder(cps.view(B * K, self.num_control_points, 2)).view(
                B, K, self.horizon, 2
            )
            l_tr = torch.abs(trajs - target_traj[:, None, :, :2]).mean(dim=(2, 3))
            l_cp = torch.abs(cps - target_cp[:, None, :, :]).mean(dim=(2, 3))

            l_pn = balanced_pen_transition_loss(
                pens,
                target_pen[:, None, :].expand(-1, K, -1),
                state[:, 2:3].expand(-1, K),
                per_sample=True,
            )

            if len(self.d2_mat) > 0:
                d2c = torch.matmul(self.d2_mat, cps.view(B * K, self.num_control_points, 2)).view(B, K, -1, 2)
                l_sm = (d2c ** 2).sum(dim=-1).mean(dim=-1)
            else:
                l_sm = torch.zeros_like(l_tr)

            tot = l_tr + 0.5 * l_cp + self.smoothness_weight * l_sm + self.pen_loss_weight * l_pn
            best_idx = tot.argmin(dim=1)
            loss_win = tot[torch.arange(B, device=obs.device), best_idx].mean()
            loss_all = tot.mean()
            loss_gate = F.cross_entropy(mlogits, best_idx)
            total_loss = 0.95 * loss_win + 0.05 * loss_all + 0.1 * loss_gate
            metrics = {
                "loss_win": loss_win.detach(),
                "loss_gate": loss_gate.detach(),
                "total_loss": total_loss.detach(),
            }
            return total_loss, metrics

        pred_cp, pen_logits, (mu_po, lv_po), (mu_pr, lv_pr) = head(cond, target_cp=target_cp)
        pred_traj = self.bspline_decoder(pred_cp)
        loss_cp = F.l1_loss(pred_cp, target_cp)
        loss_traj = F.l1_loss(pred_traj, target_traj[..., :2])
        kl_div = torch.mean(
            0.5
            * torch.sum(
                lv_pr - lv_po + (torch.exp(lv_po) + (mu_po - mu_pr) ** 2) / torch.exp(lv_pr) - 1.0,
                dim=1,
            )
        )
        if len(self.d2_mat) > 0:
            diff2 = torch.matmul(self.d2_mat, pred_cp)
            loss_smooth = torch.mean(torch.sum(diff2 ** 2, dim=-1))
        else:
            loss_smooth = torch.tensor(0.0, device=obs.device)
        loss_pen = balanced_pen_transition_loss(pen_logits, target_pen, state[:, 2])
        total_loss = (
            loss_traj
            + 0.5 * loss_cp
            + self.smoothness_weight * loss_smooth
            + self.pen_loss_weight * loss_pen
            + self.kl_weight * kl_div
        )
        metrics = {
            "loss_traj": loss_traj.detach(),
            "loss_cp": loss_cp.detach(),
            "loss_kl": kl_div.detach(),
            "loss_pen": loss_pen.detach(),
            "total_loss": total_loss.detach(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def predict(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        task_id: int | str = 0,
        mode: str = "argmax",
        temp: float = 0.4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = str(task_id)
        cond = self._encode_cond(obs, state)
        head = self.slots[key]
        if isinstance(head, WTASlotHead):
            cps, pens, mlogits = head(cond)
            if mode == "sample":
                t = max(1e-3, temp if temp > 0 else 0.8)
                idx = torch.multinomial(F.softmax(mlogits / t, dim=-1), 1).squeeze(1)
            else:
                idx = mlogits.argmax(dim=1)
            b = torch.arange(obs.shape[0], device=obs.device)
            pred_cp = cps[b, idx]
            pen_logits = pens[b, idx]
        else:
            pred_cp, pen_logits, _, _ = head(cond, target_cp=None, temp=temp)
        pred_traj = self.bspline_decoder(pred_cp)
        return pred_cp, pred_traj, torch.sigmoid(pen_logits)


# ---------------------------------------------------------------------------
# Baseline: Microscopic (x, y, pen) GRU Autoregressive Sequence Policy
# ---------------------------------------------------------------------------


class GRUSequencePolicy(nn.Module):
    """[Baseline] B-Spline 제어점 대신 원본 (x, y, pen) 시계열을 자기회귀 예측하는 GRU 정책.

    동일한 4계층 ConvNet 백본과 상태 임베딩을 공유하고, GRUCell로 horizon 스텝의
    (x, y, pen)을 순차 디코딩한다. 제어점 청킹 대비 오차 누적 및 추론 지연 비교용.
    """

    def __init__(
        self,
        horizon: int = CHUNK_SIZE,
        feat_dim: int = 256,
        state_embed: int = 64,
        hidden_dim: int = 256,
        pen_loss_weight: float = 0.3,
        use_spatial_pointer: bool = False,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.pen_loss_weight = pen_loss_weight
        self.use_spatial_pointer = use_spatial_pointer

        self.encoder = build_encoder(feat_dim=feat_dim)
        self.state_net = build_state_net(state_dim=3, state_embed=state_embed)
        in_dim = feat_dim + state_embed + (64 + 7 if use_spatial_pointer else 0)
        self.init_hidden = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
        )
        self.step_in = nn.Linear(3, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.step_out = nn.Linear(hidden_dim, 3)

    def _encode_cond(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        v = self.encoder(obs)
        s = self.state_net(state)
        if self.use_spatial_pointer:
            rc = compute_res_cue(obs)
            sp = compute_spatial_pointer(obs, state)
            return torch.cat([v, s, rc, sp], dim=1)
        return torch.cat([v, s], dim=1)

    def forward(
        self, obs: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, 2, 128, 128), state: (B, 3) -> pred_traj (B, H, 2), pen_logits (B, H)."""
        cond = self._encode_cond(obs, state)
        h = self.init_hidden(cond)
        cur = state
        trajs, pens = [], []
        for _ in range(self.horizon):
            x = F.relu(self.step_in(cur))
            h = self.gru_cell(x, h)
            raw = self.step_out(h)
            xy = torch.sigmoid(raw[:, :2])
            pen_logit = raw[:, 2]
            trajs.append(xy)
            pens.append(pen_logit)
            cur = torch.cat([xy, torch.sigmoid(pen_logit).unsqueeze(1)], dim=1)
        return torch.stack(trajs, dim=1), torch.stack(pens, dim=1)

    def loss(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        target_cp: Optional[torch.Tensor] = None,
        target_traj: Optional[torch.Tensor] = None,
        target_pen: Optional[torch.Tensor] = None,
        task_id: int | str = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del target_cp, task_id
        pred_traj, pen_logits = self.forward(obs, state)
        loss_traj = F.l1_loss(pred_traj, target_traj[..., :2])
        loss_pen = balanced_pen_transition_loss(pen_logits, target_pen, state[:, 2])
        total_loss = loss_traj + self.pen_loss_weight * loss_pen
        metrics = {
            "loss_traj": loss_traj.detach(),
            "loss_pen": loss_pen.detach(),
            "total_loss": total_loss.detach(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def predict(
        self, obs: torch.Tensor, state: torch.Tensor, mode: str = "best", temp: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del mode, temp
        pred_traj, pen_logits = self.forward(obs, state)
        # rollout_bspline 호환을 위해 (B, 6, 2) 균등 샘플 제어점 프록시 함께 반환
        idx = torch.linspace(0, self.horizon - 1, steps=6, device=obs.device).long()
        pred_cp_proxy = pred_traj[:, idx]
        return pred_cp_proxy, pred_traj, torch.sigmoid(pen_logits)


# ---------------------------------------------------------------------------
# Model D: Task-ID-Free Autonomous Slot Routing Policy (AutonomousABKPolicy)
# ---------------------------------------------------------------------------


class ResidualWTASlotHead(nn.Module):
    """[Model D] K-가설 WTA 슬롯 헤드 (Slot 0: 기저 좌표, Slot s>0: Near-Zero 잔차 ΔP)."""

    def __init__(
        self,
        in_dim: int,
        num_cp: int = 6,
        horizon: int = CHUNK_SIZE,
        num_modes: int = 4,
        is_residual: bool = False,
        zero_init_std: float = 1e-3,
        pointer_dim: int = 7,
    ):
        super().__init__()
        self.num_cp = num_cp
        self.horizon = horizon
        self.num_modes = num_modes
        self.is_residual = is_residual
        self.pointer_dim = pointer_dim

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.LayerNorm(384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )
        self.pts_head = nn.Linear(256, num_modes * num_cp * 2)
        self.pen_head = nn.Linear(256, num_modes * horizon)
        # 잔여 공간 단서(pointer 7차원)를 sel_head에 직결 결합하여 외곽선 완료 후 모드 전환 신속 인지
        self.sel_head = nn.Sequential(
            nn.Linear(256 + pointer_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_modes),
        )
        self.prior_weight: Optional[nn.Parameter]

        if is_residual:
            self.prior_weight = nn.Parameter(torch.tensor(1.0))
            nn.init.normal_(self.pts_head.weight, mean=0.0, std=zero_init_std)
            nn.init.zeros_(self.pts_head.bias)
            nn.init.normal_(self.pen_head.weight, mean=0.0, std=zero_init_std)
            nn.init.zeros_(self.pen_head.bias)
        else:
            self.register_parameter("prior_weight", None)

    def compose_residual(
        self,
        base_cp: torch.Tensor,
        base_pen: torch.Tensor,
        delta_cp: torch.Tensor,
        delta_pen: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Logit-공간 잔차 합성: P_s = sigmoid(alpha_s * logit(clamp(P_base, 0.05, 0.95)) + dP_s).
        - base_cp : (B, M, 2) 또는 (B, 1, M, 2) 직전까지 누적된 기저 제어점 예측
        - delta_cp: (B, K, M, 2) 신규 슬롯 s가 예측한 K개 후보 모드 잔차 로짓
        - 학습 극초반 zero-init으로 dP_s ≈ 0, alpha_s ≈ 1 이므로 기저 정책을 훼손하지 않고 안정 초기화.
        """
        if self.prior_weight is None:
            return delta_cp, delta_pen
        # 확률 공간 경계(0, 1)에서 logit 기울기 발산 방지 및 0~1 외삽 차단용 클램핑
        base_logit_cp = torch.logit(base_cp.clamp(0.05, 0.95))
        base_pen_safe = base_pen.clamp(-2.5, 2.5)
        if delta_cp.dim() == 4 and base_logit_cp.dim() == 3:
            # 멀티모드 훈련 시 (B, M, 2) 기저를 (B, 1, M, 2)로 브로드캐스트
            base_logit_cp = base_logit_cp.unsqueeze(1)
            base_pen_safe = base_pen_safe.unsqueeze(1)
        cps = torch.sigmoid(self.prior_weight * base_logit_cp + delta_cp)
        pens = self.prior_weight * base_pen_safe + delta_pen
        return cps, pens

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = cond.shape[0]
        h = self.trunk(cond)
        pts = self.pts_head(h).view(B, self.num_modes, self.num_cp, 2)
        if not self.is_residual:
            pts = torch.sigmoid(pts)
        pens = self.pen_head(h).view(B, self.num_modes, self.horizon)
        pointer = cond[:, -self.pointer_dim:]
        sel_in = torch.cat([h, pointer], dim=-1)
        sel = self.sel_head(sel_in)
        return pts, pens, sel


class AutonomousABKPolicy(nn.Module):
    """[Model D] Task-ID 없는 자율 프로토타입 라우팅 + 잔차 슬롯 확장 ABK-VLA 정책.

    핵심 메커니즘:
      1. Hybrid Conv-Spatial 프로토타입 키 (slot_keys): encoder Conv4 특징 + 다중스케일 목표 실루엣
      2. Task-ID-Free Softmax 라우팅 (T=0.05): 기학습 도안 w_s >= 0.99, 미학습 도안은 기저 혼합
      3. Novelty Detection 기반 동적 확장: 기존 백본·슬롯 영구 동결 + ResidualWTASlotHead 추가
      4. Select-then-Blend: 각 슬롯 내 sel_head로 최적 가설 k* 먼저 선택 후 슬롯 간 가중 합성
    """

    def __init__(
        self,
        num_control_points: int = 6,
        horizon: int = CHUNK_SIZE,
        feat_dim: int = 256,
        state_embed: int = 64,
        res_grid: int = 8,
        num_modes: int = 6,
        temperature: float = 0.05,
        smoothness_weight: float = 0.02,
        pen_loss_weight: float = 0.3,
        zero_init_std: float = 1e-3,
        routing_mode: str = "blend",
        key_conv_weight: float = 0.5,
        key_spatial_weight: float = 1.0,
    ):
        super().__init__()
        if routing_mode not in ("blend", "argmax"):
            raise ValueError(f"Unsupported routing_mode: {routing_mode}")
        self.num_control_points = num_control_points
        self.horizon = horizon
        self.feat_dim = feat_dim
        self.state_embed = state_embed
        self.res_dim = res_grid * res_grid
        self.res_grid = res_grid
        self.pointer_dim = 7
        self.cond_dim = feat_dim + state_embed + self.res_dim + self.pointer_dim
        self.num_modes = num_modes
        self.temperature = temperature
        self.smoothness_weight = smoothness_weight
        self.pen_loss_weight = pen_loss_weight
        self.zero_init_std = zero_init_std
        self.routing_mode = routing_mode
        self.key_conv_weight = key_conv_weight
        self.key_spatial_weight = key_spatial_weight

        self.encoder = build_encoder(feat_dim=feat_dim)
        # 물리 상태(px, py, pen) 전용 순수 상태망 (Task 0 이후 동결되어도 도안 무관 일관성 유지)
        self.state_net = build_state_net(state_dim=3, state_embed=state_embed)
        self.slots = nn.ModuleList()

        # Conv4 (4096-d) + 32x32 (1024-d) + 16x16 (256-d) = 5376-d prototype key
        self.key_dim = 256 * 4 * 4 + 32 * 32 + 16 * 16
        self.register_buffer("slot_keys", torch.empty(0, self.key_dim))

        self.bspline_decoder = BSplineDecoder(
            num_control_points=num_control_points,
            horizon=horizon,
            degree=min(3, num_control_points - 1),
        )
        _, d2_np = build_difference_matrices(num_control_points)
        self.register_buffer("d2_mat", torch.tensor(d2_np, dtype=torch.float32))

        self._freeze_backbone = False
        self._active_slot_idx = -1
        self._cached_target_img: Optional[torch.Tensor] = None
        self._cached_target_key: Optional[torch.Tensor] = None
        self._cached_w_all: Optional[torch.Tensor] = None

    def reset_cache(self) -> None:
        """에피소드 롤아웃 시작 전 정적 타깃 키 및 가중치 캐시 초기화."""
        self._cached_target_img = None
        self._cached_target_key = None
        self._cached_w_all = None

    def _compute_spatial_pointer(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """미완성 잔차 맵 공간 포인터 산출 (호환성 메서드)."""
        return compute_spatial_pointer(obs, state)

    def _conv_spatial_trunk(self, obs0: torch.Tensor) -> torch.Tensor:
        """Flatten/Linear 직전 Conv4 공간 특징 맵 추출 (encoder[:-3])."""
        x = obs0
        for layer in list(self.encoder.children())[:-3]:
            x = layer(x)
        return x

    @torch.no_grad()
    def _compute_target_key(self, target_imgs: torch.Tensor) -> torch.Tensor:
        """
        목표 도안에서 5,376차원 L2-정규화 Hybrid Visual Prototype Key 생성 -> (B, 5376).
        - conv_feat: (B, 4096) — Conv4 특징맵 (256 x 4 x 4)을 L2 정규화 (의미/텍스처 성분)
        - sp_feat  : (B, 1280) — 목표 도안 AvgPool(32x32 + 16x16 = 1024 + 256) L2 정규화 (공간 배치 성분)
        - 최종 결합: k = normalize([w_conv * conv_feat; w_sp * sp_feat]) (기본 0.5 : 1.0 -> 에너지 20% : 80%)
        """
        if target_imgs.dim() == 3:
            tgt = target_imgs.unsqueeze(1)
        elif target_imgs.dim() == 4 and target_imgs.shape[1] == 2:
            tgt = target_imgs[:, 1:2]
        else:
            tgt = target_imgs
        tgt = tgt.float()
        if tgt.max().item() > 1.5:
            tgt = tgt / 255.0

        was_training = self.encoder.training
        self.encoder.eval()
        # 캔버스 채널을 0으로 고정하고 목표 도안만 넣어 초기 관찰 obs0: (B, 2, 128, 128) 구성
        obs0 = torch.cat([torch.zeros_like(tgt), tgt], dim=1)
        conv_map = self._conv_spatial_trunk(obs0)  # (B, 256, 8, 8)
        conv_feat = F.normalize(
            F.adaptive_avg_pool2d(conv_map, (4, 4)).flatten(1), p=2, dim=-1
        )  # (B, 4096)
        p32 = F.avg_pool2d(tgt, kernel_size=4, stride=4)  # (B, 1, 32, 32)
        p32 = F.avg_pool2d(p32, kernel_size=3, stride=1, padding=1).flatten(1)  # (B, 1024)
        p16 = F.avg_pool2d(tgt, kernel_size=8, stride=8).flatten(1)  # (B, 256)
        sp_feat = F.normalize(torch.cat([p32, p16], dim=1), p=2, dim=-1)  # (B, 1280)
        if was_training and not self._freeze_backbone:
            self.encoder.train()
        return F.normalize(
            torch.cat([conv_feat * self.key_conv_weight, sp_feat * self.key_spatial_weight], dim=1),
            p=2,
            dim=-1,
        )

    @torch.no_grad()
    def refresh_slot_key(self, slot_idx: int, target_imgs: torch.Tensor) -> None:
        """인코더 동결 직후 지정 슬롯의 프로토타입 키를 최종 특징으로 갱신."""
        proto = F.normalize(
            self._compute_target_key(target_imgs.to(self.slot_keys.device)).mean(dim=0, keepdim=True),
            p=2,
            dim=-1,
        )
        self.slot_keys[slot_idx] = proto.squeeze(0)

    def freeze_backbone_and_prior_slots(self, up_to_exclusive: Optional[int] = None) -> None:
        """Task 0 완료 후 비전 인코더·상태망 및 기학습 슬롯을 영구 동결."""
        self._freeze_backbone = True
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.state_net.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.state_net.eval()

        limit = len(self.slots) if up_to_exclusive is None else up_to_exclusive
        for i in range(limit):
            for p in self.slots[i].parameters():
                p.requires_grad = False
            self.slots[i].eval()

    @torch.no_grad()
    def register_or_expand_slot(
        self,
        target_imgs: torch.Tensor,
        novelty_threshold: float = 0.85,
        force_expand: bool = False,
    ) -> Tuple[int, bool, float]:
        """타겟 배치의 프로토타입 유사도 기반 자율 신규 슬롯 확장 및 동결."""
        dev = self.slot_keys.device
        q_proto = F.normalize(
            self._compute_target_key(target_imgs.to(dev)).mean(dim=0, keepdim=True), p=2, dim=-1
        )
        if len(self.slots) == 0:
            head = ResidualWTASlotHead(
                in_dim=self.cond_dim,
                num_cp=self.num_control_points,
                horizon=self.horizon,
                num_modes=self.num_modes,
                is_residual=False,
            ).to(dev)
            self.slots.append(head)
            self.slot_keys = q_proto.clone()
            self._active_slot_idx = 0
            return 0, True, 0.0

        sims = torch.matmul(q_proto, self.slot_keys.t()).squeeze(0)
        max_sim = float(sims.max().item())
        best_idx = int(sims.argmax().item())

        if force_expand or max_sim < novelty_threshold:
            new_idx = len(self.slots)
            self.freeze_backbone_and_prior_slots(up_to_exclusive=new_idx)
            head = ResidualWTASlotHead(
                in_dim=self.cond_dim,
                num_cp=self.num_control_points,
                horizon=self.horizon,
                num_modes=self.num_modes,
                is_residual=True,
                zero_init_std=self.zero_init_std,
            ).to(dev)
            self.slots.append(head)
            self.slot_keys = torch.cat([self.slot_keys, q_proto], dim=0)
            self._active_slot_idx = new_idx
            return new_idx, True, max_sim

        self._active_slot_idx = best_idx
        return best_idx, False, max_sim

    def compute_routing_weights(
        self, target_imgs: torch.Tensor, max_slots: Optional[int] = None
    ) -> torch.Tensor:
        """Task-ID 없이 Softmax(q · K^T / T) 가중치 (B, S) 계산."""
        if len(self.slots) == 0:
            raise RuntimeError("등록된 슬롯이 없습니다. register_or_expand_slot()을 먼저 호출하세요.")
        keys = self.slot_keys if max_slots is None else self.slot_keys[:max_slots]
        q = self._compute_target_key(target_imgs.to(keys.device))
        logits = torch.matmul(q, keys.t()) / max(1e-6, self.temperature)
        return F.softmax(logits, dim=-1)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._freeze_backbone:
            self.encoder.eval()
            self.state_net.eval()
            for idx, slot in enumerate(self.slots):
                if idx != self._active_slot_idx:
                    slot.eval()
        return self

    def _encode_cond(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # 미완성 잔차 공간 단서(target - canvas) 및 공간 포인터(중심/최근접 변위) 직접 결합
        pointer = compute_spatial_pointer(obs, state)
        st_feat = self.state_net(state)
        res_cue = compute_res_cue(obs, self.res_grid)
        return torch.cat([self.encoder(obs), st_feat, res_cue, pointer], dim=1)

    @staticmethod
    def _select_slot_mode(
        pts: torch.Tensor,
        pens: torch.Tensor,
        sel_logits: torch.Tensor,
        mode: str = "best",
        temp: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = pts.shape[0]
        if mode == "sample":
            t = max(1e-3, temp if temp > 0 else 0.8)
            idx = torch.multinomial(F.softmax(sel_logits / t, dim=-1), 1).squeeze(1)
        else:
            idx = sel_logits.argmax(dim=1)
        b = torch.arange(B, device=pts.device)
        return pts[b, idx], pens[b, idx]

    def forward(
        self, obs: torch.Tensor, state: torch.Tensor, mode: str = "best", temp: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """task_id 없는 순전파: (pred_cp, pred_traj, pen_logits) 반환."""
        if len(self.slots) == 0:
            raise RuntimeError("등록된 슬롯이 없습니다. register_or_expand_slot()을 먼저 호출하세요.")
        cond = self._encode_cond(obs, state)
        pred_cp, pen_logits, _ = self._compute_all_slot_predictions(
            obs, cond, upto_slot=len(self.slots), mode=mode, temp=temp
        )
        pred_traj = self.bspline_decoder(pred_cp)
        return pred_cp, pred_traj, pen_logits

    def _compute_all_slot_predictions(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        upto_slot: int,
        mode: str = "best",
        temp: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """슬롯 0..upto_slot-1의 Select-then-Blend 결과 (weighted_cp, weighted_pen, weights) 반환."""
        q = self._compute_target_key(obs[:, 1:2])
        w_all = F.softmax(
            torch.matmul(q, self.slot_keys[:upto_slot].t()) / max(1e-6, self.temperature), dim=-1
        )

        slot_cps, slot_pens = [], []
        p0, pn0, s0 = self.slots[0](cond)
        cp0, pen0 = self._select_slot_mode(p0, pn0, s0, mode=mode, temp=temp)
        slot_cps.append(cp0)
        slot_pens.append(pen0)

        # 단일 슬롯 0 활성 시(예: apple 도안 추론) 즉시 반환
        if upto_slot == 1:
            return cp0, pen0, w_all

        # Select-then-Blend: 각 슬롯 s=1..S-1의 선택된 모드 출력을 직전 가중 기저 위에 잔차 누적
        for s in range(1, upto_slot):
            w_prior = F.softmax(
                torch.matmul(q, self.slot_keys[:s].t()) / max(1e-6, self.temperature), dim=-1
            )  # (B, s)
            base_cp = torch.sum(w_prior.unsqueeze(-1).unsqueeze(-1) * torch.stack(slot_cps, dim=1), dim=1)  # (B, M, 2)
            base_pen = torch.sum(w_prior.unsqueeze(-1) * torch.stack(slot_pens, dim=1), dim=1)  # (B, H)
            dp, dpn, ss = self.slots[s](cond)  # dp: (B, K, M, 2), dpn: (B, K, H), ss: (B, K)
            dcp, dpen = self._select_slot_mode(dp, dpn, ss, mode=mode, temp=temp)
            cp_s, pen_s = self.slots[s].compose_residual(base_cp, base_pen, dcp, dpen)
            slot_cps.append(cp_s)
            slot_pens.append(pen_s)

        # 전체 슬롯 출력에 대한 Softmax 라우팅 가중 평균: P = sum_s w_s * P_s
        final_cp = torch.sum(
            w_all.unsqueeze(-1).unsqueeze(-1) * torch.stack(slot_cps, dim=1), dim=1
        ).clamp(0.0, 1.0)  # (B, M, 2)
        final_pen = torch.sum(w_all.unsqueeze(-1) * torch.stack(slot_pens, dim=1), dim=1)  # (B, H)
        return final_cp, final_pen, w_all

    def loss(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        target_cp: torch.Tensor,
        target_traj: torch.Tensor,
        target_pen: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """task_id 없이 활성 슬롯 s (자동 라우팅/최신 확장 슬롯)를 WTA 잔차 학습."""
        if len(self.slots) == 0:
            self.register_or_expand_slot(obs[:, 1:2])
        s_idx = self._active_slot_idx if self._active_slot_idx >= 0 else len(self.slots) - 1
        B = obs.shape[0]
        cond = self._encode_cond(obs, state)
        head = self.slots[s_idx]
        K = head.num_modes

        if s_idx == 0:
            cand_cps, cand_pens, mlogits = head(cond)
        else:
            with torch.no_grad():
                prior_cp, prior_pen, _ = self._compute_all_slot_predictions(
                    obs, cond.detach(), upto_slot=s_idx, mode="best", temp=0.0
                )
            delta_cps, delta_pens, mlogits = head(cond)
            cand_cps, cand_pens = head.compose_residual(prior_cp, prior_pen, delta_cps, delta_pens)

        trajs = self.bspline_decoder(cand_cps.view(B * K, self.num_control_points, 2)).view(
            B, K, self.horizon, 2
        )
        l_tr = torch.abs(trajs - target_traj[:, None, :, :2]).mean(dim=(2, 3))
        l_cp = torch.abs(cand_cps - target_cp[:, None, :, :]).mean(dim=(2, 3))
        l_pn = balanced_pen_transition_loss(
            cand_pens,
            target_pen[:, None, :].expand(-1, K, -1),
            state[:, 2:3].expand(-1, K),
            per_sample=True,
        )
        if len(self.d2_mat) > 0:
            d2c = torch.matmul(self.d2_mat, cand_cps.view(B * K, self.num_control_points, 2)).view(
                B, K, -1, 2
            )
            l_sm = (d2c ** 2).sum(dim=-1).mean(dim=-1)
        else:
            l_sm = torch.zeros_like(l_tr)

        tot = l_tr + 0.5 * l_cp + self.smoothness_weight * l_sm + self.pen_loss_weight * l_pn
        best_idx = tot.argmin(dim=1)
        b_idx = torch.arange(B, device=obs.device)
        loss_win = tot[b_idx, best_idx].mean()
        loss_all = tot.mean()

        # 1. 공중 비행(pen==0) 및 펜 전환 프레임 감지 + 6.0배 역빈도 Focal Loss
        pen_mean = target_pen.mean(dim=1)
        is_flight = (pen_mean < 0.5)
        pen_diff = torch.abs(target_pen[:, 1:] - target_pen[:, :-1]).max(dim=1).values
        pen_trans_init = torch.abs(target_pen[:, 0] - state[:, 2])
        is_transition = (pen_diff > 0.5) | (pen_trans_init > 0.5)

        gate_sample_weight = torch.where(
            is_flight | is_transition,
            torch.full_like(pen_mean, 6.0),
            torch.ones_like(pen_mean),
        )
        gate_sample_weight = gate_sample_weight / (gate_sample_weight.mean() + 1e-6)

        log_p = F.log_softmax(mlogits, dim=-1)
        p = torch.exp(log_p)
        pt = p[b_idx, best_idx].clamp(min=1e-6, max=1.0)
        log_pt = log_p[b_idx, best_idx]
        focal_weight = (1.0 - pt) ** 2.0
        loss_gate = (gate_sample_weight * focal_weight * (-log_pt)).mean()

        # 2. 펜-일관성 선택 손실 (Pen-Consistent Gating Supervision)
        # Softmax 확률 분포로 가중합된 기대 펜 상태와 target_pen 간의 직접 교차 엔트로피
        # cand_pen_probs를 detach하여 각 모드의 내부 표현을 훼손하지 않고 오직 sel_head의 확률만 감독!
        gate_probs = F.softmax(mlogits, dim=-1)  # (B, K)
        cand_pen_probs = torch.sigmoid(cand_pens).detach()  # (B, K, H)
        expected_pen = torch.sum(gate_probs.unsqueeze(-1) * cand_pen_probs, dim=1).clamp(1e-6, 1.0 - 1e-6)
        loss_pen_exp = F.binary_cross_entropy(expected_pen, target_pen)

        total_loss = 0.90 * loss_win + 0.05 * loss_all + 0.10 * loss_gate + 0.10 * loss_pen_exp
        metrics = {
            "loss_win": loss_win.detach(),
            "loss_gate": loss_gate.detach(),
            "loss_pen_exp": loss_pen_exp.detach(),
            "total_loss": total_loss.detach(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def predict(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        mode: str = "best",
        temp: float = 0.0,
        routing_mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """task_id 없이 자율 라우팅 추론 (Select-then-Blend 또는 Hard Argmax, 정적 타깃 키 캐시 가속)."""
        rmode = self.routing_mode if routing_mode is None else routing_mode
        if rmode not in ("blend", "argmax"):
            raise ValueError(f"Unsupported routing_mode: {rmode}")

        tgt = obs[:, 1:2]  # (B, 1, 128, 128) 목표 도안 채널만 추출
        upto_slot = len(self.slots)
        # 에피소드 내 정적 목표 이미지가 동일하면 키 추출/유사도 연산 1회만 수행 (캐시 히트)
        if self._cached_target_img is not None and torch.equal(tgt, self._cached_target_img):
            q = self._cached_target_key
            w_all = self._cached_w_all
        else:
            q = self._compute_target_key(tgt)  # (B, 5376)
            # Softmax 라우팅 가중치: w_all = softmax(<q, K_0..S-1> / T) -> (B, S)
            w_all = F.softmax(
                torch.matmul(q, self.slot_keys[:upto_slot].t()) / max(1e-6, self.temperature), dim=-1
            )
            self._cached_target_img = tgt.clone()
            self._cached_target_key = q
            self._cached_w_all = w_all

        cond = self._encode_cond(obs, state)  # (B, cond_dim = 256+64+15)
        if rmode == "argmax":
            # Hard Argmax 라우팅: 최고 유사도 슬롯 j* 까지만 잔차 prefix 계산 후 단독 선택 (off-diagonal 혼입 0%)
            top_slot = w_all.argmax(dim=-1)
            active_limit = int(top_slot.max().item()) + 1
        else:
            # Select-then-Blend 라우팅: 전체 슬롯 0..S-1 가중합
            top_slot = None
            active_limit = upto_slot

        slot_cps, slot_pens = [], []
        p0, pn0, s0 = self.slots[0](cond)
        cp0, pen0 = self._select_slot_mode(p0, pn0, s0, mode=mode, temp=temp)
        slot_cps.append(cp0)
        slot_pens.append(pen0)

        if active_limit > 1:
            for s in range(1, active_limit):
                w_prior = F.softmax(
                    torch.matmul(q, self.slot_keys[:s].t()) / max(1e-6, self.temperature), dim=-1
                )
                base_cp = torch.sum(w_prior.unsqueeze(-1).unsqueeze(-1) * torch.stack(slot_cps, dim=1), dim=1)
                base_pen = torch.sum(w_prior.unsqueeze(-1) * torch.stack(slot_pens, dim=1), dim=1)
                dp, dpn, ss = self.slots[s](cond)
                dcp, dpen = self._select_slot_mode(dp, dpn, ss, mode=mode, temp=temp)
                cp_s, pen_s = self.slots[s].compose_residual(base_cp, base_pen, dcp, dpen)
                slot_cps.append(cp_s)
                slot_pens.append(pen_s)

        stacked_cp = torch.stack(slot_cps, dim=1)
        stacked_pen = torch.stack(slot_pens, dim=1)
        if rmode == "argmax":
            b_idx = torch.arange(obs.shape[0], device=obs.device)
            final_cp = stacked_cp[b_idx, top_slot].clamp(0.0, 1.0)
            final_pen = stacked_pen[b_idx, top_slot]
        else:
            final_cp = torch.sum(
                w_all.unsqueeze(-1).unsqueeze(-1) * stacked_cp, dim=1
            ).clamp(0.0, 1.0)
            final_pen = torch.sum(w_all.unsqueeze(-1) * stacked_pen, dim=1)
        pred_traj = self.bspline_decoder(final_cp)
        return final_cp, pred_traj, torch.sigmoid(final_pen)



