"""tools/bspline_augment.py

[Model C 핵심 모듈] B-Spline 제어점 공간 기하 증강(Geometric Augmentation) 및 균형 펜 전환 손실.
- No DiffVG, No 미분 렌더링: 순수 해석적 B-Spline 선형 기저(B * P)의 어파인 동변성 활용.
- 주요 증강 기법:
  1. Synchronized Affine Equivariance: T(B @ P) == B @ T(P).
     이미지(canvas, target)와 펜 상태, B-Spline 제어점(M=6)을 기하학적으로 동일하게 변환하여
     공유 인코더와 슬롯 헤드가 캔버스 전역에서 위치/크기/회전 불변 궤적을 학습하도록 강제.
  2. Smooth C^2 Interior Control-Point Elastic Deformation:
     양 끝 경계 제어점(P0, P_{M-1}) 가중치는 0.25배로 억제하고 내부 제어점(P1..P_{M-2})에
     미세 탄성 섭동을 부여한 뒤 B @ P로 재구성하여 물리적 저크(Jerk=0) 없는 궤적 변형 생성.
  3. Covariate-Shift Pen State Recovery (Analytical B-Spline DAgger):
     롤아웃 시 발생하는 ±1.5~3px 펜 이탈 오차를 state에만 반영하고 목표 제어점은 도안 위를 유지하여,
     고정점 정체(Fixed-Point Stall)를 방지하고 자율 복귀 벡터장을 학습.
  4. Canvas-Only Incremental Raster Jitter:
     ch0(누적 캔버스)에만 미세 픽셀 시프트를 적용해 사람 시연 vs 롤아웃 캔버스 간 래스터 차이 흡수.
- balanced_pen_transition_loss:
  기존 pen_up_weight=2.0 편향으로 인해 다음 획 시작점(예: sprout 지면선) 상공에서 펜이 내려오지
  못하던 정체 현상을 해소하기 위해 기본 가중치를 1.0으로 균형화하고 획 전환/착지 구간을 집중 학습.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dtra.core.bspline_core import BSplineDecoder


def balanced_pen_transition_loss(
    pen_logits: torch.Tensor,
    target_pen: torch.Tensor,
    state_pen: torch.Tensor,
    transition_boost: float = 2.5,
    landing_boost: float = 2.5,
    per_sample: bool = False,
) -> torch.Tensor:
    """펜 업/다운 전환 및 착지 구간(up -> down)을 집중 감독하는 균형 BCE 손실.

    Args:
        pen_logits: (..., H) 예측 펜 로짓
        target_pen: (..., H) 정답 펜 상태 [0, 1]
        state_pen: (...,) 현재 스텝 직전 펜 상태
        per_sample: True이면 마지막 시간축(H)만 평균하여 샘플별 손실 텐서 반환
    """
    bce = F.binary_cross_entropy_with_logits(pen_logits, target_pen, reduction="none")
    has_transition = (
        (target_pen.min(dim=-1).values < 0.5) & (target_pen.max(dim=-1).values > 0.5)
    ).float().unsqueeze(-1)
    is_landing = (
        (state_pen < 0.5) & (target_pen.max(dim=-1).values > 0.5)
    ).float().unsqueeze(-1)

    # 스텝 단위 시간축 전환(pen[t] != pen[t-1]) 집중 가중치
    dt = torch.zeros_like(target_pen)
    dt[..., 1:] = torch.abs(target_pen[..., 1:] - target_pen[..., :-1])
    if state_pen.shape == target_pen[..., 0].shape:
        dt[..., 0] = torch.abs(target_pen[..., 0] - (state_pen > 0.5).float())
    step_trans = (dt > 0.5).float()

    weight = 1.0 + transition_boost * has_transition + landing_boost * is_landing + 2.0 * step_trans
    weighted = bce * weight
    return weighted.mean(dim=-1) if per_sample else weighted.mean()


class BSplineGeometricAugmentor(nn.Module):
    """제어점 공간 동변 기하 증강기 (CPU/GPU/XPU 배치 텐서 단위 고속 처리)."""

    def __init__(
        self,
        num_control_points: int = 6,
        horizon: int = 16,
        degree: int = 3,
        p_affine: float = 0.5,
        rot_deg: float = 4.0,
        shift_lim: float = 0.04,
        scale_lim: float = 0.06,
        p_elastic: float = 0.65,
        elastic_std: float = 0.004,
        p_cov: float = 0.70,
        cov_std: float = 0.015,
        p_canvas_jitter: float = 0.5,
        canvas_shift_px: float = 1.5,
        p_flip: float = 0.0,
    ):
        super().__init__()
        self.num_control_points = num_control_points
        self.horizon = horizon
        self.p_affine = p_affine
        self.rot_deg = rot_deg
        self.shift_lim = shift_lim
        self.scale_lim = scale_lim
        self.p_elastic = p_elastic
        self.elastic_std = elastic_std
        self.p_cov = p_cov
        self.cov_std = cov_std
        self.p_canvas_jitter = p_canvas_jitter
        self.canvas_shift_px = canvas_shift_px
        self.p_flip = p_flip

        self.bspline_decoder = BSplineDecoder(
            num_control_points=num_control_points,
            horizon=horizon,
            degree=degree,
        )

    @torch.no_grad()
    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        cp: torch.Tensor,
        raw_action: torch.Tensor,
        pen: torch.Tensor,
        p_affine: Optional[float] = None,
        rot_deg: Optional[float] = None,
        shift_lim: Optional[float] = None,
        scale_lim: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """배치 단위로 기하 일관성을 유지하며 관측·상태·제어점·궤적을 동시 증강."""
        B, _, H_img, _ = obs.shape
        device = obs.device

        p_aff = self.p_affine if p_affine is None else p_affine
        r_deg = self.rot_deg if rot_deg is None else rot_deg
        sh_lim = self.shift_lim if shift_lim is None else shift_lim
        sc_lim = self.scale_lim if scale_lim is None else scale_lim

        obs_aug = obs.clone()
        state_aug = state.clone()
        cp_aug = cp.clone()
        act_aug = raw_action.clone()

        # 0. Optional mirror flip (주의: 비대칭 시연 방향 혼선이 우려될 경우 기본 0.0 유지)
        if self.p_flip > 0.0:
            flip_mask = torch.rand(B, device=device) < self.p_flip
            if flip_mask.any():
                obs_aug[flip_mask] = torch.flip(obs_aug[flip_mask], dims=[-1])
                state_aug[flip_mask, 0] = 1.0 - state_aug[flip_mask, 0]
                cp_aug[flip_mask, :, 0] = 1.0 - cp_aug[flip_mask, :, 0]

        # 1. Affine Equivariance: T(B @ P) == B @ T(P)
        if p_aff > 0.0:
            m_aff = (torch.rand(B, device=device) < p_aff).float()
            ang = (torch.rand(B, device=device) * 2.0 - 1.0) * (r_deg * math.pi / 180.0) * m_aff
            sc = 1.0 + ((torch.rand(B, device=device) * 2.0 - 1.0) * sc_lim) * m_aff
            tx = ((torch.rand(B, device=device) * 2.0 - 1.0) * sh_lim) * m_aff
            ty = ((torch.rand(B, device=device) * 2.0 - 1.0) * sh_lim) * m_aff

            ca = sc * torch.cos(ang)
            sa = sc * torch.sin(ang)

            def warp_pts(pts: torch.Tensor) -> torch.Tensor:
                orig_shape = pts.shape
                q = pts.reshape(B, -1, 2) - 0.5
                qx = q[..., 0] * ca[:, None] - q[..., 1] * sa[:, None] + 0.5 + tx[:, None]
                qy = q[..., 0] * sa[:, None] + q[..., 1] * ca[:, None] + 0.5 + ty[:, None]
                return torch.stack([qx, qy], dim=-1).reshape(orig_shape).clamp(0.0, 1.0)

            state_aug[:, :2] = warp_pts(state_aug[:, :2])
            cp_aug = warp_pts(cp_aug)

            ic = (1.0 / sc) * torch.cos(-ang)
            isa = (1.0 / sc) * torch.sin(-ang)
            bx = -(ic * (2.0 * tx) - isa * (2.0 * ty))
            by = -(isa * (2.0 * tx) + ic * (2.0 * ty))
            theta = torch.zeros(B, 2, 3, device=device, dtype=obs_aug.dtype)
            theta[:, 0, 0] = ic
            theta[:, 0, 1] = -isa
            theta[:, 0, 2] = bx
            theta[:, 1, 0] = isa
            theta[:, 1, 1] = ic
            theta[:, 1, 2] = by
            grid = F.affine_grid(theta, obs_aug.size(), align_corners=False)
            obs_aug = F.grid_sample(obs_aug, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        # 2. Smooth C^2 Interior Control-Point Elastic Perturbation
        if self.p_elastic > 0.0 and self.elastic_std > 0.0:
            m_cp = (torch.rand(B, device=device) < self.p_elastic).float().view(B, 1, 1)
            cp_noise = torch.randn_like(cp_aug) * self.elastic_std * m_cp
            cp_noise[:, 0, :] *= 0.25
            cp_noise[:, -1, :] *= 0.25
            cp_aug = (cp_aug + cp_noise).clamp(0.0, 1.0)

        # Analytical B-Spline reconstruction guarantees exact C^2 smoothness (zero jerk)
        act_aug[..., :2] = self.bspline_decoder(cp_aug)

        # 3. Covariate-Shift State Recovery (on-contour target attraction)
        # 공중 점프(pen=0) 착지 정밀도 유지를 위해 펜-다운(pen=1) 구간 중심으로 섭동 적용
        if self.p_cov > 0.0 and self.cov_std > 0.0:
            m_cov = (torch.rand(B, device=device) < self.p_cov).float().view(B, 1)
            is_down = (state[:, 2:3] > 0.5).float()
            cov_scale = 0.1 + 0.9 * is_down
            st_delta = torch.randn(B, 2, device=device).clamp(-2.5, 2.5) * self.cov_std * m_cov * cov_scale
            state_aug[:, :2] = (state_aug[:, :2] + st_delta).clamp(0.0, 1.0)

        # 4. Canvas-only minor raster jitter (ch0 only)
        if self.p_canvas_jitter > 0.0 and self.canvas_shift_px > 0.0:
            m_cs = (torch.rand(B, device=device) < self.p_canvas_jitter).float()
            max_norm = (self.canvas_shift_px / float(H_img)) * 2.0
            cx = (torch.rand(B, device=device) * 2.0 - 1.0) * max_norm * m_cs
            cy = (torch.rand(B, device=device) * 2.0 - 1.0) * max_norm * m_cs
            th2 = torch.zeros(B, 2, 3, device=device, dtype=obs_aug.dtype)
            th2[:, 0, 0] = 1.0
            th2[:, 1, 1] = 1.0
            th2[:, 0, 2] = cx
            th2[:, 1, 2] = cy
            g2 = F.affine_grid(th2, (B, 1, H_img, H_img), align_corners=False)
            obs_aug[:, 0:1] = F.grid_sample(
                obs_aug[:, 0:1], g2, mode="bilinear", padding_mode="zeros", align_corners=False
            )

        return obs_aug, state_aug, cp_aug, act_aug, pen
