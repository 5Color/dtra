"""experiments/sim_dtra_tube_dynamics.py

[DTRA v2.1 Step 1] 자성 인력 및 탄성 복원력 기반 3D 입체 튜브 다중 링 토폴로지 아키텍처
프로토타입 구현, 6대 핵심 동역학 단위 검증 및 4-패널 클로즈업 3D 투영 시각화 스크립트.

v2.1 제1원칙 수식 동기화:
  1. 단위 초구(S^{D_topo-1}, D_topo=64) 및 구면 현 거리 d_chord(u,v) = ||u-v||_2 = sqrt(2(1-cos(u,v))) 통일
  2. 1차 타원 고리 기저 B_ring(theta) = [1, cos(theta), sin(theta)] 및 단일 닫힌 해 atan2 투영각 theta_i*
  3. 0 나누기 발산 위험이 없는 유계 탄성 순응도(Elastic Compliance): alpha_i = sigmoid(omega_i) in (0, 1)
  4. 통합 자성 인력 F_mag = cos_sim(x_t, k_i) + eta * o_{i, t-1}
                          + sum_{j!=i} M_ij * a_j(t-1) * max(0, cos_sim(x_t, k_i))  (명령 호환성 게이팅)
  5. 단위 초구 재정규화 비대칭 변형 z_tilde = normalize(z^(0) + Delta_z) 및 추론 후 snap_back(Delta_z <- 0)
  6. 제1원칙 2항 위상 정규화 손실 L_topo = lambda_r * sum(r_i) + lambda_rep * sum_{i<j} ReLU((r_i+r_j+delta)-d_chord)^2
  7. 비파괴 수초화 지름길(M_ij = eta_myelin * tanh(C_ij)) 및 모양 보존 가지 뻗기(Structural Outgrowth)

실행 방법:
  python experiments/sim_dtra_tube_dynamics.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from dtra.core.bspline_core import BSplineDecoder


def set_seed(seed: int = 42) -> None:
    """재현성을 위한 난수 시드 고정."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_savefig(fig: plt.Figure, path: Path, dpi: int = 200) -> Path:
    """Windows 파일 잠금 회피용 임시 저장 후 원자적 교체."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return path
        except OSError:
            time.sleep(0.2)
    alt = path.with_name(f"{path.stem}_{int(time.time())}.png")
    os.replace(tmp, alt)
    return alt


def _inv_softplus(y: float) -> float:
    """softplus(x) = y 를 만족하는 초기 파라미터 역산."""
    y_clamped = max(float(y), 1e-4)
    return math.log(math.expm1(y_clamped))


def _logit(p: float) -> float:
    """sigmoid(x) = p 를 만족하는 초기 순응도 파라미터 역산."""
    p_clamped = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p_clamped / (1.0 - p_clamped))


# ---------------------------------------------------------------------------
# 1. 초입체 튜브 링 계층 (HypertubeRingLayer, v2.1)
# ---------------------------------------------------------------------------


class HypertubeRingLayer(nn.Module):
    """D_topo=64 단위 초구 위 1차 타원 초입체 튜브(Hypertube) 계층 (DTRA v2.1).

    물리적 직관:
      - 각 튜브 R_i는 자성 키 k_i의 사영점 P_topo(k_i)를 중심으로 1차 주기 기저
        B_ring(theta) = [1, cos(theta), sin(theta)]가 감긴 닫힌 타원 고리에
        학습 가능 반경 r_i(두께)와 유계 순응도 alpha_i = sigmoid(omega_i)(말랑함)를 부여한 초입체 튜브다.
      - 쿼리 투영점 c(x_t)가 유입되면 1차 타원 고리의 정확한 닫힌 해 atan2로 최단 투영각 theta_i*를
        즉시 산출하고, 국소 푸리에 가중치 psi_i(theta)로 theta_i* 부근만 볼록하게 찌그러뜨린 뒤
        단위 초구 S^{D_topo-1} 위로 재정규화한다.
      - 에피소드 내 청크 연속성을 위한 관성(eta * o_{i,t-1})과 명령 호환성 게이팅된
        수초화 지름길 M_ij * a_j(t-1) * max(0, cos_sim(x_t, k_i))을 통합한다.
    """

    def __init__(
        self,
        num_tubes: int,
        d_key: int = 128,
        d_topo: int = 64,
        d_embed: int = 64,
        num_phase_samples: int = 64,
        init_radius: float = 0.35,
        init_alpha: float = 0.50,
        init_tau: float = 0.28,
        inertia_eta: float = 0.25,
        myelin_eta: float = 0.35,
        overlap_temp: float = 0.14,
        overlap_gate_gain: float = 4.0,
        train_tail_eps: float = 0.05,
        train_tail_slope: float = 4.0,
        safety_margin: float = 0.10,
        proj_topo: Optional[nn.Linear] = None,
    ) -> None:
        super().__init__()
        self.num_tubes = num_tubes
        self.d_key = d_key
        self.d_topo = d_topo
        self.d_embed = d_embed
        self.num_phase_samples = num_phase_samples
        self.eta = inertia_eta
        self.eta_myelin = myelin_eta
        self.overlap_temp = overlap_temp
        self.overlap_gate_gain = overlap_gate_gain
        self.train_tail_eps = train_tail_eps
        self.train_tail_slope = train_tail_slope
        self.safety_margin = safety_margin

        # 공유 선형 위상 사영 행렬 P_topo: (D_key) -> (D_topo)
        if proj_topo is not None:
            self.proj_topo = proj_topo
        else:
            self.proj_topo = nn.Linear(d_key, d_topo, bias=False)

        # 멀티모달 자성 키 k_i: shape (K, D_key)
        keys_init = torch.randn(num_tubes, d_key)
        if num_tubes <= d_key:
            q_mat, _ = torch.linalg.qr(keys_init.t())
            keys_init = q_mat.t()[:num_tubes].contiguous()
        else:
            keys_init = F.normalize(keys_init, p=2, dim=-1)
        self.keys = nn.Parameter(keys_init)

        # 1차 타원 고리 푸리에 계수 C_i: shape (K, 3, D_topo)
        # 3개 모드 순서: [1 (DC 중심 오프셋), cos(theta), sin(theta)]
        coeffs_init = torch.zeros(num_tubes, 3, d_topo)
        for k_idx in range(num_tubes):
            # 각 링마다 직교하는 2개 방향 (c_cos, c_sin)으로 닫힌 타원 평면 생성
            q_plane, _ = torch.linalg.qr(torch.randn(d_topo, 2))
            coeffs_init[k_idx, 1] = 0.24 * q_plane[:, 0]  # cos(theta)
            coeffs_init[k_idx, 2] = 0.24 * q_plane[:, 1]  # sin(theta)
        self.fourier_coeffs = nn.Parameter(coeffs_init)

        # 학습 가능한 반경 파라미터 rho_i: r_i = softplus(rho_i) > 0, shape (K,)
        self.rho = nn.Parameter(torch.full((num_tubes,), _inv_softplus(init_radius)))

        # 학습 가능한 유계 탄성 순응도 파라미터 omega_i: alpha_i = sigmoid(omega_i) in (0, 1), shape (K,)
        self.omega = nn.Parameter(torch.full((num_tubes,), _logit(init_alpha)))

        # 적응형 임계 게이트 바이어스 beta_i: tau_i = softplus(beta_i), shape (K,)
        self.beta = nn.Parameter(torch.full((num_tubes,), _inv_softplus(init_tau)))

        # 튜브별 의미/운동 표현 임베딩 e_i: shape (K, D_embed)
        self.ring_embed = nn.Parameter(torch.randn(num_tubes, d_embed) * 0.25)

        # 1차 주기 기저 행렬 B_ring(theta) = [1, cos(theta), sin(theta)]: shape (M_theta, 3)
        theta = torch.linspace(0.0, 2.0 * math.pi, num_phase_samples + 1)[:-1]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        basis_ring = torch.stack([torch.ones_like(theta), cos_t, sin_t], dim=-1)
        self.register_buffer("theta_grid", theta)
        self.register_buffer("cos_theta", cos_t)
        self.register_buffer("sin_theta", sin_t)
        self.register_buffer("basis_ring", basis_ring)

        # 에피소드 생애주기 버퍼: o_{i, t-1}, a_i(t-1), Delta_z_i(theta)
        self.register_buffer("inertia_state", torch.zeros(1, num_tubes))
        self.register_buffer("prev_gate_state", torch.zeros(1, num_tubes))
        self.register_buffer(
            "delta_z_state",
            torch.zeros(1, num_tubes, num_phase_samples, d_topo),
        )

        # 비파괴 수초화(Edge Myelination) & 가지 뻗기(Structural Outgrowth) 버퍼
        # - coactivation_trace C_ij: shape (K, K)
        # - myelin_matrix M_ij = eta_myelin * tanh(C_ij): shape (K, K)
        # - outgrowth_shift: 모양(fourier_coeffs), r_i, alpha_i를 건드리지 않고 중심만 재배치, shape (K, D_topo)
        self.register_buffer("coactivation_trace", torch.zeros(num_tubes, num_tubes))
        self.register_buffer("myelin_matrix", torch.zeros(num_tubes, num_tubes))
        self.register_buffer("outgrowth_shift", torch.zeros(num_tubes, d_topo))

    @property
    def radii(self) -> torch.Tensor:
        """학습 가능한 양수 튜브 반경 r_i = softplus(rho_i) + eps. shape: (K,)."""
        return F.softplus(self.rho) + 1e-3

    @property
    def compliance(self) -> torch.Tensor:
        """유계 탄성 순응도(말랑함 계수) alpha_i = sigmoid(omega_i) in (0, 1). shape: (K,)."""
        return torch.sigmoid(self.omega)

    @property
    def gate_thresholds(self) -> torch.Tensor:
        """수동 임계값 없는 적응형 게이트 바이어스 softplus(beta_i). shape: (K,)."""
        return F.softplus(self.beta)

    def reset(self) -> None:
        """에피소드 시작/종료 시 관성(o_{i,0}=0), 활성 기록(a_i(0)=0), 변위(Delta_z_i=0) 동시 초기화."""
        self.inertia_state.zero_()
        self.prev_gate_state.zero_()
        self.delta_z_state.zero_()

    def reset_myelination(self) -> None:
        """수초화 흔적 및 가지 뻗기 변위 초기화."""
        self.coactivation_trace.zero_()
        self.myelin_matrix.zero_()
        self.outgrowth_shift.zero_()

    def snap_back(self) -> None:
        """추론 직후 임계 감쇠 탄성 복원: 변위 Delta_z_i <- 0 복귀."""
        self.delta_z_state.zero_()

    def compute_base_rings(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """좌표 동기화된 기본 1차 고리 z_i^(0)(theta) in S^{D_topo-1} 산출.

        Returns:
            z0: shape (K_sel, M_theta, D_topo) — 단위 초구 위의 기본 닫힌 1차 타원 고리
        """
        if indices is None:
            k_sel = self.keys
            c_sel = self.fourier_coeffs
            out_sel = self.outgrowth_shift
        else:
            k_sel = self.keys[indices]
            c_sel = self.fourier_coeffs[indices]
            out_sel = self.outgrowth_shift[indices]

        k_hat = F.normalize(k_sel, p=2, dim=-1)
        center = self.proj_topo(k_hat) + out_sel  # (K_sel, D_topo)

        # 1차 주기 기저 결합: (M_theta, 3) x (K_sel, 3, D_topo) -> (K_sel, M_theta, D_topo)
        harmonic_ring = torch.einsum("mf,kfd->kmd", self.basis_ring, c_sel)
        u0 = center.unsqueeze(1) + harmonic_ring
        z0 = F.normalize(u0, p=2, dim=-1)  # 단위 초구 ||z||_2 = 1 정규화
        return z0

    def get_deformed_rings(self) -> torch.Tensor:
        """단위 초구 재정규화된 변형 궤도 z_tilde_i = normalize(z_i^(0) + Delta_z_i) 반환.

        Delta_z_i == 0 인 스냅백 상태에서는 부동소수점 반올림 오차 없이 z_i^(0)와 엄밀히 일치(0.0).
        """
        z0 = self.compute_base_rings().unsqueeze(0)  # (1, K, M_theta, D_topo)
        if torch.all(self.delta_z_state == 0.0):
            return z0.expand(self.delta_z_state.shape[0], -1, -1, -1)
        zero_mask = (self.delta_z_state.abs().sum(dim=-1, keepdim=True) == 0.0)
        z_norm = F.normalize(z0 + self.delta_z_state, p=2, dim=-1)
        return torch.where(zero_mask, z0.expand_as(z_norm), z_norm)

    def myelinate_and_grow_step(
        self,
        pair_i: int,
        pair_j: int,
        co_strength: float = 0.85,
        outgrowth_lr: float = 0.14,
    ) -> Dict[str, float]:
        """동시 도킹 수초화(Edge Myelination) 및 모양 보존 가지 뻗기(Structural Outgrowth).

        - 두 튜브의 고유 모양(fourier_coeffs), 반경(r_i), 순응도(alpha_i)는 100% 동결 보존.
        - 동시 발화 흔적 C_ij를 누적하여 전도도 M_ij = eta_myelin * tanh(C_ij)를 강화.
        - 위상 중심 좌표(outgrowth_shift)만 척력 안전 마진(r_i + r_j + delta) 직전까지
          서서히 가까워지도록 재배치함.
        """
        with torch.no_grad():
            self.coactivation_trace[pair_i, pair_j] += co_strength
            self.coactivation_trace[pair_j, pair_i] += co_strength
            self.myelin_matrix.copy_(self.eta_myelin * torch.tanh(self.coactivation_trace))
            self.myelin_matrix.fill_diagonal_(0.0)

            z0_all = self.compute_base_rings()  # (K, M_theta, D_topo)
            r_all = self.radii  # (K,)
            d_ij_before = float(torch.cdist(z0_all[pair_i], z0_all[pair_j]).min().item())
            safe_floor = float((r_all[pair_i] + r_all[pair_j] + self.safety_margin).item())

            if d_ij_before > safe_floor + 0.015:
                # 링 중심 벡터를 안전 마진 직전까지만 이동시키는 비례 스텝
                c_i = z0_all[pair_i].mean(dim=0)
                c_j = z0_all[pair_j].mean(dim=0)
                diff = c_j - c_i
                diff_norm = float(diff.norm().item())
                if diff_norm > 1e-6:
                    slack = d_ij_before - safe_floor
                    step_mag = min(outgrowth_lr * slack, 0.45 * slack) * 0.5
                    dir_vec = diff / diff_norm
                    self.outgrowth_shift[pair_i] += step_mag * dir_vec
                    self.outgrowth_shift[pair_j] -= step_mag * dir_vec

            z0_after = self.compute_base_rings()
            d_ij_after = float(torch.cdist(z0_after[pair_i], z0_after[pair_j]).min().item())

        return {
            "d_before": d_ij_before,
            "d_after": d_ij_after,
            "safe_floor": safe_floor,
            "m_ij": float(self.myelin_matrix[pair_i, pair_j].item()),
        }

    def forward(
        self,
        x_t: torch.Tensor,
        use_inertia: bool = True,
        use_myelination: bool = True,
        record_deformation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """통합 자성 인력, closed-form atan2 투영, 비대칭 변형 및 Soft Overlap 순전파."""
        bsz = x_t.shape[0]
        device = x_t.device
        dtype = x_t.dtype

        x_hat = F.normalize(x_t, p=2, dim=-1)  # (B, D_key)
        k_hat = F.normalize(self.keys, p=2, dim=-1)  # (K, D_key)
        cos_sim = torch.matmul(x_hat, k_hat.t())  # (B, K)

        # 1) 관성(eta * o_{i, t-1}) + 명령 호환성 게이팅 수초화 부스트
        if use_inertia:
            o_prev = (
                self.inertia_state
                if self.inertia_state.shape[0] == bsz
                else self.inertia_state.mean(dim=0, keepdim=True).expand(bsz, -1)
            )
            a_prev = (
                self.prev_gate_state
                if self.prev_gate_state.shape[0] == bsz
                else self.prev_gate_state.mean(dim=0, keepdim=True).expand(bsz, -1)
            )
        else:
            o_prev = torch.zeros_like(cos_sim)
            a_prev = torch.zeros_like(cos_sim)

        if use_myelination:
            # sum_{j != i} M_ij * a_j(t-1) * max(0, cos_sim(x_t, k_i))
            m_no_diag = self.myelin_matrix.clone()
            m_no_diag.fill_diagonal_(0.0)
            assoc_drive = torch.matmul(a_prev, m_no_diag.t())  # (B, K)
            compat_gate = F.relu(cos_sim)  # (B, K) — 무관/배타 명령 시 0.0으로 차단
            myelin_boost = assoc_drive * compat_gate  # (B, K)
        else:
            myelin_boost = torch.zeros_like(cos_sim)

        f_mag = cos_sim + self.eta * o_prev + myelin_boost  # (B, K)

        # 2) 적응형 게이트 a_i(t) = ReLU(F_mag - softplus(beta_i))
        tau = self.gate_thresholds  # (K,)
        margin = f_mag - tau.unsqueeze(0)  # (B, K)

        if self.training:
            a_full = F.relu(margin) + self.train_tail_eps * torch.sigmoid(
                self.train_tail_slope * margin
            )
            active_idx = torch.arange(self.num_tubes, device=device)
        else:
            a_full = F.relu(margin)
            active_idx = torch.nonzero((a_full > 0.0).any(dim=0), as_tuple=False).squeeze(-1)

        k_act = int(active_idx.numel())
        c_query = F.normalize(self.proj_topo(x_hat), p=2, dim=-1)  # (B, D_topo)

        if k_act == 0:
            zeros_k = torch.zeros(bsz, self.num_tubes, device=device, dtype=dtype)
            zeros_emb = torch.zeros(bsz, self.d_embed, device=device, dtype=dtype)
            zeros_topo = torch.zeros(bsz, self.d_topo, device=device, dtype=dtype)
            if use_inertia:
                self.inertia_state = zeros_k.detach()
                self.prev_gate_state = zeros_k.detach()
            if record_deformation:
                self.delta_z_state = torch.zeros(
                    bsz, self.num_tubes, self.num_phase_samples, self.d_topo,
                    device=device, dtype=dtype,
                )
            return {
                "overlap": zeros_k,
                "weights": zeros_k,
                "gate": a_full,
                "f_mag": f_mag,
                "cos_sim": cos_sim,
                "myelin_boost": myelin_boost,
                "h_avg": zeros_emb,
                "h_int": zeros_emb,
                "z_dock": zeros_topo,
                "c_query": c_query,
                "active_count": torch.tensor(0, device=device),
                "active_ratio": torch.tensor(0.0, device=device),
            }

        # 3) 활성 튜브(K_act)만 슬라이싱해 O(|A_t|) ≈ O(1) 연산 수행
        a_act = a_full[:, active_idx]  # (B, K_act)
        r_act = self.radii[active_idx]  # (K_act,)
        alpha_act = self.compliance[active_idx]  # (K_act,) in (0, 1)
        coeffs_act = self.fourier_coeffs[active_idx]  # (K_act, 3, D_topo)
        embed_act = self.ring_embed[active_idx]  # (K_act, D_embed)
        center_act = self.proj_topo(k_hat[active_idx]) + self.outgrowth_shift[active_idx]  # (K_act, D_topo)

        # 4) 1차 타원 고리 [1, cos, sin] 의 정확한 닫힌 해(Closed-form) atan2 투영각 theta_i*
        c_dc = coeffs_act[:, 0, :]   # (K_act, D_topo)
        c_cos = coeffs_act[:, 1, :]  # (K_act, D_topo)
        c_sin = coeffs_act[:, 2, :]  # (K_act, D_topo)

        p_cos = torch.einsum("bd,kd->bk", c_query, c_cos)  # (B, K_act)
        p_sin = torch.einsum("bd,kd->bk", c_query, c_sin)  # (B, K_act)
        theta_star = torch.atan2(p_sin, p_cos)  # (B, K_act) 닫힌 해 위상각
        cos_star = torch.cos(theta_star)  # (B, K_act)
        sin_star = torch.sin(theta_star)  # (B, K_act)

        u0_star = (
            (center_act + c_dc).unsqueeze(0)
            + cos_star.unsqueeze(-1) * c_cos.unsqueeze(0)
            + sin_star.unsqueeze(-1) * c_sin.unsqueeze(0)
        )  # (B, K_act, D_topo)
        z0_star = F.normalize(u0_star, p=2, dim=-1)  # (B, K_act, D_topo)

        # 5) 국소 푸리에 가중치 psi_i(theta) = ((1 + cos(theta - theta_i*)) / 2)^2
        # 및 탄성 순응도 alpha_i 기반 비대칭 변형('찌그러짐') + 단위 초구 재정규화
        cos_diff = (
            torch.einsum("m,bk->bkm", self.cos_theta, cos_star)
            + torch.einsum("m,bk->bkm", self.sin_theta, sin_star)
        )  # (B, K_act, M_theta)
        psi = (0.5 * (1.0 + cos_diff)) ** 2  # (B, K_act, M_theta)

        pull_vec = c_query.unsqueeze(1) - z0_star  # (B, K_act, D_topo)
        disp_scale = (alpha_act.unsqueeze(0) * a_act).unsqueeze(-1)  # (B, K_act, 1)
        delta_z_star = disp_scale * pull_vec  # (B, K_act, D_topo)
        z_tilde_star = F.normalize(z0_star + delta_z_star, p=2, dim=-1)  # 단위 초구 재정규화

        if record_deformation:
            delta_z_act = delta_z_star.unsqueeze(2) * psi.unsqueeze(-1)  # (B, K_act, M_theta, D_topo)
            delta_buf = torch.zeros(
                bsz, self.num_tubes, self.num_phase_samples, self.d_topo,
                device=device, dtype=dtype,
            )
            delta_buf[:, active_idx] = delta_z_act.detach()
            self.delta_z_state = delta_buf

        # 6) 구면 현 거리 d_chord(c(x_t), z_tilde_star) 기반 Soft Tube Overlap o_{i, t} in [0, 1]
        cos_contact = F.cosine_similarity(c_query.unsqueeze(1), z_tilde_star, dim=-1).clamp(-1.0, 1.0)
        d_chord = torch.sqrt(2.0 * (1.0 - cos_contact) + 1e-8)  # (B, K_act)
        gate_factor = torch.tanh(self.overlap_gate_gain * a_act)  # (B, K_act)
        overlap_act = gate_factor * torch.sigmoid(
            (r_act.unsqueeze(0) - d_chord) / self.overlap_temp
        )  # (B, K_act)

        overlap_full = torch.zeros(bsz, self.num_tubes, device=device, dtype=dtype)
        overlap_full[:, active_idx] = overlap_act

        if use_inertia:
            self.inertia_state = overlap_full.detach()
            self.prev_gate_state = a_full.detach()

        # 7) 볼록 결합 정규화 (합 <= 1) 및 집합 풀링 (가중 평균 + 2차 곱셈 간섭항)
        w_act = overlap_act / (overlap_act.sum(dim=-1, keepdim=True) + 1e-4)  # (B, K_act)
        weights_full = torch.zeros(bsz, self.num_tubes, device=device, dtype=dtype)
        weights_full[:, active_idx] = w_act

        h_avg = torch.einsum("bk,kd->bd", w_act, embed_act)  # (B, D_embed)
        we = w_act.unsqueeze(-1) * embed_act.unsqueeze(0)  # (B, K_act, D_embed)
        sum_we = we.sum(dim=1)
        sum_we_sq = (we ** 2).sum(dim=1)
        h_int = 0.5 * (sum_we ** 2 - sum_we_sq)  # (B, D_embed)
        z_dock = torch.einsum("bk,bkd->bd", w_act, z_tilde_star)  # (B, D_topo)

        return {
            "overlap": overlap_full,
            "weights": weights_full,
            "gate": a_full,
            "f_mag": f_mag,
            "cos_sim": cos_sim,
            "myelin_boost": myelin_boost,
            "h_avg": h_avg,
            "h_int": h_int,
            "z_dock": z_dock,
            "c_query": c_query,
            "active_count": torch.tensor(k_act, device=device),
            "active_ratio": torch.tensor(float(k_act) / float(self.num_tubes), device=device),
        }

    def topo_regularization_loss(
        self,
        lambda_r: float = 0.04,
        lambda_rep: float = 1.60,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """제1원칙 2항 위상 정규화 손실 L_topo (v2.1).

        수식:
          L_topo = lambda_r * sum_i r_i
                 + lambda_rep * sum_{i < j} ReLU((r_i + r_j + delta) - d_chord(R_i^(0), R_j^(0)))^2
        """
        k = self.num_tubes
        r = self.radii  # (K,)
        alpha = self.compliance  # (K,)

        # 1) 선형 체적 항: lambda_r * sum_i r_i (Lazy Expansion 방지)
        loss_vol = r.sum()

        # 2) 상삼각 구면 현 거리 척력 항: sum_{i < j} ReLU((r_i + r_j + delta) - d_chord)^2
        z0_all = self.compute_base_rings()  # (K, M_theta, D_topo)
        z0_flat = z0_all.reshape(k * self.num_phase_samples, self.d_topo)
        cos_mat = torch.matmul(z0_flat, z0_flat.t()).clamp(-1.0, 1.0)
        d_chord_all = torch.sqrt(2.0 * (1.0 - cos_mat) + 1e-8).reshape(
            k, self.num_phase_samples, k, self.num_phase_samples
        )
        d_ij = d_chord_all.amin(dim=(1, 3))  # (K, K)

        triu_mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=r.device), diagonal=1)
        req_sep = r.unsqueeze(1) + r.unsqueeze(0) + self.safety_margin
        rep_viol = F.relu(req_sep - d_ij)[triu_mask]
        loss_rep = (rep_viol ** 2).sum()

        surface_margin = (d_ij - (r.unsqueeze(1) + r.unsqueeze(0)))[triu_mask]
        min_surface_margin = float(surface_margin.min().item()) if surface_margin.numel() > 0 else 0.0
        min_center_dist = float(d_ij[triu_mask].min().item()) if surface_margin.numel() > 0 else 0.0

        total_l_topo = lambda_r * loss_vol + lambda_rep * loss_rep
        metrics = {
            "loss_topo": float(total_l_topo.item()),
            "loss_vol": float(loss_vol.item()),
            "loss_rep": float(loss_rep.item()),
            "mean_radius": float(r.mean().item()),
            "max_radius": float(r.max().item()),
            "mean_alpha": float(alpha.mean().item()),
            "min_surface_margin": min_surface_margin,
            "min_center_dist": min_center_dist,
        }
        return total_l_topo, metrics


# ---------------------------------------------------------------------------
# 2. 2단계 도킹 아키텍처 (TwoStageDTRA, v2.1)
# ---------------------------------------------------------------------------


class TwoStageDTRA(nn.Module):
    """2단계(Coarse -> Fine) 튜브 도킹 및 B-Spline 제어점 합성 아키텍처 (v2.1)."""

    def __init__(
        self,
        d_key: int = 128,
        d_topo: int = 64,
        k_coarse: int = 4,
        k_fine: int = 8,
        num_control_points: int = 6,
        horizon: int = 16,
        d_embed: int = 64,
        num_phase_samples: int = 64,
    ) -> None:
        super().__init__()
        self.d_key = d_key
        self.d_topo = d_topo
        self.k_coarse = k_coarse
        self.k_fine = k_fine
        self.num_control_points = num_control_points
        self.horizon = horizon

        self.proj_topo = nn.Linear(d_key, d_topo, bias=False)

        # 1단계: 두껍고 말랑한 범용/의미 튜브 (r_init=0.40, alpha_init=0.78)
        self.coarse_layer = HypertubeRingLayer(
            num_tubes=k_coarse,
            d_key=d_key,
            d_topo=d_topo,
            d_embed=d_embed,
            num_phase_samples=num_phase_samples,
            init_radius=0.40,
            init_alpha=0.78,
            init_tau=0.28,
            inertia_eta=0.25,
            safety_margin=0.08,
            proj_topo=self.proj_topo,
        )

        # 2단계: 얇고 단단한 정밀 운동 튜브 (r_init=0.18, alpha_init=0.24)
        self.fine_layer = HypertubeRingLayer(
            num_tubes=k_fine,
            d_key=d_key,
            d_topo=d_topo,
            d_embed=d_embed,
            num_phase_samples=num_phase_samples,
            init_radius=0.18,
            init_alpha=0.24,
            init_tau=0.30,
            inertia_eta=0.25,
            safety_margin=0.06,
            proj_topo=self.proj_topo,
        )

        self.coarse_anchor_templates = nn.Parameter(
            torch.zeros(k_coarse, num_control_points, 2)
        )
        feat_dim = d_embed * 2 + d_topo
        self.coarse_anchor_head = nn.Sequential(
            nn.Linear(feat_dim, 96),
            nn.SiLU(),
            nn.Linear(96, num_control_points * 2),
        )
        self.coarse_ctx_head = nn.Sequential(
            nn.Linear(feat_dim, d_key),
            nn.Tanh(),
        )
        self.dock_norm = nn.LayerNorm(d_key)

        self.fine_residual_head = nn.Sequential(
            nn.Linear(feat_dim, 96),
            nn.SiLU(),
            nn.Linear(96, num_control_points * 2),
        )
        self.fine_pen_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.SiLU(),
            nn.Linear(64, horizon),
        )
        self.bspline_decoder = BSplineDecoder(
            num_control_points=num_control_points, horizon=horizon, degree=3
        )

    def reset(self) -> None:
        self.coarse_layer.reset()
        self.fine_layer.reset()

    def snap_back(self) -> None:
        self.coarse_layer.snap_back()
        self.fine_layer.snap_back()

    def forward(
        self,
        x_t: torch.Tensor,
        use_inertia: bool = True,
        use_myelination: bool = True,
        record_deformation: bool = True,
    ) -> Dict[str, object]:
        bsz = x_t.shape[0]

        c_out = self.coarse_layer(
            x_t, use_inertia=use_inertia, use_myelination=use_myelination, record_deformation=record_deformation
        )
        c_feat = torch.cat([c_out["h_avg"], c_out["h_int"], c_out["z_dock"]], dim=-1)
        base_anchor = torch.einsum(
            "bk,knd->bnd", c_out["weights"], self.coarse_anchor_templates
        )
        anchor_offset = self.coarse_anchor_head(c_feat).view(bsz, self.num_control_points, 2)
        p_anchor = torch.sigmoid(base_anchor + anchor_offset)

        ctx_coarse = self.coarse_ctx_head(c_feat)
        x_fine = self.dock_norm(x_t + 0.5 * ctx_coarse)

        f_out = self.fine_layer(
            x_fine, use_inertia=use_inertia, use_myelination=use_myelination, record_deformation=record_deformation
        )
        f_feat = torch.cat([f_out["h_avg"], f_out["h_int"], f_out["z_dock"]], dim=-1)
        delta_p_fine = 0.45 * torch.tanh(
            self.fine_residual_head(f_feat).view(bsz, self.num_control_points, 2)
        )
        pen_logits = self.fine_pen_head(f_feat).unsqueeze(-1)

        p_star = torch.clamp(p_anchor + delta_p_fine, 0.0, 1.0)
        y_hat = self.bspline_decoder(p_star)

        return {
            "y_hat": y_hat,
            "p_star": p_star,
            "p_anchor": p_anchor,
            "delta_p_fine": delta_p_fine,
            "pen_logits": pen_logits,
            "x_fine": x_fine,
            "coarse": c_out,
            "fine": f_out,
        }

    def topo_regularization_loss(
        self,
        lambda_r_coarse: float = 0.015,
        lambda_r_fine: float = 0.080,
        lambda_rep: float = 1.60,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """v2.1 제1원칙 2항 위상 정규화 손실 L_topo 통합 계산."""
        l_c, m_c = self.coarse_layer.topo_regularization_loss(
            lambda_r=lambda_r_coarse, lambda_rep=lambda_rep
        )
        l_f, m_f = self.fine_layer.topo_regularization_loss(
            lambda_r=lambda_r_fine, lambda_rep=lambda_rep
        )
        total = l_c + l_f
        metrics = {
            "total_l_topo": float(total.item()),
            "coarse_r_mean": m_c["mean_radius"],
            "coarse_alpha_mean": m_c["mean_alpha"],
            "coarse_min_margin": m_c["min_surface_margin"],
            "fine_r_mean": m_f["mean_radius"],
            "fine_alpha_mean": m_f["mean_alpha"],
            "fine_min_margin": m_f["min_surface_margin"],
        }
        return total, metrics


# ---------------------------------------------------------------------------
# 3. 6가지 핵심 검증 (기존 5항목 + v2.1 수초화/가지뻗기 검증)
# ---------------------------------------------------------------------------


def build_toy_docking_task(
    d_key: int = 128,
    k_coarse: int = 4,
    k_fine: int = 8,
    num_control_points: int = 6,
    horizon: int = 16,
    seed: int = 123,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    gen = torch.Generator().manual_seed(seed)
    decoder = BSplineDecoder(num_control_points=num_control_points, horizon=horizon, degree=3)

    q_mat, _ = torch.linalg.qr(torch.randn(d_key, k_coarse + k_fine, generator=gen))
    coarse_protos = q_mat[:, :k_coarse].t()
    fine_protos = q_mat[:, k_coarse:].t()

    proto_queries, target_cps, target_pens = [], [], []
    task_names = [
        "Apple-ArcL", "Apple-ArcR",
        "Sprout-Stem", "Sprout-Leaf",
        "Star-TipA", "Star-TipB",
        "Cat-EarL", "Cat-EarR",
    ]
    coarse_centers = [
        torch.tensor([0.28, 0.30]),
        torch.tensor([0.72, 0.28]),
        torch.tensor([0.30, 0.72]),
        torch.tensor([0.72, 0.72]),
    ]
    u_t = torch.linspace(0.0, 1.0, num_control_points)

    for idx in range(k_fine):
        c_idx = idx // 2
        variant = idx % 2
        q_vec = F.normalize(0.75 * coarse_protos[c_idx] + 0.65 * fine_protos[idx], p=2, dim=-1)
        proto_queries.append(q_vec)

        base = coarse_centers[c_idx]
        sign = 1.0 if variant == 0 else -1.0
        dx = (u_t - 0.5) * 0.24 * sign
        dy = 0.14 * torch.sin(u_t * math.pi * (1.0 + 0.5 * variant)) - 0.06 * sign
        cp = torch.clamp(base.unsqueeze(0) + torch.stack([dx, dy], dim=-1), 0.05, 0.95)
        target_cps.append(cp)

        pen = torch.ones(horizon, 1)
        pen[-2:] = 0.0 if variant == 1 else 1.0
        target_pens.append(pen)

    return (
        torch.stack(proto_queries, dim=0),
        decoder(torch.stack(target_cps, dim=0)),
        torch.stack(target_pens, dim=0),
        task_names,
    )


def verify_gradient_flow(
    model: TwoStageDTRA, queries: torch.Tensor, trajs: torch.Tensor, pens: torch.Tensor
) -> Dict[str, float]:
    """[검증 1] 미분 가능성 (Gradient Flow) 점검 (v2.1: rho, omega(alpha), beta, P_topo)."""
    model.train()
    model.reset()
    model.zero_grad()

    out = model(queries, use_inertia=False, use_myelination=False, record_deformation=False)
    l_traj = F.mse_loss(out["y_hat"], trajs)  # type: ignore[arg-type]
    l_pen = F.binary_cross_entropy_with_logits(out["pen_logits"], pens)  # type: ignore[arg-type]
    l_topo, _ = model.topo_regularization_loss()
    loss = l_traj + 0.1 * l_pen + l_topo
    loss.backward()

    targets = {
        "coarse.rho": model.coarse_layer.rho.grad,
        "coarse.omega(alpha)": model.coarse_layer.omega.grad,
        "coarse.beta": model.coarse_layer.beta.grad,
        "fine.rho": model.fine_layer.rho.grad,
        "fine.omega(alpha)": model.fine_layer.omega.grad,
        "fine.beta": model.fine_layer.beta.grad,
        "proj_topo.weight": model.proj_topo.weight.grad,
    }
    grad_norms: Dict[str, float] = {}
    for name, grad in targets.items():
        assert grad is not None, f"Gradient missing for {name}"
        assert torch.isfinite(grad).all(), f"NaN/Inf in gradient of {name}"
        norm = float(grad.norm().item())
        assert norm > 0.0, f"Zero gradient for {name}"
        grad_norms[name] = norm
    return grad_norms


def verify_optimization_convergence(
    model: TwoStageDTRA,
    proto_queries: torch.Tensor,
    target_trajs: torch.Tensor,
    target_pens: torch.Tensor,
    steps: int = 200,
) -> Dict[str, List[float]]:
    """[검증 5] 200스텝 제1원칙 2항 L_topo 수렴 및 도킹 최적화 테스트."""
    with torch.no_grad():
        c_keys = torch.stack([
            proto_queries[0:2].mean(dim=0),
            proto_queries[2:4].mean(dim=0),
            proto_queries[4:6].mean(dim=0),
            proto_queries[6:8].mean(dim=0),
        ], dim=0)
        model.coarse_layer.keys.copy_(F.normalize(c_keys, p=2, dim=-1))
        model.fine_layer.keys.copy_(F.normalize(proto_queries, p=2, dim=-1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=2e-3)

    history: Dict[str, List[float]] = {
        "total_loss": [],
        "traj_mse": [],
        "l_topo": [],
        "coarse_r_mean": [],
        "fine_r_mean": [],
        "coarse_alpha_mean": [],
        "fine_alpha_mean": [],
        "coarse_min_margin": [],
        "fine_min_margin": [],
    }

    model.train()
    for _ in range(steps):
        model.reset()
        optimizer.zero_grad()

        idx = torch.arange(proto_queries.shape[0]).repeat(2)
        q_batch = F.normalize(
            proto_queries[idx] + 0.04 * torch.randn_like(proto_queries[idx]), p=2, dim=-1
        )
        out = model(q_batch, use_inertia=False, use_myelination=False, record_deformation=False)
        l_traj = F.mse_loss(out["y_hat"], target_trajs[idx])  # type: ignore[arg-type]
        l_pen = F.binary_cross_entropy_with_logits(out["pen_logits"], target_pens[idx])  # type: ignore[arg-type]
        l_topo, topo_m = model.topo_regularization_loss(
            lambda_r_coarse=0.015, lambda_r_fine=0.080, lambda_rep=1.60
        )
        loss = 20.0 * l_traj + 0.25 * l_pen + 0.15 * l_topo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        history["total_loss"].append(float(loss.item()))
        history["traj_mse"].append(float(l_traj.item()))
        history["l_topo"].append(float(l_topo.item()))
        history["coarse_r_mean"].append(topo_m["coarse_r_mean"])
        history["fine_r_mean"].append(topo_m["fine_r_mean"])
        history["coarse_alpha_mean"].append(topo_m["coarse_alpha_mean"])
        history["fine_alpha_mean"].append(topo_m["fine_alpha_mean"])
        history["coarse_min_margin"].append(topo_m["coarse_min_margin"])
        history["fine_min_margin"].append(topo_m["fine_min_margin"])

    assert history["total_loss"][-1] < history["total_loss"][0] * 0.4
    assert history["traj_mse"][-1] < 0.01
    assert history["coarse_r_mean"][-1] < 1.0 and history["fine_r_mean"][-1] < 0.8
    assert history["coarse_min_margin"][-1] > -0.05 and history["fine_min_margin"][-1] > -0.05
    return history


def verify_snap_back(model: TwoStageDTRA, query: torch.Tensor) -> Dict[str, float]:
    """[검증 2] 탄성 복원 스냅백 오차 (||z_tilde - z^(0)||_inf = 0.0) 검증."""
    model.eval()
    model.reset()
    with torch.no_grad():
        c_z0 = model.coarse_layer.compute_base_rings().unsqueeze(0)
        f_z0 = model.fine_layer.compute_base_rings().unsqueeze(0)

        _ = model(query, use_inertia=True, use_myelination=False, record_deformation=True)
        c_active_dev = float((model.coarse_layer.get_deformed_rings() - c_z0).abs().max().item())
        f_active_dev = float((model.fine_layer.get_deformed_rings() - f_z0).abs().max().item())
        assert c_active_dev > 1e-4 and f_active_dev > 1e-4

        model.snap_back()
        c_post_err = float((model.coarse_layer.get_deformed_rings() - c_z0).abs().max().item())
        f_post_err = float((model.fine_layer.get_deformed_rings() - f_z0).abs().max().item())
        assert c_post_err == 0.0 and f_post_err == 0.0

    return {
        "coarse_active_linf": c_active_dev,
        "fine_active_linf": f_active_dev,
        "coarse_snapback_linf": c_post_err,
        "fine_snapback_linf": f_post_err,
    }


def verify_episode_reset(model: TwoStageDTRA, query: torch.Tensor) -> Dict[str, float]:
    """[검증 3] 에피소드 초기화(reset()) 및 관성·이전 활성 기록 초기화 검증."""
    model.eval()
    model.reset()
    with torch.no_grad():
        _ = model(query, use_inertia=True, use_myelination=False, record_deformation=True)
        c_inertia_t1 = float(model.coarse_layer.inertia_state.max().item())
        f_inertia_t1 = float(model.fine_layer.inertia_state.max().item())
        assert c_inertia_t1 > 0.0 and f_inertia_t1 > 0.0

        out_t2 = model(query, use_inertia=True, use_myelination=False, record_deformation=True)
        c_boost_t2 = float((out_t2["coarse"]["f_mag"] - out_t2["coarse"]["cos_sim"]).max().item())  # type: ignore[index]
        f_boost_t2 = float((out_t2["fine"]["f_mag"] - out_t2["fine"]["cos_sim"]).max().item())    # type: ignore[index]
        assert c_boost_t2 > 0.0 and f_boost_t2 > 0.0

        model.reset()
        c_in_reset = float(model.coarse_layer.inertia_state.abs().max().item())
        f_in_reset = float(model.fine_layer.inertia_state.abs().max().item())
        c_ag_reset = float(model.coarse_layer.prev_gate_state.abs().max().item())
        f_ag_reset = float(model.fine_layer.prev_gate_state.abs().max().item())
        assert max(c_in_reset, f_in_reset, c_ag_reset, f_ag_reset) == 0.0

        out_new = model(query, use_inertia=True, use_myelination=False, record_deformation=True)
        boost_new = max(
            float((out_new["coarse"]["f_mag"] - out_new["coarse"]["cos_sim"]).abs().max().item()),  # type: ignore[index]
            float((out_new["fine"]["f_mag"] - out_new["fine"]["cos_sim"]).abs().max().item()),      # type: ignore[index]
        )
        assert boost_new == 0.0

    model.reset()
    return {
        "coarse_inertia_t1": c_inertia_t1,
        "fine_inertia_t1": f_inertia_t1,
        "coarse_fmag_boost_t2": c_boost_t2,
        "fine_fmag_boost_t2": f_boost_t2,
        "post_reset_max": max(c_in_reset, f_in_reset, c_ag_reset, f_ag_reset),
        "boost_after_reset": boost_new,
    }


def verify_sparse_o1_activation(model: TwoStageDTRA, query: torch.Tensor) -> Dict[str, float]:
    """[검증 4] 학습 모드(완만한 꼬리) vs 추론 모드(O(|A_t|) ≈ O(1) 희소 절단) 검증."""
    model.reset()
    with torch.no_grad():
        model.train()
        out_train = model(query, use_inertia=False, use_myelination=False, record_deformation=False)
        c_tr_cnt = int(out_train["coarse"]["active_count"].item())  # type: ignore[index]
        f_tr_cnt = int(out_train["fine"]["active_count"].item())    # type: ignore[index]
        c_tr_ratio = float(out_train["coarse"]["active_ratio"].item())  # type: ignore[index]
        f_tr_ratio = float(out_train["fine"]["active_ratio"].item())    # type: ignore[index]

        model.eval()
        out_eval = model(query, use_inertia=False, use_myelination=False, record_deformation=False)
        c_ev_cnt = int(out_eval["coarse"]["active_count"].item())  # type: ignore[index]
        f_ev_cnt = int(out_eval["fine"]["active_count"].item())    # type: ignore[index]
        c_ev_ratio = float(out_eval["coarse"]["active_ratio"].item())  # type: ignore[index]
        f_ev_ratio = float(out_eval["fine"]["active_ratio"].item())    # type: ignore[index]

    assert c_tr_ratio == 1.0 and f_tr_ratio == 1.0
    assert 0 < c_ev_cnt < model.k_coarse and 0 < f_ev_cnt < model.k_fine
    return {
        "coarse_train_count": c_tr_cnt,
        "coarse_train_ratio": c_tr_ratio,
        "fine_train_count": f_tr_cnt,
        "fine_train_ratio": f_tr_ratio,
        "coarse_eval_count": c_ev_cnt,
        "coarse_eval_ratio": c_ev_ratio,
        "fine_eval_count": f_ev_cnt,
        "fine_eval_ratio": f_ev_ratio,
    }


def verify_myelination_and_outgrowth(
    model: TwoStageDTRA, proto_queries: torch.Tensor, pair_i: int = 4, pair_j: int = 5
) -> Dict[str, object]:
    """[검증 6 (v2.1)] 수초화 위상 지름길(모양 보존 가지 뻗기 + 명령 호환성 게이팅 부스트) 검증."""
    layer = model.fine_layer
    layer.reset()
    layer.reset_myelination()
    model.eval()

    with torch.no_grad():
        coeffs_before = layer.fourier_coeffs.clone()
        radii_before = layer.radii.clone()
        alpha_before = layer.compliance.clone()
        z0_before = layer.compute_base_rings().cpu().numpy()

        dist_history: List[float] = []
        m_ij_history: List[float] = []
        safe_floor = 0.0

        # 12스텝 동시 도킹 수초화 및 가지 뻗기 시뮬레이션
        for step in range(12):
            res = layer.myelinate_and_grow_step(pair_i, pair_j, co_strength=0.35, outgrowth_lr=0.18)
            if step == 0:
                dist_history.append(res["d_before"])
                m_ij_history.append(0.0)
            dist_history.append(res["d_after"])
            m_ij_history.append(res["m_ij"])
            safe_floor = res["safe_floor"]

        z0_after = layer.compute_base_rings().cpu().numpy()

        # 고유 모양(푸리에 계수), 반경 r_i, 순응도 alpha_i 100% 보존 검증
        shape_err = float((layer.fourier_coeffs - coeffs_before).abs().max().item())
        radius_err = float((layer.radii - radii_before).abs().max().item())
        alpha_err = float((layer.compliance - alpha_before).abs().max().item())
        assert shape_err == 0.0 and radius_err == 0.0 and alpha_err == 0.0
        assert dist_history[-1] < dist_history[0], "Expected co-docked tubes to grow closer"
        assert dist_history[-1] >= safe_floor - 1e-3, "Must stop safely outside repulsion floor"

        # 명령 호환성 게이팅 검증:
        # Step t=1: pair_i 활성(a_i > 0)
        layer.reset()
        q_i = F.normalize(layer.keys[pair_i : pair_i + 1], p=2, dim=-1)
        _ = layer(q_i, use_inertia=True, use_myelination=True, record_deformation=False)
        a_i_prev = float(layer.prev_gate_state[0, pair_i].item())
        assert a_i_prev > 0.0

        # Case A (호환 명령): pair_j와 부분 호환되는 쿼리 (cos_sim(x, k_j) > 0) -> 양의 M_ij 부스트 발생
        q_compat = F.normalize(0.55 * layer.keys[pair_j : pair_j + 1] + 0.45 * layer.keys[pair_i : pair_i + 1], p=2, dim=-1)
        out_compat = layer(q_compat, use_inertia=True, use_myelination=True, record_deformation=False)
        compat_boost = float(out_compat["myelin_boost"][0, pair_j].item())
        assert compat_boost > 0.0, f"Expected positive myelination boost on compatible query, got {compat_boost}"

        # Case B (배타/무관 명령): k_j와 반대/직교 쿼리 (cos_sim(x, k_j) <= 0) -> M_ij 부스트 0.0 차단
        layer.reset()
        _ = layer(q_i, use_inertia=True, use_myelination=True, record_deformation=False)
        q_incompat = F.normalize(-layer.keys[pair_j : pair_j + 1] + 0.35 * layer.keys[pair_i : pair_i + 1], p=2, dim=-1)
        out_incompat = layer(q_incompat, use_inertia=True, use_myelination=True, record_deformation=False)
        incompat_boost = float(out_incompat["myelin_boost"][0, pair_j].item())
        assert incompat_boost == 0.0, f"Expected 0.0 misfire boost on incompatible query, got {incompat_boost}"

    layer.reset()
    return {
        "pair": (pair_i, pair_j),
        "shape_err": shape_err,
        "radius_err": radius_err,
        "alpha_err": alpha_err,
        "d_before": dist_history[0],
        "d_after": dist_history[-1],
        "safe_floor": safe_floor,
        "m_ij_final": m_ij_history[-1],
        "compat_boost": compat_boost,
        "incompat_boost": incompat_boost,
        "dist_history": dist_history,
        "m_ij_history": m_ij_history,
        "z0_before": z0_before,
        "z0_after": z0_after,
    }


# ---------------------------------------------------------------------------
# 4. 4-패널 클로즈업 3D 투영 시각화 (outputs/dtra_step1_dynamics_clean.png)
# ---------------------------------------------------------------------------


def _pca_project_basis(points_nd: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    mean = points_nd.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(points_nd - mean, full_matrices=False)
    return mean.squeeze(0), vt[:n_components].T


def _project_3d(pts: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (pts - mean) @ basis


def _plot_3d_solid_tube(
    ax: plt.Axes,
    curve_3d: np.ndarray,
    tube_radius: float,
    color: str,
    alpha: float = 0.65,
    n_phi: int = 14,
) -> None:
    pts = np.vstack([curve_3d, curve_3d[0:1]])
    n_pts = pts.shape[0]
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[-2]
    tangents[-1] = tangents[0]
    tangents = tangents / (np.linalg.norm(tangents, axis=-1, keepdims=True) + 1e-8)

    ref = np.array([0.0, 0.0, 1.0]) if abs(tangents[0, 2]) < 0.85 else np.array([1.0, 0.0, 0.0])
    n_vec = np.cross(tangents[0], ref)
    n_vec = n_vec / (np.linalg.norm(n_vec) + 1e-8)

    normals = np.zeros_like(pts)
    binormals = np.zeros_like(pts)
    normals[0] = n_vec
    binormals[0] = np.cross(tangents[0], normals[0])

    for i in range(1, n_pts):
        v = np.cross(tangents[i - 1], tangents[i])
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-7:
            axis = v / v_norm
            angle = math.acos(float(np.clip(np.dot(tangents[i - 1], tangents[i]), -1.0, 1.0)))
            n_prev = normals[i - 1]
            n_rot = (
                n_prev * math.cos(angle)
                + np.cross(axis, n_prev) * math.sin(angle)
                + axis * np.dot(axis, n_prev) * (1.0 - math.cos(angle))
            )
            normals[i] = n_rot - np.dot(n_rot, tangents[i]) * tangents[i]
            normals[i] = normals[i] / (np.linalg.norm(normals[i]) + 1e-8)
        else:
            normals[i] = normals[i - 1]
        binormals[i] = np.cross(tangents[i], normals[i])

    phi = np.linspace(0.0, 2.0 * math.pi, n_phi)
    cos_p, sin_p = np.cos(phi)[None, :], np.sin(phi)[None, :]
    x_surf = pts[:, 0:1] + tube_radius * (normals[:, 0:1] * cos_p + binormals[:, 0:1] * sin_p)
    y_surf = pts[:, 1:2] + tube_radius * (normals[:, 1:2] * cos_p + binormals[:, 1:2] * sin_p)
    z_surf = pts[:, 2:3] + tube_radius * (normals[:, 2:3] * cos_p + binormals[:, 2:3] * sin_p)
    ax.plot_surface(x_surf, y_surf, z_surf, color=color, alpha=alpha, linewidth=0.0, antialiased=True, shade=True)


def render_four_panel_visualization(
    model: TwoStageDTRA,
    sample_query: torch.Tensor,
    history: Dict[str, List[float]],
    snap_stats: Dict[str, float],
    sparse_stats: Dict[str, float],
    myelin_stats: Dict[str, object],
    out_path: Path,
    lang: str = "en",
) -> Path:
    """4-패널 클로즈업 위상 동역학 시각화 (v2.1: outputs/dtra_step1_dynamics_clean.png, 200 DPI)."""
    is_ko = (lang == "ko")
    if is_ko:
        plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "Noto Sans KR", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["mathtext.fontset"] = "dejavusans"

    model.eval()
    model.reset()

    with torch.no_grad():
        c_z0 = model.coarse_layer.compute_base_rings().cpu().numpy()
        f_z0 = model.fine_layer.compute_base_rings().cpu().numpy()
        c_radii = model.coarse_layer.radii.cpu().numpy()
        f_radii = model.fine_layer.radii.cpu().numpy()

        out_active = model(sample_query, use_inertia=False, use_myelination=False, record_deformation=True)
        c_def = model.coarse_layer.get_deformed_rings()[0].cpu().numpy()
        f_def = model.fine_layer.get_deformed_rings()[0].cpu().numpy()
        c_gate = out_active["coarse"]["gate"][0].cpu().numpy()  # type: ignore[index]
        f_gate = out_active["fine"]["gate"][0].cpu().numpy()    # type: ignore[index]
        q_c_src = out_active["coarse"]["c_query"][0].cpu().numpy()  # type: ignore[index]
        dock_c_src = out_active["coarse"]["z_dock"][0].cpu().numpy()  # type: ignore[index]
        dock_f_src = out_active["fine"]["z_dock"][0].cpu().numpy()    # type: ignore[index]
        model.snap_back()

    all_pts = np.concatenate([
        c_z0.reshape(-1, model.d_topo),
        f_z0.reshape(-1, model.d_topo),
        c_def.reshape(-1, model.d_topo),
        f_def.reshape(-1, model.d_topo),
        q_c_src[None, :],
    ], axis=0)
    pca_mean, pca_basis = _pca_project_basis(all_pts, n_components=3)

    c_z0_3d = _project_3d(c_z0, pca_mean, pca_basis)
    f_z0_3d = _project_3d(f_z0, pca_mean, pca_basis)
    c_def_3d = _project_3d(c_def, pca_mean, pca_basis)
    f_def_3d = _project_3d(f_def, pca_mean, pca_basis)
    q_c_3d = _project_3d(q_c_src, pca_mean, pca_basis)
    dock_c_3d = _project_3d(dock_c_src, pca_mean, pca_basis)
    dock_f_3d = _project_3d(dock_f_src, pca_mean, pca_basis)

    xyz_all = np.concatenate([c_z0_3d.reshape(-1, 3), f_z0_3d.reshape(-1, 3)], axis=0)
    lim_min = xyz_all.min(axis=0) - 0.10
    lim_max = xyz_all.max(axis=0) + 0.10
    vis_scale = 0.13

    coarse_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf"]
    fine_colors = ["#ff7f0e", "#d62728", "#e377c2", "#bcbd22", "#8c564b", "#f28e2b", "#e15759", "#76b7b2"]

    fig = plt.figure(figsize=(16.5, 12.0), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.24)

    # --- Panel 1: Global Idle Manifold ---
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    for i in range(model.k_coarse):
        _plot_3d_solid_tube(ax1, c_z0_3d[i], float(c_radii[i]) * vis_scale, coarse_colors[i % 4], alpha=0.52)
    for j in range(model.k_fine):
        _plot_3d_solid_tube(ax1, f_z0_3d[j], float(f_radii[j]) * vis_scale, fine_colors[j % 8], alpha=0.75)
    ax1.set_xlim(lim_min[0], lim_max[0])
    ax1.set_ylim(lim_min[1], lim_max[1])
    ax1.set_zlim(lim_min[2], lim_max[2])
    ax1.view_init(elev=22, azim=-52)
    if is_ko:
        ax1.set_title(
            f"패널 1: 대기 상태의 기억 공간 (64차원 → 3D 압축)\n"
            f"큰 동작 고리(굵음) 반경={c_radii.mean():.2f}, 말랑함 α={history['coarse_alpha_mean'][-1]:.2f} | "
            f"정밀 고리(얇음) 반경={f_radii.mean():.2f}, α={history['fine_alpha_mean'][-1]:.2f}",
            fontsize=10.0, fontweight="bold", pad=8,
        )
        ax1.set_xlabel("축 1 (PC1)", fontsize=8)
        ax1.set_ylabel("축 2 (PC2)", fontsize=8)
        ax1.set_zlabel("축 3 (PC3)", fontsize=8)
    else:
        ax1.set_title(
            f"Panel 1: Idle Manifold (D_topo=64 -> PCA 3D)\n"
            f"Coarse r={c_radii.mean():.2f} (α={history['coarse_alpha_mean'][-1]:.2f}) | "
            f"Fine r={f_radii.mean():.2f} (α={history['fine_alpha_mean'][-1]:.2f})",
            fontsize=10.5, fontweight="bold", pad=8,
        )
        ax1.set_xlabel("PC1", fontsize=8)
        ax1.set_ylabel("PC2", fontsize=8)
        ax1.set_zlabel("PC3", fontsize=8)

    # --- Panel 2: Close-up Active Deformation & 2-Stage Docking ---
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    act_pts = [q_c_3d[None, :], dock_c_3d[None, :], dock_f_3d[None, :]]
    for i in range(model.k_coarse):
        is_act = c_gate[i] > 0.0
        col = coarse_colors[i % 4] if is_act else "#cfd4da"
        alp = 0.62 if is_act else 0.14
        if is_act:
            closed_idle = np.vstack([c_z0_3d[i], c_z0_3d[i, 0:1]])
            ax2.plot(closed_idle[:, 0], closed_idle[:, 1], closed_idle[:, 2], "--", color=col, lw=1.4, alpha=0.85)
            act_pts.append(c_def_3d[i])
            act_pts.append(c_z0_3d[i])
        _plot_3d_solid_tube(ax2, c_def_3d[i], float(c_radii[i]) * vis_scale, col, alpha=alp)

    for j in range(model.k_fine):
        is_act = f_gate[j] > 0.0
        col = fine_colors[j % 8] if is_act else "#d6dbe0"
        alp = 0.88 if is_act else 0.12
        if is_act:
            closed_idle = np.vstack([f_z0_3d[j], f_z0_3d[j, 0:1]])
            ax2.plot(closed_idle[:, 0], closed_idle[:, 1], closed_idle[:, 2], "--", color=col, lw=1.3, alpha=0.85)
            act_pts.append(f_def_3d[j])
            act_pts.append(f_z0_3d[j])
        _plot_3d_solid_tube(ax2, f_def_3d[j], float(f_radii[j]) * vis_scale, col, alpha=alp)

    lbl_q = "현재 상황(질문) c(x_t)" if is_ko else "Query c(x_t)"
    lbl_s1 = "1단계 합체점 (큰 동작)" if is_ko else "Stage1 Dock"
    lbl_s2 = "2단계 합체점 (정밀 동작)" if is_ko else "Stage2 Dock"
    ax2.scatter([q_c_3d[0]], [q_c_3d[1]], [q_c_3d[2]], c="#ffd700", edgecolors="black", s=180, marker="*", zorder=10, label=lbl_q)
    ax2.scatter([dock_c_3d[0]], [dock_c_3d[1]], [dock_c_3d[2]], c="#1f77b4", edgecolors="black", s=75, marker="o", zorder=10, label=lbl_s1)
    ax2.scatter([dock_f_3d[0]], [dock_f_3d[1]], [dock_f_3d[2]], c="#d62728", edgecolors="black", s=75, marker="^", zorder=10, label=lbl_s2)
    ax2.plot([dock_c_3d[0], dock_f_3d[0]], [dock_c_3d[1], dock_f_3d[1]], [dock_c_3d[2], dock_f_3d[2]], "k-", lw=2.0)

    # 활성 도킹 영역을 중심으로 클로즈업 줌
    zoom_pts = np.concatenate(act_pts, axis=0)
    z_min = zoom_pts.min(axis=0) - 0.10
    z_max = zoom_pts.max(axis=0) + 0.10
    ax2.set_xlim(z_min[0], z_max[0])
    ax2.set_ylim(z_min[1], z_max[1])
    ax2.set_zlim(z_min[2], z_max[2])
    ax2.view_init(elev=22, azim=-52)
    ax2.legend(loc="upper left", fontsize=8.0, framealpha=0.9)
    if is_ko:
        ax2.set_title(
            f"패널 2: 자석처럼 휘어지며 2단계 도킹 (확대)\n"
            f"필요한 고리만 켜짐(O(1)): 큰 고리 {int(sparse_stats['coarse_eval_count'])}/{model.k_coarse}개, "
            f"정밀 고리 {int(sparse_stats['fine_eval_count'])}/{model.k_fine}개만 연산",
            fontsize=10.0, fontweight="bold", pad=8,
        )
        ax2.set_xlabel("축 1 (PC1)", fontsize=8)
        ax2.set_ylabel("축 2 (PC2)", fontsize=8)
        ax2.set_zlabel("축 3 (PC3)", fontsize=8)
    else:
        ax2.set_title(
            f"Panel 2: Close-up Closed-Form Deformation & Docking\n"
            f"Sparse Eval Active: Coarse {int(sparse_stats['coarse_eval_count'])}/{model.k_coarse}, "
            f"Fine {int(sparse_stats['fine_eval_count'])}/{model.k_fine}",
            fontsize=10.5, fontweight="bold", pad=8,
        )
        ax2.set_xlabel("PC1", fontsize=8)
        ax2.set_ylabel("PC2", fontsize=8)
        ax2.set_zlabel("PC3", fontsize=8)

    # --- Panel 3: Edge Myelination & Structural Outgrowth ---
    ax3 = fig.add_subplot(gs[1, 0])
    m_steps = np.arange(len(myelin_stats["dist_history"]))  # type: ignore[arg-type]
    ax3_twin = ax3.twinx()

    l1 = ax3.plot(
        m_steps, myelin_stats["dist_history"], "o-", color="#2563eb", lw=2.2,  # type: ignore[arg-type]
        label=("두 고리 사이 거리 d_chord (가까워짐 ↓)" if is_ko else "Chord Dist d(R_i, R_j)"),
    )
    l2 = ax3.axhline(
        myelin_stats["safe_floor"], color="#dc2626", linestyle="--", lw=1.6,  # type: ignore[arg-type]
        label=(f"충돌 방지 안전선 r_i+r_j+δ ({myelin_stats['safe_floor']:.3f})" if is_ko else f"Safety Floor r_i+r_j+δ ({myelin_stats['safe_floor']:.3f})"),
    )
    l3 = ax3_twin.plot(
        m_steps, myelin_stats["m_ij_history"], "^--", color="#059669", lw=2.0,  # type: ignore[arg-type]
        label=("신경 연결(미엘린) 강도 M_ij (강화 ↑)" if is_ko else "Myelination M_ij = η tanh(C_ij)"),
    )

    ax3.set_xlabel("같이 작업한 경험 횟수 (스텝)" if is_ko else "Co-Docking Experience Step", fontsize=9.5)
    ax3.set_ylabel("고리 중심 간 거리 (d_chord)" if is_ko else "Chord Distance d_chord", fontsize=9.5, color="#1d4ed8")
    ax3_twin.set_ylabel("신경 연결 강도 M_ij" if is_ko else "Coupling Conductance M_ij", fontsize=9.5, color="#047857")
    ax3.grid(True, alpha=0.3)

    lines = l1 + [l2] + l3
    labels = [ln.get_label() for ln in lines]
    ax3.legend(lines, labels, loc="center right", fontsize=8.0, framealpha=0.92)
    if is_ko:
        ax3.set_title(
            "패널 3: 자주 같이 쓰면 신경 연결 강화 + 고리 모양 훼손 없이 가까워짐\n"
            f"모양 변형 오차={myelin_stats['shape_err']:.1f} | 같은 명령 도움=+{myelin_stats['compat_boost']:.3f} vs 다른 명령 오작동={myelin_stats['incompat_boost']:.1f}",
            fontsize=10.0, fontweight="bold", pad=8,
        )
    else:
        ax3.set_title(
            "Panel 3: Edge Myelination & Shape-Preserving Outgrowth\n"
            f"Shape Err={myelin_stats['shape_err']:.1f} | Compat Boost=+{myelin_stats['compat_boost']:.3f} vs Incompat={myelin_stats['incompat_boost']:.1f}",
            fontsize=10.5, fontweight="bold", pad=8,
        )

    # --- Panel 4: Snap-back & 2-Term L_topo Convergence ---
    ax4 = fig.add_subplot(gs[1, 1])
    steps_x = np.arange(1, len(history["total_loss"]) + 1)
    ax4.plot(steps_x, history["total_loss"], color="#111827", lw=2.2, label=("전체 오차 (Total Loss)" if is_ko else "Total Loss"))
    ax4.plot(steps_x, history["traj_mse"], color="#2563eb", lw=1.8, label=("움직임 궤적 오차 (Traj MSE)" if is_ko else "Trajectory MSE"))
    ax4.plot(steps_x, history["l_topo"], color="#d97706", lw=1.8, linestyle="--", label=("고리 크기+간격 유지 오차 (L_topo)" if is_ko else "2-Term L_topo (Vol + Repulsion)"))
    ax4.set_yscale("log")
    if is_ko:
        from matplotlib.ticker import FuncFormatter
        ax4.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" if 1e-4 <= v <= 10 else f"{v:.0e}"))
    ax4.grid(True, which="both", linestyle=":", alpha=0.45)
    ax4.set_xlabel("학습 반복 횟수 (Step)" if is_ko else "Optimization Step", fontsize=9.5)
    ax4.set_ylabel("오차 크기 (로그 스케일)" if is_ko else "Loss (log scale)", fontsize=9.5)
    ax4.legend(loc="upper right", fontsize=8.2, framealpha=0.9)
    if is_ko:
        ax4.set_title(
            "패널 4: 작업 후 100% 원상복귀(스냅백) & 200스텝 학습 수렴 곡선\n"
            f"스냅백 복원 오차 ||z_tilde - z^(0)||_inf = {max(snap_stats['coarse_snapback_linf'], snap_stats['fine_snapback_linf']):.1f} (망각 0%)",
            fontsize=10.0, fontweight="bold", pad=8,
        )
        summary_text = (
            f"작업 중 최대 찌그러짐:\n"
            f"  큰 고리   = {snap_stats['coarse_active_linf']:.4f}\n"
            f"  정밀 고리 = {snap_stats['fine_active_linf']:.4f}\n"
            f"작업 끝 원상복귀 후 오차:\n"
            f"  큰 고리   = {snap_stats['coarse_snapback_linf']:.1e}\n"
            f"  정밀 고리 = {snap_stats['fine_snapback_linf']:.1e}\n"
            f"최종 궤적 오차 = {history['traj_mse'][-1]:.5f}\n"
            f"최종 위상 오차 = {history['l_topo'][-1]:.3f}"
        )
    else:
        ax4.set_title(
            "Panel 4: Post-Inference Snap-back & 2-Term L_topo Convergence\n"
            f"Snap-back Error ||z_tilde - z^(0)||_inf = {max(snap_stats['coarse_snapback_linf'], snap_stats['fine_snapback_linf']):.1f}",
            fontsize=10.5, fontweight="bold", pad=8,
        )
        summary_text = (
            f"Active Max |Delta z|:\n"
            f"  Coarse = {snap_stats['coarse_active_linf']:.4f}\n"
            f"  Fine   = {snap_stats['fine_active_linf']:.4f}\n"
            f"After snap_back():\n"
            f"  Coarse Err = {snap_stats['coarse_snapback_linf']:.1e}\n"
            f"  Fine Err   = {snap_stats['fine_snapback_linf']:.1e}\n"
            f"Final Traj MSE = {history['traj_mse'][-1]:.5f}\n"
            f"Final L_topo   = {history['l_topo'][-1]:.3f}"
        )
    ax4.text(
        0.04, 0.06, summary_text, transform=ax4.transAxes, fontsize=8.0,
        family=("Malgun Gothic" if is_ko else "monospace"),
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#64748b", alpha=0.94),
    )

    if is_ko:
        fig.suptitle(
            "DTRA v2.1 1단계 원리 검증: 1차 타원 고리·2단계 도킹·신경 연결 강화·100% 탄성 복원",
            fontsize=13.0, fontweight="bold", y=0.99,
        )
    else:
        fig.suptitle(
            "DTRA v2.1 Step 1: 1st-Order Elliptical Hypertubes, Closed-form Docking, Myelination & Snap-back",
            fontsize=13.0, fontweight="bold", y=0.99,
        )
    return safe_savefig(fig, out_path, dpi=200)


# ---------------------------------------------------------------------------
# 5. 메인 실행 루틴
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ko", action="store_true", help="한국어 직관 해설 버전 PNG도 함께 저장")
    args = parser.parse_args()
    set_seed(42)
    print("=" * 74)
    print("DTRA v2.1 Step 1: Hypertube Topology Simulation & Unit Verification")
    print("=" * 74)

    proto_queries, target_trajs, target_pens, _ = build_toy_docking_task(
        d_key=128, k_coarse=4, k_fine=8, num_control_points=6, horizon=16
    )
    model = TwoStageDTRA(
        d_key=128, d_topo=64, k_coarse=4, k_fine=8, num_control_points=6, horizon=16
    )

    # 1. 미분 가능성 (Gradient Flow)
    grad_norms = verify_gradient_flow(model, proto_queries, target_trajs, target_pens)
    print("\n[1/6 PASS] Gradient Flow Verification (v2.1, No NaN/Inf):")
    for pname, gnorm in grad_norms.items():
        print(f"  - {pname:<22s}: ||grad|| = {gnorm:.6e}")

    # 5. 제1원칙 2항 L_topo 200스텝 수렴 테스트
    history = verify_optimization_convergence(
        model, proto_queries, target_trajs, target_pens, steps=200
    )
    print("\n[5/6 PASS] 2-Term L_topo 200-Step Toy Optimization Convergence:")
    print(f"  - Total Loss   : {history['total_loss'][0]:.4f} -> {history['total_loss'][-1]:.4f}")
    print(f"  - Traj MSE     : {history['traj_mse'][0]:.5f} -> {history['traj_mse'][-1]:.5f}")
    print(f"  - L_topo       : {history['l_topo'][0]:.4f} -> {history['l_topo'][-1]:.4f}")
    print(
        f"  - Coarse r/α   : r={history['coarse_r_mean'][-1]:.4f}, α={history['coarse_alpha_mean'][-1]:.4f}, "
        f"min_margin={history['coarse_min_margin'][-1]:.4f}"
    )
    print(
        f"  - Fine   r/α   : r={history['fine_r_mean'][-1]:.4f}, α={history['fine_alpha_mean'][-1]:.4f}, "
        f"min_margin={history['fine_min_margin'][-1]:.4f}"
    )

    test_query = proto_queries[4:5]

    # 2. 탄성 복원 스냅백 오차 검증
    snap_stats = verify_snap_back(model, test_query)
    print("\n[2/6 PASS] Zero-Forgetting Elastic Snap-back Verification:")
    print(f"  - Active Deformation L_inf : Coarse={snap_stats['coarse_active_linf']:.6f}, Fine={snap_stats['fine_active_linf']:.6f}")
    print(f"  - Post-Snap-back Error L_inf: Coarse={snap_stats['coarse_snapback_linf']:.1f}, Fine={snap_stats['fine_snapback_linf']:.1f}")

    # 3. 에피소드 초기화(reset())
    reset_stats = verify_episode_reset(model, test_query)
    print("\n[3/6 PASS] Episode Lifecycle reset() & Cross-Chunk Inertia Verification:")
    print(f"  - Inertia after t=1        : Coarse={reset_stats['coarse_inertia_t1']:.4f}, Fine={reset_stats['fine_inertia_t1']:.4f}")
    print(f"  - F_mag Boost at t=2       : Coarse=+{reset_stats['coarse_fmag_boost_t2']:.4f}, Fine=+{reset_stats['fine_fmag_boost_t2']:.4f}")
    print(f"  - Post-reset Buffer / Boost: buffer={reset_stats['post_reset_max']:.1f}, boost={reset_stats['boost_after_reset']:.1f}")

    # 4. 추론 O(1) 희소 활성화
    sparse_stats = verify_sparse_o1_activation(model, test_query)
    print("\n[4/6 PASS] Inference O(|A_t|) ≈ O(1) Sparse Activation vs Training Soft Tail:")
    print(
        f"  - Train Mode (Soft Tail)   : "
        f"Coarse {int(sparse_stats['coarse_train_count'])}/{model.k_coarse} ({sparse_stats['coarse_train_ratio']*100:.1f}%), "
        f"Fine {int(sparse_stats['fine_train_count'])}/{model.k_fine} ({sparse_stats['fine_train_ratio']*100:.1f}%)"
    )
    print(
        f"  - Eval Mode  (O(1) Pruned) : "
        f"Coarse {int(sparse_stats['coarse_eval_count'])}/{model.k_coarse} ({sparse_stats['coarse_eval_ratio']*100:.1f}%), "
        f"Fine {int(sparse_stats['fine_eval_count'])}/{model.k_fine} ({sparse_stats['fine_eval_ratio']*100:.1f}%)"
    )

    # 6. 수초화 위상 지름길 & 모양 보존 가지 뻗기
    myelin_stats = verify_myelination_and_outgrowth(model, proto_queries, pair_i=4, pair_j=5)
    print("\n[6/6 PASS] Edge Myelination & Structural Outgrowth Verification:")
    print(f"  - Shape / r / α Drift      : shape={myelin_stats['shape_err']:.1f}, r={myelin_stats['radius_err']:.1f}, α={myelin_stats['alpha_err']:.1f}")
    print(
        f"  - Co-Docked Chord Dist     : {myelin_stats['d_before']:.4f} -> {myelin_stats['d_after']:.4f} "
        f"(Safety Floor={myelin_stats['safe_floor']:.4f}, M_ij={myelin_stats['m_ij_final']:.4f})"
    )
    print(
        f"  - Command-Gated Recall     : Compatible=+{myelin_stats['compat_boost']:.4f} | "
        f"Incompatible={myelin_stats['incompat_boost']:.1f} (Zero Misfire)"
    )

    out_img = PROJECT_ROOT / "outputs" / "figures" / "dtra_step1_dynamics_clean.png"
    saved_path = render_four_panel_visualization(
        model=model,
        sample_query=test_query,
        history=history,
        snap_stats=snap_stats,
        sparse_stats=sparse_stats,
        myelin_stats=myelin_stats,
        out_path=out_img,
        lang="en",
    )
    print(f"\n[SAVED] 4-Panel Close-up Visualization -> {saved_path}")
    if args.ko:
        out_img_ko = PROJECT_ROOT / "outputs" / "figures" / "dtra_step1_dynamics_clean_ko.png"
        saved_ko = render_four_panel_visualization(
            model=model,
            sample_query=test_query,
            history=history,
            snap_stats=snap_stats,
            sparse_stats=sparse_stats,
            myelin_stats=myelin_stats,
            out_path=out_img_ko,
            lang="ko",
        )
        print(f"[SAVED] 한국어 해설판 시각화 -> {saved_ko}")
    print("=" * 74)


if __name__ == "__main__":
    main()
