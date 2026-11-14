"""experiments/ablate_dtra_scientific.py

DTRA v2.1 가설의 인과적 분리·공정 통제·스트레스 검증을 위한 통합 과학 절제(Ablation) 벤치마크.

핵심 설계 원칙:
  1. 기존 `experiments/benchmark_dtra_3d_metaworld.py`(Zenodo DOI: 10.5281/zenodo.24218483)는
     절대 수정하지 않고 모델·데이터 로더·OOD 섭동 유틸을 재사용/서브클래싱합니다.
  2. 6개 통제군(Arm)의 공정 비교 보장:
     - 모든 모델이 동일한 Conv4 공간 인코더(256-d), 고유수용감각 네트(64-d),
       Cubic B-Spline 디코더(H=16, Nk=6, 4-DoF), 6-mode WTA 손실을 공유합니다.
     - K=8 기준 전체 파라미터 총량을 ~5.25M 수준으로 정밀 일치시킵니다.
     - [Arm 1] ABK-VLA (15x15 DoG, Vision-Only): 기존 대각선 복사 단축키(L1478-1480)를
       완전히 제거하고 모든 학습 단계 s=0..7에서 과거 모든 태스크 e=0..s를 실제 재평가해
       엄밀한 8x8 하삼각 망각률 행렬을 산출합니다(Argmax & Blend 모두 기록).
     - [Arm 2] Matched Frozen MLP (Vision-Only, lang=0): Conv4 시각 키(vis_64, 언어=0) +
       파라미터를 맞춘 단순 동결 MLP 슬롯(하이퍼튜브/탄성/2단계 도킹 없음).
     - [Arm 3] DTRA Full (Vision-Only, lang=0): Conv4 시각 키(vis_64, 언어=0) +
       초입체 튜브 링 + 2단계 도킹 + 학습 탄성 순응도(alpha learned) + record_deformation=True.
     - [Arm 4] Matched Frozen MLP (Multimodal): 동일 멀티모달 키([0.55*vis_64; 0.835*lang_64]) +
       파라미터를 맞춘 단순 동결 MLP 슬롯.
     - [Arm 5] DTRA Rigid (Multimodal, alpha=0): 동일 멀티모달 키 + 초입체 튜브 링 구조는
       유지하되 탄성 순응도 alpha_i=0 강제(강체 투영, No Snap-Back).
     - [Arm 6] DTRA Full (Multimodal, alpha=learned): 전체 DTRA 모델 + record_deformation=True.
  3. 4가지 평가 조건 + 실측 텔레메트리 + CPU 지연 시간 분해:
     - [Clean ID]: held-out train_tasks[35:50] 15 에피소드, 전체 8x8 행렬 평가
     - [Official Test Split]: Meta-World 공식 ml1.test_tasks 평가 및 train/test 구조 차이 명시
     - [OOD Visual Severe]: RGB 가우시안 노이즈(sigma=0.20) 적용 시 성공률 및 라우팅 정확도
     - [Paraphrased Instructions]: Arms 4~6 대상 미학습 동의어 지시문(태스크당 3종) 평가
     - [Snap-Back Telemetry]: Arms 3, 5, 6 대상 에피소드별 ||delta_z_star||, raw pull,
       snap_back() 호출 전후 변위 버퍼 잔차를 Clean ID vs OOD Severe 각각 기록
     - [Latency Breakdown]: CPU 기준 Key Search O(K) ms vs Active Execution O(1)/O(K) ms 정밀 분해

사용 예시:
  1) 단일 시드 무결성 및 1차 검증:
     python experiments/ablate_dtra_scientific.py --seeds 42
  2) 다중 시드(3-seed) 전체 과학 검증:
     python experiments/ablate_dtra_scientific.py --seeds 42 100 2026
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import metaworld
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from experiments.benchmark_autonomous_cl import calc_cl_metrics, set_seed
from experiments.benchmark_dtra_3d_metaworld import (
    DTRA3DPolicy,
    HypertubeNode,
    LightweightTextEncoder,
    OUT_DIR,
    TASK_INSTRUCTIONS,
    TASK_SPECS_8,
    apply_ood_rgb_perturbation,
    pick_device,
    sync,
)
from experiments.benchmark_metaworld_cl import (
    Autonomous3DABKPolicy,
    MetaWorldBSplineSlotHead,
    build_rgb_spatial_encoder,
    collect_or_load_task_data,
)
from tools.bspline_core import BSplineDecoder, build_difference_matrices


# ==============================================================================
# 0. 동의어 패러프레이징 지시문(태스크당 3종, 학습 문장에 미등장한 어휘/어순 변형)
# ==============================================================================

PARAPHRASED_INSTRUCTIONS: Dict[str, List[str]] = {
    "button-press-v3": [
        "push the button down",
        "depress the crimson switch mounted on the tabletop",
        "move the gripper ahead and tap down the tabletop button",
    ],
    "door-open-v3": [
        "latch onto the handle and swing the miniature door outward",
        "seize the grip bar to open the hinged door panel",
        "grab the knob on the small door and yank it open",
    ],
    "window-open-v3": [
        "slide the casement sash laterally along the track",
        "shove the glass pane latch horizontally until it opens",
        "glide the window grip to the side to unseal the pane",
    ],
    "faucet-open-v3": [
        "rotate the spigot valve anticlockwise to turn on water",
        "twist the tap lever leftwards so water flows",
        "spin the faucet knob counter clockwise to release water",
    ],
    "drawer-open-v3": [
        "open the cabinet drawer",
        "grab the front grip and slide out the storage tray",
        "hold the drawer knob and draw the compartment toward you",
    ],
    "drawer-close-v3": [
        "shut the extended cabinet tray by shoving it back in",
        "press the pulled out drawer front inward until flush",
        "push the cabinet drawer completely inside",
    ],
    "handle-press-v3": [
        "force the bar switch straight downward completely",
        "apply downward pressure onto the crank lever to engage it",
        "depress the mechanical handle straight down to the base",
    ],
    "coffee-button-v3": [
        "tap the espresso maker switch to start brewing",
        "move toward the coffee brewer and activate the brew pad",
        "press the start button located on the espresso machine",
    ],
}

ARM_ORDER: List[str] = [
    "arm1_abk_dog_vision",
    "arm2_frozen_mlp_vision",
    "arm3_dtra_full_vision",
    "arm4_frozen_mlp_multimodal",
    "arm5_dtra_rigid_alpha0",
    "arm6_dtra_full_multimodal",
]

ARM_TITLES: Dict[str, str] = {
    "arm1_abk_dog_vision": "[Arm 1] ABK-VLA (15x15 DoG, Vision-Only)",
    "arm2_frozen_mlp_vision": "[Arm 2] Matched Frozen MLP (Vision-Only, lang=0)",
    "arm3_dtra_full_vision": "[Arm 3] DTRA Full (Vision-Only, lang=0)",
    "arm4_frozen_mlp_multimodal": "[Arm 4] Matched Frozen MLP (Multimodal)",
    "arm5_dtra_rigid_alpha0": "[Arm 5] DTRA Rigid (Multimodal, alpha=0)",
    "arm6_dtra_full_multimodal": "[Arm 6] DTRA Full (Multimodal, alpha=learned)",
}


# ==============================================================================
# 1. 통제 비교군 아키텍처 정의 (모든 모델 ~5.25M 파라미터 & 동일 Conv4/WTA 공유)
# ==============================================================================


class MatchedABK3DPolicy(Autonomous3DABKPolicy):
    """[Arm 1] 파라미터 총량(~5.25M)을 DTRA와 맞춘 Autonomous 3D ABK-VLA 서브클래스.

    기존 `Autonomous3DABKPolicy`의 15x15 DoG+Conv4(7,936-d) 시각 키, O(K) 잔차 사슬,
    신규성 확장 기준(0.996), 코사인 유사도 라우팅을 100% 그대로 유지하고
    슬롯 은닉 차원(slot_hidden_dim=492)만 확장해 K=8 기준 ~5.25M 파라미터 동등성을 맞춥니다.
    """

    def __init__(self, slot_hidden_dim: int = 492, **kwargs) -> None:
        super().__init__(**kwargs)
        self.slot_hidden_dim = slot_hidden_dim

    @torch.no_grad()
    def register_or_expand_slot(
        self, init_imgs: torch.Tensor, novelty_threshold: float = 0.996
    ) -> Tuple[int, bool, float]:
        dev = self.slot_exemplar_keys.device
        ex_imgs = init_imgs[: self.n_exemplars].to(dev)
        if len(ex_imgs) < self.n_exemplars:
            reps = (self.n_exemplars + len(ex_imgs) - 1) // len(ex_imgs)
            ex_imgs = ex_imgs.repeat(reps, 1, 1, 1)[: self.n_exemplars]
        ex_keys = self._compute_key(ex_imgs)  # shape: (M_ex, 7936)

        if len(self.slots) == 0:
            head = MetaWorldBSplineSlotHead(
                256 + 64,
                self.num_cp,
                self.action_dim,
                self.num_modes,
                hidden_dim=self.slot_hidden_dim,
                is_residual=False,
            ).to(dev)
            self.slots.append(head)
            self.slot_exemplar_keys = ex_keys.unsqueeze(0)  # shape: (1, M_ex, 7936)
            self._active_idx = 0
            return 0, True, 0.0

        sims = self._slot_similarities(ex_keys)  # shape: (M_ex, S)
        max_sim = float(sims.max(dim=-1).values.mean().item())
        best_idx = int(sims.mean(dim=0).argmax().item())
        if max_sim < novelty_threshold:
            self.freeze_prior()
            new_idx = len(self.slots)
            head = MetaWorldBSplineSlotHead(
                256 + 64,
                self.num_cp,
                self.action_dim,
                self.num_modes,
                hidden_dim=self.slot_hidden_dim,
                is_residual=True,
            ).to(dev)
            self.slots.append(head)
            self.slot_exemplar_keys = torch.cat([self.slot_exemplar_keys, ex_keys.unsqueeze(0)], dim=0)
            self._active_idx = new_idx
            return new_idx, True, max_sim
        self._active_idx = best_idx
        return best_idx, False, max_sim

    def loss_from_cached_trunk(
        self,
        cond: torch.Tensor,
        base_cp: Optional[torch.Tensor],
        tgt_cp: torch.Tensor,
        tgt_act: torch.Tensor,
    ) -> torch.Tensor:
        """Task 1..7 동결 백본·이전 슬롯 출력(cond, base_cp) 재활용 고속 등가 손실."""
        s_idx = self._active_idx
        B = cond.shape[0]
        head = self.slots[s_idx]
        if s_idx == 0:
            cand_cps, sel = head(cond)  # shape: (B, M, Nk, 4), (B, M)
        else:
            delta_cps, sel = head(cond)
            cand_cps = head.compose_residual(base_cp, delta_cps)

        acts = self.decoder(cand_cps.view(B * self.num_modes, self.num_cp, self.action_dim)).view(
            B, self.num_modes, self.horizon, self.action_dim
        ) * 2.0 - 1.0  # shape: (B, M, H, 4)
        l_act = torch.abs(acts - tgt_act[:, None]).mean(dim=(2, 3))
        l_cp = torch.abs(cand_cps - tgt_cp[:, None]).mean(dim=(2, 3))
        d2 = torch.matmul(self.d2_mat, cand_cps.view(B * self.num_modes, self.num_cp, self.action_dim)).view(
            B, self.num_modes, -1, self.action_dim
        )
        l_sm = (d2 ** 2).sum(dim=-1).mean(dim=-1)
        tot = l_act + 0.5 * l_cp + 0.01 * l_sm  # shape: (B, M)
        best = tot.argmin(dim=1)
        b_idx = torch.arange(B, device=cond.device)
        loss_win = tot[b_idx, best].mean()
        loss_all = tot.mean()
        loss_gate = F.cross_entropy(sel, best)
        return 0.85 * loss_win + 0.05 * loss_all + 0.10 * loss_gate


class MatchedFrozenMLPSlot(nn.Module):
    """[Arm 2 / Arm 4] 하이퍼튜브·위상 투영·탄성 복원 없는 순수 동결 MLP + 6-mode WTA 슬롯.

    입력 [cond(320); x_mag(128)] = 448-d를 3층 LayerNorm+ReLU MLP(hidden_dim=359)로
    인코딩하여 DTRA(5.25M)와 동일한 용량 및 6-mode B-Spline 제어점 출력을 제공합니다.
    """

    def __init__(
        self,
        in_dim: int = 448,
        hidden_dim: int = 359,
        num_cp: int = 6,
        action_dim: int = 4,
        num_modes: int = 6,
    ) -> None:
        super().__init__()
        self.num_cp = num_cp
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pts_head = nn.Linear(hidden_dim, num_modes * num_cp * action_dim)
        self.sel_head = nn.Linear(hidden_dim, num_modes)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """feat: shape (B, 448) -> cand_cps: (B, M, Nk, 4) in (0, 1), sel: (B, M)."""
        B = feat.shape[0]
        h = self.trunk(feat)  # shape: (B, hidden_dim)
        pts = torch.sigmoid(self.pts_head(h)).view(B, self.num_modes, self.num_cp, self.action_dim)
        sel = self.sel_head(h)  # shape: (B, M)
        return pts, sel


class HybridVisualKeyExtractor(nn.Module):
    """저역 통과 필터(3x3 가우시안) + 15x15 부호 DoG 윤곽 + 4x4 전역 문맥 하이브리드 시각 키 추출기.

    Conv4 가중치 변화(T0 학습 드리프트)와 고주파 가우시안 노이즈(sigma=0.20)에 완전히 불변인
    64차원 L2 정규화 시각 키 vis_64를 생성하며, 학습 파라미터 없이 고정 직교 투영 버퍼만 보유합니다.
    Arms 2, 4(Frozen MLP)와 Arms 3, 5, 6(DTRA)이 완전히 동일한 키 공간(Key Parity)을 공유합니다.
    """

    def __init__(self) -> None:
        super().__init__()
        lp_k1 = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32) / 16.0
        self.register_buffer("lp_kernel", lp_k1.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        gen = torch.Generator().manual_seed(20260401)
        q_dog = torch.linalg.qr(torch.randn(3840, 64, generator=gen)).Q
        q_lp4 = torch.linalg.qr(torch.randn(64, 48, generator=gen)).Q.t().contiguous()
        self.register_buffer("q_dog", q_dog)  # shape: (3840, 64)
        self.register_buffer("q_lp4", q_lp4)  # shape: (48, 64)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, 3, H, W) in [0, 1] -> vis_64: (B, 64) unit norm."""
        imgs_lp = F.conv2d(imgs, self.lp_kernel, padding=1, groups=3)
        blur15 = F.avg_pool2d(imgs_lp, kernel_size=15, stride=1, padding=7)
        dog = imgs_lp - blur15

        p32 = F.adaptive_avg_pool2d(dog, (32, 32)).flatten(1)  # shape: (B, 3072)
        p16 = F.adaptive_avg_pool2d(dog, (16, 16)).flatten(1)  # shape: (B, 768)
        dog_vec = torch.cat([p32, p16], dim=-1)
        dog_vec = F.normalize(dog_vec - dog_vec.mean(dim=-1, keepdim=True), p=2, dim=-1)
        dog_64 = F.normalize(torch.matmul(dog_vec, self.q_dog), p=2, dim=-1)

        lp4 = F.adaptive_avg_pool2d(imgs_lp, (4, 4)).flatten(1)  # shape: (B, 48)
        lp4 = F.normalize(lp4 - lp4.mean(dim=-1, keepdim=True), p=2, dim=-1)
        lp4_64 = F.normalize(torch.matmul(lp4, self.q_lp4), p=2, dim=-1)
        return F.normalize(0.60 * dog_64 + 0.80 * lp4_64, p=2, dim=-1)


def _orthogonalize_key_against_bank(
    ex_keys: torch.Tensor,
    prev_bank: Optional[torch.Tensor],
    vis_strength: float = 0.20,
    lang_strength: float = 1.00,
    min_margin: float = 0.08,
) -> torch.Tensor:
    """기존 동결 튜브 키 방향과의 간섭 성분을 마진 보존 범위 내에서 직교 투영으로 제거합니다."""
    proto = F.normalize(ex_keys.mean(dim=0), p=2, dim=-1)
    if prev_bank is None or prev_bank.numel() == 0:
        return proto
    bu = F.normalize(prev_bank, p=2, dim=-1)

    def _proj_sub(vec: torch.Tensor, bank_slice: torch.Tensor) -> torch.Tensor:
        if bank_slice.abs().max() < 1e-6 or vec.norm() < 1e-6:
            return torch.zeros_like(vec)
        bn = F.normalize(bank_slice, p=2, dim=-1)
        u, s, _ = torch.linalg.svd(bn.t(), full_matrices=False)
        q_basis = u[:, s > 1e-6]
        if q_basis.numel() == 0:
            return torch.zeros_like(vec)
        return torch.matmul(q_basis, torch.matmul(q_basis.t(), vec))

    par_vis = _proj_sub(proto[:64], bu[:, :64])
    par_lang = _proj_sub(proto[64:], bu[:, 64:])
    par_full = torch.cat([vis_strength * par_vis, lang_strength * par_lang], dim=-1)
    if par_full.norm() < 1e-6:
        return proto

    prev_scores = torch.matmul(ex_keys, bu.t()).max(dim=-1).values

    def _keeps_margin(gamma: float) -> bool:
        cand = F.normalize(proto - gamma * par_full, p=2, dim=-1)
        return float((torch.matmul(ex_keys, cand) - prev_scores).min().item()) >= min_margin

    if not _keeps_margin(0.0):
        return proto
    if _keeps_margin(1.0):
        return F.normalize(proto - par_full, p=2, dim=-1)
    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if _keeps_margin(mid):
            lo = mid
        else:
            hi = mid
    return F.normalize(proto - lo * par_full, p=2, dim=-1)


class ManifoldAnchorHead(nn.Module):
    """Stage 1 순수 매니폴드 기반 전역 앵커 헤드(원본 Conv4 특징 완전 분리)."""

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 284,
        num_cp: int = 6,
        action_dim: int = 4,
        d_key: int = 128,
    ) -> None:
        super().__init__()
        self.num_cp = num_cp
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.anchor_out = nn.Linear(hidden_dim, num_cp * action_dim)
        self.ctx_out = nn.Linear(hidden_dim, d_key)

    def forward(self, z_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """z_feat: (B, 128) -> p_anchor: (B, Nk, 4) in (0, 1), ctx_coarse: (B, 128) unit norm."""
        h = self.net(z_feat)
        p_anchor = torch.sigmoid(self.anchor_out(h)).view(z_feat.shape[0], self.num_cp, self.action_dim)
        ctx_coarse = F.normalize(self.ctx_out(h), p=2, dim=-1)
        return p_anchor, ctx_coarse


class DenoisingFineHead(nn.Module):
    """Stage 2 매니폴드 뼈대(P_manifold) + 유계 Conv4 잔차(res_cps) 분리 헤드."""

    def __init__(
        self,
        man_in: int = 320,
        res_in: int = 384,
        man_h: int = 240,
        res_h: int = 192,
        num_modes: int = 6,
        num_cp: int = 6,
        action_dim: int = 4,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.num_cp = num_cp
        self.action_dim = action_dim
        out_cp = num_modes * num_cp * action_dim
        self.man_net = nn.Sequential(
            nn.Linear(man_in, man_h),
            nn.LayerNorm(man_h),
            nn.SiLU(),
            nn.Linear(man_h, man_h),
            nn.LayerNorm(man_h),
            nn.SiLU(),
        )
        self.man_pts = nn.Linear(man_h, out_cp)
        self.man_sel = nn.Linear(man_h, num_modes)
        self.res_net = nn.Sequential(
            nn.Linear(res_in, res_h),
            nn.LayerNorm(res_h),
            nn.SiLU(),
            nn.Linear(res_h, res_h),
            nn.LayerNorm(res_h),
            nn.SiLU(),
        )
        self.res_pts = nn.Linear(res_h, out_cp)

    def forward(
        self,
        z_dock_f: torch.Tensor,
        h_avg_f: torch.Tensor,
        ctx_coarse: torch.Tensor,
        f_st: torch.Tensor,
        x_conv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """매니폴드 제어점 뼈대(raw_man_cps), Conv4 유계 잔차(raw_res_cps), 모드 게이트(sel) 반환."""
        bsz = z_dock_f.shape[0]
        man_feat = torch.cat([z_dock_f, h_avg_f, ctx_coarse, f_st], dim=-1)
        hm = self.man_net(man_feat)
        raw_man_cps = self.man_pts(hm).view(bsz, self.num_modes, self.num_cp, self.action_dim)
        sel = self.man_sel(hm)

        res_feat = torch.cat([x_conv, f_st, z_dock_f], dim=-1)
        hr = self.res_net(res_feat)
        raw_res_cps = self.res_pts(hr).view(bsz, self.num_modes, self.num_cp, self.action_dim)
        return raw_man_cps, raw_res_cps, sel


class MatchedFrozenMLPPolicy(nn.Module):
    """[Arm 2 / Arm 4] DTRA와 백본·키·파라미터 총량(~5.25M)을 완벽히 맞춘 동결 MLP 대조군.

    DTRA와 비교 시 순수하게 다음 4가지 기하·동역학 요소만 제거됩니다:
      - 단위 초구(S^63) 초입체 튜브 링 및 1차 타원 주기 기저
      - 2단계 Coarse-to-Fine 기하학적 도킹
      - 유계 탄성 변형(alpha) 및 추론 후 snap_back() 복원
      - 에피소드 관성(inertia) 및 위상/겹침 정규화 L_topo
    추론 시에는 동일한 쿼리 키 x_mag와 슬롯 프로토타입 키 간 코사인 유사도 argmax로
    Task-ID 없이 단 1개의 슬롯만 O(1) 실행합니다.
    """

    def __init__(
        self,
        vision_only: bool = False,
        novelty_threshold: float = 0.96,
        num_cp: int = 6,
        horizon: int = 16,
        action_dim: int = 4,
        feat_dim: int = 256,
        d_key: int = 128,
        num_modes: int = 6,
        slot_hidden_dim: int = 359,
    ) -> None:
        super().__init__()
        self.vision_only = vision_only
        self.novelty_threshold = novelty_threshold
        self.num_cp = num_cp
        self.horizon = horizon
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.d_key = d_key
        self.num_modes = num_modes
        self.slot_hidden_dim = slot_hidden_dim

        # DTRA와 완전히 동일한 Conv4 공간 인코더 + EEF state_net + 직교 시각 사영 + 텍스트 인코더
        self.encoder = build_rgb_spatial_encoder(feat_dim=feat_dim)
        self.state_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.text_encoder = LightweightTextEncoder(d_lang=64)
        self.visual_key_extractor = HybridVisualKeyExtractor()

        self.slots = nn.ModuleList()
        self.register_buffer("slot_keys", torch.empty(0, d_key))
        self.decoder = BSplineDecoder(num_control_points=num_cp, horizon=horizon, degree=3)
        _, d2_np = build_difference_matrices(num_cp)
        self.register_buffer("d2_mat", torch.tensor(d2_np, dtype=torch.float32))
        self._frozen_trunk = False
        self._active_idx = -1

    def reset(self) -> None:
        """추론 에피소드 인터페이스 통일용 no-op."""
        return None

    def _compose_key(self, vis_64: torch.Tensor, instructions: List[str], device: torch.device) -> torch.Tensor:
        """Vision-Only(lang=0) 또는 Multimodal([0.55*vis; 0.835*lang]) 자성 쿼리 생성."""
        if self.vision_only:
            # 순수 시각 조건: 텍스트 차원을 완전 0으로 마스킹하고 vis_64만 L2 정규화 -> shape: (B, 128)
            return F.normalize(torch.cat([vis_64, torch.zeros_like(vis_64)], dim=-1), p=2, dim=-1)
        lang_64 = self.text_encoder(instructions, device=device)  # shape: (B, 64)
        return F.normalize(torch.cat([0.55 * vis_64, 0.835 * lang_64], dim=-1), p=2, dim=-1)

    @torch.no_grad()
    def encode_visual_and_key(self, imgs: torch.Tensor, instructions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        was_tr = self.encoder.training
        self.encoder.eval()
        x = self.encoder(imgs)  # shape: (B, 256)
        if was_tr and not self._frozen_trunk:
            self.encoder.train()
        vis_64 = self.visual_key_extractor(imgs)  # shape: (B, 64)
        x_mag = self._compose_key(vis_64, instructions, imgs.device)
        return x, x_mag

    def _forward_trunk_with_grad(
        self, imgs: torch.Tensor, st: torch.Tensor, instructions: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(imgs)  # shape: (B, 256)
        f_st = self.state_net(st)  # shape: (B, 64)
        cond = torch.cat([x, f_st], dim=1)  # shape: (B, 320)
        vis_64 = self.visual_key_extractor(imgs)  # shape: (B, 64)
        x_mag = self._compose_key(vis_64, instructions, imgs.device)  # shape: (B, 128)
        return cond, x_mag

    def freeze_prior(self) -> None:
        self._frozen_trunk = True
        for mod in (self.encoder, self.state_net):
            for p in mod.parameters():
                p.requires_grad = False
            mod.eval()
        for sl in self.slots:
            for p in sl.parameters():
                p.requires_grad = False
            sl.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._frozen_trunk:
            self.encoder.eval()
            self.state_net.eval()
            for idx, sl in enumerate(self.slots):
                if idx != self._active_idx:
                    sl.eval()
        return self

    @torch.no_grad()
    def register_or_expand_task(self, init_imgs: torch.Tensor, instruction: str) -> Dict[str, object]:
        dev = next(self.encoder.parameters()).device
        ex_imgs = init_imgs[:8].to(dev)
        _, ex_keys = self.encode_visual_and_key(ex_imgs, [instruction] * len(ex_imgs))  # shape: (M_ex, 128)
        proto_key = F.normalize(ex_keys.mean(dim=0), p=2, dim=-1)  # shape: (128,)

        if len(self.slots) > 0:
            self.freeze_prior()
            sims = torch.matmul(ex_keys, self.slot_keys.t()).mean(dim=0)  # shape: (K,)
            max_sim = float(sims.max().item())
            best_idx = int(sims.argmax().item())
        else:
            max_sim, best_idx = 0.0, 0

        if len(self.slots) == 0 or max_sim < self.novelty_threshold:
            slot = MatchedFrozenMLPSlot(
                in_dim=320 + self.d_key,
                hidden_dim=self.slot_hidden_dim,
                num_cp=self.num_cp,
                action_dim=self.action_dim,
                num_modes=self.num_modes,
            ).to(dev)
            self.slots.append(slot)
            self.slot_keys = (
                proto_key.unsqueeze(0)
                if self.slot_keys.numel() == 0
                else torch.cat([self.slot_keys, proto_key.unsqueeze(0)], dim=0)
            )
            idx = len(self.slots) - 1
            expanded = True
        else:
            idx = best_idx
            expanded = False
            for p in self.slots[idx].parameters():
                p.requires_grad = True
            self.slots[idx].train()

        self._active_idx = idx
        return {
            "slot_idx": idx,
            "expanded": expanded,
            "max_sim": max_sim,
            "would_expand_at_096": bool(len(self.slots) == 1 or max_sim < 0.96),
            "total_slots": len(self.slots),
        }

    def loss_from_features(
        self, cond: torch.Tensor, x_mag: torch.Tensor, tgt_cp: torch.Tensor, tgt_act: torch.Tensor
    ) -> torch.Tensor:
        B = cond.shape[0]
        feat = torch.cat([cond, x_mag], dim=-1)  # shape: (B, 448)
        cand_cps, sel = self.slots[self._active_idx](feat)  # shape: (B, M, Nk, 4), (B, M)
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
        b_idx = torch.arange(B, device=cond.device)
        return 0.85 * tot[b_idx, best].mean() + 0.05 * tot.mean() + 0.10 * F.cross_entropy(sel, best)

    def loss(
        self, img: torch.Tensor, st: torch.Tensor, tgt_cp: torch.Tensor, tgt_act: torch.Tensor, instruction: str
    ) -> torch.Tensor:
        cond, x_mag = self._forward_trunk_with_grad(img, st, [instruction] * img.shape[0])
        return self.loss_from_features(cond, x_mag, tgt_cp, tgt_act)

    @torch.no_grad()
    def predict(self, img: torch.Tensor, st: torch.Tensor, instruction: str) -> Tuple[np.ndarray, Dict[str, int]]:
        B = img.shape[0]
        cond, x_mag = self._forward_trunk_with_grad(img, st, [instruction] * B)
        x_hat = F.normalize(x_mag, p=2, dim=-1)  # shape: (B, 128)
        sims = torch.matmul(x_hat, self.slot_keys.t())  # shape: (B, K) — O(K) 키 검색
        best_idx = int(sims[0].argmax().item())
        feat = torch.cat([cond, x_mag], dim=-1)  # shape: (B, 448)
        cand_cps, sel = self.slots[best_idx](feat)  # O(1) 단일 슬롯 실행
        best_mode = sel.argmax(dim=1)
        p_star = cand_cps[torch.arange(B, device=img.device), best_mode].clamp(0.0, 1.0)
        act = self.decoder(p_star) * 2.0 - 1.0
        return act[0].detach().cpu().numpy(), {"slot_idx": best_idx, "fine_idx": best_idx}


class ControlledDTRAPolicy(DTRA3DPolicy):
    """[Arm 3 / Arm 5 / Arm 6] DTRA v2.1의 과학적 인과 분리 및 실측 텔레메트리 서브클래스.

    지원 모드:
      - vision_only=True ([Arm 3]): 텍스트 임베딩을 0으로 차단(lang=0), 순수 vis_64로만
        초입체 튜브 라우팅 및 위상 투영 수행.
      - rigid_alpha_zero=True ([Arm 5]): Hypertube 링·2단계 도킹·L_topo·관성은 그대로 유지하되
        탄성 순응도 alpha_i = 0 으로 고정해 강체 투영(delta_z_star = 0, No Snap-Back)만 허용.
      - record_deformation=True ([Arm 3, 5, 6]): 추론 시 실제 _eval_single_active_tube에
        record_deformation=True 와 변위 버퍼(coarse_delta_z / fine_delta_z)를 전달하고,
        snap_back() 호출 직전·직후 변위 노름(||delta_z_star||) 및 잔차를 실측 로깅.
    """

    def __init__(
        self,
        vision_only: bool = False,
        rigid_alpha_zero: bool = False,
        disable_coarse_merge: bool = True,
        coarse_hidden_dim: int = 284,
        fine_man_hidden: int = 240,
        fine_res_hidden: int = 192,
        conv_dropout_p: float = 0.30,
        conv_noise_sigma: float = 0.05,
        res_scale: float = 0.22,
        lambda_key_rep: float = 2.0,
        **kwargs,
    ) -> None:
        if disable_coarse_merge:
            kwargs["coarse_novelty_threshold"] = 1.01
        super().__init__(**kwargs)
        self.vision_only = vision_only
        self.rigid_alpha_zero = rigid_alpha_zero
        self.disable_coarse_merge = disable_coarse_merge
        self.coarse_hidden_dim = coarse_hidden_dim
        self.fine_man_hidden = fine_man_hidden
        self.fine_res_hidden = fine_res_hidden
        self.conv_dropout_p = conv_dropout_p
        self.conv_noise_sigma = conv_noise_sigma
        self.res_scale = res_scale
        self.lambda_key_rep = lambda_key_rep
        self.visual_key_extractor = HybridVisualKeyExtractor()

    def _compose_key(self, vis_64: torch.Tensor, instructions: List[str], device: torch.device) -> torch.Tensor:
        if self.vision_only:
            # 순수 시각 조건(lang=0): 128-d 차원 및 proj_topo 파라미터 수를 동일하게 유지하되 언어 성분 완전 차단
            return F.normalize(torch.cat([vis_64, torch.zeros_like(vis_64)], dim=-1), p=2, dim=-1)
        lang_64 = self.text_encoder(instructions, device=device)  # shape: (B, 64)
        return F.normalize(torch.cat([0.55 * vis_64, 0.835 * lang_64], dim=-1), p=2, dim=-1)

    @torch.no_grad()
    def encode_visual_and_key(self, imgs: torch.Tensor, instructions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        was_tr = self.encoder.training
        self.encoder.eval()
        x = self.encoder(imgs)
        if was_tr and not self._frozen_trunk:
            self.encoder.train()
        vis_64 = self.visual_key_extractor(imgs)
        x_mag = self._compose_key(vis_64, instructions, imgs.device)
        return x, x_mag

    def _forward_trunk_with_grad(
        self, imgs: torch.Tensor, st: torch.Tensor, instructions: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(imgs)
        f_st = self.state_net(st)  # shape: (B, 64)
        cond = torch.cat([x, f_st], dim=1)  # shape: (B, 320)
        vis_64 = self.visual_key_extractor(imgs)  # shape: (B, 64)
        x_mag = self._compose_key(vis_64, instructions, imgs.device)  # shape: (B, 128)
        return cond, x_mag

    def _compute_single_tube_topo_loss(
        self,
        tubes: nn.ModuleList,
        target_idx: int,
        frozen_rings: torch.Tensor,
        frozen_radii: torch.Tensor,
        lambda_r: float,
        lambda_rep: float,
    ) -> torch.Tensor:
        """L_topo 기본 손실에 L_topo 마진(delta=0.08) 기반 키 코사인 반발 손실을 추가합니다."""
        base_loss = super()._compute_single_tube_topo_loss(
            tubes, target_idx, frozen_rings, frozen_radii, lambda_r, lambda_rep
        )
        if target_idx <= 0:
            return base_loss
        delta_margin = 0.08
        k_cur = F.normalize(tubes[target_idx].key, p=2, dim=-1)
        prev_keys = F.normalize(
            torch.stack([tubes[j].key.detach() for j in range(target_idx)], dim=0),
            p=2,
            dim=-1,
        )
        cos_ij = torch.matmul(prev_keys, k_cur)
        key_rep = (F.relu(cos_ij - (1.0 - delta_margin)) ** 2).sum()
        return base_loss + self.lambda_key_rep * key_rep

    @torch.no_grad()
    def register_or_expand_task(self, init_imgs: torch.Tensor, instruction: str) -> Dict[str, object]:
        """Coarse 합병 차단 + 마진 보존 직교 키 초기화 + 매니폴드/잔차 분리 헤드 등록."""
        dev = next(self.encoder.parameters()).device
        ex_imgs = init_imgs[:8].to(dev)
        _, ex_keys = self.encode_visual_and_key(ex_imgs, [instruction] * len(ex_imgs))  # shape: (M_ex, 128)
        lang_strength = 0.20 if self.vision_only else 1.00

        if len(self.coarse_tubes) > 0:
            self.freeze_prior()
            c_bank = F.normalize(torch.stack([t.key for t in self.coarse_tubes], dim=0), p=2, dim=-1)
            c_sims = torch.matmul(ex_keys, c_bank.t()).mean(dim=0)
            c_max_sim = float(c_sims.max().item())
            c_best_idx = int(c_sims.argmax().item())
        else:
            c_bank = None
            c_max_sim, c_best_idx = 0.0, 0

        if len(self.coarse_tubes) == 0 or self.disable_coarse_merge or c_max_sim < self.coarse_novelty_threshold:
            c_init_key = _orthogonalize_key_against_bank(
                ex_keys, c_bank, vis_strength=0.20, lang_strength=lang_strength, min_margin=0.08
            )
            c_head = ManifoldAnchorHead(
                in_dim=self.d_embed * 2,
                hidden_dim=self.coarse_hidden_dim,
                num_cp=self.num_cp,
                action_dim=self.action_dim,
                d_key=self.d_key,
            ).to(dev)
            c_node = HypertubeNode(
                init_key=c_init_key,
                d_key=self.d_key,
                d_topo=self.d_topo,
                d_embed=self.d_embed,
                init_radius=0.45,
                init_alpha=0.78,
                init_tau=0.42,
                head_module=c_head,
            ).to(dev)
            self.coarse_tubes.append(c_node)
            c_idx = len(self.coarse_tubes) - 1
            c_expanded = True
            self._coarse_trainable = True
        else:
            c_idx = c_best_idx
            c_expanded = False
            self._coarse_trainable = False

        empty_buf = torch.empty(0, device=dev)
        _, _, z_dock_c, h_avg_c, _, _ = self._run_tube_stage(
            self.coarse_tubes, ex_keys, empty_buf, empty_buf,
            [c_idx], use_inertia=False, record_deformation=False,
        )
        _, ctx_coarse_ex = self.coarse_tubes[c_idx].head(torch.cat([z_dock_c, h_avg_c], dim=-1))
        ex_fine_keys = F.normalize(ex_keys + 0.35 * ctx_coarse_ex, p=2, dim=-1)

        if len(self.fine_tubes) > 0:
            f_bank = F.normalize(torch.stack([t.key for t in self.fine_tubes], dim=0), p=2, dim=-1)
            f_sims = torch.matmul(ex_fine_keys, f_bank.t()).mean(dim=0)
            f_max_sim = float(f_sims.max().item())
            f_best_idx = int(f_sims.argmax().item())
        else:
            f_bank = None
            f_max_sim, f_best_idx = 0.0, 0

        if len(self.fine_tubes) == 0 or f_max_sim < self.fine_novelty_threshold:
            f_init_key = _orthogonalize_key_against_bank(
                ex_fine_keys, f_bank, vis_strength=0.20, lang_strength=lang_strength, min_margin=0.08
            )
            f_head = DenoisingFineHead(
                man_in=self.d_embed * 2 + self.d_key + 64,
                res_in=self.feat_dim + 64 + self.d_embed,
                man_h=self.fine_man_hidden,
                res_h=self.fine_res_hidden,
                num_modes=self.num_modes,
                num_cp=self.num_cp,
                action_dim=self.action_dim,
            ).to(dev)
            f_node = HypertubeNode(
                init_key=f_init_key,
                d_key=self.d_key,
                d_topo=self.d_topo,
                d_embed=self.d_embed,
                init_radius=0.25,
                init_alpha=0.24,
                init_tau=0.45,
                head_module=f_head,
            ).to(dev)
            self.fine_tubes.append(f_node)
            f_idx = len(self.fine_tubes) - 1
            f_expanded = True
        else:
            f_idx = f_best_idx
            f_expanded = False
            for p in self.fine_tubes[f_idx].parameters():
                p.requires_grad = True
            self.fine_tubes[f_idx].train()

        if self.rigid_alpha_zero:
            # Arm 5(강체 투영): 학습 그래프에서도 omega를 동결해 alpha_i=0 불변 강제
            for node in list(self.coarse_tubes) + list(self.fine_tubes):
                node.omega.requires_grad_(False)

        self._active_coarse_idx = c_idx
        self._active_fine_idx = f_idx
        self._sync_state_buffers(dev)
        self._cache_frozen_rings()
        self.refresh_eval_caches()
        return {
            "coarse_idx": c_idx,
            "coarse_expanded": c_expanded,
            "coarse_max_sim": c_max_sim,
            "fine_idx": f_idx,
            "fine_expanded": f_expanded,
            "fine_max_sim": f_max_sim,
            "total_coarse": len(self.coarse_tubes),
            "total_fine": len(self.fine_tubes),
            "would_expand_fine_at_096": bool(len(self.fine_tubes) == 1 or f_max_sim < 0.96),
        }

    def _run_tube_stage(
        self,
        tubes: nn.ModuleList,
        x_mag: torch.Tensor,
        inertia_buf: torch.Tensor,
        delta_buf: torch.Tensor,
        active_indices: List[int],
        use_inertia: bool,
        record_deformation: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """rigid_alpha_zero=True일 때만 alpha_sel=0 강제, 그 외는 DTRA v2.1과 100% 동일 연산."""
        if not self.rigid_alpha_zero:
            return super()._run_tube_stage(
                tubes, x_mag, inertia_buf, delta_buf, active_indices, use_inertia, record_deformation
            )

        bsz = x_mag.shape[0]
        k_all = len(tubes)
        x_hat = F.normalize(x_mag, p=2, dim=-1)  # shape: (B, D_key)
        c_query = F.normalize(self.proj_topo(x_hat), p=2, dim=-1)  # shape: (B, D_topo)

        k_bank = F.normalize(torch.stack([t.key for t in tubes], dim=0), p=2, dim=-1)
        cos_all = torch.matmul(x_hat, k_bank.t())
        o_prev = inertia_buf.expand(bsz, -1) if (use_inertia and inertia_buf.numel() > 0) else torch.zeros_like(cos_all)
        f_mag_all = cos_all + self.eta * o_prev

        idx_t = torch.tensor(active_indices, device=x_mag.device, dtype=torch.long)
        sel_nodes = [tubes[i] for i in active_indices]
        k_sel = k_bank[idx_t]
        coeffs_sel = torch.stack([n.fourier_coeffs for n in sel_nodes], dim=0)
        r_sel = F.softplus(torch.cat([n.rho for n in sel_nodes], dim=0)) + 1e-3
        tau_sel = F.softplus(torch.cat([n.beta for n in sel_nodes], dim=0))
        embed_sel = torch.stack([n.ring_embed for n in sel_nodes], dim=0)

        margin = f_mag_all[:, idx_t] - tau_sel.unsqueeze(0)
        if self.training:
            a_act = F.relu(margin) + self.train_tail_eps * torch.sigmoid(self.train_tail_slope * margin)
        else:
            a_act = F.relu(margin)
            if (a_act <= 0.0).all():
                a_act = torch.clamp(margin + 0.25, min=0.05)

        center_sel = self.proj_topo(k_sel)
        c_cos = coeffs_sel[:, 1, :]
        c_sin = coeffs_sel[:, 2, :]
        theta_star = torch.atan2(
            torch.einsum("bd,kd->bk", c_query, c_sin),
            torch.einsum("bd,kd->bk", c_query, c_cos) + 1e-8,
        )
        b_star = torch.stack([torch.ones_like(theta_star), torch.cos(theta_star), torch.sin(theta_star)], dim=-1)
        z0_star = F.normalize(
            center_sel.unsqueeze(0) + torch.einsum("bkf,kfd->bkd", b_star, coeffs_sel),
            p=2,
            dim=-1,
        )  # shape: (B, K_act, D_topo)

        # [Arm 5 강체 투영] alpha = 0 -> delta_z_star = 0, z_tilde_star = z0_star (기본 초구 궤도 고정)
        z_tilde_star = z0_star
        cos_contact = F.cosine_similarity(c_query.unsqueeze(1), z_tilde_star, dim=-1).clamp(-1.0, 1.0)
        dist_contact = torch.sqrt(2.0 * (1.0 - cos_contact) + 1e-8)
        overlap_act = torch.tanh(self.overlap_gate_gain * a_act) * torch.sigmoid(
            (r_sel.unsqueeze(0) - dist_contact) / self.overlap_temp
        )
        w_act = overlap_act / (overlap_act.sum(dim=-1, keepdim=True) + 1e-4)
        z_dock = torch.einsum("bk,bkd->bd", w_act, z_tilde_star)
        h_avg = torch.einsum("bk,kd->bd", w_act, embed_sel)

        new_inertia = torch.zeros(1, k_all, device=x_mag.device)
        new_inertia[:, idx_t] = overlap_act[:1].detach()
        return w_act, overlap_act, z_dock, h_avg, f_mag_all, new_inertia

    def _forward_and_loss(
        self,
        cond: torch.Tensor,
        x_mag: torch.Tensor,
        tgt_cp: torch.Tensor,
        tgt_act: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """매니폴드 주도 P_manifold + Conv4 디노이징 유계 잔차 분리 손실."""
        bsz = cond.shape[0]
        x_conv = cond[:, : self.feat_dim]
        f_st = cond[:, self.feat_dim :]

        c_idx = self._active_coarse_idx
        _, _, z_dock_c, h_avg_c, _, _ = self._run_tube_stage(
            self.coarse_tubes, x_mag, self.coarse_inertia, self.coarse_delta_z,
            [c_idx], use_inertia=False, record_deformation=False,
        )
        # Stage 1: 오염 가능한 x_conv 없이 순수 스냅백 위상 좌표(z_dock_c, h_avg_c)만 사용
        c_feat = torch.cat([z_dock_c, h_avg_c], dim=-1)
        if self._coarse_trainable:
            p_anchor, ctx_coarse = self.coarse_tubes[c_idx].head(c_feat)
        else:
            with torch.no_grad():
                p_anchor, ctx_coarse = self.coarse_tubes[c_idx].head(c_feat)

        x_fine = F.normalize(x_mag + 0.35 * ctx_coarse, p=2, dim=-1)
        f_idx = self._active_fine_idx
        _, overlap_f, z_dock_f, h_avg_f, _, _ = self._run_tube_stage(
            self.fine_tubes, x_fine, self.fine_inertia, self.fine_delta_z,
            [f_idx], use_inertia=False, record_deformation=False,
        )

        # 최대 자성 공명도 기반 신뢰도 게이트 clamp(key_cos_max, 0, 1)
        f_bank = F.normalize(
            torch.stack([t.key.detach() for t in self.fine_tubes[: f_idx + 1]], dim=0),
            p=2,
            dim=-1,
        )
        key_cos_max = torch.matmul(F.normalize(x_fine, p=2, dim=-1), f_bank.t()).max(dim=-1).values
        conf_gate = torch.clamp(key_cos_max, 0.0, 1.0)

        # 학습 시 Conv4 특징 드롭아웃(p=0.3) + 약한 가우시안 노이즈(sigma=0.05) 주입
        if self.training and not deterministic:
            x_conv_in = F.dropout(x_conv, p=self.conv_dropout_p, training=True)
            x_conv_in = x_conv_in + self.conv_noise_sigma * torch.randn_like(x_conv_in)
        else:
            x_conv_in = x_conv

        raw_man_cps, raw_res_cps, sel = self.fine_tubes[f_idx].head(
            z_dock_f, h_avg_f, ctx_coarse, f_st, x_conv_in
        )
        delta_man = (0.55 * overlap_f.unsqueeze(-1).unsqueeze(-1)) * torch.tanh(raw_man_cps)
        p_manifold = torch.clamp(p_anchor.unsqueeze(1) + delta_man, 0.0, 1.0)
        gate_scale = (self.res_scale * conf_gate.unsqueeze(-1) * overlap_f).unsqueeze(-1).unsqueeze(-1)
        delta_res = gate_scale * torch.tanh(raw_res_cps)
        cand_cps = torch.clamp(p_manifold + delta_res, 0.0, 1.0)

        acts = self.decoder(cand_cps.view(bsz * self.num_modes, self.num_cp, self.action_dim)).view(
            bsz, self.num_modes, self.horizon, self.action_dim
        ) * 2.0 - 1.0
        l_act = torch.abs(acts - tgt_act[:, None]).mean(dim=(2, 3))
        l_cp = torch.abs(cand_cps - tgt_cp[:, None]).mean(dim=(2, 3))
        d2 = torch.matmul(self.d2_mat, cand_cps.view(bsz * self.num_modes, self.num_cp, self.action_dim)).view(
            bsz, self.num_modes, -1, self.action_dim
        )
        l_sm = (d2 ** 2).sum(dim=-1).mean(dim=-1)
        tot = l_act + 0.5 * l_cp + 0.01 * l_sm

        best = tot.argmin(dim=1)
        b_idx = torch.arange(bsz, device=cond.device)
        loss_win = tot[b_idx, best].mean()
        loss_all = tot.mean()
        loss_gate = F.cross_entropy(sel, best)

        l_topo_f = self._compute_single_tube_topo_loss(
            self.fine_tubes, f_idx, self._frozen_f_rings, self._frozen_f_radii,
            lambda_r=0.06, lambda_rep=0.80,
        )
        l_topo_c = (
            self._compute_single_tube_topo_loss(
                self.coarse_tubes, c_idx, self._frozen_c_rings, self._frozen_c_radii,
                lambda_r=0.015, lambda_rep=0.80,
            )
            if self._coarse_trainable
            else 0.0
        )
        return 0.85 * loss_win + 0.05 * loss_all + 0.10 * loss_gate + 0.05 * (l_topo_f + l_topo_c)

    def loss_from_features(
        self,
        cond: torch.Tensor,
        x_mag: torch.Tensor,
        tgt_cp: torch.Tensor,
        tgt_act: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Task 1..7 동결 트렁크 출력(cond, x_mag) 기반 고속 등가 손실."""
        return self._forward_and_loss(cond, x_mag, tgt_cp, tgt_act, deterministic=deterministic)

    def loss(
        self,
        img: torch.Tensor,
        st: torch.Tensor,
        tgt_cp: torch.Tensor,
        tgt_act: torch.Tensor,
        instruction: str,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Task 0 및 엔드투엔드 평가용 손실(loss_from_features와 100% 동일 연산)."""
        cond, x_mag = self._forward_trunk_with_grad(img, st, [instruction] * img.shape[0])
        return self._forward_and_loss(cond, x_mag, tgt_cp, tgt_act, deterministic=deterministic)

    @torch.no_grad()
    def _eval_single_active_tube(
        self,
        node,
        f_mag_sel: torch.Tensor,
        x_mag: torch.Tensor,
        record_deformation: bool = False,
        delta_buf: Optional[torch.Tensor] = None,
        tube_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """단 1개의 활성 튜브 O(1) 추론 + 실측 변위(delta_z_star, raw pull, activation) 계측."""
        x_hat = F.normalize(x_mag, p=2, dim=-1)  # shape: (B, 128)
        c_query = F.normalize(self.proj_topo(x_hat), p=2, dim=-1)  # shape: (B, 64)
        k_sel = F.normalize(node.key.unsqueeze(0), p=2, dim=-1)  # shape: (1, 128)

        r_sel = F.softplus(node.rho) + 1e-3
        alpha_sel = torch.zeros_like(node.omega) if self.rigid_alpha_zero else torch.sigmoid(node.omega)
        tau_sel = F.softplus(node.beta)

        margin = f_mag_sel - tau_sel.unsqueeze(0)  # shape: (B, 1)
        a_act = F.relu(margin)
        if (a_act <= 0.0).all():
            a_act = torch.clamp(margin + 0.25, min=0.05)

        center_sel = self.proj_topo(k_sel)  # shape: (1, 64)
        coeffs_sel = node.fourier_coeffs.unsqueeze(0)  # shape: (1, 3, 64)
        p_cos = torch.einsum("bd,kd->bk", c_query, coeffs_sel[:, 1, :])
        p_sin = torch.einsum("bd,kd->bk", c_query, coeffs_sel[:, 2, :])
        theta_star = torch.atan2(p_sin, p_cos + 1e-8)
        cos_star, sin_star = torch.cos(theta_star), torch.sin(theta_star)
        b_star = torch.stack([torch.ones_like(cos_star), cos_star, sin_star], dim=-1)  # shape: (B, 1, 3)

        z0_star = F.normalize(
            center_sel.unsqueeze(0) + torch.einsum("bkf,kfd->bkd", b_star, coeffs_sel),
            p=2,
            dim=-1,
        )  # shape: (B, 1, 64)
        pull_vec = c_query.unsqueeze(1) - z0_star  # shape: (B, 1, 64)
        delta_z_star = (alpha_sel.unsqueeze(0) * a_act).unsqueeze(-1) * pull_vec  # shape: (B, 1, 64)
        z_tilde_star = F.normalize(z0_star + delta_z_star, p=2, dim=-1)  # shape: (B, 1, 64) on S^63

        # 텔레메트리 수집용 스칼라 캐싱 (CPU 지연 타이밍에는 사용하지 않음)
        self._last_eval_stats = (
            float(delta_z_star.norm(dim=-1).mean().item()),
            float(pull_vec.norm(dim=-1).mean().item()),
            float(a_act.mean().item()),
            float(alpha_sel.mean().item()),
        )

        if record_deformation and delta_buf is not None and delta_buf.numel() > 0:
            cos_diff = (
                torch.einsum("m,bk->bkm", self.cos_theta, cos_star)
                + torch.einsum("m,bk->bkm", self.sin_theta, sin_star)
            )
            psi = (0.5 * (1.0 + cos_diff)) ** 2  # shape: (B, 1, 48)
            delta_buf[:, tube_idx : tube_idx + 1] = (delta_z_star.unsqueeze(2) * psi.unsqueeze(-1))[:1]

        cos_contact = F.cosine_similarity(c_query.unsqueeze(1), z_tilde_star, dim=-1).clamp(-1.0, 1.0)
        dist_contact = torch.sqrt(2.0 * (1.0 - cos_contact) + 1e-8)
        overlap_act = torch.tanh(self.overlap_gate_gain * a_act) * torch.sigmoid(
            (r_sel.unsqueeze(0) - dist_contact) / self.overlap_temp
        )
        return overlap_act, a_act, z_tilde_star.squeeze(1), node.ring_embed.unsqueeze(0).expand(x_mag.shape[0], -1)

    @torch.no_grad()
    def predict(
        self, img: torch.Tensor, st: torch.Tensor, instruction: str, collect_telemetry: bool = False
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """O(1) 희소 활성 추론 + 실제 record_deformation=True 변위 기록 + snap_back() 실측 복원."""
        bsz = img.shape[0]
        if self._eval_c_keys.shape[0] != len(self.coarse_tubes) or self._eval_f_keys.shape[0] != len(self.fine_tubes):
            self.refresh_eval_caches()

        cond, x_mag = self._forward_trunk_with_grad(img, st, [instruction] * bsz)
        x_conv = cond[:, : self.feat_dim]
        f_st = cond[:, self.feat_dim :]

        # Stage 1: Coarse O(1) 단일 튜브 실행 (record_deformation=True + delta_buf 기록)
        c_idx, f_mag_c = self._select_active_eval_index(
            self._eval_c_keys, self._eval_c_taus, x_mag,
            self.coarse_inertia, self.coarse_prev_gate, self.coarse_myelin,
        )
        overlap_c, a_c, z_dock_c, h_avg_c = self._eval_single_active_tube(
            self.coarse_tubes[c_idx],
            f_mag_c,
            x_mag,
            record_deformation=True,
            delta_buf=self.coarse_delta_z,
            tube_idx=c_idx,
        )
        c_delta_norm, c_pull_norm, c_act_mean, c_alpha_val = self._last_eval_stats

        self.coarse_inertia.zero_()
        self.coarse_inertia[:, c_idx] = overlap_c[0, 0]
        self.coarse_prev_gate.zero_()
        self.coarse_prev_gate[:, c_idx] = a_c[0, 0]
        c_feat = torch.cat([z_dock_c, h_avg_c], dim=-1)
        p_anchor, ctx_coarse = self.coarse_tubes[c_idx].head(c_feat)

        # Stage 2: Fine O(1) 단일 튜브 실행 (record_deformation=True + delta_buf 기록)
        x_fine = F.normalize(x_mag + 0.35 * ctx_coarse, p=2, dim=-1)
        f_idx, f_mag_f = self._select_active_eval_index(
            self._eval_f_keys, self._eval_f_taus, x_fine,
            self.fine_inertia, self.fine_prev_gate, self.fine_myelin,
        )
        overlap_f, a_f, z_dock_f, h_avg_f = self._eval_single_active_tube(
            self.fine_tubes[f_idx],
            f_mag_f,
            x_fine,
            record_deformation=True,
            delta_buf=self.fine_delta_z,
            tube_idx=f_idx,
        )
        f_delta_norm, f_pull_norm, f_act_mean, f_alpha_val = self._last_eval_stats

        self.fine_inertia.zero_()
        self.fine_inertia[:, f_idx] = overlap_f[0, 0]
        self.fine_prev_gate.zero_()
        self.fine_prev_gate[:, f_idx] = a_f[0, 0]

        key_cos_max = torch.matmul(F.normalize(x_fine, p=2, dim=-1), self._eval_f_keys.t()).max(dim=-1).values
        conf_gate = torch.clamp(key_cos_max, 0.0, 1.0)
        raw_man_cps, raw_res_cps, sel = self.fine_tubes[f_idx].head(
            z_dock_f, h_avg_f, ctx_coarse, f_st, x_conv
        )
        best_mode = sel.argmax(dim=1)
        b_arange = torch.arange(bsz, device=img.device)
        delta_man_best = (0.55 * overlap_f.unsqueeze(-1)) * torch.tanh(raw_man_cps[b_arange, best_mode])
        p_manifold_best = torch.clamp(p_anchor + delta_man_best, 0.0, 1.0)
        gate_scale = (self.res_scale * conf_gate.unsqueeze(-1) * overlap_f).unsqueeze(-1)
        delta_res_best = gate_scale * torch.tanh(raw_res_cps[b_arange, best_mode])

        p_star = torch.clamp(p_manifold_best + delta_res_best, 0.0, 1.0)
        act = self.decoder(p_star) * 2.0 - 1.0

        info: Dict[str, object] = {"coarse_idx": c_idx, "fine_idx": f_idx}
        if collect_telemetry:
            pre_c_max = float(self.coarse_delta_z.abs().max().item()) if self.coarse_delta_z.numel() > 0 else 0.0
            pre_f_max = float(self.fine_delta_z.abs().max().item()) if self.fine_delta_z.numel() > 0 else 0.0
            pre_c_l2 = (
                float(self.coarse_delta_z[:, c_idx].norm(dim=-1).max().item())
                if self.coarse_delta_z.numel() > 0
                else 0.0
            )
            pre_f_l2 = (
                float(self.fine_delta_z[:, f_idx].norm(dim=-1).max().item())
                if self.fine_delta_z.numel() > 0
                else 0.0
            )
            self.snap_back()
            post_c_max = float(self.coarse_delta_z.abs().max().item()) if self.coarse_delta_z.numel() > 0 else 0.0
            post_f_max = float(self.fine_delta_z.abs().max().item()) if self.fine_delta_z.numel() > 0 else 0.0
            info.update(
                {
                    "coarse_delta_z_star_norm": c_delta_norm,
                    "fine_delta_z_star_norm": f_delta_norm,
                    "mean_delta_z_star_norm": 0.5 * (c_delta_norm + f_delta_norm),
                    "coarse_pull_norm": c_pull_norm,
                    "fine_pull_norm": f_pull_norm,
                    "coarse_act": c_act_mean,
                    "fine_act": f_act_mean,
                    "coarse_alpha": c_alpha_val,
                    "fine_alpha": f_alpha_val,
                    "pre_snap_coarse_buf_max": pre_c_max,
                    "pre_snap_fine_buf_max": pre_f_max,
                    "pre_snap_buf_max": max(pre_c_max, pre_f_max),
                    "pre_snap_buf_l2_max": max(pre_c_l2, pre_f_l2),
                    "post_snap_coarse_buf_max": post_c_max,
                    "post_snap_fine_buf_max": post_f_max,
                    "post_snap_buf_max": max(post_c_max, post_f_max),
                }
            )
        else:
            self.snap_back()

        return act[0].detach().cpu().numpy(), info


# ==============================================================================
# 2. 고속·등가 학습 및 환경 관리 유틸리티
# ==============================================================================


def train_arm_task_fast(
    model: nn.Module,
    dataset: Dict,
    instruction: str,
    device: torch.device,
    epochs: int = 55,
    lr: float = 1e-3,
    task_idx: int = 0,
    verify_equivalence: bool = True,
) -> Tuple[int, float]:
    """GPU 상주 텐서 슬라이싱 및 Task 1..7 동결 트렁크 캐싱을 이용한 수학적 100% 등가 학습.

    Task 0에서는 Conv4 인코더가 학습되므로 전체 이미지 배치를 매 에폭 통과시키고,
    Task 1..7에서는 동결된 인코더·상태네트·이전 슬롯 출력을 1회 사전 계산해
    에폭당 학습 시간을 단축하되 배치 0에서 전체 loss와 오차 < 1e-5 검증을 수행합니다.
    """
    x_i = (torch.from_numpy(dataset["imgs"]).permute(0, 3, 1, 2).float() / 255.0).to(device)
    x_s = torch.from_numpy(dataset["states"]).to(device)
    y_c = torch.from_numpy(dataset["cps"]).to(device)
    y_a = torch.from_numpy(dataset["acts"]).to(device)
    n_samples = x_i.shape[0]
    batch_size = 64

    use_frozen_cache = task_idx > 0
    cached_t1 = cached_t2 = None

    if use_frozen_cache:
        model.eval()
        t1_list, t2_list = [], []
        with torch.no_grad():
            for s in range(0, n_samples, batch_size):
                bi, bs = x_i[s : s + batch_size], x_s[s : s + batch_size]
                if isinstance(model, MatchedABK3DPolicy):
                    cond = torch.cat([model.encoder(bi), model.state_net(bs)], dim=1)
                    if model._active_idx == 0:
                        b_cp = torch.zeros_like(y_c[s : s + batch_size])
                    else:
                        q = model._compute_key(bi)
                        stacked_prior, w_prior = model._run_prefix(q, cond, upto=model._active_idx)
                        b_cp = torch.sum(w_prior.unsqueeze(-1).unsqueeze(-1) * stacked_prior, dim=1)
                    t1_list.append(cond)
                    t2_list.append(b_cp)
                else:
                    cond, x_mag = model._forward_trunk_with_grad(bi, bs, [instruction] * len(bi))
                    t1_list.append(cond)
                    t2_list.append(x_mag)
        cached_t1 = torch.cat(t1_list, dim=0)
        cached_t2 = torch.cat(t2_list, dim=0)

        if verify_equivalence and n_samples >= 8:
            model.train()
            b0 = slice(0, min(batch_size, n_samples))
            with torch.no_grad():
                if isinstance(model, MatchedABK3DPolicy):
                    cond_chk = torch.cat([model.encoder(x_i[b0]), model.state_net(x_s[b0])], dim=1)
                    assert float((cond_chk - cached_t1[b0]).abs().max().item()) < 1e-4
                    full_l = model.loss(x_i[b0], x_s[b0], y_c[b0], y_a[b0])
                    fast_l = model.loss_from_cached_trunk(cached_t1[b0], cached_t2[b0], y_c[b0], y_a[b0])
                elif isinstance(model, ControlledDTRAPolicy):
                    cond_chk, x_chk = model._forward_trunk_with_grad(x_i[b0], x_s[b0], [instruction] * len(x_i[b0]))
                    assert float((cond_chk - cached_t1[b0]).abs().max().item()) < 1e-4
                    assert float((x_chk - cached_t2[b0]).abs().max().item()) < 1e-4
                    full_l = model.loss(x_i[b0], x_s[b0], y_c[b0], y_a[b0], instruction=instruction, deterministic=True)
                    fast_l = model.loss_from_features(
                        cached_t1[b0], cached_t2[b0], y_c[b0], y_a[b0], deterministic=True
                    )
                else:
                    cond_chk, x_chk = model._forward_trunk_with_grad(x_i[b0], x_s[b0], [instruction] * len(x_i[b0]))
                    assert float((cond_chk - cached_t1[b0]).abs().max().item()) < 1e-4
                    assert float((x_chk - cached_t2[b0]).abs().max().item()) < 1e-4
                    full_l = model.loss(x_i[b0], x_s[b0], y_c[b0], y_a[b0], instruction=instruction)
                    fast_l = model.loss_from_features(cached_t1[b0], cached_t2[b0], y_c[b0], y_a[b0])
                diff = float((full_l - fast_l).abs().item())
                assert diff < 1e-4, f"Frozen cache equivalence violation at task {task_idx}: diff={diff}"

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    t_start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device="cpu").to(device)
        for s in range(0, n_samples, batch_size):
            idx = perm[s : s + batch_size]
            if not use_frozen_cache:
                if isinstance(model, MatchedABK3DPolicy):
                    loss = model.loss(x_i[idx], x_s[idx], y_c[idx], y_a[idx])
                else:
                    loss = model.loss(x_i[idx], x_s[idx], y_c[idx], y_a[idx], instruction=instruction)
            else:
                if isinstance(model, MatchedABK3DPolicy):
                    loss = model.loss_from_cached_trunk(cached_t1[idx], cached_t2[idx], y_c[idx], y_a[idx])
                else:
                    loss = model.loss_from_features(cached_t1[idx], cached_t2[idx], y_c[idx], y_a[idx])

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()
    sync(device)

    if hasattr(model, "refresh_eval_caches"):
        model.refresh_eval_caches()

    del x_i, x_s, y_c, y_a, cached_t1, cached_t2
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    return n_trainable, float(time.perf_counter() - t_start)


class MetaWorldEvalRegistry:
    """평가용 ML1 분할 메타데이터를 캐싱하되 단일 MuJoCo GL 컨텍스트만 순차 사용하도록 관리.

    Windows OpenGL/GLFW 환경에서는 복수의 `render_mode='rgb_array'` 환경을 동시에 열어두면
    렌더링 컨텍스트가 서로 덮어써져 프레임이 훼손되므로, `ml1` 분할 객체만 캐시하고
    평가 호출마다 단일 `env`를 열어 `finally: env.close()`로 즉시 반환합니다.
    """

    def __init__(self, tasks: Sequence[str], ml1_seed: int = 42) -> None:
        self.tasks = list(tasks)
        self.ml1_seed = ml1_seed
        self._ml1: Dict[str, object] = {}
        self.split_diagnostics: Dict[str, Dict] = {}

    def prepare_all(self) -> Dict[str, Dict]:
        """모든 태스크의 train/test 분할 정의 및 목표 좌표 분포 차이를 검사·기록."""
        for name in self.tasks:
            ml1 = metaworld.ML1(name, seed=self.ml1_seed)
            self._ml1[name] = ml1
            env = ml1.train_classes[name]()
            try:
                tr_goals, te_goals = [], []
                for t in ml1.train_tasks:
                    env.set_task(t)
                    env.reset()
                    tr_goals.append(np.array(env._target_pos, dtype=np.float32))
                for t in ml1.test_tasks:
                    env.set_task(t)
                    env.reset()
                    te_goals.append(np.array(env._target_pos, dtype=np.float32))
            finally:
                env.close()

            tr_g = np.stack(tr_goals, axis=0)
            te_g = np.stack(te_goals, axis=0)
            overlap = len(set(map(tuple, np.round(tr_g, 5))) & set(map(tuple, np.round(te_g, 5))))
            self.split_diagnostics[name] = {
                "n_train_tasks": len(ml1.train_tasks),
                "n_test_tasks": len(ml1.test_tasks),
                "train_goal_mean": [round(float(x), 4) for x in tr_g.mean(axis=0)],
                "test_goal_mean": [round(float(x), 4) for x in te_g.mean(axis=0)],
                "train_goal_std": [round(float(x), 4) for x in tr_g.std(axis=0)],
                "test_goal_std": [round(float(x), 4) for x in te_g.std(axis=0)],
                "exact_goal_pose_overlap": int(overlap),
            }
        return self.split_diagnostics

    def get_env_and_tasks(self, task_name: str, split: str, n_eval: int) -> Tuple[object, List]:
        ml1 = self._ml1[task_name]
        cls_map = ml1.train_classes if split == "train_heldout" else ml1.test_classes
        env = cls_map[task_name](render_mode="rgb_array", camera_name="corner", width=128, height=128)
        if split == "train_heldout":
            tasks_slice = ml1.train_tasks[35 : 35 + n_eval]
        elif split == "official_test":
            tasks_slice = ml1.test_tasks[:n_eval]
        else:
            env.close()
            raise ValueError(f"Unknown split: {split}")
        return env, list(tasks_slice)

    def close(self) -> None:
        self._ml1.clear()


# ==============================================================================
# 3. 통합 평가 + Snap-Back 실측 텔레메트리 + CPU 지연 시간 분해
# ==============================================================================


def eval_arm_task(
    model: nn.Module,
    registry: MetaWorldEvalRegistry,
    task_name: str,
    device: torch.device,
    n_eval: int = 15,
    exec_steps: int = 8,
    task_id: Optional[int] = None,
    split: str = "train_heldout",
    routing_mode: Optional[str] = None,
    noise_std: float = 0.0,
    instructions_override: Optional[List[str]] = None,
    collect_telemetry: bool = False,
    seed: int = 42,
) -> Dict:
    """Clean ID / Official Test Split / OOD Severe / Paraphrased / Snap-Back 공용 평가기."""
    env, eval_tasks = registry.get_env_and_tasks(task_name, split=split, n_eval=n_eval)
    model.eval()
    base_instr = TASK_INSTRUCTIONS[task_name]
    rng = torch.Generator().manual_seed(seed + (task_id or 0) * 100)

    succs = 0
    matched_slots = 0

    ep_mean_dz, ep_max_dz = [], []
    chunk_dz_all, chunk_pull_all, chunk_act_all = [], [], []
    pre_snap_abs_all, pre_snap_l2_all, post_snap_abs_all = [], [], []
    coarse_dz_all, fine_dz_all = [], []

    try:
        for ep_i, t in enumerate(eval_tasks):
            instr = (
                instructions_override[ep_i % len(instructions_override)]
                if instructions_override is not None
                else base_instr
            )
            env.set_task(t)
            obs, _ = env.reset()
            f0 = np.flipud(env.render()).copy()
            t_f0 = torch.from_numpy(f0).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            t_f0 = apply_ood_rgb_perturbation(t_f0, noise_std, 1.0, 0.0, rng)

            if isinstance(model, MatchedABK3DPolicy):
                model.reset_cache()
                w = model.set_episode_init_frame(t_f0)[0].detach().cpu().numpy()
                if task_id is not None:
                    matched_slots += int(np.argmax(w) == task_id)
            else:
                model.reset()

            ep_matched = False
            done = False
            cur_ep_dz = []

            for step_idx in range(0, 150, exec_steps):
                fr = np.flipud(env.render()).copy()
                tx = torch.from_numpy(fr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                tx = apply_ood_rgb_perturbation(tx, noise_std, 1.0, 0.0, rng)
                ts = torch.from_numpy(obs[:4].astype(np.float32)).unsqueeze(0).to(device)

                if isinstance(model, MatchedABK3DPolicy):
                    chunk = model.predict(tx, ts, routing_mode=routing_mode)
                elif isinstance(model, ControlledDTRAPolicy):
                    chunk, info_route = model.predict(tx, ts, instruction=instr, collect_telemetry=collect_telemetry)
                    if step_idx == 0 and task_id is not None:
                        ep_matched = (int(info_route["fine_idx"]) == task_id)
                    if collect_telemetry:
                        dz = float(info_route["mean_delta_z_star_norm"])
                        cur_ep_dz.append(dz)
                        chunk_dz_all.append(dz)
                        coarse_dz_all.append(float(info_route["coarse_delta_z_star_norm"]))
                        fine_dz_all.append(float(info_route["fine_delta_z_star_norm"]))
                        chunk_pull_all.append(
                            0.5 * (float(info_route["coarse_pull_norm"]) + float(info_route["fine_pull_norm"]))
                        )
                        chunk_act_all.append(
                            0.5 * (float(info_route["coarse_act"]) + float(info_route["fine_act"]))
                        )
                        pre_snap_abs_all.append(float(info_route["pre_snap_buf_max"]))
                        pre_snap_l2_all.append(float(info_route["pre_snap_buf_l2_max"]))
                        post_snap_abs_all.append(float(info_route["post_snap_buf_max"]))
                else:
                    chunk, info_route = model.predict(tx, ts, instruction=instr)
                    if step_idx == 0 and task_id is not None:
                        ep_matched = (int(info_route["slot_idx"]) == task_id)

                for a in chunk[:exec_steps]:
                    obs, _, _, _, info = env.step(np.clip(a, -1.0, 1.0))
                    if info.get("success", 0) > 0:
                        done = True
                        break
                if done:
                    break

            if not isinstance(model, MatchedABK3DPolicy) and task_id is not None:
                matched_slots += int(ep_matched)
            succs += int(done)
            if collect_telemetry and cur_ep_dz:
                ep_mean_dz.append(float(np.mean(cur_ep_dz)))
                ep_max_dz.append(float(np.max(cur_ep_dz)))
    finally:
        env.close()

    out: Dict[str, object] = {
        "success_rate": float(succs / max(1, len(eval_tasks))),
        "routing_top1_acc": float(matched_slots / max(1, len(eval_tasks))) if task_id is not None else None,
        "n_episodes": len(eval_tasks),
    }
    if collect_telemetry and chunk_dz_all:
        out["telemetry"] = {
            "ep_delta_z_star_mean": float(np.mean(ep_mean_dz)),
            "ep_delta_z_star_max": float(np.max(ep_max_dz)),
            "chunk_delta_z_star_mean": float(np.mean(chunk_dz_all)),
            "chunk_delta_z_star_max": float(np.max(chunk_dz_all)),
            "coarse_delta_z_star_mean": float(np.mean(coarse_dz_all)),
            "coarse_delta_z_star_max": float(np.max(coarse_dz_all)),
            "fine_delta_z_star_mean": float(np.mean(fine_dz_all)),
            "fine_delta_z_star_max": float(np.max(fine_dz_all)),
            "raw_pull_norm_mean": float(np.mean(chunk_pull_all)),
            "raw_pull_norm_max": float(np.max(chunk_pull_all)),
            "gate_act_mean": float(np.mean(chunk_act_all)),
            "pre_snap_buf_abs_mean": float(np.mean(pre_snap_abs_all)),
            "pre_snap_buf_abs_max": float(np.max(pre_snap_abs_all)),
            "pre_snap_buf_l2_max": float(np.max(pre_snap_l2_all)),
            "post_snap_buf_abs_max": float(np.max(post_snap_abs_all)),
        }
    return out


def aggregate_telemetry(per_task_telems: List[Dict]) -> Dict[str, float]:
    """8개 태스크에서 수집한 Snap-Back 텔레메트리를 단일 요약 수치로 집계."""
    if not per_task_telems:
        return {}
    return {
        "ep_delta_z_star_mean": float(np.mean([t["ep_delta_z_star_mean"] for t in per_task_telems])),
        "ep_delta_z_star_max": float(np.max([t["ep_delta_z_star_max"] for t in per_task_telems])),
        "coarse_delta_z_star_mean": float(np.mean([t["coarse_delta_z_star_mean"] for t in per_task_telems])),
        "fine_delta_z_star_mean": float(np.mean([t["fine_delta_z_star_mean"] for t in per_task_telems])),
        "raw_pull_norm_mean": float(np.mean([t["raw_pull_norm_mean"] for t in per_task_telems])),
        "raw_pull_norm_max": float(np.max([t["raw_pull_norm_max"] for t in per_task_telems])),
        "gate_act_mean": float(np.mean([t["gate_act_mean"] for t in per_task_telems])),
        "pre_snap_buf_abs_mean": float(np.mean([t["pre_snap_buf_abs_mean"] for t in per_task_telems])),
        "pre_snap_buf_abs_max": float(np.max([t["pre_snap_buf_abs_max"] for t in per_task_telems])),
        "pre_snap_buf_l2_max": float(np.max([t["pre_snap_buf_l2_max"] for t in per_task_telems])),
        "post_snap_buf_abs_max": float(np.max([t["post_snap_buf_abs_max"] for t in per_task_telems])),
    }


@torch.no_grad()
def measure_cpu_latency_breakdown(
    model_factory,
    datasets: Dict[str, Dict],
    tasks: Sequence[str],
    routing_mode: str = "argmax",
    k_targets: Tuple[int, ...] = (1, 8),
    n_warmup: int = 12,
    n_iters: int = 80,
) -> Dict[str, object]:
    """CPU 엣지 환경에서 K=1 및 K=8 기준 추론 지연 시간을 단계별로 정밀 분해 측정.

    분해 항목:
      1) Trunk + Key Encode (ms): Conv4 공간 특징 + state_net + 도그/텍스트 키 추출 (K와 무관)
      2) Key Search O(K) (ms): 쿼리 키와 K개 저장 키·예시 간 행렬곱 및 관성/게이트 선택
      3) Active Exec O(1) vs O(K) (ms): 선택된 슬롯/튜브 실행 + atan2/변형/snap_back + 스플라인 복원
      4) End-to-End Chunk (ms): model.predict() 전체 1청크 소요 시간
    """
    cpu = torch.device("cpu")
    cpu_model = model_factory().to(cpu)
    cpu_model.eval()
    tx = torch.rand(1, 3, 128, 128, device=cpu)
    ts = torch.rand(1, 4, device=cpu)
    instr = TASK_INSTRUCTIONS[tasks[0]]

    by_k: Dict[str, Dict[str, float]] = {}
    # 지연 측정에서는 정확히 K=1 및 K=8 크기의 키/슬롯/튜브 뱅크를 강제 구성해 아키텍처 스케일링을 엄밀 검증
    if hasattr(cpu_model, "novelty_threshold"):
        cpu_model.novelty_threshold = 1.01
    if hasattr(cpu_model, "coarse_novelty_threshold"):
        cpu_model.coarse_novelty_threshold = 1.01
        cpu_model.fine_novelty_threshold = 1.01

    for k_idx, t_name in enumerate(tasks):
        init_t = torch.from_numpy(datasets[t_name]["init_frames"][:8]).permute(0, 3, 1, 2).float() / 255.0
        if isinstance(cpu_model, MatchedABK3DPolicy):
            cpu_model.register_or_expand_slot(init_t, novelty_threshold=1.01)
            assert len(cpu_model.slots) == k_idx + 1, f"Expected {k_idx + 1} ABK slots, got {len(cpu_model.slots)}"
        else:
            cpu_model.register_or_expand_task(init_t, TASK_INSTRUCTIONS[t_name])
            if isinstance(cpu_model, MatchedFrozenMLPPolicy):
                assert len(cpu_model.slots) == k_idx + 1
            else:
                assert len(cpu_model.coarse_tubes) == k_idx + 1 and len(cpu_model.fine_tubes) == k_idx + 1
        if hasattr(cpu_model, "refresh_eval_caches"):
            cpu_model.refresh_eval_caches()

        k_cur = k_idx + 1
        if k_cur not in k_targets:
            continue

        if isinstance(cpu_model, MatchedABK3DPolicy):
            cpu_model.reset_cache()
            cpu_model.set_episode_init_frame(tx)

            def _run_trunk() -> Tuple[torch.Tensor, torch.Tensor]:
                q = cpu_model._compute_key(tx)
                cond = torch.cat([cpu_model.encoder(tx), cpu_model.state_net(ts)], dim=1)
                return q, cond

            def _run_search(q: torch.Tensor) -> Tuple[torch.Tensor, int]:
                sims = cpu_model._slot_similarities(q)
                w = F.softmax(sims / max(1e-6, cpu_model.temperature), dim=-1)
                limit = int(w.argmax(dim=-1).item()) + 1 if routing_mode == "argmax" else len(cpu_model.slots)
                return w, limit

            def _run_active(q: torch.Tensor, cond: torch.Tensor, w: torch.Tensor, limit: int) -> torch.Tensor:
                stacked, _ = cpu_model._run_prefix(q, cond, upto=limit)
                if routing_mode == "argmax":
                    top = w.argmax(dim=-1)
                    cp = stacked[0:1, top[0]]
                else:
                    cp = torch.sum(w.unsqueeze(-1).unsqueeze(-1) * stacked, dim=1)
                return cpu_model.decoder(cp.clamp(0.0, 1.0)) * 2.0 - 1.0

            def _run_e2e():
                cpu_model.predict(tx, ts, routing_mode=routing_mode)

        elif isinstance(cpu_model, MatchedFrozenMLPPolicy):
            cpu_model.reset()

            def _run_trunk() -> Tuple[torch.Tensor, torch.Tensor]:
                return cpu_model._forward_trunk_with_grad(tx, ts, [instr])

            def _run_search(x_mag: torch.Tensor) -> int:
                x_hat = F.normalize(x_mag, p=2, dim=-1)
                sims = torch.matmul(x_hat, cpu_model.slot_keys.t())
                return int(sims[0].argmax().item())

            def _run_active(cond: torch.Tensor, x_mag: torch.Tensor, idx: int) -> torch.Tensor:
                feat = torch.cat([cond, x_mag], dim=-1)
                cps, sel = cpu_model.slots[idx](feat)
                p = cps[0:1, sel.argmax(dim=1)[0]].clamp(0.0, 1.0)
                return cpu_model.decoder(p) * 2.0 - 1.0

            def _run_e2e():
                cpu_model.predict(tx, ts, instr)

        else:
            cpu_model.reset()

            def _run_trunk() -> Tuple[torch.Tensor, torch.Tensor]:
                return cpu_model._forward_trunk_with_grad(tx, ts, [instr])

            def _run_search(x_mag: torch.Tensor) -> Tuple[int, torch.Tensor, int, torch.Tensor]:
                c_idx, f_mag_c = cpu_model._select_active_eval_index(
                    cpu_model._eval_c_keys, cpu_model._eval_c_taus, x_mag,
                    cpu_model.coarse_inertia, cpu_model.coarse_prev_gate, cpu_model.coarse_myelin,
                )
                f_idx, f_mag_f = cpu_model._select_active_eval_index(
                    cpu_model._eval_f_keys, cpu_model._eval_f_taus, x_mag,
                    cpu_model.fine_inertia, cpu_model.fine_prev_gate, cpu_model.fine_myelin,
                )
                return c_idx, f_mag_c, f_idx, f_mag_f

            def _run_active(
                cond: torch.Tensor, x_mag: torch.Tensor,
                c_idx: int, f_mag_c: torch.Tensor, f_idx: int, f_mag_f: torch.Tensor,
            ) -> torch.Tensor:
                x_conv = cond[:, : cpu_model.feat_dim]
                f_st = cond[:, cpu_model.feat_dim :]
                ov_c, a_c, z_c, h_c = cpu_model._eval_single_active_tube(
                    cpu_model.coarse_tubes[c_idx], f_mag_c, x_mag,
                    record_deformation=True, delta_buf=cpu_model.coarse_delta_z, tube_idx=c_idx,
                )
                p_anc, ctx_c = cpu_model.coarse_tubes[c_idx].head(torch.cat([z_c, h_c], dim=-1))
                x_f = F.normalize(x_mag + 0.35 * ctx_c, p=2, dim=-1)
                ov_f, a_f, z_f, h_f = cpu_model._eval_single_active_tube(
                    cpu_model.fine_tubes[f_idx], f_mag_f, x_f,
                    record_deformation=True, delta_buf=cpu_model.fine_delta_z, tube_idx=f_idx,
                )
                key_cos_max = torch.matmul(x_f, cpu_model._eval_f_keys.t()).max(dim=-1).values
                conf_gate = torch.clamp(key_cos_max, 0.0, 1.0)
                man_cps, res_cps, sel = cpu_model.fine_tubes[f_idx].head(z_f, h_f, ctx_c, f_st, x_conv)
                best_m = sel.argmax(dim=1)[0]
                d_man = (0.55 * ov_f.unsqueeze(-1)) * torch.tanh(man_cps[0:1, best_m])
                p_man = torch.clamp(p_anc + d_man, 0.0, 1.0)
                d_res = (cpu_model.res_scale * conf_gate.unsqueeze(-1) * ov_f).unsqueeze(-1) * torch.tanh(
                    res_cps[0:1, best_m]
                )
                act = cpu_model.decoder(torch.clamp(p_man + d_res, 0.0, 1.0)) * 2.0 - 1.0
                cpu_model.snap_back()
                return act

            def _run_e2e():
                cpu_model.predict(tx, ts, instr, collect_telemetry=False)

        # 워밍업
        for _ in range(n_warmup):
            _run_e2e()

        # 1) Trunk + Key Encode
        t0 = time.perf_counter()
        for _ in range(n_iters):
            out_tr = _run_trunk()
        trunk_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

        # 2) Key Search O(K)
        if isinstance(cpu_model, MatchedABK3DPolicy):
            q_fixed, cond_fixed = out_tr
            t0 = time.perf_counter()
            for _ in range(n_iters):
                w_out, lim_out = _run_search(q_fixed)
            search_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = _run_active(q_fixed, cond_fixed, w_out, lim_out)
            active_ms = (time.perf_counter() - t0) * 1000.0 / n_iters
        elif isinstance(cpu_model, MatchedFrozenMLPPolicy):
            cond_fixed, x_fixed = out_tr
            t0 = time.perf_counter()
            for _ in range(n_iters):
                s_idx = _run_search(x_fixed)
            search_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = _run_active(cond_fixed, x_fixed, s_idx)
            active_ms = (time.perf_counter() - t0) * 1000.0 / n_iters
        else:
            cond_fixed, x_fixed = out_tr
            t0 = time.perf_counter()
            for _ in range(n_iters):
                c_idx, f_mag_c, f_idx, f_mag_f = _run_search(x_fixed)
            search_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = _run_active(cond_fixed, x_fixed, c_idx, f_mag_c, f_idx, f_mag_f)
            active_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

        # 4) End-to-End chunk latency
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _run_e2e()
        e2e_ms = (time.perf_counter() - t0) * 1000.0 / n_iters

        by_k[f"K={k_cur}"] = {
            "trunk_key_ms": float(trunk_ms),
            "key_search_ok_ms": float(search_ms),
            "active_exec_ms": float(active_ms),
            "e2e_chunk_ms": float(e2e_ms),
        }

    del cpu_model
    return by_k


# ==============================================================================
# 4. Arm 생성 및 단일 시드 전체 파이프라인 실행
# ==============================================================================


def build_arm_factory(arm_key: str):
    """6개 대조군 Arm 키에 대응하는 파라미터 일치(~5.25M) 모델 팩토리 반환."""
    if arm_key == "arm1_abk_dog_vision":
        return lambda: MatchedABK3DPolicy(
            slot_hidden_dim=492, num_cp=6, horizon=16, action_dim=4, num_modes=6, temperature=0.003
        )
    if arm_key == "arm2_frozen_mlp_vision":
        # 저역 통과 하이브리드 시각 키 전용(Key Parity): 시각 기반 8슬롯 독립 유지
        return lambda: MatchedFrozenMLPPolicy(
            vision_only=True, novelty_threshold=1.01, num_cp=6, horizon=16, action_dim=4, num_modes=6, slot_hidden_dim=359
        )
    if arm_key == "arm3_dtra_full_vision":
        # 저역 통과 하이브리드 시각 키 전용(lang=0): Coarse/Fine 자동 합병 비활성화로 8개 독립 튜브(5.25M) 유지
        return lambda: ControlledDTRAPolicy(
            vision_only=True,
            rigid_alpha_zero=False,
            disable_coarse_merge=True,
            coarse_novelty_threshold=1.01,
            fine_novelty_threshold=1.01,
        )
    if arm_key == "arm4_frozen_mlp_multimodal":
        return lambda: MatchedFrozenMLPPolicy(
            vision_only=False, novelty_threshold=0.96, num_cp=6, horizon=16, action_dim=4, num_modes=6, slot_hidden_dim=359
        )
    if arm_key == "arm5_dtra_rigid_alpha0":
        return lambda: ControlledDTRAPolicy(
            vision_only=False,
            rigid_alpha_zero=True,
            disable_coarse_merge=True,
            coarse_novelty_threshold=1.01,
            fine_novelty_threshold=0.96,
        )
    if arm_key == "arm6_dtra_full_multimodal":
        return lambda: ControlledDTRAPolicy(
            vision_only=False,
            rigid_alpha_zero=False,
            disable_coarse_merge=True,
            coarse_novelty_threshold=1.01,
            fine_novelty_threshold=0.96,
        )
    raise ValueError(f"Unknown arm key: {arm_key}")


def run_single_arm_for_seed(
    arm_key: str,
    seed: int,
    datasets: Dict[str, Dict],
    tasks: List[str],
    registry: MetaWorldEvalRegistry,
    device: torch.device,
    args,
) -> Dict:
    """단일 시드에서 1개 대조군의 8-태스크 순차 학습 및 4가지 조건 전체 평가 수행."""
    k_total = len(tasks)
    print(f"\n--- {ARM_TITLES[arm_key]} (seed={seed}) ---", flush=True)
    set_seed(seed)
    factory = build_arm_factory(arm_key)
    model = factory().to(device)

    r_mat = np.zeros((k_total, k_total), dtype=np.float32)
    r_route = np.zeros((k_total, k_total), dtype=np.float32)
    r_abk_blend = np.zeros((k_total, k_total), dtype=np.float32) if arm_key == "arm1_abk_dog_vision" else None
    exp_logs = []
    train_times = []

    for s_idx, t_name in enumerate(tasks):
        ds = datasets[t_name]
        init_t = torch.from_numpy(ds["init_frames"][:8]).permute(0, 3, 1, 2).float() / 255.0

        if isinstance(model, MatchedABK3DPolicy):
            idx, exp, msim = model.register_or_expand_slot(init_t, novelty_threshold=0.996)
            exp_logs.append({"step": s_idx, "task": t_name, "slot_idx": idx, "expanded": exp, "max_sim": msim})
            print(f"  [Expand T{s_idx}] {t_name:17s} | max_sim={msim:.5f} | slot={idx} (exp={exp})", flush=True)
        elif isinstance(model, MatchedFrozenMLPPolicy):
            info = model.register_or_expand_task(init_t, TASK_INSTRUCTIONS[t_name])
            exp_logs.append({"step": s_idx, "task": t_name, **info})
            print(
                f"  [Expand T{s_idx}] {t_name:17s} | max_sim={info['max_sim']:.5f} | "
                f"slot={info['slot_idx']} (exp={info['expanded']}, @0.96={info['would_expand_at_096']})",
                flush=True,
            )
        else:
            info = model.register_or_expand_task(init_t, TASK_INSTRUCTIONS[t_name])
            exp_logs.append({"step": s_idx, "task": t_name, **info})
            print(
                f"  [Expand T{s_idx}] {t_name:17s} | C={info['coarse_idx']}(exp={info['coarse_expanded']}, sim={info['coarse_max_sim']:.4f}) | "
                f"F={info['fine_idx']}(exp={info['fine_expanded']}, sim={info['fine_max_sim']:.4f})",
                flush=True,
            )

        _, tr_sec = train_arm_task_fast(
            model, ds, TASK_INSTRUCTIONS[t_name], device, epochs=args.epochs, lr=1e-3, task_idx=s_idx
        )
        train_times.append(tr_sec)

        if s_idx == 0:
            if isinstance(model, MatchedABK3DPolicy):
                model.refresh_slot_0_keys(init_t)
                model.freeze_prior()
            else:
                model.freeze_prior()

        # 엄밀한 8x8 행렬: 단축키/대각선 복사 없이 모든 과거 태스크 e=0..s_idx 실제 재평가
        is_final_step = (s_idx == k_total - 1)
        collect_tel_now = is_final_step and isinstance(model, ControlledDTRAPolicy)
        if is_final_step:
            clean_telems: List[Dict] = []

        for e_idx in range(s_idx + 1):
            e_name = tasks[e_idx]
            if arm_key == "arm1_abk_dog_vision":
                ev_a = eval_arm_task(
                    model, registry, e_name, device, n_eval=args.n_eval, task_id=e_idx,
                    split="train_heldout", routing_mode="argmax", seed=seed,
                )
                ev_b = eval_arm_task(
                    model, registry, e_name, device, n_eval=args.n_eval, task_id=e_idx,
                    split="train_heldout", routing_mode="blend", seed=seed,
                )
                r_mat[e_idx, s_idx] = ev_a["success_rate"]
                r_route[e_idx, s_idx] = ev_a["routing_top1_acc"] or 0.0
                r_abk_blend[e_idx, s_idx] = ev_b["success_rate"]
                print(
                    f"    Eval T{e_idx} {e_name:17s} | Argmax SR={ev_a['success_rate']*100:5.1f}% (Route={ev_a['routing_top1_acc']*100:5.1f}%) | "
                    f"Blend SR={ev_b['success_rate']*100:5.1f}% | train={tr_sec:.1f}s",
                    flush=True,
                )
            else:
                ev = eval_arm_task(
                    model, registry, e_name, device, n_eval=args.n_eval, task_id=e_idx,
                    split="train_heldout", collect_telemetry=collect_tel_now, seed=seed,
                )
                r_mat[e_idx, s_idx] = ev["success_rate"]
                r_route[e_idx, s_idx] = ev["routing_top1_acc"] or 0.0
                if collect_tel_now and "telemetry" in ev:
                    clean_telems.append(ev["telemetry"])
                print(
                    f"    Eval T{e_idx} {e_name:17s} | SR={ev['success_rate']*100:5.1f}% | "
                    f"Route={ev['routing_top1_acc']*100:5.1f}% | train={tr_sec:.1f}s",
                    flush=True,
                )

    total_params = sum(p.numel() for p in model.parameters())
    cl_clean = calc_cl_metrics(r_mat)

    # ------------------------------------------------------------------
    # 최종 K=8 모델에 대한 추가 스트레스 조건 (Test Split / OOD / Paraphrase) + Snap-Back 실측
    # ------------------------------------------------------------------
    collect_tel = isinstance(model, ControlledDTRAPolicy)
    routing_default = "argmax" if arm_key == "arm1_abk_dog_vision" else None

    clean_final_srs = [float(r_mat[i, k_total - 1]) for i in range(k_total)]
    clean_final_rts = [float(r_route[i, k_total - 1]) for i in range(k_total)]

    # B. Official Test Split (ml1.test_tasks)
    test_srs, test_rts = [], []
    abk_blend_test_srs, abk_blend_ood_srs = [], []
    for e_idx, e_name in enumerate(tasks):
        ev = eval_arm_task(
            model, registry, e_name, device, n_eval=args.n_test_eval, task_id=e_idx,
            split="official_test", routing_mode=routing_default, seed=seed + 1000,
        )
        test_srs.append(ev["success_rate"])
        test_rts.append(ev["routing_top1_acc"] or 0.0)
        if arm_key == "arm1_abk_dog_vision":
            ev_bt = eval_arm_task(
                model, registry, e_name, device, n_eval=args.n_test_eval, task_id=e_idx,
                split="official_test", routing_mode="blend", seed=seed + 1000,
            )
            abk_blend_test_srs.append(ev_bt["success_rate"])

    # C. OOD Visual Severe (Gaussian noise sigma=0.20) + Snap-Back Telemetry
    ood_srs, ood_rts, ood_telems = [], [], []
    for e_idx, e_name in enumerate(tasks):
        ev = eval_arm_task(
            model, registry, e_name, device, n_eval=args.n_ood_eval, task_id=e_idx,
            split="train_heldout", routing_mode=routing_default, noise_std=0.20,
            collect_telemetry=collect_tel, seed=seed + 999,
        )
        ood_srs.append(ev["success_rate"])
        ood_rts.append(ev["routing_top1_acc"] or 0.0)
        if "telemetry" in ev:
            ood_telems.append(ev["telemetry"])
        if arm_key == "arm1_abk_dog_vision":
            ev_bo = eval_arm_task(
                model, registry, e_name, device, n_eval=args.n_ood_eval, task_id=e_idx,
                split="train_heldout", routing_mode="blend", noise_std=0.20, seed=seed + 999,
            )
            abk_blend_ood_srs.append(ev_bo["success_rate"])

    # D. Paraphrased Instructions (Arms 4~6)
    para_srs, para_rts = [], []
    if arm_key in ("arm4_frozen_mlp_multimodal", "arm5_dtra_rigid_alpha0", "arm6_dtra_full_multimodal"):
        for e_idx, e_name in enumerate(tasks):
            ev = eval_arm_task(
                model, registry, e_name, device, n_eval=args.n_eval, task_id=e_idx,
                split="train_heldout", instructions_override=PARAPHRASED_INSTRUCTIONS[e_name],
                seed=seed,
            )
            para_srs.append(ev["success_rate"])
            para_rts.append(ev["routing_top1_acc"] or 0.0)

    # E. 학습된 alpha 통계 (DTRA Arms)
    alpha_stats = None
    if isinstance(model, ControlledDTRAPolicy):
        if model.rigid_alpha_zero:
            alpha_stats = {"coarse_alphas": [0.0] * len(model.coarse_tubes), "fine_alphas": [0.0] * len(model.fine_tubes)}
        else:
            alpha_stats = {
                "coarse_alphas": [round(float(torch.sigmoid(n.omega).item()), 4) for n in model.coarse_tubes],
                "fine_alphas": [round(float(torch.sigmoid(n.omega).item()), 4) for n in model.fine_tubes],
            }

    print(
        f"  [Summary {arm_key} s={seed}] Params={total_params:,} | "
        f"Clean SR={cl_clean['avg_final_f1']*100:5.1f}% (FR={cl_clean['avg_forgetting']*100:4.1f}%, BWT={cl_clean['avg_bwt']*100:+5.1f}%) | "
        f"Test={np.mean(test_srs)*100:5.1f}% | OOD={np.mean(ood_srs)*100:5.1f}% (Route={np.mean(ood_rts)*100:5.1f}%)"
        + (f" | Para={np.mean(para_srs)*100:5.1f}% (Route={np.mean(para_rts)*100:5.1f}%)" if para_srs else ""),
        flush=True,
    )

    out_arm: Dict[str, object] = {
        "arm_key": arm_key,
        "title": ARM_TITLES[arm_key],
        "seed": seed,
        "total_params": int(total_params),
        "train_seconds_total": float(sum(train_times)),
        "R": r_mat.tolist(),
        "R_route": r_route.tolist(),
        "clean_id": {
            **cl_clean,
            "avg_route_acc": float(np.mean(clean_final_rts)),
            "per_task_sr": clean_final_srs,
            "per_task_route": clean_final_rts,
        },
        "official_test_split": {
            "avg_sr": float(np.mean(test_srs)),
            "avg_route_acc": float(np.mean(test_rts)),
            "per_task_sr": test_srs,
            "per_task_route": test_rts,
        },
        "ood_severe_sigma020": {
            "avg_sr": float(np.mean(ood_srs)),
            "avg_route_acc": float(np.mean(ood_rts)),
            "per_task_sr": ood_srs,
            "per_task_route": ood_rts,
        },
        "expansion_logs": exp_logs,
        "alpha_stats": alpha_stats,
    }
    if r_abk_blend is not None:
        out_arm["abk_blend_clean"] = {
            "R": r_abk_blend.tolist(),
            **calc_cl_metrics(r_abk_blend),
            "test_avg_sr": float(np.mean(abk_blend_test_srs)),
            "ood_severe_sr": float(np.mean(abk_blend_ood_srs)),
        }
    if para_srs:
        out_arm["paraphrased_instructions"] = {
            "avg_sr": float(np.mean(para_srs)),
            "avg_route_acc": float(np.mean(para_rts)),
            "per_task_sr": para_srs,
            "per_task_route": para_rts,
        }
    if collect_tel:
        out_arm["snapback_telemetry"] = {
            "clean_id": aggregate_telemetry(clean_telems),
            "ood_severe": aggregate_telemetry(ood_telems),
        }

    del model
    gc.collect()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    return out_arm


# ==============================================================================
# 5. 다중 시드 통계 집계 & 논문·README용 Markdown 표 출력
# ==============================================================================


def _fmt_mean_std(vals: List[float], scale: float = 100.0, signed: bool = False) -> str:
    arr = np.array(vals, dtype=np.float64) * scale
    if len(arr) <= 1:
        return f"{arr[0]:+.1f}%" if signed else f"{arr[0]:.1f}%"
    m, s = float(arr.mean()), float(arr.std(ddof=0))
    return (f"{m:+.1f}% ± {s:.1f}" if signed else f"{m:.1f}% ± {s:.1f}")


def aggregate_and_print_tables(results: Dict) -> Dict:
    """전체 시드 결과를 집계하여 논문 및 README에 바로 삽입 가능한 Markdown 표 출력."""
    arm_keys = results["arms"]
    seeds = results["seeds"]
    lat_map = results["cpu_latency_breakdown"]
    agg: Dict[str, Dict] = {}

    for ak in arm_keys:
        runs = [results["per_seed"][str(s)][ak] for s in seeds if ak in results["per_seed"].get(str(s), {})]
        if not runs:
            continue
        clean_srs = [r["clean_id"]["avg_final_f1"] for r in runs]
        clean_frs = [r["clean_id"]["avg_forgetting"] for r in runs]
        clean_bwts = [r["clean_id"]["avg_bwt"] for r in runs]
        clean_rts = [r["clean_id"]["avg_route_acc"] for r in runs]
        test_srs = [r["official_test_split"]["avg_sr"] for r in runs]
        ood_srs = [r["ood_severe_sigma020"]["avg_sr"] for r in runs]
        ood_rts = [r["ood_severe_sigma020"]["avg_route_acc"] for r in runs]
        para_srs = [r["paraphrased_instructions"]["avg_sr"] for r in runs if "paraphrased_instructions" in r]
        para_rts = [r["paraphrased_instructions"]["avg_route_acc"] for r in runs if "paraphrased_instructions" in r]

        entry: Dict[str, object] = {
            "title": ARM_TITLES[ak],
            "total_params": runs[0]["total_params"],
            "clean_avg_sr_mean": float(np.mean(clean_srs)),
            "clean_avg_sr_std": float(np.std(clean_srs)),
            "clean_fr_mean": float(np.mean(clean_frs)),
            "clean_fr_std": float(np.std(clean_frs)),
            "clean_bwt_mean": float(np.mean(clean_bwts)),
            "clean_bwt_std": float(np.std(clean_bwts)),
            "clean_route_mean": float(np.mean(clean_rts)),
            "test_avg_sr_mean": float(np.mean(test_srs)),
            "test_avg_sr_std": float(np.std(test_srs)),
            "ood_severe_sr_mean": float(np.mean(ood_srs)),
            "ood_severe_sr_std": float(np.std(ood_srs)),
            "ood_severe_route_mean": float(np.mean(ood_rts)),
            "paraphrase_sr_mean": float(np.mean(para_srs)) if para_srs else None,
            "paraphrase_sr_std": float(np.std(para_srs)) if para_srs else None,
            "paraphrase_route_mean": float(np.mean(para_rts)) if para_rts else None,
            "cpu_latency": lat_map.get(ak, {}),
        }
        if "snapback_telemetry" in runs[0]:
            tel_agg = {}
            for cond in ("clean_id", "ood_severe"):
                sub_runs = [r["snapback_telemetry"][cond] for r in runs]
                tel_agg[cond] = {k: float(np.mean([sr[k] for sr in sub_runs])) for k in sub_runs[0].keys()}
            entry["snapback_telemetry_mean"] = tel_agg
        agg[ak] = entry

    results["aggregated"] = agg

    # 1. Main Comparison Table (Markdown)
    print("\n" + "=" * 110)
    print(f"### Table 1: Controlled Scientific Ablation (Seeds={seeds}, K=8 Meta-World Tasks, Parameter-Matched ~5.25M)")
    print("=" * 110)
    header = (
        "| Control Arm | Params | Clean Avg SR | FR | BWT | Official Test SR | "
        "OOD Severe SR (sigma=0.20) | Paraphrase SR | Key Search O(K) ms (K=1 -> K=8) | Active Exec O(1)/O(K) ms (K=1 -> K=8) |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    print(header)
    print(sep)
    for ak in arm_keys:
        if ak not in agg:
            continue
        runs = [results["per_seed"][str(s)][ak] for s in seeds if ak in results["per_seed"].get(str(s), {})]
        sr_str = _fmt_mean_std([r["clean_id"]["avg_final_f1"] for r in runs])
        fr_str = _fmt_mean_std([r["clean_id"]["avg_forgetting"] for r in runs])
        bwt_str = _fmt_mean_std([r["clean_id"]["avg_bwt"] for r in runs], signed=True)
        te_str = _fmt_mean_std([r["official_test_split"]["avg_sr"] for r in runs])
        ood_str = _fmt_mean_std([r["ood_severe_sigma020"]["avg_sr"] for r in runs])
        para_runs = [r["paraphrased_instructions"]["avg_sr"] for r in runs if "paraphrased_instructions" in r]
        para_str = _fmt_mean_std(para_runs) if para_runs else "N/A (Vision)"
        lat = lat_map.get(ak, {})
        k1, k8 = lat.get("K=1", {}), lat.get("K=8", {})
        ks_str = f"{k1.get('key_search_ok_ms', 0.0):.3f} -> {k8.get('key_search_ok_ms', 0.0):.3f}"
        ae_str = f"{k1.get('active_exec_ms', 0.0):.2f} -> {k8.get('active_exec_ms', 0.0):.2f}"
        pm_str = f"{agg[ak]['total_params']/1e6:.2f}M"
        print(f"| {ARM_TITLES[ak]} | {pm_str} | {sr_str} | {fr_str} | {bwt_str} | {te_str} | {ood_str} | {para_str} | {ks_str} | {ae_str} |")
        if ak == "arm1_abk_dog_vision" and all("abk_blend_clean" in r for r in runs):
            b_sr = _fmt_mean_std([r["abk_blend_clean"]["avg_final_f1"] for r in runs])
            b_fr = _fmt_mean_std([r["abk_blend_clean"]["avg_forgetting"] for r in runs])
            b_bwt = _fmt_mean_std([r["abk_blend_clean"]["avg_bwt"] for r in runs], signed=True)
            b_te = _fmt_mean_std([r["abk_blend_clean"].get("test_avg_sr", 0.0) for r in runs])
            b_ood = _fmt_mean_std([r["abk_blend_clean"].get("ood_severe_sr", 0.0) for r in runs])
            blat = lat_map.get("arm1_abk_dog_vision_blend", {})
            bk1, bk8 = blat.get("K=1", {}), blat.get("K=8", {})
            bks = f"{bk1.get('key_search_ok_ms', 0.0):.3f} -> {bk8.get('key_search_ok_ms', 0.0):.3f}"
            bae = f"{bk1.get('active_exec_ms', 0.0):.2f} -> {bk8.get('active_exec_ms', 0.0):.2f}"
            print(f"| [Arm 1b-Blend] ABK-VLA (15x15 DoG, Blend Ref) | {pm_str} | {b_sr} | {b_fr} | {b_bwt} | {b_te} | {b_ood} | N/A (Vision) | {bks} | {bae} |")

    # 2. Routing Accuracy Table (Markdown)
    print("\n### Table 2: Task-ID-Free Top-1 Routing Accuracy Breakdown")
    print("| Control Arm | Clean ID RouteAcc | Official Test RouteAcc | OOD Severe RouteAcc | Paraphrase RouteAcc |")
    print("|---|---:|---:|---:|---:|")
    for ak in arm_keys:
        if ak not in agg:
            continue
        runs = [results["per_seed"][str(s)][ak] for s in seeds if ak in results["per_seed"].get(str(s), {})]
        c_r = _fmt_mean_std([r["clean_id"]["avg_route_acc"] for r in runs])
        t_r = _fmt_mean_std([r["official_test_split"]["avg_route_acc"] for r in runs])
        o_r = _fmt_mean_std([r["ood_severe_sigma020"]["avg_route_acc"] for r in runs])
        p_runs = [r["paraphrased_instructions"]["avg_route_acc"] for r in runs if "paraphrased_instructions" in r]
        p_r = _fmt_mean_std(p_runs) if p_runs else "N/A"
        print(f"| {ARM_TITLES[ak]} | {c_r} | {t_r} | {o_r} | {p_r} |")

    # 3. Snap-Back Telemetry Table (Markdown)
    print("\n### Table 3: Empirical Elastic Deformation (||delta_z_star||) & Snap-Back Buffer Residuals")
    print("| Arm | Condition | Ep Mean ||delta_z_star|| | Ep Max ||delta_z_star|| | Raw Pull Mean ||c-z0|| | Pre-Snap Buf Max | Post-Snap Residual |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for ak in ("arm3_dtra_full_vision", "arm5_dtra_rigid_alpha0", "arm6_dtra_full_multimodal"):
        if ak not in agg or "snapback_telemetry_mean" not in agg[ak]:
            continue
        for cond, cname in (("clean_id", "Clean ID"), ("ood_severe", "OOD Severe (0.20)")):
            td = agg[ak]["snapback_telemetry_mean"][cond]
            print(
                f"| {ARM_TITLES[ak]} | {cname} | {td['ep_delta_z_star_mean']:.5f} | "
                f"{td['ep_delta_z_star_max']:.5f} | {td['raw_pull_norm_mean']:.5f} | "
                f"{td['pre_snap_buf_abs_max']:.5f} | {td['post_snap_buf_abs_max']:.1f} |"
            )
    print("=" * 110 + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--n-demos", type=int, default=30)
    parser.add_argument("--n-eval", type=int, default=15)
    parser.add_argument("--n-ood-eval", type=int, default=15)
    parser.add_argument("--n-test-eval", type=int, default=15)
    parser.add_argument("--arms", nargs="+", default=ARM_ORDER)
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(OUT_DIR / "ablate_dtra_scientific_results.json"),
        help="결과 JSON 저장 경로",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    tasks = [name for name, _ in TASK_SPECS_8]

    print("=" * 88)
    print(f"DTRA Controlled Scientific Ablation Benchmark | seeds={args.seeds} | epochs={args.epochs}")
    print(f"Arms: {args.arms}")
    print("=" * 88)

    datasets = {
        name: collect_or_load_task_data(name, pcls, n_episodes=args.n_demos, H=16, Nk=6)
        for name, pcls in TASK_SPECS_8
    }

    registry = MetaWorldEvalRegistry(tasks, ml1_seed=42)
    try:
        split_diag = registry.prepare_all()
        print("\n=== Meta-World ML1 Split Diagnostics (train_tasks vs test_tasks) ===")
        for t_name, sd in split_diag.items():
            print(
                f"  {t_name:17s} | tr_n={sd['n_train_tasks']}, te_n={sd['n_test_tasks']} | "
                f"tr_mean={sd['train_goal_mean']} vs te_mean={sd['test_goal_mean']} | "
                f"exact_goal_overlap={sd['exact_goal_pose_overlap']}"
            )

        # CPU 엣지 지연 시간 분해 측정 (아키텍처별 K=1 vs K=8)
        print("\n=== Measuring CPU Latency Breakdown: Key Search O(K) ms vs Active Execution ms ===")
        lat_map: Dict[str, Dict] = {}
        for ak in args.arms:
            rmode = "argmax" if ak == "arm1_abk_dog_vision" else "o1"
            lat_map[ak] = measure_cpu_latency_breakdown(
                build_arm_factory(ak), datasets, tasks, routing_mode=rmode
            )
            k1, k8 = lat_map[ak]["K=1"], lat_map[ak]["K=8"]
            print(
                f"  {ak:28s} | KeySearch O(K): {k1['key_search_ok_ms']:.3f}->{k8['key_search_ok_ms']:.3f} ms | "
                f"ActiveExec: {k1['active_exec_ms']:.2f}->{k8['active_exec_ms']:.2f} ms | "
                f"E2E Chunk: {k1['e2e_chunk_ms']:.2f}->{k8['e2e_chunk_ms']:.2f} ms",
                flush=True,
            )
        if "arm1_abk_dog_vision" in args.arms:
            lat_map["arm1_abk_dog_vision_blend"] = measure_cpu_latency_breakdown(
                build_arm_factory("arm1_abk_dog_vision"), datasets, tasks, routing_mode="blend"
            )
            b1, b8 = lat_map["arm1_abk_dog_vision_blend"]["K=1"], lat_map["arm1_abk_dog_vision_blend"]["K=8"]
            print(
                f"  {'arm1_abk_blend (reference)':28s} | KeySearch O(K): {b1['key_search_ok_ms']:.3f}->{b8['key_search_ok_ms']:.3f} ms | "
                f"ActiveExec O(K): {b1['active_exec_ms']:.2f}->{b8['active_exec_ms']:.2f} ms | "
                f"E2E Chunk: {b1['e2e_chunk_ms']:.2f}->{b8['e2e_chunk_ms']:.2f} ms",
                flush=True,
            )

        results: Dict = {
            "tasks": tasks,
            "seeds": args.seeds,
            "arms": args.arms,
            "config": {
                "epochs": args.epochs,
                "n_demos": args.n_demos,
                "n_eval": args.n_eval,
                "n_ood_eval": args.n_ood_eval,
                "n_test_eval": args.n_test_eval,
            },
            "instructions": TASK_INSTRUCTIONS,
            "paraphrased_instructions": PARAPHRASED_INSTRUCTIONS,
            "ml1_split_diagnostics": split_diag,
            "cpu_latency_breakdown": lat_map,
            "per_seed": {},
        }

        for seed in args.seeds:
            seed_key = str(seed)
            results["per_seed"][seed_key] = {}
            for ak in args.arms:
                arm_res = run_single_arm_for_seed(ak, seed, datasets, tasks, registry, device, args)
                results["per_seed"][seed_key][ak] = arm_res
                # 시드·암 완료마다 중간 저장 (안전 체크포인트)
                out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        results = aggregate_and_print_tables(results)
        out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SAVED] Scientific ablation results -> {out_json}", flush=True)
    finally:
        registry.close()


if __name__ == "__main__":
    main()
