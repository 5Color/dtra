"""experiments/benchmark_dtra_libero.py

[DTRA-LIBERO] 24GB VRAM 듀얼 카메라 + 고유수용감각 + 언어 멀티모달 기반
7-DoF B-Spline DTRA 지속학습 벤치마크 (LIBERO-Goal & LIBERO-10).

4종 파라미터 동급 비교군:
  1. arm_frozen_mlp   : 동결 멀티모달 슬롯 + Flat 청크 MLP (비-위상, 비-스플라인, 비-탄성)
  2. arm_er_5pct      : 단일 공유 멀티모달 B-Spline 트렁크 + 5% Experience Replay
  3. arm_dtra_rigid   : DTRA 2-Stage Hypertube (alpha=0 강체 프로토타입 고정)
  4. arm_dtra_elastic : DTRA 2-Stage Hypertube (alpha=learned 탄성 변형 + snap_back + M_ij 수초화)

주요 기능:
  - 128x128 듀얼 카메라 (agentview + robot0_eye_in_hand) Frozen ResNet18 + SpatialSoftmax
  - 로봇 고유수용감각 (ee_pos(3) + gripper_qpos(2) + joint_pos(7) = 12-d) + 언어 명령 (128-d)
  - 고속 벡터화 7-DoF Cubic B-Spline (H=16, M=8) 제어점 사전 적합 및 미분 가능 BSplineDecoder
  - RTX 3090 24GB VRAM 활용 스마트 GPU 텐서 캐싱 + batch_size>=128 + bfloat16/float16 AMP
  - 순차 태스크 학습 후 전체 하삼각 R 행렬 평가: Mean SR, FR, BWT, RouteAcc, VRAM Peak, Throughput

사용법:
  python experiments/benchmark_dtra_libero.py --suite libero_goal --max-tasks 5 --epochs 25 --n-eval 5
  python experiments/benchmark_dtra_libero.py --suite libero_10 --max-tasks 5 --epochs 30 --n-eval 5
"""

from __future__ import annotations

import argparse
import copy
import ctypes
from dataclasses import dataclass
import gc
import json
import math
import os
import random
import sys
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if os.name == "nt":
    try:
        ctypes.CDLL("ucrtbase")._setmaxstdio(8192)
    except Exception:
        pass

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError:
    benchmark = None  # type: ignore[assignment]
    get_libero_path = None  # type: ignore[assignment]
    OffScreenRenderEnv = None  # type: ignore[assignment]

from dtra.core.bspline_core import (
    BSplineDecoder,
    build_clamped_knot_vector,
    build_design_matrix,
    build_difference_matrices,
)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_cl_metrics(
    R: np.ndarray, baseline_diag: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """지속학습 표준 지표 산출: Mean Final SR, Forgetting Rate (FR), Backward Transfer (BWT), Forward Transfer (FWT)."""
    K = R.shape[0]
    if K == 1:
        return {"mean_final_sr": float(R[0, 0]), "forgetting_rate": 0.0, "bwt": 0.0, "fwt": 0.0}
    forgets = [max(0.0, float(np.max(R[i, i : K - 1]) - R[i, K - 1])) for i in range(K - 1)]
    bwts = [float(R[i, K - 1] - R[i, i]) for i in range(K - 1)]
    if baseline_diag is not None and len(baseline_diag) >= K:
        fwts = [float(R[i, i] - baseline_diag[i]) for i in range(1, K)]
    else:
        fwts = [float(R[i, i] - R[0, 0]) for i in range(1, K)]
    return {
        "mean_final_sr": float(np.mean(R[:, K - 1])),
        "forgetting_rate": float(np.mean(forgets)) if forgets else 0.0,
        "bwt": float(np.mean(bwts)) if bwts else 0.0,
        "fwt": float(np.mean(fwts)) if fwts else 0.0,
    }


def forward_fill_R_for_milestone(
    R_raw: np.ndarray, measured: np.ndarray, k_end: int
) -> np.ndarray:
    """마일스톤 K=k_end 기준 [0:K, 0:K] 행렬에서 미측정 중간 셀을 직전 측정값으로 보수 채움."""
    sub = R_raw[:k_end, :k_end].copy()
    sub_m = measured[:k_end, :k_end]
    for i in range(k_end):
        last_val = float(sub[i, i])
        for j in range(i + 1, k_end - 1):
            if sub_m[i, j]:
                last_val = float(sub[i, j])
            else:
                sub[i, j] = last_val
    return sub


def save_json_atomic(path: str, payload: dict) -> None:
    """임시 파일 기록 후 교체하여 마일스톤 중간 저장 중 JSON 손상 방지."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(max(float(y), 1e-6)))


def _det_hash(tag: str, val: str) -> int:
    """프로세스 간 완벽히 결정론적인 CRC32 해시."""
    return zlib.crc32(f"{tag}:{val}".encode("utf-8")) & 0xFFFFFFFF


def _orthogonalize_key_against_bank(
    raw_key: torch.Tensor, existing_keys: Sequence[torch.Tensor]
) -> torch.Tensor:
    """태스크별 단위 초구 프로토타입 키 정규화(공유 배경 장면 간섭 방지 위해 쿼리 정렬 유지)."""
    k = F.normalize(raw_key.detach().clone(), dim=-1)
    if not existing_keys:
        return k
    proj = k.clone()
    for ek in existing_keys:
        ek_u = F.normalize(ek.detach(), dim=-1)
        proj = proj - torch.dot(proj, ek_u) * ek_u
    norm = proj.norm()
    if float(norm.item()) < 1e-4:
        return k
    ortho = proj / norm
    return F.normalize(0.95 * k + 0.05 * ortho, dim=-1)


def _refine_key_baseline_standard(
    raw_key: torch.Tensor,
    existing_keys: Sequence[torch.Tensor],
    sample_queries: Optional[torch.Tensor] = None,
    margin: float = 0.08,
    sep_cap: float = 0.72,
    steps: int = 80,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    [Baseline (arm_frozen_mlp) 표준 마진 정제]:
    DTRA v3.0의 침범 금지 제약(No-Encroachment Constraint) 없이
    신규 슬롯 키만 기존 키들로부터 밀어내는 기존 대조 마진 방식.
    고밀도(K=10->30->50->100) 누적 시 보로노이 침범 현상을 대조 검증하기 위해 유지.
    """
    k0 = F.normalize(raw_key.detach().clone().float(), dim=-1)
    q_norm = (
        F.normalize(sample_queries.detach().float(), dim=-1)
        if (sample_queries is not None and sample_queries.numel() > 0)
        else k0.unsqueeze(0)
    )
    if not existing_keys:
        return k0.to(raw_key.dtype), {
            "max_cos_before": 0.0,
            "max_cos_after": 0.0,
            "query_route_acc": 1.0,
        }

    dev = k0.device
    E = F.normalize(torch.stack([ek.detach().to(dev).float() for ek in existing_keys], dim=0), dim=-1)
    q_norm = q_norm.to(dev)
    max_cos_before = float((E @ k0).max().item())
    old_max_per_q = (q_norm @ E.T).max(dim=-1).values

    with torch.enable_grad():
        u = k0.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=0.025)
        for _ in range(steps):
            opt.zero_grad()
            z = F.normalize(u, dim=-1)
            q_sim = q_norm @ z
            l_hinge = F.relu(old_max_per_q + margin - q_sim).pow(2).mean()
            l_sep = F.relu((E @ z) - sep_cap).pow(2).sum()
            l_anc = F.relu(0.88 - (k0 @ z)).pow(2) * 10.0
            loss = 3.0 * l_hinge + 1.5 * l_sep + l_anc
            loss.backward()
            opt.step()

    best_z = F.normalize(u.detach(), dim=-1)
    best_cos = float((E @ best_z).max().item())
    best_acc = float(((q_norm @ best_z) > old_max_per_q).float().mean().item())
    return best_z.to(raw_key.dtype), {
        "max_cos_before": round(max_cos_before, 4),
        "max_cos_after": round(best_cos, 4),
        "query_route_acc": round(best_acc, 4),
    }


def _refine_key_with_contrastive_margin(
    raw_key: torch.Tensor,
    existing_keys: Sequence[torch.Tensor],
    sample_queries: Optional[torch.Tensor] = None,
    margin: float = 0.04,
    margin_safe: float = 0.02,
    sep_cap: float = 0.72,
    steps: int = 50,
    existing_anchors: Optional[Sequence[torch.Tensor]] = None,
    existing_self_sims: Optional[Sequence[float]] = None,
    existing_val_queries: Optional[Sequence[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """침범 금지 제약(No-Encroachment Constraint) 기반 대조 키 마진 정제 (DTRA v3.0).

    1. 각 태스크 j 등록 시 원본 쿼리 평균 앵커 mu_j와 자기 유사도 하한 s_jj = cos(mu_j, k_j)를 캐싱한다.
    2. 신규 키 k_i를 정제할 때 모든 기존 태스크 j < i에 대해:
         - 침범 금지 불변 조건: cos(mu_j, k_i_new) <= cos(mu_j, k_j) - margin_safe
         - 자기 앵커 보존 조건: cos(mu_i, k_i_new) >= cos(mu_i, mu_i_raw) * 0.95
       를 강제한다.
    3. 정제 업데이트가 기존 0..i-1 태스크 중 어느 하나라도 검증 샘플 RouteAcc를 하락시키면
       해당 반발 성분을 직교 투영으로 제거하고 스텝 크기를 축소/롤백한다.
    """
    if sample_queries is not None and sample_queries.numel() > 0:
        q_norm = F.normalize(sample_queries.detach().float(), dim=-1)
        mu_i = F.normalize(q_norm.mean(dim=0), dim=-1)
    else:
        mu_i = F.normalize(raw_key.detach().float().clone(), dim=-1)
        q_norm = mu_i.unsqueeze(0)

    mu_i_raw = F.normalize(raw_key.detach().float().clone(), dim=-1)
    min_self_sim = max(0.95, float(torch.dot(mu_i, mu_i_raw).item()) * 0.95)

    if not existing_keys:
        out_k = mu_i.to(raw_key.dtype)
        out_k._anchor_mu = mu_i.detach()  # type: ignore[attr-defined]
        out_k._self_sim = 1.0  # type: ignore[attr-defined]
        out_k._val_queries = q_norm[:96].detach()  # type: ignore[attr-defined]
        return out_k, {
            "max_cos_before": 0.0,
            "max_cos_after": 0.0,
            "query_route_acc": 1.0,
            "self_sim": 1.0,
        }

    dev = mu_i.device
    E = torch.stack([F.normalize(ek.detach().float(), dim=-1) for ek in existing_keys], dim=0).to(dev)
    if existing_anchors is not None and len(existing_anchors) == len(existing_keys):
        M_anchors = torch.stack(
            [F.normalize(a.detach().float(), dim=-1) for a in existing_anchors], dim=0
        ).to(dev)
    else:
        M_anchors = torch.stack(
            [
                getattr(ek, "_anchor_mu", F.normalize(ek.detach().float(), dim=-1)).to(dev)
                for ek in existing_keys
            ],
            dim=0,
        )

    if existing_self_sims is not None and len(existing_self_sims) == len(existing_keys):
        s_self = torch.tensor(list(existing_self_sims), device=dev, dtype=torch.float32)
    else:
        s_self = torch.tensor(
            [
                float(getattr(ek, "_self_sim", float((M_anchors[j] @ E[j]).item())))
                for j, ek in enumerate(existing_keys)
            ],
            device=dev,
            dtype=torch.float32,
        )

    if existing_val_queries is not None and len(existing_val_queries) == len(existing_keys):
        val_q_list = [vq.to(dev) for vq in existing_val_queries]
    else:
        val_q_list = [
            getattr(ek, "_val_queries", M_anchors[j].unsqueeze(0)).to(dev)
            for j, ek in enumerate(existing_keys)
        ]

    base_val_accs: List[float] = []
    base_val_best_other: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for j, vq in enumerate(val_q_list):
        sims_j = vq @ E.T
        own_s = sims_j[:, j]
        if E.shape[0] > 1:
            mask_other = torch.ones(E.shape[0], dtype=torch.bool, device=dev)
            mask_other[j] = False
            best_other = sims_j[:, mask_other].max(dim=-1).values
        else:
            best_other = torch.full_like(own_s, -1.0)
        base_val_best_other.append((own_s, best_other))
        base_val_accs.append(float((own_s > best_other).float().mean().item()))

    q_norm = q_norm.to(dev)
    s_raw = E @ mu_i
    max_cos_before = float(s_raw.max().item())
    old_max_per_q = (q_norm @ E.T).max(dim=-1).values
    raw_acc = float(((q_norm @ mu_i) > old_max_per_q).float().mean().item())

    # 침범 금지 상한: cos(mu_j, k_i_new) <= cos(mu_j, k_j) - margin_safe
    encroach_bound = s_self - margin_safe

    def _eval_candidate(z_cand: torch.Tensor) -> Tuple[float, float, float, float, float]:
        self_sim_c = float((mu_i @ z_cand).item())
        enc_viol = float(F.relu((M_anchors @ z_cand) - encroach_bound).max().item())
        min_old_acc_diff = 0.0
        for j, vq in enumerate(val_q_list):
            own_s, best_other = base_val_best_other[j]
            new_other = torch.maximum(best_other, vq @ z_cand)
            acc_after = float((own_s > new_other).float().mean().item())
            diff = acc_after - base_val_accs[j]
            if diff < min_old_acc_diff:
                min_old_acc_diff = diff
        cur_acc = float(((q_norm @ z_cand) > old_max_per_q).float().mean().item())
        cur_cos = float((E @ z_cand).max().item())
        return self_sim_c, enc_viol, min_old_acc_diff, cur_acc, cur_cos

    best_z = mu_i.clone()
    _, _, _, best_acc, best_cos = _eval_candidate(best_z)

    with torch.enable_grad():
        u = mu_i.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=0.015)
        for _ in range(steps):
            opt.zero_grad()
            z = F.normalize(u, dim=-1)
            q_sim = q_norm @ z
            l_hinge = F.relu(old_max_per_q + margin - q_sim).pow(2).mean()
            e_sim = E @ z
            m_sim = M_anchors @ z
            l_sep = F.relu(e_sim - sep_cap).pow(2).sum()
            l_no_encroach = F.relu(m_sim - encroach_bound).pow(2).sum() * 40.0
            l_self = F.relu(0.992 - (mu_i @ z)).pow(2) * 50.0
            loss = 4.0 * l_hinge + 1.5 * l_sep + l_no_encroach + l_self
            loss.backward()
            opt.step()

            with torch.no_grad():
                z_c = F.normalize(u, dim=-1)
                delta = z_c - mu_i
                enc_mask = (M_anchors @ z_c) > encroach_bound
                if bool(enc_mask.any().item()):
                    for m_vec in M_anchors[enc_mask]:
                        proj_coeff = torch.dot(delta, m_vec)
                        if float(proj_coeff.item()) > 0.0:
                            delta = delta - proj_coeff * m_vec
                for beta in (1.0, 0.5, 0.25, 0.1):
                    z_try = F.normalize(mu_i + beta * delta, dim=-1)
                    s_sim, enc_v, old_diff, c_acc, c_cos = _eval_candidate(z_try)
                    if (
                        s_sim >= min_self_sim
                        and enc_v <= 1e-5
                        and old_diff >= -1e-6
                        and c_acc >= raw_acc - 1e-6
                    ):
                        if c_cos < best_cos - 1e-4 or (c_acc > best_acc + 1e-4 and c_cos <= best_cos):
                            best_z = z_try.clone()
                            best_cos = c_cos
                            best_acc = c_acc
                            break

    out_k = best_z.to(raw_key.dtype)
    self_sim_final = float((mu_i @ best_z).item())
    out_k._anchor_mu = mu_i.detach()  # type: ignore[attr-defined]
    out_k._self_sim = self_sim_final  # type: ignore[attr-defined]
    out_k._val_queries = q_norm[:96].detach()  # type: ignore[attr-defined]
    return out_k, {
        "max_cos_before": round(max_cos_before, 4),
        "max_cos_after": round(best_cos, 4),
        "query_route_acc": round(best_acc, 4),
        "self_sim": round(self_sim_final, 4),
    }


# ---------------------------------------------------------------------------
# 1. 고속 7-DoF B-Spline 적합 및 LIBERO HDF5 데이터셋 로더
# ---------------------------------------------------------------------------


def build_bspline_ridge_solver(
    horizon: int = 16,
    num_control_points: int = 8,
    degree: int = 3,
    reg_lambda: float = 1e-2,
) -> np.ndarray:
    """Tikhonov 2차 차분 정규화 B-Spline 일괄 최소제곱 역행렬 (M, H) 생성."""
    knots = build_clamped_knot_vector(num_control_points, degree)
    t_eval = np.linspace(0.0, 1.0, horizon)
    b_mat, _ = build_design_matrix(t_eval, num_control_points, degree, knots)
    d2 = build_difference_matrices(num_control_points)[1]
    lhs = b_mat.T @ b_mat + reg_lambda * (d2.T @ d2) + 1e-6 * np.eye(num_control_points)
    solver = np.linalg.solve(lhs, b_mat.T).astype(np.float32)
    return solver  # (M, H)


def fit_bspline_chunks_batch(
    actions_seq: np.ndarray, solver_mat: np.ndarray, clip_bound: float = 1.25
) -> np.ndarray:
    """(N, H, D) 행동 청크 배치를 벡터화 행렬곱으로 (N, M, D) 제어점으로 일괄 변환."""
    ctrl = np.einsum("mh,nhd->nmd", solver_mat, actions_seq.astype(np.float32))
    return np.clip(ctrl, -clip_bound, clip_bound).astype(np.float32)


class LightweightTextEncoder:
    """외부 토크나이저 의존성 없는 결정론적 위치/N-gram 해싱 텍스트 인코더."""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self._cache: Dict[str, torch.Tensor] = {}

    def encode(self, text: str, device: torch.device) -> torch.Tensor:
        if text not in self._cache:
            vec = np.zeros(self.dim, dtype=np.float32)
            cleaned = f" {text.lower().strip()} "
            words = cleaned.split()
            for pos, w in enumerate(words):
                idx = _det_hash("w", w) % self.dim
                vec[idx] += 2.0
                p_idx = _det_hash("wp", f"{pos}:{w}") % self.dim
                p_sign = 1.0 if (_det_hash("wps", f"{pos}:{w}") % 2 == 0) else -1.0
                vec[p_idx] += p_sign * 1.2
            for bi in range(len(words) - 1):
                bg = f"{words[bi]}_{words[bi + 1]}"
                b_idx = _det_hash("wb2", bg) % self.dim
                b_sign = 1.0 if (_det_hash("wb2s", bg) % 2 == 0) else -1.0
                vec[b_idx] += b_sign * 1.5
            for ti in range(len(words) - 2):
                tg = f"{words[ti]}_{words[ti + 1]}_{words[ti + 2]}"
                t_idx = _det_hash("wb3", tg) % self.dim
                t_sign = 1.0 if (_det_hash("wb3s", tg) % 2 == 0) else -1.0
                vec[t_idx] += t_sign * 1.2
            for n in (2, 3, 4):
                for i in range(max(0, len(cleaned) - n + 1)):
                    gram = cleaned[i : i + n]
                    idx = _det_hash(f"c{n}", gram) % self.dim
                    sign = 1.0 if (_det_hash("cs", gram) % 2 == 0) else -1.0
                    vec[idx] += sign * 0.5
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec /= norm
            self._cache[text] = torch.from_numpy(vec)
        return self._cache[text].to(device)


class LiberoTaskDataset:
    """단일 LIBERO 태스크 HDF5 시연 데이터를 (듀얼 카메라 + 고유수용 + 7-DoF 청크)로 로드."""

    def __init__(
        self,
        task_idx: int,
        task_name: str,
        language: str,
        hdf5_path: str,
        horizon: int = 16,
        num_cp: int = 8,
        stride: int = 1,
        max_demos: Optional[int] = None,
        cache_on_gpu: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        self.task_idx = task_idx
        self.task_name = task_name
        self.language = language
        self.hdf5_path = hdf5_path
        self.horizon = horizon
        self.num_cp = num_cp

        solver = build_bspline_ridge_solver(horizon=horizon, num_control_points=num_cp)
        decoder_b, _ = build_design_matrix(np.linspace(0.0, 1.0, horizon), num_cp, 3)
        decoder_b = decoder_b.astype(np.float32)

        agent_list, eih_list, prop_list, act_list, seq_next_list = [], [], [], [], []
        offset = 0

        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(list(f["data"].keys()))
            if max_demos is not None:
                demos = demos[:max_demos]
            self.num_demos = len(demos)

            for dname in demos:
                grp = f["data"][dname]
                obs = grp["obs"]
                av = obs["agentview_rgb"][:]  # (T, H, W, 3) uint8
                eih = obs["eye_in_hand_rgb"][:]  # (T, H, W, 3) uint8
                ee_pos = obs["ee_pos"][:].astype(np.float32)
                grip = obs["gripper_states"][:].astype(np.float32)
                jnt = obs["joint_states"][:].astype(np.float32)
                proprio = np.concatenate([ee_pos, grip, jnt], axis=-1)  # (T, 12)
                actions = grp["actions"][:].astype(np.float32)  # (T, 7)

                T = len(actions)
                if T < 4:
                    continue

                # 에피소드 종료 직전 정지-그리퍼 홀드 패딩 (tail horizon // 2 활용)
                tail_hold = np.zeros((horizon, 7), dtype=np.float32)
                tail_hold[:, 6] = actions[-1, 6]
                padded_act = np.concatenate([actions, tail_hold], axis=0)

                idxs = list(range(0, max(1, T - 2), stride))
                chunks = np.stack([padded_act[t : t + horizon] for t in idxs], axis=0)

                agent_list.append(np.transpose(av[idxs], (0, 3, 1, 2)).copy())
                eih_list.append(np.transpose(eih[idxs], (0, 3, 1, 2)).copy())
                prop_list.append(proprio[idxs])
                act_list.append(chunks)

                n_ep = len(idxs)
                nxt = np.arange(offset, offset + n_ep, dtype=np.int64)
                nxt[:-1] = nxt[:-1] + 1
                seq_next_list.append(nxt)
                offset += n_ep

        agent_arr = np.concatenate(agent_list, axis=0)
        eih_arr = np.concatenate(eih_list, axis=0)
        prop_arr = np.concatenate(prop_list, axis=0)
        act_arr = np.concatenate(act_list, axis=0)
        ctrl_arr = fit_bspline_chunks_batch(act_arr, solver)
        recon_arr = np.einsum("hm,nmd->nhd", decoder_b, ctrl_arr)
        self.bspline_fit_rmse = float(np.sqrt(np.mean((recon_arr - act_arr) ** 2)))

        self.num_samples = len(act_arr)
        self.agent_u8 = torch.from_numpy(agent_arr)
        self.eih_u8 = torch.from_numpy(eih_arr)
        self.proprio = torch.from_numpy(prop_arr)
        self.actions_gt = torch.from_numpy(act_arr)
        self.ctrl_gt = torch.from_numpy(ctrl_arr)
        self.seq_next = torch.from_numpy(np.concatenate(seq_next_list, axis=0))

        self.data_bytes = (
            self.agent_u8.numel()
            + self.eih_u8.numel()
            + self.proprio.numel() * 4
            + self.actions_gt.numel() * 4
            + self.ctrl_gt.numel() * 4
        )
        self.storage_device = device if (cache_on_gpu and device.type == "cuda") else torch.device("cpu")
        if self.storage_device.type == "cuda":
            self.agent_u8 = self.agent_u8.to(device, non_blocking=True)
            self.eih_u8 = self.eih_u8.to(device, non_blocking=True)
            self.proprio = self.proprio.to(device, non_blocking=True)
            self.actions_gt = self.actions_gt.to(device, non_blocking=True)
            self.ctrl_gt = self.ctrl_gt.to(device, non_blocking=True)
        else:
            self.agent_u8 = self.agent_u8.pin_memory()
            self.eih_u8 = self.eih_u8.pin_memory()
            self.proprio = self.proprio.pin_memory()
            self.actions_gt = self.actions_gt.pin_memory()
            self.ctrl_gt = self.ctrl_gt.pin_memory()

    def sample_batch(self, indices: torch.Tensor, target_device: torch.device):
        idx_store = indices.to(self.storage_device)
        return (
            self.agent_u8[idx_store].to(target_device, non_blocking=True),
            self.eih_u8[idx_store].to(target_device, non_blocking=True),
            self.proprio[idx_store].to(target_device, non_blocking=True),
            self.actions_gt[idx_store].to(target_device, non_blocking=True),
            self.ctrl_gt[idx_store].to(target_device, non_blocking=True),
        )


# ---------------------------------------------------------------------------
# 2. 듀얼 카메라 ResNet18 비전 백본 + SpatialSoftmax + 초구 라우팅 키 추출기
# ---------------------------------------------------------------------------


class FrozenResNet18DualCam(nn.Module):
    """사전학습 ImageNet ResNet18 (layer1..3 동결) 기반 128x128 듀얼 카메라 특징 추출기."""

    def __init__(self):
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        self.trunk = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True):
        # BatchNorm 통계 고정 유지
        return super().train(False)

    @staticmethod
    def _random_shift_aug(x: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
        if max_shift <= 0:
            return x
        B, C, H, W = x.shape
        padded = F.pad(x, (max_shift, max_shift, max_shift, max_shift), mode="replicate")
        dy = torch.randint(0, 2 * max_shift + 1, (1,), device=x.device).item()
        dx = torch.randint(0, 2 * max_shift + 1, (1,), device=x.device).item()
        return padded[:, :, dy : dy + H, dx : dx + W]

    @torch.no_grad()
    def forward(
        self, agent_u8: torch.Tensor, eih_u8: torch.Tensor, augment: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def _prep(img_u8: torch.Tensor) -> torch.Tensor:
            x = img_u8.float() / 255.0
            if augment:
                x = self._random_shift_aug(x, max_shift=4)
            return (x - self.img_mean) / self.img_std

        fa = self.trunk(_prep(agent_u8))  # (B, 256, 8, 8)
        fe = self.trunk(_prep(eih_u8))  # (B, 256, 8, 8)
        return fa.float(), fe.float()


class MultimodalMagneticKeyExtractor(nn.Module):
    """
    공통 기본 멀티모달 쿼리 추출기 (Key Extractor Parity 보장):
    - forward(): 모든 비교군(arm_frozen_mlp, arm_dtra_rigid, arm_dtra_elastic)에 동일하게 제공되는
      기본 멀티모달(DualCam g+2x2 풀링 + proprio + lang) 단위 초구 쿼리 추출.
    - route_top_m(): DTRA v3.0 전용 2단계 라우팅 (1차 Top-M 스캔 후 동일/유사 언어 후보군 내
      공통 모드 차감[Common-Mode Cancellation] + 중심화 4x4 공간 패치/DoG 윤곽/고유수용 잔차 재순위화).
    """

    def __init__(self, d_topo: int = 64, d_lang: int = 128, seed: int = 42):
        super().__init__()
        self.d_topo = d_topo
        self.d_lang = d_lang
        self.vis_dim = 2560  # 표준 DualCam (g 256 + 2x2 1024) * 2 = 2560 (Key Extractor Parity)
        self.rerank_vis_dim = 6400  # fa_2(1024) + fa_4(4096) + fa_dog(256) + fe_2(1024)
        self.d_rerank = 256
        gen = torch.Generator().manual_seed(seed + 1337)
        w_vis = torch.randn(self.vis_dim, d_topo, generator=gen) / math.sqrt(float(self.vis_dim))
        w_prop = torch.randn(12, d_topo, generator=gen) / math.sqrt(12.0)
        w_lang = torch.randn(d_lang, d_topo, generator=gen) / math.sqrt(float(d_lang))
        w_rerank = torch.randn(self.rerank_vis_dim, self.d_rerank, generator=gen) / math.sqrt(
            float(self.rerank_vis_dim)
        )
        self.register_buffer("w_vis", w_vis)
        self.register_buffer("w_prop", w_prop)
        self.register_buffer("w_lang", w_lang)
        self.register_buffer("w_rerank", w_rerank)

        # DTRA v3.0 Top-M Common-Mode Cancellation을 위한 슬롯별 순수 시각/고유수용/언어 프로토타입 뱅크
        self.slot_vis_sigs: List[torch.Tensor] = []
        self.slot_prop_sigs: List[torch.Tensor] = []
        self.slot_lang_sigs: List[torch.Tensor] = []
        self.register_buffer("vis_sig_bank", torch.empty(0, self.d_rerank))
        self.register_buffer("prop_sig_bank", torch.empty(0, 12))
        self.register_buffer("lang_sig_bank", torch.empty(0, d_lang))

    def reset_slot_signatures(self) -> None:
        self.slot_vis_sigs.clear()
        self.slot_prop_sigs.clear()
        self.slot_lang_sigs.clear()
        dev = self.w_vis.device
        self.vis_sig_bank = torch.empty(0, self.d_rerank, device=dev)
        self.prop_sig_bank = torch.empty(0, 12, device=dev)
        self.lang_sig_bank = torch.empty(0, self.d_lang, device=dev)

    @staticmethod
    def _pool_cam(feat: torch.Tensor) -> torch.Tensor:
        g = feat.mean(dim=(2, 3))  # (B, 256)
        q = F.adaptive_avg_pool2d(feat, (2, 2)).flatten(1)  # (B, 1024)
        return torch.cat([g, q], dim=-1)  # (B, 1280)

    @staticmethod
    def _extract_centered_spatial_features(fa: torch.Tensor, fe: torch.Tensor) -> torch.Tensor:
        # Agentview: 채널별 전역 배경 평균 차감(Centering) 후 2x2, 4x4 격자 + 3x3 DoG 윤곽 추출
        fa_c = fa - fa.mean(dim=(2, 3), keepdim=True)
        fa_2 = F.adaptive_avg_pool2d(fa_c, (2, 2)).flatten(1)  # (B, 1024)
        fa_4 = F.adaptive_avg_pool2d(fa_c, (4, 4)).flatten(1)  # (B, 4096)
        fa_dog = (fa - F.avg_pool2d(fa, kernel_size=3, stride=1, padding=1)).abs().mean(dim=(2, 3))  # (B, 256)

        # Eye-in-hand: 중심화 2x2 격자
        fe_c = fe - fe.mean(dim=(2, 3), keepdim=True)
        fe_2 = F.adaptive_avg_pool2d(fe_c, (2, 2)).flatten(1)  # (B, 1024)

        return torch.cat([fa_2, fa_4, fa_dog, fe_2], dim=-1)  # (B, 6400)

    @torch.no_grad()
    def extract_rerank_signature(
        self, fa: torch.Tensor, fe: torch.Tensor, proprio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vis_rerank_raw = self._extract_centered_spatial_features(fa, fe)
        vis_proj = F.normalize(vis_rerank_raw @ self.w_rerank, dim=-1)
        return vis_proj, proprio

    @torch.no_grad()
    def register_slot_signature(
        self,
        vis_sig: torch.Tensor,
        prop_sig: torch.Tensor,
        lang_emb: torch.Tensor,
        device: torch.device,
    ) -> None:
        v_mean = F.normalize(vis_sig.mean(dim=0) if vis_sig.dim() == 2 else vis_sig, dim=-1).detach().to(device)
        p_mean = (prop_sig.mean(dim=0) if prop_sig.dim() == 2 else prop_sig).detach().to(device)
        l_mean = F.normalize(lang_emb.mean(dim=0) if lang_emb.dim() == 2 else lang_emb, dim=-1).detach().to(device)
        self.slot_vis_sigs.append(v_mean)
        self.slot_prop_sigs.append(p_mean)
        self.slot_lang_sigs.append(l_mean)
        self.vis_sig_bank = torch.stack(self.slot_vis_sigs, dim=0).to(device)
        self.prop_sig_bank = torch.stack(self.slot_prop_sigs, dim=0).to(device)
        self.lang_sig_bank = torch.stack(self.slot_lang_sigs, dim=0).to(device)

    @torch.no_grad()
    def forward(
        self,
        fa: torch.Tensor,
        fe: torch.Tensor,
        proprio: torch.Tensor,
        lang_emb: torch.Tensor,
    ) -> torch.Tensor:
        vis = torch.cat([self._pool_cam(fa), self._pool_cam(fe)], dim=-1)
        kv = F.normalize(vis @ self.w_vis, dim=-1)
        kp = F.normalize(proprio @ self.w_prop, dim=-1)
        kl = F.normalize(lang_emb @ self.w_lang, dim=-1)
        fused = 0.25 * kv + 0.10 * kp + 0.65 * kl
        return F.normalize(fused, dim=-1)

    @torch.no_grad()
    def route_top_m(
        self,
        fa: torch.Tensor,
        fe: torch.Tensor,
        proprio: torch.Tensor,
        lang_emb: torch.Tensor,
        key_bank: torch.Tensor,
        top_m: int = 5,
        margin_thresh: float = 0.12,
        extra_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [DTRA v3.0 전용 2단계 라우팅]:
        1단계 O(K) 멀티모달 코사인 스캔으로 Top-M (M=5) 후보를 선별하고,
        동일/유사 지시어 그룹 후보들 간 공통 모드(언어 및 평균 시각 배경)를 차감(Centering/Whitening)하여
        순수 시각 격자 편차(DualCam 4x4/2x2 + DoG 윤곽) + 고유수용 잔차로 2차 재순위화 수행.
        Returns: (selected_indices (B,), stage1_scores (B, K), x_query (B, D_topo))
        """
        x_q = self.forward(fa, fe, proprio, lang_emb)
        s1 = x_q @ key_bank.T  # (B, K)
        if extra_bias is not None:
            s1 = s1 + extra_bias

        B, K = s1.shape
        if K <= 1 or self.vis_sig_bank.shape[0] != K:
            return s1.argmax(dim=-1), s1, x_q

        m_eff = min(top_m, K)
        top_vals, top_idx = torch.topk(s1, k=m_eff, dim=-1)  # (B, M)
        margins = top_vals[:, 0] - top_vals[:, 1]

        # Top-M 후보 중 1위와 언어 임베딩이 동일(>0.999)하거나 유사(>0.88)하면서 마진이 좁은 후보군 선별
        q_lang = F.normalize(lang_emb, dim=-1)  # (B, D_lang)
        cand_langs = self.lang_sig_bank[top_idx]  # (B, M, D_lang)
        q_lang_sims = (cand_langs * q_lang.unsqueeze(1)).sum(dim=-1)  # (B, M)
        score_gaps = top_vals[:, 0:1] - top_vals  # (B, M)
        exact_dup_mask = (q_lang_sims > 0.998) & (score_gaps < margin_thresh)  # (B, M)
        sim_lang_mask = (q_lang_sims > 0.88) & (score_gaps < 0.06)  # (B, M)
        need_rerank = (exact_dup_mask.sum(dim=-1) >= 2) | (sim_lang_mask.sum(dim=-1) >= 2) | (margins < 0.025)

        selected = top_idx[:, 0].clone()
        if not bool(need_rerank.any().item()):
            return selected, s1, x_q

        q_vis, q_prop = self.extract_rerank_signature(fa, fe, proprio)
        rerank_rows = torch.nonzero(need_rerank, as_tuple=False).flatten()
        for b_tensor in rerank_rows:
            b = int(b_tensor.item())
            if int(exact_dup_mask[b].sum().item()) >= 2:
                # Case A: 지시어가 100% 동일한 Duplicate-Lang 후보군 -> 공통 언어 완전 소거 후 시각+고유수용 잔차로 분리
                c_idx = top_idx[b, exact_dup_mask[b]]
                s_lang_cm = torch.zeros(c_idx.shape[0], device=s1.device)
                w_l, w_v, w_p = 0.0, 0.25, 0.04
            else:
                # Case B: 지시어가 유사(front vs back 등)한 후보군 -> 공통 문장 모드 차감 후 차별 단어 잔차 증폭
                c_idx = top_idx[b, sim_lang_mask[b]] if int(sim_lang_mask[b].sum().item()) >= 2 else top_idx[b, :2]
                p_lang = self.lang_sig_bank[c_idx]  # (C, D_lang)
                mu_lang = p_lang.mean(dim=0, keepdim=True)
                p_lang_res = F.normalize(p_lang - mu_lang, dim=-1)
                q_lang_res = F.normalize(q_lang[b : b + 1] - mu_lang, dim=-1)
                s_lang_cm = (q_lang_res @ p_lang_res.T).squeeze(0)  # (C,)
                w_l, w_v, w_p = 0.20, 0.04, 0.01

            # 후보군 내 공통 시각 배경 모드 차감 (Common-Mode Centering & L2 Whitening)
            p_vis = self.vis_sig_bank[c_idx]  # (C, d_rerank)
            mu_vis = p_vis.mean(dim=0, keepdim=True)
            p_vis_res = F.normalize(p_vis - mu_vis, dim=-1)
            q_vis_res = F.normalize(q_vis[b : b + 1] - mu_vis, dim=-1)
            s_vis_cm = (q_vis_res @ p_vis_res.T).squeeze(0)  # (C,)

            # 후보군 내 고유수용 잔차 정규화 내적
            p_prop = self.prop_sig_bank[c_idx]  # (C, 12)
            mu_prop = p_prop.mean(dim=0, keepdim=True)
            std_prop = p_prop.std(dim=0, keepdim=True).clamp_min(1e-2)
            p_prop_res = F.normalize((p_prop - mu_prop) / std_prop, dim=-1)
            q_prop_res = F.normalize((q_prop[b : b + 1] - mu_prop) / std_prop, dim=-1)
            s_prop_cm = (q_prop_res @ p_prop_res.T).squeeze(0)  # (C,)

            s_comb = s1[b, c_idx] + w_l * s_lang_cm + w_v * s_vis_cm + w_p * s_prop_cm
            selected[b] = c_idx[int(s_comb.argmax().item())]

        return selected, s1, x_q


class SpatialSoftmax(nn.Module):
    """특징 맵 (B, C, H, W) 에서 채널별 연속 공간 기대좌표 (x, y) 추출."""

    def __init__(self, height: int = 8, width: int = 8, temperature: float = 1.0):
        super().__init__()
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        attn = F.softmax(x.reshape(B, C, H * W) / self.temperature, dim=-1)
        ex = torch.sum(attn * self.pos_x, dim=-1)
        ey = torch.sum(attn * self.pos_y, dim=-1)
        return torch.cat([ex, ey], dim=-1)  # (B, 2*C)


class SpatialContextAdapter(nn.Module):
    """듀얼 카메라 공간 키포인트 + 고유수용 + 언어 컨텍스트 융합 모듈."""

    def __init__(self, d_lang: int = 128, ctx_dim: int = 256, n_kp: int = 48):
        super().__init__()
        self.conv_agent = nn.Conv2d(256, n_kp, kernel_size=1)
        self.conv_eih = nn.Conv2d(256, n_kp, kernel_size=1)
        self.spatial_sm = SpatialSoftmax(8, 8)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(12, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.lang_mlp = nn.Sequential(
            nn.Linear(d_lang, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        in_dim = (n_kp * 3) * 2 + 128 + 64
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, ctx_dim),
            nn.LayerNorm(ctx_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        fa: torch.Tensor,
        fe: torch.Tensor,
        proprio: torch.Tensor,
        lang_emb: torch.Tensor,
    ) -> torch.Tensor:
        pa = self.conv_agent(fa)
        pe = self.conv_eih(fe)
        kp_a = self.spatial_sm(pa)
        kp_e = self.spatial_sm(pe)
        gap_a = pa.mean(dim=(2, 3))
        gap_e = pe.mean(dim=(2, 3))
        pr = self.proprio_mlp(proprio)
        lg = self.lang_mlp(lang_emb)
        return self.fusion(torch.cat([kp_a, gap_a, kp_e, gap_e, pr, lg], dim=-1))


# ---------------------------------------------------------------------------
# 3. 4종 비교 정책 모델 구현 (동일 파라미터 체급)
# ---------------------------------------------------------------------------


class FrozenMLPSlot(nn.Module):
    """Arm 1 단일 태스크 동결 슬롯: Flat Chunk MLP (비-스플라인, 비-위상)."""

    def __init__(self, ctx_dim: int = 256, horizon: int = 16, action_dim: int = 7):
        super().__init__()
        self.ctx_encoder = SpatialContextAdapter(ctx_dim=ctx_dim)
        self.head = nn.Sequential(
            nn.Linear(ctx_dim, 320),
            nn.LayerNorm(320),
            nn.SiLU(),
            nn.Linear(320, 320),
            nn.LayerNorm(320),
            nn.SiLU(),
            nn.Linear(320, horizon * action_dim),
        )
        self.horizon = horizon
        self.action_dim = action_dim

    def forward(self, fa, fe, proprio, lang_emb) -> torch.Tensor:
        ctx = self.ctx_encoder(fa, fe, proprio, lang_emb)
        return self.head(ctx).view(-1, self.horizon, self.action_dim)


class KnotMotionBasisSlot(nn.Module):
    """매듭 공간(N_k=6) 사전 수축 공유 모션 기저 단일 슬롯 (동결 시 requires_grad=False로 분리)."""

    def __init__(self, d_basis: int = 64, num_knots: int = 6, num_cp: int = 8, action_dim: int = 7):
        super().__init__()
        self.d_basis = d_basis
        self.num_knots = num_knots
        self.num_cp = num_cp
        self.action_dim = action_dim
        self.knot_net = nn.Sequential(
            nn.Linear(d_basis, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, num_knots * 16),
            nn.LayerNorm(num_knots * 16),
            nn.SiLU(),
        )
        self.knot_to_cp_head = nn.Linear(num_knots * 16, num_cp * action_dim)

    def forward(self, coeff: torch.Tensor) -> torch.Tensor:
        knot_feat = self.knot_net(coeff)
        return self.knot_to_cp_head(knot_feat).view(-1, self.num_cp, self.action_dim)


class SharedBasisBank(nn.Module):
    """
    DTRA v3.0 공유 B-스플라인 기저 뱅크 (SharedBasisBank):
    - 매듭 공간(N_k=6) 사전 수축 기반의 공유 모션 기저와 튜브별 경량 위상 계수/미세 잔차를 분리해
      K=100 파라미터 증가를 억제하면서 단일 태스크 표현력(90%+ SR)을 보존.
    - 동결된 기저 슬롯은 requires_grad=False로 즉시 분리하고 옵티마이저 그룹에서 완전 제외하여
      AdamW weight_decay / momentum에 의한 Silent Drift를 0.0 (L_inf == 0.0)으로 원천 차단.
    """

    def __init__(
        self,
        num_knots: int = 6,
        num_cp: int = 8,
        action_dim: int = 7,
        d_basis: int = 64,
        novelty_thresh: float = 0.88,
        max_bases: int = 48,
    ):
        super().__init__()
        self.num_knots = num_knots
        self.num_cp = num_cp
        self.action_dim = action_dim
        self.d_basis = d_basis
        self.novelty_thresh = novelty_thresh
        self.max_bases = max_bases

        self.basis_slots = nn.ModuleList()
        self.basis_sigs: List[torch.Tensor] = []
        self.frozen_slot_ids: set = set()

    @property
    def active_basis_count(self) -> int:
        return len(self.basis_slots)

    def freeze_all_bases(self) -> None:
        """현재까지 등록된 모든 기저 슬롯을 동결(requires_grad=False)하여 후속 AdamW 드리프트를 차단."""
        for idx, slot in enumerate(self.basis_slots):
            for p in slot.parameters():
                p.requires_grad_(False)
                p.grad = None
            self.frozen_slot_ids.add(idx)

    def allocate_or_share_basis(
        self, motion_sig: Optional[torch.Tensor], device: torch.device
    ) -> Tuple[int, bool]:
        """
        기존 기저들과 모션 서명 유사도를 비교하여 임계치 이상이면 기존 동결 기저를 공유(is_new=False),
        새로운 모션 패턴이면 신규 기저 슬롯을 할당(is_new=True).
        """
        self.freeze_all_bases()
        if motion_sig is None:
            sig = F.normalize(torch.randn(self.num_knots * self.action_dim, device=device), dim=-1)
        else:
            sig = F.normalize(motion_sig.detach().flatten().float().to(device), dim=-1)

        if len(self.basis_slots) > 0 and len(self.basis_sigs) == len(self.basis_slots):
            sig_bank = torch.stack(self.basis_sigs, dim=0).to(device)
            d_min = min(sig.shape[0], sig_bank.shape[1])
            sims = F.normalize(sig_bank[:, :d_min], dim=-1) @ F.normalize(sig[:d_min], dim=-1)
            best_sim, best_idx_t = sims.max(dim=0)
            if float(best_sim.item()) >= self.novelty_thresh or len(self.basis_slots) >= self.max_bases:
                return int(best_idx_t.item()), False

        new_slot = KnotMotionBasisSlot(
            d_basis=self.d_basis, num_knots=self.num_knots, num_cp=self.num_cp, action_dim=self.action_dim
        ).to(device)
        self.basis_slots.append(new_slot)
        self.basis_sigs.append(sig.detach())
        return len(self.basis_slots) - 1, True

    def eval_basis(self, basis_idx: int, coeff: torch.Tensor) -> torch.Tensor:
        """동결 슬롯도 coeff로의 역전파는 허용하되 슬롯 자체 requires_grad=False로 파라미터 드리프트 0.0 보장."""
        slot = self.basis_slots[basis_idx]
        return slot(coeff)


class FrozenMLPMultimodalPolicy(nn.Module):
    """
    Arm 1 (Baseline): 태스크별 동결 자성 키 + Flat MLP 슬롯.
    공정한 Ablation을 위해 DTRA v3.0 전용 기법(No-Encroachment Constraint 및 Top-M Common-Mode Cancellation)을
    적용하지 않고, 기본 멀티모달 쿼리(forward) + 표준 마진 키 + 1단계 코사인 argmax 라우팅을 사용.
    """

    def __init__(
        self,
        backbone: FrozenResNet18DualCam,
        key_extractor: MultimodalMagneticKeyExtractor,
        horizon: int = 16,
        action_dim: int = 7,
    ):
        super().__init__()
        self.backbone = backbone
        self.key_extractor = key_extractor
        self.horizon = horizon
        self.action_dim = action_dim
        self.slots = nn.ModuleList()
        self.slot_keys: List[torch.Tensor] = []
        self.register_buffer("key_bank", torch.empty(0, key_extractor.d_topo))

    @property
    def active_basis_count(self) -> int:
        return len(self.slots)

    def add_task_slot(
        self,
        raw_mean_key: torch.Tensor,
        device: torch.device,
        sample_queries: Optional[torch.Tensor] = None,
    ) -> Tuple[int, Dict[str, float]]:
        for slot in self.slots:
            for p in slot.parameters():
                p.requires_grad_(False)
                p.grad = None
        refined_key, key_stats = _refine_key_baseline_standard(
            raw_mean_key.to(device), self.slot_keys, sample_queries=sample_queries
        )
        self.slot_keys.append(refined_key.detach())
        self.key_bank = torch.stack(self.slot_keys, dim=0).to(device)
        slot = FrozenMLPSlot(ctx_dim=256, horizon=self.horizon, action_dim=self.action_dim).to(device)
        self.slots.append(slot)
        return len(self.slots) - 1, key_stats

    @torch.no_grad()
    def route(self, fa, fe, proprio, lang_emb) -> torch.Tensor:
        x_q = self.key_extractor(fa, fe, proprio, lang_emb)
        sims = x_q @ self.key_bank.T
        return sims.argmax(dim=-1)

    def reset_episode(self) -> None:
        pass

    def chunk_snap_back(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def episode_snap_back(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def snap_back(self) -> Tuple[float, float]:
        return self.episode_snap_back()

    @torch.no_grad()
    def predict_action_chunk(self, agent_u8, eih_u8, proprio, lang_emb) -> Tuple[torch.Tensor, int, float, float]:
        fa, fe = self.backbone(agent_u8, eih_u8, augment=False)
        sel = int(self.route(fa, fe, proprio, lang_emb)[0].item())
        traj = self.slots[sel](fa, fe, proprio, lang_emb)
        return traj, sel, 0.0, 0.0


class ExperienceReplayPolicy(nn.Module):
    """Arm 2: 단일 공유 멀티모달 B-Spline 정책 + 5% Experience Replay."""

    def __init__(
        self,
        backbone: FrozenResNet18DualCam,
        bspline_decoder: BSplineDecoder,
        ctx_dim: int = 256,
        num_cp: int = 8,
        action_dim: int = 7,
    ):
        super().__init__()
        self.backbone = backbone
        self.bspline_decoder = bspline_decoder
        self.num_cp = num_cp
        self.action_dim = action_dim
        self.ctx_encoder = SpatialContextAdapter(ctx_dim=ctx_dim)
        self.head = nn.Sequential(
            nn.Linear(ctx_dim, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Linear(384, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Linear(384, num_cp * action_dim),
        )

    @property
    def active_basis_count(self) -> int:
        return 1

    def forward_train(self, fa, fe, proprio, lang_emb) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = self.ctx_encoder(fa, fe, proprio, lang_emb)
        cp = self.head(ctx).view(-1, self.num_cp, self.action_dim)
        traj = self.bspline_decoder(cp)
        return cp, traj

    def reset_episode(self) -> None:
        pass

    def chunk_snap_back(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def episode_snap_back(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def snap_back(self) -> Tuple[float, float]:
        return self.episode_snap_back()

    @torch.no_grad()
    def predict_action_chunk(self, agent_u8, eih_u8, proprio, lang_emb) -> Tuple[torch.Tensor, int, float, float]:
        fa, fe = self.backbone(agent_u8, eih_u8, augment=False)
        _, traj = self.forward_train(fa, fe, proprio, lang_emb)
        return traj, 0, 0.0, 0.0


class LIBEROHypertubeNode(nn.Module):
    """
    DTRA v3.0 2-Stage 탄성 튜브 노드:
    - 매듭 공간(N_k=6) 공유 모션 기저(SharedBasisBank) + 튜브별 경량 위상 계수 및 미세 잔차(Micro-Residual) 결합
    - 이중 시정수 스냅백(Two-Timescale Snap-Back):
      * chunk_snap_back(): 매 16-스텝 청크 끝에서 공간 탄성 변위(delta_z)만 0.000으로 즉시 리셋,
        에피소드 내 진행 위상/불응기 흔적(episodic_trace, Conf Gate, 그리퍼/관절 상태 변화 게이팅)은 유지
      * episode_snap_back(): 에피소드 종료 시 delta_z, episodic_trace 전부 0.000으로 초기화
    """

    def __init__(
        self,
        d_topo: int = 64,
        ctx_dim: int = 256,
        num_knots: int = 6,
        num_cp: int = 8,
        d_basis: int = 64,
        action_dim: int = 7,
        elastic: bool = True,
        res_scale: float = 0.70,
        init_key: Optional[torch.Tensor] = None,
        basis_idx: int = 0,
    ):
        super().__init__()
        self.d_topo = d_topo
        self.ctx_dim = ctx_dim
        self.num_knots = num_knots
        self.num_cp = num_cp
        self.d_basis = d_basis
        self.action_dim = action_dim
        self.elastic = elastic
        self.res_scale = res_scale
        self.basis_idx = basis_idx

        z0_init = (
            F.normalize(init_key.detach().clone(), dim=-1)
            if init_key is not None
            else F.normalize(torch.randn(d_topo), dim=-1)
        )
        self.z0_raw = nn.Parameter(z0_init.clone())
        self.register_buffer("mag_key", z0_init.clone())
        self.radius_raw = nn.Parameter(torch.tensor(_inv_softplus(0.25)))
        self.omega_alpha = nn.Parameter(torch.tensor(-0.4))

        # 48-키포인트 고해상도 공간 어댑터 유지 (단일 태스크 90%+ 표현력 보장)
        self.ctx_encoder = SpatialContextAdapter(ctx_dim=ctx_dim, n_kp=48)
        self.theta_head = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
        # 에피소드 내 진행 위상/불응기 흔적(episodic_trace) 및 Conf Gate (그리퍼/관절 변화 게이팅)
        self.trace_proj = nn.Sequential(
            nn.Linear(ctx_dim + 12, ctx_dim),
            nn.LayerNorm(ctx_dim),
            nn.Tanh(),
        )
        self.conf_gate_net = nn.Sequential(
            nn.Linear(ctx_dim + 12, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Stage S0: 위상 링 앵커 (z_base + B_ring(theta) -> P_coarse 및 초기 기저 계수)
        self.anchor_head = nn.Sequential(
            nn.Linear(d_topo + 3, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Linear(192, num_cp * action_dim),
        )
        self.anchor_coeff_head = nn.Linear(d_topo + 3, d_basis)

        # Stage S1: 공유 매듭 기저 계수 헤드 + 튜브별 경량 미세 잔차(Micro-Residual) 헤드
        fine_in = ctx_dim + d_topo + d_topo
        self.fine_coeff_head = nn.Sequential(
            nn.Linear(fine_in, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Linear(160, d_basis),
        )
        self.micro_res_head = nn.Sequential(
            nn.Linear(fine_in, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Linear(160, num_cp * action_dim),
        )
        self.res_gate = nn.Sequential(
            nn.Linear(fine_in, 64),
            nn.SiLU(),
            nn.Linear(64, num_cp * action_dim),
            nn.Sigmoid(),
        )

        # 런타임 동적 상태 버퍼 (Two-Timescale Snap-Back 대상)
        self.register_buffer("delta_z_state", torch.zeros(1, d_topo))
        self.register_buffer("episodic_trace", torch.zeros(1, ctx_dim))
        self.register_buffer("prev_proprio_state", torch.zeros(1, 12))
        self.has_prev_proprio: bool = False
        self.conf_gate_state: float = 0.0
        self.last_delta_z_norm: float = 0.0
        self.last_alpha: float = 0.0

    @property
    def z0(self) -> torch.Tensor:
        return F.normalize(self.z0_raw, dim=-1)

    @property
    def radius(self) -> torch.Tensor:
        return F.softplus(self.radius_raw)

    @property
    def alpha(self) -> torch.Tensor:
        if not self.elastic:
            return self.omega_alpha.new_zeros(())
        return torch.sigmoid(self.omega_alpha)

    @property
    def episodic_trace_norm(self) -> float:
        return float(self.episodic_trace.norm(dim=-1).max().item())

    def chunk_snap_back(self) -> Tuple[float, float]:
        """매 16-스텝 청크 끝: 공간 탄성 변위(delta_z)만 0.000으로 즉시 리셋, episodic_trace는 보존."""
        pre_dz = self.last_delta_z_norm
        self.delta_z_state.zero_()
        self.last_delta_z_norm = 0.0
        post_dz = float(self.delta_z_state.norm(dim=-1).max().item())
        return pre_dz, post_dz

    def episode_snap_back(self) -> Tuple[float, float]:
        """에피소드 종료 시: delta_z, episodic_trace, 그리퍼/관절 상태 흔적을 전부 0.000으로 완전 초기화."""
        pre_norm = max(self.last_delta_z_norm, self.episodic_trace_norm)
        self.delta_z_state.zero_()
        self.episodic_trace.zero_()
        self.prev_proprio_state.zero_()
        self.has_prev_proprio = False
        self.conf_gate_state = 0.0
        self.last_delta_z_norm = 0.0
        post_norm = float(
            self.delta_z_state.norm(dim=-1).max().item() + self.episodic_trace.norm(dim=-1).max().item()
        )
        return pre_norm, post_norm

    def snap_back(self) -> float:
        pre, _ = self.episode_snap_back()
        return pre

    def forward(
        self,
        fa: torch.Tensor,
        fe: torch.Tensor,
        proprio: torch.Tensor,
        lang_emb: torch.Tensor,
        x_query: torch.Tensor,
        shared_basis_bank: Optional[SharedBasisBank] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x_query.shape[0]
        ctx = self.ctx_encoder(fa, fe, proprio, lang_emb)

        # Conf Gate & 그리퍼(3:5)/관절(5:12) 변화량 게이팅 기반 에피소드 진행 흔적 업데이트
        trace_in = torch.cat([ctx, proprio], dim=-1)
        conf_g = self.conf_gate_net(trace_in)  # (B, 1)
        step_trace = self.trace_proj(trace_in) * conf_g  # (B, ctx_dim)

        if not self.training:
            with torch.no_grad():
                p_cur = proprio[:1]
                if self.has_prev_proprio:
                    grip_delta = (p_cur[:, 3:5] - self.prev_proprio_state[:, 3:5]).norm(dim=-1, keepdim=True)
                    joint_delta = (p_cur[:, 5:12] - self.prev_proprio_state[:, 5:12]).norm(dim=-1, keepdim=True)
                    state_change_gate = torch.clamp(0.25 + 2.0 * grip_delta + 0.5 * joint_delta, 0.15, 1.0)
                else:
                    state_change_gate = torch.ones((1, 1), device=proprio.device, dtype=proprio.dtype)
                    self.has_prev_proprio = True
                self.prev_proprio_state.copy_(p_cur)
                eff_gate = float((conf_g[:1] * state_change_gate).mean().item())
                self.conf_gate_state = eff_gate
                self.episodic_trace.copy_(
                    (1.0 - 0.30 * eff_gate) * self.episodic_trace + (0.30 * eff_gate) * step_trace[:1]
                )

        # 학습-평가 간 분포 일치(Zero Train-Test Mismatch) 유지
        ctx_eff = ctx + 0.08 * step_trace

        # 1차 타원 고리 기저 B_ring(theta) = [1, cos(theta), sin(theta)]
        ang = F.normalize(self.theta_head(ctx_eff), dim=-1)
        b_ring = torch.cat([torch.ones(B, 1, device=ctx.device, dtype=ctx.dtype), ang], dim=-1)

        z_base = self.z0.unsqueeze(0).expand(B, -1)
        a_val = self.alpha

        if self.elastic:
            z_tilde = F.normalize(z_base + a_val * (x_query - z_base), dim=-1)
            delta_z = z_tilde - z_base
        else:
            z_tilde = z_base
            delta_z = torch.zeros_like(z_base)

        with torch.no_grad():
            self.delta_z_state.copy_(delta_z[:1].detach())
            self.last_delta_z_norm = float(delta_z.norm(dim=-1).mean().item())
            self.last_alpha = float(a_val.item())

        ring_in = torch.cat([z_base, b_ring], dim=-1)
        p_coarse = self.anchor_head(ring_in).view(B, self.num_cp, self.action_dim)
        c_coarse = self.anchor_coeff_head(ring_in)

        fine_in = torch.cat([ctx_eff, delta_z, z_tilde], dim=-1)
        c_fine = self.fine_coeff_head(fine_in)
        p_micro = self.micro_res_head(fine_in).view(B, self.num_cp, self.action_dim)
        gate = self.res_gate(fine_in).view(B, self.num_cp, self.action_dim)

        if shared_basis_bank is not None and len(shared_basis_bank.basis_slots) > self.basis_idx:
            p_basis = shared_basis_bank.eval_basis(self.basis_idx, c_coarse + c_fine)
            delta_p = 0.55 * p_basis + 0.45 * p_micro
        else:
            delta_p = p_micro

        cp = p_coarse + self.res_scale * gate * delta_p
        return cp, delta_z


class DTRALiberoPolicy(nn.Module):
    """Arm 3 (Rigid, alpha=0) & Arm 4 (Full Elastic, alpha=learned) 통합 DTRA v3.0 정책."""

    def __init__(
        self,
        backbone: FrozenResNet18DualCam,
        key_extractor: MultimodalMagneticKeyExtractor,
        bspline_decoder: BSplineDecoder,
        elastic: bool = True,
        d_topo: int = 64,
        num_knots: int = 6,
        num_cp: int = 8,
        d_basis: int = 64,
        action_dim: int = 7,
        eta_inertia: float = 0.20,
        eta_myelin: float = 0.25,
    ):
        super().__init__()
        self.backbone = backbone
        self.key_extractor = key_extractor
        self.key_extractor.reset_slot_signatures()
        self.bspline_decoder = bspline_decoder
        self.elastic = elastic
        self.d_topo = d_topo
        self.num_knots = num_knots
        self.num_cp = num_cp
        self.d_basis = d_basis
        self.action_dim = action_dim
        self.eta_inertia = eta_inertia
        self.eta_myelin = eta_myelin

        self.shared_basis_bank = SharedBasisBank(
            num_knots=num_knots,
            num_cp=num_cp,
            action_dim=action_dim,
            d_basis=d_basis,
            novelty_thresh=0.88,
            max_bases=48,
        )
        self.tubes = nn.ModuleList()
        self.register_buffer("mag_key_bank", torch.empty(0, d_topo))
        self.register_buffer("myelin_mat", torch.zeros(1, 1))
        self.register_buffer("coact_trace", torch.zeros(1, 1))
        self.prev_act_vec: Optional[torch.Tensor] = None
        self.prev_selected: Optional[int] = None

    @property
    def active_basis_count(self) -> int:
        return self.shared_basis_bank.active_basis_count

    def freeze_completed_slots(self, active_slot_id: Optional[int] = None) -> None:
        """완료된 모든 과거 튜브 및 공유 기저 슬롯을 requires_grad=False로 동결 (AdamW Silent Drift 차단)."""
        for idx, t in enumerate(self.tubes):
            if active_slot_id is not None and idx == active_slot_id:
                for p in t.parameters():
                    p.requires_grad_(True)
            else:
                for p in t.parameters():
                    p.requires_grad_(False)
                    p.grad = None
        if active_slot_id is None:
            self.shared_basis_bank.freeze_all_bases()
        else:
            active_basis_idx = self.tubes[active_slot_id].basis_idx
            for b_idx, b_slot in enumerate(self.shared_basis_bank.basis_slots):
                if b_idx != active_basis_idx or (b_idx in self.shared_basis_bank.frozen_slot_ids):
                    for p in b_slot.parameters():
                        p.requires_grad_(False)
                        p.grad = None

    def get_trainable_params(self, slot_id: int) -> List[nn.Parameter]:
        """현재 학습할 신규 튜브 및 신규(비동결) 기저 슬롯의 파라미터만 반환 (동결 슬롯 100% 제외)."""
        self.freeze_completed_slots(active_slot_id=slot_id)
        node = self.tubes[slot_id]
        params: List[nn.Parameter] = [p for p in node.parameters() if p.requires_grad]
        b_idx = node.basis_idx
        if 0 <= b_idx < len(self.shared_basis_bank.basis_slots) and (
            b_idx not in self.shared_basis_bank.frozen_slot_ids
        ):
            b_slot = self.shared_basis_bank.basis_slots[b_idx]
            params.extend([p for p in b_slot.parameters() if p.requires_grad])
        return params

    def add_task_tube(
        self,
        raw_mean_key: torch.Tensor,
        device: torch.device,
        sample_queries: Optional[torch.Tensor] = None,
        motion_sig: Optional[torch.Tensor] = None,
    ) -> Tuple[int, Dict[str, float]]:
        existing_keys = [t.mag_key for t in self.tubes]
        self.freeze_completed_slots(active_slot_id=None)

        if motion_sig is None:
            motion_sig = getattr(raw_mean_key, "_motion_sig", None)
        basis_idx, is_new_basis = self.shared_basis_bank.allocate_or_share_basis(motion_sig, device)

        refined_key, key_stats = _refine_key_with_contrastive_margin(
            raw_mean_key.to(device), existing_keys, sample_queries=sample_queries
        )
        key_stats["basis_idx"] = float(basis_idx)
        key_stats["is_new_basis"] = 1.0 if is_new_basis else 0.0
        key_stats["active_basis_count"] = float(self.shared_basis_bank.active_basis_count)

        node = LIBEROHypertubeNode(
            d_topo=self.d_topo,
            ctx_dim=256,
            num_knots=self.num_knots,
            num_cp=self.num_cp,
            d_basis=self.d_basis,
            action_dim=self.action_dim,
            elastic=self.elastic,
            init_key=refined_key,
            basis_idx=basis_idx,
        ).to(device)
        self.tubes.append(node)
        self.mag_key_bank = torch.stack([t.mag_key.detach() for t in self.tubes], dim=0).to(device)

        n = len(self.tubes)
        new_m = torch.zeros(n, n, device=device)
        new_c = torch.zeros(n, n, device=device)
        if n > 1:
            new_m[:-1, :-1] = self.myelin_mat.to(device)
            new_c[:-1, :-1] = self.coact_trace.to(device)
        self.myelin_mat = new_m
        self.coact_trace = new_c
        return n - 1, key_stats

    def update_hebbian_edges(self, sims: torch.Tensor, sims_next: torch.Tensor) -> None:
        """시간적 인접 청크 간 공동 활성화 흔적 C_ij -> 수초화 지름길 M_ij 갱신."""
        if len(self.tubes) < 2:
            return
        with torch.no_grad():
            a_t = F.softmax(sims * 6.0, dim=-1)
            a_nxt = F.softmax(sims_next * 6.0, dim=-1)
            coact = (a_t.T @ a_nxt) / max(1, a_t.shape[0])
            self.coact_trace = 0.90 * self.coact_trace + 0.10 * coact
            self.myelin_mat = self.eta_myelin * torch.tanh(self.coact_trace)
            self.myelin_mat.fill_diagonal_(0.0)

    def _get_frozen_topo_banks(self, active_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if active_idx <= 0:
            z_e = self.tubes[0].z0.new_empty(0, self.d_topo)
            r_e = self.tubes[0].radius.new_empty(0)
            return z_e, r_e
        z_bank = torch.stack([self.tubes[j].z0.detach() for j in range(active_idx)], dim=0)
        r_bank = torch.stack([self.tubes[j].radius.detach() for j in range(active_idx)], dim=0)
        return z_bank, r_bank

    def topo_loss(
        self,
        active_idx: int,
        frozen_z0_bank: Optional[torch.Tensor] = None,
        frozen_r_bank: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """동결된 구 튜브는 고정한 채 신규 튜브 z0의 구면 척력 + 반지름 콤팩트성 정규화 (K=100 벡터화)."""
        node = self.tubes[active_idx]
        l_comp = 0.02 * node.radius
        if active_idx == 0:
            return l_comp
        z_other, r_other = (
            (frozen_z0_bank, frozen_r_bank)
            if (frozen_z0_bank is not None and frozen_r_bank is not None)
            else self._get_frozen_topo_banks(active_idx)
        )
        d_chord = (node.z0.unsqueeze(0) - z_other).norm(dim=-1)
        target_sep = node.radius + r_other + 0.35
        l_rep = F.relu(target_sep - d_chord).pow(2).sum()
        return l_comp + 0.25 * l_rep

    def chunk_snap_back(self) -> Tuple[float, float]:
        """매 16-스텝 청크 끝: 공간 탄성 변위(delta_z)만 0.000으로 즉시 리셋, 관성 및 에피소드 흔적은 유지."""
        max_pre = 0.0
        max_post = 0.0
        for t in self.tubes:
            pre, post = t.chunk_snap_back()
            max_pre = max(max_pre, pre)
            max_post = max(max_post, post)
        return max_pre, max_post

    def episode_snap_back(self) -> Tuple[float, float]:
        """에피소드 종료 시: delta_z, inertia, episodic_trace를 전부 0.000으로 완전 초기화."""
        max_pre = 0.0
        max_post = 0.0
        if self.prev_act_vec is not None:
            max_pre = max(max_pre, float(self.prev_act_vec.norm().item()))
        self.prev_act_vec = None
        self.prev_selected = None
        for t in self.tubes:
            pre, post = t.episode_snap_back()
            max_pre = max(max_pre, pre)
            max_post = max(max_post, post)
        inertia_norm = 0.0 if (self.prev_act_vec is None and self.prev_selected is None) else 1.0
        return max_pre, max(max_post, inertia_norm)

    def reset_episode(self) -> None:
        self.episode_snap_back()

    def snap_back(self) -> Tuple[float, float]:
        return self.episode_snap_back()

    @torch.no_grad()
    def route(self, fa, fe, proprio, lang_emb) -> Tuple[int, torch.Tensor]:
        extra_bias = None
        if self.prev_selected is not None and self.prev_act_vec is not None and len(self.tubes) > 0:
            x_q_tmp = self.key_extractor(fa, fe, proprio, lang_emb)
            cos_sim = (x_q_tmp @ self.mag_key_bank.T)[0]
            o_prev = torch.zeros_like(cos_sim)
            o_prev[self.prev_selected] = 1.0
            myelin_boost = (self.prev_act_vec @ self.myelin_mat.T) * F.relu(cos_sim)
            extra_bias = (self.eta_inertia * o_prev + myelin_boost).unsqueeze(0)

        sel_t, f_mag, x_q = self.key_extractor.route_top_m(
            fa, fe, proprio, lang_emb, self.mag_key_bank, top_m=5, margin_thresh=0.12, extra_bias=extra_bias
        )
        sel = int(sel_t[0].item())
        self.prev_selected = sel
        self.prev_act_vec = F.softmax(f_mag[0] * 6.0, dim=-1)
        return sel, x_q

    @torch.no_grad()
    def predict_action_chunk(self, agent_u8, eih_u8, proprio, lang_emb) -> Tuple[torch.Tensor, int, float, float]:
        fa, fe = self.backbone(agent_u8, eih_u8, augment=False)
        sel, x_q = self.route(fa, fe, proprio, lang_emb)
        node = self.tubes[sel]
        cp, _ = node(fa, fe, proprio, lang_emb, x_q, shared_basis_bank=self.shared_basis_bank)
        traj = self.bspline_decoder(cp)
        return traj, sel, node.last_alpha, node.last_delta_z_norm


# ---------------------------------------------------------------------------
# 4. 학습 & 폐루프 LIBERO 롤아웃 평가 엔진
# ---------------------------------------------------------------------------


def compute_task_keys(
    dataset: LiberoTaskDataset,
    backbone: FrozenResNet18DualCam,
    key_extractor: MultimodalMagneticKeyExtractor,
    text_enc: LightweightTextEncoder,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lang_vec = text_enc.encode(dataset.language, device)
    n_sample = min(256, dataset.num_samples)
    idx = torch.linspace(0, dataset.num_samples - 1, n_sample, dtype=torch.long)
    av, eih, pr, _, ctrl_gt = dataset.sample_batch(idx, device)
    fa, fe = backbone(av, eih, augment=False)
    lg_batch = lang_vec.unsqueeze(0).expand(n_sample, -1)
    keys = key_extractor(fa, fe, pr, lg_batch)
    vis_sig, prop_sig = key_extractor.extract_rerank_signature(fa, fe, pr)
    key_extractor.register_slot_signature(vis_sig, prop_sig, lang_vec, device)

    mean_key = F.normalize(keys.mean(dim=0), dim=-1)
    # 매듭 공간 모션 서명 추출 (제어점 평균 + 시간 차분 스펙트럼)
    ctrl_mean = ctrl_gt.mean(dim=0).flatten()
    ctrl_diff = (ctrl_gt[:, 1:, :] - ctrl_gt[:, :-1, :]).abs().mean(dim=0).flatten()
    mean_key._motion_sig = F.normalize(torch.cat([ctrl_mean, ctrl_diff], dim=0), dim=-1).detach()
    return mean_key, keys.detach()


def extract_cpu_replay_subset(
    dataset: LiberoTaskDataset,
    text_enc: LightweightTextEncoder,
    ratio: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Experience Replay용 5% 샘플만 CPU RAM에 경량 보관(VRAM 0 MB 점유)."""
    keep_n = max(16, int(round(ratio * dataset.num_samples)))
    pick = torch.randperm(dataset.num_samples)[:keep_n]
    return {
        "agent_u8": dataset.agent_u8[pick].detach().cpu(),
        "eih_u8": dataset.eih_u8[pick].detach().cpu(),
        "proprio": dataset.proprio[pick].detach().cpu(),
        "actions_gt": dataset.actions_gt[pick].detach().cpu(),
        "ctrl_gt": dataset.ctrl_gt[pick].detach().cpu(),
        "lang_vec": text_enc.encode(dataset.language, torch.device("cpu")).detach().cpu(),
    }


def _sample_replay_minibatch(
    replay_buffers: List[Dict[str, torch.Tensor]],
    rep_bs: int,
    device: torch.device,
    max_tasks_per_step: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_buf = len(replay_buffers)
    if n_buf <= max_tasks_per_step:
        chosen = replay_buffers
    else:
        perm_b = torch.randperm(n_buf)[:max_tasks_per_step].tolist()
        chosen = [replay_buffers[i] for i in perm_b]

    per_old = max(4, rep_bs // max(1, len(chosen)))
    avs, eihs, prs, acts, ctrls, lgs = [], [], [], [], [], []
    for buf in chosen:
        total_b = int(buf["proprio"].shape[0])
        take = min(per_old, total_b)
        pick = torch.randint(0, total_b, (take,))
        avs.append(buf["agent_u8"][pick])
        eihs.append(buf["eih_u8"][pick])
        prs.append(buf["proprio"][pick])
        acts.append(buf["actions_gt"][pick])
        ctrls.append(buf["ctrl_gt"][pick])
        lgs.append(buf["lang_vec"].unsqueeze(0).expand(take, -1))
    return (
        torch.cat(avs, dim=0).to(device, non_blocking=True),
        torch.cat(eihs, dim=0).to(device, non_blocking=True),
        torch.cat(prs, dim=0).to(device, non_blocking=True),
        torch.cat(acts, dim=0).to(device, non_blocking=True),
        torch.cat(ctrls, dim=0).to(device, non_blocking=True),
        torch.cat(lgs, dim=0).to(device, non_blocking=True),
    )


def train_arm_on_task(
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
    key_stats: Dict[str, float] = {}

    if arm_name == "arm_frozen_mlp":
        mean_key, sample_queries = compute_task_keys(
            dataset, policy.backbone, policy.key_extractor, text_enc, device
        )
        slot_id, key_stats = policy.add_task_slot(mean_key, device, sample_queries=sample_queries)
        slot = policy.slots[slot_id]
        params = [p for p in slot.parameters() if p.requires_grad]
    elif arm_name == "arm_er_5pct":
        params = [p for p in policy.parameters() if p.requires_grad]
    elif arm_name in ("arm_dtra_rigid", "arm_dtra_elastic"):
        mean_key, sample_queries = compute_task_keys(
            dataset, policy.backbone, policy.key_extractor, text_enc, device
        )
        slot_id, key_stats = policy.add_task_tube(mean_key, device, sample_queries=sample_queries)
        node = policy.tubes[slot_id]
        params = policy.get_trainable_params(slot_id)
        frozen_z0_bank, frozen_r_bank = policy._get_frozen_topo_banks(slot_id)
    else:
        raise ValueError(f"Unknown arm: {arm_name}")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    total_samples = 0
    last_loss = 0.0
    rep_bs = max(16, batch_size // 4)

    for ep in range(epochs):
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

                if arm_name == "arm_frozen_mlp":
                    pred_traj = slot(fa, fe, pr, lg)
                    loss = F.smooth_l1_loss(pred_traj, act_gt, beta=0.05)

                elif arm_name == "arm_er_5pct":
                    # 현재 태스크 손실 + 5% 과거 태스크 리플레이(CPU 버퍼에서 미니배치만 CUDA 전송)
                    cp_cur, traj_cur = policy.forward_train(fa, fe, pr, lg)
                    loss_cur = F.smooth_l1_loss(traj_cur, act_gt, beta=0.05) + 0.25 * F.mse_loss(
                        cp_cur, ctrl_gt
                    )
                    if replay_buffers:
                        o_av, o_eih, o_pr, o_act, o_ctrl, o_lg = _sample_replay_minibatch(
                            replay_buffers, rep_bs=rep_bs, device=device
                        )
                        o_fa, o_fe = policy.backbone(o_av, o_eih, augment=True)
                        o_cp, o_traj = policy.forward_train(o_fa, o_fe, o_pr, o_lg)
                        rb_loss = F.smooth_l1_loss(o_traj, o_act, beta=0.05) + 0.25 * F.mse_loss(
                            o_cp, o_ctrl
                        )
                        loss = loss_cur + rb_loss
                    else:
                        loss = loss_cur

                else:  # DTRA Rigid / Full Elastic
                    x_q = policy.key_extractor(fa, fe, pr, lg)
                    cp, _ = node(fa, fe, pr, lg, x_q, shared_basis_bank=policy.shared_basis_bank)
                    traj = policy.bspline_decoder(cp)
                    loss = (
                        F.smooth_l1_loss(traj, act_gt, beta=0.05)
                        + 0.25 * F.mse_loss(cp, ctrl_gt)
                        + 0.05 * policy.topo_loss(slot_id, frozen_z0_bank, frozen_r_bank)
                    )
                    if ep >= epochs - 3 and len(policy.tubes) >= 2:
                        nxt_idx = dataset.seq_next[b_idx]
                        n_av, n_eih, n_pr, _, _ = dataset.sample_batch(nxt_idx, device)
                        n_fa, n_fe = policy.backbone(n_av, n_eih, augment=False)
                        n_xq = policy.key_extractor(n_fa, n_fe, n_pr, lg)
                        policy.update_hebbian_edges(
                            x_q @ policy.mag_key_bank.T, n_xq @ policy.mag_key_bank.T
                        )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 2.0)
            optimizer.step()
            ep_loss_sum += float(loss.item()) * b_len
            ep_samples += b_len
            total_samples += b_len
        last_loss = ep_loss_sum / max(1, ep_samples)
        scheduler.step()

    if arm_name == "arm_er_5pct" and replay_buffers is not None:
        replay_buffers.append(extract_cpu_replay_subset(dataset, text_enc, ratio=0.05))

    if arm_name == "arm_frozen_mlp":
        for p in slot.parameters():
            p.requires_grad_(False)
            p.grad = None

    if arm_name in ("arm_dtra_rigid", "arm_dtra_elastic"):
        policy.freeze_completed_slots(active_slot_id=None)
        policy.episode_snap_back()

    dt = max(1e-4, time.perf_counter() - t0)
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        if device.type == "cuda"
        else 0.0
    )
    stat: Dict[str, Any] = {
        "train_time_s": round(dt, 2),
        "samples_per_sec": round(total_samples / dt, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "final_loss": round(last_loss, 5),
        "trainable_params": int(sum(p.numel() for p in params)),
        "active_basis_count": int(getattr(policy, "active_basis_count", 1)),
    }
    if key_stats:
        stat["key_stats"] = key_stats
    return stat


def extract_obs_tensors(
    obs: dict, lang_vec: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    av = torch.from_numpy(np.transpose(obs["agentview_image"], (2, 0, 1)).copy()).unsqueeze(0).to(device)
    eih = (
        torch.from_numpy(np.transpose(obs["robot0_eye_in_hand_image"], (2, 0, 1)).copy())
        .unsqueeze(0)
        .to(device)
    )
    prop_np = np.concatenate(
        [
            obs["robot0_eef_pos"],
            obs["robot0_gripper_qpos"],
            obs["robot0_joint_pos"],
        ],
        axis=-1,
    ).astype(np.float32)
    pr = torch.from_numpy(prop_np).unsqueeze(0).to(device)
    lg = lang_vec.unsqueeze(0)
    return av, eih, pr, lg


@dataclass
class TaskSpec:
    slot_idx: int
    suite_name: str
    bm_task_idx: int
    task_name: str
    language: str
    hdf5_path: str
    bddl_file: str
    init_states_path: str
    max_steps: int


def load_task_init_states(spec: TaskSpec):
    """PyTorch 2.6+ weights_only=True 기본값 호환 LIBERO 초기 상태 안전 로드."""
    return torch.load(spec.init_states_path, weights_only=False)


class SingleActiveEnvManager:
    """동시 다중 MuJoCo/OpenGL 자산 충돌 및 Windows CRT 파일 디스크립터(50개/환경) 누수를 원천 차단."""

    def __init__(self):
        self.cur_bddl_file: Optional[str] = None
        self.env: Optional[OffScreenRenderEnv] = None
        self._base_fd: Optional[int] = None

    @staticmethod
    def _probe_next_fd(probe_path: str) -> Optional[int]:
        try:
            fd = os.open(probe_path, os.O_RDONLY)
            os.close(fd)
            return fd
        except Exception:
            return None

    def get(self, bddl_file: str) -> OffScreenRenderEnv:
        if self.env is not None and self.cur_bddl_file == bddl_file:
            return self.env
        self.close()
        if self._base_fd is None and os.path.exists(bddl_file):
            self._base_fd = self._probe_next_fd(bddl_file)
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            hard_reset=False,
            camera_heights=128,
            camera_widths=128,
            camera_names=["agentview", "robot0_eye_in_hand"],
        )
        self.cur_bddl_file = bddl_file
        return self.env

    def close(self) -> None:
        probe_file = self.cur_bddl_file
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
            self.cur_bddl_file = None
            gc.collect()
            if os.name == "nt":
                low_fd = max(4, (self._base_fd + 1) if self._base_fd is not None else 4)
                try:
                    os.closerange(low_fd, 1024)
                except Exception:
                    pass


@torch.no_grad()
def measure_active_latency_ms(
    arm_name: str,
    policy: nn.Module,
    device: torch.device,
    n_warmup: int = 8,
    n_iters: int = 40,
) -> Dict[str, float]:
    """K개 슬롯/튜브 적재 상태에서 단일 스텝 추론 지연(전체 e2e & 백본 제외 라우팅/헤드) 실측."""
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

    e2e_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        policy.predict_action_chunk(av, eih, pr, lg)
        _sync()
        e2e_times.append((time.perf_counter() - t0) * 1000.0)

    fa, fe = policy.backbone(av, eih, augment=False)
    for _ in range(n_warmup):
        if arm_name == "arm_frozen_mlp":
            sel = int(policy.route(fa, fe, pr, lg)[0].item())
            _ = policy.slots[sel](fa, fe, pr, lg)
        elif arm_name == "arm_er_5pct":
            _ = policy.forward_train(fa, fe, pr, lg)
        else:
            sel, x_q = policy.route(fa, fe, pr, lg)
            cp, _ = policy.tubes[sel](fa, fe, pr, lg, x_q, shared_basis_bank=policy.shared_basis_bank)
            _ = policy.bspline_decoder(cp)
    _sync()

    head_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        if arm_name == "arm_frozen_mlp":
            sel = int(policy.route(fa, fe, pr, lg)[0].item())
            _ = policy.slots[sel](fa, fe, pr, lg)
        elif arm_name == "arm_er_5pct":
            _ = policy.forward_train(fa, fe, pr, lg)
        else:
            sel, x_q = policy.route(fa, fe, pr, lg)
            cp, _ = policy.tubes[sel](fa, fe, pr, lg, x_q, shared_basis_bank=policy.shared_basis_bank)
            _ = policy.bspline_decoder(cp)
        _sync()
        head_times.append((time.perf_counter() - t0) * 1000.0)

    policy.reset_episode()
    return {
        "active_latency_ms": round(float(np.median(e2e_times)), 3),
        "routing_head_ms": round(float(np.median(head_times)), 3),
    }


def _calc_policy_params_mb(policy: nn.Module) -> float:
    """공통 동결 ResNet 백본을 제외한 활성 정책(슬롯/튜브/공유기저) 파라미터 메모리(MB) 산출."""
    total_bytes = sum(
        p.numel() * p.element_size()
        for name, p in policy.named_parameters()
        if not name.startswith("backbone.")
    )
    return round(float(total_bytes) / (1024.0 * 1024.0), 3)


def _calc_dup_lang_route_acc(
    R_route: np.ndarray,
    dup_lang_indices: set,
    k_cur: int,
    t_col: int,
) -> float:
    """현재 마일스톤(0..k_cur-1) 내 동일 지시어(Duplicate-lang) 태스크들의 평균 라우팅 정확도 산출."""
    dup_in_range = [i for i in range(k_cur) if i in dup_lang_indices]
    if not dup_in_range:
        return float(np.mean([R_route[i, t_col] for i in range(k_cur)]))
    return float(np.mean([R_route[i, t_col] for i in dup_in_range]))


def eval_policy_on_task(
    policy: nn.Module,
    spec: TaskSpec,
    text_enc: LightweightTextEncoder,
    n_eval: int,
    exec_steps: int,
    device: torch.device,
    env_mgr: Optional[SingleActiveEnvManager] = None,
) -> Dict[str, float]:
    t_eval_start = time.perf_counter()
    init_states = load_task_init_states(spec)
    lang_vec = text_enc.encode(spec.language, device)

    owns_env = env_mgr is None
    if env_mgr is not None:
        env = env_mgr.get(spec.bddl_file)
    else:
        env = OffScreenRenderEnv(
            bddl_file_name=spec.bddl_file,
            camera_heights=128,
            camera_widths=128,
            camera_names=["agentview", "robot0_eye_in_hand"],
        )

    successes = 0
    route_hits = 0
    route_total = 0
    alphas: List[float] = []
    dz_norms: List[float] = []
    chunk_post_snaps: List[float] = []
    post_snaps: List[float] = []
    steps_done: List[int] = []
    max_steps = spec.max_steps

    try:
        policy.eval()
        for ep in range(n_eval):
            policy.reset_episode()
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))

            step_idx = 0
            ep_success = False
            while step_idx < max_steps:
                av, eih, pr, lg = extract_obs_tensors(obs, lang_vec, device)
                chunk, sel, a_val, dz_val = policy.predict_action_chunk(av, eih, pr, lg)
                chunk_np = np.clip(chunk[0].cpu().numpy(), -1.0, 1.0)
                route_total += 1
                if sel == spec.slot_idx:
                    route_hits += 1
                alphas.append(a_val)
                dz_norms.append(dz_val)

                for k in range(min(exec_steps, chunk_np.shape[0])):
                    obs, _, _, _ = env.step(chunk_np[k])
                    step_idx += 1
                    if env.check_success():
                        ep_success = True
                        break
                    if step_idx >= max_steps:
                        break
                # 매 청크 종료 시 Two-Timescale Chunk Snap-Back (delta_z -> 0.000, episodic_trace 유지)
                if hasattr(policy, "chunk_snap_back"):
                    _, c_post = policy.chunk_snap_back()
                    chunk_post_snaps.append(c_post)
                if ep_success:
                    break

            if hasattr(policy, "episode_snap_back"):
                _, post_norm = policy.episode_snap_back()
            else:
                _, post_norm = policy.snap_back()
            post_snaps.append(post_norm)
            if ep_success:
                successes += 1
                steps_done.append(step_idx)
    finally:
        if owns_env:
            env.close()

    return {
        "sr": float(successes / max(1, n_eval)),
        "route_acc": float(route_hits / max(1, route_total)),
        "mean_alpha": float(np.mean(alphas)) if alphas else 0.0,
        "mean_delta_z": float(np.mean(dz_norms)) if dz_norms else 0.0,
        "chunk_post_snap_norm": float(np.max(chunk_post_snaps)) if chunk_post_snaps else 0.0,
        "post_snap_norm": float(np.max(post_snaps)) if post_snaps else 0.0,
        "mean_success_steps": float(np.mean(steps_done)) if steps_done else float(max_steps),
        "eval_time_s": round(time.perf_counter() - t_eval_start, 2),
    }


# ---------------------------------------------------------------------------
# 5. 메인 벤치마크 오케스트레이션 (100-태스크 스트리밍 + 마일스톤 평가 지원)
# ---------------------------------------------------------------------------


def _parse_int_csv(raw: str) -> List[int]:
    if not raw.strip():
        return []
    return sorted({int(tok.strip()) for tok in raw.split(",") if tok.strip()})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DTRA-LIBERO 24GB VRAM Multimodal Continual Learning")
    p.add_argument(
        "--suite",
        type=str,
        default="libero_goal",
        choices=["libero_goal", "libero_10", "libero_90", "libero_spatial", "libero_object", "libero_100"],
    )
    p.add_argument(
        "--suites",
        type=str,
        default="",
        help="쉼표 구분 멀티 스위트 체이닝 (예: libero_10,libero_90). 미지정 시 --suite 사용",
    )
    p.add_argument("--max-tasks", type=int, default=5, help="순차 학습할 최대 태스크 수 (예: 10, 100)")
    p.add_argument("--task-indices", nargs="+", type=int, default=None, help="단일 스위트 명시적 태스크 ID 목록")
    p.add_argument("--max-demos", type=int, default=None, help="태스크당 시연 수 제한 (None=전체 50)")
    p.add_argument("--stride", type=int, default=1, help="시연 프레임 샘플링 스트라이드")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--num-cp", type=int, default=8)
    p.add_argument("--exec-steps", type=int, default=8)
    p.add_argument("--n-eval", type=int, default=5)
    p.add_argument("--max-rollout-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--arms",
        nargs="+",
        default=["arm_frozen_mlp", "arm_dtra_elastic", "arm_dtra_rigid", "arm_er_5pct"],
    )
    p.add_argument(
        "--stream-tasks",
        action="store_true",
        help="현재 태스크 T_k 1개만 VRAM에 올렸다 학습 직후 즉시 해제 (K=100 Peak VRAM <= 4GB)",
    )
    p.add_argument(
        "--eval-milestones",
        type=str,
        default="",
        help="쉼표 구분 마일스톤 평가 시점 (예: 10,30,50,70,100). 비마일스톤은 대각선 R[k,k]만 평가",
    )
    p.add_argument("--fast-eval-endpoints", action="store_true", help="중간 R 대신 t==i 및 t==K-1만 평가")
    p.add_argument(
        "--resume",
        action="store_true",
        help="기존 out-json이 있으면 completed arm과 tasks 메타데이터를 보존하고 미완료 arm만 이어서 실행",
    )
    p.add_argument("--out-json", type=str, default="")
    return p.parse_args()


def build_task_specs(args: argparse.Namespace) -> Tuple[List[str], List[TaskSpec]]:
    suite_list = [s.strip() for s in args.suites.split(",") if s.strip()] if args.suites.strip() else [args.suite]
    datasets_root = get_libero_path("datasets")
    init_root = get_libero_path("init_states")

    specs: List[TaskSpec] = []
    for sname in suite_list:
        bm = benchmark.get_benchmark(sname)(task_order_index=0)
        if args.task_indices is not None and len(suite_list) == 1:
            ids = [int(x) for x in args.task_indices]
        else:
            ids = list(range(bm.get_num_tasks()))

        step_cap = args.max_rollout_steps or (
            250
            if sname in ("libero_goal", "libero_spatial", "libero_object")
            else (320 if sname == "libero_90" else 420)
        )
        for bm_i in ids:
            if len(specs) >= args.max_tasks:
                break
            t = bm.get_task(bm_i)
            hdf5_path = os.path.join(datasets_root, bm.get_task_demonstration(bm_i))
            if not os.path.exists(hdf5_path):
                raise FileNotFoundError(
                    f"Missing LIBERO dataset HDF5 for {sname}[{bm_i}] ({t.name}): {hdf5_path}"
                )
            specs.append(
                TaskSpec(
                    slot_idx=len(specs),
                    suite_name=sname,
                    bm_task_idx=bm_i,
                    task_name=t.name,
                    language=t.language,
                    hdf5_path=hdf5_path,
                    bddl_file=bm.get_task_bddl_file_path(bm_i),
                    init_states_path=os.path.join(init_root, t.problem_folder, t.init_states_file),
                    max_steps=step_cap,
                )
            )
        if len(specs) >= args.max_tasks:
            break
    return suite_list, specs


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    suite_list, task_specs = build_task_specs(args)
    num_tasks = len(task_specs)
    lang_counts = Counter(sp.language.strip().lower() for sp in task_specs)
    dup_lang_indices = {sp.slot_idx for sp in task_specs if lang_counts[sp.language.strip().lower()] > 1}

    suite_tag = "_".join(suite_list)
    default_out = (
        str(PROJECT_ROOT / "outputs" / "libero" / f"benchmark_dtra_{args.suite}.json")
        if (not args.suites.strip() and not args.stream_tasks and not args.eval_milestones)
        else str(PROJECT_ROOT / "outputs" / "libero" / f"benchmark_dtra_{suite_tag}_k{num_tasks}.json")
    )
    out_path = args.out_json or default_out

    raw_milestones = _parse_int_csv(args.eval_milestones)
    milestone_list = [m for m in raw_milestones if 1 <= m <= num_tasks]
    if raw_milestones and num_tasks not in milestone_list and num_tasks > 0:
        milestone_list.append(num_tasks)
    milestone_set = set(milestone_list)

    print(
        f"=== DTRA-LIBERO Benchmark | suites={suite_list} | K={num_tasks} | stream={args.stream_tasks} | "
        f"milestones={milestone_list or 'all'} | batch={args.batch_size} | epochs={args.epochs} | "
        f"n_eval={args.n_eval} | device={device} ===",
        flush=True,
    )

    cached_datasets: Optional[List[LiberoTaskDataset]] = None
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

    if not args.stream_tasks:
        cached_datasets = []
        total_mb = 0.0
        for sp in task_specs:
            ds = LiberoTaskDataset(
                task_idx=sp.slot_idx,
                task_name=sp.task_name,
                language=sp.language,
                hdf5_path=sp.hdf5_path,
                horizon=args.horizon,
                num_cp=args.num_cp,
                stride=args.stride,
                max_demos=args.max_demos,
                cache_on_gpu=(total_mb < 9000.0),
                device=device,
            )
            cached_datasets.append(ds)
            ds_mb = ds.data_bytes / (1024.0 * 1024.0)
            total_mb += ds_mb
            task_meta_rows[sp.slot_idx].update(
                {
                    "demos": ds.num_demos,
                    "chunks": ds.num_samples,
                    "bspline_rmse": round(ds.bspline_fit_rmse, 5),
                }
            )
            print(
                f"  [Data T{sp.slot_idx} ({sp.suite_name}:{sp.bm_task_idx})] {sp.task_name} | "
                f"demos={ds.num_demos} | chunks={ds.num_samples} | bspline_rmse={ds.bspline_fit_rmse:.4f} | "
                f"{ds_mb:.1f} MB ({ds.storage_device})",
                flush=True,
            )
    else:
        print("  [Stream Mode] Single-task CUDA upload + immediate post-task release enabled.", flush=True)

    # 2. 공통 동결 백본 & B-Spline 디코더 생성
    backbone = FrozenResNet18DualCam().to(device)
    key_extractor = MultimodalMagneticKeyExtractor(d_topo=64, d_lang=128, seed=args.seed).to(device)
    bspline_decoder = BSplineDecoder(num_control_points=args.num_cp, horizon=args.horizon, degree=3).to(
        device
    )
    text_enc = LightweightTextEncoder(dim=128)

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
            prev_tasks_by_name = {
                (t.get("name") or t.get("task_name")): t
                for t in prev_res.get("tasks", [])
                if isinstance(t, dict)
            }
            for tm in results["tasks"]:
                pt = prev_tasks_by_name.get(tm["name"])
                if pt:
                    for k in ("demos", "chunks", "bspline_rmse", "data_mb"):
                        if k in pt and k not in tm:
                            tm[k] = pt[k]
            done_arms = [
                an for an, av in results["arms"].items() if isinstance(av, dict) and av.get("status") == "completed"
            ]
            print(f"  [Resume] Loaded {out_path} (completed arms: {done_arms})", flush=True)
        except Exception as exc:
            print(f"  [Resume Warning] Failed to load {out_path}: {exc}", flush=True)

    env_mgr = SingleActiveEnvManager()
    try:
        for arm_name in args.arms:
            prev_arm = results["arms"].get(arm_name)
            if args.resume and isinstance(prev_arm, dict) and prev_arm.get("status") == "completed":
                print(f"\n--- Skipping Completed Arm (resume): {arm_name} ---", flush=True)
                continue
            set_seed(args.seed)
            arm_t0 = time.perf_counter()
            arm_eval_time_s = 0.0
            eval_plan = [
                len(range(k))
                if (milestone_set and k in milestone_set)
                else (1 if (milestone_set or (args.fast_eval_endpoints and k < num_tasks)) else k)
                for k in range(1, num_tasks + 1)
            ]
            total_evals_arm = sum(eval_plan)
            done_evals_arm = 0
            print(f"\n--- Running Arm: {arm_name} (planned_evals={total_evals_arm}) ---", flush=True)
            if arm_name == "arm_frozen_mlp":
                policy = FrozenMLPMultimodalPolicy(
                    backbone, key_extractor, horizon=args.horizon, action_dim=7
                ).to(device)
            elif arm_name == "arm_er_5pct":
                policy = ExperienceReplayPolicy(
                    backbone, bspline_decoder, ctx_dim=256, num_cp=args.num_cp, action_dim=7
                ).to(device)
            elif arm_name == "arm_dtra_rigid":
                policy = DTRALiberoPolicy(
                    backbone, key_extractor, bspline_decoder, elastic=False, d_topo=64, num_cp=args.num_cp
                ).to(device)
            elif arm_name == "arm_dtra_elastic":
                policy = DTRALiberoPolicy(
                    backbone, key_extractor, bspline_decoder, elastic=True, d_topo=64, num_cp=args.num_cp
                ).to(device)
            else:
                raise ValueError(f"Unsupported arm: {arm_name}")

            R_sr = np.zeros((num_tasks, num_tasks), dtype=np.float32)
            R_route = np.zeros((num_tasks, num_tasks), dtype=np.float32)
            measured_mask = np.zeros((num_tasks, num_tasks), dtype=bool)
            replay_buffers: Optional[List[Dict[str, torch.Tensor]]] = (
                [] if arm_name == "arm_er_5pct" else None
            )
            train_logs: List[Dict[str, Any]] = []
            eval_details: Dict[str, dict] = {}
            arm_milestones: Dict[str, dict] = {}
            ckpt_pt = f"{out_path}.{arm_name}.pt"
            saved_t_cur = -1
            if args.resume and isinstance(prev_arm, dict) and prev_arm.get("status") == "running":
                if isinstance(prev_arm.get("eval_details"), dict):
                    eval_details = dict(prev_arm["eval_details"])
                if isinstance(prev_arm.get("milestones"), dict):
                    arm_milestones = dict(prev_arm["milestones"])
                if os.path.exists(ckpt_pt):
                    try:
                        ckpt_obj = torch.load(ckpt_pt, map_location=device, weights_only=False)
                        saved_t_cur = int(ckpt_obj.get("t_cur", -1))
                        policy = ckpt_obj["policy"].to(device)
                        key_extractor = ckpt_obj["key_extractor"].to(device)
                        policy.key_extractor = key_extractor
                        train_logs = list(ckpt_obj.get("train_logs", []))
                        replay_buffers = ckpt_obj.get("replay_buffers", replay_buffers)
                        print(f"  [Model Resume] Loaded {ckpt_pt} at t_cur={saved_t_cur}", flush=True)
                    except Exception as exc:
                        print(f"  [Model Resume Warning] {exc}", flush=True)
                        saved_t_cur = -1
                if eval_details:
                    print(
                        f"  [Mid-Arm Resume] Restored {len(eval_details)} completed rollout evals "
                        f"and milestones {list(arm_milestones.keys())} for {arm_name}",
                        flush=True,
                    )
            results["arms"][arm_name] = {
                "status": "running",
                "completed_tasks": 0,
                "milestones": arm_milestones,
                "eval_details": eval_details,
            }

            for t_cur in range(num_tasks):
                sp = task_specs[t_cur]
                if t_cur <= saved_t_cur and len(train_logs) > t_cur:
                    tr_stat = train_logs[t_cur]
                else:
                    if cached_datasets is not None:
                        cur_ds = cached_datasets[t_cur]
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
                            # 단일 태스크 예외적 대용량 방어 (Peak VRAM <= 4GB 보장)
                            cur_ds.agent_u8 = cur_ds.agent_u8.cpu()
                            cur_ds.eih_u8 = cur_ds.eih_u8.cpu()
                            cur_ds.proprio = cur_ds.proprio.cpu()
                            cur_ds.actions_gt = cur_ds.actions_gt.cpu()
                            cur_ds.ctrl_gt = cur_ds.ctrl_gt.cpu()
                            cur_ds.storage_device = torch.device("cpu")
                            torch.cuda.empty_cache()
                        if "chunks" not in task_meta_rows[t_cur]:
                            task_meta_rows[t_cur].update(
                                {
                                    "demos": cur_ds.num_demos,
                                    "chunks": cur_ds.num_samples,
                                    "bspline_rmse": round(cur_ds.bspline_fit_rmse, 5),
                                    "data_mb": round(ds_mb, 1),
                                }
                            )
                        print(
                            f"  [Stream Load T{t_cur} ({sp.suite_name}:{sp.bm_task_idx})] {sp.task_name} | "
                            f"chunks={cur_ds.num_samples} | {ds_mb:.1f} MB ({cur_ds.storage_device})",
                            flush=True,
                        )

                    tr_stat = train_arm_on_task(
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

                    if cached_datasets is None:
                        del cur_ds
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    try:
                        torch.save(
                            {
                                "t_cur": t_cur,
                                "policy": policy,
                                "key_extractor": key_extractor,
                                "train_logs": train_logs,
                                "replay_buffers": replay_buffers,
                            },
                            ckpt_pt,
                        )
                    except Exception:
                        pass

                ks = tr_stat.get("key_stats")
                key_msg = (
                    f" | key_cos={ks['max_cos_before']:.3f}->{ks['max_cos_after']:.3f} "
                    f"(q_acc={ks['query_route_acc']*100:.1f}%, bases={tr_stat['active_basis_count']})"
                    if ks
                    else ""
                )
                print(
                    f"  [Train T{t_cur}] time={tr_stat['train_time_s']}s | "
                    f"{tr_stat['samples_per_sec']} samp/s | vram_peak={tr_stat['peak_vram_mb']} MB | "
                    f"loss={tr_stat['final_loss']:.4f} | params={tr_stat['trainable_params']:,}{key_msg}",
                    flush=True,
                )

                k_cur = t_cur + 1
                is_milestone = (k_cur in milestone_set) if milestone_set else False
                if milestone_set:
                    eval_indices = list(range(k_cur)) if is_milestone else [t_cur]
                elif args.fast_eval_endpoints and t_cur < num_tasks - 1:
                    eval_indices = [t_cur]
                else:
                    eval_indices = list(range(k_cur))

                for ev_i in eval_indices:
                    ev_key = f"t{t_cur}_eval{ev_i}"
                    if args.resume and ev_key in eval_details:
                        ev = eval_details[ev_key]
                        tag_str = " [Restored]"
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
                        tag_str = ""
                    R_sr[ev_i, t_cur] = ev["sr"]
                    R_route[ev_i, t_cur] = ev["route_acc"]
                    measured_mask[ev_i, t_cur] = True
                    done_evals_arm += 1
                    arm_eval_time_s += float(ev.get("eval_time_s", 0.0))
                    print(
                        f"    [Eval after T{t_cur} -> T{ev_i}]{tag_str} SR={ev['sr']*100:5.1f}% | "
                        f"RouteAcc={ev['route_acc']*100:5.1f}% | alpha={ev['mean_alpha']:.3f} | "
                        f"|dz*|={ev['mean_delta_z']:.4f} -> snap={ev['post_snap_norm']:.4f} "
                        f"({ev['eval_time_s']}s)",
                        flush=True,
                    )

                arm_elapsed_s = time.perf_counter() - arm_t0
                cum_train_s = sum(x["train_time_s"] for x in train_logs)
                avg_train_task = cum_train_s / max(1, k_cur)
                avg_eval_call = arm_eval_time_s / max(1, done_evals_arm)
                est_rem_s = avg_train_task * (num_tasks - k_cur) + avg_eval_call * (total_evals_arm - done_evals_arm)
                print(
                    f"  [Progress {arm_name} T{k_cur}/{num_tasks} | evals {done_evals_arm}/{total_evals_arm}] "
                    f"elapsed={arm_elapsed_s/60:.1f}m | eta_arm={est_rem_s/60:.1f}m",
                    flush=True,
                )

                if is_milestone:
                    prev_ms = arm_milestones.get(str(k_cur)) if (args.resume and t_cur < saved_t_cur) else None
                    R_sub = forward_fill_R_for_milestone(R_sr, measured_mask, k_cur)
                    m_metrics = calc_cl_metrics(R_sub)
                    m_routes = [float(R_route[i, t_cur]) for i in range(k_cur)]
                    m_dup_route = _calc_dup_lang_route_acc(R_route, dup_lang_indices, k_cur, t_cur)
                    if prev_ms is not None:
                        lat = {
                            "active_latency_ms": prev_ms.get("active_latency_ms", 0.0),
                            "routing_head_ms": prev_ms.get("routing_head_ms", 0.0),
                        }
                        m_active_bases = int(prev_ms.get("active_basis_count", k_cur))
                        m_params_mb = float(prev_ms.get("total_params_mb", 0.0))
                    else:
                        lat = measure_active_latency_ms(arm_name, policy, device)
                        m_active_bases = int(getattr(policy, "active_basis_count", k_cur))
                        m_params_mb = _calc_policy_params_mb(policy)
                    m_peak_vram = float(max(x["peak_vram_mb"] for x in train_logs[:k_cur]))
                    m_cum_time = float(cum_train_s)
                    m_entry = {
                        "milestone_k": k_cur,
                        "mean_sr": m_metrics["mean_final_sr"],
                        "forgetting_rate": m_metrics["forgetting_rate"],
                        "bwt": m_metrics["bwt"],
                        "fwt": m_metrics["fwt"],
                        "mean_route_acc": float(np.mean(m_routes)),
                        "dup_lang_route_acc": m_dup_route,
                        "active_basis_count": m_active_bases,
                        "total_params_mb": m_params_mb,
                        "active_latency_ms": lat["active_latency_ms"],
                        "routing_head_ms": lat["routing_head_ms"],
                        "peak_vram_mb": m_peak_vram,
                        "cumulative_train_time_s": round(m_cum_time, 2),
                        "wall_elapsed_s": round(arm_elapsed_s, 1),
                        "eta_arm_s": round(est_rem_s, 1),
                        "eval_sr_vec": [float(R_sr[i, t_cur]) for i in range(k_cur)],
                        "eval_route_vec": m_routes,
                    }
                    arm_milestones[str(k_cur)] = m_entry
                    results["arms"][arm_name].update(
                        {
                            "completed_tasks": k_cur,
                            "active_basis_count": m_active_bases,
                            "total_params_mb": m_params_mb,
                            "milestones": arm_milestones,
                            "R_sr_partial": R_sub.tolist(),
                            "R_route_partial": R_route[:k_cur, :k_cur].tolist(),
                        }
                    )
                    save_json_atomic(out_path, results)
                    print(
                        f"  *** [Milestone K={k_cur} | {arm_name}] MeanSR={m_entry['mean_sr']*100:.1f}% | "
                        f"FR={m_entry['forgetting_rate']*100:.1f}% | BWT={m_entry['bwt']*100:+.1f}% | "
                        f"FWT={m_entry['fwt']*100:+.1f}% | RouteAcc={m_entry['mean_route_acc']*100:.1f}% "
                        f"(DupLang={m_dup_route*100:.1f}%) | Bases={m_active_bases} ({m_params_mb:.2f}MB) | "
                        f"Latency={m_entry['active_latency_ms']:.2f}ms (head={m_entry['routing_head_ms']:.2f}ms) | "
                        f"PeakVRAM={m_peak_vram:.0f}MB | elapsed={arm_elapsed_s/60:.1f}m | eta={est_rem_s/60:.1f}m -> Checkpoint Saved ***",
                        flush=True,
                    )

            R_filled = forward_fill_R_for_milestone(R_sr, measured_mask, num_tasks)
            metrics = calc_cl_metrics(R_filled)
            final_routes = [float(R_route[i, num_tasks - 1]) for i in range(num_tasks)]
            final_dup_route = _calc_dup_lang_route_acc(R_route, dup_lang_indices, num_tasks, num_tasks - 1)
            final_lat = measure_active_latency_ms(arm_name, policy, device)
            final_active_bases = int(getattr(policy, "active_basis_count", num_tasks))
            final_params_mb = _calc_policy_params_mb(policy)
            arm_summary = {
                "status": "completed",
                "completed_tasks": num_tasks,
                "mean_final_sr": metrics["mean_final_sr"],
                "forgetting_rate": metrics["forgetting_rate"],
                "bwt": metrics["bwt"],
                "fwt": metrics["fwt"],
                "mean_final_route_acc": float(np.mean(final_routes)),
                "dup_lang_route_acc": final_dup_route,
                "active_basis_count": final_active_bases,
                "total_params_mb": final_params_mb,
                "active_latency_ms": final_lat["active_latency_ms"],
                "routing_head_ms": final_lat["routing_head_ms"],
                "max_vram_mb": float(max(x["peak_vram_mb"] for x in train_logs)),
                "mean_samples_per_sec": float(np.mean([x["samples_per_sec"] for x in train_logs])),
                "wall_elapsed_s": round(time.perf_counter() - arm_t0, 2),
                "total_eval_time_s": round(arm_eval_time_s, 2),
                "milestones": arm_milestones,
                "R_sr": R_filled.tolist(),
                "R_route": R_route.tolist(),
                "train_logs": train_logs,
                "final_eval": {f"T{i}": eval_details[f"t{num_tasks-1}_eval{i}"] for i in range(num_tasks)},
            }
            results["arms"][arm_name] = arm_summary
            save_json_atomic(out_path, results)
            print(
                f"  ==> [{arm_name}] MeanFinalSR={arm_summary['mean_final_sr']*100:.1f}% | "
                f"FR={arm_summary['forgetting_rate']*100:.1f}% | BWT={arm_summary['bwt']*100:+.1f}% | "
                f"FWT={arm_summary['fwt']*100:+.1f}% | RouteAcc={arm_summary['mean_final_route_acc']*100:.1f}% "
                f"(DupLang={final_dup_route*100:.1f}%) | Bases={final_active_bases} ({final_params_mb:.2f}MB) | "
                f"Latency={arm_summary['active_latency_ms']:.2f}ms | PeakVRAM={arm_summary['max_vram_mb']:.0f}MB",
                flush=True,
            )

            del policy, replay_buffers
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        env_mgr.close()

    print(f"\nSaved LIBERO benchmark results to: {out_path}", flush=True)


if __name__ == "__main__":
    main()

