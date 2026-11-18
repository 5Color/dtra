"""tools/bspline_core.py

B-Spline 기하학 기저 및 정규화 유계 최소제곱(BVLS) 핵심 라이브러리.
- Cox-de Boor 점화식 기반 B-Spline 기저 행렬 생성
- 2차 차분 평활화 정규화 (Smoothness Regularization)
- 유계 최소제곱법 (Bounded Variable Least Squares, BVLS)
- Chunk 이음매 C^1 / C^0 연속성 바인딩 (Collinear Tangent Enforcement)
- PyTorch 호환 선형 해석적 재구성 연산자 (Differentiable Matrix Reconstruction)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt
import torch


# ---------------------------------------------------------------------------
# 1. B-Spline 매듭 벡터 및 기저 행렬 생성
# ---------------------------------------------------------------------------

def build_clamped_knot_vector(num_control_points: int, degree: int = 3) -> np.ndarray:
    """양 끝점에서 곡선이 시작/끝 제어점을 통과하도록 하는 Clamped 매듭 벡터 생성.
    
    총 매듭점 수 = M + degree + 1
    내부 매듭점 수 = M - degree - 1
    """
    m = num_control_points
    k = degree
    if m < k + 1:
        raise ValueError(f"제어점 개수 M({m})은 차수 degree({k}) + 1 이상이어야 합니다.")

    num_internal = m - k - 1
    if num_internal > 0:
        internal_knots = np.linspace(0.0, 1.0, num_internal + 2)[1:-1]
    else:
        internal_knots = np.array([], dtype=np.float64)

    return np.concatenate([
        np.zeros(k + 1, dtype=np.float64),
        internal_knots,
        np.ones(k + 1, dtype=np.float64)
    ])


def parameterize_points(points: np.ndarray, method: str = "chord") -> np.ndarray:
    """2D 점열에 매개변수 t in [0, 1] 할당.
    
    Args:
        points: (N, 2) 좌표 배열
        method: 'chord' (누적 유클리드 호 길이), 'uniform' (균등)
    """
    n = len(points)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    if method == "uniform":
        return np.linspace(0.0, 1.0, n, dtype=np.float64)

    diffs = np.diff(points, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    total_dist = float(np.sum(dists))

    if total_dist < 1e-9:
        return np.linspace(0.0, 1.0, n, dtype=np.float64)

    s = np.concatenate([[0.0], np.cumsum(dists)])
    t = s / s[-1]
    t[-1] = 1.0
    return t


def build_design_matrix(
    t_eval: np.ndarray,
    num_control_points: int,
    degree: int = 3,
    knots: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """t_eval 위치에서 B-Spline 기저 함수값 행렬 B (N, M) 계산."""
    if knots is None:
        knots = build_clamped_knot_vector(num_control_points, degree)
    t_eval = np.clip(np.asarray(t_eval, dtype=np.float64), 0.0, 1.0)
    b_mat = interp.BSpline.design_matrix(t_eval, knots, degree).toarray()
    return b_mat, knots


def build_difference_matrices(m: int) -> Tuple[np.ndarray, np.ndarray]:
    """1차 차분 행렬 D1 (m-1, m) 및 2차 차분 행렬 D2 (m-2, m) 구성.
    
    D1: 인접 제어점 거리 벌어짐 페널티
    D2: 인접 제어점 꺾임/가속도/곡률 페널티 (루프 아티팩트 원천 억제)
    """
    if m >= 2:
        d1 = np.zeros((m - 1, m), dtype=np.float64)
        for i in range(m - 1):
            d1[i, i] = -1.0
            d1[i, i + 1] = 1.0
    else:
        d1 = np.zeros((0, m), dtype=np.float64)

    if m >= 3:
        d2 = np.zeros((m - 2, m), dtype=np.float64)
        for i in range(m - 2):
            d2[i, i] = 1.0
            d2[i, i + 1] = -2.0
            d2[i, i + 2] = 1.0
    else:
        d2 = np.zeros((0, m), dtype=np.float64)

    return d1, d2


# ---------------------------------------------------------------------------
# 2. 정규화 유계 최소제곱(BVLS) 피팅 및 C^1 연속성 바인딩
# ---------------------------------------------------------------------------

def fit_bspline_chunk_bvls(
    points: np.ndarray,
    num_control_points: int = 6,
    degree: int = 3,
    param_method: str = "chord",
    smoothness_weight: float = 0.05,
    bounds: Tuple[float, float] = (-0.05, 1.05),
    c1_anchor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    c0_anchor: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """점열을 B-Spline 제어점 [M, 2]으로 최적 피팅.
    
    C^1 연속성 바인딩:
        c1_anchor=(P_start, P_tangent)가 주어지면,
        P[0] = P_start, P[1] = P_tangent 로 정확히 고정하고
        나머지 P[2:]만 BVLS 최적화로 푼다. (대수적 변수 소거법 사용)
    
    C^0 연속성 바인딩:
        c0_anchor=P_start 만 주어지면,
        P[0] = P_start 로 고정하고 나머지 P[1:]만 BVLS 최적화로 푼다.
    
    Args:
        points: (N, 2) 궤적 점열
        num_control_points: 목표 제어점 수 M
        degree: 차수 (기본 3)
        param_method: 'chord' 또는 'uniform'
        smoothness_weight: 2차 차분 평활화 가중치 (루프 방지)
        bounds: 제어점 허용 좌표 범위 [-0.05, 1.05]
        c1_anchor: (P0, P1) 튜플 (C^1 연속성 보장)
        c0_anchor: P0 (C^0 시작점 연속성 보장)
        
    Returns:
        (control_points, knots, degree)
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        raise ValueError("입력 점열이 비어 있습니다.")

    m = num_control_points
    k = min(degree, m - 1)

    if n <= 1:
        cp = np.repeat(points[:1], m, axis=0)
        knots = build_clamped_knot_vector(m, k)
        return cp, knots, k

    t_eval = parameterize_points(points, method=param_method)
    knots = build_clamped_knot_vector(m, k)
    b_mat = interp.BSpline.design_matrix(t_eval, knots, k).toarray()

    # 정규화 차분 행렬
    d1, d2 = build_difference_matrices(m)
    scale_n = n / 16.0  # 기본 청크 크기(16) 대비 스케일
    lam2 = float(smoothness_weight * scale_n)
    lam1 = float(0.1 * lam2)

    # 설계행렬 B (H x M)과 정규화된 2차 차분행렬 D2 ((M-2) x M)을 수직 결합해
    # 최소제곱 목적식 ||B P - Y||^2 + lam2 ||D2 P||^2 (C2 곡률 평활화) 구성
    blocks_a = [b_mat]
    blocks_bx = [points[:, 0]]
    blocks_by = [points[:, 1]]

    if len(d2) > 0 and lam2 > 0:
        blocks_a.append(np.sqrt(lam2) * d2)
        blocks_bx.append(np.zeros(len(d2), dtype=np.float64))
        blocks_by.append(np.zeros(len(d2), dtype=np.float64))

    if len(d1) > 0 and lam1 > 0:
        blocks_a.append(np.sqrt(lam1) * d1)
        blocks_bx.append(np.zeros(len(d1), dtype=np.float64))
        blocks_by.append(np.zeros(len(d1), dtype=np.float64))

    a_mat = np.vstack(blocks_a)
    bx = np.concatenate(blocks_bx)
    by = np.concatenate(blocks_by)

    # 1. C^1 연속성 바인딩 (P0, P1 고정 -> P2:M 변수 소거 풀이)
    if c1_anchor is not None and m >= 3:
        p0, p1 = np.asarray(c1_anchor[0], dtype=np.float64), np.asarray(c1_anchor[1], dtype=np.float64)
        p0 = np.clip(p0, bounds[0], bounds[1])
        p1 = np.clip(p1, bounds[0], bounds[1])

        p_fixed = np.vstack([p0, p1])  # (2, 2)
        a_fixed = a_mat[:, :2]
        a_free = a_mat[:, 2:]

        bx_sub = bx - a_fixed @ p_fixed[:, 0]
        by_sub = by - a_fixed @ p_fixed[:, 1]

        num_free = m - 2
        lb = np.full(num_free, bounds[0], dtype=np.float64)
        ub = np.full(num_free, bounds[1], dtype=np.float64)

        rx = opt.lsq_linear(a_free, bx_sub, bounds=(lb, ub), tol=1e-6)
        ry = opt.lsq_linear(a_free, by_sub, bounds=(lb, ub), tol=1e-6)

        cp_free = np.column_stack([rx.x, ry.x])
        cp = np.vstack([p_fixed, cp_free])

    # 2. C^0 연속성 바인딩 (P0만 고정 -> P1:M 변수 소거 풀이)
    elif c0_anchor is not None and m >= 2:
        p0 = np.asarray(c0_anchor, dtype=np.float64)
        p0 = np.clip(p0, bounds[0], bounds[1])

        a_fixed = a_mat[:, :1]
        a_free = a_mat[:, 1:]

        bx_sub = bx - a_fixed @ np.array([p0[0]])
        by_sub = by - a_fixed @ np.array([p0[1]])

        num_free = m - 1
        lb = np.full(num_free, bounds[0], dtype=np.float64)
        ub = np.full(num_free, bounds[1], dtype=np.float64)

        rx = opt.lsq_linear(a_free, bx_sub, bounds=(lb, ub), tol=1e-6)
        ry = opt.lsq_linear(a_free, by_sub, bounds=(lb, ub), tol=1e-6)

        cp_free = np.column_stack([rx.x, ry.x])
        cp = np.vstack([p0[None, :], cp_free])

    # 3. 비구속 BVLS 풀이
    else:
        lb = np.full(m, bounds[0], dtype=np.float64)
        ub = np.full(m, bounds[1], dtype=np.float64)

        rx = opt.lsq_linear(a_mat, bx, bounds=(lb, ub), tol=1e-6)
        ry = opt.lsq_linear(a_mat, by, bounds=(lb, ub), tol=1e-6)

        cp = np.column_stack([rx.x, ry.x])

    cp = np.clip(cp, bounds[0], bounds[1])
    return cp, knots, k


# ---------------------------------------------------------------------------
# 3. 곡선 재구성 (Numpy & PyTorch Analytic Operator)
# ---------------------------------------------------------------------------

def reconstruct_bspline_np(
    control_points: np.ndarray,
    knots: np.ndarray,
    degree: int = 3,
    num_points: int = 16,
    t_eval: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Numpy/SciPy 기반 B-Spline 곡선 샘플링."""
    spl = interp.BSpline(knots, control_points, degree)
    if t_eval is None:
        t_eval = np.linspace(0.0, 1.0, num_points)
    t_eval = np.clip(t_eval, 0.0, 1.0)
    return spl(t_eval)


class BSplineDecoder(torch.nn.Module):
    """PyTorch 미분 가능한 해석적 B-Spline 재구성 계층 (Frozen Matrix Operator).
    
    P_chunk [B, M, 2] -> Trajectory [B, H, 2]
    수식: Trajectory = B_matrix @ P_chunk
    """

    def __init__(self, num_control_points: int = 6, horizon: int = 16, degree: int = 3):
        super().__init__()
        self.num_control_points = num_control_points
        self.horizon = horizon
        self.degree = degree

        knots = build_clamped_knot_vector(num_control_points, degree)
        t_eval = np.linspace(0.0, 1.0, horizon)
        b_mat, _ = build_design_matrix(t_eval, num_control_points, degree, knots)

        # Frozen 기저 행렬 등록 (학습되지 않는 불변 물리 기저)
        self.register_buffer("b_mat", torch.tensor(b_mat, dtype=torch.float32))
        self.register_buffer("knots", torch.tensor(knots, dtype=torch.float32))

    def forward(self, control_points: torch.Tensor) -> torch.Tensor:
        """
        미분 가능한 선형 행렬곱 기반 Cubic B-Spline 디코더 (학습·추론 공용).

        수식: Y_hat = B @ P, 여기서 B in R^(H x M) 은 사전 계산된 클램프 기저 행렬.
        Args:
            control_points: (B, M, D) 또는 (B, M, 2) 정규화된 제어점 텐서
        Returns:
            trajectory: (B, H, D) 또는 (B, H, 2) 복원된 물리 궤적 텐서
        """
        # 선형 사상: (H, M) x (B, M, D) -> (B, H, D) (경사 역전파 완전 지원)
        return torch.matmul(self.b_mat, control_points)


# ---------------------------------------------------------------------------
# 4. C^1 연속성 검증 및 품질 메트릭 유틸리티
# ---------------------------------------------------------------------------

def compute_c1_tangent_discrepancy(cp_prev: np.ndarray, cp_next: np.ndarray) -> Tuple[float, float]:
    """두 연속 청크 간 위치(C0) 및 접선 벡터(C1) 불일치 오차 계산.
    
    C0 오차: || P_next[0] - P_prev[-1] ||
    C1 접선 불일치: || (P_next[1] - P_next[0]) - (P_prev[-1] - P_prev[-2]) ||
    """
    c0_err = float(np.linalg.norm(cp_next[0] - cp_prev[-1]))
    tangent_prev = cp_prev[-1] - cp_prev[-2]
    tangent_next = cp_next[1] - cp_next[0]
    c1_err = float(np.linalg.norm(tangent_next - tangent_prev))
    return c0_err, c1_err


def compute_chunk_metrics(
    original_points: np.ndarray,
    reconstructed_points: np.ndarray,
    control_points: np.ndarray,
) -> Dict[str, float]:
    """재구성 정확도 및 제어점 유계성 메트릭 산출."""
    diff = original_points - reconstructed_points
    dist = np.hypot(diff[:, 0], diff[:, 1])
    mse = float(np.mean(np.sum(diff ** 2, axis=-1)))
    rmse = float(np.sqrt(mse))
    max_err = float(np.max(dist))

    min_coord = float(np.min(control_points))
    max_coord = float(np.max(control_points))
    overshoot = float(max(0.0, -min_coord - 0.05) + max(0.0, max_coord - 1.05))

    return {
        "rmse": rmse,
        "rmse_px_128": rmse * 128.0,
        "max_err_px_128": max_err * 128.0,
        "min_cp_coord": min_coord,
        "max_cp_coord": max_coord,
        "overshoot": overshoot,
    }
