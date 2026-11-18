"""experiments/test_bspline_compression.py

드로잉 궤적 시계열(x, y) 데이터를 고정된 소수의 B-Spline 제어점(Control Points)으로
최적 압축하고 복원하는 실험 스크립트 (v2 - 기하학적 결함 개선판).

[v2 주요 개선 사항]:
  1. 2차 차분 평활화 정규화 (2nd-order Difference Smoothness Regularization):
     - 목적식: min_P ||B P - X||^2 + λ2 ||D2 P||^2 + λ1 ||D1 P||^2
     - 인접 제어점의 급격한 꺾임/지그재그 진동을 억제하여 꽈배기처럼 꼬이는 자가교차 루프(Loop Artifact) 원천 제거.
  2. 제어점 바운딩 및 박스 제약 (Bounded Least Squares):
     - scipy.optimize.lsq_linear를 사용하여 제어점이 [-0.05, 1.05] 캔버스 유효 영역을 벗어나는 오버슈팅 방지.
  3. 스트로크 길이에 따른 적응형 M 할당 (Adaptive M Assignment):
     - 전체 도안 복원 시 긴 외곽선(N>=150)은 M=7~8, 중간 획은 M=5~6, 짧은 줄기/잎(N<40)은 M=3~4로 최적 분배.
  4. 자가교차(Self-intersection / Loop) 정량 검출기 및 개선 전후(v1 vs v2) 비교 리포트 내장.

사용법:
  # 기본 실행 (사과 대상 테스트 및 시각화)
  python experiments/test_bspline_compression.py --target apple

  # 고양이(cat) 뾰족한 귀 모서리 보존 테스트
  python experiments/test_bspline_compression.py --target cat

  # 벤치마크 모드 (사과 + 고양이 동시 실행 및 비교 저장)
  python experiments/test_bspline_compression.py --run-benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt

# 프로젝트 루트 경로 등록
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from train.dataset import pen_down_runs
except ImportError:
    def pen_down_runs(traj: np.ndarray) -> np.ndarray:
        """traj[:, 2] (pen_down)이 1인 구간들의 [start, end) 인덱스."""
        pen = (traj[:, 2] > 0.5).astype(np.int8)
        edges = np.diff(np.concatenate([[0], pen, [0]]))
        return np.stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)], axis=1)


# ---------------------------------------------------------------------------
# 1. 기하학적 분석 유틸리티 (자가교차 / 루프 감지)
# ---------------------------------------------------------------------------

def count_self_intersections(curve_points: np.ndarray) -> int:
    """복원된 2D 곡선에서 선분 간 자가교차(Loop Artifact) 횟수 정량 카운트.
    
    인접한 선분을 제외하고, 선분 i와 선분 j가 서로 교차하는 횟수를 센다.
    """
    pts = np.asarray(curve_points, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return 0

    count = 0
    # 선분 (p1-p2)와 (p3-p4) 교차 판정
    for i in range(n - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        for j in range(i + 2, n - 1):
            # 닫힌 곡선 시작/끝 연결부 인접 선분 제외
            if i == 0 and j == n - 2:
                continue
            p3 = pts[j]
            p4 = pts[j + 1]

            # 2D 외적을 이용한 선분 교차 검정
            c1 = (p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])
            c2 = (p4[0] - p3[0]) * (p2[1] - p3[1]) - (p4[1] - p3[1]) * (p2[0] - p3[0])
            c3 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
            c4 = (p2[0] - p1[0]) * (p4[1] - p1[1]) - (p2[1] - p1[1]) * (p4[0] - p1[0])

            if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
                count += 1
    return count


# ---------------------------------------------------------------------------
# 2. B-Spline 기저 및 정규화 최소제곱(Regularized Bounded Least Squares)
# ---------------------------------------------------------------------------

def parameterize_curve(points: np.ndarray, method: str = "chord") -> np.ndarray:
    """궤적 점열에 매개변수 t in [0, 1] 할당.
    
    Args:
        points: (N, 2) 좌표 배열
        method: 'chord' (누적 유클리드 호 길이), 'uniform' (균등), 'centripetal' (제곱근 거리)
    """
    n = len(points)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)

    if method == "uniform":
        return np.linspace(0.0, 1.0, n, dtype=np.float64)

    diffs = np.diff(points, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])

    if method == "centripetal":
        dists = np.sqrt(np.maximum(dists, 1e-12))

    total_dist = float(np.sum(dists))
    if total_dist < 1e-9:
        return np.linspace(0.0, 1.0, n, dtype=np.float64)

    s = np.concatenate([[0.0], np.cumsum(dists)])
    t = s / s[-1]
    t[-1] = 1.0
    return t


def build_clamped_knot_vector(num_control_points: int, degree: int) -> np.ndarray:
    """양 끝점에서 곡선이 시작/끝 제어점을 통과하도록 하는 Clamped 매듭 벡터 생성."""
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


def build_difference_matrices(num_control_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """1차 차분 행렬 D1 및 2차 차분 행렬 D2 구축.
    
    D1: (M-1, M) -> 인접 제어점 거리 벌어짐 페널티
    D2: (M-2, M) -> 인접 제어점 꺾임/가속도/곡률 페널티 (루프 억제 핵심)
    """
    m = num_control_points
    
    # 1차 차분: P_{i+1} - P_i
    if m >= 2:
        d1 = np.zeros((m - 1, m), dtype=np.float64)
        for i in range(m - 1):
            d1[i, i] = -1.0
            d1[i, i + 1] = 1.0
    else:
        d1 = np.zeros((0, m), dtype=np.float64)

    # 2차 차분: P_i - 2*P_{i+1} + P_{i+2}
    if m >= 3:
        d2 = np.zeros((m - 2, m), dtype=np.float64)
        for i in range(m - 2):
            d2[i, i] = 1.0
            d2[i, i + 1] = -2.0
            d2[i, i + 2] = 1.0
    else:
        d2 = np.zeros((0, m), dtype=np.float64)

    return d1, d2


def fit_bspline_least_squares_v2(
    points: np.ndarray,
    num_control_points: int,
    degree: int = 3,
    param_method: str = "chord",
    smoothness_weight: float = 0.05,
    bounds: Tuple[float, float] = (-0.05, 1.05),
    clamp_endpoints: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """[v2] 2차 차분 평활화 정규화 및 박스 제약이 결합된 최적 제어점 역산.
    
    최적화 문제:
        min_{P} || B * P - X ||_F^2 + λ2 || D2 * P ||_F^2 + λ1 || D1 * P ||_F^2
        s.t.   bounds[0] <= P <= bounds[1]
               (선택) P[0] = points[0], P[-1] = points[-1]
    
    Args:
        points: (N, 2) 원본 점열
        num_control_points (M): 제어점 개수
        degree (k): B-Spline 차수 (기본 3 = Cubic)
        param_method: 'chord' 또는 'uniform'
        smoothness_weight (λ2): 2차 차분 평활화 가중치 (루프 억제)
        bounds: 제어점 허용 좌표 범위 (기본: [-0.05, 1.05])
        clamp_endpoints: 시작점과 끝점을 원본과 일치하도록 강제할지 여부
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        raise ValueError("입력 점열이 비어 있습니다.")

    actual_degree = min(degree, num_control_points - 1, max(1, n - 1))
    actual_m = min(num_control_points, n)

    if actual_m <= 1:
        cp = np.repeat(points[:1], num_control_points, axis=0)
        knots = build_clamped_knot_vector(num_control_points, actual_degree)
        return cp, knots, actual_degree, np.zeros(n, dtype=np.float64)

    t_orig = parameterize_curve(points, method=param_method)
    knots = build_clamped_knot_vector(actual_m, actual_degree)

    # (N, M) B-Spline 기저 행렬
    b_mat = interp.BSpline.design_matrix(t_orig, knots, actual_degree).toarray()

    # 정규화 행렬 구성 (스트로크 길이 N에 적응형 스케일링)
    d1, d2 = build_difference_matrices(actual_m)
    scale_n = n / 100.0
    lam2 = float(smoothness_weight * scale_n)
    lam1 = float(0.1 * lam2)

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

    # 박스 경계 (Bounds) 설정
    lb_x = np.full(actual_m, bounds[0], dtype=np.float64)
    ub_x = np.full(actual_m, bounds[1], dtype=np.float64)
    lb_y = np.full(actual_m, bounds[0], dtype=np.float64)
    ub_y = np.full(actual_m, bounds[1], dtype=np.float64)

    if clamp_endpoints and actual_m > 2:
        # 시작점과 끝점 좌표를 정확히 고정
        lb_x[0] = ub_x[0] = float(points[0, 0])
        lb_y[0] = ub_y[0] = float(points[0, 1])
        lb_x[-1] = ub_x[-1] = float(points[-1, 0])
        lb_y[-1] = ub_y[-1] = float(points[-1, 1])

    # scipy.optimize.lsq_linear로 박스 제약 선형 최소제곱 고속 풀이
    res_x = opt.lsq_linear(a_mat, bx, bounds=(lb_x, ub_x), tol=1e-6)
    res_y = opt.lsq_linear(a_mat, by, bounds=(lb_y, ub_y), tol=1e-6)

    control_points = np.column_stack([res_x.x, res_y.x])
    control_points = np.clip(control_points, bounds[0], bounds[1])

    if len(control_points) < num_control_points:
        pad_len = num_control_points - len(control_points)
        control_points = np.pad(control_points, ((0, pad_len), (0, 0)), mode="edge")
        knots = build_clamped_knot_vector(num_control_points, actual_degree)

    return control_points, knots, actual_degree, t_orig


def fit_bspline_v1_legacy(
    points: np.ndarray,
    num_control_points: int,
    degree: int = 3,
    param_method: str = "chord",
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """[비교용 v1] 정규화 및 바운딩이 없는 기존 순수 최소제곱법."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    actual_degree = min(degree, num_control_points - 1, max(1, n - 1))
    actual_m = min(num_control_points, n)
    t_orig = parameterize_curve(points, method=param_method)
    knots = build_clamped_knot_vector(actual_m, actual_degree)
    b_mat = interp.BSpline.design_matrix(t_orig, knots, actual_degree).toarray()
    cp, _, _, _ = np.linalg.lstsq(b_mat, points, rcond=None)
    return cp, knots, actual_degree, t_orig


def reconstruct_bspline(
    control_points: np.ndarray,
    knots: np.ndarray,
    degree: int,
    num_points: Optional[int] = None,
    t_eval: Optional[np.ndarray] = None,
) -> np.ndarray:
    """제어점과 매듭 벡터로부터 복원 곡선 계산."""
    spl = interp.BSpline(knots, control_points, degree)
    if t_eval is None:
        if num_points is None:
            num_points = 200
        t_eval = np.linspace(0.0, 1.0, num_points)
    t_eval = np.clip(t_eval, 0.0, 1.0)
    return spl(t_eval)


# ---------------------------------------------------------------------------
# 3. 스트로크 길이에 따른 적응형 M 할당 (Adaptive M Assignment)
# ---------------------------------------------------------------------------

def get_adaptive_m(stroke: np.ndarray, min_m: int = 4, max_m: int = 8) -> int:
    """스트로크 점 개수(N)와 기하학적 호 길이(L)에 비례하여 적응형 M 결정.
    
    규칙:
      - 짧은 디테일 / 잎 / 줄기 / 수염 (N < 40 or L < 0.15) -> M = 3 ~ 4
      - 중간 획 / 눈 / 코 (40 <= N < 150 or 0.15 <= L < 0.45) -> M = 5 ~ 6
      - 긴 외곽선 / 복잡한 몸체 (N >= 150 or L >= 0.45)      -> M = 7 ~ max_m
    """
    n = len(stroke)
    if n <= 3:
        return 3

    diffs = np.diff(stroke, axis=0)
    total_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

    if n < 40 or total_len < 0.15:
        return min_m
    elif n < 150 or total_len < 0.45:
        return min(max_m - 2, max(min_m, 6))
    else:
        return max_m


def compute_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    control_points: np.ndarray,
) -> Dict[str, float]:
    """복원 오차, 압축률, 자가교차 루프 수 및 제어점 좌표 범위 계산."""
    diff = original - reconstructed
    dist = np.hypot(diff[:, 0], diff[:, 1])
    mse = float(np.mean(np.sum(diff ** 2, axis=-1)))
    rmse = float(np.sqrt(mse))
    max_err = float(np.max(dist))
    mean_err = float(np.mean(dist))

    n = len(original)
    m = len(control_points)
    loops = count_self_intersections(reconstructed)

    min_cp = float(np.min(control_points))
    max_cp = float(np.max(control_points))
    cp_overshoot = float(max(0.0, -min_cp - 0.05) + max(0.0, max_cp - 1.05))

    return {
        "n_points": n,
        "m_control_points": m,
        "compression_ratio": float(n / m) if m > 0 else 0.0,
        "data_reduction_pct": float((1.0 - m / n) * 100.0) if n > 0 else 0.0,
        "mse": mse,
        "rmse": rmse,
        "mean_error": mean_err,
        "max_error": max_err,
        "rmse_px_128": rmse * 128.0,
        "max_err_px_128": max_err * 128.0,
        "rmse_px_512": rmse * 512.0,
        "max_err_px_512": max_err * 512.0,
        "loops_count": loops,
        "min_cp_coord": min_cp,
        "max_cp_coord": max_cp,
        "cp_overshoot": cp_overshoot,
    }


# ---------------------------------------------------------------------------
# 4. 데이터셋 로더
# ---------------------------------------------------------------------------

def load_demo_data(
    sample_path: Optional[str] = None,
    target: Optional[str] = None,
    data_root: str = "data/raw",
) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
    """JSON 데모 파일 로드."""
    root = PROJECT_ROOT / data_root
    json_path: Optional[Path] = None

    if sample_path:
        json_path = Path(sample_path)
        if not json_path.is_absolute():
            json_path = PROJECT_ROOT / json_path
    elif target:
        matches = sorted(root.glob(f"demo_{target}_*.json"))
        if matches:
            json_path = matches[0]
        else:
            raise FileNotFoundError(f"타겟 '{target}'에 매칭되는 데모 파일이 없습니다: {root}")
    else:
        matches = sorted(root.glob("demo_apple_*.json")) or sorted(root.glob("demo_*.json"))
        if matches:
            json_path = matches[0]
        else:
            raise FileNotFoundError(f"{root}에 데모 파일이 없습니다.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    traj = np.array(data["trajectory"], dtype=np.float64)
    runs = pen_down_runs(traj)
    strokes = [traj[a:b, :2] for a, b in runs if b > a]

    meta = data.get("meta", {})
    meta["file_path"] = str(json_path.resolve())
    meta["file_name"] = json_path.name
    return traj, strokes, meta


# ---------------------------------------------------------------------------
# 5. v2 시각화 렌더러 (다각도 정밀 분석 뷰)
# ---------------------------------------------------------------------------

def visualize_bspline_v2(
    primary_stroke: np.ndarray,
    results_v2: Dict[int, Dict],
    results_v1: Dict[int, Dict],
    all_strokes: List[np.ndarray],
    all_strokes_recon_fixed: List[np.ndarray],
    all_strokes_recon_adapt: List[np.ndarray],
    adaptive_m_list: List[int],
    meta: Dict,
    output_path: Path,
    smoothness: float,
):
    """v2 시각화 생성 및 저장 (루프 제거, 오버슈팅 방지, 적응형 M 비교)."""
    plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(19, 12), dpi=200)
    fig.patch.set_facecolor("#F8F9FA")

    target_name = meta.get("target", "Drawing")
    target_label = meta.get("target_label", target_name)
    title_text = (
        f"Drawing B-Spline v2 (Regularized & Bounded Least Squares)\n"
        f"Target: '{target_name}' ({target_label}) | Sample: {meta.get('file_name', '')} "
        f"| Smoothness λ={smoothness} | Bounds=[-0.05, 1.05]"
    )
    fig.suptitle(title_text, fontsize=15, fontweight="bold", y=0.98, color="#1A202C")

    color_map = {4: "#E53E3E", 6: "#3182CE", 8: "#38A169"}
    marker_map = {4: "D", 6: "s", 8: "o"}
    sorted_m = sorted(results_v2.keys())

    gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.28, left=0.06, right=0.96, top=0.91, bottom=0.07)

    # -----------------------------------------------------------------------
    # 패널 1: 원본 Stroke (조밀 점열)
    # -----------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor("#FFFFFF")
    ax0.scatter(primary_stroke[:, 0], primary_stroke[:, 1], c="#718096", s=6, alpha=0.5, label="Original Points")
    ax0.plot(primary_stroke[:, 0], primary_stroke[:, 1], color="#2D3748", lw=1.2, alpha=0.8)
    ax0.scatter(primary_stroke[0, 0], primary_stroke[0, 1], c="#38A169", s=80, marker="^", zorder=5, label="Start")
    ax0.scatter(primary_stroke[-1, 0], primary_stroke[-1, 1], c="#E53E3E", s=80, marker="v", zorder=5, label="End")

    ax0.set_title(f"Original Stroke ({len(primary_stroke)} pts)", fontsize=11, fontweight="bold", pad=8)
    ax0.set_xlim(-0.08, 1.08)
    ax0.set_ylim(-0.08, 1.08)
    ax0.invert_yaxis()
    ax0.set_aspect("equal")
    ax0.grid(True, linestyle="--", alpha=0.4)
    ax0.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax0.set_xlabel("X (norm)", fontsize=8)
    ax0.set_ylabel("Y (norm, down)", fontsize=8)

    # -----------------------------------------------------------------------
    # 패널 2, 3, 4: M = 4, 6, 8 v2 복원 및 v1 비교
    # -----------------------------------------------------------------------
    for idx, m_val in enumerate(sorted_m):
        ax_m = fig.add_subplot(gs[0, idx + 1])
        ax_m.set_facecolor("#FFFFFF")
        res2 = results_v2[m_val]
        res1 = results_v1.get(m_val)
        cp2 = res2["control_points"]
        recon2 = res2["reconstructed_dense"]
        met2 = res2["metrics"]
        col = color_map.get(m_val, "#4A5568")
        mk = marker_map.get(m_val, "o")

        # 원본 점열
        ax_m.plot(primary_stroke[:, 0], primary_stroke[:, 1], color="#E2E8F0", lw=1.2, linestyle=":", label="Original")

        # 이전 v1 곡선 (루프 또는 오버슈팅 비교 표시, 얇은 빨간 점선)
        if res1 is not None:
            recon1 = res1["reconstructed_dense"]
            ax_m.plot(recon1[:, 0], recon1[:, 1], color="#E53E3E", lw=1.0, linestyle="--", alpha=0.45, label="v1 (Legacy)")

        # v2 제어 다각형 및 제어점
        ax_m.plot(cp2[:, 0], cp2[:, 1], color=col, lw=1.2, linestyle="-.", alpha=0.6, label="Control Polygon")
        ax_m.scatter(cp2[:, 0], cp2[:, 1], color=col, s=70, marker=mk, edgecolor="#1A202C", lw=1.2, zorder=6, label=f"CPs (M={m_val})")

        # 제어점 번호 표기
        for p_idx, pt in enumerate(cp2):
            ax_m.annotate(
                f"P{p_idx}",
                (pt[0], pt[1]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                fontweight="bold",
                color=col,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFFFFF", edgecolor=col, alpha=0.7)
            )

        # v2 복원 곡선 (두꺼운 부드러운 실선)
        ax_m.plot(recon2[:, 0], recon2[:, 1], color=col, lw=2.4, alpha=0.9, zorder=4, label="v2 Recon (Smooth)")

        ax_m.set_title(f"M = {m_val} (v2 Regularized)", fontsize=11, fontweight="bold", color=col, pad=8)
        ax_m.set_xlim(-0.08, 1.08)
        ax_m.set_ylim(-0.08, 1.08)
        ax_m.invert_yaxis()
        ax_m.set_aspect("equal")
        ax_m.grid(True, linestyle="--", alpha=0.4)
        ax_m.set_xlabel("X (norm)", fontsize=8)

        # 캔버스 경계 박스 [0, 1] 가이드 점선
        rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#A0AEC0", linestyle="--", lw=0.8, alpha=0.7)
        ax_m.add_patch(rect)

        stats_box = (
            f"RMSE: {met2['rmse']:.4f} ({met2['rmse_px_128']:.1f}px@128)\n"
            f"MaxErr: {met2['max_error']:.4f} ({met2['max_err_px_128']:.1f}px)\n"
            f"Loops: {met2['loops_count']} (v1: {res1['metrics']['loops_count'] if res1 else 'N/A'})\n"
            f"CP Range: [{met2['min_cp_coord']:.2f}, {met2['max_cp_coord']:.2f}]"
        )
        ax_m.text(
            0.04, 0.04, stats_box,
            transform=ax_m.transAxes,
            fontsize=7.2,
            family="monospace",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EDF2F7", edgecolor="#CBD5E0", alpha=0.9)
        )
        ax_m.legend(loc="upper right", fontsize=7.2, framealpha=0.85)

    # -----------------------------------------------------------------------
    # 패널 5: All-in-One 오버레이 비교
    # -----------------------------------------------------------------------
    ax_all = fig.add_subplot(gs[1, 0])
    ax_all.set_facecolor("#FFFFFF")
    ax_all.plot(primary_stroke[:, 0], primary_stroke[:, 1], color="#1A202C", lw=2.2, alpha=0.3, label="Original Ground Truth")

    for m_val in sorted_m:
        res2 = results_v2[m_val]
        recon2 = res2["reconstructed_dense"]
        col = color_map.get(m_val, "#4A5568")
        ax_all.plot(recon2[:, 0], recon2[:, 1], color=col, lw=1.8, label=f"M={m_val} ({res2['metrics']['rmse_px_128']:.1f}px)")

    ax_all.set_title("Reconstruction Overlay", fontsize=11, fontweight="bold", pad=8)
    ax_all.set_xlim(-0.08, 1.08)
    ax_all.set_ylim(-0.08, 1.08)
    ax_all.invert_yaxis()
    ax_all.set_aspect("equal")
    ax_all.grid(True, linestyle="--", alpha=0.4)
    ax_all.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
    ax_all.set_xlabel("X (norm)", fontsize=8)
    ax_all.set_ylabel("Y (norm, down)", fontsize=8)

    # -----------------------------------------------------------------------
    # 패널 6: v1 vs v2 결함 제거 비교 (오버슈팅 & 루프 비교 차트)
    # -----------------------------------------------------------------------
    ax_cmp = fig.add_subplot(gs[1, 1])
    ax_cmp.set_facecolor("#FFFFFF")

    m_list = sorted_m
    x_idx = np.arange(len(m_list))
    bw = 0.35

    # v1 vs v2 Max Error 비교
    v1_max = [results_v1[m]["metrics"]["max_err_px_128"] for m in m_list]
    v2_max = [results_v2[m]["metrics"]["max_err_px_128"] for m in m_list]

    b1 = ax_cmp.bar(x_idx - bw/2, v1_max, bw, label="v1 (Legacy)", color="#CBD5E0", alpha=0.85)
    b2 = ax_cmp.bar(x_idx + bw/2, v2_max, bw, label="v2 (Regularized)", color="#3182CE", alpha=0.85)

    ax_cmp.set_xticks(x_idx)
    ax_cmp.set_xticklabels([f"M={m}" for m in m_list], fontsize=8.5)
    ax_cmp.set_ylabel("Max Error (pixels @ 128)", fontsize=8.5)
    ax_cmp.set_title("v1 vs v2 Max Error Comparison", fontsize=11, fontweight="bold", pad=8)
    ax_cmp.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_cmp.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    for rect in b1:
        h = rect.get_height()
        ax_cmp.text(rect.get_x() + rect.get_width()/2.0, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=7)
    for rect in b2:
        h = rect.get_height()
        ax_cmp.text(rect.get_x() + rect.get_width()/2.0, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    # -----------------------------------------------------------------------
    # 패널 7: 전체 도안 - 고정 M (Fixed M=6) 복원
    # -----------------------------------------------------------------------
    ax_fix = fig.add_subplot(gs[1, 2])
    ax_fix.set_facecolor("#FFFFFF")
    for s_recon in all_strokes_recon_fixed:
        ax_fix.plot(s_recon[:, 0], s_recon[:, 1], lw=2.0, color="#4A5568", solid_capstyle="round")

    total_orig = sum(len(s) for s in all_strokes)
    total_fix_cp = len(all_strokes) * 6
    ax_fix.set_title(
        f"Full Drawing: Fixed M=6\nTotal CPs: {total_fix_cp} ({total_orig/total_fix_cp:.1f}x comp)",
        fontsize=10.5, fontweight="bold", pad=8
    )
    ax_fix.set_xlim(-0.05, 1.05)
    ax_fix.set_ylim(-0.05, 1.05)
    ax_fix.invert_yaxis()
    ax_fix.set_aspect("equal")
    ax_fix.grid(True, linestyle="--", alpha=0.4)
    ax_fix.set_xlabel("X (norm)", fontsize=8)

    # -----------------------------------------------------------------------
    # 패널 8: 전체 도안 - 적응형 M (Adaptive M) 복원
    # -----------------------------------------------------------------------
    ax_adapt = fig.add_subplot(gs[1, 3])
    ax_adapt.set_facecolor("#FFFFFF")
    for s_recon in all_strokes_recon_adapt:
        ax_adapt.plot(s_recon[:, 0], s_recon[:, 1], lw=2.0, color="#3182CE", solid_capstyle="round")

    total_adapt_cp = sum(adaptive_m_list)
    ax_adapt.set_title(
        f"Full Drawing: Adaptive M per Stroke\n"
        f"Total CPs: {total_adapt_cp} ({total_orig/total_adapt_cp:.1f}x comp) [M in {min(adaptive_m_list)}~{max(adaptive_m_list)}]",
        fontsize=10.5, fontweight="bold", color="#2B6CB0", pad=8
    )
    ax_adapt.set_xlim(-0.05, 1.05)
    ax_adapt.set_ylim(-0.05, 1.05)
    ax_adapt.invert_yaxis()
    ax_adapt.set_aspect("equal")
    ax_adapt.grid(True, linestyle="--", alpha=0.4)
    ax_adapt.set_xlabel("X (norm)", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Visualizer] v2 비교 시각화 이미지가 저장되었습니다 -> {output_path.resolve()}")


# ---------------------------------------------------------------------------
# 6. 실험 실행 파이프라인
# ---------------------------------------------------------------------------

def run_experiment_v2(
    sample_path: Optional[str] = None,
    target: Optional[str] = "apple",
    m_values: List[int] = [4, 6, 8],
    degree: int = 3,
    param_method: str = "chord",
    smoothness: float = 0.05,
    clamp_endpoints: bool = False,
    output_image: Optional[str] = None,
    save_json_path: Optional[str] = None,
) -> Dict:
    print("\n" + "=" * 76)
    print(f" [Drawing-VLA] B-Spline v2 Compression Test: Target '{target}' ")
    print("=" * 76)

    # 1. 데이터 로드
    traj, strokes, meta = load_demo_data(sample_path=sample_path, target=target)
    print(f"[*] 데이터 로드 완료:")
    print(f"    - 파일명: {meta.get('file_name', 'N/A')}")
    print(f"    - 대상 타겟: {meta.get('target', 'N/A')} ({meta.get('target_label', 'N/A')})")
    print(f"    - 총 스텝: {len(traj)} | 스트로크: {len(strokes)}개")

    if not strokes:
        print("[!] 스트로크 데이터가 없습니다.")
        return {}

    # 메인 스트로크 선정 (가장 긴 외곽선)
    stroke_lens = [len(s) for s in strokes]
    main_idx = int(np.argmax(stroke_lens))
    primary_stroke = strokes[main_idx]
    print(f"    - 메인 스트로크: #{main_idx} (포인트 {len(primary_stroke)}개)")
    print(f"    - 2차 차분 평활화 정규화 λ: {smoothness}")
    print(f"    - 제어점 바운딩: [-0.05, 1.05]")
    print(f"    - 끝점 일치 고정 (Clamp Endpoints): {clamp_endpoints}")
    print("-" * 76)

    # 2. v1 vs v2 제어점 추출 및 복원
    results_v2 = {}
    results_v1 = {}

    print(f"{'M':>2} | {'Pts':>4} | {'v1 Loops':>8} | {'v2 Loops':>8} | {'v1 CP Range':>15} | {'v2 CP Range':>15} | {'v2 RMSE(128)':>12}")
    print("-" * 76)

    for m in m_values:
        # (A) 개선판 v2 (Regularized + Bounded)
        cp2, knots2, deg2, t2 = fit_bspline_least_squares_v2(
            primary_stroke,
            num_control_points=m,
            degree=degree,
            param_method=param_method,
            smoothness_weight=smoothness,
            bounds=(-0.05, 1.05),
            clamp_endpoints=clamp_endpoints,
        )
        recon_orig_t2 = reconstruct_bspline(cp2, knots2, deg2, t_eval=t2)
        recon_dense2 = reconstruct_bspline(cp2, knots2, deg2, num_points=200)
        met2 = compute_metrics(primary_stroke, recon_orig_t2, cp2)

        results_v2[m] = {
            "control_points": cp2,
            "knots": knots2,
            "degree": deg2,
            "reconstructed_dense": recon_dense2,
            "metrics": met2,
        }

        # (B) 레거시 v1 (Unregularized, Unconstrained)
        cp1, knots1, deg1, t1 = fit_bspline_v1_legacy(
            primary_stroke,
            num_control_points=m,
            degree=degree,
            param_method=param_method,
        )
        recon_orig_t1 = reconstruct_bspline(cp1, knots1, deg1, t_eval=t1)
        recon_dense1 = reconstruct_bspline(cp1, knots1, deg1, num_points=200)
        met1 = compute_metrics(primary_stroke, recon_orig_t1, cp1)

        results_v1[m] = {
            "control_points": cp1,
            "knots": knots1,
            "degree": deg1,
            "reconstructed_dense": recon_dense1,
            "metrics": met1,
        }

        v1_rng_str = f"[{met1['min_cp_coord']:.2f}, {met1['max_cp_coord']:.2f}]"
        v2_rng_str = f"[{met2['min_cp_coord']:.2f}, {met2['max_cp_coord']:.2f}]"
        print(
            f"{m:2d} | {len(primary_stroke):4d} | "
            f"{met1['loops_count']:8d} | "
            f"{met2['loops_count']:8d} | "
            f"{v1_rng_str:>15} | "
            f"{v2_rng_str:>15} | "
            f"{met2['rmse_px_128']:10.2f}px"
        )

    print("-" * 76)

    # 3. 전체 도안 복원: 고정 M(6) vs 적응형 M(Adaptive)
    recon_fixed_list = []
    recon_adapt_list = []
    adapt_m_list = []

    for s_idx, s in enumerate(strokes):
        # 고정 M=6
        cp_fix, k_fix, d_fix, _ = fit_bspline_least_squares_v2(
            s, num_control_points=6, degree=degree, smoothness_weight=smoothness, bounds=(-0.05, 1.05), clamp_endpoints=clamp_endpoints
        )
        recon_fixed_list.append(reconstruct_bspline(cp_fix, k_fix, d_fix, num_points=max(30, len(s))))

        # 적응형 M
        m_adapt = get_adaptive_m(s, min_m=4, max_m=8)
        adapt_m_list.append(m_adapt)
        cp_ad, k_ad, d_ad, _ = fit_bspline_least_squares_v2(
            s, num_control_points=m_adapt, degree=degree, smoothness_weight=smoothness, bounds=(-0.05, 1.05), clamp_endpoints=clamp_endpoints
        )
        recon_adapt_list.append(reconstruct_bspline(cp_ad, k_ad, d_ad, num_points=max(30, len(s))))

    print(f"[*] 적응형 M 할당 결과: 총 {len(strokes)}개 스트로크 -> {adapt_m_list}")

    # 4. 시각화 저장
    if output_image is None:
        output_image = f"experiments/bspline_v2_{target}_comparison.png"
    out_img_path = Path(output_image)
    if not out_img_path.is_absolute():
        out_img_path = PROJECT_ROOT / out_img_path

    visualize_bspline_v2(
        primary_stroke=primary_stroke,
        results_v2=results_v2,
        results_v1=results_v1,
        all_strokes=strokes,
        all_strokes_recon_fixed=recon_fixed_list,
        all_strokes_recon_adapt=recon_adapt_list,
        adaptive_m_list=adapt_m_list,
        meta=meta,
        output_path=out_img_path,
        smoothness=smoothness,
    )

    # 5. JSON 저장
    if save_json_path:
        out_json_path = Path(save_json_path)
        if not out_json_path.is_absolute():
            out_json_path = PROJECT_ROOT / out_json_path
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable = {
            "meta": meta,
            "target": target,
            "smoothness": smoothness,
            "results_v2": {
                m: {
                    "metrics": results_v2[m]["metrics"],
                    "control_points": results_v2[m]["control_points"].tolist(),
                }
                for m in m_values
            },
            "results_v1": {
                m: {
                    "metrics": results_v1[m]["metrics"],
                    "control_points": results_v1[m]["control_points"].tolist(),
                }
                for m in m_values
            },
            "adaptive_m_list": adapt_m_list,
        }
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"[*] 실험 결과 JSON 저장 -> {out_json_path.resolve()}")

    return {"results_v2": results_v2, "results_v1": results_v1}


def main():
    parser = argparse.ArgumentParser(description="Drawing Trajectory B-Spline Compression v2 (Regularized & Bounded)")
    parser.add_argument("--sample-path", type=str, default=None, help="테스트할 demo_*.json 파일 경로")
    parser.add_argument("--target", type=str, default="apple", help="대상 타겟 카테고리 (apple, cat, fish, star 등)")
    parser.add_argument("--m-values", type=int, nargs="+", default=[4, 6, 8], help="비교할 제어점 개수 리스트 (기본: 4 6 8)")
    parser.add_argument("--degree", type=int, default=3, help="B-Spline 차수 (기본 3)")
    parser.add_argument("--param-method", type=str, default="chord", choices=["chord", "uniform", "centripetal"])
    parser.add_argument("--smoothness", type=float, default=0.05, help="2차 차분 평활화 페널티 가중치 λ (기본: 0.05)")
    parser.add_argument("--clamp-endpoints", action="store_true", help="곡선 시작/끝점이 원본 끝점을 반드시 통과하도록 고정")
    parser.add_argument("--output", type=str, default=None, help="결과 이미지 저장 경로")
    parser.add_argument("--save-json", type=str, default="experiments/bspline_v2_results.json", help="수치 결과 JSON 저장 경로")
    parser.add_argument("--run-benchmark", action="store_true", help="사과(apple) 및 고양이(cat) 벤치마크 일괄 실행")

    args = parser.parse_args()

    if args.run_benchmark:
        print("\n========================================================================")
        print("          B-Spline v2 일괄 벤치마크 (사과 & 고양이) 시작               ")
        print("========================================================================")
        # 사과 테스트
        run_experiment_v2(
            target="apple",
            m_values=args.m_values,
            degree=args.degree,
            param_method=args.param_method,
            smoothness=args.smoothness,
            clamp_endpoints=args.clamp_endpoints,
            output_image="experiments/bspline_v2_apple_comparison.png",
            save_json_path="experiments/bspline_v2_apple_results.json",
        )
        # 고양이 테스트
        run_experiment_v2(
            target="cat",
            m_values=args.m_values,
            degree=args.degree,
            param_method=args.param_method,
            smoothness=args.smoothness,
            clamp_endpoints=args.clamp_endpoints,
            output_image="experiments/bspline_v2_cat_comparison.png",
            save_json_path="experiments/bspline_v2_cat_results.json",
        )
    else:
        run_experiment_v2(
            sample_path=args.sample_path,
            target=args.target,
            m_values=args.m_values,
            degree=args.degree,
            param_method=args.param_method,
            smoothness=args.smoothness,
            clamp_endpoints=args.clamp_endpoints,
            output_image=args.output,
            save_json_path=args.save_json,
        )


if __name__ == "__main__":
    main()
