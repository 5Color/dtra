"""tests/test_bspline_core.py

B-Spline 핵심 모듈 검증 단위 테스트:
1. 매듭 벡터 생성 및 차수 무결성
2. BVLS 피팅 및 바운딩 제약 확인 ([-0.05, 1.05])
3. C^1 연속성 바인딩 (C0 위치 일치 및 C1 접선 벡터 일치)
4. PyTorch BSplineDecoder 와 SciPy BSpline 일치성 검증 (오차 < 1e-6)
"""

import unittest
import numpy as np
import scipy.interpolate as interp
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sp in (str(_REPO_ROOT), str(_REPO_ROOT / "v1_drawing_vla" / "src"), str(_REPO_ROOT / "v1_drawing_vla" / "handvsai")):
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
import torch

from dtra.core.bspline_core import (
    build_clamped_knot_vector,
    parameterize_points,
    build_design_matrix,
    fit_bspline_chunk_bvls,
    reconstruct_bspline_np,
    BSplineDecoder,
    compute_c1_tangent_discrepancy,
    compute_chunk_metrics,
)


class TestBSplineCore(unittest.TestCase):

    def test_knot_vector_shape_and_clamping(self):
        m, k = 6, 3
        knots = build_clamped_knot_vector(m, k)
        # 총 매듭점 수 = M + k + 1 = 6 + 3 + 1 = 10
        self.assertEqual(len(knots), 10)
        # 시작과 끝 각각 k+1 = 4개가 0과 1로 고정
        np.testing.assert_array_equal(knots[:4], [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(knots[-4:], [1.0, 1.0, 1.0, 1.0])
        # 단조 증가
        self.assertTrue(np.all(np.diff(knots) >= 0.0))

    def test_bvls_bounds_enforcement(self):
        # 캔버스 밖으로 튀어나갈 수 있는 급격한 인위적 점열
        pts = np.array([
            [0.1, 0.1], [0.2, 0.9], [0.5, 1.2], [0.8, 0.9], [0.9, 0.1]
        ], dtype=np.float64)

        cp, knots, deg = fit_bspline_chunk_bvls(
            pts, num_control_points=6, degree=3, bounds=(-0.05, 1.05)
        )
        self.assertEqual(cp.shape, (6, 2))
        self.assertGreaterEqual(float(np.min(cp)), -0.05 - 1e-7)
        self.assertLessEqual(float(np.max(cp)), 1.05 + 1e-7)

    def test_c1_continuity_binding(self):
        # 1번 청크 (임의의 곡선)
        t1 = np.linspace(0, np.pi / 2, 16)
        chunk1 = np.column_stack([np.cos(t1) * 0.4 + 0.5, np.sin(t1) * 0.4 + 0.5])
        cp1, knots, _ = fit_bspline_chunk_bvls(chunk1, num_control_points=6, degree=3)

        # 2번 청크 (1번 청크에서 이어지는 곡선)
        t2 = np.linspace(np.pi / 2, np.pi, 16)
        chunk2 = np.column_stack([np.cos(t2) * 0.4 + 0.5, np.sin(t2) * 0.4 + 0.5])

        # 이전 청크의 끝점과 접선 벡터로 C1 앵커 설정
        p0_anchor = cp1[-1]
        tangent_anchor = cp1[-1] - cp1[-2]
        p1_anchor = p0_anchor + tangent_anchor

        cp2, _, _ = fit_bspline_chunk_bvls(
            chunk2, num_control_points=6, degree=3, c1_anchor=(p0_anchor, p1_anchor)
        )

        c0_err, c1_err = compute_c1_tangent_discrepancy(cp1, cp2)
        # C0 오차는 0이어야 함
        self.assertAlmostEqual(c0_err, 0.0, places=6)
        # C1 접선 방향 불일치도 0이어야 함
        self.assertAlmostEqual(c1_err, 0.0, places=6)

    def test_pytorch_decoder_numerical_identity(self):
        decoder = BSplineDecoder(num_control_points=6, horizon=16, degree=3)
        torch.manual_seed(42)
        cp_torch = torch.rand(4, 6, 2)
        traj_torch = decoder(cp_torch)

        knots = build_clamped_knot_vector(6, 3)
        t_eval = np.linspace(0.0, 1.0, 16)

        for b in range(4):
            spl = interp.BSpline(knots, cp_torch[b].numpy(), 3)
            expected = spl(t_eval)
            actual = traj_torch[b].numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_reconstruction_quality_on_line(self):
        # 직선 궤적에 대해 M=4 이상의 B-Spline은 오차가 사실상 0이어야 함
        line_pts = np.linspace([0.1, 0.2], [0.8, 0.9], 16)
        cp, knots, deg = fit_bspline_chunk_bvls(line_pts, num_control_points=4, degree=3)
        recon = reconstruct_bspline_np(cp, knots, deg, num_points=16)
        metrics = compute_chunk_metrics(line_pts, recon, cp)
        self.assertLess(metrics["rmse_px_128"], 0.05)


if __name__ == "__main__":
    unittest.main()
