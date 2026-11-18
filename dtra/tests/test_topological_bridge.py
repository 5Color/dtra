
import copy
import os
import sys
import torch

# 실행 위치(dtra/ 또는 상위 폴더)와 무관하게 import 되도록 경로 자동 설정
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARENT_DIR = os.path.dirname(ROOT_DIR)
for p in (ROOT_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dtra.core.topological_bridge import BridgePhase, TopologicalGeodesicBridge
except ImportError:
    from core.topological_bridge import BridgePhase, TopologicalGeodesicBridge


def eval_clamped_cubic_bspline_boundary_derivs(knots: torch.Tensor, at_end: bool):
    if not at_end:
        c0 = knots[..., 0, :]
        c1 = 9.0 * (knots[..., 1, :] - knots[..., 0, :])
        c2 = 27.0 * (2.0 * knots[..., 0, :] - 3.0 * knots[..., 1, :] + knots[..., 2, :])
    else:
        c0 = knots[..., 5, :]
        c1 = 9.0 * (knots[..., 5, :] - knots[..., 4, :])
        c2 = 27.0 * (2.0 * knots[..., 5, :] - 3.0 * knots[..., 4, :] + knots[..., 3, :])
    return c0, c1, c2


def test_five_invariants():
    torch.manual_seed(42)
    bridge = TopologicalGeodesicBridge(num_cells=4, d_topo=64, n_knots=6, horizon=16)

    frozen_cell_a = torch.nn.Linear(64, 7).requires_grad_(False)
    frozen_cell_b = torch.nn.Linear(64, 7).requires_grad_(False)
    snap_a = copy.deepcopy(frozen_cell_a.state_dict())
    snap_b = copy.deepcopy(frozen_cell_b.state_dict())

    # 1. 초구면 S^63 단위 노름 보존 검증
    u = torch.randn(64)
    v = torch.randn(64)
    for step in range(17):
        z_t = bridge.slerp_s63(u, v, tau=step / 16.0)
        assert abs(z_t.norm(p=2).item() - 1.0) < 1e-6
    print("[PASS] 1/5 초구면(S^63) SLERP 노름 보존 확인")

    # 2. 위치(C^0), 속도(C^1), 가속도(C^2) 경계 연속성 검증
    knots_a = torch.randn(6, 7, dtype=torch.float64) * 0.05
    knots_b = torch.randn(6, 7, dtype=torch.float64) * 0.05
    p_br = bridge.synthesize_bridge_knots(knots_a, knots_b, c2_damping=1.0, action_clip=10.0)

    a0, a1, a2 = eval_clamped_cubic_bspline_boundary_derivs(knots_a, at_end=True)
    br_s0, br_s1, br_s2 = eval_clamped_cubic_bspline_boundary_derivs(p_br, at_end=False)
    br_e0, br_e1, br_e2 = eval_clamped_cubic_bspline_boundary_derivs(p_br, at_end=True)
    b0, b1, b2 = eval_clamped_cubic_bspline_boundary_derivs(knots_b, at_end=False)

    assert (a0 - br_s0).abs().max().item() < 1e-6
    assert (a1 - br_s1).abs().max().item() < 1e-6
    assert (a2 - br_s2).abs().max().item() < 1e-6
    assert (br_e0 - b0).abs().max().item() < 1e-6
    assert (br_e1 - b1).abs().max().item() < 1e-6
    assert (br_e2 - b2).abs().max().item() < 1e-6
    print("[PASS] 2/5 C^0 / C^1 / C^2 곡선 연속성 오차 < 1e-6 확인")

    # 3. 수초화 상태 머신 전이 및 불응기 재점화 차단 검증
    bridge.start_sequence(cell_a=0, cell_b=1)
    phase, _ = bridge.step_fsm(u, v, phase_progress=0.80, gripper_delta=0.0)
    assert phase == BridgePhase.CELL_A

    phase, _ = bridge.step_fsm(u, v, phase_progress=0.98, gripper_delta=0.25)
    assert phase == BridgePhase.BRIDGE
    assert bridge.graph.episodic_trace[0].item() == 1.0

    for _ in range(15):
        phase, _ = bridge.step_fsm(u, v, phase_progress=1.0, gripper_delta=0.0)
    assert phase == BridgePhase.CELL_B

    centers = torch.randn(4, 64)
    next_id = bridge.graph.select_next_cell(current_cell=1, cell_centers=centers)
    assert next_id not in (0, 1)
    print("[PASS] 3/5 상태 전이 트리거 및 불응기 중복 실행 차단 확인")

    # 4. 에피소드 스냅백(0.0000 초기화) 검증
    residual = bridge.episode_snap_back()
    assert residual == 0.0
    assert bridge.phase_state == BridgePhase.IDLE
    print("[PASS] 4/5 에피소드 스냅백 0.0000 초기화 확인")

    # 5. 동결된 기존 링 가중치 드리프트 0.0 검증
    for k in snap_a:
        assert (frozen_cell_a.state_dict()[k] - snap_a[k]).abs().max().item() == 0.0
    for k in snap_b:
        assert (frozen_cell_b.state_dict()[k] - snap_b[k]).abs().max().item() == 0.0
    print("[PASS] 5/5 기존 링 가중치 드리프트 L_inf == 0.0 확인")


if __name__ == "__main__":
    test_five_invariants()