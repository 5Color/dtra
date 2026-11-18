"""tests/test_bspline_policy.py

B-Spline VLA 정책 아키텍처 단위 테스트:
1. BSplineVisuomotorPolicy forward shape 및 loss 계산 검증
2. BSplineCVAEPolicy forward shape, KL divergence 및 loss 검증
3. 역전파(Backpropagation) 기울기 소실/폭주 여부 점검
4. 배치 추론 predict 속도 및 결정론적 출력 검증
"""

import unittest
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sp in (str(_REPO_ROOT), str(_REPO_ROOT / "v1_drawing_vla" / "src"), str(_REPO_ROOT / "v1_drawing_vla" / "handvsai")):
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
import torch
import torch.nn.functional as F

from train.bspline_policy import (
    AutonomousABKPolicy,
    BSplineCVAEPolicy,
    BSplineVisuomotorPolicy,
    GRUSequencePolicy,
)


class TestBSplinePolicy(unittest.TestCase):

    def setUp(self):
        self.B = 4
        self.H = 16
        self.M = 6
        self.obs = torch.rand(self.B, 2, 128, 128)
        self.state = torch.rand(self.B, 3)
        self.target_cp = torch.rand(self.B, self.M, 2)
        self.target_act = torch.rand(self.B, self.H, 3)
        self.target_pen = torch.randint(0, 2, (self.B, self.H)).float()

    def test_mlp_policy_forward_and_loss(self):
        policy = BSplineVisuomotorPolicy(num_control_points=self.M, horizon=self.H)
        loss, metrics = policy.loss(
            self.obs, self.state, self.target_cp, self.target_act, self.target_pen
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

        # 역전파 검증
        loss.backward()
        for name, param in policy.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertTrue(torch.all(torch.isfinite(param.grad)), f"NaN/Inf grad in {name}")

        # 추론 검증
        pred_cp, pred_traj, pen_prob = policy.predict(self.obs, self.state)
        self.assertEqual(pred_cp.shape, (self.B, self.M, 2))
        self.assertEqual(pred_traj.shape, (self.B, self.H, 2))
        self.assertEqual(pen_prob.shape, (self.B, self.H))
        self.assertTrue((pred_cp >= 0.0).all() and (pred_cp <= 1.0).all())

    def test_cvae_policy_forward_and_loss(self):
        policy = BSplineCVAEPolicy(num_control_points=self.M, horizon=self.H, latent_dim=8)
        loss, metrics = policy.loss(
            self.obs, self.state, self.target_cp, self.target_act, self.target_pen
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_kl", metrics)
        self.assertGreaterEqual(metrics["loss_kl"].item(), 0.0)

        loss.backward()
        for name, param in policy.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.all(torch.isfinite(param.grad)))

        pred_cp, pred_traj, pen_prob = policy.predict(self.obs, self.state)
        self.assertEqual(pred_cp.shape, (self.B, self.M, 2))
        self.assertEqual(pred_traj.shape, (self.B, self.H, 2))

    def test_gru_sequence_policy(self):
        policy = GRUSequencePolicy(horizon=self.H)
        pred_traj, pen_logits = policy(self.obs, self.state)
        self.assertEqual(pred_traj.shape, (self.B, self.H, 2))
        self.assertEqual(pen_logits.shape, (self.B, self.H))
        self.assertTrue((pred_traj >= 0.0).all() and (pred_traj <= 1.0).all())

        target_pen = torch.randint(0, 2, (self.B, self.H)).float()
        loss, metrics = policy.loss(self.obs, self.state, target_traj=pred_traj.detach(), target_pen=target_pen)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_traj", metrics)
        loss.backward()
        self.assertIsNotNone(policy.gru_cell.weight_ih.grad)

        cp_proxy, pred_t, pen_p = policy.predict(self.obs, self.state)
        self.assertEqual(cp_proxy.shape, (self.B, self.M, 2))
        self.assertEqual(pred_t.shape, (self.B, self.H, 2))
        self.assertEqual(pen_p.shape, (self.B, self.H))

    def test_autonomous_abk_policy_expansion_routing_and_freeze(self):
        torch.manual_seed(0)
        policy = AutonomousABKPolicy(num_control_points=self.M, horizon=self.H, temperature=0.05)

        # 서로 공간 패턴이 뚜렷이 다른 2개 도안 생성 (좌상단 사각형 vs 우하단 십자가)
        tgt_a = torch.zeros(2, 1, 128, 128)
        tgt_a[:, :, 20:56, 20:56] = 1.0
        tgt_b = torch.zeros(2, 1, 128, 128)
        tgt_b[:, :, 70:110, 88:96] = 1.0
        tgt_b[:, :, 86:94, 70:110] = 1.0

        # Task 0 최초 등록 -> Slot 0 생성
        idx0, exp0, _ = policy.register_or_expand_slot(tgt_a, novelty_threshold=0.85)
        self.assertEqual(idx0, 0)
        self.assertTrue(exp0)
        self.assertEqual(len(policy.slots), 1)
        self.assertFalse(policy.slots[0].is_residual)

        # 동일 도안 재입력 -> 확장 없이 Slot 0 재사용
        idx0_repeat, exp0_repeat, sim_same = policy.register_or_expand_slot(tgt_a, novelty_threshold=0.85)
        self.assertEqual(idx0_repeat, 0)
        self.assertFalse(exp0_repeat)
        self.assertGreater(sim_same, 0.99)

        # 신규 도안 입력 -> 유사도 < 0.85 감지, 백본/Slot 0 동결 후 Slot 1 확장
        idx1, exp1, sim_novel = policy.register_or_expand_slot(tgt_b, novelty_threshold=0.85)
        self.assertEqual(idx1, 1)
        self.assertTrue(exp1)
        self.assertLess(sim_novel, 0.85)
        self.assertEqual(len(policy.slots), 2)
        self.assertTrue(policy.slots[1].is_residual)
        self.assertIsNotNone(policy.slots[1].prior_weight)
        self.assertAlmostEqual(policy.slots[1].prior_weight.item(), 1.0, places=5)
        self.assertLess(policy.slots[1].pts_head.weight.abs().max().item(), 0.02)

        # 동결 검증: encoder, state_net, Slot 0는 requires_grad=False / Slot 1은 True
        self.assertTrue(all(not p.requires_grad for p in policy.encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in policy.state_net.parameters()))
        self.assertTrue(all(not p.requires_grad for p in policy.slots[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in policy.slots[1].parameters()))

        # Task-ID 없는 순전파 / 추론 / 역전파 검증
        obs_b = torch.cat([torch.zeros(2, 1, 128, 128), tgt_b], dim=1)
        st_b = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 1.0]])
        cp_fwd, traj_fwd, pen_logits = policy(obs_b, st_b)
        self.assertEqual(cp_fwd.shape, (2, self.M, 2))
        self.assertEqual(traj_fwd.shape, (2, self.H, 2))
        self.assertEqual(pen_logits.shape, (2, self.H))

        cp_pred, traj_pred, pen_prob = policy.predict(obs_b, st_b)
        self.assertEqual(cp_pred.shape, (2, self.M, 2))
        self.assertEqual(traj_pred.shape, (2, self.H, 2))
        self.assertTrue((pen_prob >= 0.0).all() and (pen_prob <= 1.0).all())

        # Softmax 라우팅 가중치 날카로움 검증 (대각 >= 0.90)
        w = policy.compute_routing_weights(torch.cat([tgt_a[:1], tgt_b[:1]], dim=0))
        self.assertEqual(w.shape, (2, 2))
        self.assertTrue(torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-5))
        self.assertGreater(w[0, 0].item(), 0.90)
        self.assertGreater(w[1, 1].item(), 0.90)

        # Slot 1만 그래디언트 갱신되고 Slot 0 그래디언트는 None 확인
        loss, _ = policy.loss(
            obs_b, st_b, torch.rand(2, self.M, 2), torch.rand(2, self.H, 2), torch.ones(2, self.H)
        )
        loss.backward()
        self.assertIsNone(policy.slots[0].pts_head.weight.grad)
        self.assertIsNotNone(policy.slots[1].pts_head.weight.grad)

        # 공간 포인터 피처 및 추론 프루닝 동작 검증
        policy.eval()
        pointer = policy._compute_spatial_pointer(obs_b, st_b)
        self.assertEqual(pointer.shape, (2, 7))
        # 단일 샘플 추론 시 프루닝 활성화 테스트
        obs_single = obs_b[:1]
        st_single = st_b[:1]
        cp_single, traj_single, pen_single = policy.predict(obs_single, st_single)
        self.assertEqual(cp_single.shape, (1, self.M, 2))
        self.assertEqual(traj_single.shape, (1, self.H, 2))
        self.assertEqual(pen_single.shape, (1, self.H))

        # Hard Argmax 라우팅 검증: Slot 0에 매칭되는 쿼리는 Slot 1 교란과 무관하게 동일한 출력 유지
        obs_a0 = torch.cat([torch.zeros(1, 1, 128, 128), tgt_a[:1]], dim=1)
        st_a0 = st_b[:1]
        policy.reset_cache()
        cp_arg1, _, _ = policy.predict(obs_a0, st_a0, routing_mode="argmax")
        with torch.no_grad():
            policy.slots[1].pts_head.bias.add_(5.0)
        policy.reset_cache()
        cp_arg2, _, _ = policy.predict(obs_a0, st_a0, routing_mode="argmax")
        self.assertTrue(torch.allclose(cp_arg1, cp_arg2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
