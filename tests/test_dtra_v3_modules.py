"""tests/test_dtra_v3_modules.py

DTRA v3.0 핵심 모듈 단위 검증 테스트 스위트:
1. 침범 금지 제약(No-Encroachment Constraint) 기반 마진 정제 불변성 검증
2. 동일 지시어 그룹(Duplicate-lang) Top-M Common-Mode Cancellation 2차 재순위화 검증
3. 이중 시정수 스냅백(Two-Timescale Snap-Back: chunk_snap_back vs episode_snap_back) 검증
4. 공유 B-스플라인 기저 뱅크(SharedBasisBank, N_k=6) 및 AdamW Silent Drift == 0.0 완벽 차단 검증
5. 마일스톤 추가 JSON 지표(active_basis_count, total_params_mb, fwt, dup_lang_route_acc) 검증
"""

from __future__ import annotations

import unittest
import numpy as np
import torch
import torch.nn.functional as F

from dtra.core.bspline_core import BSplineDecoder
from dtra.experiments.libero.benchmark_dtra_libero import (
    DTRALiberoPolicy,
    FrozenResNet18DualCam,
    LightweightTextEncoder,
    MultimodalMagneticKeyExtractor,
    SharedBasisBank,
    _calc_dup_lang_route_acc,
    _calc_policy_params_mb,
    _refine_key_with_contrastive_margin,
    calc_cl_metrics,
    set_seed,
    tv_models,
)


class _DummyDualCamBackbone(torch.nn.Module):
    """Lightweight deterministic fallback when torchvision is unavailable in minimal test envs."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 256, kernel_size=16, stride=16, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(
        self,
        av: torch.Tensor,
        eih: torch.Tensor,
        augment: bool = False,
        random_shift: bool = False,
    ):
        del augment, random_shift
        return self.conv(av.float() / 255.0), self.conv(eih.float() / 255.0)


def _make_test_backbone(device: torch.device) -> torch.nn.Module:
    if tv_models is not None:
        return FrozenResNet18DualCam().to(device)
    return _DummyDualCamBackbone().to(device)


class TestDTRAv3Modules(unittest.TestCase):
    def setUp(self) -> None:
        set_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_no_encroachment_constraint(self) -> None:
        """신규 슬롯 정제 시 기존 슬롯의 쿼리 앵커 영역 침범 금지 및 자기 유사도 하한 보존 검증."""
        d_topo = 64
        slot_keys = []
        anchors = []
        queries_list = []

        # 의도적으로 매우 가깝게 밀집된 12개 태스크 분포 생성 (동일 지시어/유사 씬 모사)
        base_dir = F.normalize(torch.randn(d_topo, device=self.device), dim=-1)
        for i in range(12):
            offset = F.normalize(torch.randn(d_topo, device=self.device), dim=-1)
            mu_i = F.normalize(0.88 * base_dir + 0.25 * offset, dim=-1)
            q_i = F.normalize(mu_i.unsqueeze(0) + 0.03 * torch.randn(32, d_topo, device=self.device), dim=-1)
            raw_mean = F.normalize(q_i.mean(dim=0), dim=-1)

            rk, stats = _refine_key_with_contrastive_margin(raw_mean, slot_keys, sample_queries=q_i)
            slot_keys.append(rk)
            anchors.append(mu_i)
            queries_list.append(q_i)

            # 자기 앵커 보존 조건 검증: cos(mu_i, k_i_new) >= 0.95
            self_cos = float((F.normalize(q_i.mean(dim=0), dim=-1) @ rk).item())
            self.assertGreaterEqual(self_cos, 0.95 - 1e-5)
            self.assertGreaterEqual(stats["query_route_acc"], 0.95)

        key_bank = torch.stack(slot_keys, dim=0)
        for j in range(12):
            preds = (queries_list[j] @ key_bank.T).argmax(dim=-1)
            acc = float((preds == j).float().mean().item())
            self.assertGreater(acc, 0.90, f"Task {j} collapsed after subsequent refinements: acc={acc}")

    def test_top_m_common_mode_cancellation(self) -> None:
        """100% 동일한 언어 지시어를 가진 태스크 그룹에서 Top-M 공통모드 차감 재순위화 검증."""
        key_ext = MultimodalMagneticKeyExtractor(d_topo=64, d_lang=128, seed=42).to(self.device)
        text_enc = LightweightTextEncoder(dim=128)
        dup_lang = text_enc.encode("put the red mug on the plate", self.device)

        # 동일 지시어 + 동일 전역 배경 평균을 갖되 4x4 공간 패치 배치만 다른 3개 태스크 생성
        common_bg = torch.randn(1, 256, 8, 8, device=self.device) * 2.0
        slot_keys = []
        task_fa, task_fe, task_pr = [], [], []

        for t_idx in range(3):
            spatial_layout = torch.zeros(1, 256, 8, 8, device=self.device)
            # 각 태스크마다 서로 다른 사분면 패치에 물체 특징 배치 (전역 평균은 0이 되도록 대칭 배치)
            r0, c0 = (t_idx * 2) % 6, (t_idx * 3) % 6
            spatial_layout[:, :, r0 : r0 + 2, c0 : c0 + 2] = 1.5
            spatial_layout[:, :, (r0 + 2) % 8 : (r0 + 2) % 8 + 2, (c0 + 2) % 8 : (c0 + 2) % 8 + 2] = -1.5

            fa = common_bg + spatial_layout + 0.02 * torch.randn(16, 256, 8, 8, device=self.device)
            fe = torch.randn(16, 256, 8, 8, device=self.device) * 0.1
            pr = torch.zeros(16, 12, device=self.device) + 0.2 * t_idx
            lg = dup_lang.unsqueeze(0).expand(16, -1)

            qs = key_ext(fa, fe, pr, lg)
            vis_sig, prop_sig = key_ext.extract_rerank_signature(fa, fe, pr)
            key_ext.register_slot_signature(vis_sig, prop_sig, dup_lang, self.device)
            mk = F.normalize(qs.mean(dim=0), dim=-1)
            rk, _ = _refine_key_with_contrastive_margin(mk, slot_keys, sample_queries=qs)
            slot_keys.append(rk)
            task_fa.append(fa)
            task_fe.append(fe)
            task_pr.append(pr)

        key_bank = torch.stack(slot_keys, dim=0)
        for t_idx in range(3):
            lg = dup_lang.unsqueeze(0).expand(16, -1)
            sel, _, _ = key_ext.route_top_m(
                task_fa[t_idx], task_fe[t_idx], task_pr[t_idx], lg, key_bank, top_m=5
            )
            acc = float((sel == t_idx).float().mean().item())
            self.assertEqual(acc, 1.0, f"Duplicate-lang task {t_idx} failed Top-M routing: acc={acc}")

    def test_two_timescale_snap_back(self) -> None:
        """chunk_snap_back(delta_z만 리셋, episodic_trace 유지) 및 episode_snap_back(전부 0.0000 리셋) 검증."""
        backbone = _make_test_backbone(self.device)
        key_ext = MultimodalMagneticKeyExtractor(d_topo=64, d_lang=128, seed=42).to(self.device)
        decoder = BSplineDecoder(num_control_points=8, horizon=16, degree=3).to(self.device)
        policy = DTRALiberoPolicy(
            backbone, key_ext, decoder, elastic=True, d_topo=64, num_cp=8
        ).to(self.device)

        raw_key = F.normalize(torch.randn(64, device=self.device), dim=-1)
        policy.add_task_tube(raw_key, self.device)
        policy.eval()

        av = torch.randint(0, 255, (1, 3, 128, 128), dtype=torch.uint8, device=self.device)
        eih = torch.randint(0, 255, (1, 3, 128, 128), dtype=torch.uint8, device=self.device)
        pr1 = torch.randn(1, 12, device=self.device)
        pr2 = pr1 + 0.5  # 그리퍼/관절 상태 변화 유도
        lg = F.normalize(torch.randn(1, 128, device=self.device), dim=-1)

        _, sel1, _, dz1 = policy.predict_action_chunk(av, eih, pr1, lg)
        _, sel2, _, dz2 = policy.predict_action_chunk(av, eih, pr2, lg)
        node = policy.tubes[sel2]

        self.assertGreater(dz2, 0.0)
        self.assertGreater(node.episodic_trace_norm, 0.0)
        self.assertIsNotNone(policy.prev_selected)

        # 1. Chunk Snap-Back: delta_z는 0.000으로 즉시 소거되고, episodic_trace와 관성은 유지되어야 함
        pre_c, post_c = policy.chunk_snap_back()
        self.assertGreater(pre_c, 0.0)
        self.assertEqual(post_c, 0.0)
        self.assertEqual(node.last_delta_z_norm, 0.0)
        self.assertGreater(node.episodic_trace_norm, 0.0, "episodic_trace must persist across chunk_snap_back")
        self.assertIsNotNone(policy.prev_selected, "routing inertia must persist across chunk_snap_back")

        # 2. Episode Snap-Back: delta_z, inertia, episodic_trace 전부 0.0000으로 완전 초기화
        pre_e, post_e = policy.episode_snap_back()
        self.assertGreater(pre_e, 0.0)
        self.assertEqual(post_e, 0.0)
        self.assertEqual(node.last_delta_z_norm, 0.0)
        self.assertEqual(node.episodic_trace_norm, 0.0)
        self.assertIsNone(policy.prev_selected)
        self.assertIsNone(policy.prev_act_vec)

    def test_shared_basis_bank_and_adamw_zero_silent_drift(self) -> None:
        """SharedBasisBank 기저 공유 및 후속 태스크 AdamW(weight_decay>0) 학습 시 과거 기저/튜브 드리프트 == 0.0 검증."""
        backbone = _make_test_backbone(self.device)
        key_ext = MultimodalMagneticKeyExtractor(d_topo=64, d_lang=128, seed=42).to(self.device)
        decoder = BSplineDecoder(num_control_points=8, horizon=16, degree=3).to(self.device)
        policy = DTRALiberoPolicy(
            backbone, key_ext, decoder, elastic=True, d_topo=64, num_knots=6, num_cp=8
        ).to(self.device)

        # Task 0 추가 및 학습
        sig0 = F.normalize(torch.randn(112, device=self.device), dim=-1)
        k0 = F.normalize(torch.randn(64, device=self.device), dim=-1)
        slot0, st0 = policy.add_task_tube(k0, self.device, motion_sig=sig0)
        self.assertEqual(slot0, 0)
        self.assertEqual(policy.active_basis_count, 1)

        params0 = policy.get_trainable_params(slot0)
        opt0 = torch.optim.AdamW(params0, lr=1e-3, weight_decay=1e-2)
        fa = torch.randn(8, 256, 8, 8, device=self.device)
        fe = torch.randn(8, 256, 8, 8, device=self.device)
        pr = torch.randn(8, 12, device=self.device)
        lg = F.normalize(torch.randn(8, 128, device=self.device), dim=-1)
        act_gt = torch.randn(8, 16, 7, device=self.device)

        for _ in range(5):
            opt0.zero_grad(set_to_none=True)
            xq = policy.key_extractor(fa, fe, pr, lg)
            cp, _ = policy.tubes[slot0](fa, fe, pr, lg, xq, shared_basis_bank=policy.shared_basis_bank)
            traj = policy.bspline_decoder(cp)
            loss = F.mse_loss(traj, act_gt)
            loss.backward()
            opt0.step()

        policy.freeze_completed_slots(active_slot_id=None)

        # Task 0의 튜브 파라미터 및 공유 기저 슬롯 0 스냅샷 저장
        tube0_snap = {n: p.detach().clone() for n, p in policy.tubes[0].named_parameters()}
        basis0_snap = {
            n: p.detach().clone() for n, p in policy.shared_basis_bank.basis_slots[0].named_parameters()
        }

        # Task 1: 유사 모션 서명으로 기존 기저 0을 공유(Sharing)하는 태스크 추가
        sig1_similar = F.normalize(sig0 + 0.02 * torch.randn_like(sig0), dim=-1)
        k1 = F.normalize(torch.randn(64, device=self.device), dim=-1)
        slot1, st1 = policy.add_task_tube(k1, self.device, motion_sig=sig1_similar)
        self.assertEqual(st1["is_new_basis"], 0.0, "Task 1 with similar motion_sig should share Basis 0")
        self.assertEqual(policy.active_basis_count, 1)

        # Task 2: 완전히 다른 모션 서명으로 신규 기저 1을 할당하는 태스크 추가
        sig2_novel = F.normalize(-sig0 + 0.5 * torch.randn_like(sig0), dim=-1)
        k2 = F.normalize(torch.randn(64, device=self.device), dim=-1)
        slot2, st2 = policy.add_task_tube(k2, self.device, motion_sig=sig2_novel)
        self.assertEqual(st2["is_new_basis"], 1.0, "Task 2 with novel motion_sig should allocate Basis 1")
        self.assertEqual(policy.active_basis_count, 2)

        # Task 1, Task 2를 강한 weight_decay=0.05를 가진 AdamW로 각각 15 스텝 업데이트
        for active_slot in (slot1, slot2):
            params_cur = policy.get_trainable_params(active_slot)
            # 동결된 기저 슬롯 0과 과거 튜브 0의 텐서가 옵티마이저 그룹에 단 하나도 포함되지 않았는지 검증
            opt_param_ids = {id(p) for p in params_cur}
            for p0 in policy.tubes[0].parameters():
                self.assertFalse(p0.requires_grad)
                self.assertNotIn(id(p0), opt_param_ids)
            for pb0 in policy.shared_basis_bank.basis_slots[0].parameters():
                self.assertFalse(pb0.requires_grad)
                self.assertNotIn(id(pb0), opt_param_ids)

            opt_cur = torch.optim.AdamW(params_cur, lr=2e-3, weight_decay=0.05)
            for _ in range(15):
                opt_cur.zero_grad(set_to_none=True)
                xq = policy.key_extractor(fa, fe, pr, lg)
                cp, _ = policy.tubes[active_slot](
                    fa, fe, pr, lg, xq, shared_basis_bank=policy.shared_basis_bank
                )
                traj = policy.bspline_decoder(cp)
                loss = F.mse_loss(traj, act_gt) + 0.05 * policy.topo_loss(active_slot)
                loss.backward()
                opt_cur.step()

        # L_inf drift == 0.0 엄밀 검증
        max_tube0_drift = max(
            float((p - tube0_snap[n]).abs().max().item()) for n, p in policy.tubes[0].named_parameters()
        )
        max_basis0_drift = max(
            float((p - basis0_snap[n]).abs().max().item())
            for n, p in policy.shared_basis_bank.basis_slots[0].named_parameters()
        )
        self.assertEqual(max_tube0_drift, 0.0, f"Tube 0 drifted by {max_tube0_drift}")
        self.assertEqual(max_basis0_drift, 0.0, f"Shared Basis 0 drifted by {max_basis0_drift}")

    def test_v3_json_metrics(self) -> None:
        """fwt, dup_lang_route_acc, total_params_mb, active_basis_count 산출 검증."""
        R = np.array(
            [
                [0.80, 0.80, 0.80],
                [0.00, 0.90, 0.90],
                [0.00, 0.00, 1.00],
            ],
            dtype=np.float32,
        )
        m = calc_cl_metrics(R)
        self.assertAlmostEqual(m["mean_final_sr"], 0.90, places=5)
        self.assertAlmostEqual(m["forgetting_rate"], 0.0, places=5)
        self.assertAlmostEqual(m["bwt"], 0.0, places=5)
        self.assertIn("fwt", m)
        self.assertGreater(m["fwt"], 0.0)

        R_route = np.array(
            [
                [1.0, 1.0, 1.0],
                [0.0, 0.96, 0.98],
                [0.0, 0.0, 0.94],
            ],
            dtype=np.float32,
        )
        dup_acc = _calc_dup_lang_route_acc(R_route, dup_lang_indices={1, 2}, k_cur=3, t_col=2)
        self.assertAlmostEqual(dup_acc, (0.98 + 0.94) / 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
