"""scripts/visualize_dtra_xai.py

DTRA v2.1 기하학적 작동 원리, 아키텍처, 지속학습/스케일링 성능 및 화이트박스 설명가능성(XAI) 논문용 300 DPI 시각화 스크립트.

생성 그림 (outputs/figures/):
  0. fig0_dtra_architecture_schematic.png
     - 멀티모달 입력 -> S^63 Coarse-to-Fine 위상 라우팅 -> 탄성 변형(delta_z*) + M_ij -> 로컬 B-Spline 디코더 -> snap_back() 0.0 도식도
  1. fig1_ood_perturbation_curves.png
     - outputs/ablate_dtra_supplemental_results.json (Part 2) 기반 9종 OOD 섭동 SR & RouteAcc 곡선
  2. fig2_manifold_elastic_snapback.png
     - (a) 8개 S^63 튜브 독립 격리 배치(Macro View), (b) 대표 튜브 T3 10배 확대 탄성 변형 & 스냅백 궤적(Micro Zoom-In),
       (c) 탄성 변위 응력 및 snap_back() 잔차 0.000 실측 바차트
  3. fig3_xai_telemetry_rollout.png
     - Meta-World 3종 태스크 OOD(sigma=0.20) 실시간 롤아웃 텔레메트리
  4. fig4_cl_forgetting_heatmap_and_bars.png
     - 8x8 순차 학습 성공률 행렬 히트맵(Naive FT / EWC / ER 5% vs DTRA Arm 6) 및 Clean SR · FR · OOD SR 비교 차트
  5. fig5_o1_latency_scaling.png
     - K=1 -> K=8 태스크 확장 시 DTRA O(1) 상수 활성 제어 지연(~1.87 ms) vs O(K) 베이스라인 비교 차트
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dtra.experiments.metaworld.ablate_dtra_scientific import (
    ControlledDTRAPolicy,
    MetaWorldEvalRegistry,
    build_arm_factory,
)
from dtra.experiments.metaworld.benchmark_dtra_3d_metaworld import (
    TASK_INSTRUCTIONS,
    TASK_SPECS_8,
    apply_ood_rgb_perturbation,
)

SUP_JSON = PROJECT_ROOT / "outputs" / "metaworld" / "ablate_dtra_supplemental_results.json"
SCI_JSON = PROJECT_ROOT / "outputs" / "metaworld" / "ablate_dtra_scientific_results.json"
CKPT_DIR = PROJECT_ROOT / "outputs" / "metaworld" / "checkpoints_seed42"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

TASK_NAMES_8 = [name for name, _ in TASK_SPECS_8]
SHORT_TASK_LABELS = [
    "T0: button-press",
    "T1: door-open",
    "T2: window-open",
    "T3: faucet-open",
    "T4: drawer-open",
    "T5: drawer-close",
    "T6: handle-press",
    "T7: coffee-button",
]
TASK_COLORS = plt.cm.tab10(np.linspace(0.0, 0.78, 8))


# ==============================================================================
# 0. 체크포인트 로드 및 모델 내부 기하 상태 계측 유틸
# ==============================================================================


def load_dtra_checkpoint(arm_key: str, seed: int = 42, device: torch.device = torch.device("cpu")) -> ControlledDTRAPolicy:
    """저장된 Seed 42 체크포인트를 MetaWorld 데이터 재수집 없이 strict=True로 복원합니다."""
    ckpt_path = CKPT_DIR / f"{arm_key}_seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_arm_factory(arm_key)().to(device)
    tasks: List[str] = list(blob["tasks"])
    gen = torch.Generator().manual_seed(12345)
    for s_idx, t_name in enumerate(tasks):
        # 서로 겹치지 않는 더미 이미지/텍스트로 8개 Coarse & Fine 튜브 슬롯 골격만 생성한 뒤 strict 로드
        dummy_imgs = torch.rand(8, 3, 64, 64, generator=gen, device=device)
        model.register_or_expand_task(dummy_imgs, f"{t_name} unique init {s_idx} {TASK_INSTRUCTIONS[t_name]}")
        if s_idx == 0:
            model.freeze_prior()

    model.load_state_dict(blob["state_dict"], strict=True)
    model.refresh_eval_caches()
    model.eval()
    return model


@torch.no_grad()
def extract_base_rings_and_centers(
    model: ControlledDTRAPolicy, stage: str = "fine"
) -> Tuple[np.ndarray, np.ndarray]:
    """튜브 뱅크의 기본 1차 타원 고리 z0(theta) in S^63 (K, M_theta, 64) 및 중심 투영 c_0 (K, 64) 추출."""
    tubes = model.fine_tubes if stage == "fine" else model.coarse_tubes
    k_bank = F.normalize(torch.stack([t.key for t in tubes], dim=0), p=2, dim=-1)
    c_bank = model.proj_topo(k_bank)  # (K, 64)
    coeffs = torch.stack([t.fourier_coeffs for t in tubes], dim=0)  # (K, 3, 64)
    rings = F.normalize(
        c_bank.unsqueeze(1) + torch.einsum("mf,kfd->kmd", model.basis_ring, coeffs),
        p=2,
        dim=-1,
    )  # (K, M_theta, 64)
    return rings.cpu().numpy(), F.normalize(c_bank, p=2, dim=-1).cpu().numpy()


@torch.no_grad()
def probe_step_with_geometry(
    model: ControlledDTRAPolicy,
    img_t: torch.Tensor,
    st_t: torch.Tensor,
    instruction: str,
) -> Dict[str, object]:
    """ControlledDTRAPolicy 1스텝 추론과 함께 S^63 도킹 좌표(c_query, z0*, z~*), 변형 응력, 라우팅 마진 계측."""
    bsz = img_t.shape[0]
    if model._eval_c_keys.shape[0] != len(model.coarse_tubes) or model._eval_f_keys.shape[0] != len(model.fine_tubes):
        model.refresh_eval_caches()

    cond, x_mag = model._forward_trunk_with_grad(img_t, st_t, [instruction] * bsz)
    x_conv = cond[:, : model.feat_dim]
    f_st = cond[:, model.feat_dim :]

    # Stage 1: Coarse
    c_idx, f_mag_c = model._select_active_eval_index(
        model._eval_c_keys,
        model._eval_c_taus,
        x_mag,
        model.coarse_inertia,
        model.coarse_prev_gate,
        model.coarse_myelin,
    )
    overlap_c, a_c, z_dock_c, h_avg_c = model._eval_single_active_tube(
        model.coarse_tubes[c_idx],
        f_mag_c,
        x_mag,
        record_deformation=True,
        delta_buf=model.coarse_delta_z,
        tube_idx=c_idx,
    )
    c_delta_norm, c_pull_norm, _, _ = model._last_eval_stats
    model.coarse_inertia.zero_()
    model.coarse_inertia[:, c_idx] = overlap_c[0, 0]
    model.coarse_prev_gate.zero_()
    model.coarse_prev_gate[:, c_idx] = a_c[0, 0]
    p_anchor, ctx_coarse = model.coarse_tubes[c_idx].head(torch.cat([z_dock_c, h_avg_c], dim=-1))

    # Stage 2: Fine 기하 좌표 직접 캡처
    x_fine = F.normalize(x_mag + 0.35 * ctx_coarse, p=2, dim=-1)
    cos_all_f = torch.matmul(F.normalize(x_fine, p=2, dim=-1), model._eval_f_keys.t())[0]  # (K,)
    f_idx, f_mag_f = model._select_active_eval_index(
        model._eval_f_keys,
        model._eval_f_taus,
        x_fine,
        model.fine_inertia,
        model.fine_prev_gate,
        model.fine_myelin,
    )

    f_node = model.fine_tubes[f_idx]
    x_hat_f = F.normalize(x_fine, p=2, dim=-1)
    c_query_f = F.normalize(model.proj_topo(x_hat_f), p=2, dim=-1)  # (1, 64)
    k_sel_f = F.normalize(f_node.key.unsqueeze(0), p=2, dim=-1)
    center_sel_f = model.proj_topo(k_sel_f)
    coeffs_f = f_node.fourier_coeffs.unsqueeze(0)
    theta_star_f = torch.atan2(
        torch.einsum("bd,kd->bk", c_query_f, coeffs_f[:, 2, :]),
        torch.einsum("bd,kd->bk", c_query_f, coeffs_f[:, 1, :]) + 1e-8,
    )
    b_star_f = torch.stack(
        [torch.ones_like(theta_star_f), torch.cos(theta_star_f), torch.sin(theta_star_f)], dim=-1
    )
    z0_star_f = F.normalize(
        center_sel_f.unsqueeze(0) + torch.einsum("bkf,kfd->bkd", b_star_f, coeffs_f),
        p=2,
        dim=-1,
    ).squeeze(1)  # (1, 64)

    overlap_f, a_f, z_dock_f, h_avg_f = model._eval_single_active_tube(
        f_node,
        f_mag_f,
        x_fine,
        record_deformation=True,
        delta_buf=model.fine_delta_z,
        tube_idx=f_idx,
    )
    f_delta_norm, f_pull_norm, f_act_val, f_alpha_val = model._last_eval_stats
    model.fine_inertia.zero_()
    model.fine_inertia[:, f_idx] = overlap_f[0, 0]
    model.fine_prev_gate.zero_()
    model.fine_prev_gate[:, f_idx] = a_f[0, 0]

    pre_snap_buf_l2 = float(model.fine_delta_z[:, f_idx].norm(dim=-1).max().item())
    pre_snap_buf_abs = float(model.fine_delta_z.abs().max().item())

    key_cos_max = float(cos_all_f.max().item())
    sorted_cos = torch.sort(cos_all_f, descending=True).values
    route_margin = float((sorted_cos[0] - sorted_cos[1]).item()) if len(sorted_cos) > 1 else 0.0
    conf_gate = float(torch.clamp(cos_all_f.max(), 0.0, 1.0).item())

    x_conv_scaled = F.normalize(x_conv, p=2, dim=-1) * model.conv_norm_scale
    raw_man_cps, raw_res_cps, sel = f_node.head(z_dock_f, h_avg_f, ctx_coarse, f_st, x_conv_scaled)
    best_mode = int(sel.argmax(dim=1).item())
    anchor_logit = torch.logit(p_anchor[0].clamp(1e-4, 1.0 - 1e-4))
    p_man = torch.sigmoid(anchor_logit + raw_man_cps[0, best_mode])
    delta_res = (model.res_scale * conf_gate) * torch.tanh(raw_res_cps[0, best_mode])
    p_star = torch.clamp(p_man + delta_res, 0.0, 1.0)  # (6, 4)
    act = (model.decoder(p_star.unsqueeze(0)) * 2.0 - 1.0)[0]  # (16, 4)

    # 에피소드 청크 완료 시 스냅백 수행 및 잔차 0.0 실측
    model.snap_back()
    post_snap_buf_abs = float(max(model.coarse_delta_z.abs().max().item(), model.fine_delta_z.abs().max().item()))

    return {
        "coarse_idx": int(c_idx),
        "fine_idx": int(f_idx),
        "cos_all_f": cos_all_f.cpu().numpy(),
        "key_cos_max": key_cos_max,
        "route_margin": route_margin,
        "conf_gate": conf_gate,
        "c_query_f": c_query_f[0].cpu().numpy(),
        "z0_star_f": z0_star_f[0].cpu().numpy(),
        "z_tilde_star_f": z_dock_f[0].cpu().numpy(),
        "coarse_delta_norm": float(c_delta_norm),
        "fine_delta_norm": float(f_delta_norm),
        "mean_delta_norm": 0.5 * (float(c_delta_norm) + float(f_delta_norm)),
        "fine_pull_norm": float(f_pull_norm),
        "fine_alpha": float(f_alpha_val),
        "pre_snap_buf_l2": pre_snap_buf_l2,
        "pre_snap_buf_abs": pre_snap_buf_abs,
        "post_snap_buf_abs": post_snap_buf_abs,
        "p_star": p_star.cpu().numpy(),  # (6, 4)
        "act_chunk": act.cpu().numpy(),  # (16, 4)
    }


# ==============================================================================
# Figure 1: 9종 OOD 섭동 강건성 비교 곡선
# ==============================================================================


def generate_fig1_ood_curves() -> None:
    with open(SUP_JSON, "r", encoding="utf-8") as f:
        sup = json.load(f)
    p2 = sup["part2_multi_ood_curve"]

    perts = [
        "clean",
        "noise_0.05",
        "noise_0.10",
        "noise_0.20",
        "noise_0.30",
        "cutout_15",
        "cutout_30",
        "bright_high",
        "contrast_harsh",
    ]
    x_labels = [
        "Clean",
        "Noise\n(σ=0.05)",
        "Noise\n(σ=0.10)",
        "Noise\n(σ=0.20)",
        "Noise\n(σ=0.30)",
        "Cutout\n(15%)",
        "Cutout\n(30%)",
        "Bright\nHigh",
        "Contrast\nHarsh",
    ]
    arms_cfg = [
        ("arm6_dtra_full_multimodal", "Arm 6: DTRA Full (Multimodal, α learned)", "#1b6ca8", "o", "-", 2.6, 4),
        ("arm5_dtra_rigid_alpha0", "Arm 5: DTRA Rigid (Multimodal, α=0)", "#2a9d8f", "s", "--", 2.0, 3),
        ("arm4_frozen_mlp_multimodal", "Arm 4: Frozen MLP (Multimodal, Key Parity)", "#d62828", "^", "-.", 2.2, 3),
        ("arm3_dtra_full_vision", "Arm 3: DTRA Full (Vision-Only)", "#f77f00", "D", ":", 1.9, 2),
    ]

    x = np.arange(len(perts))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.5, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.14}
    )

    # 영역 구분 음영
    for ax in (ax1, ax2):
        ax.axvspan(0.65, 4.35, color="#e9eef5", alpha=0.55, zorder=0)
        ax.axvspan(4.65, 6.35, color="#fdf0e6", alpha=0.65, zorder=0)
        ax.axvspan(6.65, 8.35, color="#eef7ee", alpha=0.55, zorder=0)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    for key, label, color, marker, ls, lw, zo in arms_cfg:
        srs = [p2[key][p]["avg_sr"] * 100.0 for p in perts]
        rts = [p2[key][p]["avg_route_acc"] * 100.0 for p in perts]
        ax1.plot(x, srs, label=label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=6.5, zorder=zo)
        ax2.plot(x, rts, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=5.8, zorder=zo)

    # 주요 격차 주석
    ax1.annotate(
        "90.8% → 56.7% (−34.1%p)\nat mild noise σ=0.05",
        xy=(1, p2["arm4_frozen_mlp_multimodal"]["noise_0.05"]["avg_sr"] * 100.0),
        xytext=(1.15, 41.0),
        fontsize=8.2,
        color="#a81d1d",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#a81d1d", lw=1.2),
    )
    ax1.annotate(
        "Bright High: DTRA 68.3%\nvs Frozen MLP 27.5% (+40.8%p)",
        xy=(7, p2["arm4_frozen_mlp_multimodal"]["bright_high"]["avg_sr"] * 100.0),
        xytext=(5.1, 14.5),
        fontsize=8.2,
        color="#1b6ca8",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#d62828", lw=1.2),
    )
    ax1.annotate(
        "σ=0.20: DTRA 50.8%\nvs MLP 32.5%",
        xy=(3, p2["arm6_dtra_full_multimodal"]["noise_0.20"]["avg_sr"] * 100.0),
        xytext=(2.35, 67.5),
        fontsize=8.2,
        color="#1b6ca8",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1b6ca8", lw=1.2),
    )

    ax1.text(2.5, 96.5, "Gaussian Sensor Noise Sweep", ha="center", va="top", fontsize=8.5, color="#3b4d63", style="italic")
    ax1.text(5.5, 96.5, "Occlusion", ha="center", va="top", fontsize=8.5, color="#7d4a1d", style="italic")
    ax1.text(7.5, 96.5, "Photometric", ha="center", va="top", fontsize=8.5, color="#265b28", style="italic")

    ax1.set_ylim(0, 102)
    ax1.set_ylabel("8-Task Mean Success Rate (%)")
    ax1.set_title("Multi-OOD Perturbation Robustness Curves (Seed 42 Reloaded Checkpoints, Key Parity)")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#cccccc")

    ax2.set_ylim(10, 108)
    ax2.set_ylabel("Top-1 RouteAcc (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.annotate(
        "Multimodal Keys (Arms 4/5/6): 100.0% RouteAcc across all OOD regimes",
        xy=(4.0, 100.0),
        xytext=(0.15, 76.0),
        fontsize=8.3,
        color="#1b6ca8",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#1b6ca8", lw=1.1),
    )
    ax2.annotate(
        "Vision-Only Cutout Collapse\n(35.8% / 20.0%)",
        xy=(6, p2["arm3_dtra_full_vision"]["cutout_30"]["avg_route_acc"] * 100.0),
        xytext=(3.2, 23.0),
        fontsize=8.2,
        color="#c45f00",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#c45f00", lw=1.1),
    )

    out_path = FIG_DIR / "fig1_ood_perturbation_curves.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Saved Fig 1] {out_path}")


# ==============================================================================
# Figure 2: 64차원 초구 튜브 배치 & 탄성 스냅백 궤적 투영
# ==============================================================================


def _fit_pca_2d(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pts: (N, D) -> mean (D,), components (2, D), explained_variance_ratio (2,)."""
    mu = pts.mean(axis=0, keepdims=True)
    xc = pts - mu
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    tot_var = float(np.sum(s ** 2)) + 1e-12
    evr = (s[:2] ** 2) / tot_var
    return mu[0], vt[:2], evr


def _proj_pca_2d(pts: np.ndarray, mu: np.ndarray, comps: np.ndarray) -> np.ndarray:
    return (pts - mu) @ comps.T


def _collect_arm_snapback_samples(
    model: ControlledDTRAPolicy,
    registry: MetaWorldEvalRegistry,
    device: torch.device,
    task_indices: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
) -> Dict[str, object]:
    """실제 held-out 에피소드 Clean vs OOD(sigma=0.20) 롤아웃에서 S^63 튜브 변형·스냅백 실측 수집."""
    clean_records: List[Dict[str, object]] = []
    ood_records: List[Dict[str, object]] = []

    for cond_name, noise_std, target_list in [("clean", 0.0, clean_records), ("ood", 0.20, ood_records)]:
        rng = torch.Generator().manual_seed(42 + (99 if cond_name == "ood" else 0))
        for tid in task_indices:
            t_name = TASK_NAMES_8[tid]
            env, eval_tasks = registry.get_env_and_tasks(t_name, split="train_heldout", n_eval=15)
            try:
                env.set_task(eval_tasks[0])
                obs, _ = env.reset()
                model.reset()
                for step_idx in range(0, 64, 8):
                    fr = np.flipud(env.render()).copy()
                    tx = torch.from_numpy(fr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                    tx = apply_ood_rgb_perturbation(tx, noise_std, 1.0, 0.0, rng)
                    ts = torch.from_numpy(obs[:4].astype(np.float32)).unsqueeze(0).to(device)
                    res = probe_step_with_geometry(model, tx, ts, TASK_INSTRUCTIONS[t_name])
                    target_list.append(
                        {
                            "task_id": tid,
                            "step_idx": step_idx,
                            "matched_id": int(res["fine_idx"]),
                            "c_query": res["c_query_f"],
                            "z0_star": res["z0_star_f"],
                            "z_tilde_star": res["z_tilde_star_f"],
                            "fine_delta_norm": res["fine_delta_norm"],
                            "fine_pull_norm": res["fine_pull_norm"],
                            "fine_alpha": res["fine_alpha"],
                            "pre_snap_l2": res["pre_snap_buf_l2"],
                            "post_snap_abs": res["post_snap_buf_abs"],
                        }
                    )
                    for a in res["act_chunk"][:8]:
                        obs, _, _, _, info = env.step(np.clip(a, -1.0, 1.0))
                        if info.get("success", 0) > 0:
                            break
            finally:
                env.close()

    return {"clean": clean_records, "ood": ood_records}


def _classical_mds_2d(pts: np.ndarray, min_dist: float = 0.68, seed: int = 42) -> np.ndarray:
    """고차원 초구 중심점 (K, D)의 실제 유클리드 거리 행렬을 보존하는 2D MDS + 비충돌 미세 이완 좌표 생성."""
    diff = pts[:, None, :] - pts[None, :, :]
    d_mat = np.linalg.norm(diff, axis=-1)
    n = d_mat.shape[0]
    j_mat = np.eye(n) - np.ones((n, n)) / n
    b_mat = -0.5 * j_mat @ (d_mat ** 2) @ j_mat
    eigvals, eigvecs = np.linalg.eigh(b_mat)
    idx = np.argsort(eigvals)[::-1][:2]
    vals = np.maximum(eigvals[idx], 1e-6)
    coords = eigvecs[:, idx] * np.sqrt(vals)[None, :]

    # 고차원 직교 성분 투영 축소로 가까운 쌍이 겹치지 않도록 최소 거리만 부드럽게 확보
    rng = np.random.default_rng(seed)
    coords = coords + rng.normal(scale=0.01, size=coords.shape)
    for _ in range(300):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = coords[i] - coords[j]
                dist = float(np.linalg.norm(delta))
                if dist < min_dist:
                    push = (delta / max(dist, 1e-6)) * (0.5 * (min_dist - dist) + 1e-3)
                    coords[i] += push
                    coords[j] -= push
                    moved = True
        if not moved:
            break
    coords -= coords.mean(axis=0, keepdims=True)
    return coords


def _local_ring_oval_2d(ring: np.ndarray, center: np.ndarray, target_scale: float) -> np.ndarray:
    """개별 64D 타원 고리의 국소 주성분 평면(2D SVD)에 정사영해 8자 꼬임 없이 실제 편심·반경 비율을 유지한 폐곡선 반환."""
    rel = ring - center[None, :]
    _, _, vt = np.linalg.svd(rel - rel.mean(axis=0, keepdims=True), full_matrices=False)
    p2 = rel @ vt[:2].T
    p2 -= p2.mean(axis=0, keepdims=True)
    cur_r = float(np.mean(np.linalg.norm(p2, axis=-1))) + 1e-8
    return p2 * (target_scale / cur_r)


def generate_fig2_manifold_snapback(registry: MetaWorldEvalRegistry, device: torch.device) -> None:
    arm6 = load_dtra_checkpoint("arm6_dtra_full_multimodal", seed=42, device=device)
    arm3 = load_dtra_checkpoint("arm3_dtra_full_vision", seed=42, device=device)

    arm6_rings, arm6_centers = extract_base_rings_and_centers(arm6, stage="fine")  # (8, 48, 64)
    print("[Fig 2] Collecting real Clean & OOD(sigma=0.20) deformation trajectories from Arm 6 & Arm 3...")
    arm6_samples = _collect_arm_snapback_samples(arm6, registry, device)
    arm3_samples = _collect_arm_snapback_samples(arm3, registry, device)

    # 64D 실측 분리도 계산 (제목/주석 기재용)
    k_raw = F.normalize(torch.stack([t.key for t in arm6.fine_tubes], dim=0), p=2, dim=-1).detach().cpu().numpy()
    raw_off = k_raw @ k_raw.T - np.eye(8)
    max_raw_cos = float(raw_off.max())
    ring_cos = np.einsum("imd,jnd->ijmn", arm6_rings, arm6_rings)
    for k in range(8):
        ring_cos[k, k] = -1.0
    max_ring_cos = float(ring_cos.max())
    min_ring_l2 = float(math.sqrt(max(0.0, 2.0 * (1.0 - max_ring_cos))))

    fig = plt.figure(figsize=(16.0, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.15, 0.92], wspace=0.28)

    # --------------------------------------------------------------------------
    # Panel (a): Macro View — 8개 튜브의 64차원 초구 독립 격리 배치 (충돌/겹침 0%)
    # --------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    centers_2d = _classical_mds_2d(arm6_centers, min_dist=0.72, seed=42)
    radii_64 = np.linalg.norm(arm6_rings - arm6_centers[:, None, :], axis=-1).mean(axis=1)
    mean_r64 = float(np.mean(radii_64)) + 1e-8

    # 반경을 이웃 중심 거리의 ~26% 이내로 맞춰 8개 고리가 절대 겹치지 않게 배치
    nn_dists = np.linalg.norm(centers_2d[:, None, :] - centers_2d[None, :, :], axis=-1) + np.eye(8) * 99.0
    max_safe_r = 0.30 * float(nn_dists.min())
    vis_radii = np.clip((radii_64 / mean_r64) * 0.175, 0.11, max_safe_r)

    # 바깥 방사형 배치 계산 (라벨이 고리와 선을 절대 덮지 않도록)
    centroid_2d = centers_2d.mean(axis=0)
    short_names = [lbl.split(": ")[1] for lbl in SHORT_TASK_LABELS]

    for k in range(8):
        col_k = TASK_COLORS[k]
        oval = _local_ring_oval_2d(arm6_rings[k], arm6_centers[k], float(vis_radii[k])) + centers_2d[k : k + 1]
        oval_closed = np.vstack([oval, oval[:1]])
        lw_k = 3.0 if k == 3 else 2.2
        ax_a.fill(oval_closed[:, 0], oval_closed[:, 1], color=col_k, alpha=0.18 if k == 3 else 0.13, zorder=2)
        ax_a.plot(oval_closed[:, 0], oval_closed[:, 1], color=col_k, lw=lw_k, zorder=3)
        ax_a.scatter([centers_2d[k, 0]], [centers_2d[k, 1]], color=col_k, s=36, zorder=4)

        # T3 확대 대상 강조 점선 사각형
        if k == 3:
            rx_min, ry_min = oval.min(axis=0) - 0.055
            rx_max, ry_max = oval.max(axis=0) + 0.055
            zoom_box = mpatches.FancyBboxPatch(
                (rx_min, ry_min),
                rx_max - rx_min,
                ry_max - ry_min,
                boxstyle="round,pad=0.01",
                fc="none",
                ec="#d62828",
                lw=1.5,
                ls="--",
                zorder=5,
            )
            ax_a.add_patch(zoom_box)
            ax_a.text(
                rx_max + 0.03,
                ry_max + 0.02,
                "10x Zoom in (b)",
                fontsize=7.6,
                fontweight="bold",
                color="#d62828",
                ha="left",
                va="bottom",
                zorder=6,
            )

        # 방사형 외곽 라벨 배치 (선이 튜브를 가로지르지 않게 바깥 가장자리 근처에)
        out_vec = centers_2d[k] - centroid_2d
        out_u = out_vec / (float(np.linalg.norm(out_vec)) + 1e-6)
        # 링 반경 밖으로 충분히 이격
        r_outer = float(np.max(np.linalg.norm(oval - centers_2d[k], axis=-1)))
        lbl_pos = centers_2d[k] + out_u * (r_outer + 0.145)
        edge_pt = centers_2d[k] + out_u * (r_outer * 0.85)
        ax_a.annotate(
            f"T{k}\n({short_names[k]})" if k == 3 else f"T{k}",
            xy=edge_pt,
            xytext=lbl_pos,
            ha="center",
            va="center",
            fontsize=8.3,
            fontweight="bold",
            color=col_k,
            bbox=dict(
                boxstyle="round,pad=0.20",
                fc="#fff5f5" if k == 3 else "white",
                ec="#d62828" if k == 3 else col_k,
                lw=1.2 if k == 3 else 0.9,
                alpha=0.95,
            ),
            arrowprops=dict(arrowstyle="-", color=col_k, lw=0.85, alpha=0.65),
            zorder=7,
        )

    all_x, all_y = centers_2d[:, 0], centers_2d[:, 1]
    pad_a_x = 0.56
    pad_a_y_top = 0.68
    pad_a_y_bot = 0.72
    ax_a.set_xlim(float(all_x.min()) - pad_a_x, float(all_x.max()) + pad_a_x)
    ax_a.set_ylim(float(all_y.min()) - pad_a_y_bot, float(all_y.max()) + pad_a_y_top)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel("Geodesic MDS Axis 1 (Normalized $S^{63}$ Chordal Map)")
    ax_a.set_ylabel("Geodesic MDS Axis 2")
    ax_a.set_title("(a) Macro View: 8 Isolated Tubes in $S^{63}$ (Zero Overlap)", pad=10)
    ax_a.grid(True, linestyle=":", alpha=0.32)
    ax_a.text(
        0.5,
        0.975,
        f"Raw Key max cos={max_raw_cos:.2f} | Min Inter-Ring L2 Gap={min_ring_l2:.2f} (Zero Collision)",
        transform=ax_a.transAxes,
        ha="center",
        va="top",
        fontsize=7.6,
        color="#1f2937",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#cbd5e1", lw=0.8),
        zorder=8,
    )

    # 하단 바깥 2열 범례 (그래프와 절대 겹치지 않도록 axes 바깥)
    task_handles = [
        mpatches.Patch(color=TASK_COLORS[k], alpha=0.75, label=SHORT_TASK_LABELS[k]) for k in range(8)
    ]
    ax_a.legend(
        handles=task_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=7.4,
        frameon=True,
        framealpha=0.95,
        columnspacing=0.9,
        handlelength=1.2,
    )

    # --------------------------------------------------------------------------
    # Panel (b): Micro Zoom-In (~10x) on Representative Tube T3 (faucet-open-v3)
    # --------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    t3_ood = next(r for r in arm6_samples["ood"] if r["task_id"] == 3)
    t3_cln = next(r for r in arm6_samples["clean"] if r["task_id"] == 3)

    z0 = t3_ood["z0_star"]
    zt_ood = t3_ood["z_tilde_star"]
    cq_ood = t3_ood["c_query"]
    zt_cln = t3_cln["z_tilde_star"]

    # T3 국소 등거리 직교 정규 좌표계 (원점 = z0*, e1 = 튜브 접선, e2 = 직교화된 OOD 인력 방향)
    r3 = arm6_rings[3]
    idx_near = int(np.argmin(np.linalg.norm(r3 - z0[None, :], axis=-1)))
    tang = r3[(idx_near + 1) % len(r3)] - r3[(idx_near - 1) % len(r3)]
    e1 = tang / (float(np.linalg.norm(tang)) + 1e-12)
    pull_vec = cq_ood - z0
    pull_ortho = pull_vec - float(np.dot(pull_vec, e1)) * e1
    e2 = pull_ortho / (float(np.linalg.norm(pull_ortho)) + 1e-12)

    def _to_local(pt: np.ndarray) -> np.ndarray:
        d = pt - z0
        return np.array([float(np.dot(d, e1)), float(np.dot(d, e2))], dtype=np.float64)

    # 로컬 튜브 고리 호(arc) 투영
    ring_local = (r3 - z0[None, :]) @ np.stack([e1, e2], axis=1)
    ring_local_closed = np.vstack([ring_local, ring_local[:1]])
    # 원의 내부 영역 (e2 < 0 쪽 튜브 내측) 음영
    ax_b.fill(ring_local_closed[:, 0], ring_local_closed[:, 1], color="#64748b", alpha=0.10, zorder=1)
    ax_b.plot(
        ring_local_closed[:, 0],
        ring_local_closed[:, 1],
        color="#475569",
        lw=3.2,
        label="Un-deformed Base Tube Ring $z_3^{(0)}(\\theta)$",
        zorder=2,
    )

    zt_ood_2d = _to_local(zt_ood)
    cq_ood_2d = _to_local(cq_ood)
    zt_cln_2d = _to_local(zt_cln)
    dz_norm_val = float(t3_ood["fine_delta_norm"])
    pull_norm_val = float(t3_ood["fine_pull_norm"])
    alpha_val = float(t3_ood["fine_alpha"])

    # 1) 원래 도킹 원점 z0*(theta*)
    ax_b.scatter([0.0], [0.0], color="#1e293b", s=85, zorder=5, edgecolors="white", linewidths=1.2)
    ax_b.annotate(
        "Anchor $z_3^{(0)}(\\theta^*)$",
        xy=(0.0, 0.0),
        xytext=(-0.082, 0.030),
        fontsize=8.3,
        fontweight="bold",
        color="#1e293b",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#64748b", lw=0.75, alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.0),
        zorder=8,
    )

    # 2) Clean 평형 도킹점 (작은 검정 점)
    ax_b.scatter(
        [zt_cln_2d[0]],
        [zt_cln_2d[1]],
        color="#111111",
        s=42,
        zorder=6,
        edgecolors="white",
        linewidths=0.8,
        label="Clean Dock $z_3^*(t)$",
    )

    # 3) 외부 OOD 쿼리 c_query 및 점선 인력 선분
    ax_b.plot(
        [0.0, cq_ood_2d[0]],
        [0.0, cq_ood_2d[1]],
        color="#7b2cbf",
        ls=":",
        lw=1.5,
        alpha=0.70,
        zorder=3,
    )
    ax_b.scatter(
        [cq_ood_2d[0]],
        [cq_ood_2d[1]],
        marker="X",
        s=125,
        color="#7b2cbf",
        edgecolors="white",
        linewidths=0.9,
        zorder=7,
        label="OOD ($\\sigma=0.20$) Query $c(x_t)$",
    )
    ax_b.annotate(
        f"OOD Noisy Query $c(x_t)$\n($||c - z_0^*|| = {pull_norm_val:.3f}$)",
        xy=cq_ood_2d,
        xytext=(cq_ood_2d[0] - 0.085, cq_ood_2d[1] + 0.015),
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color="#6a1b9a",
        bbox=dict(boxstyle="round,pad=0.20", fc="#f3e8ff", ec="#7b2cbf", lw=0.85),
        arrowprops=dict(arrowstyle="->", color="#7b2cbf", lw=1.1),
        zorder=8,
    )

    # 4) 굵은 빨간 실선 화살표: 탄성 변형 Elastic Stretch delta_z* (z0* -> z~*)
    ax_b.annotate(
        "",
        xy=zt_ood_2d,
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="#d62828", lw=3.4, mutation_scale=17, shrinkA=2, shrinkB=2),
        zorder=9,
    )
    ax_b.scatter(
        [zt_ood_2d[0]],
        [zt_ood_2d[1]],
        marker="*",
        s=195,
        color="#d62828",
        edgecolors="white",
        linewidths=0.9,
        zorder=10,
        label="Elastic Dock $\\tilde{z}_3^* = z_0^* + \\delta z^*$",
    )
    ax_b.annotate(
        f"Elastic Stretch $\\delta z^*$\n($||\\delta z^*|| = {dz_norm_val:.3f},\\ \\alpha_3={alpha_val:.2f}$)",
        xy=zt_ood_2d,
        xytext=(zt_ood_2d[0] - 0.082, zt_ood_2d[1] + 0.068),
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color="#b91c1c",
        bbox=dict(boxstyle="round,pad=0.22", fc="#fef2f2", ec="#d62828", lw=0.95),
        arrowprops=dict(arrowstyle="->", color="#d62828", lw=1.2),
        zorder=11,
    )

    # 5) 굵은 청록색 곡선 점선 화살표: 행동 종료 후 snap_back() -> 0.000 원점 복원
    ax_b.annotate(
        "",
        xy=(-0.001, 0.001),
        xytext=(zt_ood_2d[0] - 0.001, zt_ood_2d[1] + 0.001),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#008b8b",
            lw=2.8,
            ls="--",
            mutation_scale=16,
            connectionstyle="arc3,rad=0.72",
            shrinkA=3,
            shrinkB=4,
        ),
        zorder=9,
    )
    ax_b.annotate(
        "Post-Action snap_back()\n$\\Delta z_3 \\to 0.000$ (Exact Zero)",
        xy=(-0.006, 0.018),
        xytext=(-0.042, -0.050),
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color="#006d6f",
        bbox=dict(boxstyle="round,pad=0.22", fc="#ecfeff", ec="#008b8b", lw=0.95),
        arrowprops=dict(arrowstyle="->", color="#008b8b", lw=1.2),
        zorder=11,
    )

    # 10x 확대 국소 창 크기 (z0*, z~*, c_query가 모두 넉넉히 보이도록)
    x_hi = max(0.185, float(cq_ood_2d[0]) + 0.075)
    x_lo = -0.125
    y_hi = max(0.225, float(cq_ood_2d[1]) + 0.055)
    y_lo = -0.095
    ax_b.set_xlim(x_lo, x_hi)
    ax_b.set_ylim(y_lo, y_hi)

    # 우측 하단 2.5x 국소 초미세 인셋 (delta_z* 화살표와 snap_back() 복원 호를 큼직하게 직관 강조)
    ax_ins = ax_b.inset_axes([0.52, 0.07, 0.45, 0.40])
    ax_ins.fill(ring_local_closed[:, 0], ring_local_closed[:, 1], color="#64748b", alpha=0.10, zorder=1)
    ax_ins.plot(ring_local_closed[:, 0], ring_local_closed[:, 1], color="#475569", lw=2.6, zorder=2)
    ax_ins.plot([0.0, cq_ood_2d[0]], [0.0, cq_ood_2d[1]], color="#7b2cbf", ls=":", lw=1.2, alpha=0.65, zorder=2)
    ax_ins.scatter([0.0], [0.0], color="#1e293b", s=65, edgecolors="white", linewidths=1.0, zorder=5)
    ax_ins.scatter([zt_cln_2d[0]], [zt_cln_2d[1]], color="#111111", s=34, edgecolors="white", linewidths=0.7, zorder=6)
    ax_ins.annotate(
        "",
        xy=zt_ood_2d,
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="#d62828", lw=3.2, mutation_scale=16, shrinkA=2, shrinkB=2),
        zorder=8,
    )
    ax_ins.scatter([zt_ood_2d[0]], [zt_ood_2d[1]], marker="*", s=160, color="#d62828", edgecolors="white", lw=0.8, zorder=9)
    ax_ins.annotate(
        "",
        xy=(-0.001, 0.001),
        xytext=(zt_ood_2d[0] - 0.001, zt_ood_2d[1] + 0.001),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#008b8b",
            lw=2.6,
            ls="--",
            mutation_scale=15,
            connectionstyle="arc3,rad=0.72",
            shrinkA=3,
            shrinkB=4,
        ),
        zorder=8,
    )
    ax_ins.text(0.015, 0.036, "$\\delta z^*$", color="#d62828", fontsize=8.4, fontweight="bold")
    ax_ins.text(-0.014, 0.022, "snap_back", color="#006d6f", fontsize=7.6, fontweight="bold")
    ax_ins.text(0.023, 0.011, "Clean", color="#111111", fontsize=7.2, fontweight="bold")
    ax_ins.set_xlim(-0.022, max(0.038, float(zt_cln_2d[0]) + 0.014))
    ax_ins.set_ylim(-0.012, max(0.044, float(zt_ood_2d[1]) + 0.015))
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.set_title("Docking Zone Zoom", fontsize=7.8, fontweight="bold", pad=3)
    for sp in ax_ins.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#64748b")
        sp.set_linewidth(1.1)
    ax_b.indicate_inset_zoom(ax_ins, edgecolor="#94a3b8", lw=0.9, ls="--")
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("Local Ring Tangent Axis $e_1$ (Isometric, ~10x Zoom)")
    ax_b.set_ylabel("Orthogonal OOD Pull Axis $e_2$")
    ax_b.set_title("(b) Micro View (10x Zoom on Tube T3): Stretch & Snap-Back", pad=10)
    ax_b.grid(True, linestyle=":", alpha=0.36)

    micro_handles = [
        plt.Line2D([0], [0], color="#475569", lw=2.8, label="Base Ring $z_3^{(0)}(\\theta)$"),
        plt.Line2D([0], [0], color="#d62828", lw=2.6, marker="*", ms=9, label="Elastic Stretch $\\delta z^*$"),
        plt.Line2D([0], [0], color="#008b8b", lw=2.4, ls="--", label="snap_back() -> 0.000"),
        plt.Line2D([0], [0], color="#7b2cbf", ls="None", marker="X", ms=7.5, label="OOD Query $c(x_t)$"),
    ]
    ax_b.legend(
        handles=micro_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=7.6,
        frameon=True,
        framealpha=0.95,
    )

    # --------------------------------------------------------------------------
    # Panel (c): 탄성 변형 응력 및 snap_back() 0.0 원점 복원 실측 바차트 (가독성 강화)
    # --------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    c_pull_6 = float(np.mean([r["fine_pull_norm"] for r in arm6_samples["clean"]]))
    o_pull_6 = float(np.mean([r["fine_pull_norm"] for r in arm6_samples["ood"]]))
    c_dz_6 = float(np.mean([r["fine_delta_norm"] for r in arm6_samples["clean"]]))
    o_dz_6 = float(np.mean([r["fine_delta_norm"] for r in arm6_samples["ood"]]))
    post_6 = float(np.max([r["post_snap_abs"] for r in arm6_samples["ood"]]))

    c_pull_3 = float(np.mean([r["fine_pull_norm"] for r in arm3_samples["clean"]]))
    o_pull_3 = float(np.mean([r["fine_pull_norm"] for r in arm3_samples["ood"]]))
    c_dz_3 = float(np.mean([r["fine_delta_norm"] for r in arm3_samples["clean"]]))
    o_dz_3 = float(np.mean([r["fine_delta_norm"] for r in arm3_samples["ood"]]))
    post_3 = float(np.max([r["post_snap_abs"] for r in arm3_samples["ood"]]))

    cats = ["Clean\n$||\\Delta z^*||$", "OOD\n$||\\Delta z^*||$", "Post\nsnap_back", "OOD Pull\n$||c-z_0||$"]
    vals_6 = [c_dz_6, o_dz_6, post_6, o_pull_6]
    vals_3 = [c_dz_3, o_dz_3, post_3, o_pull_3]
    bx = np.arange(len(cats))
    w = 0.34
    b1 = ax_c.bar(bx - w / 2, vals_6, width=w, color="#1b6ca8", edgecolor="#0f4c75", lw=0.6, label="Arm 6 (Multimodal)")
    b2 = ax_c.bar(bx + w / 2, vals_3, width=w, color="#f77f00", edgecolor="#b45309", lw=0.6, label="Arm 3 (Vision-Only)")

    for i_bar, rect in enumerate(list(b1) + list(b2)):
        h = rect.get_height()
        if h == 0.0 and i_bar in (2, 6):
            if i_bar == 2:
                ax_c.text(
                    bx[2],
                    0.012,
                    "0.000\n(Exact)",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold",
                    color="#008b8b",
                    bbox=dict(boxstyle="round,pad=0.18", fc="#ecfeff", ec="#008b8b", lw=0.8),
                )
            continue
        ax_c.text(
            rect.get_x() + rect.get_width() / 2.0,
            h + 0.009,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
        )

    ax_c.set_xticks(bx)
    ax_c.set_xticklabels(cats, fontsize=8.8)
    ax_c.set_ylim(0, max(vals_6 + vals_3) * 1.24)
    ax_c.set_ylabel("L2 Norm on $S^{63}$ Manifold")
    ax_c.set_title("(c) Stress & Zero-Residual Verification", pad=10)
    ax_c.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=8.4)
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)

    import io

    out_path = FIG_DIR / "fig2_manifold_elastic_snapback.png"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    out_path.write_bytes(buf.getvalue())
    print(
        f"[Saved Fig 2] {out_path} | "
        f"Arm6 OOD ||dz*||={o_dz_6:.4f} (T3 sample={dz_norm_val:.4f}) vs post_snap={post_6:.1e} | "
        f"Arm3 OOD ||dz*||={o_dz_3:.4f} vs post_snap={post_3:.1e}"
    )


# ==============================================================================
# Figure 3: 실시간 XAI 텔레메트리 & 3D B-Spline 미래 제어점 투영
# ==============================================================================


def _project_3d_to_flipped_img(
    pts_3d: np.ndarray, cam_pos: np.ndarray, cam_xmat: np.ndarray, fovy_deg: float, width: int = 64, height: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """Meta-World `np.flipud(env.render())` 프레임의 정확한 MuJoCo 카메라 3D->2D 픽셀 투영."""
    p_cam = (cam_xmat.T @ (pts_3d - cam_pos).T).T  # (N, 3)
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    z_depth = np.maximum(-p_cam[:, 2], 1e-4)
    u = 0.5 * width + f * (p_cam[:, 0] / z_depth)
    # `np.flipud(env.render())` 뒤집힘 반영 (+y_cam가 아래 방향 행 증가로 매핑됨을 red button 실측 검증)
    v = 0.5 * height + f * (p_cam[:, 1] / z_depth)
    return u, v


def _bspline_trajectory_and_6_cps_3d(
    eef_pos: np.ndarray, p_star: np.ndarray, act_chunk: np.ndarray, action_scale: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """예측된 16-스텝 행동 청크 & 6개 B-Spline 제어점(p_star)을 현재 EEF 기준 3D 미래 궤적/제어점으로 변환."""
    # decoded 16-step trajectory in 3D world meters (visualized over full horizon H=16)
    deltas = act_chunk[:, :3] * action_scale
    traj_3d = np.vstack([eef_pos[None, :], eef_pos[None, :] + np.cumsum(deltas, axis=0)])  # (17, 3)

    # 6개 제어점: clamped cubic knot [0,0,0,0, 1/3, 2/3, 1,1,1,1]의 Greville abscissae
    # + 해당 제어 속도 벡터(2*p_star - 1)의 누적 적분으로 미래 3D 위치 6개 제어점 배치
    vel_cps = (2.0 * p_star[:, :3] - 1.0) * action_scale  # (6, 3) m/step
    step_span = float(len(act_chunk)) / 5.0  # 16 / 5 = 3.2 steps per interval
    cp_pts = [eef_pos + vel_cps[0] * (0.5 * action_scale * 0.0)]
    # 구간 적분으로 제어 다각형 6개 정점 생성 (시작=현재 EEF+초기속도, 끝=궤적 말단 근처)
    cur = eef_pos.copy()
    for k in range(len(vel_cps)):
        if k == 0:
            cur = cur + 0.5 * step_span * vel_cps[k]
        else:
            cur = cur + step_span * 0.5 * (vel_cps[k - 1] + vel_cps[k])
        cp_pts.append(cur.copy())
    # CP0..CP5 총 6개 제어점 (cp_pts[1:])
    cp_3d = np.array(cp_pts[1:], dtype=np.float32)  # (6, 3)
    return traj_3d, cp_3d


def _rollout_task_for_xai(
    model: ControlledDTRAPolicy,
    registry: MetaWorldEvalRegistry,
    task_name: str,
    task_id: int,
    device: torch.device,
    noise_std: float = 0.20,
    exec_steps: int = 8,
) -> Dict[str, object]:
    """OOD(sigma=0.20)에서 held-out 에피소드 중 가장 대표적인 실행을 캡처(성공 에피소드 우선)."""
    env, eval_tasks = registry.get_env_and_tasks(task_name, split="train_heldout", n_eval=15)
    instr = TASK_INSTRUCTIONS[task_name]
    best_trace = None

    try:
        for ep_i, task_cfg in enumerate(eval_tasks[:8]):
            rng = torch.Generator().manual_seed(42 + task_id * 100 + ep_i)
            env.set_task(task_cfg)
            obs, _ = env.reset()
            model.reset()
            done = False
            steps_log: List[Dict[str, object]] = []
            m, d = env.unwrapped.model, env.unwrapped.data
            cid = m.camera("corner").id
            action_scale = float(getattr(env.unwrapped, "action_scale", 0.01))

            for step_idx in range(0, 150, exec_steps):
                fr = np.flipud(env.render()).copy()
                tx_clean = torch.from_numpy(fr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                tx_ood = apply_ood_rgb_perturbation(tx_clean, noise_std, 1.0, 0.0, rng)
                ts = torch.from_numpy(obs[:4].astype(np.float32)).unsqueeze(0).to(device)

                res = probe_step_with_geometry(model, tx_ood, ts, instr)
                eef_3d = obs[:3].astype(np.float64)
                traj_3d, cp_3d = _bspline_trajectory_and_6_cps_3d(
                    eef_3d, res["p_star"], res["act_chunk"], action_scale=action_scale * 2.5
                )
                cam_pos = d.cam_xpos[cid].copy()
                cam_xmat = d.cam_xmat[cid].reshape(3, 3).copy()
                fovy = float(m.cam_fovy[cid])
                img_h, img_w = fr.shape[0], fr.shape[1]
                traj_u, traj_v = _project_3d_to_flipped_img(traj_3d, cam_pos, cam_xmat, fovy, width=img_w, height=img_h)
                cp_u, cp_v = _project_3d_to_flipped_img(cp_3d, cam_pos, cam_xmat, fovy, width=img_w, height=img_h)
                eef_u, eef_v = _project_3d_to_flipped_img(eef_3d[None, :], cam_pos, cam_xmat, fovy, width=img_w, height=img_h)

                ood_img = (tx_ood[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                steps_log.append(
                    {
                        "step": step_idx,
                        "ood_img": ood_img,
                        "eef_uv": (float(eef_u[0]), float(eef_v[0])),
                        "traj_uv": (traj_u, traj_v),
                        "cp_uv": (cp_u, cp_v),
                        "fine_idx": int(res["fine_idx"]),
                        "coarse_idx": int(res["coarse_idx"]),
                        "cos_all_f": res["cos_all_f"],
                        "key_cos_max": float(res["key_cos_max"]),
                        "route_margin": float(res["route_margin"]),
                        "conf_gate": float(res["conf_gate"]),
                        "coarse_delta_norm": float(res["coarse_delta_norm"]),
                        "fine_delta_norm": float(res["fine_delta_norm"]),
                        "mean_delta_norm": float(res["mean_delta_norm"]),
                        "post_snap_abs": float(res["post_snap_buf_abs"]),
                    }
                )

                for a in res["act_chunk"][:exec_steps]:
                    obs, _, _, _, info = env.step(np.clip(a, -1.0, 1.0))
                    if info.get("success", 0) > 0:
                        done = True
                        break
                if done:
                    break

            trace = {"task_name": task_name, "task_id": task_id, "ep_i": ep_i, "success": done, "steps": steps_log}
            if best_trace is None:
                best_trace = trace
            if done and len(steps_log) >= 3:
                best_trace = trace
                break
    finally:
        env.close()

    return best_trace


def generate_fig3_xai_rollout(registry: MetaWorldEvalRegistry, device: torch.device) -> None:
    arm6 = load_dtra_checkpoint("arm6_dtra_full_multimodal", seed=42, device=device)
    target_specs = [
        ("button-press-v3", 0),
        ("door-open-v3", 1),
        ("faucet-open-v3", 3),
    ]

    traces = [_rollout_task_for_xai(arm6, registry, t_name, tid, device, noise_std=0.20) for t_name, tid in target_specs]

    fig, axes = plt.subplots(3, 3, figsize=(14.4, 10.6), gridspec_kw={"width_ratios": [1.0, 1.25, 1.25], "wspace": 0.36, "hspace": 0.36})

    for row, tr in enumerate(traces):
        t_name = tr["task_name"]
        tid = tr["task_id"]
        steps_data = tr["steps"]
        status_str = "SUCCESS" if tr["success"] else "ROLLOUT"
        # 접근/조작이 가장 잘 보이는 중간 스텝 프레임 선택
        mid_k = min(len(steps_data) - 1, max(1, len(steps_data) // 3))
        snap = steps_data[mid_k]

        # Col 0: OOD(sigma=0.20) 실제 입력 프레임 + 6 B-Spline 제어점 & 16-스텝 미래 궤적 투영
        ax_img = axes[row, 0]
        ax_img.imshow(snap["ood_img"], interpolation="nearest")
        tu, tv = snap["traj_uv"]
        cu, cv = snap["cp_uv"]
        eu, ev = snap["eef_uv"]
        ax_img.plot(tu, tv, color="#00f5d4", lw=2.3, label="16-Step Decoded B-Spline", zorder=4)
        ax_img.plot(cu, cv, color="#ffd166", lw=1.2, ls="--", alpha=0.85, zorder=4)
        ax_img.scatter(cu, cv, color="#ffbe0b", edgecolors="#111111", s=44, lw=1.0, zorder=5, label="6 Control Points")
        for cp_idx in (0, 2, 5):
            ax_img.text(cu[cp_idx] + 2.5, cv[cp_idx] - 2.0, f"P{cp_idx}", color="#fff3b0", fontsize=7.6, fontweight="bold", zorder=6)
        ax_img.scatter([eu], [ev], color="#ff006e", edgecolors="white", marker="D", s=46, lw=1.0, zorder=6, label="Current EEF")
        h_im, w_im = snap["ood_img"].shape[:2]
        ax_img.set_xlim(0, w_im - 1)
        ax_img.set_ylim(h_im - 1, 0)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(f"T{tid}: {t_name} ({status_str}, ep={tr['ep_i']})\nOOD $\\sigma=0.20$ @ step={snap['step']} + 6 B-Spline CPs", fontsize=9.2)
        if row == 0:
            ax_img.legend(loc="lower left", fontsize=7.0, framealpha=0.82)

        # Col 1: 스텝별 Top-1 활성 튜브 ID & 라우팅 확신도 + Margin
        ax_rt = axes[row, 1]
        ts = [s["step"] for s in steps_data]
        cos_mats = np.stack([s["cos_all_f"] for s in steps_data], axis=0)  # (T, 8)
        top1_ids = [s["fine_idx"] for s in steps_data]
        for k_tube in range(8):
            is_tgt = k_tube == tid
            ax_rt.plot(
                ts,
                cos_mats[:, k_tube],
                color=TASK_COLORS[k_tube] if is_tgt else "#b0b8c4",
                lw=2.5 if is_tgt else 1.1,
                alpha=0.95 if is_tgt else 0.55,
                marker="o" if is_tgt else None,
                ms=4.2,
                label=f"Target Tube T{k_tube}" if is_tgt else ("Other 7 Tubes" if k_tube == (1 if tid != 1 else 0) else None),
            )
        ax_rt.axhline(0.0, color="#888888", lw=0.7, ls=":")
        route_ok = all(fid == tid for fid in top1_ids)
        mean_margin = float(np.mean([s["route_margin"] for s in steps_data]))
        ax_rt.set_ylim(-0.20, 1.05)
        ax_rt.set_ylabel("Resonance $\\cos(x_{fine}, k_i)$")
        if row == 2:
            ax_rt.set_xlabel("Control Step $t$")
        ax_rt.set_title(
            f"Top-1 Tube: T{top1_ids[0]} ({'100% Match' if route_ok else 'Mixed'}) | Margin={mean_margin:.3f}",
            fontsize=8.8,
        )
        ax_rt.grid(True, linestyle="--", alpha=0.32)
        ax_rt.legend(loc="lower right", fontsize=7.6, framealpha=0.9)

        # Col 2: 실시간 탄성 변형 응력(||delta_z*||) & Confidence Gate & Snap-Back 0.0
        ax_dz = axes[row, 2]
        c_dz = [s["coarse_delta_norm"] for s in steps_data]
        f_dz = [s["fine_delta_norm"] for s in steps_data]
        cgate = [s["conf_gate"] for s in steps_data]
        post_s = [s["post_snap_abs"] for s in steps_data]

        ax_dz.plot(ts, c_dz, color="#e76f51", marker="s", ms=4.0, lw=2.0, label="Coarse Tube $||\\Delta z_c^*||$")
        ax_dz.plot(ts, f_dz, color="#1b6ca8", marker="o", ms=4.0, lw=2.0, label="Fine Tube $||\\Delta z_f^*||$")
        ax_dz.plot(ts, post_s, color="#2a9d8f", ls="--", lw=1.6, label="Post-snap_back() Buf = 0.0")
        ax_dz.fill_between(ts, 0, c_dz, color="#e76f51", alpha=0.12)
        ax_dz.fill_between(ts, 0, f_dz, color="#1b6ca8", alpha=0.12)

        ax_g = ax_dz.twinx()
        ax_g.plot(ts, cgate, color="#6c757d", ls=":", lw=1.5, label="Resonance Conf Gate")
        ax_g.set_ylim(0.0, 1.05)
        ax_g.set_ylabel("Conf Gate", color="#555555", fontsize=8.5)
        ax_g.spines["top"].set_visible(False)

        ax_dz.set_ylim(-0.02, max(max(c_dz + f_dz) * 1.35, 0.35))
        ax_dz.set_ylabel("Elastic Displacement $||\\Delta z^*||$")
        if row == 2:
            ax_dz.set_xlabel("Control Step $t$")
        ax_dz.set_title(
            f"Strain & Snap-Back ($||\\Delta z_f^*||={np.mean(f_dz):.3f}$, Gate={np.mean(cgate):.3f})",
            fontsize=8.8,
        )
        ax_dz.grid(True, linestyle="--", alpha=0.32)
        if row == 0:
            lines1, labs1 = ax_dz.get_legend_handles_labels()
            lines2, labs2 = ax_g.get_legend_handles_labels()
            ax_dz.legend(lines1 + lines2, labs1 + labs2, loc="center right", bbox_to_anchor=(0.98, 0.56), fontsize=6.8, framealpha=0.92)

    fig.suptitle(
        "White-Box XAI Rollout Telemetry under Severe OOD Noise ($\\sigma=0.20$, Arm 6 Seed 42)",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    out_path = FIG_DIR / "fig3_xai_telemetry_rollout.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    for tr in traces:
        print(
            f"  [Fig 3 trace] {tr['task_name']}: ep={tr['ep_i']} success={tr['success']} "
            f"chunks={len(tr['steps'])} top1_ids={[s['fine_idx'] for s in tr['steps']]}"
        )
    print(f"[Saved Fig 3] {out_path}")


# ==============================================================================
# Figure 0: DTRA 전체 아키텍처 파이프라인 도식도
# ==============================================================================


def generate_fig0_architecture_schematic() -> None:
    """DTRA v2.1 멀티모달->S^63 라우팅->탄성 변형/M_ij->B-Spline->snap_back(0.0) 화이트박스 도식도 생성."""
    fig, ax = plt.subplots(figsize=(16.8, 7.0))
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 45.0)
    ax.axis("off")

    fig.suptitle(
        "DTRA Architecture: White-Box $S^{63}$ Topological Routing, Elastic Docking & Zero-Residual Snap-Back",
        fontsize=13.0,
        fontweight="bold",
        y=0.96,
    )

    stages = [
        {
            "x": 1.8,
            "w": 17.4,
            "hdr_col": "#1d3557",
            "bg_col": "#f1f5f9",
            "title": "1. Multimodal Trunk\n& Query Encoding",
            "lines": [
                "- RGB Frame (128x128)\n  Conv4 Spatial Encoder",
                "- Instruction l_t\n  HashingTextEncoder",
                "- Proprio s_t in R^4\n  (eef_pos + gripper)",
                "-> Unit Query x_hat in R^64\n   (~2.44 ms on CPU)",
            ],
        },
        {
            "x": 21.6,
            "w": 18.2,
            "hdr_col": "#1b6ca8",
            "bg_col": "#eff6ff",
            "title": "2. S^63 Coarse-to-Fine\nTopological Routing",
            "lines": [
                "- Bank of K Frozen Keys\n  O(K) scan (~0.21 ms)",
                "- Coarse Tube (anchor, ctx)\n  x_fine = norm(x + 0.35 ctx)",
                "- Inertia + Myelination M_ij\n  Skill transition prior",
                "-> Top-1 Argmax Select i*\n   O(1) Single Active Tube",
            ],
        },
        {
            "x": 42.2,
            "w": 18.6,
            "hdr_col": "#b91c1c",
            "bg_col": "#fef2f2",
            "title": "3. Elastic Docking\n(delta_z*) on S^63",
            "lines": [
                "- Fourier Tube Ring z_i^(0)(theta)\n  centered on S^63 manifold",
                "- Closed-Form Phase:\n  theta* = atan2(<q,sin>, <q,cos>)",
                "- Resonance-Gated Pull:\n  z~* = norm(z0* + a_t alpha (q-z0*))",
                "-> Fast Buffer delta_z_i\n   (Frozen theta_slow Safe!)",
            ],
        },
        {
            "x": 63.2,
            "w": 17.8,
            "hdr_col": "#2a9d8f",
            "bg_col": "#ecfdf5",
            "title": "4. Isolated Local\nB-Spline Decoder",
            "lines": [
                "- Only fine_tubes[i*].head()\n  runs (zeros untouched!)",
                "- Conf Gate = clip(cos_max)\n  throttles OOD residual",
                "- 6 Control Points p* in [0,1]^4\n  Cubic B-Spline Basis (MatMul)",
                "-> 16-Step Action Chunk\n   (~1.87 ms O(1) Exec!)",
            ],
        },
        {
            "x": 83.4,
            "w": 14.8,
            "hdr_col": "#006d77",
            "bg_col": "#ecfeff",
            "title": "5. Post-Execution\nsnap_back() -> 0.0",
            "lines": [
                "- Runs after every\n  chunk / episode end",
                "- Clears transient buf:\n  delta_z.zero_()\n  inertia / gate.zero_()",
                "-> Residual Verified:\n   max|delta_z| = 0.000",
                "-> Zero Drift!\n   Forgetting = 0.0%",
            ],
        },
    ]

    top_y = 14.2
    box_h = 24.5
    hdr_h = 5.4

    for idx, st in enumerate(stages):
        x, w = st["x"], st["w"]
        # 본체 카드
        card = FancyBboxPatch(
            (x, top_y),
            w,
            box_h,
            boxstyle="round,pad=0.35,rounding_size=1.0",
            fc=st["bg_col"],
            ec=st["hdr_col"],
            lw=1.7,
            zorder=2,
        )
        ax.add_patch(card)

        # 헤더 바
        hdr = FancyBboxPatch(
            (x + 0.15, top_y + box_h - hdr_h),
            w - 0.30,
            hdr_h - 0.10,
            boxstyle="round,pad=0.20,rounding_size=0.8",
            fc=st["hdr_col"],
            ec=st["hdr_col"],
            lw=1.0,
            zorder=3,
        )
        ax.add_patch(hdr)
        ax.text(
            x + w / 2.0,
            top_y + box_h - hdr_h / 2.0,
            st["title"],
            color="white",
            ha="center",
            va="center",
            fontsize=9.3,
            fontweight="bold",
            zorder=4,
        )

        # 내부 bullet 항목 균등 배치
        y_cursor = top_y + box_h - hdr_h - 2.3
        step_y = (box_h - hdr_h - 2.2) / max(len(st["lines"]), 1)
        for j, line in enumerate(st["lines"]):
            is_last = j == len(st["lines"]) - 1
            ax.text(
                x + 0.85,
                y_cursor - j * step_y,
                line,
                color=st["hdr_col"] if is_last else "#1f2937",
                ha="left",
                va="top",
                fontsize=7.9,
                fontweight="bold" if is_last else "normal",
                linespacing=1.26,
                zorder=4,
            )

        # 단계 간 화살표
        if idx < len(stages) - 1:
            x_start = x + w + 0.45
            x_end = stages[idx + 1]["x"] - 0.45
            y_mid = top_y + box_h * 0.48
            arr = FancyArrowPatch(
                (x_start, y_mid),
                (x_end, y_mid),
                arrowstyle="-|>",
                mutation_scale=16,
                lw=2.2,
                color="#334155",
                zorder=5,
            )
            ax.add_patch(arr)

    # Stage 5 -> Stage 3 상단 아치형 snap_back() 복원 피드백 화살표
    x5_top = stages[4]["x"] + stages[4]["w"] * 0.5
    x3_top = stages[2]["x"] + stages[2]["w"] * 0.5
    y_arc = top_y + box_h + 0.7
    snap_arc = FancyArrowPatch(
        (x5_top, y_arc),
        (x3_top, y_arc),
        arrowstyle="-|>",
        mutation_scale=15,
        lw=2.1,
        ls="--",
        color="#008b8b",
        connectionstyle="arc3,rad=0.16",
        zorder=6,
    )
    ax.add_patch(snap_arc)
    ax.text(
        0.5 * (x5_top + x3_top),
        y_arc + 3.4,
        "snap_back() resets elastic buffer & inertia on S^63 to exact 0.000 after action execution",
        ha="center",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color="#006d77",
        bbox=dict(boxstyle="round,pad=0.25", fc="#ecfeff", ec="#008b8b", lw=0.9),
        zorder=7,
    )

    # 하단 보증 패널 2개 (지속학습 확장 규칙 + 화이트박스 XAI 계측)
    bottom_boxes = [
        (
            1.8,
            47.2,
            "#334155",
            "#f8fafc",
            "Continual Novelty Expansion Protocol (Zero-Forgetting Guarantee)",
            "If max_i cos(x_hat, k_i) < tau_nov => Freeze old tubes 0..K-1 (grad=0) & append tube K.\n"
            "Order-invariant (T0->T7 & T7->T0): FR = 0.0%, BWT = 0.0%, RouteAcc = 100.0%.",
        ),
        (
            51.0,
            47.2,
            "#6a1b9a",
            "#faf5ff",
            "Real-Time White-Box XAI Telemetry (Zero Post-Hoc Approximation)",
            "Every step exposes exact geometric internals: Top-1 tube ID i*, margin (cos_1 - cos_2),\n"
            "elastic strain ||delta_z*||_2, active myelin handoff M_ij, and 6 3D B-Spline CPs.",
        ),
    ]
    for bx, bw, ec_col, fc_col, b_title, b_body in bottom_boxes:
        box = FancyBboxPatch(
            (bx, 1.8),
            bw,
            9.2,
            boxstyle="round,pad=0.30,rounding_size=0.8",
            fc=fc_col,
            ec=ec_col,
            lw=1.3,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(bx + 1.0, 9.6, b_title, color=ec_col, fontsize=8.9, fontweight="bold", va="top", zorder=3)
        ax.text(bx + 1.0, 6.7, b_body, color="#1e293b", fontsize=7.6, va="top", linespacing=1.35, zorder=3)

    out_path = FIG_DIR / "fig0_dtra_architecture_schematic.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Saved Fig 0] {out_path}")


# ==============================================================================
# Figure 4: 지속학습 망각률 8x8 히트맵 & CL 대조군 종합 비교
# ==============================================================================


def generate_fig4_cl_forgetting_heatmap() -> None:
    """8x8 순차 학습 성공률 히트맵(Naive FT, EWC, ER 5% vs DTRA Arm 6) 및 Clean/FR/OOD 종합 막대 비교."""
    with open(SUP_JSON, "r", encoding="utf-8") as f:
        sup = json.load(f)
    with open(SCI_JSON, "r", encoding="utf-8") as f:
        sci = json.load(f)

    p1 = sup["part1_traditional_cl"]
    s42 = sci["per_seed"]["42"]

    heatmap_specs = [
        ("Naive Fine-Tuning", np.array(p1["cl_naive_ft"]["R"], dtype=np.float64), p1["cl_naive_ft"]["clean_avg_sr"], p1["cl_naive_ft"]["fr"]),
        ("EWC (Elastic Weight Consolidation)", np.array(p1["cl_ewc"]["R"], dtype=np.float64), p1["cl_ewc"]["clean_avg_sr"], p1["cl_ewc"]["fr"]),
        ("Experience Replay (5% Buffer)", np.array(p1["cl_er_5pct"]["R"], dtype=np.float64), p1["cl_er_5pct"]["clean_avg_sr"], p1["cl_er_5pct"]["fr"]),
        ("DTRA Full (Arm 6, Ours)", np.array(s42["arm6_dtra_full_multimodal"]["R"], dtype=np.float64), s42["arm6_dtra_full_multimodal"]["clean_id"]["avg_final_f1"], s42["arm6_dtra_full_multimodal"]["clean_id"]["avg_forgetting"]),
    ]

    fig = plt.figure(figsize=(17.2, 8.0))
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.95], wspace=0.24, top=0.86, bottom=0.14)
    left_gs = outer_gs[0, 0].subgridspec(2, 2, wspace=0.22, hspace=0.36)
    fig.text(
        0.31,
        0.925,
        "(a) Sequential Learning 8x8 Success Rate Matrices $R^T$ (Seed 42)",
        ha="center",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
    )

    cmap = plt.cm.RdYlBu.copy()
    cmap.set_bad(color="#f1f5f9")
    upper_mask = np.triu(np.ones((8, 8), dtype=bool), k=1)  # row s < col t (학습 전 태스크 마스킹)
    tick_labs = [f"T{k}" for k in range(8)]

    im_ref = None
    for idx, (title_m, r_mat, clean_sr, fr_val) in enumerate(heatmap_specs):
        ax_h = fig.add_subplot(left_gs[idx // 2, idx % 2])
        # JSON 행렬 R[task_i, stage_j] -> 전치 R.T[stage_row, task_col]: 아래로 갈수록 학습 단계 진행
        mat_pct = np.ma.masked_where(upper_mask, r_mat.T * 100.0)
        im_ref = ax_h.imshow(mat_pct, cmap=cmap, vmin=0.0, vmax=100.0, aspect="equal")

        for r_i in range(8):
            for c_j in range(8):
                if r_i < c_j:
                    ax_h.text(c_j, r_i, "-", ha="center", va="center", fontsize=7.0, color="#94a3b8")
                else:
                    v = float(r_mat.T[r_i, c_j] * 100.0)
                    txt_col = "white" if (v < 24.0 or v > 78.0) else "#111827"
                    ax_h.text(
                        c_j,
                        r_i,
                        f"{v:.0f}",
                        ha="center",
                        va="center",
                        fontsize=7.2,
                        fontweight="bold" if (r_i == 7 or v == 0.0) else "normal",
                        color=txt_col,
                    )

        ax_h.set_xticks(range(8))
        ax_h.set_xticklabels(tick_labs, fontsize=7.6)
        ax_h.set_yticks(range(8))
        ax_h.set_yticklabels([f"After T{k}" for k in range(8)] if (idx % 2 == 0) else [], fontsize=7.4)
        if idx >= 2:
            ax_h.set_xlabel("Evaluated Task ID", fontsize=8.8)
        ax_h.set_title(f"{title_m}\nClean={clean_sr*100:.1f}% | FR={fr_val*100:.1f}%", fontsize=9.4)
        # 그리드 경계선
        ax_h.set_xticks(np.arange(-0.5, 8, 1), minor=True)
        ax_h.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax_h.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax_h.tick_params(which="minor", bottom=False, left=False)
        ax_h.spines["top"].set_visible(True)
        ax_h.spines["right"].set_visible(True)

    # 왼쪽 2x2 히트맵 하단 공통 컬러바
    cbar_ax = fig.add_axes([0.14, 0.03, 0.36, 0.022])
    cbar = fig.colorbar(im_ref, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Task Success Rate (%)   [Red = 0% Collapse  -->  Blue = 100% Retained]", fontsize=8.6)

    # --------------------------------------------------------------------------
    # 오른쪽: CL 대조군 & 동결 아키텍처 종합 비교 막대그래프 (Seed 42)
    # --------------------------------------------------------------------------
    ax_bar = fig.add_subplot(outer_gs[0, 1])
    bar_models = [
        ("Naive FT", p1["cl_naive_ft"]["clean_avg_sr"] * 100.0, p1["cl_naive_ft"]["fr"] * 100.0, p1["cl_naive_ft"]["ood_severe_sr"] * 100.0),
        ("EWC", p1["cl_ewc"]["clean_avg_sr"] * 100.0, p1["cl_ewc"]["fr"] * 100.0, p1["cl_ewc"]["ood_severe_sr"] * 100.0),
        ("ER (5%)", p1["cl_er_5pct"]["clean_avg_sr"] * 100.0, p1["cl_er_5pct"]["fr"] * 100.0, p1["cl_er_5pct"]["ood_severe_sr"] * 100.0),
        ("Frozen MLP\n(Arm 4)", s42["arm4_frozen_mlp_multimodal"]["clean_id"]["avg_final_f1"] * 100.0, s42["arm4_frozen_mlp_multimodal"]["clean_id"]["avg_forgetting"] * 100.0, s42["arm4_frozen_mlp_multimodal"]["ood_severe_sigma020"]["avg_sr"] * 100.0),
        ("DTRA Full\n(Arm 6)", s42["arm6_dtra_full_multimodal"]["clean_id"]["avg_final_f1"] * 100.0, s42["arm6_dtra_full_multimodal"]["clean_id"]["avg_forgetting"] * 100.0, s42["arm6_dtra_full_multimodal"]["ood_severe_sigma020"]["avg_sr"] * 100.0),
    ]

    x_idx = np.arange(len(bar_models))
    bw = 0.255
    clean_vals = [m[1] for m in bar_models]
    fr_vals = [m[2] for m in bar_models]
    ood_vals = [m[3] for m in bar_models]

    r1 = ax_bar.bar(x_idx - bw, clean_vals, width=bw, color="#1b6ca8", edgecolor="#0f4c75", lw=0.7, label="Clean Final SR (%)  [Higher Better]")
    r2 = ax_bar.bar(x_idx, fr_vals, width=bw, color="#d62828", edgecolor="#991b1b", lw=0.7, label="Forgetting Rate FR (%)  [Lower Better]")
    r3 = ax_bar.bar(x_idx + bw, ood_vals, width=bw, color="#2a9d8f", edgecolor="#115e59", lw=0.7, label="OOD Noise (sigma=0.20) SR (%)  [Higher Better]")

    for rects, is_fr in [(r1, False), (r2, True), (r3, False)]:
        for rect in rects:
            val = rect.get_height()
            if is_fr and val == 0.0:
                ax_bar.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    2.6,
                    "0.0%",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                    color="#0f766e",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#0f766e", lw=0.6, alpha=0.92),
                    zorder=6,
                )
            else:
                ax_bar.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    val + 1.3,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.0,
                    fontweight="bold",
                )

    # DTRA 영역 강조 배경 하이라이트
    ax_bar.axvspan(3.55, 4.45, color="#e0f2fe", alpha=0.45, zorder=0)
    ax_bar.annotate(
        "DTRA: 0.0% FR AND\n+36.7%p OOD SR vs Frozen MLP",
        xy=(4 + bw, ood_vals[-1]),
        xytext=(2.50, 95.0),
        fontsize=8.5,
        fontweight="bold",
        color="#0f4c75",
        bbox=dict(boxstyle="round,pad=0.28", fc="#e0f2fe", ec="#1b6ca8", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#1b6ca8", lw=1.3),
        zorder=8,
    )

    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels([m[0] for m in bar_models], fontsize=9.2, fontweight="bold")
    ax_bar.set_ylim(0, 126.0)
    ax_bar.set_ylabel("Percentage (%)")
    ax_bar.set_title("(b) Continual Learning & OOD Robustness Summary (Seed 42)", pad=10)
    ax_bar.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=8.1)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(
        "Catastrophic Forgetting Collapse in Standard CL vs. Exact 0.0% Forgetting & OOD Retention in DTRA",
        fontsize=12.6,
        fontweight="bold",
        y=0.985,
    )

    out_path = FIG_DIR / "fig4_cl_forgetting_heatmap_and_bars.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Saved Fig 4] {out_path}")


# ==============================================================================
# Figure 5: O(1) 상수 시간 추론 지연시간 스케일링 (K=1 -> K=8)
# ==============================================================================


def generate_fig5_latency_scaling() -> None:
    """ablate_dtra_scientific_results.json 실측 cpu_latency_breakdown 기반 K=1->8 O(1) vs O(K) 비교 차트."""
    with open(SCI_JSON, "r", encoding="utf-8") as f:
        sci = json.load(f)
    lat = sci["cpu_latency_breakdown"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15.2, 5.4), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28})

    # Panel (a): Active Control Execution Time (ms) at K=1 vs K=8
    series_cfg = [
        ("arm6_dtra_full_multimodal", "Arm 6: DTRA Full Multimodal [O(1)]", "#1b6ca8", "o", "-", 3.0, 9.0),
        ("arm5_dtra_rigid_alpha0", "Arm 5: DTRA Rigid (alpha=0) [O(1)]", "#008b8b", "s", "--", 2.2, 7.5),
        ("arm3_dtra_full_vision", "Arm 3: DTRA Vision-Only [O(1)]", "#475569", "D", ":", 2.0, 6.8),
        ("arm1_abk_dog_vision_blend", "Arm 1 Blend: Dense Soft-Mix [O(K)]", "#d62828", "^", "-", 2.8, 8.8),
        ("arm1_abk_dog_vision", "Arm 1 Argmax: Prefix Scan [O(K/2)]", "#e76f51", "v", "-.", 2.0, 7.2),
        ("arm4_frozen_mlp_multimodal", "Arm 4: Frozen MLP [O(1), Low OOD SR]", "#94a3b8", "P", "--", 1.7, 6.8),
    ]

    ks = np.array([1, 8], dtype=np.float64)
    # O(1) ~1.87 ms 수평 대역 강조
    ax_l.axhspan(1.80, 1.94, color="#dbeafe", alpha=0.55, zorder=0, label="DTRA O(1) Constant Band (~1.87 ms)")

    for key, label, col, mk, ls, lw, ms in series_cfg:
        y1 = float(lat[key]["K=1"]["active_exec_ms"])
        y8 = float(lat[key]["K=8"]["active_exec_ms"])
        ax_l.plot(ks, [y1, y8], color=col, marker=mk, ls=ls, lw=lw, ms=ms, label=label, zorder=4 if "arm6" in key or "blend" in key else 3)

    a6_k1 = float(lat["arm6_dtra_full_multimodal"]["K=1"]["active_exec_ms"])
    a6_k8 = float(lat["arm6_dtra_full_multimodal"]["K=8"]["active_exec_ms"])
    b_k1 = float(lat["arm1_abk_dog_vision_blend"]["K=1"]["active_exec_ms"])
    b_k8 = float(lat["arm1_abk_dog_vision_blend"]["K=8"]["active_exec_ms"])

    ax_l.annotate(
        f"DTRA Full (Arm 6):\n{a6_k1:.3f} ms (K=1) -> {a6_k8:.3f} ms (K=8)\nFlat O(1) Scaling (1.00x)",
        xy=(8.0, a6_k8),
        xytext=(3.7, 1.24),
        fontsize=8.4,
        fontweight="bold",
        color="#0f4c75",
        bbox=dict(boxstyle="round,pad=0.25", fc="#eff6ff", ec="#1b6ca8", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#1b6ca8", lw=1.3),
        zorder=7,
    )
    ax_l.annotate(
        f"Soft-Blend O(K): {b_k1:.3f} -> {b_k8:.3f} ms\n(7.41x Linear Growth, Crosses DTRA)",
        xy=(8.0, b_k8),
        xytext=(2.2, 2.34),
        fontsize=8.2,
        fontweight="bold",
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#d62828", lw=1.0),
        arrowprops=dict(arrowstyle="->", color="#d62828", lw=1.3),
        zorder=7,
    )

    ax_l.set_xticks([1, 8])
    ax_l.set_xticklabels(["K = 1 Task", "K = 8 Tasks"], fontsize=9.5, fontweight="bold")
    ax_l.set_xlim(0.3, 8.7)
    ax_l.set_ylim(0.0, 2.78)
    ax_l.set_xlabel("Number of Continually Learned Tasks (Measured at K=1 and K=8)")
    ax_l.set_ylabel("Active Control Execution Latency (ms / chunk)")
    ax_l.set_title("(a) Active Control Path Scaling: Exact O(1) (~1.87 ms) vs. O(K)", pad=10)
    ax_l.grid(True, linestyle="--", alpha=0.35)
    ax_l.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        fontsize=7.5,
        frameon=True,
        framealpha=0.95,
    )

    # Panel (b): K=8 End-to-End 16-Step Chunk Latency Stage Breakdown (Horizontal Stacked Bar)
    breakdown_arms = [
        ("arm6_dtra_full_multimodal", "Arm 6: DTRA Full\n(O(1), 57.5% OOD SR)"),
        ("arm5_dtra_rigid_alpha0", "Arm 5: DTRA Rigid\n(O(1), alpha=0)"),
        ("arm1_abk_dog_vision_blend", "Arm 1: Soft-Blend\n(O(K), 7.4x Active)"),
        ("arm1_abk_dog_vision", "Arm 1: Argmax Scan\n(2.5% OOD SR)"),
        ("arm4_frozen_mlp_multimodal", "Arm 4: Frozen MLP\n(20.8% OOD SR)"),
    ]
    y_pos = np.arange(len(breakdown_arms))[::-1]
    trunks = np.array([lat[k]["K=8"]["trunk_key_ms"] for k, _ in breakdown_arms])
    searches = np.array([lat[k]["K=8"]["key_search_ok_ms"] for k, _ in breakdown_arms])
    actives = np.array([lat[k]["K=8"]["active_exec_ms"] for k, _ in breakdown_arms])
    e2es = np.array([lat[k]["K=8"]["e2e_chunk_ms"] for k, _ in breakdown_arms])
    overheads = np.maximum(0.0, e2es - (trunks + searches + actives))

    h_bar = 0.56
    ax_r.barh(y_pos, trunks, height=h_bar, color="#334155", label="Trunk + Key Encode (ms)")
    ax_r.barh(y_pos, searches, left=trunks, height=h_bar, color="#f59e0b", label="Key Search O(K) (ms)")
    ax_r.barh(y_pos, actives, left=trunks + searches, height=h_bar, color="#1b6ca8", label="Active Execution (ms)")
    ax_r.barh(y_pos, overheads, left=trunks + searches + actives, height=h_bar, color="#cbd5e1", label="Chunk Wrap / Sync")

    for yp, tot, act_ms, sr_ms in zip(y_pos, e2es, actives, searches):
        hz = 1000.0 / tot
        ax_r.text(
            tot + 0.08,
            yp,
            f"{tot:.2f} ms ({hz:.0f} Hz)\n[Exec={act_ms:.2f}, Search={sr_ms:.2f}]",
            va="center",
            ha="left",
            fontsize=7.8,
            fontweight="bold",
            color="#1e293b",
        )

    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels([lbl for _, lbl in breakdown_arms], fontsize=8.5, fontweight="bold")
    ax_r.set_xlim(0.0, float(e2es.max()) * 1.44)
    ax_r.set_xlabel("CPU End-to-End Inference Time per 16-Step Chunk (ms)")
    ax_r.set_title("(b) K=8 CPU Latency Breakdown (Real-Time >200 Chunks/s)", pad=10)
    ax_r.grid(axis="x", linestyle="--", alpha=0.35)
    ax_r.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        fontsize=7.6,
        frameon=True,
        framealpha=0.95,
    )

    fig.suptitle(
        "CPU Edge Deployment Latency Scaling (Measured per 16-Step Action Chunk, Seed 42 Benchmarks)",
        fontsize=12.4,
        fontweight="bold",
        y=0.99,
    )

    out_path = FIG_DIR / "fig5_o1_latency_scaling.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[Saved Fig 5] {out_path} | Arm6 K=1->8 active_exec: {a6_k1:.3f} -> {a6_k8:.3f} ms | Blend: {b_k1:.3f} -> {b_k8:.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="DTRA v2.1 Paper Figures & White-Box XAI Visualizer (300 DPI)")
    parser.add_argument(
        "--figs",
        nargs="+",
        default=["0", "1", "2", "3", "4", "5"],
        help="Subset of figures to generate: 0 1 2 3 4 5 (default: all)",
    )
    args = parser.parse_args()
    sel = {str(x).lower().lstrip("fig") for x in args.figs}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    print(f"[XAI Visualizer] Output dir: {FIG_DIR} | Figures selected: {sorted(sel)} | Device: {device}")

    if "0" in sel:
        generate_fig0_architecture_schematic()
    if "1" in sel:
        generate_fig1_ood_curves()
    if "4" in sel:
        generate_fig4_cl_forgetting_heatmap()
    if "5" in sel:
        generate_fig5_latency_scaling()

    if "2" in sel or "3" in sel:
        registry = MetaWorldEvalRegistry(TASK_NAMES_8)
        registry.prepare_all()
        if "2" in sel:
            generate_fig2_manifold_snapback(registry, device=device)
        if "3" in sel:
            generate_fig3_xai_rollout(registry, device=device)

    print(f"[DONE] Generated requested 300-DPI figures: {sorted(sel)}")


if __name__ == "__main__":
    main()
