"""scripts/generate_libero100_figures.py

DTRA v3.0 논문(dtra_v30_preprint.tex)용 고해상도 300 DPI 시각화 생성기.
1) fig6_dtra_v3_architecture_schematic.png:
   - DTRA v3.0 아키텍처 (No-Encroachment + Top-M CMC + SharedBasisBank + Two-Timescale Snap-Back)
   - Cell -> Region (K=100 Implemented) -> Organ (Future Work) 개념 경계 명시
2) fig7_libero100_scaling_curves.png:
   - outputs/libero/benchmark_dtra_libero_10_libero_90_k100*.json 실측 데이터를 파싱하여
     K=10..100 마일스톤 Mean SR, Dup-Lang RouteAcc, SharedBasisBank 포화 곡선, K=100 비교 바차트 생성
     (다중 시드 JSON이 존재하면 자동으로 Mean ± Std 오차막대 렌더링)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
LIBERO_DIR = PROJECT_ROOT / "outputs" / "libero"

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


def _draw_box(
    ax,
    xy: Tuple[float, float],
    w: float,
    h: float,
    title: str,
    body: str,
    fc: str,
    ec: str,
    title_color: str = "#111827",
    fontsize_title: float = 9.0,
    fontsize_body: float = 7.8,
    lw: float = 1.5,
    linestyle: str = "-",
) -> None:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=linestyle,
    )
    ax.add_patch(box)
    cx = xy[0] + w / 2.0
    if body:
        ax.text(
            cx,
            xy[1] + h * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=fontsize_title,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            cx,
            xy[1] + h * 0.31,
            body,
            ha="center",
            va="center",
            fontsize=fontsize_body,
            color="#374151",
            linespacing=1.25,
        )
    else:
        ax.text(
            cx,
            xy[1] + h * 0.50,
            title,
            ha="center",
            va="center",
            fontsize=fontsize_title,
            fontweight="bold",
            color=title_color,
        )


def _draw_arrow(ax, p1: Tuple[float, float], p2: Tuple[float, float], color: str = "#374151", rad: float = 0.0) -> None:
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>,head_length=5.5,head_width=3.5",
        connectionstyle=f"arc3,rad={rad}",
        color=color,
        linewidth=1.5,
    )
    ax.add_patch(arr)


def generate_fig6_architecture_schematic(out_path: Path) -> None:
    fig = plt.figure(figsize=(14.2, 6.4))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.96])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.50,
        0.965,
        "Dynamic Topology Ring Architecture: Hyperspherical Topological Isolation & Shared Basis Compression",
        ha="center",
        va="center",
        fontsize=11.4,
        fontweight="bold",
        color="#0F172A",
    )

    # Row 1: Implemented Single-Ring Region Pipeline (K=100)
    _draw_box(
        ax,
        (0.015, 0.56),
        0.175,
        0.33,
        "Multimodal Inputs\n(Strict Parity)",
        "Dual-Cam RGB (128×128)\nFrozen ResNet-18 (L1–L3)\n+ 12-d Proprioception\n+ LightweightTextEncoder\n  (QR Hash 128-d, No VLM)",
        fc="#F1F5F9",
        ec="#475569",
        fontsize_title=9.5,
        fontsize_body=8.2,
    )

    _draw_box(
        ax,
        (0.215, 0.56),
        0.21,
        0.33,
        "2-Stage Routing on S⁶³\n(Task-ID-Free)",
        "Stage 1: O(K) Scan → Top-M\nStage 2: Top-M Common-Mode\n  Cancellation (Center/Whiten)\nNo-Encroachment Constraint:\n  cos(μ_j, k_i) ≤ s_jj − m_safe",
        fc="#EFF6FF",
        ec="#2563EB",
        title_color="#1E3A8A",
        fontsize_title=9.5,
        fontsize_body=8.2,
    )

    _draw_box(
        ax,
        (0.45, 0.56),
        0.22,
        0.33,
        "Active Hypertube & Basis\n(O(|A_t|) ≈ O(1))",
        "Elliptical Ring Basis B_ring(θ)\nElastic Bend Δz* (α_i ∈ (0, 1))\nSharedBasisBank (K=100 → 27\n  frozen knot bases, ∇θ = 0)\n+ Micro-Residual & Conf Gate",
        fc="#ECFDF5",
        ec="#059669",
        title_color="#065F46",
        fontsize_title=9.5,
        fontsize_body=8.2,
    )

    _draw_box(
        ax,
        (0.695, 0.56),
        0.29,
        0.33,
        "C² B-Spline & Two-Timescale\nSnap-Back (Zero Drift)",
        "Clamped Cubic B-Spline (N_k=6 → H=16)\n7-DoF Continuous Action Chunk\n1) chunk_snap_back(): Δz → 0.0000\n   (keeps episodic_trace & Conf Gate)\n2) episode_snap_back(): all buffers → 0.0",
        fc="#FFF7ED",
        ec="#EA580C",
        title_color="#9A3412",
        fontsize_title=9.5,
        fontsize_body=8.1,
    )

    _draw_arrow(ax, (0.19, 0.725), (0.215, 0.725), "#334155")
    _draw_arrow(ax, (0.425, 0.725), (0.45, 0.725), "#2563EB")
    _draw_arrow(ax, (0.67, 0.725), (0.695, 0.725), "#059669")

    arr_snap = FancyArrowPatch(
        (0.84, 0.546),
        (0.56, 0.546),
        arrowstyle="-|>,head_length=5,head_width=3",
        connectionstyle="arc3,rad=-0.15",
        color="#DC2626",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(arr_snap)
    ax.text(
        0.70,
        0.464,
        "Exact Topological Freeze (L∞ Drift = 0.0000, AdamW Silent Drift = 0.0)",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color="#B91C1C",
    )

    # Bottom section: Scope Boundary & Cell -> Region -> Organ Hierarchy
    scope_box = FancyBboxPatch(
        (0.015, 0.03),
        0.97,
        0.405,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#FAFAFA",
        edgecolor="#94A3B8",
        linewidth=1.2,
    )
    ax.add_patch(scope_box)
    ax.text(
        0.50,
        0.40,
        "Implementation Scope Boundary & Conceptual Scaling Hierarchy (Cell → Region → Organ)",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color="#1E293B",
    )

    _draw_box(
        ax,
        (0.04, 0.065),
        0.275,
        0.29,
        "[IMPLEMENTED] Cell Level\nAtomic Motor Memory Unit",
        "Single S⁶³ Hypertube Ring\nC² Cubic B-Spline Control Knots\nNo-Encroachment Boundary\nZero-Gradient Freeze (∇θ = 0)",
        fc="#DCFCE7",
        ec="#16A34A",
        title_color="#14532D",
        fontsize_title=9.5,
        fontsize_body=8.3,
    )
    _draw_box(
        ax,
        (0.362, 0.065),
        0.275,
        0.29,
        "[IMPLEMENTED] Region Level\nDTRA v3.0 Single-Ring (K=100)",
        "Single Cortical Column Module\nK=100 LIBERO Tasks Validated\n27 Shared Bases (73% Compression)\nO(1) Active Kernel & White-Box Stack",
        fc="#DBEAFE",
        ec="#2563EB",
        title_color="#1E3A8A",
        fontsize_title=9.5,
        fontsize_body=8.3,
    )
    _draw_box(
        ax,
        (0.684, 0.065),
        0.275,
        0.29,
        "[FUTURE ROADMAP] Organ Level\nConceptual Multi-Ring (10⁴–10⁸)",
        "Not Implemented in DTRA v3.0\nMulti-Region Hierarchical Routing\nIsometric Zero-Padding [v; 0]\nSub-linear O(log K) Lookup Vision",
        fc="#F3F4F6",
        ec="#6B7280",
        title_color="#374151",
        fontsize_title=9.5,
        fontsize_body=8.3,
        linestyle="--",
    )

    _draw_arrow(ax, (0.315, 0.21), (0.362, 0.21), "#16A34A")
    _draw_arrow(ax, (0.637, 0.21), (0.684, 0.21), "#6B7280")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _extract_true_bases(arm_obj: dict, k_int: int, fallback: int) -> int:
    tl = arm_obj.get("train_logs", [])
    if isinstance(tl, list) and len(tl) >= k_int:
        val = tl[k_int - 1].get("active_basis_count")
        if val is not None:
            return int(val)
    return int(fallback)


def load_libero_seed_jsons(json_paths: List[Path]) -> List[dict]:
    loaded = []
    for p in json_paths:
        locked = p.with_name(p.stem + "_locked.json")
        # Support mapping base k100.json -> seed42_locked.json cleanly
        if p.name == "benchmark_dtra_libero_10_libero_90_k100.json":
            s42_locked = p.with_name("benchmark_dtra_libero_10_libero_90_k100_seed42_locked.json")
            target = s42_locked if s42_locked.exists() else p
        else:
            target = locked if locked.exists() else p
        if target.exists():
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not target.name.endswith("_locked.json") and isinstance(data.get("arms"), dict):
                data["arms"] = {
                    k: v for k, v in data["arms"].items()
                    if v.get("status") == "completed" or "100" in v.get("milestones", {})
                }
            loaded.append(data)
    return loaded


def aggregate_milestone_series(
    datasets: List[dict], arm: str, milestones: List[int], key: str, scale: float = 100.0
) -> Tuple[List[int], List[float], List[float]]:
    ks, means, stds = [], [], []
    for k in milestones:
        mk = str(k)
        vals = []
        for d in datasets:
            arm_obj = d.get("arms", {}).get(arm, {})
            ms = arm_obj.get("milestones", {}).get(mk)
            if not ms:
                continue
            if key == "active_basis_count":
                vals.append(float(_extract_true_bases(arm_obj, k, ms.get("active_basis_count", k))))
            else:
                vals.append(float(ms.get(key, 0.0)) * scale)
        if vals:
            ks.append(k)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0)
    return ks, means, stds


def generate_fig7_libero100_scaling(json_paths: List[Path], out_path: Path) -> None:
    datasets = load_libero_seed_jsons(json_paths)
    if not datasets:
        raise FileNotFoundError("No LIBERO-100 JSON files found.")

    milestones = [10, 30, 50, 70, 100]
    styles = {
        "arm_frozen_mlp": ("Baseline: Frozen MLP (1-Stage Argmax)", "#2563EB", "s", "--"),
        "arm_er_5pct": ("Baseline: Experience Replay (5%)", "#64748B", "^", ":"),
        "arm_dtra_rigid": ("DTRA v3.0 Rigid (alpha=0 Ablation)", "#D97706", "D", "-."),
        "arm_dtra_elastic": ("DTRA v3.0 Full (Elastic Snap-Back, Ours)", "#059669", "o", "-"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8))
    ax_sr, ax_dup, ax_basis, ax_bar = axes.flatten()

    # Panel A: Mean SR across K=10..100
    for arm, (label, color, marker, ls) in styles.items():
        ks, means, stds = aggregate_milestone_series(datasets, arm, milestones, "mean_sr", 100.0)
        if not ks:
            continue
        ax_sr.errorbar(
            ks,
            means,
            yerr=stds if any(s > 0 for s in stds) else None,
            label=label,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.2,
            markersize=6.5,
            capsize=3.5,
        )
        if arm in ("arm_dtra_elastic", "arm_frozen_mlp") and 100 in ks:
            idx = ks.index(100)
            dy = 2.6 if arm == "arm_dtra_elastic" else -4.2
            ax_sr.annotate(
                f"{means[idx]:.2f}%",
                (100, means[idx]),
                xytext=(92, means[idx] + dy),
                fontsize=8.5,
                fontweight="bold",
                color=color,
            )
    ax_sr.set_title("(A) LIBERO-100 Milestone Mean Success Rate (K=10 -> 100)")
    ax_sr.set_xlabel("Learned Tasks K")
    ax_sr.set_ylabel("Mean Success Rate (%)")
    ax_sr.set_xticks(milestones)
    ax_sr.set_ylim(34.0, 103.0)
    ax_sr.grid(True, linestyle="--", alpha=0.35)
    ax_sr.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.8)

    # Panel B: Dup-Lang RouteAcc (and Overall RouteAcc thin overlay)
    mlp_dup_k100 = None
    for arm in ("arm_frozen_mlp", "arm_dtra_rigid", "arm_dtra_elastic"):
        label, color, marker, ls = styles[arm]
        ks, dup_m, dup_s = aggregate_milestone_series(datasets, arm, milestones, "dup_lang_route_acc", 100.0)
        _, rt_m, _ = aggregate_milestone_series(datasets, arm, milestones, "mean_route_acc", 100.0)
        if not ks:
            continue
        ax_dup.errorbar(
            ks,
            dup_m,
            yerr=dup_s if any(s > 0 for s in dup_s) else None,
            label=f"{label.split(':')[0]} Dup-Lang" if arm != "arm_dtra_elastic" else "DTRA v3.0 Full Dup-Lang (Ours)",
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.3,
            markersize=6.5,
            capsize=3.5,
        )
        if arm == "arm_frozen_mlp":
            ax_dup.plot(ks, rt_m, color=color, linestyle=":", linewidth=1.4, alpha=0.75, label="Frozen MLP Overall RouteAcc")
            if 100 in ks:
                idx = ks.index(100)
                mlp_dup_k100 = dup_m[idx]
                ax_dup.annotate(
                    f"{mlp_dup_k100:.2f}%\n(Voronoi Crash)",
                    (100, mlp_dup_k100),
                    xytext=(68, mlp_dup_k100 + 3.0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                    fontsize=8.2,
                    fontweight="bold",
                    color=color,
                )
        elif arm == "arm_dtra_elastic" and 100 in ks:
            idx = ks.index(100)
            el_dup = dup_m[idx]
            gap_str = f" (+{el_dup - mlp_dup_k100:.2f}%p)" if mlp_dup_k100 is not None else ""
            ax_dup.annotate(
                f"{el_dup:.2f}%{gap_str}",
                (100, el_dup),
                xytext=(55, 91.5),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                fontsize=8.2,
                fontweight="bold",
                color=color,
            )
    ax_dup.set_title("(B) Duplicate-Instruction (Dup-Lang) Routing Retention")
    ax_dup.set_xlabel("Learned Tasks K")
    ax_dup.set_ylabel("Routing Accuracy (%)")
    ax_dup.set_xticks(milestones)
    ax_dup.set_ylim(30.0, 106.0)
    ax_dup.grid(True, linestyle="--", alpha=0.35)
    ax_dup.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.8)

    # Panel C: SharedBasisBank Sub-Linear Saturation vs Uncompressed Linear Bases
    ks_b, bs_m, bs_s = aggregate_milestone_series(datasets, "arm_dtra_elastic", milestones, "active_basis_count", 1.0)
    ax_basis.plot(milestones, milestones, color="#2563EB", linestyle="--", marker="s", linewidth=1.8, label="Baseline Unshared Slots (O(K) = 100)")
    if ks_b:
        ax_basis.errorbar(
            ks_b,
            bs_m,
            yerr=bs_s if any(s > 0 for s in bs_s) else None,
            color="#059669",
            marker="o",
            linewidth=2.4,
            markersize=7.0,
            capsize=3.5,
            label="DTRA SharedBasisBank (73% Compression)",
        )
        for k_v, b_v in zip(ks_b, bs_m):
            ax_basis.annotate(f"{int(round(b_v))}", (k_v, b_v), xytext=(k_v - 2.0, b_v + 4.5), fontsize=8.5, fontweight="bold", color="#065F46")
    ax_basis.fill_between(ks_b, bs_m, milestones[: len(ks_b)], color="#10B981", alpha=0.14)
    ax_basis.set_title("(C) Sub-Linear Shared Motor Basis Growth (SharedBasisBank)")
    ax_basis.set_xlabel("Learned Tasks K")
    ax_basis.set_ylabel("Active Primitive Bases")
    ax_basis.set_xticks(milestones)
    ax_basis.set_ylim(0, 112)
    ax_basis.grid(True, linestyle="--", alpha=0.35)
    ax_basis.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=8.0)

    # Panel D: K=100 Final Comparison Bars (with explicit [PENDING] mark if arm_er_5pct K=100 incomplete)
    bar_arms = ["arm_frozen_mlp", "arm_er_5pct", "arm_dtra_rigid", "arm_dtra_elastic"]
    bar_labels = ["Frozen MLP\n(Baseline)", "ER 5%\n(Replay)", "DTRA Rigid\n(alpha=0)", "DTRA Elastic\n(Ours Full)"]
    x = np.arange(len(bar_arms))
    w = 0.25

    sr_vals, rt_vals, dup_vals = [], [], []
    pending_arms = []
    for arm in bar_arms:
        ks_k, sr_m, _ = aggregate_milestone_series(datasets, arm, [100], "mean_sr", 100.0)
        _, rt_m, _ = aggregate_milestone_series(datasets, arm, [100], "mean_route_acc", 100.0)
        _, dp_m, _ = aggregate_milestone_series(datasets, arm, [100], "dup_lang_route_acc", 100.0)
        if ks_k:
            sr_vals.append(sr_m[0])
            rt_vals.append(rt_m[0])
            dup_vals.append(dp_m[0])
        else:
            sr_vals.append(0.0)
            rt_vals.append(0.0)
            dup_vals.append(0.0)
            pending_arms.append(arm)

    b1 = ax_bar.bar(x - w, sr_vals, width=w, color="#059669", label="Final Mean SR (%)")
    b2 = ax_bar.bar(x, rt_vals, width=w, color="#2563EB", label="Overall RouteAcc (%)")
    b3 = ax_bar.bar(x + w, dup_vals, width=w, color="#D97706", label="Dup-Lang Route (%)")

    for idx, arm in enumerate(bar_arms):
        if arm in pending_arms:
            ax_bar.text(idx, 18.0, "K=100\n[PENDING]", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#64748B", bbox=dict(boxstyle="round,pad=0.25", fc="#F1F5F9", ec="#94A3B8"))
        else:
            for rect in (b1[idx], b2[idx], b3[idx]):
                h = rect.get_height()
                ax_bar.text(rect.get_x() + rect.get_width() / 2.0, h + 1.5, f"{h:.2f}", ha="center", va="bottom", fontsize=6.1, fontweight="bold")

    ax_bar.set_title("(D) K=100 Final Performance Summary")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(bar_labels, fontsize=8.3)
    ax_bar.set_ylabel("Score (%)")
    ax_bar.set_ylim(0, 114)
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_bar.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.5)

    n_seeds = len(datasets)
    fig.suptitle(
        f"LIBERO-100 (K=100 Tasks) Continual Learning Scaling Benchmark (Observed Seeds N={n_seeds})",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_fig8_basis_and_encroachment(json_paths: List[Path], out_path: Path) -> None:
    """Appendix D visual: 100->27 SharedBasisBank assignment, reuse histogram & No-Encroachment margin."""
    datasets = load_libero_seed_jsons(json_paths)
    if not datasets:
        return
    d0 = datasets[0]
    ae = d0.get("arms", {}).get("arm_dtra_elastic", {})
    tl = ae.get("train_logs", [])
    if len(tl) < 10:
        return

    tasks = np.arange(1, len(tl) + 1)
    basis_ids = np.array([int(x.get("key_stats", {}).get("basis_idx", 0)) for x in tl])
    is_new = np.array([bool(x.get("key_stats", {}).get("is_new_basis", 0.0)) for x in tl])
    cos_before = np.array([float(x.get("key_stats", {}).get("max_cos_before", 0.0)) for x in tl])
    cos_after = np.array([float(x.get("key_stats", {}).get("max_cos_after", 0.0)) for x in tl])
    n_bases = int(basis_ids.max()) + 1

    fig = plt.figure(figsize=(12.8, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.25)
    ax_map = fig.add_subplot(gs[0, :])
    ax_cos = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    # Panel A: Sparse 100 x 27 assignment scatter / stem
    ax_map.scatter(tasks[~is_new], basis_ids[~is_new], s=26, color="#2563EB", alpha=0.82, label="Reused Existing Basis")
    ax_map.scatter(tasks[is_new], basis_ids[is_new], s=48, color="#059669", marker="D", edgecolor="#064E3B", linewidth=0.8, label="Newly Allocated Basis (Total 27)")
    ax_map.axvline(10.5, color="#94A3B8", linestyle="--", linewidth=1.2)
    ax_map.text(5.5, n_bases - 1.5, "libero_10", ha="center", fontsize=8.5, fontweight="bold", color="#475569")
    ax_map.text(55.0, n_bases - 1.5, "libero_90 (90 sequential diverse + Dup-Lang tasks)", ha="center", fontsize=8.5, fontweight="bold", color="#475569")
    ax_map.set_title("(A) SharedBasisBank Sparse Task-to-Basis Allocation (K=100 Tasks -> 27 Frozen Bases, 73% Compression)")
    ax_map.set_xlabel("Sequential Task Index T_k (1 .. 100)")
    ax_map.set_ylabel("Assigned Basis ID (0 .. 26)")
    ax_map.set_xlim(0, 102)
    ax_map.set_ylim(-1.2, n_bases + 1.5)
    ax_map.grid(True, linestyle="--", alpha=0.30)
    ax_map.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8.3)

    # Panel B: No-Encroachment Max Cosine Before vs After projection
    ax_cos.plot(tasks[1:], cos_before[1:], color="#DC2626", alpha=0.55, linewidth=1.3, label="Raw Max Prior Cos (Before)")
    ax_cos.plot(tasks[1:], cos_after[1:], color="#059669", linewidth=1.7, label="Guarded Max Prior Cos (After No-Encroachment)")
    ax_cos.fill_between(tasks[1:], cos_after[1:], cos_before[1:], where=cos_before[1:] > cos_after[1:], color="#10B981", alpha=0.18)
    ax_cos.set_title("(B) No-Encroachment Key Projection Across K=100 Tasks")
    ax_cos.set_xlabel("Sequential Task Index T_k")
    ax_cos.set_ylabel("Max Cosine Sim to Prior Keys")
    ax_cos.set_ylim(0.25, 1.03)
    ax_cos.grid(True, linestyle="--", alpha=0.35)
    ax_cos.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.8)

    # Panel C: Per-basis reuse count
    counts = np.bincount(basis_ids, minlength=n_bases)
    bids = np.arange(n_bases)
    colors = ["#059669" if c >= 6 else "#2563EB" for c in counts]
    ax_hist.bar(bids, counts, color=colors, alpha=0.88)
    for b, c in zip(bids, counts):
        ax_hist.text(b, c + 0.15, str(c), ha="center", va="bottom", fontsize=7.0, color="#1E293B")
    ax_hist.set_title("(C) Task Sharing Frequency Across the 27 Frozen Bases")
    ax_hist.set_xlabel("SharedBasisBank Primitive ID (0 .. 26)")
    ax_hist.set_ylabel("Number of Assigned Tasks")
    ax_hist.set_xticks(np.arange(0, n_bases, 2))
    ax_hist.set_ylim(0, max(counts) + 1.8)
    ax_hist.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("DTRA v3.0 Telemetry Diagnostics: SharedBasisBank & No-Encroachment Projection (Seed 42)", fontsize=12.0, fontweight="bold", y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DTRA v3.0 Architecture & LIBERO-100 Scaling Figures")
    parser.add_argument(
        "--jsons",
        nargs="+",
        default=[
            str(LIBERO_DIR / "benchmark_dtra_libero_10_libero_90_k100.json"),
            str(LIBERO_DIR / "benchmark_dtra_libero_10_libero_90_k100_seed100.json"),
            str(LIBERO_DIR / "benchmark_dtra_libero_10_libero_90_k100_seed2026.json"),
        ],
    )
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    generate_fig6_architecture_schematic(FIG_DIR / "fig6_dtra_v3_architecture_schematic.png")
    generate_fig7_libero100_scaling([Path(p) for p in args.jsons], FIG_DIR / "fig7_libero100_scaling_curves.png")
    generate_fig8_basis_and_encroachment([Path(p) for p in args.jsons], FIG_DIR / "fig8_libero100_basis_and_callstack.png")


if __name__ == "__main__":
    main()
