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
        boxstyle="round,pad=0.008,rounding_size=0.014",
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
        "Dual-Cam RGB (128×128)\nFrozen ResNet-18 (L1–L3)\n+ 12-d Proprioception\n+ LightweightTextEncoder\n  (CRC32 Hash 128-d, No VLM)",
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
        "Stage 1: O(K) Scan → Top-M (M=5)\nStage 2: 3-Modality CMC\n  (4×4+DoG Vis / Prop / Lang)\nNo-Encroachment Constraint:\n  cos(μ_j, k_i) ≤ s_jj − m_safe",
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
        "LIBERO: theta_head + SpatialAdapter\n  (MW: Elliptical Ring B_ring(θ))\nSpherical Elastic Interp. α_i ∈ (0, 1)\nSharedBasisBank (K=100 → 27\n  frozen knot bases, ∇θ = 0)",
        fc="#ECFDF5",
        ec="#059669",
        title_color="#065F46",
        fontsize_title=9.5,
        fontsize_body=8.0,
    )

    _draw_box(
        ax,
        (0.695, 0.56),
        0.29,
        0.33,
        "C² B-Spline & Episode-Boundary\nState Reset (Zero Drift)",
        "Clamped Cubic B-Spline (N_cp=8 → H=16)\n7-DoF Continuous Action Chunk\n1) LIBERO: episode_snap_back() resets\n   routing inertia & diag. buffers → 0.0\n2) MW: active chunk + ep. snap-back",
        fc="#FFF7ED",
        ec="#EA580C",
        title_color="#9A3412",
        fontsize_title=9.5,
        fontsize_body=8.0,
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
        "[IMPLEMENTED] Region Level\nDTRA Single-Ring (K=100)",
        "Single Cortical Column Module\nK=100 LIBERO Tasks Validated\n27 Shared Bases (73% Slot / 5.3% Tot.)\nO(1) Active Kernel & White-Box Stack",
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
        "Not Implemented in Current DTRA\nMulti-Region Hierarchical Routing\nIsometric Zero-Padding [v; 0]\nSub-linear O(log K) Lookup Vision",
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
            # Non-destructively merge supplemental Seed 2026 arms (rigid / er5) if finalized
            if "seed2026" in target.name and isinstance(data.get("arms"), dict):
                for arm_k, supp_fn in [
                    ("arm_dtra_rigid", "benchmark_dtra_libero_k100_seed2026_rigid.json"),
                    ("arm_er_5pct", "benchmark_dtra_libero_k100_seed2026_er5.json"),
                ]:
                    supp_p = target.parent / supp_fn
                    if arm_k not in data["arms"] and supp_p.exists():
                        try:
                            with open(supp_p, "r", encoding="utf-8") as sf:
                                sd = json.load(sf)
                            av = sd.get("arms", {}).get(arm_k, {})
                            if av.get("status") == "completed" or "100" in av.get("milestones", {}):
                                data["arms"][arm_k] = av
                        except Exception:
                            pass
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
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return ks, means, stds


def generate_fig7_libero100_scaling(json_paths: List[Path], out_path: Path) -> None:
    datasets = load_libero_seed_jsons(json_paths)
    if not datasets:
        raise FileNotFoundError("No LIBERO-100 JSON files found.")
    n_seeds = len(datasets)

    # Load matched-encoder external CL baselines (DER++, L2P, O-LoRA) if present
    ext_cl_path = LIBERO_DIR / "benchmark_dtra_libero_v31_ext_cl_k100_seed42.json"
    ext_datasets: List[dict] = []
    if ext_cl_path.exists():
        with open(ext_cl_path, "r", encoding="utf-8") as ef:
            ext_datasets = [json.load(ef)]

    milestones = [10, 30, 50, 70, 100]
    styles = {
        "arm_frozen_mlp": (f"Frozen MLP (1-Stage, N={n_seeds})", "#2563EB", "s", "--"),
        "arm_er_5pct": (f"Experience Replay 5% (N={n_seeds})", "#64748B", "^", ":"),
        "arm_dtra_rigid": (f"DTRA Rigid (α=0, N={n_seeds})", "#D97706", "D", "-."),
        "arm_dtra_elastic": (f"DTRA Elastic (Ours, N={n_seeds})", "#059669", "o", "-"),
    }
    ext_styles = {
        "arm_derpp": ("DER++ (5% Buf., S42)", "#9333EA", "v", "--"),
        "arm_l2p": ("L2P (Prompt Pool, S42)", "#DC2626", "X", ":"),
        "arm_olora": ("O-LoRA (Subspace, S42)", "#B45309", "P", ":"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    ax_sr, ax_dup, ax_basis, ax_bar = axes.flatten()

    # Panel A: Mean SR across K=10..100 (Core 4 Arms + 3 External CL Baselines)
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
            markersize=6.2,
            capsize=3.2,
        )
        if arm in ("arm_dtra_elastic", "arm_frozen_mlp") and 100 in ks:
            idx = ks.index(100)
            dy = 3.2 if arm == "arm_dtra_elastic" else -5.5
            ax_sr.annotate(
                f"{means[idx]:.2f}%",
                (100, means[idx]),
                xytext=(88, means[idx] + dy),
                fontsize=8.2,
                fontweight="bold",
                color=color,
            )
    if ext_datasets:
        ext_k100_sr: Dict[str, float] = {}
        for arm, (label, color, marker, ls) in ext_styles.items():
            ks, means, _ = aggregate_milestone_series(ext_datasets, arm, milestones, "mean_sr", 100.0)
            if not ks:
                continue
            ax_sr.plot(
                ks,
                means,
                label=label,
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=1.9,
                markersize=6.0,
                alpha=0.92,
            )
            if 100 in ks:
                ext_k100_sr[arm] = means[ks.index(100)]
                if arm == "arm_derpp":
                    ax_sr.annotate(
                        f"{ext_k100_sr[arm]:.1f}%",
                        (100, ext_k100_sr[arm]),
                        xytext=(88, ext_k100_sr[arm] - 6.5),
                        fontsize=7.8,
                        fontweight="bold",
                        color=color,
                    )
        if "arm_l2p" in ext_k100_sr and "arm_olora" in ext_k100_sr:
            ax_sr.annotate(
                f"{ext_k100_sr['arm_l2p']:.1f}% / {ext_k100_sr['arm_olora']:.1f}%",
                (100, ext_k100_sr["arm_l2p"]),
                xytext=(84, 6.0),
                fontsize=7.6,
                fontweight="bold",
                color="#DC2626",
            )
    ax_sr.set_title("(A) LIBERO-100 Milestone Mean Success Rate (K = 10 → 100)")
    ax_sr.set_xlabel("Learned Tasks K")
    ax_sr.set_ylabel("Mean Success Rate (%)")
    ax_sr.set_xticks(milestones)
    ax_sr.set_ylim(-4.0, 104.0)
    ax_sr.grid(True, linestyle="--", alpha=0.35)
    ax_sr.legend(loc="lower center", bbox_to_anchor=(0.42, 0.14), frameon=True, framealpha=0.94, fontsize=6.6, ncol=2)

    # Panel B: Dup-Lang RouteAcc (and Overall RouteAcc thin overlay)
    mlp_dup_k100 = None
    dup_labels = {
        "arm_frozen_mlp": "Frozen MLP Dup-Lang (1-Stage)",
        "arm_dtra_rigid": f"DTRA Rigid Dup-Lang (α=0, N={n_seeds})",
        "arm_dtra_elastic": f"DTRA Elastic Dup-Lang (Ours, N={n_seeds})",
    }
    for arm in ("arm_frozen_mlp", "arm_dtra_rigid", "arm_dtra_elastic"):
        _, color, marker, ls = styles[arm]
        ks, dup_m, dup_s = aggregate_milestone_series(datasets, arm, milestones, "dup_lang_route_acc", 100.0)
        _, rt_m, _ = aggregate_milestone_series(datasets, arm, milestones, "mean_route_acc", 100.0)
        if not ks:
            continue
        ax_dup.errorbar(
            ks,
            dup_m,
            yerr=dup_s if any(s > 0 for s in dup_s) else None,
            label=dup_labels[arm],
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
                    xytext=(66, mlp_dup_k100 + 3.5),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                    fontsize=8.2,
                    fontweight="bold",
                    color=color,
                )
        elif arm == "arm_dtra_elastic" and 100 in ks:
            idx = ks.index(100)
            el_dup = dup_m[idx]
            gap_str = f" (+{el_dup - mlp_dup_k100:.2f} pp)" if mlp_dup_k100 is not None else ""
            ax_dup.annotate(
                f"{el_dup:.2f}%{gap_str}",
                (100, el_dup),
                xytext=(52, 91.0),
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
    ax_dup.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.6)

    # Panel C: SharedBasisBank Sub-Linear Saturation vs Uncompressed Linear Bases
    ks_b, bs_m, bs_s = aggregate_milestone_series(datasets, "arm_dtra_elastic", milestones, "active_basis_count", 1.0)
    ax_basis.plot(milestones, milestones, color="#2563EB", linestyle="--", marker="s", linewidth=1.8, label="Baseline Unshared Slots (O(K) = 100)")
    if ks_b:
        final_b = int(round(bs_m[-1]))
        red_pct = int(round((1.0 - final_b / 100.0) * 100.0))
        ax_basis.errorbar(
            ks_b,
            bs_m,
            yerr=bs_s if any(s > 0 for s in bs_s) else None,
            color="#059669",
            marker="o",
            linewidth=2.4,
            markersize=7.0,
            capsize=3.5,
            label=f"DTRA SharedBasisBank ({red_pct}% Basis-Slot Reduction)",
        )
        for k_v, b_v in zip(ks_b, bs_m):
            dy_b = -6.5 if k_v == 10 else 4.5
            dx_b = 1.5 if k_v == 10 else -2.0
            ax_basis.annotate(f"{int(round(b_v))}", (k_v, b_v), xytext=(k_v + dx_b, b_v + dy_b), fontsize=8.5, fontweight="bold", color="#065F46")
    ax_basis.fill_between(ks_b, bs_m, milestones[: len(ks_b)], color="#10B981", alpha=0.14)
    ax_basis.set_title("(C) Sub-Linear Shared Motor Basis Growth (SharedBasisBank)")
    ax_basis.set_xlabel("Learned Tasks K")
    ax_basis.set_ylabel("Active Primitive Bases")
    ax_basis.set_xticks(milestones)
    ax_basis.set_ylim(-2, 112)
    ax_basis.grid(True, linestyle="--", alpha=0.35)
    ax_basis.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.8)

    # Panel D: K=100 Final Comparison Bars across all 7 Parity Arms (Mean SR & Forgetting Rate)
    bar_specs = [
        ("arm_olora", "O-LoRA\n(S42)", ext_datasets, False),
        ("arm_l2p", "L2P\n(S42)", ext_datasets, False),
        ("arm_derpp", "DER++\n(S42)", ext_datasets, False),
        ("arm_er_5pct", f"ER 5%\n(N={n_seeds})", datasets, True),
        ("arm_frozen_mlp", f"Frozen MLP\n(N={n_seeds})", datasets, True),
        ("arm_dtra_rigid", f"DTRA Rigid\n(N={n_seeds})", datasets, True),
        ("arm_dtra_elastic", f"DTRA Elastic\n(N={n_seeds})", datasets, True),
    ]
    x = np.arange(len(bar_specs))
    w = 0.36
    sr_vals, sr_errs, fr_vals, fr_errs, labels_d, has_err = [], [], [], [], [], []
    for arm, lbl, ds_list, is_multi in bar_specs:
        labels_d.append(lbl)
        has_err.append(is_multi and len(ds_list) > 1)
        ks_k, sr_m, sr_s = aggregate_milestone_series(ds_list, arm, [100], "mean_sr", 100.0)
        _, fr_m, fr_s = aggregate_milestone_series(ds_list, arm, [100], "forgetting_rate", 100.0)
        if ks_k:
            sr_vals.append(sr_m[0])
            sr_errs.append(sr_s[0] if is_multi else 0.0)
            fr_vals.append(fr_m[0])
            fr_errs.append(fr_s[0] if is_multi else 0.0)
        else:
            sr_vals.append(0.0); sr_errs.append(0.0)
            fr_vals.append(0.0); fr_errs.append(0.0)

    b1 = ax_bar.bar(x - w / 2, sr_vals, width=w, color="#059669", label="Final Mean SR (%) ↑")
    b2 = ax_bar.bar(x + w / 2, fr_vals, width=w, color="#EF4444", alpha=0.82, label="Forgetting Rate FR (%) ↓")
    # Draw error bars ONLY for multi-seed arms (no zero-cap error bars on single-seed S42 baselines)
    multi_idx = [i for i, flag in enumerate(has_err) if flag]
    if multi_idx:
        ax_bar.errorbar(
            x[multi_idx] - w / 2,
            [sr_vals[i] for i in multi_idx],
            yerr=[sr_errs[i] for i in multi_idx],
            fmt="none",
            ecolor="#111827",
            elinewidth=1.1,
            capsize=2.5,
        )
        ax_bar.errorbar(
            x[multi_idx] + w / 2,
            [fr_vals[i] for i in multi_idx],
            yerr=[fr_errs[i] for i in multi_idx],
            fmt="none",
            ecolor="#111827",
            elinewidth=1.1,
            capsize=2.5,
        )

    for rect, err in zip(b1, sr_errs):
        h = rect.get_height()
        ax_bar.text(rect.get_x() + rect.get_width() / 2.0, h + err + 1.8, f"{h:.1f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#065F46")
    for rect, err in zip(b2, fr_errs):
        h = rect.get_height()
        ax_bar.text(rect.get_x() + rect.get_width() / 2.0, h + err + 1.8, f"{h:.1f}", ha="center", va="bottom", fontsize=6.6, color="#991B1B")

    ax_bar.set_title("(D) K = 100 Final Mean SR vs. Forgetting (7 Parity Arms)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels_d, fontsize=7.4)
    ax_bar.set_ylabel("Percentage (%)")
    ax_bar.set_ylim(0, 108)
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_bar.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=7.5)

    fig.suptitle(
        f"LIBERO-100 (K = 100 Tasks) Continual Learning Scaling Benchmark (Core Arms N = {n_seeds}, ddof = 1)",
        fontsize=12.4,
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

    n_tasks = len(tl)
    tasks = np.arange(1, n_tasks + 1)
    basis_ids = np.array([int(x.get("key_stats", {}).get("basis_idx", 0)) for x in tl])
    is_new = np.array([bool(x.get("key_stats", {}).get("is_new_basis", 0.0)) for x in tl])
    cos_before = np.array([float(x.get("key_stats", {}).get("max_cos_before", 0.0)) for x in tl])
    cos_after = np.array([float(x.get("key_stats", {}).get("max_cos_after", 0.0)) for x in tl])
    n_bases = int(basis_ids.max()) + 1
    red_pct = int(round((1.0 - n_bases / n_tasks) * 100.0))

    fig = plt.figure(figsize=(12.8, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.25)
    ax_map = fig.add_subplot(gs[0, :])
    ax_cos = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    # Panel A: Sparse 100 x 27 assignment scatter / stem
    ax_map.scatter(tasks[~is_new], basis_ids[~is_new], s=26, color="#2563EB", alpha=0.82, label="Reused Existing Basis")
    ax_map.scatter(tasks[is_new], basis_ids[is_new], s=48, color="#059669", marker="D", edgecolor="#064E3B", linewidth=0.8, label=f"Newly Allocated Basis (Total {n_bases})")
    ax_map.axvline(10.5, color="#94A3B8", linestyle="--", linewidth=1.2)
    ax_map.text(5.5, n_bases - 0.8, "libero_10", ha="center", fontsize=8.5, fontweight="bold", color="#475569")
    ax_map.text(56.0, n_bases - 0.8, "libero_90 (90 sequential diverse + Dup-Lang tasks)", ha="center", fontsize=8.5, fontweight="bold", color="#475569")
    ax_map.set_title(f"(A) SharedBasisBank Sparse Task-to-Basis Allocation (K = {n_tasks} → {n_bases} Frozen Bases, {red_pct}% Basis-Slot Reduction)")
    ax_map.set_xlabel(f"Sequential Task Index $T_k$ (1 .. {n_tasks})")
    ax_map.set_ylabel(f"Assigned Basis ID (0 .. {n_bases - 1})")
    ax_map.set_xlim(0, n_tasks + 2)
    ax_map.set_ylim(-1.2, n_bases + 2.2)
    ax_map.grid(True, linestyle="--", alpha=0.30)
    ax_map.legend(loc="upper left", bbox_to_anchor=(0.01, 0.86), frameon=True, framealpha=0.94, fontsize=8.1)

    # Panel B: No-Encroachment Max Cosine Before vs After projection
    ax_cos.plot(tasks[1:], cos_before[1:], color="#DC2626", alpha=0.55, linewidth=1.3, label="Raw Max Prior Cos (Before)")
    ax_cos.plot(tasks[1:], cos_after[1:], color="#059669", linewidth=1.7, label="Guarded Max Prior Cos (After No-Encroachment)")
    ax_cos.fill_between(tasks[1:], cos_after[1:], cos_before[1:], where=cos_before[1:] > cos_after[1:], color="#10B981", alpha=0.18)
    ax_cos.set_title(f"(B) No-Encroachment Key Projection Across K = {n_tasks} Tasks")
    ax_cos.set_xlabel("Sequential Task Index $T_k$")
    ax_cos.set_ylabel("Max Cosine Sim to Prior Keys")
    ax_cos.set_ylim(0.25, 1.03)
    ax_cos.grid(True, linestyle="--", alpha=0.35)
    ax_cos.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.8)

    # Panel C: Per-basis reuse count
    counts = np.bincount(basis_ids, minlength=n_bases)
    bids = np.arange(n_bases)
    hi_mask = counts >= 6
    ax_hist.bar(bids[hi_mask], counts[hi_mask], color="#059669", alpha=0.88, label="High Reuse (≥ 6 tasks)")
    ax_hist.bar(bids[~hi_mask], counts[~hi_mask], color="#2563EB", alpha=0.88, label="Standard Reuse (≤ 5 tasks)")
    for b, c in zip(bids, counts):
        ax_hist.text(b, c + 0.15, str(c), ha="center", va="bottom", fontsize=7.0, color="#1E293B")
    ax_hist.set_title(f"(C) Task Sharing Frequency Across the {n_bases} Frozen Bases")
    ax_hist.set_xlabel(f"SharedBasisBank Primitive ID (0 .. {n_bases - 1})")
    ax_hist.set_ylabel("Number of Assigned Tasks")
    ax_hist.set_xticks(np.arange(0, n_bases, 2))
    ax_hist.set_ylim(0, max(counts) + 2.4)
    ax_hist.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_hist.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=7.8)

    fig.suptitle("DTRA Telemetry Diagnostics: SharedBasisBank & No-Encroachment Projection (Seed 42)", fontsize=12.0, fontweight="bold", y=0.98)
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
