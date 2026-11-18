"""experiments/eval_routing_perturbation.py

[Experiment 1] 목표 이미지 아핀 섭동/노이즈/가림 하 자율 프로토타입 라우팅 강건성 곡선
& Hybrid Key (Conv4 vs Multi-Scale Spatial Pooling) 비중 Ablation + OOD 폐루프 롤아웃 평가.

평가 항목:
  1) 5가지 키 비중 조합 (Conv4^2 / Spatial^2 에너지 비율):
     - Spatial-Only  (0% / 100%): (w_conv=0.0, w_sp=1.0)
     - ABK Default   (20% / 80%): (w_conv=0.5, w_sp=1.0)
     - Balanced      (50% / 50%): (w_conv=1.0, w_sp=1.0)
     - Conv4-Heavy   (80% / 20%): (w_conv=1.0, w_sp=0.5)
     - Conv4-Only    (100% / 0%): (w_conv=1.0, w_sp=0.0)
  2) 5가지 섭동 축:
     - Translation shift (px): 0, 2, 4, 8, 12, 16
     - Rotation (deg):         0, 5, 10, 15, 25, 35
     - Scale jitter (±%):      0%, 5%, 10%, 15%, 20%
     - Gaussian noise (sigma): 0.00, 0.05, 0.10, 0.20, 0.30, 0.40
     - Random block occlusion: 0%, 10%, 20%, 30%, 40%
  3) OOD 섭동 목표 이미지에 대한 실제 Closed-Loop Rollout F1 비교 (Select-then-Blend vs Argmax)

사용법:
  python experiments/eval_routing_perturbation.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from experiments.benchmark_autonomous_cl import eval_episode_rollout, set_seed
from experiments.benchmark_three_models import pick_device, safe_savefig
from train.bspline_policy import AutonomousABKPolicy

OUT_DIR = PROJECT_ROOT / "outputs"


def load_canonical_targets(tasks: List[str]) -> Dict[str, np.ndarray]:
    targets = {}
    for t in tasks:
        npz_path = PROJECT_ROOT / "data" / "bspline" / f"bspline_vla_dataset_{t}_m6_h16_s8.npz"
        with np.load(npz_path, allow_pickle=True) as data:
            targets[t] = data["target_images"][0]
    return targets


@torch.no_grad()
def extract_key_components(
    model: AutonomousABKPolicy, target_imgs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hybrid Visual Prototype Key의 두 성분 분리 추출.
    - conv_feat: (B, 4096) — Conv4 특징맵(256 x 4 x 4) L2 정규화 벡터 (의미/텍스처 강인성)
    - sp_feat  : (B, 1280) — 목표 도안 AvgPool(32x32 + 16x16 = 1024 + 256) L2 정규화 벡터 (공간/픽셀 정합)
    """
    tgt = target_imgs.float()
    if tgt.dim() == 3:
        tgt = tgt.unsqueeze(1)
    if tgt.max().item() > 1.5:
        tgt = tgt / 255.0

    model.encoder.eval()
    # 캔버스 채널을 0으로 둔 초기 관찰 obs0: (B, 2, 128, 128)
    obs0 = torch.cat([torch.zeros_like(tgt), tgt], dim=1)
    conv_map = model._conv_spatial_trunk(obs0)
    conv_feat = F.normalize(F.adaptive_avg_pool2d(conv_map, (4, 4)).flatten(1), p=2, dim=-1)
    p32 = F.avg_pool2d(tgt, kernel_size=4, stride=4)
    p32 = F.avg_pool2d(p32, kernel_size=3, stride=1, padding=1).flatten(1)
    p16 = F.avg_pool2d(tgt, kernel_size=8, stride=8).flatten(1)
    sp_feat = F.normalize(torch.cat([p32, p16], dim=1), p=2, dim=-1)
    return conv_feat, sp_feat


def compose_key(conv_feat: torch.Tensor, sp_feat: torch.Tensor, w_conv: float, w_sp: float) -> torch.Tensor:
    """
    가중 결합 프로토타입 키 생성: k = normalize([w_conv * conv_feat; w_sp * sp_feat]) -> (B, 5376).
    에너지 비율 = w_conv^2 : w_sp^2 (기본 ABK 설정: w_conv=0.5, w_sp=1.0 -> 20% : 80%).
    """
    return F.normalize(torch.cat([conv_feat * w_conv, sp_feat * w_sp], dim=1), p=2, dim=-1)


def apply_affine_tensor(
    img: torch.Tensor, dx_px: float = 0.0, dy_px: float = 0.0, angle_deg: float = 0.0, scale: float = 1.0
) -> torch.Tensor:
    """(B, 1, H, W) 이미지에 아핀 변환 적용 (배경 0 패딩)."""
    B, _, H, W = img.shape
    theta = math.radians(angle_deg)
    inv_s = 1.0 / max(1e-4, scale)
    tx = -2.0 * dx_px / W
    ty = -2.0 * dy_px / H
    mat = torch.tensor(
        [
            [
                inv_s * math.cos(theta),
                inv_s * math.sin(theta),
                tx * math.cos(theta) + ty * math.sin(theta),
            ],
            [
                -inv_s * math.sin(theta),
                inv_s * math.cos(theta),
                -tx * math.sin(theta) + ty * math.cos(theta),
            ],
        ],
        dtype=torch.float32,
        device=img.device,
    ).unsqueeze(0).expand(B, -1, -1)
    grid = F.affine_grid(mat, size=img.size(), align_corners=False)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False).clamp(0.0, 1.0)


def generate_perturbed_batch(
    base_imgs: torch.Tensor, kind: str, level: float, n_draws: int, rng: np.random.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(K, 1, 128, 128) -> (K * n_draws, 1, 128, 128) 섭동 쿼리 및 정답 슬롯 인덱스 생성."""
    K, _, H, W = base_imgs.shape
    queries = []
    labels = []
    for k in range(K):
        src = base_imgs[k : k + 1]
        for d in range(n_draws):
            if kind == "translation":
                if level == 0:
                    dx, dy = 0.0, 0.0
                else:
                    ang = rng.uniform(0, 2 * math.pi)
                    dx, dy = level * math.cos(ang), level * math.sin(ang)
                q = apply_affine_tensor(src, dx_px=dx, dy_px=dy)
            elif kind == "rotation":
                sign = -1.0 if d % 2 else 1.0
                ang = 0.0 if level == 0 else sign * rng.uniform(0.7 * level, level)
                q = apply_affine_tensor(src, angle_deg=ang)
            elif kind == "scale":
                if level == 0:
                    sc = 1.0
                else:
                    sc = (1.0 - level) if d % 2 == 0 else (1.0 + level)
                q = apply_affine_tensor(src, scale=sc)
            elif kind == "gaussian_noise":
                noise = torch.randn_like(src) * level
                q = (src + noise).clamp(0.0, 1.0)
            elif kind == "occlusion":
                q = src.clone()
                if level > 0:
                    side = max(4, int(round(H * math.sqrt(level))))
                    y0 = int(rng.integers(0, H - side + 1))
                    x0 = int(rng.integers(0, W - side + 1))
                    q[:, :, y0 : y0 + side, x0 : x0 + side] = 0.0
            else:
                raise ValueError(kind)
            queries.append(q)
            labels.append(k)
    return torch.cat(queries, dim=0), torch.tensor(labels, dtype=torch.long, device=base_imgs.device)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    rng = np.random.default_rng(42)
    device = pick_device()

    bench = json.loads((OUT_DIR / "autonomous_cl_benchmark.json").read_text(encoding="utf-8"))
    tasks = bench["tasks"]
    targets = load_canonical_targets(tasks)

    ckpt = torch.load(OUT_DIR / "autonomous_model_d_final.pth", map_location="cpu", weights_only=False)
    model = AutonomousABKPolicy(num_control_points=6, horizon=16, num_modes=6, temperature=0.05).to(device)
    for idx in range(len(tasks)):
        t_tensor = torch.from_numpy(targets[tasks[idx]].astype(np.float32) / 255.0)[None, None].to(device)
        model.register_or_expand_slot(t_tensor, force_expand=(idx > 0))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    base_imgs = torch.cat(
        [torch.from_numpy(targets[t].astype(np.float32) / 255.0)[None, None].to(device) for t in tasks], dim=0
    )
    proto_conv, proto_sp = extract_key_components(model, base_imgs)

    ablation_configs = [
        ("Spatial-Only (0/100)", 0.0, 1.0),
        ("ABK Default (20/80)", 0.5, 1.0),
        ("Balanced (50/50)", 1.0, 1.0),
        ("Conv4-Heavy (80/20)", 1.0, 0.5),
        ("Conv4-Only (100/0)", 1.0, 0.0),
    ]

    perturbations = {
        "translation": [0.0, 2.0, 4.0, 8.0, 12.0, 16.0],
        "rotation": [0.0, 5.0, 10.0, 15.0, 25.0, 35.0],
        "scale": [0.0, 0.05, 0.10, 0.15, 0.20],
        "gaussian_noise": [0.0, 0.05, 0.10, 0.20, 0.30, 0.40],
        "occlusion": [0.0, 0.10, 0.20, 0.30, 0.40],
    }

    n_draws = 20
    curve_results: Dict[str, Dict] = {}
    print("\n=== Experiment 1A: Perturbation Routing Robustness & Key Weight Ablation ===")
    for p_kind, levels in perturbations.items():
        curve_results[p_kind] = {"levels": levels, "configs": {name: {"top1_acc": [], "diag_conf": []} for name, _, _ in ablation_configs}}
        for lvl in levels:
            q_imgs, labels = generate_perturbed_batch(base_imgs, p_kind, lvl, n_draws=n_draws, rng=rng)
            q_conv, q_sp = extract_key_components(model, q_imgs)
            for name, wc, ws in ablation_configs:
                keys = compose_key(proto_conv, proto_sp, wc, ws)
                q_keys = compose_key(q_conv, q_sp, wc, ws)
                sims = torch.matmul(q_keys, keys.t())
                probs = F.softmax(sims / 0.05, dim=-1)
                preds = sims.argmax(dim=-1)
                acc = float((preds == labels).float().mean().item()) * 100.0
                diag_c = float(probs[torch.arange(len(labels), device=device), labels].mean().item()) * 100.0
                curve_results[p_kind]["configs"][name]["top1_acc"].append(acc)
                curve_results[p_kind]["configs"][name]["diag_conf"].append(diag_c)

        print(f"  [{p_kind}] levels={levels}")
        for name, _, _ in ablation_configs:
            accs = [round(a, 1) for a in curve_results[p_kind]["configs"][name]["top1_acc"]]
            print(f"    {name:22s}: Top-1 Acc (%) = {accs}")

    # Experiment 1B: Closed-loop rollout F1 under representative OOD perturbations
    print("\n=== Experiment 1B: OOD Target Closed-Loop Rollout F1 (Select-then-Blend vs Argmax) ===")
    ood_specs = [
        ("clean", lambda x: x),
        ("shift_4px", lambda x: apply_affine_tensor(x, dx_px=4.0, dy_px=-3.0)),
        ("rot_8deg", lambda x: apply_affine_tensor(x, angle_deg=8.0)),
        ("noise_0.15", lambda x: (x + torch.randn_like(x) * 0.15).clamp(0.0, 1.0)),
        ("occlusion_20pct", lambda x: _occlude(x, 0.20)),
    ]

    def _occlude(x: torch.Tensor, frac: float) -> torch.Tensor:
        out = x.clone()
        side = int(round(128 * math.sqrt(frac)))
        y0, x0 = 64 - side // 2, 64 - side // 2
        out[:, :, y0 : y0 + side, x0 : x0 + side] = 0.0
        return out

    # Ensure deterministic noise for Experiment 1B
    torch.manual_seed(42)
    ood_rollout = {}
    for ood_name, fn in ood_specs:
        ood_rollout[ood_name] = {}
        for rmode in ["blend", "argmax"]:
            model.routing_mode = rmode
            f1s, cds = [], []
            per_t = {}
            for i, t_name in enumerate(tasks):
                src_t = base_imgs[i : i + 1]
                pert_t = fn(src_t)
                pert_np = np.clip((pert_t[0, 0].detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
                res, _ = eval_episode_rollout(
                    model, pert_np, device=device, task_name=t_name, task_id=None, max_iters=100, exec_steps=16, patience=8
                )
                per_t[t_name] = res["f1"]
                f1s.append(res["f1"])
                cds.append(res["chamfer_dist"])
            ood_rollout[ood_name][rmode] = {
                "avg_f1": float(np.mean(f1s)),
                "avg_chamfer": float(np.mean(cds)),
                "per_task_f1": per_t,
            }
        print(
            f"  {ood_name:16s} | Blend F1={ood_rollout[ood_name]['blend']['avg_f1']*100:5.2f}% | "
            f"Argmax F1={ood_rollout[ood_name]['argmax']['avg_f1']*100:5.2f}%"
        )

    out_json = OUT_DIR / "exp1_routing_perturbation.json"
    out_json.write_text(
        json.dumps({"tasks": tasks, "curves": curve_results, "ood_rollout": ood_rollout}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Plot 2x3 panels (5 perturbation Top-1 accuracy curves + 1 OOD Rollout F1 bar chart)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=160)
    fig.patch.set_facecolor("#F8FAFC")
    palette = ["#94A3B8", "#2563EB", "#16A34A", "#F59E0B", "#DC2626"]
    titles = {
        "translation": ("Translation Shift", "Shift Radius (px)"),
        "rotation": ("In-Plane Rotation", "Max Angle (±deg)"),
        "scale": ("Scale Jitter", "Scale Perturbation (±%)"),
        "gaussian_noise": ("Additive Gaussian Noise", "Noise Std (σ)"),
        "occlusion": ("Random Square Occlusion", "Occluded Area Ratio (%)"),
    }

    for ax, p_kind in zip(axes.flat[:5], perturbations.keys()):
        levels = curve_results[p_kind]["levels"]
        x_vals = [l * 100 if p_kind in ("scale", "occlusion") else l for l in levels]
        for (name, _, _), col in zip(ablation_configs, palette):
            lw = 2.6 if "Default" in name else 1.7
            ls = "-" if "Default" in name else "--"
            ax.plot(x_vals, curve_results[p_kind]["configs"][name]["top1_acc"], marker="o", linewidth=lw, linestyle=ls, color=col, label=name)
        t_str, x_str = titles[p_kind]
        ax.set_title(t_str, fontweight="bold")
        ax.set_xlabel(x_str)
        ax.set_ylabel("Top-1 Routing Accuracy (%)")
        ax.set_ylim(0, 105)
        ax.axhline(25.0, color="#CBD5E1", linestyle=":", linewidth=1.2)
        ax.grid(alpha=0.3)

    # Panel 6: OOD Closed-Loop Rollout F1
    ax_bar = axes.flat[5]
    ood_keys = [k for k, _ in ood_specs]
    xb = np.arange(len(ood_keys))
    w = 0.36
    b_vals = [ood_rollout[k]["blend"]["avg_f1"] * 100 for k in ood_keys]
    a_vals = [ood_rollout[k]["argmax"]["avg_f1"] * 100 for k in ood_keys]
    ax_bar.bar(xb - w / 2, b_vals, width=w, color="#2563EB", label="Select-then-Blend")
    ax_bar.bar(xb + w / 2, a_vals, width=w, color="#16A34A", label="Hard Argmax")
    ax_bar.set_title("OOD Closed-Loop Rollout Avg F1 (%)", fontweight="bold")
    ax_bar.set_xticks(xb)
    ax_bar.set_xticklabels(ood_keys, rotation=20, ha="right", fontsize=8)
    ax_bar.set_ylim(0, 100)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend(fontsize=8)

    handles, leg_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.96), frameon=True, fontsize=8.5)
    plt.suptitle("Experiment 1: OOD Target Routing Robustness & Key Weight Ablation (80 queries/level)", fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    saved_png = safe_savefig(OUT_DIR / "exp1_routing_perturbation.png")
    print(f"\n[Done] Saved {out_json} and {saved_png}")


if __name__ == "__main__":
    main()
