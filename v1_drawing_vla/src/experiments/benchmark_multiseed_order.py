"""experiments/benchmark_multiseed_order.py

[Experiment 2] Multi-Seed & Task-Order Robustness Benchmark (3 Seeds × Forward + Reverse Order).
동일한 프로토콜(epochs_t0=10, epochs_slot=12/cat=16, batch_size=64, max_iters=100, exec_steps=16,
T=0.05, threshold=0.85)에서 Autonomous ABK-VLA (Model D)를:
  1) Forward order: apple -> sprout -> star -> cat (seeds 42, 43, 44)
  2) Reverse order: cat -> star -> sprout -> apple (seeds 42, 43, 44)
에 대해 학습·평가하여 평균 ± 표준편차를 산출하고, 각 체크포인트에서
Select-then-Blend(T=0.05)와 Hard Argmax 라우팅을 동시에 비교합니다.

사용법:
  python experiments/benchmark_multiseed_order.py --seeds 42 43 44 --orders forward reverse
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "v1_drawing_vla" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from experiments.benchmark_autonomous_cl import (
    calc_cl_metrics,
    eval_episode_rollout,
    set_seed,
    train_one_task,
)
from experiments.benchmark_three_models import pick_device, prepare_task_loaders, safe_savefig
from tools.bspline_augment import BSplineGeometricAugmentor
from train.bspline_policy import AutonomousABKPolicy

OUT_DIR = PROJECT_ROOT / "outputs" / "v1_archive"
FORWARD_TASKS = ["apple", "sprout", "star", "cat"]
REVERSE_TASKS = ["cat", "star", "sprout", "apple"]


def run_single_model_d(
    task_names: List[str],
    seed: int,
    device: torch.device,
    epochs_t0: int = 10,
    epochs_slot: int = 12,
    batch_size: int = 64,
    max_iters: int = 100,
    exec_steps: int = 16,
    num_modes: int = 6,
    temperature: float = 0.05,
    novelty_threshold: float = 0.85,
) -> Dict:
    """
    단일 (task_order, seed) 조합에 대해 AutonomousABKPolicy(Model D)를 순차 학습·평가.
    - 태스크 s=0 학습 후 인코더/상태망/슬롯0을 영구 동결(94.44% frozen).
    - 이후 태스크 s>=1 도착 시 코사인 유사도 < tau(0.85)이면 자율적으로 신규 Residual WTA 슬롯 확장.
    - 매 단계 및 최종 시점에서 Select-then-Blend(T=0.05)와 Hard Argmax R 행렬 (K, K) 동시 계측.
    """
    set_seed(seed)
    K = len(task_names)

    loaders, targets_info = [], []
    for name in task_names:
        cache_path = PROJECT_ROOT / "data" / "bspline" / f"bspline_vla_dataset_{name}_m6_h16_s8.npz"
        tr_l, val_l, tgt_img = prepare_task_loaders(cache_path, batch_size=batch_size, seed=seed)
        loaders.append((tr_l, val_l))
        targets_info.append((name, tgt_img))

    augmentor = BSplineGeometricAugmentor(num_control_points=6, horizon=16).to(device)
    model = AutonomousABKPolicy(
        num_control_points=6,
        horizon=16,
        num_modes=num_modes,
        temperature=temperature,
    ).to(device)

    R_blend = np.zeros((K, K), dtype=np.float32)
    R_argmax = np.zeros((K, K), dtype=np.float32)
    expansion_logs = []
    t_start = time.time()

    for step_idx, (t_name, tgt_img) in enumerate(targets_info):
        tr_loader, _ = loaders[step_idx]
        tgt_tensor = torch.from_numpy(tgt_img.astype(np.float32) / 255.0)[None, None].to(device)

        s_idx, expanded, max_sim = model.register_or_expand_slot(
            tgt_tensor, novelty_threshold=novelty_threshold
        )
        expansion_logs.append(
            {
                "step": step_idx,
                "task": t_name,
                "slot_idx": int(s_idx),
                "expanded": bool(expanded),
                "max_prior_sim": float(max_sim),
            }
        )

        n_train = len(tr_loader.dataset)
        if step_idx == 0:
            ep = epochs_t0
            rep = 1
            p_aff, rot_d, sh_l, sc_l = 0.60, 6.0, 0.08, 0.10
            lr_step = 3e-4
        else:
            ep = max(epochs_slot, 16) if t_name == "cat" else epochs_slot
            rep = 5 if (n_train < 1500 or t_name == "cat") else (4 if n_train < 2500 else 3)
            p_aff, rot_d, sh_l, sc_l = 0.45, 4.0, 0.04, 0.06
            lr_step = 1e-3

        train_one_task(
            model=model,
            augmentor=augmentor,
            train_loader=tr_loader,
            device=device,
            epochs=ep,
            lr=lr_step,
            repeat=rep,
            p_affine=p_aff,
            rot_deg=rot_d,
            shift_lim=sh_l,
            scale_lim=sc_l,
            desc=f"seed={seed} | {'->'.join(task_names[:2])} | T{step_idx}:{t_name}",
            task_id=None,
            replay_buf=None,
        )

        if step_idx == 0:
            model.refresh_slot_key(0, tgt_tensor)
            model.freeze_backbone_and_prior_slots(up_to_exclusive=1)

        # Evaluate seen tasks under Select-then-Blend (and Hard Argmax on final step + diagonal)
        for eval_idx in range(step_idx + 1):
            e_name, e_tgt = targets_info[eval_idx]
            model.routing_mode = "blend"
            res_b, _ = eval_episode_rollout(
                model, e_tgt, device=device, task_name=e_name, task_id=None, max_iters=max_iters, exec_steps=exec_steps
            )
            R_blend[eval_idx, step_idx] = res_b["f1"]

            if eval_idx == step_idx or step_idx == K - 1:
                model.routing_mode = "argmax"
                res_a, _ = eval_episode_rollout(
                    model, e_tgt, device=device, task_name=e_name, task_id=None, max_iters=max_iters, exec_steps=exec_steps
                )
                R_argmax[eval_idx, step_idx] = res_a["f1"]

    # Fill upper-triangle unseen diagonals for argmax
    for step_idx in range(1, K - 1):
        for eval_idx in range(step_idx):
            R_argmax[eval_idx, step_idx] = R_argmax[eval_idx, eval_idx]

    # Evaluate final details
    final_blend, final_argmax = {}, {}
    for eval_idx, (e_name, e_tgt) in enumerate(targets_info):
        model.routing_mode = "blend"
        rb, _ = eval_episode_rollout(model, e_tgt, device=device, task_name=e_name, max_iters=max_iters, exec_steps=exec_steps)
        model.routing_mode = "argmax"
        ra, _ = eval_episode_rollout(model, e_tgt, device=device, task_name=e_name, max_iters=max_iters, exec_steps=exec_steps)
        final_blend[e_name] = rb
        final_argmax[e_name] = ra
        R_blend[eval_idx, K - 1] = rb["f1"]
        R_argmax[eval_idx, K - 1] = ra["f1"]

    # Routing matrix
    model.eval()
    with torch.no_grad():
        tgt_stack = torch.cat(
            [torch.from_numpy(t_img.astype(np.float32) / 255.0)[None, None] for _, t_img in targets_info], dim=0
        ).to(device)
        rw = model.compute_routing_weights(tgt_stack).detach().cpu().numpy()

    cl_blend = calc_cl_metrics(R_blend)
    cl_argmax = calc_cl_metrics(R_argmax)
    diag_conf = float(np.mean(np.diag(rw)))
    top1_acc = float(np.mean(np.argmax(rw, axis=1) == np.arange(K)))

    return {
        "seed": seed,
        "tasks": task_names,
        "elapsed_sec": float(time.time() - t_start),
        "routing_diag_mean": diag_conf,
        "routing_top1_acc": top1_acc,
        "routing_matrix": rw.tolist(),
        "expansion_logs": expansion_logs,
        "blend": {
            "summary": cl_blend,
            "avg_chamfer": float(np.mean([final_blend[t]["chamfer_dist"] for t in task_names])),
            "avg_nmj": float(np.mean([final_blend[t]["normalized_jerk"] for t in task_names])),
            "avg_component_recall": float(np.mean([final_blend[t]["component_recall"] for t in task_names])),
            "avg_episode_infer_ms": float(np.mean([final_blend[t]["episode_infer_ms"] for t in task_names])),
            "per_task_f1": {t: final_blend[t]["f1"] for t in task_names},
            "R": R_blend.tolist(),
        },
        "argmax": {
            "summary": cl_argmax,
            "avg_chamfer": float(np.mean([final_argmax[t]["chamfer_dist"] for t in task_names])),
            "avg_nmj": float(np.mean([final_argmax[t]["normalized_jerk"] for t in task_names])),
            "avg_component_recall": float(np.mean([final_argmax[t]["component_recall"] for t in task_names])),
            "avg_episode_infer_ms": float(np.mean([final_argmax[t]["episode_infer_ms"] for t in task_names])),
            "per_task_f1": {t: final_argmax[t]["f1"] for t in task_names},
            "R": R_argmax.tolist(),
        },
    }


def aggregate_runs(runs: List[Dict], mode_key: str) -> Dict:
    f1s = [r[mode_key]["summary"]["avg_final_f1"] * 100 for r in runs]
    frs = [r[mode_key]["summary"]["avg_forgetting"] * 100 for r in runs]
    bwts = [r[mode_key]["summary"]["avg_bwt"] * 100 for r in runs]
    cds = [r[mode_key]["avg_chamfer"] for r in runs]
    nmjs = [r[mode_key]["avg_nmj"] for r in runs]
    crs = [r[mode_key]["avg_component_recall"] * 100 for r in runs]
    lats = [r[mode_key]["avg_episode_infer_ms"] for r in runs]
    diag = [r["routing_diag_mean"] * 100 for r in runs]
    return {
        "n_seeds": len(runs),
        "avg_f1_mean_std": [float(np.mean(f1s)), float(np.std(f1s, ddof=0))],
        "fr_mean_std": [float(np.mean(frs)), float(np.std(frs, ddof=0))],
        "bwt_mean_std": [float(np.mean(bwts)), float(np.std(bwts, ddof=0))],
        "cd_mean_std": [float(np.mean(cds)), float(np.std(cds, ddof=0))],
        "nmj_mean_std": [float(np.mean(nmjs)), float(np.std(nmjs, ddof=0))],
        "cr_mean_std": [float(np.mean(crs)), float(np.std(crs, ddof=0))],
        "lat_mean_std": [float(np.mean(lats)), float(np.std(lats, ddof=0))],
        "routing_diag_mean_std": [float(np.mean(diag)), float(np.std(diag, ddof=0))],
    }


def save_outputs(all_data: Dict, out_json: Path, out_png: Path) -> None:
    out_json.write_text(json.dumps(all_data, indent=2, ensure_ascii=False), encoding="utf-8")

    if not all_data.get("aggregates"):
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=200)
    fig.patch.set_facecolor("#F8FAFC")

    entries = []
    for order_name in ("forward", "reverse"):
        if order_name in all_data["aggregates"]:
            for rmode in ("blend", "argmax"):
                entries.append((f"{order_name}\n({rmode})", all_data["aggregates"][order_name][rmode]))

    if not entries:
        plt.close()
        return

    labels = [e[0] for e in entries]
    x = np.arange(len(labels))
    cols = ["#2563EB", "#16A34A", "#7C3AED", "#0D9488"][: len(labels)]

    for ax, metric, title, ylim, pad in [
        (axes[0], "avg_f1_mean_std", "Average Final F1 (%) ↑", (0, 102), 1.8),
        (axes[1], "fr_mean_std", "Forgetting Rate (%) ↓", (0, 20), 0.7),
        (axes[2], "routing_diag_mean_std", "Diagonal Routing Conf. (%) ↑", (90, 102.2), 0.35),
    ]:
        means = [e[1][metric][0] for e in entries]
        stds = [e[1][metric][1] for e in entries]
        ax.bar(x, means, yerr=stds, capsize=5, color=cols, alpha=0.92)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for i, (m, s) in enumerate(zip(means, stds)):
            y_txt = min(ylim[1] - pad * 0.7, m + s + pad)
            ax.text(i, y_txt, f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="medium")

    plt.suptitle("Experiment 2: Multi-Seed & Task-Order Robustness (Mean ± Std)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    safe_savefig(out_png)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--orders", nargs="+", choices=["forward", "reverse"], default=["forward", "reverse"])
    parser.add_argument("--epochs-t0", type=int, default=10)
    parser.add_argument("--epochs-slot", type=int, default=12)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    out_json = OUT_DIR / "exp2_multiseed_order.json"
    out_png = OUT_DIR / "exp2_multiseed_order.png"

    data = {"runs": {"forward": [], "reverse": []}, "aggregates": {}}
    if out_json.exists():
        try:
            loaded = json.loads(out_json.read_text(encoding="utf-8"))
            if "runs" in loaded:
                data["runs"].update(loaded["runs"])
        except Exception:
            pass

    for order in args.orders:
        tasks = FORWARD_TASKS if order == "forward" else REVERSE_TASKS
        existing_seeds = {r["seed"] for r in data["runs"][order]}
        for sd in args.seeds:
            if sd in existing_seeds:
                print(f"[Skip] {order} seed={sd} already cached in {out_json.name}")
                continue
            print(f"\n=== Running {order.upper()} | seed={sd} | tasks={tasks} ===")
            res = run_single_model_d(
                tasks,
                seed=sd,
                device=device,
                epochs_t0=args.epochs_t0,
                epochs_slot=args.epochs_slot,
            )
            data["runs"][order].append(res)
            print(
                f"  -> [DONE {order} seed={sd}] Blend F1={res['blend']['summary']['avg_final_f1']*100:.2f}% "
                f"FR={res['blend']['summary']['avg_forgetting']*100:.2f}% | "
                f"Argmax F1={res['argmax']['summary']['avg_final_f1']*100:.2f}% "
                f"FR={res['argmax']['summary']['avg_forgetting']*100:.2f}% | "
                f"Routing={res['routing_diag_mean']*100:.2f}%"
            )
            # Incremental checkpoint save
            for ord_k in ("forward", "reverse"):
                if data["runs"][ord_k]:
                    data["aggregates"][ord_k] = {
                        "blend": aggregate_runs(data["runs"][ord_k], "blend"),
                        "argmax": aggregate_runs(data["runs"][ord_k], "argmax"),
                    }
            save_outputs(data, out_json, out_png)

    for ord_k in ("forward", "reverse"):
        if data["runs"][ord_k]:
            data["aggregates"][ord_k] = {
                "blend": aggregate_runs(data["runs"][ord_k], "blend"),
                "argmax": aggregate_runs(data["runs"][ord_k], "argmax"),
            }
    save_outputs(data, out_json, out_png)
    print("\n=== Experiment 2 Aggregates ===")
    print(json.dumps(data["aggregates"], indent=2))


if __name__ == "__main__":
    main()
