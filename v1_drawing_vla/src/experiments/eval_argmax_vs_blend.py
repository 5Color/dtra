"""experiments/eval_argmax_vs_blend.py

[Experiment 3] Select-then-Blend (Softmax T sweep) vs Hard Argmax Slot Routing 비교 실험.
기존 학습된 AutonomousABKPolicy 체크포인트(outputs/autonomous_model_d_final.pth)를 사용하여
동일한 4-Task 순차 평가 기준에서 tail-weight(off-diagonal <0.04%) 혼입 제거 시
망각률(FR), F1, Chamfer Distance, Jerk, Component Recall, 추론 지연시간(Latency) 변화를 검증합니다.

사용법:
  python experiments/eval_argmax_vs_blend.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from experiments.benchmark_autonomous_cl import (
    calc_cl_metrics,
    eval_episode_rollout,
    set_seed,
)
from experiments.benchmark_three_models import pick_device, safe_savefig
from train.bspline_policy import AutonomousABKPolicy

OUT_DIR = PROJECT_ROOT / "outputs"


def load_canonical_targets(tasks: List[str]) -> Dict[str, np.ndarray]:
    """각 태스크의 대표 목표 도안 이미지 (128, 128) uint8 로드."""
    targets = {}
    for t in tasks:
        npz_path = PROJECT_ROOT / "data" / "bspline" / f"bspline_vla_dataset_{t}_m6_h16_s8.npz"
        with np.load(npz_path, allow_pickle=True) as data:
            targets[t] = data["target_images"][0]
    return targets


def eval_policy_mode(
    model: AutonomousABKPolicy,
    targets: Dict[str, np.ndarray],
    tasks: List[str],
    r_diag: List[float],
    device: torch.device,
    routing_mode: str,
    temperature: float,
    max_iters: int = 100,
    exec_steps: int = 16,
) -> Dict:
    """
    동일 체크포인트에서 라우팅 모드('blend' vs 'argmax') 및 Softmax 온도 T만 바꿔
    전 태스크 폐루프 롤아웃을 재평가하고 망각률 FR(R[:, -1] vs R[i, i])을 산출.
    - R 행렬: (K, K), 대각원소 R[i,i]는 학습 직후 F1, R[i,K-1]은 최종 시점 F1.
    """
    model.routing_mode = routing_mode
    model.temperature = temperature
    K = len(tasks)

    # Reconstruct R matrix with historical diagonals R[i,i] and newly measured final column R[i,K-1]
    R = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        R[i, i] = r_diag[i]

    per_task = {}
    canvases = {}
    latencies = []
    cds, nmjs, crs = [], [], []
    for i, t_name in enumerate(tasks):
        res, canv = eval_episode_rollout(
            model,
            targets[t_name],
            device=device,
            task_name=t_name,
            task_id=None,
            max_iters=max_iters,
            exec_steps=exec_steps,
            patience=8,
        )
        R[i, K - 1] = res["f1"]
        per_task[t_name] = res
        canvases[t_name] = canv
        latencies.append(res["episode_infer_ms"])
        cds.append(res["chamfer_dist"])
        nmjs.append(res["normalized_jerk"])
        crs.append(res["component_recall"])

    cl = calc_cl_metrics(R)
    return {
        "routing_mode": routing_mode,
        "temperature": float(temperature),
        "avg_final_f1": float(cl["avg_final_f1"]),
        "forgetting_rate": float(cl["avg_forgetting"]),
        "bwt": float(cl["avg_bwt"]),
        "avg_chamfer": float(np.mean(cds)),
        "avg_nmj": float(np.mean(nmjs)),
        "avg_component_recall": float(np.mean(crs)),
        "avg_episode_infer_ms": float(np.mean(latencies)),
        "per_task": per_task,
    }


def plot_exp3(results: Dict[str, dict]) -> Path:
    """저장된 Experiment 3 결과 딕셔너리를 기반으로 비교 막대 그래프를 생성합니다."""
    labels = list(results.keys())
    avg_f1s = [results[k]["avg_final_f1"] * 100 for k in labels]
    frs = [results[k]["forgetting_rate"] * 100 for k in labels]
    lats = [results[k]["avg_episode_infer_ms"] for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=200)
    fig.patch.set_facecolor("#F8FAFC")
    x = np.arange(len(labels))
    colors = ["#64748B", "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#16A34A"]

    axes[0].bar(x, avg_f1s, color=colors)
    axes[0].set_title("Average Final F1 (%) ↑", fontweight="bold", pad=8)
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(avg_f1s):
        axes[0].text(i, v + 1.5, f"{v:.2f}%", ha="center", va="bottom", fontsize=8.5)

    axes[1].bar(x, frs, color=colors)
    axes[1].set_title("Forgetting Rate (%) ↓", fontweight="bold", pad=8)
    axes[1].set_ylim(0, max(frs) * 1.18)
    for i, v in enumerate(frs):
        axes[1].text(i, v + 1.4, f"{v:.2f}%", ha="center", va="bottom", fontsize=8.5)

    axes[2].bar(x, lats, color=colors)
    axes[2].set_title("Episode Latency (ms) ↓", fontweight="bold", pad=8)
    axes[2].set_ylim(0, max(lats) * 1.18)
    for i, v in enumerate(lats):
        axes[2].text(i, v + 12, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8.5)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Experiment 3: Select-then-Blend Temperature Sweep vs. Hard Argmax Routing", fontweight="bold")
    plt.tight_layout()
    return safe_savefig(OUT_DIR / "exp3_argmax_vs_blend.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: Blend vs. Hard Argmax routing evaluation")
    parser.add_argument("--plot-only", action="store_true", help="Re-generate figure from outputs/exp3_argmax_vs_blend.json")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "exp3_argmax_vs_blend.json"
    if args.plot_only:
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        saved_png = plot_exp3(payload["configs"])
        print(f"[Plot-Only] Regenerated {saved_png} from {out_json}")
        return

    set_seed(42)
    device = pick_device()

    bench_path = OUT_DIR / "autonomous_cl_benchmark.json"
    ckpt_path = OUT_DIR / "autonomous_model_d_final.pth"
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    tasks = bench["tasks"]
    r_mat = np.array(bench["models"]["model_5_auto_d"]["R"], dtype=np.float32)
    r_diag = [float(r_mat[i, i]) for i in range(len(tasks))]

    targets = load_canonical_targets(tasks)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = AutonomousABKPolicy(
        num_control_points=6,
        horizon=16,
        num_modes=6,
        temperature=float(ckpt.get("temperature", 0.05)),
    ).to(device)
    for idx in range(len(tasks)):
        t_tensor = torch.from_numpy(targets[tasks[idx]].astype(np.float32) / 255.0)[None, None].to(device)
        model.register_or_expand_slot(t_tensor, force_expand=(idx > 0))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    configs = [
        ("blend_T0.01", "blend", 0.01),
        ("blend_T0.05 (Default)", "blend", 0.05),
        ("blend_T0.10", "blend", 0.10),
        ("blend_T0.25", "blend", 0.25),
        ("blend_T0.50", "blend", 0.50),
        ("hard_argmax", "argmax", 0.05),
    ]

    results = {}
    print("\n=== Experiment 3: Select-then-Blend vs Hard Argmax Slot Routing ===")
    for name, rmode, temp in configs:
        out = eval_policy_mode(
            model,
            targets,
            tasks,
            r_diag=r_diag,
            device=device,
            routing_mode=rmode,
            temperature=temp,
            max_iters=int(bench["hyperparams"]["max_iters"]),
            exec_steps=16,
        )
        results[name] = out
        f1s = [round(out["per_task"][t]["f1"] * 100, 2) for t in tasks]
        print(
            f"  {name:22s} | Avg F1={out['avg_final_f1']*100:6.2f}% | "
            f"FR={out['forgetting_rate']*100:5.2f}% | CD={out['avg_chamfer']:.4f} | "
            f"CR={out['avg_component_recall']*100:5.1f}% | Lat={out['avg_episode_infer_ms']:5.1f}ms | F1={f1s}"
        )

    out_json.write_text(
        json.dumps({"tasks": tasks, "r_diag": r_diag, "configs": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    saved_png = plot_exp3(results)
    print(f"\n[Done] Saved {out_json} and {saved_png}")


if __name__ == "__main__":
    main()
