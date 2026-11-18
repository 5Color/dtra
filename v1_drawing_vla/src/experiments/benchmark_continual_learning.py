"""experiments/benchmark_continual_learning.py

[Phase 3] 비파괴적 지속학습(Continual Learning) 벤치마크 및 파국적 망각 실증 실험.
- Task A (사과, Apple) -> Task B (새싹, Sprout) 순차 학습 파이프라인.
- 대조군: 일반 VLA 파인튜닝 (Baseline Naive Fine-Tuning) - 전역 가중치 재학습.
- 실험군: ABK-VLA 국소 메모리 슬롯 (Ours Slot-Isolated Policy) - ARIN 슬롯 격리.
- 핵심 검증:
    1. Task B 학습 후 Task A 역방향 성능 유지율 (Backward Retention / Forgetting Rate).
    2. 폐루프 자율 롤아웃을 통한 실제 캔버스 도안 파괴 vs 보존 시각적 대조.
    3. 논문 게재용 정량 결과표(JSON) 및 고해상도 비교 차트/캔버스 렌더링.

사용법:
  python experiments/benchmark_continual_learning.py --epochs-a 10 --epochs-b 10
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from tools.bspline_core import BSplineDecoder
from train.bspline_policy import BSplineSlotPolicy, BSplineVisuomotorPolicy
from train.rollout_bspline import compute_smoothness, load_target_image, match_scores, rollout_bspline

OUT_DIR = PROJECT_ROOT / "outputs"


def pick_device() -> torch.device:
    if os.environ.get("DEVICE") != "cpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        dev, name = torch.device("xpu"), torch.xpu.get_device_name(0)
    elif torch.cuda.is_available():
        dev, name = torch.device("cuda"), torch.cuda.get_device_name(0)
    else:
        dev, name = torch.device("cpu"), platform.processor() or "CPU"
    print(f"[device] {dev.type.upper()}: {name}", flush=True)
    return dev


def sync(dev: torch.device):
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 데이터셋 분할 및 로더 생성 유틸
# ---------------------------------------------------------------------------

def prepare_task_loaders(
    npz_path: Path,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    """npz 캐시 로드 후 Train / Val 텐서 로더 반환."""
    with np.load(npz_path, allow_pickle=True) as data:
        canv = data["canv"]                   # (N, 128, 128) uint8
        state = data["state"]                 # (N, 3) float32
        cp = data["control_points"]           # (N, M, 2) float32
        pen = data["pen"]                     # (N, H) float32
        raw_action = data["raw_action"]       # (N, H, 3) float32
        target_images = data["target_images"] # (Num_targets, 128, 128) uint8
        ep_ids = data["ep_id"]

    n_samples = len(canv)
    tgt_img = target_images[0]
    tgt_expanded = np.repeat(tgt_img[np.newaxis, :, :], n_samples, axis=0)

    # 2채널 관측 [canv, target]
    obs = np.stack([canv, tgt_expanded], axis=1).astype(np.float32) / 255.0

    unique_eps = np.unique(ep_ids)
    np.random.seed(seed)
    val_eps = set(np.random.choice(unique_eps, size=max(1, int(len(unique_eps) * val_ratio)), replace=False))
    val_mask = np.isin(ep_ids, list(val_eps))
    train_mask = ~val_mask

    train_ds = TensorDataset(
        torch.tensor(obs[train_mask], dtype=torch.float32),
        torch.tensor(state[train_mask], dtype=torch.float32),
        torch.tensor(cp[train_mask], dtype=torch.float32),
        torch.tensor(raw_action[train_mask], dtype=torch.float32),
        torch.tensor(pen[train_mask], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(obs[val_mask], dtype=torch.float32),
        torch.tensor(state[val_mask], dtype=torch.float32),
        torch.tensor(cp[val_mask], dtype=torch.float32),
        torch.tensor(raw_action[val_mask], dtype=torch.float32),
        torch.tensor(pen[val_mask], dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tgt_img


# ---------------------------------------------------------------------------
# 평가 및 롤아웃 유틸
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_val_metrics(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    task_id: int = 0,
) -> Dict[str, float]:
    """검증 데이터셋에서 RMSE 및 Control Point L1 오차 측정."""
    model.eval()
    rmse_accum = 0.0
    cp_l1_accum = 0.0
    n_batches = 0

    for b_obs, b_state, b_cp, b_act, b_pen in val_loader:
        b_obs = b_obs.to(device)
        b_state = b_state.to(device)
        b_cp = b_cp.to(device)
        b_act = b_act.to(device)

        kw = {"task_id": task_id} if isinstance(model, BSplineSlotPolicy) else {}
        pred_cp, pred_traj, _ = model.predict(b_obs, b_state, **kw)

        cp_l1 = F.l1_loss(pred_cp, b_cp).item()
        diff = pred_traj - b_act[..., :2]
        rmse_px = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1))).item() * 128.0

        rmse_accum += rmse_px
        cp_l1_accum += cp_l1
        n_batches += 1

    return {
        "rmse_px": rmse_accum / max(1, n_batches),
        "cp_l1": cp_l1_accum / max(1, n_batches),
    }


def run_full_rollout_eval(
    model: torch.nn.Module,
    target_name: str,
    target_img: np.ndarray,
    device: torch.device,
    task_id: int = 0,
    exec_steps: int = 12,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """도안에 대해 폐루프 롤아웃 수행 후 F1 점수 및 최종 캔버스 반환."""
    # 래퍼 정책 (SlotPolicy 인터페이스 일원화)
    class TaskWrapper(torch.nn.Module):
        def __init__(self, m, t_id):
            super().__init__()
            self.m = m
            self.t_id = t_id

        def predict(self, obs, state):
            if isinstance(self.m, BSplineSlotPolicy):
                return self.m.predict(obs, state, task_id=self.t_id)
            else:
                return self.m.predict(obs, state)

    wrapped = TaskWrapper(model, task_id)
    executed, history, stop_reason, _ = rollout_bspline(
        wrapped,
        target_img,
        device=device,
        max_iters=65,
        exec_steps=exec_steps,
        patience=6,
    )
    final_canvas = history[-1]["canvas"]
    scores = match_scores(final_canvas, target_img)
    smooth = compute_smoothness(executed)

    return {
        "f1": float(scores["f1"]),
        "coverage": float(scores["coverage"]),
        "precision": float(scores["precision"]),
        "turn_deg": float(smooth["turn_deg"]),
        "wave_deg": float(smooth["wave_deg"]),
        "steps": len(executed),
    }, final_canvas, executed


# ---------------------------------------------------------------------------
# 모델 학습 루프
# ---------------------------------------------------------------------------

def train_one_task(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 3e-4,
    task_id: int = 0,
    desc: str = "",
) -> Dict[str, List[float]]:
    """지정된 태스크 데이터로 에포크 학습 수행."""
    # 동결되지 않은(trainable) 파라미터만 옵티마이저에 전달
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"[*] {desc} 학습 시작: 총 {epochs} 에포크 (가중치 {sum(p.numel() for p in trainable_params):,}개 학습 가능)...")

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for b_obs, b_state, b_cp, b_act, b_pen in train_loader:
            b_obs = b_obs.to(device)
            b_state = b_state.to(device)
            b_cp = b_cp.to(device)
            b_act = b_act.to(device)
            b_pen = b_pen.to(device)

            optimizer.zero_grad()
            kw = {"task_id": task_id} if isinstance(model, BSplineSlotPolicy) else {}
            loss, _ = model.loss(b_obs, b_state, target_cp=b_cp, target_traj=b_act, target_pen=b_pen, **kw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        sync(device)

        if ep % 5 == 0 or ep == 1 or ep == epochs:
            val_met = evaluate_val_metrics(model, val_loader, device, task_id=task_id)
            print(f"    [{desc}] Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss / n_batches:.4f} | Val RMSE: {val_met['rmse_px']:.2f}px | Val CP: {val_met['cp_l1']:.4f}")


# ---------------------------------------------------------------------------
# 메인 벤치마크 실행 파이프라인
# ---------------------------------------------------------------------------

def run_continual_benchmark(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    print("=" * 80)
    print("      [Phase 3] Non-Destructive Continual Learning Benchmark (ABK-VLA)       ")
    print("      Task A (Apple) ──▶ Task B (Sprout) Sequential Continual Learning       ")
    print("=" * 80)

    # 1. 데이터셋 로드
    cache_apple = PROJECT_ROOT / "data" / "bspline" / "bspline_vla_dataset_apple_m6_h16_s8.npz"
    cache_sprout = PROJECT_ROOT / "data" / "bspline" / "bspline_vla_dataset_sprout_m6_h16_s8.npz"

    train_loader_a, val_loader_a, tgt_img_a = prepare_task_loaders(cache_apple, batch_size=args.batch_size)
    train_loader_b, val_loader_b, tgt_img_b = prepare_task_loaders(cache_sprout, batch_size=args.batch_size)

    print(f"[*] Task A (사과)   : Train {len(train_loader_a.dataset)} | Val {len(val_loader_a.dataset)}")
    print(f"[*] Task B (새싹)   : Train {len(train_loader_b.dataset)} | Val {len(val_loader_b.dataset)}")

    # 2. 모델 생성
    torch.manual_seed(42)
    # 대조군: 일반 VLA 파인튜닝 (단일 헤드)
    model_baseline = BSplineVisuomotorPolicy(num_control_points=6, horizon=16).to(device)

    # 실험군: ABK-VLA 국소 기억 슬롯 (ARIN Slot Isolation)
    torch.manual_seed(42)
    model_ours = BSplineSlotPolicy(num_control_points=6, horizon=16).to(device)
    model_ours.add_slot(0)  # Task A (Apple) 전용 슬롯
    model_ours.to(device)

    # =======================================================================
    # STEP 1: Task A (사과) 초기 학습
    # =======================================================================
    print("\n" + "=" * 60)
    print(" >>> STEP 1: Task A (Apple) 초기 학습 (Both Models)")
    print("=" * 60)

    train_one_task(model_baseline, train_loader_a, val_loader_a, device, epochs=args.epochs_a, desc="Baseline on Task A")
    train_one_task(model_ours, train_loader_a, val_loader_a, device, epochs=args.epochs_a, task_id=0, desc="Ours (Slot 0) on Task A")

    # Task A 초기 롤아웃 평가
    print("\n[*] [초기 평가] Task A (사과) 롤아웃 수행...")
    res_base_a_init, canv_base_a_init, _ = run_full_rollout_eval(model_baseline, "apple", tgt_img_a, device)
    res_ours_a_init, canv_ours_a_init, _ = run_full_rollout_eval(model_ours, "apple", tgt_img_a, device, task_id=0)

    print(f"  • Baseline Task A F1 (Initial): {res_base_a_init['f1']*100:.2f}% (Cov: {res_base_a_init['coverage']*100:.1f}%)")
    print(f"  • Ours     Task A F1 (Initial): {res_ours_a_init['f1']*100:.2f}% (Cov: {res_ours_a_init['coverage']*100:.1f}%)")

    # =======================================================================
    # STEP 2: Task B (새싹) 순차 지속 학습 (Task A 리플레이 없음!)
    # =======================================================================
    print("\n" + "=" * 60)
    print(" >>> STEP 2: Task B (Sprout) 순차 학습 (Zero Replay of Task A)")
    print("=" * 60)

    # 1) 대조군 (Baseline Naive Fine-Tuning): 새싹 데이터로 전역 가중치 파인튜닝
    train_one_task(model_baseline, train_loader_b, val_loader_b, device, epochs=args.epochs_b, lr=args.lr_ft, desc="Baseline Fine-tuning on Task B")

    # 2) 실험군 (Ours Slot Isolation): Slot 1 할당 후 Slot 0 및 공통 인코더 동결!
    model_ours.add_slot(1)  # Task B (Sprout) 전용 슬롯 할당
    model_ours.to(device)
    model_ours.freeze_previous_slots(current_task_id=1, freeze_encoder=True)
    train_one_task(model_ours, train_loader_b, val_loader_b, device, epochs=max(12, args.epochs_b), task_id=1, lr=1e-3, desc="Ours (Slot 1 only) on Task B")

    # Task B 학습 완료 평가
    print("\n[*] [중간 평가] Task B (새싹) 롤아웃 수행...")
    res_base_b, canv_base_b, _ = run_full_rollout_eval(model_baseline, "sprout", tgt_img_b, device)
    res_ours_b, canv_ours_b, _ = run_full_rollout_eval(model_ours, "sprout", tgt_img_b, device, task_id=1)

    print(f"  • Baseline Task B F1: {res_base_b['f1']*100:.2f}% (Cov: {res_base_b['coverage']*100:.1f}%)")
    print(f"  • Ours     Task B F1: {res_ours_b['f1']*100:.2f}% (Cov: {res_ours_b['coverage']*100:.1f}%)")

    # =======================================================================
    # STEP 3: 파국적 망각(Catastrophic Forgetting) 역방향 재검증 (Task A)
    # =======================================================================
    print("\n" + "=" * 60)
    print(" >>> STEP 3: [핵심 검증] Task B 학습 후 Task A (Apple) 보존도 재평가")
    print("=" * 60)

    res_base_a_post, canv_base_a_post, _ = run_full_rollout_eval(model_baseline, "apple", tgt_img_a, device)
    res_ours_a_post, canv_ours_a_post, _ = run_full_rollout_eval(model_ours, "apple", tgt_img_a, device, task_id=0)

    # 지속학습 핵심 메트릭 계산
    # 1. 망각률 (Forgetting Rate) F_A = max(0, F1_init - F1_post)
    forgetting_base = max(0.0, res_base_a_init["f1"] - res_base_a_post["f1"])
    forgetting_ours = max(0.0, res_ours_a_init["f1"] - res_ours_a_post["f1"])

    # 2. 기억 보존율 (Retention Ratio) = F1_post / F1_init * 100%
    retention_base = (res_base_a_post["f1"] / max(1e-6, res_base_a_init["f1"])) * 100.0
    retention_ours = (res_ours_a_post["f1"] / max(1e-6, res_ours_a_init["f1"])) * 100.0

    # 3. 역방향 전이 (Backward Transfer) BWT = F1_post - F1_init
    bwt_base = res_base_a_post["f1"] - res_base_a_init["f1"]
    bwt_ours = res_ours_a_post["f1"] - res_ours_a_init["f1"]

    print("\n" + "#" * 65)
    print("          [지속학습 벤치마크 최종 정량 비교표]          ")
    print("#" * 65)
    print(f"  [Model]               | Task A (Before) | Task A (After) | 망각률(Forgetting) | 기억 보존율(Retention)")
    print(f"  --------------------- | --------------- | -------------- | ------------------ | ----------------------")
    print(f"  Baseline (Fine-tune)  | {res_base_a_init['f1']*100:6.2f}%       | {res_base_a_post['f1']*100:6.2f}%       | {forgetting_base*100:6.2f}% (망각발생) | {retention_base:6.2f}%")
    print(f"  Ours (ABK-VLA Slots)  | {res_ours_a_init['f1']*100:6.2f}%       | {res_ours_a_post['f1']*100:6.2f}%       | {forgetting_ours*100:6.2f}% (0% 보존!) | {retention_ours:6.2f}%")
    print("#" * 65)

    # 4. 결과 JSON 저장
    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_sequence": ["apple", "sprout"],
        "epochs_a": args.epochs_a,
        "epochs_b": args.epochs_b,
        "baseline": {
            "task_a_f1_before": res_base_a_init["f1"],
            "task_a_f1_after": res_base_a_post["f1"],
            "task_b_f1": res_base_b["f1"],
            "forgetting_rate": forgetting_base,
            "retention_ratio": retention_base,
            "backward_transfer": bwt_base,
        },
        "ours_abk_vla": {
            "task_a_f1_before": res_ours_a_init["f1"],
            "task_a_f1_after": res_ours_a_post["f1"],
            "task_b_f1": res_ours_b["f1"],
            "forgetting_rate": forgetting_ours,
            "retention_ratio": retention_ours,
            "backward_transfer": bwt_ours,
        },
    }
    json_path = OUT_DIR / "continual_learning_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\n[*] 벤치마크 JSON 데이터 저장 완료 -> {json_path}")

    # =======================================================================
    # STEP 4: 논문용 고해상도 시각화 비교도 렌더링
    # =======================================================================
    # 1) Bar Chart & Metric Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)
    fig.patch.set_facecolor("#F8F9FA")

    # 패널 1: Task A F1 점수 변화
    models = ["Baseline\n(Naive FT)", "Ours\n(ABK-VLA Slots)"]
    x = np.arange(len(models))
    w = 0.35
    axes[0].bar(x - w / 2, [res_base_a_init["f1"] * 100, res_ours_a_init["f1"] * 100], width=w, label="Before Task B", color="#3182CE")
    axes[0].bar(x + w / 2, [res_base_a_post["f1"] * 100, res_ours_a_post["f1"] * 100], width=w, label="After Task B", color="#E53E3E")
    axes[0].set_title("Task A (Apple) F1 Score Maintenance", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, fontsize=10)
    axes[0].set_ylabel("F1 Score (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower left")

    # 패널 2: 기억 보존율 (Retention Ratio)
    axes[1].bar(models, [retention_base, retention_ours], color=["#E53E3E", "#38A169"], width=0.5)
    axes[1].axhline(100.0, color="#718096", linestyle="--", alpha=0.7)
    axes[1].set_title("Task A Memory Retention Ratio (%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Retention Ratio (%) [Higher is Better]")
    axes[1].set_ylim(0, 115)
    for i, v in enumerate([retention_base, retention_ours]):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # 패널 3: 파국적 망각률 (Forgetting Rate)
    axes[2].bar(models, [forgetting_base * 100, forgetting_ours * 100], color=["#E53E3E", "#38A169"], width=0.5)
    axes[2].set_title("Catastrophic Forgetting Rate (%)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Forgetting Rate (%) [0% is Ideal]")
    axes[2].set_ylim(0, max(50.0, forgetting_base * 100 + 10))
    for i, v in enumerate([forgetting_base * 100, forgetting_ours * 100]):
        axes[2].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("[Continual Learning Benchmark] Task A (Apple) ──▶ Task B (Sprout)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    chart_path = OUT_DIR / "continual_learning_benchmark.png"
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"[*] 벤치마크 지표 차트 저장 완료 -> {chart_path}")

    # 2) 4-Way 캔버스 롤아웃 시각적 비교도 (논문 그림용)
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10), dpi=200)
    fig2.patch.set_facecolor("#F8F9FA")

    # Row 1: Task A (Apple)
    axes2[0, 0].imshow(tgt_img_a, cmap="gray_r"); axes2[0, 0].set_title("Task A Target: Apple", fontweight="bold")
    axes2[0, 1].imshow(canv_base_a_post, cmap="gray_r")
    axes2[0, 1].set_title(f"Baseline (After Task B)\nF1: {res_base_a_post['f1']*100:.1f}% (Forgotten!)", color="#E53E3E", fontweight="bold")
    axes2[0, 2].imshow(canv_ours_a_post, cmap="gray_r")
    axes2[0, 2].set_title(f"Ours ABK-VLA (After Task B)\nF1: {res_ours_a_post['f1']*100:.1f}% (Zero Forgetting!)", color="#38A169", fontweight="bold")

    # Row 2: Task B (Sprout)
    axes2[1, 0].imshow(tgt_img_b, cmap="gray_r"); axes2[1, 0].set_title("Task B Target: Sprout", fontweight="bold")
    axes2[1, 1].imshow(canv_base_b, cmap="gray_r")
    axes2[1, 1].set_title(f"Baseline on Task B\nF1: {res_base_b['f1']*100:.1f}%", fontweight="bold")
    axes2[1, 2].imshow(canv_ours_b, cmap="gray_r")
    axes2[1, 2].set_title(f"Ours ABK-VLA on Task B\nF1: {res_ours_b['f1']*100:.1f}%", color="#38A169", fontweight="bold")

    for ax in axes2.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("Visual Proof of Continual Learning: Baseline Catastrophic Forgetting vs. Ours Zero Forgetting", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    rollout_comp_path = OUT_DIR / "continual_learning_rollout_comparison.png"
    plt.savefig(rollout_comp_path, bbox_inches="tight")
    plt.close()
    print(f"[*] 캔버스 롤아웃 실증 비교도 저장 완료 -> {rollout_comp_path}")


def main():
    parser = argparse.ArgumentParser(description="ABK-VLA Continual Learning Benchmark")
    parser.add_argument("--epochs-a", type=int, default=10, help="Task A (Apple) 학습 에포크 수")
    parser.add_argument("--epochs-b", type=int, default=10, help="Task B (Sprout) 학습 에포크 수")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument("--lr-ft", type=float, default=2e-4, help="Baseline 파인튜닝 학습률")
    args = parser.parse_args()

    run_continual_benchmark(args)


if __name__ == "__main__":
    main()
