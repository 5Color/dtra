"""train/train_bspline_vla.py

[Phase 2] 적응형 B-Spline 매듭점 기반 VLA 정책 네트워크(ABK-VLA) 학습 스크립트.
- 행동 공간: 기존 [16, 3] 시계열에서 고정된 M=6 (12차원) 제어점 + 펜 상태로 압축.
- 파라미터 불변 해석적 B-Spline 디코더(BSplineDecoder)를 통한 종단간 궤적 최적화.
- MLP 회귀 헤드 및 CVAE 헤드 지원 (--head mlp | cvae).
- Intel XPU / CUDA 가속 지원 및 실시간 복원 픽셀 오차(128px 기준) 모니터링.

사용법:
  # 기본 MLP 정책 30 에포크 고속 학습
  python train/train_bspline_vla.py --tag bspline_v1 --epochs 30 --head mlp

  # CVAE 잠재 정책 학습
  python train/train_bspline_vla.py --tag bspline_cvae --epochs 50 --head cvae
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "v1_drawing_vla" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dtra.core.bspline_core import BSplineDecoder
from train.bspline_policy import BSplineCVAEPolicy, BSplineVisuomotorPolicy
from train.dataset import CHUNK_SIZE, OBS_SIZE, DrawingVLADataset

OUT_DIR = PROJECT_ROOT / "outputs" / "v1_archive"


# ---------------------------------------------------------------------------
# 디바이스 가속 (XPU / CUDA / CPU)
# ---------------------------------------------------------------------------

def pick_device() -> torch.device:
    if os.environ.get("DEVICE") != "cpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        dev, name = torch.device("xpu"), torch.xpu.get_device_name(0)
    elif torch.cuda.is_available():
        dev, name = torch.device("cuda"), torch.cuda.get_device_name(0)
    else:
        dev, name = torch.device("cpu"), platform.processor() or "CPU"
    print(f"[device] {dev.type.upper()}: {name}", flush=True)
    return dev


def sync(dev: torch.device) -> None:
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 데이터셋 로드 및 텐서 변환
# ---------------------------------------------------------------------------

def load_or_build_cache(
    cache_path: Path,
    targets: Optional[list[str]] = None,
    num_control_points: int = 6,
    chunk_size: int = 16,
    stride: int = 4,
) -> Dict[str, np.ndarray]:
    """npz 캐시가 있으면 즉시 로드하고 없으면 변환 파이프라인 호출."""
    if cache_path.exists():
        print(f"[data] B-Spline 캐시 로드: {cache_path.name}", flush=True)
        with np.load(cache_path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}

    print(f"[data] 캐시가 없습니다. 데이터셋 실시간 변환 파이프라인 가동...")
    from tools.convert_dataset_to_bspline import convert_dataset_sliding

    ds = DrawingVLADataset(root=PROJECT_ROOT / "data" / "raw", chunk_size=chunk_size, targets=targets)
    converted = convert_dataset_sliding(
        ds,
        num_control_points=num_control_points,
        chunk_size=chunk_size,
        stride=stride,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        canv=converted["canv"],
        state=converted["state"],
        control_points=converted["control_points"],
        pen=converted["pen"],
        raw_action=converted["raw_action"],
        is_pad=converted["is_pad"],
        ep_id=converted["ep_id"],
        step_idx=converted["step_idx"],
        target_images=converted["target_images"],
        target_names=converted["target_names"],
        knots=converted["knots"],
        m=num_control_points,
        h=chunk_size,
    )
    return converted


# ---------------------------------------------------------------------------
# 메인 학습 함수
# ---------------------------------------------------------------------------

def train_bspline_vla(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    device = pick_device()

    # 캐시 경로
    cache_file = PROJECT_ROOT / "data" / "bspline" / f"bspline_vla_dataset_m{args.num_control_points}_h{args.chunk_size}_s{args.stride}.npz"
    data = load_or_build_cache(
        cache_file,
        targets=args.targets,
        num_control_points=args.num_control_points,
        chunk_size=args.chunk_size,
        stride=args.stride,
    )

    canv = data["canv"]                 # (N, 128, 128) uint8
    state = data["state"]               # (N, 3) float32
    cp = data["control_points"]         # (N, M, 2) float32
    pen = data["pen"]                   # (N, H) float32
    raw_action = data["raw_action"]     # (N, H, 3) float32
    target_images = data["target_images"] # (Num_targets, 128, 128) uint8
    ep_ids = data["ep_id"]
    n_samples = len(canv)

    # 타겟 이미지 매핑
    target_obs = target_images[0]  # 단일 타겟 기본
    target_obs_expanded = np.repeat(target_obs[np.newaxis, :, :], n_samples, axis=0)

    # 2채널 관측 텐서 구성: [canv, target] (N, 2, 128, 128)
    obs = np.stack([canv, target_obs_expanded], axis=1).astype(np.float32) / 255.0

    # Train / Val 에피소드 단위 분할 (에피소드 인덱스 기준 85:15 분할)
    unique_eps = np.unique(ep_ids)
    np.random.seed(42)
    val_eps = set(np.random.choice(unique_eps, size=max(1, int(len(unique_eps) * 0.15)), replace=False))
    val_mask = np.isin(ep_ids, list(val_eps))
    train_mask = ~val_mask

    train_obs = torch.tensor(obs[train_mask], dtype=torch.float32)
    train_state = torch.tensor(state[train_mask], dtype=torch.float32)
    train_cp = torch.tensor(cp[train_mask], dtype=torch.float32)
    train_act = torch.tensor(raw_action[train_mask], dtype=torch.float32)
    train_pen = torch.tensor(pen[train_mask], dtype=torch.float32)

    val_obs = torch.tensor(obs[val_mask], dtype=torch.float32)
    val_state = torch.tensor(state[val_mask], dtype=torch.float32)
    val_cp = torch.tensor(cp[val_mask], dtype=torch.float32)
    val_act = torch.tensor(raw_action[val_mask], dtype=torch.float32)
    val_pen = torch.tensor(pen[val_mask], dtype=torch.float32)

    train_ds = TensorDataset(train_obs, train_state, train_cp, train_act, train_pen)
    val_ds = TensorDataset(val_obs, val_state, val_cp, val_act, val_pen)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[*] 데이터 분할 완료: Train {len(train_ds)} 샘플 | Val {len(val_ds)} 샘플 | 배치 크기 {args.batch_size}")

    # 모델 구축
    if args.head == "cvae":
        policy = BSplineCVAEPolicy(
            num_control_points=args.num_control_points,
            horizon=args.chunk_size,
            smoothness_weight=args.smoothness,
        ).to(device)
    else:
        policy = BSplineVisuomotorPolicy(
            num_control_points=args.num_control_points,
            horizon=args.chunk_size,
            smoothness_weight=args.smoothness,
        ).to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"[*] 모델 구성 완료: {policy.__class__.__name__} (파라미터 수: {sum(p.numel() for p in policy.parameters()):,})")
    print(f"[*] 학습 시작: 총 {args.epochs} 에포크...")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse_px": [],
        "val_cp_l1": [],
    }

    best_val_rmse = float("inf")
    best_weights = None

    t0_train = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        policy.train()
        train_loss_accum = 0.0
        n_train_batches = 0

        for b_obs, b_state, b_cp, b_act, b_pen in train_loader:
            b_obs = b_obs.to(device)
            b_state = b_state.to(device)
            b_cp = b_cp.to(device)
            b_act = b_act.to(device)
            b_pen = b_pen.to(device)

            optimizer.zero_grad()
            loss, metrics = policy.loss(b_obs, b_state, target_cp=b_cp, target_traj=b_act, target_pen=b_pen)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accum += loss.item()
            n_train_batches += 1

        scheduler.step()
        sync(device)

        # 평가 (Validation)
        policy.eval()
        val_loss_accum = 0.0
        val_cp_l1_accum = 0.0
        val_rmse_px_accum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for b_obs, b_state, b_cp, b_act, b_pen in val_loader:
                b_obs = b_obs.to(device)
                b_state = b_state.to(device)
                b_cp = b_cp.to(device)
                b_act = b_act.to(device)
                b_pen = b_pen.to(device)

                loss, metrics = policy.loss(b_obs, b_state, target_cp=b_cp, target_traj=b_act, target_pen=b_pen)
                pred_cp, pred_traj, _ = policy.predict(b_obs, b_state)

                cp_l1 = F.l1_loss(pred_cp, b_cp).item()
                diff = pred_traj - b_act[..., :2]
                rmse_px = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1))).item() * 128.0

                val_loss_accum += loss.item()
                val_cp_l1_accum += cp_l1
                val_rmse_px_accum += rmse_px
                n_val_batches += 1

        avg_train_loss = train_loss_accum / max(1, n_train_batches)
        avg_val_loss = val_loss_accum / max(1, n_val_batches)
        avg_val_rmse = val_rmse_px_accum / max(1, n_val_batches)
        avg_val_cp = val_cp_l1_accum / max(1, n_val_batches)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_rmse_px"].append(avg_val_rmse)
        history["val_cp_l1"].append(avg_val_cp)

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            best_weights = copy.deepcopy(policy.state_dict())

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | Val CP L1: {avg_val_cp:.4f} | "
                f"Val RMSE: {avg_val_rmse:.2f}px | LR: {lr_now:.2e}",
                flush=True,
            )

    train_time = time.perf_counter() - t0_train
    print(f"\n[*] 학습 완료 (총 {train_time:.1f}초, 최고 Val RMSE: {best_val_rmse:.2f}px)")

    # 모델 체크포인트 저장
    model_save_path = OUT_DIR / f"bspline_vla_policy{tag}.pth"
    torch.save(
        {
            "state_dict": best_weights or policy.state_dict(),
            "head": args.head,
            "num_control_points": args.num_control_points,
            "chunk_size": args.chunk_size,
            "best_val_rmse": best_val_rmse,
            "args": vars(args),
        },
        model_save_path,
    )
    print(f"[*] 정책 체크포인트 저장 -> {model_save_path}")

    # 손실 곡선 시각화 저장
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    ax1.plot(history["train_loss"], label="Train Loss", color="#3182CE", lw=1.8)
    ax1.plot(history["val_loss"], label="Val Loss", color="#E53E3E", lw=1.8)
    ax1.set_title("ABK-VLA Training Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.plot(history["val_rmse_px"], label="Val Trajectory RMSE (128px)", color="#38A169", lw=1.8)
    ax2.axhline(best_val_rmse, color="#D69E2E", linestyle="--", label=f"Best: {best_val_rmse:.2f}px")
    ax2.set_title("Validation Reconstruction Error (px)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE (pixels)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend()

    curve_path = OUT_DIR / f"bspline_loss_curve{tag}.png"
    plt.tight_layout()
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    print(f"[*] 손실 곡선 저장 -> {curve_path}")

    # 학습 로그 JSON 저장
    log_path = OUT_DIR / f"bspline_train_log{tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_rmse_px": best_val_rmse,
                "total_train_time_sec": train_time,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"[*] 학습 로그 JSON 저장 -> {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ABK-VLA Policy Network")
    parser.add_argument("--tag", type=str, default="test", help="출력 파일 식별 태그")
    parser.add_argument("--head", choices=["mlp", "cvae"], default="mlp", help="정책 헤드 타입 (mlp | cvae)")
    parser.add_argument("--epochs", type=int, default=30, help="학습 에포크 수 (기본 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기 (기본 64)")
    parser.add_argument("--lr", type=float, default=3e-4, help="학습률 (기본 3e-4)")
    parser.add_argument("--num-control-points", "-m", type=int, default=6, help="제어점 개수 (기본 6)")
    parser.add_argument("--chunk-size", "-H", type=int, default=16, help="청크 horizon (기본 16)")
    parser.add_argument("--stride", type=int, default=8, help="슬라이딩 윈도우 스텝 간격 (기본 8)")
    parser.add_argument("--smoothness", type=float, default=0.02, help="2차 차분 평활화 가중치")
    parser.add_argument("--targets", nargs="*", default=["apple"], help="학습 도안 타겟")
    args = parser.parse_args()

    train_bspline_vla(args)


if __name__ == "__main__":
    main()
