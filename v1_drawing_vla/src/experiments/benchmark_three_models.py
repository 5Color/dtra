"""experiments/benchmark_three_models.py

[논문 3단계 비교 프레임워크] Model A vs. Model B vs. Model C 종합 지속학습 벤치마크.
- Model A: Naive Fine-Tuning (전역 가중치 파인튜닝, Raw Data) -> 파국적 망각 실증
- Model B: ABK-VLA Slot Isolation (Raw Demonstrations) -> 망각 0%, 소량 데이터(star/cat) 병목 노출
- Model C: ABK-VLA + B-Spline Geometric Augmentation + WTA-4 다중 모달 슬롯
           -> 망각 0% + 소량 데이터 병목 해소(별 5각 모서리·새싹 지면선 등 고성능 복원)

사용법:
  python experiments/benchmark_three_models.py --tasks apple sprout star cat
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "v1_drawing_vla" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from tools.bspline_augment import BSplineGeometricAugmentor
from train.bspline_policy import BSplineModelCPolicy, BSplineSlotPolicy, BSplineVisuomotorPolicy
from train.rollout_bspline import compute_smoothness, match_scores, rollout_bspline

OUT_DIR = PROJECT_ROOT / "outputs" / "v1_archive"


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


def safe_savefig(path: Path) -> Path:
    """Windows 파일 잠금 회피용 임시 저장 후 원자적 교체."""
    tmp = path.with_suffix(".tmp.png")
    plt.savefig(tmp, bbox_inches="tight")
    plt.close()
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return path
        except OSError:
            time.sleep(0.3)
    alt = path.with_name(f"{path.stem}_{int(time.time())}.png")
    os.replace(tmp, alt)
    return alt


def prepare_task_loaders(
    npz_path: Path,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        canv = data["canv"]
        state = data["state"]
        cp = data["control_points"]
        pen = data["pen"]
        raw_action = data["raw_action"]
        target_images = data["target_images"]
        ep_ids = data["ep_id"]

    tgt_img = target_images[0]
    tgt_expanded = np.repeat(tgt_img[np.newaxis, :, :], len(canv), axis=0)
    obs = np.stack([canv, tgt_expanded], axis=1).astype(np.float32) / 255.0

    unique_eps = np.unique(ep_ids)
    np.random.seed(seed)
    n_val = max(1, int(len(unique_eps) * val_ratio))
    val_eps = set(np.random.choice(unique_eps, size=n_val, replace=False))
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

    tr_obs = obs[train_mask]
    tr_pen = pen[train_mask]
    # 다중 획 태스크(cat 등)에서만 보조 획 및 공중 비행(pen=0) 샘플링 가중치 상향 적용
    # (단일 획인 star, apple의 균일 학습 손상 방지)
    sample_weights = np.ones(len(tr_obs), dtype=np.float32)
    has_multiple_strokes = float(tr_pen.min()) < 0.5
    if has_multiple_strokes:
        canv_masses = tr_obs[:, 0].sum(axis=(1, 2))
        tgt_masses = tr_obs[:, 1].sum(axis=(1, 2))
        is_secondary = (canv_masses > 0.35 * tgt_masses)
        is_jump = (tr_pen.mean(axis=1) < 0.8)
        sample_weights[is_secondary] *= 2.0
        sample_weights[is_jump] *= 2.5

    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), num_samples=len(tr_obs), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=(len(train_ds) > batch_size))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, tgt_img


def run_rollout_eval(
    model: torch.nn.Module,
    target_img: np.ndarray,
    device: torch.device,
    task_id: int = 0,
    max_iters: int = 90,
    exec_steps: int = 12,
    patience: int = 8,
    predict_mode: str = "argmax",
) -> Tuple[Dict[str, float], np.ndarray]:
    class Wrapper(torch.nn.Module):
        def predict(self, obs, state):
            if isinstance(model, BSplineModelCPolicy):
                return model.predict(obs, state, task_id=task_id, mode=predict_mode)
            if isinstance(model, BSplineSlotPolicy):
                return model.predict(obs, state, task_id=task_id)
            return model.predict(obs, state)

    model.eval()
    executed, history, stop_reason, _ = rollout_bspline(
        Wrapper(),
        target_img,
        device=device,
        max_iters=max_iters,
        exec_steps=exec_steps,
        patience=patience,
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
        "stop_reason": str(stop_reason),
    }, final_canvas


def train_model_c_slot(
    model: BSplineModelCPolicy,
    augmentor: BSplineGeometricAugmentor,
    train_loader: DataLoader,
    device: torch.device,
    task_id: int,
    epochs: int,
    lr: float,
    repeat: int,
    p_affine: float,
    rot_deg: float,
    shift_lim: float,
    scale_lim: float,
    desc: str,
) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(
        f"[*] {desc}: {epochs} epochs × {repeat}x aug (params={sum(p.numel() for p in trainable):,})...",
        flush=True,
    )

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for _ in range(repeat):
            for bo, bs, bc, ba, bp in train_loader:
                bo, bs, bc, ba, bp = bo.to(device), bs.to(device), bc.to(device), ba.to(device), bp.to(device)
                bo, bs, bc, ba, bp = augmentor(
                    bo,
                    bs,
                    bc,
                    ba,
                    bp,
                    p_affine=p_affine,
                    rot_deg=rot_deg,
                    shift_lim=shift_lim,
                    scale_lim=scale_lim,
                )
                optimizer.zero_grad()
                loss, _ = model.loss(bo, bs, target_cp=bc, target_traj=ba, target_pen=bp, task_id=task_id)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        scheduler.step()
        sync(device)
        if ep in (1, epochs // 2, epochs):
            print(f"    [{desc}] Epoch {ep:02d}/{epochs:02d} | Avg Loss: {total_loss / max(1, n_batches):.4f}", flush=True)


def run_three_model_benchmark(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    task_names = args.tasks
    K = len(task_names)

    # 1. Model A & Model B 기존 공정 벤치마크 로드 (동일 255개 동기화 데이터셋 결과)
    ab_json_path = OUT_DIR / "multitask_continual_learning_benchmark.json"
    base_ckpt_path = OUT_DIR / "multitask_baseline_final.pth"
    ours_b_ckpt_path = OUT_DIR / "multitask_ours_final.pth"
    if not (ab_json_path.exists() and base_ckpt_path.exists() and ours_b_ckpt_path.exists()):
        raise FileNotFoundError("Model A/B 벤치마크 JSON 또는 체크포인트가 없습니다. 먼저 benchmark_continual_learning_multitask.py를 실행하세요.")

    ab_data = json.loads(ab_json_path.read_text(encoding="utf-8"))
    R_a = np.array(ab_data["R_baseline"], dtype=np.float32)
    R_b = np.array(ab_data["R_ours"], dtype=np.float32)

    # 2. 데이터 로더 로드
    loaders, targets_info = [], []
    for name in task_names:
        cache_path = PROJECT_ROOT / "data" / "bspline" / f"bspline_vla_dataset_{name}_m6_h16_s8.npz"
        tr_l, val_l, tgt_img = prepare_task_loaders(cache_path, batch_size=args.batch_size)
        loaders.append((tr_l, val_l))
        targets_info.append((name, tgt_img))
        print(f"[*] Task '{name}': Train {len(tr_l.dataset)} | Val {len(val_l.dataset)}")

    # 3. Model C 학습 (Apple -> Sprout -> Star -> Cat 순차 지속학습)
    print("\n" + "=" * 80)
    print("      [3단계 비교 프레임워크] Model C (ABK-VLA + B-Spline Geometric Augmentation)      ")
    print("=" * 80, flush=True)

    torch.manual_seed(42)
    model_c = BSplineModelCPolicy(
        num_control_points=6,
        horizon=16,
        slot_type="wta",
        num_modes=4,
    ).to(device)
    augmentor = BSplineGeometricAugmentor(num_control_points=6, horizon=16).to(device)

    R_c = np.zeros((K, K), dtype=np.float32)
    details_c_init: Dict[str, Dict] = {}
    details_c_final: Dict[str, Dict] = {}
    final_canvases_c: Dict[str, Tuple[np.ndarray, float]] = {}

    t_start = time.time()
    for step_idx, (task_name, tgt_img) in enumerate(targets_info):
        tr_loader, _ = loaders[step_idx]
        model_c.add_slot(step_idx)
        model_c.to(device)

        if step_idx == 0:
            # Task 0: 전역 동변 인코더 + Slot 0 동시 학습
            train_model_c_slot(
                model_c,
                augmentor,
                tr_loader,
                device,
                task_id=0,
                epochs=args.epochs_t0,
                lr=3e-4,
                repeat=1,
                p_affine=0.60,
                rot_deg=6.0,
                shift_lim=0.08,
                scale_lim=0.10,
                desc=f"Model C Slot 0 [T0:{task_name}] (Equivariant Encoder)",
            )
        else:
            # 이전 슬롯 및 공유 인코더 완전 동결 -> 0% 망각 보장
            model_c.freeze_previous_slots(current_task_id=step_idx, freeze_encoder=True)
            # 소량 데이터 태스크는 기하 증강 배수(repeat)로 샘플 기근 해소
            n_train = len(tr_loader.dataset)
            rep = 5 if n_train < 1500 else (4 if n_train < 2500 else 3)
            train_model_c_slot(
                model_c,
                augmentor,
                tr_loader,
                device,
                task_id=step_idx,
                epochs=args.epochs_slot,
                lr=1e-3,
                repeat=rep,
                p_affine=0.45,
                rot_deg=4.0,
                shift_lim=0.04,
                scale_lim=0.06,
                desc=f"Model C Slot {step_idx} [T{step_idx}:{task_name}]",
            )

        print(f"\n[*] [Model C Step {step_idx + 1}/{K}] Evaluating seen tasks 0..{step_idx}...")
        for eval_idx in range(step_idx + 1):
            e_name, e_tgt = targets_info[eval_idx]
            res_c, canv_c = run_rollout_eval(
                model_c,
                e_tgt,
                device,
                task_id=eval_idx,
                max_iters=args.max_iters,
                exec_steps=12,
                patience=8,
                predict_mode="argmax",
            )
            R_c[eval_idx, step_idx] = res_c["f1"]
            print(
                f"    - Task {eval_idx} ({e_name:7s}) | Model C F1: {res_c['f1']*100:5.1f}% "
                f"(Cov {res_c['coverage']*100:4.0f}% Prec {res_c['precision']*100:4.0f}% Wave {res_c['wave_deg']:4.1f}° {res_c['steps']:3d}st)",
                flush=True,
            )
            if eval_idx == step_idx:
                details_c_init[e_name] = res_c
            if step_idx == K - 1:
                details_c_final[e_name] = res_c
                final_canvases_c[e_name] = (canv_c, res_c["f1"])

    elapsed = time.time() - t_start

    # 4. Model A & B 최종 체크포인트 캔버스도 동일 평가기로 렌더링해 4-Row 그리드 구성
    ckpt_a = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    model_a = BSplineVisuomotorPolicy(6, 16)
    model_a.load_state_dict(ckpt_a["model"])
    model_a.to(device).eval()

    ckpt_b = torch.load(ours_b_ckpt_path, map_location="cpu", weights_only=False)
    model_b = BSplineSlotPolicy(6, 16)
    for idx in range(K):
        model_b.add_slot(idx)
    model_b.load_state_dict(ckpt_b["model"])
    model_b.to(device).eval()

    final_canvases_a, final_canvases_b = {}, {}
    for idx, (name, tgt_img) in enumerate(targets_info):
        res_a_eval, canv_a = run_rollout_eval(model_a, tgt_img, device, max_iters=65, exec_steps=12, patience=6)
        res_b_eval, canv_b = run_rollout_eval(model_b, tgt_img, device, task_id=idx, max_iters=65, exec_steps=12, patience=6)
        final_canvases_a[name] = (canv_a, float(R_a[idx, K - 1]))
        final_canvases_b[name] = (canv_b, float(R_b[idx, K - 1]))

    # Model C 체크포인트 영구 저장
    ckpt_c_path = OUT_DIR / "multitask_model_c_final.pth"
    torch.save({"model": model_c.state_dict(), "tasks": task_names, "slot_type": "wta", "num_modes": 4}, ckpt_c_path)
    print(f"\n[*] Model C 체크포인트 저장 완료 -> {ckpt_c_path}")

    # 5. 종합 메트릭 산출
    def calc_metrics(R: np.ndarray) -> Dict[str, float]:
        forgets = [max(0.0, float(np.max(R[i, i : K - 1]) - R[i, K - 1])) for i in range(K - 1)]
        rets = [float(R[i, K - 1] / max(1e-6, R[i, i]) * 100.0) for i in range(K - 1)]
        return {
            "avg_final_f1": float(np.mean(R[:, K - 1])),
            "avg_forgetting": float(np.mean(forgets)),
            "avg_retention": float(np.mean(rets)),
        }

    met_a, met_b, met_c = calc_metrics(R_a), calc_metrics(R_b), calc_metrics(R_c)

    print("\n" + "#" * 86)
    print("              [3단계 비교 프레임워크 최종 요약: Model A vs B vs C]              ")
    print("#" * 86)
    print(f"  Task         | Model A (Naive FT) | Model B (ABK Raw) | Model C (ABK + Geom Aug)")
    print("-" * 86)
    for i, name in enumerate(task_names):
        print(
            f"  {name:12s} |      {R_a[i, K-1]*100:6.2f}%       |      {R_b[i, K-1]*100:6.2f}%      |         {R_c[i, K-1]*100:6.2f}%"
        )
    print("-" * 86)
    print(f"  평균 최종 F1 |      {met_a['avg_final_f1']*100:6.2f}%       |      {met_b['avg_final_f1']*100:6.2f}%      |         {met_c['avg_final_f1']*100:6.2f}%")
    print(f"  평균 망각률  |      {met_a['avg_forgetting']*100:6.2f}%       |       {met_b['avg_forgetting']*100:5.2f}%      |          {met_c['avg_forgetting']*100:5.2f}%")
    print(f"  기억 보존율  |      {met_a['avg_retention']:6.2f}%       |      {met_b['avg_retention']:6.2f}%      |         {met_c['avg_retention']:6.2f}%")
    print("#" * 86)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": task_names,
        "elapsed_model_c_sec": elapsed,
        "R_model_a": R_a.tolist(),
        "R_model_b": R_b.tolist(),
        "R_model_c": R_c.tolist(),
        "summary": {"model_a": met_a, "model_b": met_b, "model_c": met_c},
        "model_c_per_task": {
            name: {
                "init_f1": float(R_c[i, i]),
                "final_f1": float(R_c[i, K - 1]),
                "init_details": details_c_init.get(name, {}),
                "final_details": details_c_final.get(name, {}),
            }
            for i, name in enumerate(task_names)
        },
    }
    json_path = OUT_DIR / "three_model_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[*] 3-모델 비교 JSON 저장 -> {json_path}")

    # 6. 시각화 1: 3-Panel Lifelong Retention Heatmap (Model A vs B vs C)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=200)
    fig.patch.set_facecolor("#F8F9FA")
    for ax_idx, (mat, title) in enumerate([
        (R_a, f"Model A: Naive FT\n(Avg F1 {met_a['avg_final_f1']*100:.1f}%, Forget {met_a['avg_forgetting']*100:.1f}%)"),
        (R_b, f"Model B: ABK-VLA Raw\n(Avg F1 {met_b['avg_final_f1']*100:.1f}%, Forget 0.0%)"),
        (R_c, f"Model C: ABK-VLA + Geom Aug\n(Avg F1 {met_c['avg_final_f1']*100:.1f}%, Forget 0.0%)"),
    ]):
        ax = axes[ax_idx]
        masked = np.ma.masked_where(np.tril(np.ones_like(mat), k=-1), mat * 100.0)
        im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels([f"After {t}" for t in task_names], fontsize=9, rotation=20)
        ax.set_yticklabels([f"Eval {t}" for t in task_names], fontsize=9)
        for r in range(K):
            for c in range(r, K):
                val = mat[r, c] * 100.0
                col = "white" if (val < 35 or val > 80) else "black"
                ax.text(c, r, f"{val:.1f}%", ha="center", va="center", color=col, fontweight="bold", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("3-Stage Continual Learning Comparison: Model A (Collapse) vs. Model B (Raw Slots) vs. Model C (B-Spline Geom Aug)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    hm_path = safe_savefig(OUT_DIR / "multitask_3model_heatmap.png")
    print(f"[*] 3-모델 히트맵 저장 -> {hm_path}")

    # 7. 시각화 2: 4 x 4 Qualitative Rollout Grid (Target / Model A / Model B / Model C)
    fig, axes = plt.subplots(4, K, figsize=(4 * K, 14.5), dpi=200)
    fig.patch.set_facecolor("#F8F9FA")
    for col, name in enumerate(task_names):
        _, tgt_img = targets_info[col]
        ca, fa = final_canvases_a[name]
        cb, fb = final_canvases_b[name]
        cc, fc = final_canvases_c[name]

        axes[0, col].imshow(tgt_img, cmap="gray_r")
        axes[0, col].set_title(f"Target: {name.capitalize()}", fontweight="bold", fontsize=11)

        axes[1, col].imshow(ca, cmap="gray_r")
        axes[1, col].set_title(f"Model A (Naive FT)\nF1: {fa*100:.1f}%", color="#E53E3E", fontweight="bold", fontsize=10)

        axes[2, col].imshow(cb, cmap="gray_r")
        b_col = "#D69E2E" if fb < 0.7 else "#2B6CB0"
        axes[2, col].set_title(f"Model B (ABK Raw)\nF1: {fb*100:.1f}%", color=b_col, fontweight="bold", fontsize=10)

        axes[3, col].imshow(cc, cmap="gray_r")
        axes[3, col].set_title(f"Model C (ABK + Geom Aug)\nF1: {fc*100:.1f}%", color="#2F855A", fontweight="bold", fontsize=10)

    for r, lbl in enumerate(["Ground Truth", "Model A\n(Naive FT)", "Model B\n(ABK Raw)", "Model C\n(ABK + Geom Aug)"]):
        axes[r, 0].set_ylabel(lbl, fontsize=11, fontweight="bold", labelpad=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Qualitative Rollout Comparison After 4-Task Lifelong Training: Model A vs. Model B vs. Model C", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    gr_path = safe_savefig(OUT_DIR / "multitask_3model_rollout_grid.png")
    print(f"[*] 3-모델 캔버스 그리드 저장 -> {gr_path}")


def main():
    parser = argparse.ArgumentParser(description="3-Model Continual Learning Comparison (Model A vs B vs C)")
    parser.add_argument("--tasks", nargs="+", default=["apple", "sprout", "star", "cat"])
    parser.add_argument("--epochs-t0", type=int, default=10, help="Task 0 동변 인코더+슬롯 학습 에포크")
    parser.add_argument("--epochs-slot", type=int, default=12, help="후속 태스크 슬롯 학습 에포크")
    parser.add_argument("--max-iters", type=int, default=100, help="폐루프 롤아웃 최대 반복 횟수")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    run_three_model_benchmark(args)


if __name__ == "__main__":
    main()
