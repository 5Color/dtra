"""experiments/benchmark_autonomous_cl.py

[정식 지속학습 벤치마크] Task-ID-Free Autonomous ABK-VLA (Model D) 5-모델 비교 평가.

동일 데이터 분할(seed=42), 동일 에폭, 동일 기하 증강 조건에서 4-Task 순차 학습
(apple -> sprout -> star -> cat)을 수행하며 다음 5개 모델을 비교한다:
  1. Model 1 (GRU + Naive FT)           : 미시적 (x, y, pen) 자기회귀 시계열 베이스라인
  2. Model 2 (ABK-VLA + Naive FT)       : B-Spline 제어점 단순 순차 덮어쓰기 (망각 하한선)
  3. Model 3 (ABK-VLA + Replay)         : 이전 태스크 버퍼 리허설 기반 지속학습 대조군
  4. Model 4 (ABK-VLA + Oracle Slots)   : task_id 정답 제공 상한선 (Model C)
  5. Model 5 (Autonomous ABK-VLA, D)    : task_id 0% 입력 + 프로토타입 Softmax 라우팅 + 잔차 슬롯

출력 산출물:
  - outputs/autonomous_cl_benchmark.json
  - outputs/autonomous_cl_heatmap.png
  - outputs/autonomous_cl_rollout_grid.png (4 x 5 비교 그리드)
  - outputs/autonomous_routing_matrix.png  (4 x 4 자율 슬롯 가중치)

사용법:
  python experiments/benchmark_autonomous_cl.py --tasks apple sprout star cat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from experiments.benchmark_three_models import (
    OUT_DIR,
    pick_device,
    prepare_task_loaders,
    safe_savefig,
    sync,
)
from tools.bspline_augment import BSplineGeometricAugmentor
from train.bspline_policy import (
    AutonomousABKPolicy,
    BSplineModelCPolicy,
    BSplineVisuomotorPolicy,
    GRUSequencePolicy,
)
from train.dataset import pen_down_runs
from train.rollout_bspline import compute_smoothness, match_scores, rollout_bspline


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_cl_metrics(R: np.ndarray) -> Dict[str, float]:
    """공통 지속학습 표준 지표 산출 (Avg Final F1, Forgetting Rate, Backward Transfer)."""
    K = R.shape[0]
    forgets = [max(0.0, float(np.max(R[i, i : K - 1]) - R[i, K - 1])) for i in range(K - 1)]
    bwts = [float(R[i, K - 1] - R[i, i]) for i in range(K - 1)]
    return {
        "avg_final_f1": float(np.mean(R[:, K - 1])),
        "avg_forgetting": float(np.mean(forgets)) if forgets else 0.0,
        "avg_bwt": float(np.mean(bwts)) if bwts else 0.0,
    }


def compute_chamfer_distance(executed, target_img: np.ndarray) -> float:
    """예측 궤적 점군과 정답 점군 간 양방향 Chamfer Distance 산출 (좌표계 [0, 1] 기준, 낮을수록 우수)."""
    if executed is None or len(executed) == 0:
        return 1.0
    arr = np.asarray(executed, dtype=np.float32)
    pen_down = arr[:, 2] > 0.5
    if pen_down.sum() == 0:
        return 1.0
    pred_pts = arr[pen_down, :2]

    tgt_y, tgt_x = np.where(target_img > 32)
    if len(tgt_y) == 0:
        return 0.0
    tgt_pts = np.column_stack([tgt_x / 127.0, tgt_y / 127.0]).astype(np.float32)

    tree_tgt = cKDTree(tgt_pts)
    d_p2t, _ = tree_tgt.query(pred_pts)
    tree_pred = cKDTree(pred_pts)
    d_t2p, _ = tree_pred.query(tgt_pts)
    return float(0.5 * (d_p2t.mean() + d_t2p.mean()))


def compute_normalized_mean_jerk(executed) -> float:
    """펜-다운 연속 구간에서의 3차 미분(Jerk) 기반 궤적 매끄러움 지표 (낮을수록 우수)."""
    if executed is None or len(executed) < 4:
        return 0.0
    arr = np.asarray(executed, dtype=np.float32)
    runs = pen_down_runs(arr)
    jerks = []
    for a, b in runs:
        if b - a >= 4:
            pts = arr[a:b, :2]
            j = pts[3:] - 3 * pts[2:-1] + 3 * pts[1:-2] - pts[:-3]
            mag = np.linalg.norm(j, axis=1)
            jerks.extend(mag.tolist())
    if not jerks:
        return 0.0
    return float(np.mean(jerks) * 1e3)


def compute_component_recall(canvas_img: np.ndarray, target_img: np.ndarray, task_name: str) -> float:
    """목표 도안의 전체 연결 요소(외곽선, 잎, 줄기, 눈, 코 등) 중 실제 복원된 파트 비율 (높을수록 우수)."""
    canv_bin = canvas_img > 32
    tgt_bin = target_img > 32

    labeled, num_features = ndimage.label(tgt_bin)
    components = []
    for c in range(1, num_features + 1):
        mask = (labeled == c)
        if mask.sum() >= 15:
            components.append(mask)

    if task_name == "apple" and len(components) == 1:
        y_grid, _ = np.mgrid[0:128, 0:128] / 127.0
        leaf_mask = tgt_bin & (y_grid < 0.28)
        body_mask = tgt_bin & (y_grid >= 0.28)
        components = [body_mask, leaf_mask]
    elif task_name == "sprout" and len(components) == 1:
        y_grid, _ = np.mgrid[0:128, 0:128] / 127.0
        leaves_mask = tgt_bin & (y_grid < 0.60)
        stem_ground_mask = tgt_bin & (y_grid >= 0.60)
        components = [leaves_mask, stem_ground_mask]

    if not components:
        return 0.0

    recalled = 0
    for comp in components:
        comp_mass = comp.sum()
        if comp_mass == 0:
            continue
        covered = (comp & canv_bin).sum()
        if (covered / comp_mass) >= 0.30:
            recalled += 1
    return float(recalled / len(components))


def eval_episode_rollout(
    model: torch.nn.Module,
    target_img: np.ndarray,
    device: torch.device,
    task_name: str = "",
    task_id: Optional[int] = None,
    max_iters: int = 100,
    exec_steps: int = 16,
    patience: int = 8,
) -> Tuple[Dict[str, float], np.ndarray]:
    """단일 도안에 대한 폐루프 롤아웃 점수, 다차원 구조 지표 및 추론 지연 측정."""
    if hasattr(model, "reset_cache"):
        model.reset_cache()

    class _PredictAdapter(torch.nn.Module):
        def predict(self, obs, state):
            if isinstance(model, BSplineModelCPolicy):
                return model.predict(obs, state, task_id=task_id, mode="argmax")
            return model.predict(obs, state)

    model.eval()
    executed, history, stop_reason, latencies = rollout_bspline(
        _PredictAdapter(),
        target_img,
        device=device,
        max_iters=max_iters,
        exec_steps=exec_steps,
        patience=patience,
    )
    final_canvas = history[-1]["canvas"]
    scores = match_scores(final_canvas, target_img)
    smooth = compute_smoothness(executed)
    cd = compute_chamfer_distance(executed, target_img)
    nmj = compute_normalized_mean_jerk(executed)
    comp_rec = compute_component_recall(final_canvas, target_img, task_name)
    return {
        "f1": float(scores["f1"]),
        "coverage": float(scores["coverage"]),
        "precision": float(scores["precision"]),
        "chamfer_dist": float(cd),
        "normalized_jerk": float(nmj),
        "component_recall": float(comp_rec),
        "turn_deg": float(smooth["turn_deg"]),
        "wave_deg": float(smooth["wave_deg"]),
        "steps": int(len(executed)),
        "stop_reason": str(stop_reason),
        "avg_step_infer_ms": float(np.mean(latencies)) if latencies else 0.0,
        "episode_infer_ms": float(np.sum(latencies)) if latencies else 0.0,
        "n_chunks": int(len(latencies)),
    }, final_canvas


def compute_loss_for_model(
    model: torch.nn.Module,
    bo: torch.Tensor,
    bs: torch.Tensor,
    bc: torch.Tensor,
    ba: torch.Tensor,
    bp: torch.Tensor,
    task_id: Optional[int],
) -> torch.Tensor:
    """모델 유형에 맞춰 동일한 손실 인터페이스 호출."""
    if isinstance(model, BSplineModelCPolicy):
        loss, _ = model.loss(bo, bs, target_cp=bc, target_traj=ba, target_pen=bp, task_id=task_id)
        return loss
    if isinstance(model, AutonomousABKPolicy):
        loss, _ = model.loss(bo, bs, target_cp=bc, target_traj=ba, target_pen=bp)
        return loss
    if isinstance(model, GRUSequencePolicy):
        loss, _ = model.loss(bo, bs, target_traj=ba, target_pen=bp)
        return loss
    loss, _ = model.loss(bo, bs, target_cp=bc, target_traj=ba, target_pen=bp)
    return loss


class BalancedReplayBuffer:
    """Experience Replay 대조군용 태스크별 균형 리허설 버퍼."""

    def __init__(self, per_task_cap: int = 256, seed: int = 42):
        self.per_task_cap = per_task_cap
        self.rng = np.random.default_rng(seed)
        self.tasks: List[Tuple[torch.Tensor, ...]] = []

    def add_task_dataset(self, ds) -> None:
        n = len(ds)
        idx = self.rng.choice(n, size=min(self.per_task_cap, n), replace=False)
        tensors = tuple(t[idx].cpu() for t in ds.tensors)
        self.tasks.append(tensors)

    def sample_batch(
        self, batch_size: int, device: torch.device
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        if not self.tasks:
            return None
        per_t = max(1, batch_size // len(self.tasks))
        chunks = [[] for _ in range(len(self.tasks[0]))]
        for task_tensors in self.tasks:
            n = len(task_tensors[0])
            idx = torch.randint(0, n, (per_t,))
            for k, t in enumerate(task_tensors):
                chunks[k].append(t[idx])
        return tuple(torch.cat(c, dim=0).to(device) for c in chunks)


def train_one_task(
    model: torch.nn.Module,
    augmentor: BSplineGeometricAugmentor,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    repeat: int,
    p_affine: float,
    rot_deg: float,
    shift_lim: float,
    scale_lim: float,
    desc: str,
    task_id: Optional[int] = None,
    replay_buf: Optional[BalancedReplayBuffer] = None,
) -> int:
    """통일된 기하 증강 및 CosineAnnealingLR 스케줄러로 단일 태스크 학습. 학습 가능 파라미터 수 반환."""
    base_params, gate_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("prior_weight"):
            gate_params.append(p)
        else:
            base_params.append(p)
    trainable = base_params + gate_params
    n_trainable = sum(p.numel() for p in trainable)
    param_groups = [{"params": base_params, "lr": lr}]
    if gate_params:
        param_groups.append({"params": gate_params, "lr": lr * 5.0})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(
        f"[*] {desc}: {epochs} epochs × {repeat}x aug (trainable={n_trainable:,})...",
        flush=True,
    )

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for _ in range(repeat):
            for bo, bs, bc, ba, bp in train_loader:
                bo, bs, bc, ba, bp = (
                    bo.to(device),
                    bs.to(device),
                    bc.to(device),
                    ba.to(device),
                    bp.to(device),
                )
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
                loss = compute_loss_for_model(model, bo, bs, bc, ba, bp, task_id=task_id)

                if replay_buf is not None and len(replay_buf.tasks) > 0:
                    rep = replay_buf.sample_batch(bo.shape[0], device)
                    if rep is not None:
                        ro, rs, rc, ra, rp = augmentor(
                            *rep,
                            p_affine=p_affine,
                            rot_deg=rot_deg,
                            shift_lim=shift_lim,
                            scale_lim=scale_lim,
                        )
                        loss_rep = compute_loss_for_model(model, ro, rs, rc, ra, rp, task_id=task_id)
                        loss = 0.5 * loss + 0.5 * loss_rep

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1

        scheduler.step()
        sync(device)
        if ep in (1, epochs // 2, epochs):
            print(
                f"    [{desc}] Epoch {ep:02d}/{epochs:02d} | Avg Loss: {total_loss / max(1, n_batches):.4f}",
                flush=True,
            )

    return n_trainable


def overlay_target_canvas(canvas: np.ndarray, target_img: np.ndarray) -> np.ndarray:
    """예측 캔버스 아래에 연한 파스텔 블루 목표 윤곽을 깔아 4×5 그리드 가독성 강화."""
    rgb = np.ones((*canvas.shape, 3), dtype=np.float32)
    tgt_mask = np.clip(target_img.astype(np.float32) / 255.0, 0.0, 1.0)
    # 타겟 가이드: 연한 하늘색 (#BFDBFE)
    rgb[..., 0] -= 0.25 * tgt_mask
    rgb[..., 1] -= 0.14 * tgt_mask
    rgb[..., 2] -= 0.02 * tgt_mask

    ink = np.clip(canvas.astype(np.float32) / 255.0, 0.0, 1.0)
    rgb = rgb * (1.0 - ink[..., None])
    return np.clip(rgb, 0.0, 1.0)


def run_benchmark(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    task_names: List[str] = args.tasks
    K = len(task_names)

    loaders, targets_info = [], []
    for name in task_names:
        cache_path = PROJECT_ROOT / "data" / "bspline" / f"bspline_vla_dataset_{name}_m6_h16_s8.npz"
        tr_l, val_l, tgt_img = prepare_task_loaders(cache_path, batch_size=args.batch_size, seed=args.seed)
        loaders.append((tr_l, val_l))
        targets_info.append((name, tgt_img))
        print(f"[*] Task '{name}': Train {len(tr_l.dataset)} | Val {len(val_l.dataset)}")

    augmentor = BSplineGeometricAugmentor(num_control_points=6, horizon=16).to(device)

    all_model_specs = [
        ("model_1_gru", "Model 1: GRU + Naive FT", "gru"),
        ("model_2_naive", "Model 2: ABK-VLA + Naive FT", "naive"),
        ("model_3_replay", "Model 3: ABK-VLA + Exp. Replay", "replay"),
        ("model_4_oracle_c", "Model 4: ABK-VLA + Oracle (Model C)", "oracle_c"),
        ("model_5_auto_d", "Model 5: Autonomous ABK-VLA (Model D)", "auto_d"),
    ]

    model_specs = (
        [("model_5_auto_d", "Model 5: Autonomous ABK-VLA (Model D)", "auto_d")]
        if getattr(args, "only_model_d", False)
        else all_model_specs
    )

    json_path = OUT_DIR / "autonomous_cl_benchmark.json"
    results_store: Dict[str, Dict] = {}
    if getattr(args, "only_model_d", False) and json_path.exists():
        try:
            prev_data = json.loads(json_path.read_text(encoding="utf-8"))
            results_store = prev_data.get("models", {})
            print("[*] 기존 모델 1~4 벤치마크 데이터 로드 완료 (Model D 단독 갱신 모드)")
        except Exception:
            pass

    final_canvases: Dict[str, Dict[str, Tuple[np.ndarray, float]]] = {}
    zero_shot_logs: Dict[str, Dict] = {}
    expansion_logs: List[Dict] = []
    routing_matrix = np.zeros((K, K), dtype=np.float32)

    t_all = time.time()

    for key, label, kind in model_specs:
        print("\n" + "=" * 86)
        print(f"   [{label}] 4-Task 순차 학습 시작 ({' -> '.join(task_names)})")
        print("=" * 86, flush=True)

        set_seed(args.seed)
        n_modes = getattr(args, "num_modes", 6)
        if kind == "gru":
            model = GRUSequencePolicy(horizon=16, use_spatial_pointer=True).to(device)
        elif kind in ("naive", "replay"):
            model = BSplineVisuomotorPolicy(6, 16, use_spatial_pointer=True).to(device)
        elif kind == "oracle_c":
            model = BSplineModelCPolicy(6, 16, slot_type="wta", num_modes=n_modes, use_spatial_pointer=True).to(device)
        else:
            model = AutonomousABKPolicy(
                num_control_points=6,
                horizon=16,
                num_modes=n_modes,
                temperature=args.temperature,
            ).to(device)

        replay_buf = BalancedReplayBuffer(per_task_cap=args.replay_cap, seed=args.seed) if kind == "replay" else None
        R = np.zeros((K, K), dtype=np.float32)
        trainable_per_step: List[int] = []
        final_details: Dict[str, Dict] = {}
        final_canvases[key] = {}

        t_model_start = time.time()

        for step_idx, (t_name, tgt_img) in enumerate(targets_info):
            tr_loader, _ = loaders[step_idx]
            tgt_tensor = torch.from_numpy(tgt_img.astype(np.float32) / 255.0)[None, None].to(device)

            if kind == "oracle_c":
                model.add_slot(step_idx)
                model.to(device)
                if step_idx > 0:
                    model.freeze_previous_slots(current_task_id=step_idx, freeze_encoder=True)

            elif kind == "auto_d":
                # (1) Zero-Shot Forward Transfer 평가: 신규 슬롯 확장 직전 기존 슬롯 Softmax 혼합만으로 측정
                if step_idx > 0:
                    model.eval()
                    with torch.no_grad():
                        prior_w = model.compute_routing_weights(tgt_tensor)[0].detach().cpu().numpy().tolist()
                    zs_res, _ = eval_episode_rollout(
                        model,
                        tgt_img,
                        device,
                        task_name=t_name,
                        max_iters=args.max_iters,
                        exec_steps=args.exec_steps,
                    )
                    zero_shot_logs[t_name] = {
                        "zero_shot_f1": float(zs_res["f1"]),
                        "zero_shot_cov": float(zs_res["coverage"]),
                        "zero_shot_prec": float(zs_res["precision"]),
                        "prior_routing_weights": [float(x) for x in prior_w],
                    }
                    print(
                        f"    [Zero-Shot Forward Transfer -> Unseen '{t_name}'] "
                        f"F1={zs_res['f1']*100:.1f}% (Cov {zs_res['coverage']*100:.0f}%, "
                        f"Prec {zs_res['precision']*100:.0f}%) | prior_w={[round(w, 3) for w in prior_w]}",
                        flush=True,
                    )

                # (2) 자율 Novelty Detection & 동적 잔차 슬롯 확장 (task_id 없음)
                s_idx, expanded, max_sim = model.register_or_expand_slot(
                    tgt_tensor, novelty_threshold=args.novelty_threshold
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
                print(
                    f"    [Novelty Detection] Task '{t_name}': max_sim={max_sim:.4f} "
                    f"(thr={args.novelty_threshold:.2f}) -> expanded={expanded}, active_slot={s_idx}",
                    flush=True,
                )

            # 통일된 에폭 / 증강 스케줄
            n_train = len(tr_loader.dataset)
            if step_idx == 0:
                ep = args.epochs_t0
                rep = 1
                p_aff, rot_d, sh_l, sc_l = 0.60, 6.0, 0.08, 0.10
                lr_step = 3e-4
            else:
                ep = max(args.epochs_slot, 16) if t_name == "cat" else args.epochs_slot
                rep = 5 if (n_train < 1500 or t_name == "cat") else (4 if n_train < 2500 else 3)
                p_aff, rot_d, sh_l, sc_l = 0.45, 4.0, 0.04, 0.06
                lr_step = 1e-3 if kind in ("oracle_c", "auto_d") else 3e-4

            n_tr_p = train_one_task(
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
                desc=f"{label} | T{step_idx}:{t_name}",
                task_id=step_idx if kind == "oracle_c" else None,
                replay_buf=replay_buf,
            )
            trainable_per_step.append(n_tr_p)

            if kind == "auto_d" and step_idx == 0:
                model.refresh_slot_key(0, tgt_tensor)
                model.freeze_backbone_and_prior_slots(up_to_exclusive=1)

            if replay_buf is not None:
                replay_buf.add_task_dataset(tr_loader.dataset)

            # 평가: 지금까지 학습한 태스크 0..step_idx 롤아웃 평가
            print(f"    [{label}] Step {step_idx + 1}/{K} seen-task evaluation:")
            for eval_idx in range(step_idx + 1):
                e_name, e_tgt = targets_info[eval_idx]
                res, canv = eval_episode_rollout(
                    model,
                    e_tgt,
                    device=device,
                    task_name=e_name,
                    task_id=eval_idx if kind == "oracle_c" else None,
                    max_iters=args.max_iters,
                    exec_steps=args.exec_steps,
                )
                R[eval_idx, step_idx] = res["f1"]
                print(
                    f"      - {e_name:7s}: F1={res['f1']*100:5.1f}% "
                    f"(Cov {res['coverage']*100:4.0f}% Prec {res['precision']*100:4.0f}% "
                    f"{res['episode_infer_ms']:5.1f}ms)",
                    flush=True,
                )
                if step_idx == K - 1:
                    final_details[e_name] = res
                    final_canvases[key][e_name] = (canv, res["f1"])

        # Model D 최종 4x4 자율 라우팅 가중치 행렬 측정 (task_id=None)
        if kind == "auto_d":
            model.eval()
            with torch.no_grad():
                tgt_stack = torch.cat(
                    [
                        torch.from_numpy(t_img.astype(np.float32) / 255.0)[None, None]
                        for _, t_img in targets_info
                    ],
                    dim=0,
                ).to(device)
                routing_matrix = model.compute_routing_weights(tgt_stack).detach().cpu().numpy()

            ckpt_d_path = OUT_DIR / "autonomous_model_d_final.pth"
            torch.save(
                {"model": model.state_dict(), "tasks": task_names, "temperature": args.temperature},
                ckpt_d_path,
            )
            print(f"[*] Model D 체크포인트 저장 -> {ckpt_d_path}")

        total_params = sum(p.numel() for p in model.parameters())
        metrics = calc_cl_metrics(R)
        mean_ep_ms = float(np.mean([final_details[t]["episode_infer_ms"] for t in task_names]))
        mean_step_ms = float(np.mean([final_details[t]["avg_step_infer_ms"] for t in task_names]))

        results_store[key] = {
            "label": label,
            "kind": kind,
            "elapsed_sec": float(time.time() - t_model_start),
            "total_params": int(total_params),
            "trainable_params_per_step": trainable_per_step,
            "avg_episode_infer_ms": mean_ep_ms,
            "avg_step_infer_ms": mean_step_ms,
            "R": R.tolist(),
            "summary": metrics,
            "final_per_task": final_details,
        }

    # ------------------------------------------------------------------
    # 콘솔 요약 리포트 출력
    # ------------------------------------------------------------------
    print("\n" + "#" * 102)
    print("         [정식 5-모델 지속학습 벤치마크 최종 요약: Task-ID-Free Autonomous ABK-VLA]         ")
    print("#" * 102)
    header = "  Metric / Task       | " + " | ".join(f"{results_store[k]['kind']:^13s}" for k, _, _ in model_specs)
    print(header)
    print("-" * 102)
    for i, name in enumerate(task_names):
        vals = " | ".join(
            f"   {np.array(results_store[k]['R'])[i, K-1]*100:5.1f}%    " for k, _, _ in model_specs
        )
        print(f"  {name:19s} | {vals}")
    print("-" * 102)
    for m_key, m_lbl in [
        ("avg_final_f1", "Avg Final F1"),
        ("avg_forgetting", "Forgetting Rate"),
        ("avg_bwt", "Backward Transfer (BWT)"),
    ]:
        vals = " | ".join(
            f"   {results_store[k]['summary'][m_key]*100.0:+6.2f}%  " for k, _, _ in model_specs
        )
        print(f"  {m_lbl:23s} | {vals}")

    # 다차원 기하학·구조 지표 (Chamfer Distance, Normalized Jerk, Component Recall)
    for metric_name, metric_lbl in [
        ("chamfer_dist", "Mean Chamfer Dist (v)"),
        ("normalized_jerk", "Mean Norm. Jerk (v)"),
        ("component_recall", "Mean Comp. Recall (^)"),
    ]:
        vals = []
        for k, _, _ in model_specs:
            m_vals = [results_store[k]["final_per_task"][t][metric_name] for t in task_names]
            mean_v = float(np.mean(m_vals))
            if metric_name == "component_recall":
                vals.append(f"   {mean_v*100:5.1f}%   ")
            elif metric_name == "chamfer_dist":
                vals.append(f"   {mean_v:7.4f}  ")
            else:
                vals.append(f"   {mean_v:7.2f}  ")
        print(f"  {metric_lbl:23s} | " + " | ".join(vals))

    vals_ep_ms = " | ".join(
        f"  {results_store[k]['avg_episode_infer_ms']:6.1f} ms  " for k, _, _ in model_specs
    )
    vals_params = " | ".join(
        f" {results_store[k]['trainable_params_per_step'][-1]/1e6:5.2f}M (tr) " for k, _, _ in model_specs
    )
    print(f"  {'Episode Infer (ms)':23s} | {vals_ep_ms}")
    print(f"  {'Trainable Params T3':23s} | {vals_params}")
    print("#" * 102)

    print("\n[Model D Autonomous Routing Matrix (Softmax w, T={:.2f})]:".format(args.temperature))
    print("                " + "  ".join(f"Slot {j} ({t[:4]})" for j, t in enumerate(task_names)))
    for i, t in enumerate(task_names):
        row_s = "  ".join(f"   {routing_matrix[i, j]:6.4f}  " for j in range(K))
        print(f"  Input {t:7s}: {row_s}")

    # ------------------------------------------------------------------
    # JSON 저장
    # ------------------------------------------------------------------
    out_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "seed": args.seed,
        "tasks": task_names,
        "hyperparams": {
            "epochs_t0": args.epochs_t0,
            "epochs_slot": args.epochs_slot,
            "batch_size": args.batch_size,
            "max_iters": args.max_iters,
            "temperature": args.temperature,
            "novelty_threshold": args.novelty_threshold,
            "replay_cap_per_task": args.replay_cap,
        },
        "total_elapsed_sec": float(time.time() - t_all),
        "models": results_store,
        "model_d_zero_shot_forward_transfer": zero_shot_logs,
        "model_d_expansion_logs": expansion_logs,
        "model_d_routing_weights": routing_matrix.tolist(),
    }
    json_path = OUT_DIR / "autonomous_cl_benchmark.json"
    json_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(f"\n[*] 벤치마크 JSON 저장 -> {json_path}")

    # ------------------------------------------------------------------
    # 시각화 1: 5-모델 지속학습 F1 유지율 히트맵 (autonomous_cl_heatmap.png)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.6), dpi=200)
    fig.patch.set_facecolor("#F8F9FA")
    short_titles = [
        "1. GRU + Naive FT",
        "2. ABK + Naive FT",
        "3. ABK + Replay",
        "4. Oracle Slots (C)",
        "5. Autonomous (Model D)",
    ]
    for ax_idx, ((k, _, _), stitle) in enumerate(zip(all_model_specs, short_titles)):
        ax = axes[ax_idx]
        if k in results_store:
            mat = np.array(results_store[k]["R"], dtype=np.float32)
            sm = results_store[k]["summary"]
            masked = np.ma.masked_where(np.tril(np.ones_like(mat), k=-1), mat * 100.0)
            im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=100)
            ax.set_title(
                f"{stitle}\nF1 {sm['avg_final_f1']*100:.1f}% | Forget {sm['avg_forgetting']*100:.1f}%\nBWT {sm['avg_bwt']*100:+.1f}%",
                fontsize=10.0,
                fontweight="bold",
                pad=8,
            )
            for r in range(K):
                for c in range(r, K):
                    val = mat[r, c] * 100.0
                    col = "white" if (val < 35 or val > 80) else "black"
                    ax.text(c, r, f"{val:.1f}%", ha="center", va="center", color=col, fontweight="bold", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_title(stitle, fontsize=10.5, fontweight="bold")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels([f"After {t}" for t in task_names], fontsize=8, rotation=22)
        ax.set_yticklabels([f"Eval {t}" for t in task_names], fontsize=8)

    fig.tight_layout(rect=[0, 0.02, 1, 0.88])
    fig.subplots_adjust(top=0.82)
    plt.suptitle(
        "5-Model Continual Learning F1 Retention Heatmaps (apple -> sprout -> star -> cat)",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
    )
    hm_path = safe_savefig(OUT_DIR / "autonomous_cl_heatmap.png")
    print(f"[*] 5-모델 히트맵 저장 -> {hm_path}")

    # ------------------------------------------------------------------
    # 시각화 2: Model D 단독 롤아웃 시각화 (눈·코·수염 점프 복원 정밀 확인)
    # ------------------------------------------------------------------
    if "model_5_auto_d" in final_canvases:
        fig_d, axes_d = plt.subplots(1, K, figsize=(4.2 * K, 4.8), dpi=200)
        fig_d.patch.set_facecolor("#F8F9FA")
        if K == 1:
            axes_d = np.array([axes_d])
        for col_idx, (t_name, tgt_img) in enumerate(targets_info):
            ax = axes_d[col_idx]
            canv_d, f1_d = final_canvases["model_5_auto_d"][t_name]
            ax.imshow(overlay_target_canvas(canv_d, tgt_img))
            f1_color = "#2F855A" if f1_d >= 0.80 else ("#D69E2E" if f1_d >= 0.55 else "#C53030")
            ax.set_title(f"Model D: {t_name.capitalize()}\nF1: {f1_d*100:.1f}%", color=f1_color, fontweight="bold", fontsize=11, pad=6)
            ax.set_xticks([])
            ax.set_yticks([])
        fig_d.tight_layout(rect=[0, 0.02, 1, 0.88])
        fig_d.subplots_adjust(top=0.82)
        fig_d.suptitle("Model D Autonomous Rollout (With Uncovered Spatial Pointer & Pruned Inference)", fontsize=13, fontweight="bold", y=0.96)
        safe_savefig(OUT_DIR / "autonomous_model_d_rollout.png")
        print(f"[*] Model D 롤아웃 단독 확인 그리드 저장 -> {OUT_DIR / 'autonomous_model_d_rollout.png'}")

    # ------------------------------------------------------------------
    # 시각화 3: 최종 4 × 5 캔버스 롤아웃 시각 비교 그리드 (autonomous_cl_rollout_grid.png)
    # ------------------------------------------------------------------
    if len(final_canvases) == len(all_model_specs):
        fig, axes = plt.subplots(K, 5, figsize=(18.5, max(4.0, 3.6 * K + 1.2)), dpi=200)
        fig.patch.set_facecolor("#F8F9FA")
        if K == 1:
            axes = np.array([axes]) if axes.ndim == 1 else axes.reshape(1, 5)
        col_headers = [
            "Model 1\nGRU + Naive FT",
            "Model 2\nABK + Naive FT",
            "Model 3\nABK + Exp. Replay",
            "Model 4\nOracle Slots (Model C)",
            "Model 5\nAutonomous ABK (Model D)",
        ]
        for r_idx, (t_name, tgt_img) in enumerate(targets_info):
            for c_idx, (k, _, _) in enumerate(all_model_specs):
                ax = axes[r_idx, c_idx]
                canv, f1 = final_canvases[k][t_name]
                ax.imshow(overlay_target_canvas(canv, tgt_img))
                f1_color = "#2F855A" if f1 >= 0.80 else ("#D69E2E" if f1 >= 0.55 else "#C53030")
                title_prefix = f"{col_headers[c_idx]}\n" if r_idx == 0 else ""
                ax.set_title(f"{title_prefix}F1: {f1*100:.1f}%", color=f1_color, fontweight="bold", fontsize=9.5, pad=4)
                ax.set_xticks([])
                ax.set_yticks([])
                if c_idx == 0:
                    ax.set_ylabel(t_name.capitalize(), fontsize=11.5, fontweight="bold", labelpad=8)

        fig.tight_layout(rect=[0, 0.02, 1, 0.92])
        fig.subplots_adjust(top=0.88)
        fig.suptitle(
            "Qualitative Final Rollout Comparison (4 Tasks × 5 Models, faint blue = Target)",
            fontsize=14,
            fontweight="bold",
            y=0.96,
        )
        grid_path = safe_savefig(OUT_DIR / "autonomous_cl_rollout_grid.png")
        print(f"[*] 4×5 롤아웃 그리드 저장 -> {grid_path}")

    # ------------------------------------------------------------------
    # 시각화 4: Model D 자율 라우팅 가중치 4 × 4 히트맵 (autonomous_routing_matrix.png)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=200)
    fig.patch.set_facecolor("#F8F9FA")
    im = ax.imshow(routing_matrix * 100.0, cmap="YlGnBu", vmin=0.0, vmax=100.0)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"Slot {j}\n({t.capitalize()})" for j, t in enumerate(task_names)], fontsize=9.5)
    ax.set_yticklabels([f"Query:\n{t.capitalize()}" for t in task_names], fontsize=9.5)
    for i in range(K):
        for j in range(K):
            v = routing_matrix[i, j] * 100.0
            col = "white" if v > 55.0 else "black"
            ax.text(j, i, f"{v:.2f}%", ha="center", va="center", color=col, fontweight="bold", fontsize=10)
    ax.set_title(
        f"Model D Task-ID-Free Softmax Routing Weights (T={args.temperature:.2f})",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Routing Weight (%)")
    plt.tight_layout()
    rt_path = safe_savefig(OUT_DIR / "autonomous_routing_matrix.png")
    print(f"[*] 자율 라우팅 매트릭스 저장 -> {rt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="5-Model Task-ID-Free Autonomous CL Benchmark")
    parser.add_argument("--tasks", nargs="+", default=["apple", "sprout", "star", "cat"])
    parser.add_argument("--epochs-t0", type=int, default=10)
    parser.add_argument("--epochs-slot", type=int, default=12)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--exec-steps", type=int, default=16, help="청크당 실행 스텝 수 (기본 16)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--novelty-threshold", type=float, default=0.85)
    parser.add_argument("--replay-cap", type=int, default=256)
    parser.add_argument("--only-model-d", action="store_true", help="Model D만 단독 실행 및 벤치마크 갱신")
    parser.add_argument("--num-modes", type=int, default=6, help="Model D의 WTA 가설 수 (기본 6)")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
