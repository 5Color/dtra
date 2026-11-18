"""train/rollout_bspline.py

[Phase 1 & 2 실증] ABK-VLA 정책의 폐루프(Closed-Loop) 자율 드로잉 롤아웃 평가기.
- 빈 캔버스 + 목표 도안에서 출발하여 정책이 실시간으로 제어점(Control Points)을 예측하며 그림을 완성.
- B-Spline 불변 기저를 통한 가속도 연속($C^2$) 궤적 생성 및 저주파 물결침(Wave Degrees) 억제 검증.
- 실시간 제어 다각형(Control Polygon) 및 변곡점 매듭 시각화 애니메이션 GIF 생성.
- 최종 복원율(Coverage, Precision, F1) 및 매끈함(Turn/Wave Deg) 정량 산출.

사용법:
  # 기본 롤아웃 (사과 도안, GIF 및 최종 PNG 생성)
  python train/rollout_bspline.py --ckpt outputs/bspline_vla_policy_test_mlp.pth --target apple --exec-steps 8

  # 실행 스텝 12 설정 및 고해상도 시각화
  python train/rollout_bspline.py --ckpt outputs/bspline_vla_policy_test_mlp.pth --exec-steps 12
"""

from __future__ import annotations

import argparse
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
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "v1_drawing_vla" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dtra.core.bspline_core import (
    BSplineDecoder,
    build_clamped_knot_vector,
    reconstruct_bspline_np,
)
from train.bspline_policy import BSplineCVAEPolicy, BSplineVisuomotorPolicy
from train.dataset import (
    HOME_STATE,
    OBS_SIZE,
    STROKE_WIDTH_PX,
    SUPERSAMPLE,
    decode_png_gray,
    pen_down_runs,
    render_canvas,
    render_polyline,
)

OUT_DIR = PROJECT_ROOT / "outputs" / "v1_archive"
TILE = 384
SCALE = TILE // OBS_SIZE


# ---------------------------------------------------------------------------
# 디바이스 유틸
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


def _get_font(size: int = 15):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 평가 메트릭 (정밀도, 커버리지, F1, 선의 매끈함)
# ---------------------------------------------------------------------------

def dilate(m: np.ndarray, r: int) -> np.ndarray:
    out, (H, W) = m.copy(), m.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            s = np.zeros_like(m)
            s[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = m[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
            out |= s
    return out


def match_scores(canvas: np.ndarray, target: np.ndarray, tol: int = 2) -> Dict[str, float]:
    R, T = canvas > 64, target > 64
    cov = float((T & dilate(R, tol)).sum() / max(1, T.sum()))
    prec = float((R & dilate(T, tol)).sum() / max(1, R.sum())) if R.any() else 0.0
    f1 = 2 * cov * prec / (cov + prec) if (cov + prec) > 0 else 0.0
    return {"coverage": cov, "precision": prec, "f1": f1}


def compute_smoothness(traj: np.ndarray, step: float = 0.005) -> Dict[str, float]:
    """궤적의 매끈함 지표:
    turn_deg = 이웃 점 사이의 평균 꺾임 각도 (작을수록 부드러움)
    wave_deg = 꺾임 각도의 변화율 (물결/지그재그 진동일수록 큼 - 작을수록 물리적 관성 우수)
    """
    turns, waves = [], []
    T = np.asarray(traj, np.float64)
    runs = pen_down_runs(T.astype(np.float32))

    for a0, a1 in runs:
        P = T[a0:a1, :2]
        if len(P) < 3:
            continue
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        s_ = np.concatenate([[0.0], np.cumsum(seg)])
        if s_[-1] < step * 4:
            continue
        u = np.arange(0.0, s_[-1], step)
        R = np.column_stack([np.interp(u, s_, P[:, 0]), np.interp(u, s_, P[:, 1])])
        d = np.diff(R, axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        da = np.angle(np.exp(1j * np.diff(ang)))
        turns.append(np.abs(da))
        waves.append(np.abs(np.diff(da)))

    t = np.concatenate(turns) if turns else np.zeros(1)
    w = np.concatenate(waves) if waves else np.zeros(1)
    return {
        "turn_deg": float(np.degrees(t.mean())),
        "wave_deg": float(np.degrees(w.mean())),
    }


# ---------------------------------------------------------------------------
# 롤아웃 시뮬레이션
# ---------------------------------------------------------------------------

class IncrementalCanvas:
    def __init__(self, size: int = OBS_SIZE):
        self.size = size
        self.done = np.zeros((size, size), np.uint8)
        self.cur: list[tuple[float, float]] = []

    def add(self, x: float, y: float, pen: float) -> None:
        if pen > 0.5:
            self.cur.append((x, y))
        elif self.cur:
            self.done = np.maximum(self.done, render_polyline(np.asarray(self.cur), size=self.size))
            self.cur = []

    def image(self) -> np.ndarray:
        return np.maximum(self.done, render_polyline(np.asarray(self.cur), size=self.size)) if self.cur else self.done


def load_target_image(target_name: str) -> np.ndarray:
    """타겟 도안의 정답 128x128 캔버스 로드."""
    # 1. targets_128.json 탐색
    targets_json = PROJECT_ROOT / "v1_drawing_vla" / "handvsai" / "targets_128.json"
    if targets_json.exists():
        with open(targets_json, "r", encoding="utf-8") as f:
            d = json.load(f)
            if target_name in d:
                val = d[target_name]
                if isinstance(val, dict) and "png_128" in val:
                    return decode_png_gray(val["png_128"])
                elif isinstance(val, str):
                    return decode_png_gray(val)

    # 2. data/raw/demo_{target}_*.json 탐색
    matches = sorted((PROJECT_ROOT / "data" / "raw").glob(f"demo_{target_name}_*.json"))
    if matches:
        with open(matches[0], "r", encoding="utf-8") as f:
            demo_d = json.load(f)
            if "target_128_png" in demo_d:
                return decode_png_gray(demo_d["target_128_png"])

    raise FileNotFoundError(f"타겟 도안 '{target_name}'을 찾을 수 없습니다.")


@torch.no_grad()
def rollout_bspline(
    policy: torch.nn.Module,
    target_img: np.ndarray,
    device: torch.device,
    max_iters: int = 70,
    exec_steps: int = 8,
    patience: int = 6,
) -> Tuple[np.ndarray, List[Dict], str, List[float]]:
    """ABK-VLA 정책을 통한 폐루프 자율 롤아웃 실행."""
    ink = IncrementalCanvas()
    canvas = ink.image()
    state = np.array(HOME_STATE, dtype=np.float32)
    tgt_tensor = target_img.astype(np.float32) / 255.0

    executed: List[List[float]] = []
    history: List[Dict] = []
    infer_latencies: List[float] = []

    still = 0
    stale = 0
    reason = f"최대 {max_iters}회 반복 도달"

    for it in range(max_iters):
        # 1. 관측 텐서 구성: [canvas, target] (1, 2, 128, 128)
        canv_norm = canvas.astype(np.float32) / 255.0
        obs_tensor = torch.from_numpy(np.stack([canv_norm, tgt_tensor]))[None].to(device)
        st_tensor = torch.from_numpy(state)[None].to(device)

        # 2. 제어점 및 B-Spline 물리 궤적 추론 (지연 시간 측정)
        t0 = time.perf_counter()
        pred_cp_t, pred_traj_t, pen_prob_t = policy.predict(obs_tensor, st_tensor)
        sync_dev(device)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        infer_latencies.append(latency_ms)

        pred_cp = pred_cp_t[0].cpu().numpy()          # (M, 2)
        pred_traj = pred_traj_t[0].cpu().numpy()      # (H, 2)
        pen_probs = pen_prob_t[0].cpu().numpy()      # (H,)
        pen_down = (pen_probs > 0.5).astype(np.float32)

        # 청크 간 C^0/C^1 접선 연속성 보정:
        # 연속된 그리기(pen_down) 구간에서 이전 청크의 끝점과 매끄럽게 연결하고,
        # 완만한 곡선(cos_sim > 0.0)에서는 이전 진행 방향을 부드럽게 유지
        if it > 0 and len(executed) >= 2 and pen_down[0] > 0.5 and executed[-1][2] > 0.5:
            p_prev = np.array(executed[-1][:2], dtype=np.float32)
            v_prev = p_prev - np.array(executed[-2][:2], dtype=np.float32)
            pred_traj[0] = p_prev  # C^0 위치 오차 0 보정
            v_model = pred_traj[1] - pred_traj[0]
            norm_prev = float(np.linalg.norm(v_prev))
            norm_model = float(np.linalg.norm(v_model))
            if norm_prev > 1e-5 and norm_model > 1e-5:
                cos_a = float(np.dot(v_prev, v_model) / (norm_prev * norm_model))
                # 급격한 꼭짓점(각도 > 90도)에서는 모델의 코너 꺾기를 존중하고, 완만한 곡선에서만 접선 스무딩
                if cos_a > 0.0:
                    pred_traj[1] = 0.7 * pred_traj[1] + 0.3 * (p_prev + v_prev)

        chunk = np.column_stack([pred_traj, pen_down])  # (H, 3)

        # 3. 앞 exec_steps개 스텝만 실행 (Receding Horizon Execution)
        run_steps = min(exec_steps, len(chunk))
        for step_i in range(run_steps):
            pt = chunk[step_i]
            executed.append(pt.tolist())
            ink.add(pt[0], pt[1], pt[2])

        # 펜 상태 갱신
        state = np.array(executed[-1], dtype=np.float32)
        canvas = ink.image()

        prev_canvas = history[-1]["canvas"] if history else np.zeros_like(canvas)
        new_ink = int((canvas.astype(np.int16) - prev_canvas > 32).sum())

        history.append({
            "iter": it,
            "control_points": pred_cp,
            "chunk_traj": chunk,
            "canvas": canvas.copy(),
            "n_exec": len(executed),
            "new_ink": new_ink,
            "latency_ms": latency_ms,
        })

        # 종료 조건 1: 연속 펜-업 상태로 정지 (도안 완성 신호)
        is_pen_up_idle = (pen_down.max() < 0.5) and (np.max(np.abs(pred_traj - pred_traj[0])) < 0.02)
        still = still + 1 if (is_pen_up_idle and it > 3) else 0
        if still >= 4:
            reason = "스스로 종료 (도안 완성 및 펜-업 정지 감지)"
            break

        # 종료 조건 2: 새 잉크가 patience회 연속 0 (진행 멈춤)
        # 잔여 미완성 잉크가 5% 이상 남아있고 공중 점프 비행 중이면 점프 완료 기회 보장
        rem_ink = int((target_img.astype(np.int16) - canvas.astype(np.int16) > 32).sum())
        total_tgt_ink = max(1, int((target_img > 32).sum()))
        rem_ratio = float(rem_ink / total_tgt_ink)
        is_flying = (pen_down.max() < 0.5) and (np.max(np.abs(pred_traj - pred_traj[0])) >= 0.02)

        if new_ink > 0:
            stale = 0
        elif is_flying and rem_ratio > 0.05:
            # 다음 파트로 공중 이동 중인 경우 stale 축적 속도 완화
            stale = stale if (it % 3 != 0) else (stale + 1)
        else:
            stale = stale + 1 if it > 3 else 0

        effective_patience = (patience + 8) if rem_ratio > 0.05 else patience
        if stale >= effective_patience:
            reason = f"진행 정체 감지 ({effective_patience}회 연속 새 잉크 없음)"
            break

    return np.array(executed, dtype=np.float32), history, reason, infer_latencies


def sync_dev(dev: torch.device):
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# 시각화 및 애니메이션 프레임 생성
# ---------------------------------------------------------------------------

def render_frame(
    target_img: np.ndarray,
    h_step: Dict,
    exec_steps: int,
    it: int,
    total_iters: int,
) -> Image.Image:
    """롤아웃 진행 상황 프레임 생성: 도안 배경 + 누적 캔버스 + 실시간 B-Spline 제어 다각형."""
    # 1. 배경 (타겟 도안 연한 회색)
    base_img = Image.new("RGB", (TILE, TILE), (245, 247, 250))
    tgt_pil = Image.fromarray(target_img).resize((TILE, TILE), Image.Resampling.NEAREST)
    tgt_arr = np.array(tgt_pil)
    
    # 목표 도안 그리기 (부드러운 블루-그레이)
    mask = tgt_arr > 64
    base_arr = np.array(base_img)
    base_arr[mask] = [200, 210, 225]
    
    # 2. 누적 캔버스 그리기 (짙은 차콜)
    canv_pil = Image.fromarray(h_step["canvas"]).resize((TILE, TILE), Image.Resampling.NEAREST)
    canv_mask = np.array(canv_pil) > 64
    base_arr[canv_mask] = [26, 32, 44]
    
    frame_img = Image.fromarray(base_arr)
    draw = ImageDraw.Draw(frame_img)

    # 3. 실시간 제어 다각형(Control Polygon) 및 제어점 그리기
    cps = h_step["control_points"] * TILE
    for i in range(len(cps) - 1):
        draw.line([tuple(cps[i]), tuple(cps[i + 1])], fill=(229, 62, 62, 180), width=2)
    for i, (cx, cy) in enumerate(cps):
        r = 4 if (i == 0 or i == len(cps) - 1) else 3
        col = (229, 62, 62) if i == 0 else ((49, 130, 206) if i < len(cps) - 1 else (56, 161, 105))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col, outline=(26, 32, 44), width=1)

    # 4. 상단 정보 배너
    banner_h = 36
    final_canvas = Image.new("RGB", (TILE, TILE + banner_h), (26, 32, 44))
    final_canvas.paste(frame_img, (0, banner_h))

    banner_draw = ImageDraw.Draw(final_canvas)
    header_text = f"Step {h_step['n_exec']} | Iter {it + 1}/{total_iters} | Infer: {h_step['latency_ms']:.1f}ms"
    banner_draw.text((10, 9), header_text, fill=(240, 240, 240), font=_get_font(13))
    banner_draw.text((TILE - 95, 9), "ABK-VLA (M=6)", fill=(72, 187, 120), font=_get_font(13))

    return final_canvas


def save_final_summary_plot(
    target_img: np.ndarray,
    final_canvas: np.ndarray,
    history: List[Dict],
    executed_traj: np.ndarray,
    scores: Dict[str, float],
    smooth: Dict[str, float],
    out_path: Path,
):
    """최종 결과 고해상도 3패널 비교 이미지 생성."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
    fig.patch.set_facecolor("#F8F9FA")

    # 패널 1: 목표 도안
    ax0 = axes[0]
    ax0.set_facecolor("#FFFFFF")
    ax0.imshow(target_img, cmap="gray_r", extent=[0, 1, 1, 0])
    ax0.set_title("1. Ground Truth Target (도안)", fontsize=11, fontweight="bold", pad=8)
    ax0.set_xlim(-0.05, 1.05); ax0.set_ylim(1.05, -0.05)
    ax0.set_aspect("equal"); ax0.grid(True, linestyle="--", alpha=0.3)

    # 패널 2: ABK-VLA 자율 렌더 캔버스
    ax1 = axes[1]
    ax1.set_facecolor("#FFFFFF")
    ax1.imshow(final_canvas, cmap="gray_r", extent=[0, 1, 1, 0])
    f1_str = f"F1: {scores['f1']*100:.1f}% (Cov: {scores['coverage']*100:.1f}%, Prec: {scores['precision']*100:.1f}%)"
    ax1.set_title(f"2. Closed-Loop Rollout Canvas\n({f1_str})", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlim(-0.05, 1.05); ax1.set_ylim(1.05, -0.05)
    ax1.set_aspect("equal"); ax1.grid(True, linestyle="--", alpha=0.3)

    # 패널 3: 제어 다각형 (Control Polygons) 및 변곡점 매듭 골격
    ax2 = axes[2]
    ax2.set_facecolor("#FFFFFF")
    ax2.imshow(target_img, cmap="gray_r", extent=[0, 1, 1, 0], alpha=0.15)
    
    cmap = plt.get_cmap("tab20", len(history))
    for idx, h in enumerate(history):
        cps = h["control_points"]
        col = cmap(idx)
        ax2.plot(cps[:, 0], cps[:, 1], color=col, lw=1.2, linestyle="-.", alpha=0.7)
        ax2.scatter(cps[:, 0], cps[:, 1], color=col, s=25, edgecolor="#1A202C", lw=0.6, zorder=5)

    smooth_str = f"Turn: {smooth['turn_deg']:.2f}° | Wave (Wobble): {smooth['wave_deg']:.2f}°"
    ax2.set_title(f"3. Predicted Control Polygons & Knots\n({smooth_str})", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlim(-0.05, 1.05); ax2.set_ylim(1.05, -0.05)
    ax2.set_aspect("equal"); ax2.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(
        f"[ABK-VLA Closed-Loop Rollout Evaluation]\n"
        f"M=6 B-Spline Knots | C^2 Physical Inertia Guarantee | Total Executed Steps: {len(executed_traj)}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[*] 고해상도 최종 비교 이미지 저장 완료 -> {out_path}")


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate ABK-VLA Policy with Closed-Loop Rollout")
    parser.add_argument("--ckpt", type=str, default="outputs/bspline_vla_policy_test_mlp.pth", help="체크포인트 경로")
    parser.add_argument("--target", type=str, default="apple", help="평가 목표 도안 이름")
    parser.add_argument("--max-iters", type=int, default=70, help="최대 롤아웃 반복 횟수 (기본 70)")
    parser.add_argument("--exec-steps", type=int, default=8, help="반복당 실행할 스텝 수 (기본 8)")
    parser.add_argument("--patience", type=int, default=6, help="새 잉크 없을 때 조기 종료 인내치")
    parser.add_argument("--gif-fps", type=int, default=10, help="애니메이션 GIF FPS (기본 10)")
    args = parser.parse_args()

    device = pick_device()
    ckpt_path = PROJECT_ROOT / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

    print("=" * 75)
    print("      [ABK-VLA] Closed-Loop Visual Rollout & Waviness Evaluation       ")
    print("=" * 75)
    print(f"[*] 로드 체크포인트: {ckpt_path.name}")
    print(f"[*] 목표 도안: {args.target}")

    # 체크포인트 로드
    ckpt = torch.load(ckpt_path, map_location=device)
    m = ckpt.get("num_control_points", 6)
    h = ckpt.get("chunk_size", 16)
    head = ckpt.get("head", "mlp")

    if head == "cvae":
        policy = BSplineCVAEPolicy(num_control_points=m, horizon=h)
    else:
        policy = BSplineVisuomotorPolicy(num_control_points=m, horizon=h)

    policy.load_state_dict(ckpt["state_dict"])
    policy.to(device)
    policy.eval()
    print(f"[*] 정책 네트워크 준비 완료: {policy.__class__.__name__} (M={m}, H={h})")

    # 목표 도안 로드
    target_img = load_target_image(args.target)
    print(f"[*] 목표 도안 이미지 로드 완료 (128x128)")

    # 롤아웃 실행
    print(f"[*] 폐루프 자율 드로잉 시뮬레이션 가동 (exec_steps={args.exec_steps})...")
    t0_rollout = time.perf_counter()
    executed, history, stop_reason, latencies = rollout_bspline(
        policy,
        target_img,
        device=device,
        max_iters=args.max_iters,
        exec_steps=args.exec_steps,
        patience=args.patience,
    )
    total_time = time.perf_counter() - t0_rollout
    print(f"[*] 롤아웃 종료 ({total_time:.2f}초): {stop_reason} (총 {len(executed)}스텝, {len(history)}회 청크 실행)")

    # 정량 평가
    final_canvas = history[-1]["canvas"]
    scores = match_scores(final_canvas, target_img)
    smooth = compute_smoothness(executed)
    p50_lat = float(np.median(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    print("\n" + "-" * 50)
    print("               [정량 평가 결과표]                 ")
    print("-" * 50)
    print(f"  • 도안 일치도 F1 Score : {scores['f1']*100:.2f}%")
    print(f"  • 도안 커버리지(Coverage): {scores['coverage']*100:.2f}%")
    print(f"  • 선의 정밀도(Precision): {scores['precision']*100:.2f}%")
    print(f"  • 평균 꺾임 각도 (Turn): {smooth['turn_deg']:.2f}°")
    print(f"  • 저주파 물결침 (Wave) : {smooth['wave_deg']:.2f}°  (낮을수록 떨림 없음)")
    print(f"  • 추론 지연 시간       : p50={p50_lat:.1f}ms, p95={p95_lat:.1f}ms")
    print("-" * 50)

    # 1. 애니메이션 GIF 저장
    gif_path = OUT_DIR / f"rollout_bspline_{args.target}.gif"
    frames = [
        render_frame(target_img, h_step, args.exec_steps, idx, len(history))
        for idx, h_step in enumerate(history)
    ]
    if frames:
        # 마지막 완성 프레임은 1초간 머무르도록 10회 복제 추가
        frames_extended = frames + [frames[-1]] * 10
        frames_extended[0].save(
            gif_path,
            save_all=True,
            append_images=frames_extended[1:],
            duration=int(1000 / args.gif_fps),
            loop=0,
        )
        print(f"[*] 자율 드로잉 애니메이션 GIF 저장 완료 -> {gif_path}")

    # 2. 고해상도 최종 비교 PNG 저장
    final_png_path = OUT_DIR / f"rollout_bspline_{args.target}_final.png"
    save_final_summary_plot(target_img, final_canvas, history, executed, scores, smooth, final_png_path)

    # 3. 리포트 JSON 저장
    result_json_path = OUT_DIR / f"rollout_bspline_{args.target}.json"
    report_data = {
        "target": args.target,
        "ckpt": str(ckpt_path.name),
        "total_steps": len(executed),
        "num_chunks": len(history),
        "stop_reason": stop_reason,
        "f1": scores["f1"],
        "coverage": scores["coverage"],
        "precision": scores["precision"],
        "turn_deg": smooth["turn_deg"],
        "wave_deg": smooth["wave_deg"],
        "latency_p50_ms": p50_lat,
        "latency_p95_ms": p95_lat,
    }
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[*] 결과 리포트 JSON 저장 -> {result_json_path}")


if __name__ == "__main__":
    main()
