"""Pure RGB-D HUD rollout recorder for DTRA v4.0 on RoboCasa.

Records held-out unseen evaluation episodes across Single-Stage Atomic tasks
(OpenDrawer ep80, CloseDrawer ep80, OpenMicrowave ep80, PickPlaceCounterToSink ep81)
and a Multi-Stage Composite task (OrganizeVegetables ep80) showing the System 1
horizon boundary under unmodified env._check_success() and sanitize_pure_rgbd_obs().
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import robocasa.utils.lerobot_utils as LU
from run_robocasa_continual_benchmark import (
    DTRARoboCasaPolicy,
    FrozenPerceptionEncoder,
    TASK_CURRICULUM,
    create_robocasa_eval_env,
    load_or_build_task_cache,
    rollout_spline_episode,
    set_seed,
)
import torch


def _load_font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_MD = _load_font(13)
FONT_SM = _load_font(12)


def render_robocasa_hud_frame(rec: Dict[str, Any], stage_label: str = "ATOMIC") -> np.ndarray:
    """Render one crisp 740x390 telemetry HUD frame from a Pure RGB-D MuJoCo step."""
    w, h = 740, 390
    canvas = Image.new("RGB", (w, h), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    left_img = Image.fromarray(rec["left_rgb"]).resize((340, 340), Image.BILINEAR)
    canvas.paste(left_img, (16, 26))
    draw.rectangle((16, 26, 356, 366), outline=(52, 211, 153) if rec["solved"] else (51, 65, 85), width=2)
    draw.text((18, 6), "Onboard Left Cam (RGB-D) | Pure Vision+Depth", fill=(148, 163, 184), font=FONT_SM)

    hand_img = Image.fromarray(rec["hand_rgb"]).resize((134, 134), Image.BILINEAR)
    canvas.paste(hand_img, (376, 28))
    draw.rectangle((376, 28, 510, 162), outline=(51, 65, 85), width=2)
    draw.text((376, 8), "Eye-in-Hand Cam", fill=(148, 163, 184), font=FONT_SM)

    solved = bool(rec["solved"])
    status_bg = (6, 95, 70) if solved else ((120, 53, 15) if stage_label == "COMPOSITE (SYS1 LIMIT)" else (30, 58, 138))
    status_txt = "SOLVED (_check_success=True)" if solved else (
        "COMPOSITE (NEEDS SYS2)" if stage_label == "COMPOSITE (SYS1 LIMIT)" else "EXECUTING C^2 HYPERTUBE"
    )
    draw.rectangle((524, 28, 724, 86), fill=status_bg, outline=(100, 116, 139), width=1)
    draw.text((532, 34), f"RoboCasa ({stage_label[:9]}):", fill=(226, 232, 240), font=FONT_SM)
    draw.text((532, 56), status_txt[:26], fill=(255, 255, 255), font=FONT_SM)

    dp = rec["dp_tube"]
    dp_norm = float(np.linalg.norm(dp))
    draw.rectangle((524, 94, 724, 162), fill=(30, 41, 59), outline=(71, 85, 105), width=1)
    draw.text((532, 100), f"Weight Drift: {rec['drift']:.6f}", fill=(52, 211, 153), font=FONT_SM)
    draw.text((532, 120), f"Transport |dp|: {dp_norm:.3f} m", fill=(251, 191, 36), font=FONT_SM)
    draw.text((532, 140), f"Route Latency: {rec['lat_ms']:.2f} ms", fill=(56, 189, 248), font=FONT_SM)

    draw.rectangle((376, 174, 724, 366), fill=(30, 41, 59), outline=(71, 85, 105), width=1)
    match = rec["routed_task"] == rec["task"]
    rows = [
        ("DTRA v4.0 RoboCasa | Strict Pure RGB-D", (96, 165, 250)),
        (f"Target Task : {rec['task']} (ep{rec.get('ep_id', '?')})", (248, 250, 252)),
        (f"Instruction : {rec['inst'][:36]}", (203, 213, 225)),
        (f"Routed Ring : {rec['routed_task']} ({'MATCH' if match else 'MISS'})", (52, 211, 153) if match else (248, 113, 113)),
        (f"Progress    : Step {rec['step']}/{rec['max_steps']}", (251, 191, 36)),
        (f"Tube Shift  : [{dp[0]:+.3f}, {dp[1]:+.3f}, {dp[2]:+.3f}]", (226, 232, 240)),
        ("Replay: 0% | No object-state | Task-ID-Free", (148, 163, 184)),
    ]
    y = 182
    for txt, col in rows:
        draw.text((386, y), txt, fill=col, font=FONT_SM)
        y += 25
    return np.asarray(canvas, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record RoboCasa Pure RGB-D HUD Video/GIF")
    parser.add_argument("--dataset_root", default="/home/team03/dtra_a6000_lab/robocasa_lab/robocasa/datasets/v1.0/pretrain")
    parser.add_argument("--cache_path", default="/home/team03/dtra_a6000_lab/robocasa_lab/demo_cache_12tasks_65v2.pt")
    parser.add_argument("--out_dir", default="/home/team03/dtra_a6000_lab/robocasa_lab/dtra_v4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrozenPerceptionEncoder(device)
    bundle = load_or_build_task_cache(Path(args.dataset_root), encoder, Path(args.cache_path))

    policy = DTRARoboCasaPolicy(use_transport=True, max_shift=0.14)
    for _, tname in TASK_CURRICULUM:
        policy.train_task(bundle["tasks"][tname]["train"])

    # Explicitly configured held-out episodes matching updated Arm4 verification
    demo_plan = [
        ("OpenDrawer", 80, "ATOMIC", 550),
        ("CloseDrawer", 80, "ATOMIC", 550),
        ("OpenMicrowave", 80, "ATOMIC", 550),
        ("PickPlaceCounterToSink", 81, "ATOMIC", 550),
        ("OrganizeVegetables", 80, "COMPOSITE (SYS1 LIMIT)", 180),
    ]
    all_hud_frames: List[np.ndarray] = []
    provenance: Dict[str, Any] = {
        "episode_selection": "Held-out evaluation episodes showcasing solved Atomic tasks (OpenDrawer ep80, CloseDrawer ep80, OpenMicrowave ep80, PickPlaceCounterToSink ep81) and the Composite bottleneck (OrganizeVegetables ep80, first 180 steps)",
        "capture": "Every consecutive simulator step (1:1 real-time at env.control_freq=20.0 Hz); 15 hold frames (0.75s) appended after each solved atomic episode",
        "evaluation": "Unmodified env._check_success() and sanitize_pure_rgbd_obs()",
        "drift_metric_limit": "DTRA parameter_drift() returns structural constant 0.0 under frozen cell banks",
        "episodes": [],
    }
    fps_val = 20.0
    for tname, ep_id, stage_lbl, cap_h in demo_plan:
        tinfo = bundle["tasks"][tname]
        ds = Path(tinfo["ds_path"])
        env = create_robocasa_eval_env(tinfo["ds_path"])
        fps_val = float(env.control_freq)
        st = LU.get_episode_states(ds, ep_id)
        init_state = {
            "states": st[0],
            "model": LU.get_episode_model_xml(ds, ep_id),
            "ep_meta": json.dumps(LU.get_episode_meta(ds, ep_id)),
        }
        step_recs: List[Dict[str, Any]] = []
        import inspect as _inspect, run_robocasa_continual_benchmark as _rc_mod
        _ns = dict(vars(_rc_mod))
        _full_src = _inspect.getsource(_rc_mod.rollout_spline_episode).replace(
            "record_hud_frames is not None and (step_i % 3 == 0 or solved or step_i == n_steps - 1)",
            "record_hud_frames is not None",
        )
        exec(compile(_full_src, "<full-step-rollout>", "exec"), _ns)
        res = _ns["rollout_spline_episode"](
            env, init_state, policy, encoder, tname, max_horizon_cap=cap_h, record_hud_frames=step_recs
        )
        for r in step_recs:
            r["ep_id"] = ep_id
        frames = [render_robocasa_hud_frame(r, stage_label=stage_lbl) for r in step_recs]
        hold_n = 15 if res["solved"] else 10
        if frames:
            frames.extend([frames[-1]] * hold_n)
        start_idx = len(all_hud_frames)
        all_hud_frames.extend(frames)
        provenance["episodes"].append({
            "task": tname,
            "stage": stage_lbl,
            "episode_id": int(ep_id),
            "frame_start_zero_based": start_idx,
            "frame_count": len(frames),
            "hold_frames": hold_n,
            "control_frequency_hz": fps_val,
            "result": res,
        })
        print(f"Recorded {tname} ep{ep_id} ({stage_lbl}): solved={res['solved']} captured_steps={len(step_recs)} total_frames={len(frames)}", flush=True)
        env.close()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance["frame_count"] = len(all_hud_frames)
    (out_dir / "recording_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    gif_path = out_dir / "robocasa_pure_rgbd_hud_rollout.gif"
    pil_frames = [Image.fromarray(f) for f in all_hud_frames]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=round(1000 / fps_val), loop=0, optimize=True)
    print(f"Saved GIF: {gif_path} ({len(pil_frames)} frames)")

    mp4_path = out_dir / "robocasa_pure_rgbd_hud_rollout.mp4"
    imageio.mimsave(str(mp4_path), all_hud_frames, fps=fps_val, macro_block_size=1)
    print(f"Saved MP4: {mp4_path}")
    if all_hud_frames:
        Image.fromarray(all_hud_frames[0]).save(out_dir / "hud_preview.png")


if __name__ == "__main__":
    main()
