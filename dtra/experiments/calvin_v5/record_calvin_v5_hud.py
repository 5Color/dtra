"""Unedited Pure RGB-D HUD rollout recorder for CALVIN V5 (DTRAv5TransportController)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from calvin_env.envs.play_table_env import get_env

sys.path.insert(0, os.path.expanduser("~/dtra_a6000_lab/calvin_benchmark"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "calvin_v4"))

from record_calvin_hud import make_hud_frame
from run_calvin_dtra_v5 import (
    CalvinCameraVisionAnchorEstimatorV5,
    DTRAv5TransportController,
    FrozenPerceptionEncoder,
    load_episodes,
    set_seed,
)


def main() -> None:
    set_seed(42)
    data_dir = os.path.expanduser("~/dtra_a6000_lab/calvin_data/task_D_D_subset")
    calvin_root = os.path.expanduser("~/dtra_a6000_lab/calvin_benchmark/calvin")
    out_dir = os.path.expanduser("~/dtra_a6000_lab/calvin_benchmark")

    train_tasks = load_episodes(os.path.join(data_dir, "training"), max_per_task=10)
    val_tasks = load_episodes(os.path.join(data_dir, "validation"), max_per_task=5)
    all_task_names = sorted(list(set(train_tasks.keys()) | set(val_tasks.keys())))

    env = get_env(os.path.join(data_dir, "validation"), show_gui=False)
    task_oracle = hydra.utils.instantiate(
        OmegaConf.load(os.path.join(calvin_root, "calvin_env/conf/tasks/new_playtable_tasks.yaml"))
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = FrozenPerceptionEncoder(device)
    vis_est = CalvinCameraVisionAnchorEstimatorV5(env.cameras[0])
    policy = DTRAv5TransportController(
        encoder, device, anchor_estimator=vis_est, use_transport=True, num_knots=10
    )

    print("Training DTRA v5 (Pure RGB-D) across 34 CALVIN tasks...")
    for tn in all_task_names:
        policy.train_task(train_tasks[tn])

    demo_tasks = [
        "lift_pink_block_drawer",
        "push_into_drawer",
        "stack_block",
        "push_pink_block_right",
        "place_in_slider",
    ]
    frames_out = []
    for tn in demo_tasks:
        chosen_ep = None
        for ep in val_tasks[tn]:
            f0 = ep["frames"][0]
            obs = env.reset(robot_obs=f0["robot_obs"], scene_obs=f0["scene_obs"])
            s_info = env.get_info()
            policy.reset_episode(ep["ann"], obs, info=s_info)
            ep_frames = []
            solved = False
            max_steps = len(ep["frames"]) + 30
            lat_ms = 0.0
            st = 0
            for st in range(max_steps):
                t0 = time.perf_counter()
                act = policy.act(obs)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                obs, _, _, c_info = env.step(act)
                if tn in task_oracle.get_task_info_for_set(s_info, c_info, {tn}):
                    solved = True
                anc = policy.transport.state.cur_obj_pos.copy()
                anc_valid = (
                    policy.active_cell["cell_anchor"] is not None
                    and policy.active_cell["cell_anchor"].target_name != "fixed"
                )
                phase_str = (
                    "V5 PHASE-0 APPROACH"
                    if (policy.use_transport and anc_valid and not policy.transport.state.approach_done)
                    else "V5 PHASE-1 HYPERTUBE"
                )
                if st % 2 == 0 or solved:
                    ep_frames.append(
                        make_hud_frame(
                            obs["rgb_obs"]["rgb_static"],
                            obs["rgb_obs"]["rgb_gripper"],
                            env.cameras[0],
                            tn,
                            ep["ann"],
                            policy.active_cell["task"],
                            phase_str,
                            st + 1,
                            max_steps,
                            anc,
                            anc_valid,
                            policy.parameter_drift(),
                            lat_ms,
                            solved,
                            False,
                        )
                    )
                if solved:
                    ep_frames.extend([ep_frames[-1]] * 5)
                    break
            policy.episode_snap_back()
            ep_frames.append(
                make_hud_frame(
                    obs["rgb_obs"]["rgb_static"],
                    obs["rgb_obs"]["rgb_gripper"],
                    env.cameras[0],
                    tn,
                    ep["ann"],
                    policy.active_cell["task"],
                    "EPISODE SNAP-BACK",
                    st + 1,
                    max_steps,
                    np.zeros(3, dtype=np.float32),
                    False,
                    policy.parameter_drift(),
                    lat_ms,
                    solved,
                    True,
                )
            )
            ep_frames.extend([ep_frames[-1]] * 2)
            chosen_ep = (solved, st + 1, ep_frames)
            if solved:
                break
        solved, used_steps, ep_frames = chosen_ep
        print(f"  Recorded unedited V5 {tn}: solved={solved}, steps={used_steps}, frames={len(ep_frames)}")
        frames_out.extend(ep_frames)

    env.close()

    gif_path = os.path.join(out_dir, "calvin_v5_pure_vision_hud_rollout.gif")
    pil_frames = [Image.fromarray(f) for f in frames_out]
    pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=90, loop=0, optimize=True)
    print(f"Saved V5 GIF: {gif_path} ({len(pil_frames)} frames)")

    mp4_path = os.path.join(out_dir, "calvin_v5_pure_vision_hud_rollout.mp4")
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-r",
            "12",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            mp4_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for im in pil_frames:
            im.save(proc.stdin, format="PNG")
        proc.stdin.close()
        proc.wait()
        print(f"Saved V5 MP4: {mp4_path}")
    else:
        import imageio
        imageio.mimsave(mp4_path, frames_out, fps=12)
        print(f"Saved V5 MP4 via imageio: {mp4_path}")


if __name__ == "__main__":
    main()
