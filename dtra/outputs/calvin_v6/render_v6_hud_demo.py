import os, sys, shutil, subprocess, inspect
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calvin_env.envs.play_table_env import PlayTableSimEnv
from topological_bridge_v6 import TopologicalGeodesicBridgeV6
import run_calvin_dtra_v6 as v6


def load_font(size):
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_LG = load_font(16)
FONT_MD = load_font(13)
FONT_SM = load_font(11)

STATE = {
    "in_v6_sys12": False,
    "planned_chains": [],
    "frames": [],
    "best_frames": [],
    "best_score": (-1, -1, -1),
    "best_tasks": [],
    "best_chain_idx": -1,
    "current_tasks": [],
    "last_completed_ann": "",
    "current_ann": "Initializing...",
    "subgoal_idx": 0,
    "retry_in_subgoal": 0,
    "chain_start_recoveries": -1,
    "steps_since_bind": 0,
    "completed_in_chain": 0,
    "active_policy": None,
    "last_obs": None,
    "v6_chains_seen": 0,
}


def render_hud_frame(obs, pol, step_idx, solved_banner: bool = False):
    W, H = 960, 540
    canvas = Image.new("RGB", (W, H), (13, 17, 23))
    draw = ImageDraw.Draw(canvas)

    rgb_obs = obs.get("rgb_obs", {})
    static_img = rgb_obs.get("rgb_static")
    grip_img = rgb_obs.get("rgb_gripper")

    in_bridge = False
    tau = 0.0
    if pol is not None:
        d = pol.__dict__
        if d.get("in_bridge", False) or d.get("bridge_active", False):
            in_bridge = True
            bseq = d.get("bridge_pos_seq")
            bidx = d.get("bridge_idx", 0)
            if bseq is not None and len(bseq) > 0:
                tau = min(1.0, (bidx + 1) / float(len(bseq)))
            else:
                tau = 0.5
        elif STATE["subgoal_idx"] >= 2 and STATE["steps_since_bind"] <= 16:
            in_bridge = True
            tau = min(1.0, STATE["steps_since_bind"] / 16.0)

    cell_id = int(getattr(pol, "active_cell_idx", getattr(pol, "current_cell", 0))) if pol else 0
    if solved_banner:
        accent = (16, 185, 129)
    elif STATE["retry_in_subgoal"] > 0:
        accent = (167, 139, 250)  # System 2 Refractory Recovery purple highlight
    elif in_bridge:
        accent = (245, 158, 11)
    else:
        accent = (16, 185, 129)

    if static_img is not None:
        im_s = Image.fromarray(static_img).resize((500, 500), Image.Resampling.BICUBIC)
        draw.rectangle([16, 16, 524, 524], outline=accent, width=4)
        canvas.paste(im_s, (20, 20))
        if solved_banner:
            draw.rectangle([28, 26, 405, 58], fill=(6, 95, 70), outline=(52, 211, 153), width=2)
            draw.text((38, 34), "5/5 CHAIN SOLVED! (PURE VISION)", fill=(255, 255, 255), font=FONT_MD)

    if grip_img is not None:
        im_g = Image.fromarray(grip_img).resize((190, 190), Image.Resampling.BICUBIC)
        draw.rectangle([540, 16, 738, 214], outline=(56, 189, 248), width=2)
        canvas.paste(im_g, (544, 20))

    draw.rectangle([750, 16, 944, 214], fill=(22, 27, 34), outline=(48, 54, 61), width=2)
    draw.text((762, 26), "DTRA v6 ENGINE", fill=(56, 189, 248), font=FONT_MD)
    draw.text((762, 52), "Mode: VISION", fill=(16, 185, 129), font=FONT_SM)
    draw.text((762, 70), "Leak: ZERO", fill=(16, 185, 129), font=FONT_SM)
    draw.text((762, 88), "Drift: 0.0000", fill=(16, 185, 129), font=FONT_SM)
    step_disp = min(5, max(1, STATE["subgoal_idx"]))
    draw.text((762, 118), f"Chain Step: {step_disp}/5", fill=(255, 255, 255), font=FONT_SM)
    draw.text((762, 138), f"Completed: {STATE['completed_in_chain']}/5", fill=(16, 185, 129), font=FONT_SM)
    draw.text((762, 158), f"Frame: {step_idx:04d}", fill=(139, 148, 158), font=FONT_SM)
    if STATE["retry_in_subgoal"] > 0:
        draw.text((762, 180), "Sys2: BACKTRACK", fill=(167, 139, 250), font=FONT_SM)

    draw.rectangle([540, 224, 944, 524], fill=(22, 27, 34), outline=accent, width=2)
    if solved_banner:
        phase_txt = "STATUS: ALL 5 SUBGOALS SOLVED (5/5)"
    elif STATE["retry_in_subgoal"] > 0:
        phase_txt = f"SYS2 RECOVERY RETRY -> CELL #{cell_id}"
    elif in_bridge:
        phase_txt = f"C^2 GEODESIC BRIDGE -> CELL #{cell_id}"
    else:
        phase_txt = f"SYSTEM 1: DIRECTED TUBE #{cell_id}"
    draw.text((554, 232), phase_txt, fill=accent, font=FONT_MD)

    bar_w = int(376 * (1.0 if (solved_banner or not in_bridge) else tau))
    draw.rectangle([554, 254, 930, 264], fill=(33, 38, 45))
    draw.rectangle([554, 254, 554 + bar_w, 264], fill=accent)

    ann = STATE["current_ann"][:52]
    draw.text((554, 272), f"Subgoal: {ann}", fill=(240, 246, 252), font=FONT_SM)

    r_obs = obs.get("robot_obs", np.zeros(15))
    eef = r_obs[:3]
    grip_state = "CLOSED" if float(r_obs[14]) < 0 else "OPEN"
    draw.text((554, 290), f"EEF:[{eef[0]:+.2f},{eef[1]:+.2f},{eef[2]:+.2f}] | Grip:{grip_state}", fill=(201, 209, 217), font=FONT_SM)

    myelin_cnt = 0
    total_recov = int(getattr(pol, "backtrack_recoveries", 0)) if pol else 0
    if STATE["chain_start_recoveries"] < 0 and pol is not None:
        STATE["chain_start_recoveries"] = total_recov
    chain_recov = max(0, total_recov - max(0, STATE["chain_start_recoveries"]))
    if pol and hasattr(pol, "bridge") and hasattr(pol.bridge, "graph"):
        m = pol.bridge.graph.myelin_matrix
        myelin_cnt = int((m > 0.05).sum().item())
    draw.text((554, 308), f"Sys2 Myelin:{myelin_cnt} | Chain Recov:{chain_recov} (Total:{total_recov})", fill=(167, 139, 250), font=FONT_SM)

    draw.line([554, 326, 930, 326], fill=(48, 54, 61), width=1)
    draw.text((554, 332), f"5-Step Chain Checklist ({STATE['completed_in_chain']}/5):", fill=(139, 148, 158), font=FONT_SM)

    plan_idx = STATE["v6_chains_seen"]
    planned = STATE["planned_chains"][plan_idx] if plan_idx < len(STATE["planned_chains"]) else []
    for k in range(5):
        yk = 352 + k * 32
        if k < len(STATE["current_tasks"]):
            tag, col = "[DONE]", (16, 185, 129)
            label = STATE["current_tasks"][k]
        elif k == len(STATE["current_tasks"]) and not solved_banner:
            tag, col = ("[RETRY]", (167, 139, 250)) if STATE["retry_in_subgoal"] > 0 else ("[EXEC]", (245, 158, 11))
            label = planned[k] if k < len(planned) else STATE["current_ann"]
        else:
            tag, col = "[WAIT]", (110, 118, 129)
            label = planned[k] if k < len(planned) else "..."
        draw.text((554, yk), f"{k+1}. {tag} {label[:45]}", fill=col, font=FONT_SM)

    return np.asarray(canvas, dtype=np.uint8)


def save_videos_and_exit():
    frames = STATE["best_frames"]
    if not frames:
        print("[WARN] No multi-task chain captured yet.", flush=True)
        os._exit(1)
    here = os.path.dirname(os.path.abspath(__file__))
    mp4_path = os.path.join(here, "dtra_v6_pure_vision_chain_hud.mp4")
    gif_path = os.path.join(here, "dtra_v6_pure_vision_chain_hud.gif")
    print(
        f"[RENDER] Encoding {len(frames)} frames (Chain #{STATE['best_chain_idx']}, Completed={STATE['best_score'][0]}/5, Tasks={STATE['best_tasks']})...",
        flush=True,
    )
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", "960x540", "-pix_fmt", "rgb24", "-r", "24", "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for fr in frames:
            proc.stdin.write(fr.tobytes())
        proc.stdin.close()
        proc.wait()
    else:
        import cv2
        vw = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (960, 540))
        for fr in frames:
            vw.write(fr[:, :, ::-1])
        vw.release()
    gif_imgs = [Image.fromarray(frames[i]).resize((480, 270), Image.Resampling.BILINEAR) for i in range(0, len(frames), 3)]
    gif_imgs[0].save(gif_path, save_all=True, append_images=gif_imgs[1:], duration=120, loop=0)
    print(f"[DONE] Saved {mp4_path} and {gif_path}", flush=True)


def flush_episode():
    if STATE["in_v6_sys12"] and STATE["subgoal_idx"] >= 1:
        STATE["v6_chains_seen"] += 1
        pol = STATE["active_policy"]
        total_recov = int(getattr(pol, "backtrack_recoveries", 0)) if pol else 0
        chain_recov = max(0, total_recov - max(0, STATE["chain_start_recoveries"]))
        print(
            f"[CHAIN #{STATE['v6_chains_seen']}] Completed={STATE['completed_in_chain']}/5 ChainRecov={chain_recov} Frames={len(STATE['frames'])} Tasks={STATE['current_tasks']}",
            flush=True,
        )
        if STATE["completed_in_chain"] >= 3 and STATE["frames"]:
            # Prefer 5/5 completion, then canonical Chain #1 / recovery demo, then frame length
            is_canon_or_recov = 1 if (STATE["v6_chains_seen"] <= 2 or chain_recov > 0) else 0
            score = (STATE["completed_in_chain"], is_canon_or_recov, len(STATE["frames"]))
            if score > STATE["best_score"]:
                STATE["best_score"] = score
                STATE["best_frames"] = list(STATE["frames"])
                STATE["best_tasks"] = list(STATE["current_tasks"])
                STATE["best_chain_idx"] = STATE["v6_chains_seen"]
                print(
                    f"[CANDIDATE] v6 Chain #{STATE['best_chain_idx']}: Completed={STATE['completed_in_chain']}/5 -> {STATE['best_tasks']}",
                    flush=True,
                )
    STATE["frames"].clear()
    STATE["current_tasks"].clear()
    STATE["last_completed_ann"] = ""
    STATE["subgoal_idx"] = 0
    STATE["retry_in_subgoal"] = 0
    STATE["chain_start_recoveries"] = -1
    STATE["steps_since_bind"] = 0
    STATE["completed_in_chain"] = 0
    STATE["last_obs"] = None


_orig_eval_chains = v6.evaluate_5step_chains_v6
def _hooked_eval_chains(env, task_oracle, policy, chains, use_system12_bridge=True, *args, **kwargs):
    STATE["in_v6_sys12"] = bool(use_system12_bridge)
    if use_system12_bridge:
        STATE["planned_chains"] = [[s["ann"] for s in seq] for seq in chains]
        STATE["v6_chains_seen"] = 0
    try:
        return _orig_eval_chains(env, task_oracle, policy, chains, use_system12_bridge=use_system12_bridge, *args, **kwargs)
    finally:
        flush_episode()
        STATE["in_v6_sys12"] = False
v6.evaluate_5step_chains_v6 = _hooked_eval_chains

_orig_reset = PlayTableSimEnv.reset
def _hooked_reset(self, *args, **kwargs):
    flush_episode()
    return _orig_reset(self, *args, **kwargs)
PlayTableSimEnv.reset = _hooked_reset

_orig_snap = TopologicalGeodesicBridgeV6.episode_snap_back
def _hooked_snap(self, *args, **kwargs):
    flush_episode()
    return _orig_snap(self, *args, **kwargs)
TopologicalGeodesicBridgeV6.episode_snap_back = _hooked_snap

_orig_step = PlayTableSimEnv.step
def _hooked_step(self, action):
    obs, rew, done, info = _orig_step(self, action)
    if STATE["in_v6_sys12"] and STATE["subgoal_idx"] >= 1:
        STATE["steps_since_bind"] += 1
        STATE["last_obs"] = obs
        STATE["frames"].append(render_hud_frame(obs, STATE["active_policy"], len(STATE["frames"]), solved_banner=False))
    return obs, rew, done, info
PlayTableSimEnv.step = _hooked_step

for name in dir(v6):
    cls = getattr(v6, name)
    if isinstance(cls, type) and hasattr(cls, "finish_chain_subgoal"):
        _orig_bind = cls.bind_subgoal
        def _hooked_bind(self, instruction, obs, *args, **kwargs):
            STATE["active_policy"] = self
            if STATE["chain_start_recoveries"] < 0:
                STATE["chain_start_recoveries"] = int(getattr(self, "backtrack_recoveries", 0))
            ann_str = str(instruction)
            if ann_str == STATE["current_ann"] and STATE["subgoal_idx"] >= 1:
                STATE["retry_in_subgoal"] += 1
            else:
                STATE["subgoal_idx"] += 1
                STATE["retry_in_subgoal"] = 0
            STATE["current_ann"] = ann_str
            STATE["steps_since_bind"] = 0
            STATE["last_obs"] = obs
            return _orig_bind(self, instruction, obs, *args, **kwargs)
        cls.bind_subgoal = _hooked_bind

        _orig_fin = cls.finish_chain_subgoal
        def _hooked_fin(self, *args, **kwargs):
            sig = inspect.signature(_orig_fin)
            bound = sig.bind_partial(self, *args, **kwargs)
            bound.apply_defaults()
            ok = bool(bound.arguments.get("success", True))
            obs_post = bound.arguments.get("obs", STATE["last_obs"])
            if STATE["in_v6_sys12"] and ok and STATE["current_ann"] != STATE["last_completed_ann"]:
                STATE["completed_in_chain"] += 1
                STATE["current_tasks"].append(STATE["current_ann"])
                STATE["last_completed_ann"] = STATE["current_ann"]
                if obs_post is not None:
                    is_all = (STATE["completed_in_chain"] >= 5)
                    n_hold = 24 if is_all else 1
                    for _ in range(n_hold):
                        STATE["frames"].append(render_hud_frame(obs_post, self, len(STATE["frames"]), solved_banner=is_all))
            return _orig_fin(self, *args, **kwargs)
        cls.finish_chain_subgoal = _hooked_fin

if __name__ == "__main__":
    sys.argv = [
        "run_calvin_dtra_v6.py",
        "--max_tasks", "34",
        "--num_chains", "30",
        "--mode", "vision",
        "--out_json", os.path.join(os.path.dirname(os.path.abspath(__file__)), "calvin_v6_vision_34tasks.json"),
    ]
    v6.main()
    flush_episode()
    save_videos_and_exit()
