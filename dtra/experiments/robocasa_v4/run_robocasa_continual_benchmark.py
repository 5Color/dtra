"""DTRA v4.0 Continual Learning Benchmark on RoboCasa (Pure RGB-D, Unmodified Evaluator).

Compares four arms strictly on Pure RGB-D observations (object-state explicitly stripped),
task-ID-free inference, and unmodified ``env._check_success()`` across 12 RoboCasa pretrain
human kitchen tasks (8 Single-Stage Atomic + 4 Multi-Stage Composite):
  1. Arm1_Flat_MLP           : Shared monolithic B-Spline MLP, sequential AdamW, 0% Replay
  2. Arm2_ER_5pct            : Same B-Spline MLP with 5% episodic Experience Replay
  3. Arm3_DTRA_v4_NoTransport: Task-ID-Free CMC Region->Cell + C^2 B-Spline, 0% Replay, dp=0
  4. Arm4_DTRA_v4_Hypertube  : Full DTRA v4.0 + Pure RGB-D Object-Centric Hypertube Transport
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import robosuite
import robosuite.utils.transform_utils as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from scipy.interpolate import BSpline

import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.scripts.dataset_scripts.playback_utils import resolve_instruction_from_ep_meta
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

# 12 Downloaded Kitchen Manipulation Tasks (8 Single-Stage Atomic + 4 Multi-Stage Composite)
TASK_CURRICULUM: List[Tuple[str, str]] = [
    ("atomic", "OpenCabinet"),
    ("atomic", "CloseCabinet"),
    ("atomic", "OpenDrawer"),
    ("atomic", "CloseDrawer"),
    ("atomic", "OpenMicrowave"),
    ("atomic", "TurnOnSinkFaucet"),
    ("atomic", "TurnOnStove"),
    ("atomic", "PickPlaceCounterToSink"),
    ("composite", "AddSugarCubes"),
    ("composite", "OrganizeVegetables"),
    ("composite", "CupcakeCleanup"),
    ("composite", "DeliverBrewedCoffee"),
]

CAM_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]
STEREO_CAMS = ["robot0_agentview_left", "robot0_agentview_right"]

ALLOWED_RGBD_KEYS = frozenset(
    {
        "robot0_agentview_left_image",
        "robot0_agentview_right_image",
        "robot0_eye_in_hand_image",
        "robot0_agentview_left_depth",
        "robot0_agentview_right_depth",
        "robot0_eye_in_hand_depth",
        "robot0_base_pos",
        "robot0_base_quat",
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
        "robot0_gripper_qpos",
        "robot0_joint_pos",
    }
)

KNOTS_DEFAULT = 32
TEXT_DIM = 128
CAM_H = 84
CAM_W = 84
GRID_H = 11  # ResNet-18 layer2 spatial grid on 84x84 inputs: ceil(84/8)=11


def set_seed(seed: int) -> None:
    """Fix all Python/NumPy/PyTorch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_pure_rgbd_obs(raw_obs: Dict[str, Any]) -> Dict[str, Any]:
    """Strip all ground-truth object-state keys, enforcing strict Pure RGB-D input."""
    clean = {k: raw_obs[k] for k in ALLOWED_RGBD_KEYS if k in raw_obs}
    for k in clean:
        assert "object-state" not in k and "_obj_" not in k and "distr_" not in k
    return clean


def wrap_pi(angle: float) -> float:
    """Wrap scalar angle in radians to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def quat_to_yaw(q_xyzw: np.ndarray) -> float:
    """Extract yaw around +Z from robosuite (x, y, z, w) quaternion."""
    mat = T.quat2mat(q_xyzw)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def rotvec_diff(q_des_xyzw: np.ndarray, q_cur_xyzw: np.ndarray) -> np.ndarray:
    """Compute shortest 3D axis-angle error from current to desired orientation."""
    r_des = T.quat2mat(q_des_xyzw)
    r_cur = T.quat2mat(q_cur_xyzw)
    return T.quat2axisangle(T.mat2quat(r_des @ r_cur.T))


_BSPLINE_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def bspline_basis(t_len: int, n_knots: int = KNOTS_DEFAULT, degree: int = 3) -> np.ndarray:
    """Evaluate cached clamped C^2 cubic B-spline basis matrix of shape [t_len, n_knots]."""
    key = (t_len, n_knots)
    if key in _BSPLINE_CACHE:
        return _BSPLINE_CACHE[key]
    n_int = n_knots - degree - 1
    interior = np.linspace(0.0, 1.0, max(n_int + 2, 2))[1:-1] if n_int > 0 else np.array([])
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    ts = np.linspace(0.0, 1.0, t_len)
    basis = np.zeros((t_len, n_knots), dtype=np.float64)
    for j in range(n_knots):
        coeff = np.zeros(n_knots, dtype=np.float64)
        coeff[j] = 1.0
        basis[:, j] = BSpline(knots, coeff, degree)(ts)
    _BSPLINE_CACHE[key] = basis
    return basis


def fit_bspline(traj: np.ndarray, n_knots: int = KNOTS_DEFAULT) -> np.ndarray:
    """Project trajectory [T, D] onto clamped C^2 cubic B-spline control knots [n_knots, D]."""
    basis = bspline_basis(len(traj), n_knots, 3)
    knots, *_ = np.linalg.lstsq(basis, traj, rcond=None)
    return knots.astype(np.float32)


def eval_bspline(knots: np.ndarray, t_len: int) -> np.ndarray:
    """Evaluate C^2 cubic B-spline trajectory at t_len normalized timesteps."""
    return bspline_basis(t_len, len(knots), 3) @ knots


STOPWORDS = frozenset({"the", "a", "an", "on", "to", "from", "in", "into", "of", "and", "with", "for"})
VIS_DIM = 2304


def hash_instruction(text: str, dim: int = TEXT_DIM) -> np.ndarray:
    """Deterministic stopword-filtered action/noun/bigram hash embedding of natural language."""
    vec = np.zeros(dim, dtype=np.float32)
    raw = text.lower().replace(".", "").replace(",", "").split()
    tokens = [w for w in raw if w not in STOPWORDS] or raw
    for i, w in enumerate(tokens):
        wt = 1.8 if i == 0 else 1.0  # emphasize leading action verb (open vs close, turn vs pick)
        h1 = sum(ord(ch) * (31 ** (idx + 1)) for idx, ch in enumerate(w)) % dim
        vec[h1] += wt
        if i + 1 < len(tokens):
            bg = f"{w}_{tokens[i + 1]}"
            h2 = sum(ord(ch) * (131 ** (idx + 1)) for idx, ch in enumerate(bg)) % dim
            vec[h2] += 0.9
    norm = float(np.linalg.norm(vec))
    return vec / (norm + 1e-8) if norm > 1e-8 else vec


class FrozenPerceptionEncoder(nn.Module):
    """Shared frozen ResNet-18 spatial-pyramid + 21x21 fine dense + text encoder across all arms."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        weights = tv_models.ResNet18_Weights.DEFAULT
        res = tv_models.resnet18(weights=weights).to(device).eval()
        for p in res.parameters():
            p.requires_grad = False
        self.res = res
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        rng = np.random.RandomState(42)
        proj = rng.randn(256 * 6, 256).astype(np.float32)
        q, _ = np.linalg.qr(proj)
        self.register_buffer("spatial_proj", torch.from_numpy(q[:, :256].copy()).to(device))

    @torch.no_grad()
    def encode_views(self, imgs_hwc: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Encode 3 views into spatial-aware [3, 768] and fine multi-scale [3, 192, 21, 21] maps."""
        resized = np.stack([cv2.resize(im, (CAM_W, CAM_H)) for im in imgs_hwc], axis=0)
        x = torch.from_numpy(resized).to(self.device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        x = (x - self.mean) / self.std
        x = self.res.relu(self.res.bn1(self.res.conv1(x)))
        x = self.res.maxpool(x)
        f1 = self.res.layer1(x)
        f2 = self.res.layer2(f1)
        f3 = self.res.layer3(f2)
        f4 = self.res.layer4(f3)

        g4 = F.normalize(self.res.avgpool(f4).flatten(1), dim=1)
        sp_2x3 = F.adaptive_avg_pool2d(f3, (2, 3)).flatten(1)
        sp_proj = F.normalize(sp_2x3 @ self.spatial_proj, dim=1)
        g_comb = F.normalize(torch.cat([g4 * 0.50, sp_proj * 0.85], dim=1), dim=1)  # [3, 768]

        f2_up = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f_dense = torch.cat([F.normalize(f1, dim=1) * 0.55, F.normalize(f2_up, dim=1) * 0.85], dim=1)
        f_dense = F.normalize(f_dense, dim=1)  # [3, 192, 21, 21]
        return g_comb.cpu().numpy().astype(np.float32), f_dense.cpu().numpy().astype(np.float32)

    def make_key(self, g_views: np.ndarray, inst: str, proprio0: np.ndarray) -> np.ndarray:
        """Assemble normalized multimodal routing/conditioning key [2441]."""
        z_vis = g_views.flatten().astype(np.float32)
        z_vis = z_vis / (np.linalg.norm(z_vis) + 1e-8)
        z_txt = hash_instruction(inst, TEXT_DIM)
        prop_norm = np.tanh(proprio0 / 2.0).astype(np.float32)
        key = np.concatenate([z_vis * 0.60, z_txt * 1.15, prop_norm * 0.12])
        return (key / (np.linalg.norm(key) + 1e-8)).astype(np.float32)


# PandaOmron invariant onboard base->camera extrinsics (populated lazily from sim robot model)
_BASE_CAM_EXTRINSICS_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def get_base_cam_kinematics(sim: Any, obs_clean: Dict[str, Any], cam: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return invariant base-to-camera extrinsic E_bc [4x4] and intrinsic K [3x3] from robot kinematics."""
    t_wb = T.make_pose(obs_clean["robot0_base_pos"], T.quat2mat(obs_clean["robot0_base_quat"]))
    e_wc = get_camera_extrinsic_matrix(sim, cam)
    e_bc = np.linalg.inv(t_wb) @ e_wc
    k_mat = get_camera_intrinsic_matrix(sim, cam, CAM_H, CAM_W)
    _BASE_CAM_EXTRINSICS_CACHE[cam] = (e_bc.copy(), k_mat.copy())
    return e_bc, k_mat


def extract_contact_anchor_and_spline(st_arr: np.ndarray, acts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute proprioception-only contact anchor p_contact and 17-channel C^2 B-spline knot matrix [K, 17]."""
    t_len = len(acts)
    pos_b = st_arr[:, 7:10].astype(np.float64)
    quat_b = st_arr[:, 10:14].copy().astype(np.float64)
    for t in range(1, t_len):
        if np.dot(quat_b[t], quat_b[t - 1]) < 0.0:
            quat_b[t] = -quat_b[t]

    # Unwrap mobile base pose trajectory relative to episode start base frame
    p_b0 = st_arr[0, 0:3].astype(np.float64)
    r_b0 = T.quat2mat(st_arr[0, 3:7].astype(np.float64))
    yaw0 = quat_to_yaw(st_arr[0, 3:7].astype(np.float64))
    base_xyz = np.array([r_b0.T @ (st_arr[t, 0:3].astype(np.float64) - p_b0) for t in range(t_len)])
    base_yaw = np.unwrap([wrap_pi(quat_to_yaw(st_arr[t, 3:7].astype(np.float64)) - yaw0) for t in range(t_len)])[:, None]

    # Locate proprioceptive contact anchor (first gripper close onset, or lowest/farthest counter reach)
    grip = acts[:, 6:7].astype(np.float64)
    flips = np.where((grip[1:, 0] > 0.2) & (grip[:-1, 0] <= 0.2))[0]
    if len(flips) > 0:
        t_c = int(flips[0])
    else:
        reach = np.linalg.norm(pos_b - pos_b[0], axis=1) - 0.5 * pos_b[:, 2]
        t_c = int(np.argmax(reach))
    r_rel_c = r_b0.T @ T.quat2mat(st_arr[t_c, 3:7].astype(np.float64))
    p_contact = (r_rel_c @ pos_b[t_c] + base_xyz[t_c]).astype(np.float32)

    traj17 = np.concatenate(
        [
            pos_b,                       # 0:3 EEF pos in current base frame
            quat_b,                      # 3:7 EEF quat (sign-continuous)
            base_xyz,                    # 7:10 base rel pos in init base frame
            base_yaw,                    # 10:11 base rel yaw
            grip,                        # 11:12 gripper close
            acts[:, 7:11],               # 12:16 base vel
            acts[:, 11:12],              # 16:17 control_mode
        ],
        axis=1,
    )
    knots = fit_bspline(traj17, KNOTS_DEFAULT)  # [32, 17]
    return knots, p_contact, t_len


def load_or_build_task_cache(
    dataset_root: Path,
    encoder: FrozenPerceptionEncoder,
    cache_path: Path,
    max_train: int = 25,
    eval_eps: Sequence[int] = (80, 81, 82),
) -> Dict[str, Any]:
    """Load parquet+mp4 demos across all 12 tasks once and cache frozen multimodal features."""
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"Loaded cached frozen demo features from {cache_path}")
        return data

    # Bootstrap PandaOmron onboard camera extrinsics once from first task env
    ds0 = sorted((dataset_root / TASK_CURRICULUM[0][0] / TASK_CURRICULUM[0][1]).glob("*/lerobot"))[0]
    em0 = LU.get_env_metadata(ds0)
    ek0 = dict(em0["env_kwargs"])
    ek0.update(
        {
            "env_name": em0["env_name"],
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_object_obs": False,
            "use_camera_obs": True,
            "camera_names": CAM_NAMES,
            "camera_depths": False,
            "camera_widths": CAM_W,
            "camera_heights": CAM_H,
        }
    )
    env_boot = robosuite.make(**ek0)
    st0 = LU.get_episode_states(ds0, 0)
    reset_to(env_boot, {"states": st0[0], "model": LU.get_episode_model_xml(ds0, 0), "ep_meta": json.dumps(LU.get_episode_meta(ds0, 0))})
    clean0 = sanitize_pure_rgbd_obs(env_boot._get_observations(force_update=True))
    cam_kin = {cam: get_base_cam_kinematics(env_boot.sim, clean0, cam) for cam in STEREO_CAMS}
    env_boot.close()

    tasks_data: Dict[str, Dict[str, Any]] = {}
    t_start = time.time()
    for stage, tname in TASK_CURRICULUM:
        ds = sorted((dataset_root / stage / tname).glob("*/lerobot"))[0]
        eps = LU.get_episodes(ds)
        train_demos = []
        for ep in range(min(max_train, len(eps))):
            df = pd.read_parquet(ds / f"data/chunk-000/episode_{ep:06d}.parquet")
            st_arr = np.stack(df["observation.state"].to_list())
            acts = LU.reorder_lerobot_action(np.stack(df["action"].to_list()), ds)
            inst = resolve_instruction_from_ep_meta(LU.get_episode_meta(ds, ep))
            frames0 = []
            for cam in CAM_NAMES:
                cap = cv2.VideoCapture(str(ds / f"videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4"))
                ok, bgr = cap.read()
                cap.release()
                if not ok:
                    raise RuntimeError(f"Failed reading {tname} ep{ep} {cam}")
                frames0.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            g_v, f2_v = encoder.encode_views(frames0)
            proprio0 = np.concatenate([st_arr[0, 7:16]]).astype(np.float32)  # 9-D robot-relative proprio
            key = encoder.make_key(g_v, inst, proprio0)
            knots, p_contact, t_len = extract_contact_anchor_and_spline(st_arr, acts)

            # Extract stereo visual descriptors at projected contact pixel (base frame)
            descs: Dict[str, Optional[Tuple[int, int, float, np.ndarray]]] = {}
            for c_idx, cam in enumerate(STEREO_CAMS):
                e_bc, k_mat = cam_kin[cam]
                p_cam = (np.linalg.inv(e_bc) @ np.append(p_contact, 1.0))[:3]
                if p_cam[2] > 0.25:
                    uv = k_mat @ p_cam
                    u, v = float(uv[0] / uv[2]), float(uv[1] / uv[2])
                    if 6.0 <= u < CAM_W - 6.0 and 6.0 <= v < CAM_H - 6.0:
                        gh, gw = f2_v.shape[2], f2_v.shape[3]
                        gy = int(np.clip(round(v / CAM_H * gh), 0, gh - 1))
                        gx = int(np.clip(round(u / CAM_W * gw), 0, gw - 1))
                        descs[cam] = (gx, gy, float(p_cam[2]), f2_v[c_idx, :, gy, gx].copy())
                    else:
                        descs[cam] = None
                else:
                    descs[cam] = None

            train_demos.append(
                {
                    "ep": ep,
                    "task": tname,
                    "stage": stage,
                    "inst": inst,
                    "key": key,
                    "knots": knots,
                    "t_len": t_len,
                    "p_contact": p_contact,
                    "descs": descs,
                }
            )
        tasks_data[tname] = {
            "stage": stage,
            "ds_path": str(ds),
            "train": train_demos,
            "eval_eps": [e for e in eval_eps if e < len(eps)],
        }
        print(f"  Cached {tname:24s} ({stage}): {len(train_demos)} train demos")

    bundle = {"tasks": tasks_data, "cam_kin": cam_kin}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, cache_path)
    print(f"Saved demo cache ({time.time() - t_start:.1f}s) -> {cache_path}")
    return bundle


def compute_pure_rgbd_transport(
    env: Any,
    clean_obs: Dict[str, Any],
    f2_live: np.ndarray,
    cell: Dict[str, Any],
    max_shift: float = 0.14,
) -> Tuple[np.ndarray, float]:
    """Estimate bounded 3D Hypertube Transport dp_base strictly from Pure RGB-D fine sub-pixel match + depth."""
    p_contact = cell["p_contact"].astype(np.float64)
    descs = cell.get("descs", {})
    dp_candidates: List[np.ndarray] = []
    weights: List[float] = []

    for c_idx, cam in enumerate(STEREO_CAMS):
        info = descs.get(cam)
        if info is None:
            continue
        gx_d, gy_d, z_demo, f_tgt = info
        e_bc, k_mat = get_base_cam_kinematics(env.sim, clean_obs, cam)
        f_live = f2_live[c_idx]
        gh, gw = f_live.shape[1], f_live.shape[2]
        sim_map = np.tensordot(f_tgt, f_live, axes=([0], [0]))  # [21, 21]
        y0, y1 = max(0, gy_d - 5), min(gh, gy_d + 6)
        x0, x1 = max(0, gx_d - 5), min(gw, gx_d + 6)
        win = sim_map[y0:y1, x0:x1]
        ry, rx = np.unravel_index(np.argmax(win), win.shape)
        my, mx = y0 + int(ry), x0 + int(rx)
        best_sim = float(sim_map[my, mx])
        margin = best_sim - float(np.mean(win))
        if best_sim < 0.80 or margin < 0.30:
            continue
        conf = float(np.clip((best_sim - 0.45) / 0.40, 0.0, 1.0))

        # 3x3 sub-pixel soft-argmax refinement for continuous localization accuracy
        sy0, sy1 = max(0, my - 1), min(gh, my + 2)
        sx0, sx1 = max(0, mx - 1), min(gw, mx + 2)
        sub = sim_map[sy0:sy1, sx0:sx1]
        w_sub = np.exp((sub - best_sim) / 0.05)
        w_sub /= np.sum(w_sub)
        ys_grid, xs_grid = np.mgrid[sy0:sy1, sx0:sx1]
        my_sub = float(np.sum(w_sub * ys_grid))
        mx_sub = float(np.sum(w_sub * xs_grid))

        u_l = float((mx_sub + 0.5) / gw * CAM_W)
        v_l = float((my_sub + 0.5) / gh * CAM_H)
        ui, vi = int(np.clip(round(u_l), 1, CAM_W - 2)), int(np.clip(round(v_l), 1, CAM_H - 2))
        rd = get_real_depth_map(env.sim, clean_obs[f"{cam}_depth"])[::-1, :, 0]
        z_l = float(np.median(rd[vi - 1 : vi + 2, ui - 1 : ui + 2]))
        if not (0.25 < z_l < 2.2) or abs(z_l - z_demo) > 0.22:
            continue
        p_cam_l = np.array([(u_l - k_mat[0, 2]) * z_l / k_mat[0, 0], (v_l - k_mat[1, 2]) * z_l / k_mat[1, 1], z_l, 1.0])
        p_base_l = (e_bc @ p_cam_l)[:3]
        dp_candidates.append(p_base_l - p_contact)
        weights.append(conf)

    if len(dp_candidates) < 2:
        return np.zeros(3, dtype=np.float64), 0.0
    if float(np.linalg.norm(dp_candidates[0] - dp_candidates[1])) > 0.10:
        return np.zeros(3, dtype=np.float64), 0.0
    w_arr = np.array(weights, dtype=np.float64)
    w_arr /= np.sum(w_arr)
    dp_raw = np.sum([w * d for w, d in zip(w_arr, dp_candidates)], axis=0)
    mean_conf = float(np.mean(weights))
    dp_bounded = np.clip(mean_conf * dp_raw, -max_shift, max_shift)
    dp_bounded[2] = float(np.clip(dp_bounded[2], -0.025, 0.025))
    return dp_bounded, mean_conf


def rollout_spline_episode(
    env: Any,
    init_state: Dict[str, Any],
    policy: Any,
    encoder: FrozenPerceptionEncoder,
    true_task: str,
    max_horizon_cap: int = 550,
    record_hud_frames: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run one closed-loop episode under Strict Pure RGB-D observations and unmodified env._check_success."""
    reset_to(env, init_state)
    clean_obs = sanitize_pure_rgbd_obs(env._get_observations(force_update=True))
    ep_meta = json.loads(init_state["ep_meta"])
    inst = resolve_instruction_from_ep_meta(ep_meta)

    imgs0 = [clean_obs[f"{cam}_image"][::-1].copy() for cam in CAM_NAMES]
    g_v, f2_v = encoder.encode_views(imgs0)
    proprio0 = np.concatenate([clean_obs["robot0_base_to_eef_pos"], clean_obs["robot0_base_to_eef_quat"], clean_obs["robot0_gripper_qpos"]]).astype(np.float32)
    q_key = encoder.make_key(g_v, inst, proprio0)

    t0 = time.perf_counter()
    pred = policy.predict_episode(q_key, env, clean_obs, f2_v)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    knots = pred["knots"]  # [K, 17]
    t_len = int(np.clip(pred["t_len"], 60, max_horizon_cap))
    dp_tube = pred["dp_tube"]
    routed_task = pred["routed_task"]

    traj = eval_bspline(knots, t_len)
    taus = np.linspace(0.0, 1.0, t_len)
    bell = np.sin(np.pi * taus)
    pos_des_seq = traj[:, 0:3]
    quat_des_seq = traj[:, 3:7].copy()
    quat_des_seq /= np.linalg.norm(quat_des_seq, axis=1, keepdims=True) + 1e-8
    bxyz_des_seq = traj[:, 7:10]
    byaw_des_seq = traj[:, 10:11]
    grip_seq = traj[:, 11:12]
    bvel_seq = traj[:, 12:16]
    ctrl_seq = traj[:, 16:17]

    p_init = clean_obs["robot0_base_pos"].copy()
    r_init = T.quat2mat(clean_obs["robot0_base_quat"])
    yaw_init = quat_to_yaw(clean_obs["robot0_base_quat"])

    solved = False
    n_steps = min(t_len + 30, max_horizon_cap)
    for step_i in range(n_steps):
        idx = min(step_i, t_len - 1)
        p_base_rel = r_init.T @ (clean_obs["robot0_base_pos"] - p_init)
        yaw_rel = wrap_pi(quat_to_yaw(clean_obs["robot0_base_quat"]) - yaw_init)
        c_y, s_y = np.cos(-yaw_rel), np.sin(-yaw_rel)
        # Rotate t=0 base-frame Hypertube Transport dp_tube into current base frame
        dp_loc_x = c_y * dp_tube[0] - s_y * dp_tube[1]
        dp_loc_y = s_y * dp_tube[0] + c_y * dp_tube[1]
        dp_loc = np.array([dp_loc_x, dp_loc_y, dp_tube[2]], dtype=np.float64) * bell[idx]

        p_err = (pos_des_seq[idx] + dp_loc) - clean_obs["robot0_base_to_eef_pos"]
        r_err = rotvec_diff(quat_des_seq[idx], clean_obs["robot0_base_to_eef_quat"])

        dp_base = bxyz_des_seq[idx] - p_base_rel
        dx_loc = c_y * dp_base[0] - s_y * dp_base[1]
        dy_loc = s_y * dp_base[0] + c_y * dp_base[1]
        eyaw = wrap_pi(byaw_des_seq[idx, 0] - yaw_rel)

        action = np.zeros(12, dtype=np.float32)
        action[0:3] = np.clip(p_err * 22.0, -1.0, 1.0)
        action[3:6] = np.clip(r_err * 8.0, -1.0, 1.0)
        action[6] = 1.0 if grip_seq[idx, 0] > 0.0 else -1.0
        action[7] = np.clip(bvel_seq[idx, 0] + dx_loc * 6.0, -1.0, 1.0)
        action[8] = np.clip(bvel_seq[idx, 1] + dy_loc * 6.0, -1.0, 1.0)
        action[9] = np.clip(bvel_seq[idx, 2] + eyaw * 4.0, -1.0, 1.0)
        action[10] = np.clip(bvel_seq[idx, 3], -1.0, 1.0)
        action[11] = 1.0 if ctrl_seq[idx, 0] > 0.0 else -1.0

        raw_obs, _, _, _ = env.step(action)
        clean_obs = sanitize_pure_rgbd_obs(raw_obs)
        if env._check_success():
            solved = True
        if record_hud_frames is not None and (step_i % 3 == 0 or solved or step_i == n_steps - 1):
            record_hud_frames.append(
                {
                    "left_rgb": clean_obs["robot0_agentview_left_image"][::-1].copy(),
                    "hand_rgb": clean_obs["robot0_eye_in_hand_image"][::-1].copy(),
                    "task": true_task,
                    "routed_task": routed_task or "-",
                    "inst": inst,
                    "step": step_i + 1,
                    "max_steps": n_steps,
                    "dp_tube": dp_tube.copy(),
                    "conf": float(pred.get("conf", 0.0)),
                    "drift": float(policy.parameter_drift()),
                    "lat_ms": infer_ms,
                    "solved": solved,
                }
            )
        if solved:
            break

    return {
        "solved": bool(solved),
        "route_match": (routed_task == true_task) if routed_task is not None else None,
        "infer_ms": float(infer_ms),
        "dp_norm": float(np.linalg.norm(dp_tube)),
    }


class ContinualSplineMLP(nn.Module):
    """Shared-weight B-spline MLP baseline used by Arm1 (Flat 0%) and Arm2 (ER 5%)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPBaselinePolicy:
    """Continual MLP controller supporting 0% replay (Flat MLP) or 5% Experience Replay (ER 5%)."""

    def __init__(self, key_dim: int, device: torch.device, replay_ratio: float = 0.0, epochs: int = 80):
        self.device = device
        self.replay_ratio = replay_ratio
        self.epochs = epochs
        out_dim = KNOTS_DEFAULT * 17 + 1  # spline knots + normalized duration
        self.model = ContinualSplineMLP(key_dim, out_dim).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=8e-4, weight_decay=1e-4)
        self.w_init: Optional[torch.Tensor] = None
        self.replay_x: List[np.ndarray] = []
        self.replay_y: List[np.ndarray] = []

    def train_task(self, demos: List[Dict[str, Any]]) -> None:
        xs = np.stack([d["key"] for d in demos]).astype(np.float32)
        ys = np.stack(
            [np.concatenate([d["knots"].reshape(-1), [d["t_len"] / 500.0]]).astype(np.float32) for d in demos]
        )
        if self.replay_ratio > 0.0 and self.replay_x:
            rx = np.stack(self.replay_x).astype(np.float32)
            ry = np.stack(self.replay_y).astype(np.float32)
            all_x = np.concatenate([xs, rx], axis=0)
            all_y = np.concatenate([ys, ry], axis=0)
        else:
            all_x, all_y = xs, ys

        x_t = torch.from_numpy(all_x).to(self.device)
        y_t = torch.from_numpy(all_y).to(self.device)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(x_t), device=self.device)
            pred = self.model(x_t[perm])
            loss = F.mse_loss(pred, y_t[perm])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        if self.w_init is None:
            with torch.no_grad():
                self.w_init = torch.cat([p.detach().flatten() for p in self.model.parameters()]).clone()

        if self.replay_ratio > 0.0:
            n_keep = max(1, int(round(len(demos) * max(self.replay_ratio, 0.05))))
            for idx in range(n_keep):
                self.replay_x.append(xs[idx])
                self.replay_y.append(ys[idx])

    @torch.no_grad()
    def predict_episode(self, q_key: np.ndarray, env: Any, clean_obs: Dict[str, Any], f2_live: np.ndarray) -> Dict[str, Any]:
        self.model.eval()
        out = self.model(torch.from_numpy(q_key).to(self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
        knots = out[:-1].reshape(KNOTS_DEFAULT, 17).astype(np.float64)
        t_len = int(round(float(out[-1]) * 500.0))
        return {"knots": knots, "t_len": t_len, "dp_tube": np.zeros(3), "routed_task": None, "conf": 0.0}

    def parameter_drift(self) -> float:
        if self.w_init is None:
            return 0.0
        with torch.no_grad():
            cur = torch.cat([p.detach().flatten() for p in self.model.parameters()])
            return float(torch.norm(cur - self.w_init).item())


class DTRARoboCasaPolicy:
    """DTRA v4.0 Region-Cell CMC Routing + Frozen C^2 B-Splines + Pure RGB-D Hypertube Transport."""

    def __init__(self, use_transport: bool = True, max_shift: float = 0.14):
        self.use_transport = use_transport
        self.max_shift = max_shift
        self.regions: List[Dict[str, Any]] = []

    def train_task(self, demos: List[Dict[str, Any]]) -> None:
        task_name = demos[0]["task"]
        raw_mean = np.mean(np.stack([d["key"] for d in demos]), axis=0).astype(np.float64)
        raw_mean /= np.linalg.norm(raw_mean) + 1e-8
        # Store raw frozen region centroid & immutable cell trajectories (Zero Drift)
        self.regions.append({"task": task_name, "raw_proto": raw_mean, "cells": copy.deepcopy(demos)})

    def _cmc_norm(self, v: np.ndarray, mu: np.ndarray, alpha: float = 0.90) -> np.ndarray:
        c = v - alpha * mu
        return c / (np.linalg.norm(c) + 1e-8)

    def predict_episode(self, q_key: np.ndarray, env: Any, clean_obs: Dict[str, Any], f2_live: np.ndarray) -> Dict[str, Any]:
        # Task-ID-free Stage-2 CMC Mean-Centered Region Routing (cancels shared kitchen/stopword background)
        if len(self.regions) >= 2:
            mu_r = np.mean(np.stack([r["raw_proto"] for r in self.regions]), axis=0)
            q_r = self._cmc_norm(q_key.astype(np.float64), mu_r, 0.92)
            r_scores = [float(np.dot(q_r, self._cmc_norm(r["raw_proto"], mu_r, 0.92))) for r in self.regions]
        else:
            r_scores = [float(np.dot(q_key, r["raw_proto"])) for r in self.regions]
        reg = self.regions[int(np.argmax(r_scores))]

        # Stage-2 Within-Region Spatial-Visual + Language Cell Selection
        cells = reg["cells"]
        q_vis = q_key[:VIS_DIM] / (np.linalg.norm(q_key[:VIS_DIM]) + 1e-8)
        q_txt = q_key[VIS_DIM : VIS_DIM + TEXT_DIM] / (np.linalg.norm(q_key[VIS_DIM : VIS_DIM + TEXT_DIM]) + 1e-8)
        best_idx = 0
        best_sc = -1e9
        for i, c in enumerate(cells):
            ck = c["key"]
            cv = ck[:VIS_DIM] / (np.linalg.norm(ck[:VIS_DIM]) + 1e-8)
            ct = ck[VIS_DIM : VIS_DIM + TEXT_DIM] / (np.linalg.norm(ck[VIS_DIM : VIS_DIM + TEXT_DIM]) + 1e-8)
            sc = 0.80 * float(np.dot(q_vis, cv)) + 0.20 * float(np.dot(q_txt, ct))
            if sc > best_sc:
                best_sc = sc
                best_idx = i
        cell = cells[best_idx]

        dp_tube, conf = (
            compute_pure_rgbd_transport(env, clean_obs, f2_live, cell, max_shift=self.max_shift)
            if self.use_transport
            else (np.zeros(3, dtype=np.float64), 0.0)
        )
        return {
            "knots": cell["knots"].astype(np.float64),
            "t_len": int(cell["t_len"]),
            "dp_tube": dp_tube,
            "routed_task": reg["task"],
            "conf": conf,
        }

    def parameter_drift(self) -> float:
        return 0.0  # Structural zero drift across prior frozen Regions


def create_robocasa_eval_env(ds_path: str) -> Any:
    """Instantiate unmodified RoboCasa environment strictly in Pure RGB-D configuration."""
    ds = Path(ds_path)
    em = LU.get_env_metadata(ds)
    ek = dict(em["env_kwargs"])
    ek.update(
        {
            "env_name": em["env_name"],
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_object_obs": False,
            "use_camera_obs": True,
            "camera_names": CAM_NAMES,
            "camera_depths": True,
            "camera_widths": CAM_W,
            "camera_heights": CAM_H,
        }
    )
    return robosuite.make(**ek)


def evaluate_task(
    tname: str,
    task_info: Dict[str, Any],
    policy: Any,
    encoder: FrozenPerceptionEncoder,
    env: Any,
) -> Dict[str, Any]:
    """Evaluate all held-out unseen episodes for one task cleanly."""
    ds = Path(task_info["ds_path"])
    eps = task_info["eval_eps"]
    solved_list, route_list, lat_list, dp_list = [], [], [], []
    for ep_id in eps:
        st = LU.get_episode_states(ds, ep_id)
        init_state = {
            "states": st[0],
            "model": LU.get_episode_model_xml(ds, ep_id),
            "ep_meta": json.dumps(LU.get_episode_meta(ds, ep_id)),
        }
        res = rollout_spline_episode(env, init_state, policy, encoder, tname)
        solved_list.append(float(res["solved"]))
        if res["route_match"] is not None:
            route_list.append(float(res["route_match"]))
        lat_list.append(res["infer_ms"])
        dp_list.append(res["dp_norm"])
    return {
        "sr": float(np.mean(solved_list)) * 100.0,
        "route_acc": float(np.mean(route_list)) * 100.0 if route_list else None,
        "lat_ms": float(np.mean(lat_list)),
        "mean_dp": float(np.mean(dp_list)),
    }


def run_arm(arm_name: str, bundle: Dict[str, Any], encoder: FrozenPerceptionEncoder, device: torch.device) -> Dict[str, Any]:
    """Execute sequential 12-task continual learning + init & final unseen MuJoCo evaluations."""
    key_dim = len(bundle["tasks"][TASK_CURRICULUM[0][1]]["train"][0]["key"])
    if arm_name == "Arm1_Flat_MLP":
        policy: Any = MLPBaselinePolicy(key_dim, device, replay_ratio=0.0, epochs=80)
        replay_str = "0%"
    elif arm_name == "Arm2_ER_5pct":
        policy = MLPBaselinePolicy(key_dim, device, replay_ratio=0.05, epochs=80)
        replay_str = "5%"
    elif arm_name == "Arm3_DTRA_v4_NoTransport":
        policy = DTRARoboCasaPolicy(use_transport=False)
        replay_str = "0%"
    elif arm_name == "Arm4_DTRA_v4_Hypertube":
        policy = DTRARoboCasaPolicy(use_transport=True, max_shift=0.14)
        replay_str = "0%"
    else:
        raise ValueError(f"Unknown arm: {arm_name}")

    print(f"\n=== Running {arm_name} (12 Tasks, Pure RGB-D, Replay={replay_str}) ===")
    t0 = time.time()
    init_per_task: Dict[str, float] = {}
    final_per_task: Dict[str, float] = {}
    route_accs: List[float] = []
    lats: List[float] = []

    # Sequential continual training + immediate Init SR evaluation
    for step_idx, (stage, tname) in enumerate(TASK_CURRICULUM):
        tinfo = bundle["tasks"][tname]
        policy.train_task(tinfo["train"])
        env = create_robocasa_eval_env(tinfo["ds_path"])
        res = evaluate_task(tname, tinfo, policy, encoder, env)
        env.close()
        init_per_task[tname] = res["sr"]
        lats.append(res["lat_ms"])
        print(
            f"  [{arm_name}] Task {step_idx + 1:2d}/12 ({tname:22s}, {stage:9s}): "
            f"init_sr={res['sr']:5.1f}% | route={res['route_acc'] if res['route_acc'] is not None else '-'} | "
            f"|dp|={res['mean_dp']:.3f}m | elapsed={time.time() - t0:.1f}s"
        )

    # Full 12-Task Task-ID-Free Final Continual Evaluation Sweep
    for stage, tname in TASK_CURRICULUM:
        tinfo = bundle["tasks"][tname]
        env = create_robocasa_eval_env(tinfo["ds_path"])
        res = evaluate_task(tname, tinfo, policy, encoder, env)
        env.close()
        final_per_task[tname] = res["sr"]
        if res["route_acc"] is not None:
            route_accs.append(res["route_acc"])
        lats.append(res["lat_ms"])

    atomic_names = [t for s, t in TASK_CURRICULUM if s == "atomic"]
    comp_names = [t for s, t in TASK_CURRICULUM if s == "composite"]
    init_mean = float(np.mean(list(init_per_task.values())))
    final_mean = float(np.mean(list(final_per_task.values())))
    bwt = float(np.mean([final_per_task[t] - init_per_task[t] for _, t in TASK_CURRICULUM]))
    forgetting = float(np.mean([max(0.0, init_per_task[t] - final_per_task[t]) for _, t in TASK_CURRICULUM]))

    summary = {
        "arm": arm_name,
        "replay": replay_str,
        "num_tasks": len(TASK_CURRICULUM),
        "initial_mean_sr": round(init_mean, 2),
        "final_mean_sr": round(final_mean, 2),
        "atomic_final_sr": round(float(np.mean([final_per_task[t] for t in atomic_names])), 2),
        "composite_final_sr": round(float(np.mean([final_per_task[t] for t in comp_names])), 2),
        "forgetting_rate": round(forgetting, 2),
        "bwt": round(bwt, 2),
        "route_acc": round(float(np.mean(route_accs)), 2) if route_accs else None,
        "parameter_drift": round(policy.parameter_drift(), 6),
        "infer_latency_ms": round(float(np.mean(lats)), 3),
        "total_sec": round(time.time() - t0, 2),
        "initial_per_task_sr": init_per_task,
        "final_per_task_sr": final_per_task,
    }
    print(
        f"==> [{arm_name}] Init={init_mean:.2f}% -> Final={final_mean:.2f}% "
        f"(Atomic={summary['atomic_final_sr']:.2f}%, Composite={summary['composite_final_sr']:.2f}%) | "
        f"BWT={bwt:+.2f}% | Drift={summary['parameter_drift']:.6f} | Route={summary['route_acc']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DTRA v4.0 RoboCasa Pure RGB-D Continual Learning Benchmark")
    parser.add_argument(
        "--dataset_root",
        default="/home/team03/dtra_a6000_lab/robocasa_lab/robocasa/datasets/v1.0/pretrain",
    )
    parser.add_argument("--cache_path", default="/home/team03/dtra_a6000_lab/robocasa_lab/demo_cache_12tasks_65v2.pt")
    parser.add_argument("--out_json", default="/home/team03/dtra_a6000_lab/robocasa_lab/dtra_v4/robocasa_continual_results.json")
    parser.add_argument("--max_train", type=int, default=65)
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["Arm1_Flat_MLP", "Arm2_ER_5pct", "Arm3_DTRA_v4_NoTransport", "Arm4_DTRA_v4_Hypertube"],
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DTRA v4.0 RoboCasa Benchmark | Device: {device} | Pure RGB-D | Unmodified Evaluator ===")

    encoder = FrozenPerceptionEncoder(device)
    bundle = load_or_build_task_cache(
        Path(args.dataset_root),
        encoder,
        Path(args.cache_path),
        max_train=args.max_train,
        eval_eps=(80, 81, 82),
    )
    if args.cache_only:
        print("Cache build complete (--cache_only). Exiting cleanly.")
        return

    results: Dict[str, Any] = {}
    out_p = Path(args.out_json)
    if out_p.exists():
        try:
            results = json.loads(out_p.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    for arm in args.arms:
        results[arm] = run_arm(arm, bundle, encoder, device)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved checkpoint JSON -> {out_p}")


if __name__ == "__main__":
    main()
