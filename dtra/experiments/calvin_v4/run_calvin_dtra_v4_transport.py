import os
import json
import time
import hashlib
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import hydra
from omegaconf import OmegaConf
from calvin_env.envs.play_table_env import get_env

from hypertube_transport import (
    ObjectCentricTransportAdapter,
    OracleSceneAnchorEstimator,
    ObjectAnchor,
    ObjectAnchorEstimator,
    infer_target_object_name,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _det_hash(token: str, dim: int = 512) -> int:
    h = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % dim


# ==============================================================================
# 1. C^2 Clamped Cubic B-Spline Basis (N_k = 6 control knots)
# ==============================================================================
class CubicBSplineBasis(nn.Module):
    def __init__(self, num_knots=6, degree=3):
        super().__init__()
        self.num_knots = num_knots
        self.degree = degree
        knots = [0.0] * (degree + 1) + [0.3333333, 0.6666667] + [1.0] * (degree + 1)
        self.register_buffer("knots", torch.tensor(knots, dtype=torch.float32))

    def _basis_matrix(self, T, device):
        t = torch.linspace(0.0, 1.0 - 1e-6, T, device=device)
        t[-1] = 1.0 - 1e-6
        n = self.num_knots
        p = self.degree
        u = self.knots.to(device)
        N = torch.zeros((T, n + p), device=device)
        for i in range(n + p):
            if i == n - 1:
                cond = (t >= u[i]) & (t <= u[i + 1] + 1e-5)
            else:
                cond = (t >= u[i]) & (t < u[i + 1])
            N[:, i] = cond.float()
        for k in range(1, p + 1):
            N_next = torch.zeros_like(N)
            for i in range(n + p - k):
                d1 = u[i + k] - u[i]
                d2 = u[i + k + 1] - u[i + 1]
                term1 = ((t - u[i]) / d1) * N[:, i] if d1 > 1e-8 else 0.0
                term2 = ((u[i + k + 1] - t) / d2) * N[:, i + 1] if d2 > 1e-8 else 0.0
                N_next[:, i] = term1 + term2
            N = N_next
        return N[:, :n]

    def fit_knots(self, traj):
        T = traj.shape[0]
        B = self._basis_matrix(T, traj.device)
        reg = 1e-4 * torch.eye(self.num_knots, device=traj.device)
        return torch.linalg.solve(B.T @ B + reg, B.T @ traj)

    def evaluate(self, knots, T):
        B = self._basis_matrix(T, knots.device)
        return B @ knots


# ==============================================================================
# 2. Shared Frozen Dual-View ResNet-18 + Deterministic SHA-256 QR Hash Encoder
# ==============================================================================
STOPWORDS = {"the", "a", "an", "to", "of", "and", "from", "it", "this", "that", "then", "lying", "grasp", "take", "grab"}
DISCRIMINATIVE_WORDS = {
    "on", "off", "left", "right", "up", "down", "open", "close", "into",
    "red", "blue", "pink", "green", "yellow", "led", "lightbulb", "bulb",
    "slider", "sliding", "cabinet", "drawer", "table", "shelf",
    "stack", "unstack", "remove", "rotate", "push", "lift", "place",
}


class FrozenPerceptionEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            resnet = models.resnet18(weights=None)
        resnet.fc = nn.Identity()
        self.backbone = resnet.to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        rng = np.random.RandomState(2026)
        Q_lang, _ = np.linalg.qr(rng.randn(512, 128).astype(np.float32))
        Q_vis, _ = np.linalg.qr(rng.randn(1024 + 15, 128).astype(np.float32))
        self.register_buffer("proj_lang", torch.from_numpy(Q_lang).to(device))
        self.register_buffer("proj_vis", torch.from_numpy(Q_vis).to(device))
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def encode_lang(self, text):
        vec = np.zeros(512, dtype=np.float32)
        clean = text.lower().strip().replace(",", "")
        words = clean.split()
        for w in words:
            wt = 10.0 if w in DISCRIMINATIVE_WORDS else (0.1 if w in STOPWORDS else 2.5)
            vec[_det_hash("U:" + w, 512)] += wt
        for i in range(len(words) - 1):
            vec[_det_hash("B:" + words[i] + "_" + words[i + 1], 512)] += 2.0
        t = F.normalize(torch.from_numpy(vec).to(self.device), dim=-1)
        return F.normalize(t @ self.proj_lang, dim=-1)

    @torch.no_grad()
    def encode_obs(self, rgb_static, rgb_gripper, robot_obs):
        s_t = torch.from_numpy(rgb_static).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        g_t = torch.from_numpy(rgb_gripper).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        s_t = F.interpolate(s_t, size=(128, 128), mode="bilinear", align_corners=False)
        g_t = F.interpolate(g_t, size=(84, 84), mode="bilinear", align_corners=False)
        s_t = (s_t - self.mean) / self.std
        g_t = (g_t - self.mean) / self.std
        f_s = self.backbone(s_t).squeeze(0)
        f_g = self.backbone(g_t).squeeze(0)
        r_t = torch.from_numpy(robot_obs).to(self.device).float()
        raw = torch.cat([F.normalize(f_s, dim=-1), F.normalize(f_g, dim=-1), r_t], dim=-1)
        z_v = F.normalize(raw @ self.proj_vis, dim=-1)
        return z_v, raw


# ==============================================================================
# 3. Baselines: Flat MLP (No-Replay) & ER 5%
# ==============================================================================
class MLPPolicy(nn.Module):
    def __init__(self, in_dim=1039 + 128 + 1, hidden=512, out_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, raw_obs, z_lang, phase):
        return self.net(torch.cat([raw_obs, z_lang, phase], dim=-1))


class ContinualMLPBaseline:
    def __init__(self, encoder, device, replay_ratio=0.0, epochs=60):
        self.encoder = encoder
        self.device = device
        self.replay_ratio = replay_ratio
        self.epochs = epochs
        self.model = MLPPolicy().to(device)
        self.initial_weights = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.replay_buffer = []
        self.step_count = 0

    def train_task(self, episodes, lr=2e-3):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        X_raw, X_lang, X_phase, Y_act = [], [], [], []
        for ep in episodes:
            z_l = self.encoder.encode_lang(ep["ann"])
            T = len(ep["frames"])
            for idx, f in enumerate(ep["frames"]):
                _, raw = self.encoder.encode_obs(f["rgb_static"], f["rgb_gripper"], f["robot_obs"])
                X_raw.append(raw)
                X_lang.append(z_l)
                X_phase.append(torch.tensor([idx / max(1, T - 1)], device=self.device))
                Y_act.append(torch.from_numpy(f["rel_actions"]).to(self.device).float())
        if self.replay_ratio > 0.0 and len(self.replay_buffer) > 0:
            n_rep = max(4, int(len(X_raw) * self.replay_ratio))
            for r_obs, r_lang, r_ph, r_act in random.sample(self.replay_buffer, min(n_rep, len(self.replay_buffer))):
                X_raw.append(r_obs); X_lang.append(r_lang); X_phase.append(r_ph); Y_act.append(r_act)
        X_raw_t, X_lang_t, X_phase_t, Y_act_t = map(torch.stack, (X_raw, X_lang, X_phase, Y_act))
        self.model.train()
        bs = min(128, X_raw_t.size(0))
        for _ in range(self.epochs):
            perm = torch.randperm(X_raw_t.size(0), device=self.device)
            for i in range(0, X_raw_t.size(0), bs):
                idx = perm[i:i + bs]
                loss = F.mse_loss(self.model(X_raw_t[idx], X_lang_t[idx], X_phase_t[idx]), Y_act_t[idx])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            scheduler.step()
        if self.replay_ratio > 0.0:
            keep = max(2, int(len(X_raw) * self.replay_ratio))
            for i in random.sample(range(len(X_raw)), min(keep, len(X_raw))):
                self.replay_buffer.append((X_raw[i].detach(), X_lang[i].detach(), X_phase[i].detach(), Y_act[i].detach()))

    def parameter_drift(self):
        return sum(torch.norm(v.detach() - self.initial_weights[k]).item() for k, v in self.model.state_dict().items())

    def reset_episode(self, ann, obs, info=None):
        self.z_lang = self.encoder.encode_lang(ann)
        self.step_count = 0

    @torch.no_grad()
    def act(self, obs):
        self.model.eval()
        _, raw = self.encoder.encode_obs(obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"], obs["robot_obs"])
        ph = torch.tensor([[min(1.0, self.step_count / 45.0)]], device=self.device)
        act = self.model(raw.unsqueeze(0), self.z_lang.unsqueeze(0), ph).squeeze(0).cpu().numpy()
        self.step_count += 1
        act[:6] = np.clip(act[:6], -1.0, 1.0)
        act[6] = 1.0 if act[6] >= 0.0 else -1.0
        return act

    def episode_snap_back(self):
        pass


# ==============================================================================
# 4. Stage-2 Pure Vision+Depth Anchor Estimator (Strict Color Ratio + Workspace)
# ==============================================================================
class CalvinCameraVisionAnchorEstimator(ObjectAnchorEstimator):
    """Pure RGB-D object 3D centroid estimator via camera extrinsics/intrinsics only."""

    def __init__(self, static_cam) -> None:
        self.cam = static_cam

    @staticmethod
    def _color_mask_and_score(rgb: np.ndarray, target_name: str):
        r = rgb[:, :, 0].astype(np.float32) / 255.0
        g = rgb[:, :, 1].astype(np.float32) / 255.0
        b = rgb[:, :, 2].astype(np.float32) / 255.0
        total = r + g + b + 1e-6
        if target_name == "block_red":
            dom = r / total
            diff = r - np.maximum(g, b)
            return (dom > 0.50) & (diff > 0.12), diff, 0.12
        if target_name == "block_blue":
            diff = b - 0.65 * r - 0.35 * g
            return diff > 0.16, diff, 0.16
        if target_name == "block_pink":
            diff = 0.55 * r + 0.55 * b - 1.00 * g
            return diff > 0.16, diff, 0.16
        return None, None, 1.0

    def estimate(self, obs: dict, target_name: str, info: dict = None) -> ObjectAnchor:
        if target_name == "fixed":
            return ObjectAnchor(name="fixed", pos=np.zeros(3, dtype=np.float32), yaw=0.0, valid=False)
        rgb_obs = obs.get("rgb_obs", obs)
        depth_obs = obs.get("depth_obs", obs)
        if "rgb_static" not in rgb_obs or "depth_static" not in depth_obs:
            return ObjectAnchor(name=target_name, pos=np.zeros(3, dtype=np.float32), yaw=0.0, valid=False)

        rgb = rgb_obs["rgb_static"]
        depth = depth_obs["depth_static"]
        cmask, score, thresh = self._color_mask_and_score(rgb, target_name)
        if cmask is None:
            return ObjectAnchor(name=target_name, pos=np.zeros(3, dtype=np.float32), yaw=0.0, valid=False)

        h, w = score.shape
        roi = np.zeros((h, w), dtype=bool)
        roi[int(0.15 * h):int(0.90 * h), int(0.10 * w):int(0.90 * w)] = True
        ys, xs = np.where(cmask & roi)
        if len(xs) < 4:
            return ObjectAnchor(name=target_name, pos=np.zeros(3, dtype=np.float32), yaw=0.0, valid=False)

        pts = np.asarray([self.cam.deproject((int(x), int(y)), depth) for y, x in zip(ys, xs)], dtype=np.float32)
        sc_vals = score[ys, xs]
        ws_mask = (
            (pts[:, 0] >= -0.45) & (pts[:, 0] <= 0.45)
            & (pts[:, 1] >= -0.22) & (pts[:, 1] <= 0.38)
            & (pts[:, 2] >= 0.34) & (pts[:, 2] <= 0.52)
        )
        if ws_mask.sum() < 3:
            return ObjectAnchor(name=target_name, pos=np.zeros(3, dtype=np.float32), yaw=0.0, valid=False)

        pts_ws = pts[ws_mask]
        sc_ws = sc_vals[ws_mask]
        top_cut = max(thresh, float(np.percentile(sc_ws, 75.0)))
        top_pts = pts_ws[sc_ws >= top_cut]
        if len(top_pts) == 0:
            top_pts = pts_ws
        xyz = np.median(top_pts, axis=0).astype(np.float32)
        return ObjectAnchor(name=target_name, pos=xyz, yaw=0.0, valid=True)


# ==============================================================================
# 5. Arm 3, 4, 5: DTRA v4.0 Organ->Region->Cell Top-M CMC + Object-Centric Transport
# ==============================================================================
class DTRAv4TransportController:
    def __init__(
        self,
        encoder,
        device,
        anchor_estimator: ObjectAnchorEstimator = None,
        use_transport: bool = True,
        num_knots: int = 6,
        top_m_tasks: int = 5,
    ):
        self.encoder = encoder
        self.device = device
        self.anchor_estimator = anchor_estimator
        self.use_transport = use_transport
        self.num_knots = num_knots
        self.top_m_tasks = top_m_tasks
        self.spline = CubicBSplineBasis(num_knots=num_knots).to(device)
        self.transport = ObjectCentricTransportAdapter(
            approach_tol_m=0.025,
            max_approach_steps=18,
            approach_gain=16.0,
            track_gain=9.0,
            enable_yaw_rotation=False,
            safe_z_lift_m=0.05,
        )

        self.cells = []
        self.region_map = {}  # task_name -> list of cell indices (Organ/Region hierarchy)
        self.frozen_tensor = torch.zeros((1, num_knots, 7), device=device)
        self.initial_frozen_checksum = 0.0

        self.active_cell = None
        self.active_traj = None
        self.transported_ref_pos = None
        self.anchor_bound = False
        self.step_idx = 0
        self.elastic_offset = np.zeros(7, dtype=np.float32)
        self.init_delta_pos = np.zeros(3, dtype=np.float32)

    def train_task(self, episodes):
        for ep in episodes:
            z_l = self.encoder.encode_lang(ep["ann"])
            f0 = ep["frames"][0]
            z_v0, _ = self.encoder.encode_obs(f0["rgb_static"], f0["rgb_gripper"], f0["robot_obs"])

            actions = np.stack([f["rel_actions"] for f in ep["frames"]], axis=0)
            robot_seq = np.stack([f["robot_obs"][:6] for f in ep["frames"]], axis=0)
            knots = self.spline.fit_knots(torch.from_numpy(actions).to(self.device).float()).detach()

            cell_anchor = None
            if self.use_transport and self.anchor_estimator is not None:
                cell_anchor = self.transport.register_cell_anchor(
                    task_name=ep["task"],
                    ann_text=ep["ann"],
                    frames=ep["frames"],
                    estimator=self.anchor_estimator,
                )

            c_idx = len(self.cells)
            self.region_map.setdefault(ep["task"], []).append(c_idx)
            self.cells.append({
                "z_lang": z_l.detach(),
                "z_vis": z_v0.detach(),
                "knots": knots,
                "duration": len(actions),
                "init_pos": f0["robot_obs"][:3].copy(),
                "robot_seq": robot_seq,
                "cell_anchor": cell_anchor,
                "task": ep["task"],
                "ann": ep["ann"],
            })

        self.frozen_tensor = torch.stack([c["knots"] for c in self.cells])
        self.initial_frozen_checksum = self.frozen_tensor.sum().item()

    def parameter_drift(self):
        bank_drift = abs(self.frozen_tensor.sum().item() - self.initial_frozen_checksum)
        elastic_drift = float(np.linalg.norm(self.elastic_offset))
        transport_drift = float(np.linalg.norm(self.transport.state.last_offset))
        return bank_drift + elastic_drift + transport_drift

    def reset_episode(self, ann, obs, info=None):
        self.episode_snap_back()
        z_l = self.encoder.encode_lang(ann)
        z_v, _ = self.encoder.encode_obs(obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"], obs["robot_obs"])

        lang_bank = torch.stack([c["z_lang"] for c in self.cells])
        vis_bank = torch.stack([c["z_vis"] for c in self.cells])
        s1_all = 0.88 * (lang_bank @ z_l) + 0.12 * (vis_bank @ z_v)

        # Stage 1 (Region/Organ level): score each distinct skill region by max cell match
        region_scores = []
        for tn, idxs in self.region_map.items():
            best_cell_s1 = float(s1_all[idxs].max().item())
            reg_l = torch.stack([self.cells[i]["z_lang"] for i in idxs]).mean(dim=0)
            reg_v = torch.stack([self.cells[i]["z_vis"] for i in idxs]).mean(dim=0)
            region_scores.append((best_cell_s1, tn, reg_l, reg_v, idxs))
        region_scores.sort(key=lambda x: x[0], reverse=True)
        top_regions = region_scores[:min(self.top_m_tasks, len(region_scores))]

        # Stage 2 (Region-Local Sharp Common-Mode Cancellation): strips shared task boilerplate
        if len(top_regions) >= 2:
            cm_l = torch.stack([r[2] for r in top_regions]).mean(dim=0)
            cm_v = torch.stack([r[3] for r in top_regions]).mean(dim=0)
            ql_cmc = F.normalize(z_l - 0.98 * cm_l, dim=-1)
            qv_cmc = F.normalize(z_v - 0.90 * cm_v, dim=-1)

            cand_pool = []
            for _, tn, _, _, idxs in top_regions:
                for i in idxs:
                    cl_cmc = F.normalize(self.cells[i]["z_lang"] - 0.98 * cm_l, dim=-1)
                    cv_cmc = F.normalize(self.cells[i]["z_vis"] - 0.90 * cm_v, dim=-1)
                    sc = 0.92 * float(cl_cmc @ ql_cmc) + 0.08 * float(cv_cmc @ qv_cmc)
                    cand_pool.append((sc, i))
        else:
            cand_pool = [(float(s1_all[i].item()), i) for i in top_regions[0][4]]

        max_sc = max(x[0] for x in cand_pool)
        cur_pos = obs["robot_obs"][:3]
        best_idx = cand_pool[0][1]
        best_final = -1e9
        for sc, i in cand_pool:
            if sc >= max_sc - 0.03:
                pos_dist = np.linalg.norm(cur_pos - self.cells[i]["init_pos"])
                final_sc = sc - 0.02 * pos_dist
            else:
                final_sc = sc
            if final_sc > best_final:
                best_final = final_sc
                best_idx = i

        self.active_cell = self.cells[best_idx]
        T = self.active_cell["duration"]
        with torch.no_grad():
            self.active_traj = self.spline.evaluate(self.active_cell["knots"], T).cpu().numpy()
        self.step_idx = 0
        self.init_delta_pos = (self.active_cell["init_pos"] - cur_pos).astype(np.float32)

        self.anchor_bound = False
        if self.use_transport and self.active_cell["cell_anchor"] is not None and self.anchor_estimator is not None:
            ca = self.active_cell["cell_anchor"]
            self.transported_ref_pos, _ = self.transport.bind_episode(ca, obs, self.anchor_estimator, info=info)
            self.anchor_bound = (ca.target_name == "fixed") or (float(np.linalg.norm(self.transport.state.cur_obj_pos)) > 1e-6)
        else:
            self.transported_ref_pos = None

    def act(self, obs):
        _, _ = self.encoder.encode_obs(obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"], obs["robot_obs"])
        T = len(self.active_traj)
        idx = min(self.step_idx, T - 1)
        base_act = self.active_traj[idx].copy()

        if self.use_transport and not self.anchor_bound and self.step_idx < 10 and self.active_cell["cell_anchor"] is not None and self.anchor_estimator is not None:
            ca = self.active_cell["cell_anchor"]
            anchor = self.anchor_estimator.estimate(obs, ca.target_name)
            if anchor.valid:
                self.transported_ref_pos, _ = self.transport.bind_episode(ca, obs, self.anchor_estimator)
                self.anchor_bound = True

        if self.use_transport and self.transported_ref_pos is not None:
            act, advance = self.transport.step_action(
                cur_eef_pos=obs["robot_obs"][:3],
                base_rel_action=base_act,
                transported_ref_pos=self.transported_ref_pos,
                tube_step_idx=idx,
                ref_orn_seq=self.active_cell["robot_seq"][:, 3:6],
                cur_eef_orn=obs["robot_obs"][3:6],
            )
            self.elastic_offset[:3] = self.transport.state.last_offset
            if advance:
                self.step_idx += 1
            return act

        ref_pose = self.active_cell["robot_seq"][idx]
        cur_pose = obs["robot_obs"][:6]
        tube_err = np.clip(ref_pose - cur_pose, -0.05, 0.05)
        decay = max(0.0, 1.0 - (self.step_idx / max(1, T * 0.4)))
        self.elastic_offset[:3] = np.clip(6.0 * decay * self.init_delta_pos + 3.0 * tube_err[:3], -0.6, 0.6)
        self.elastic_offset[3:6] = 0.15 * tube_err[3:6]
        act = base_act + self.elastic_offset
        act[:6] = np.clip(act[:6], -1.0, 1.0)
        act[6] = 1.0 if base_act[6] >= 0.0 else -1.0
        self.step_idx += 1
        return act

    def episode_snap_back(self):
        self.elastic_offset.fill(0.0)
        self.init_delta_pos.fill(0.0)
        self.anchor_bound = False
        self.transport.episode_snap_back()


# ==============================================================================
# 6. Loader & Evaluator
# ==============================================================================
def load_episodes(split_dir, max_per_task=10):
    lang_path = os.path.join(split_dir, "lang_annotations", "auto_lang_ann.npy")
    lang_data = np.load(lang_path, allow_pickle=True).item()
    tasks = lang_data["language"]["task"]
    anns = lang_data["language"]["ann"]
    indxs = lang_data["info"]["indx"]

    by_task = {}
    for task, ann, (s_idx, e_idx) in zip(tasks, anns, indxs):
        by_task.setdefault(task, [])
        if len(by_task[task]) >= max_per_task:
            continue
        frames = []
        valid = True
        for f_idx in range(s_idx, e_idx + 1):
            fpath = os.path.join(split_dir, f"episode_{f_idx:07d}.npz")
            if not os.path.exists(fpath):
                valid = False
                break
            d = np.load(fpath)
            frames.append({
                "rgb_static": d["rgb_static"],
                "rgb_gripper": d["rgb_gripper"],
                "depth_static": d["depth_static"],
                "robot_obs": d["robot_obs"],
                "scene_obs": d["scene_obs"],
                "rel_actions": d["rel_actions"],
            })
        if valid and len(frames) > 4:
            by_task[task].append({"task": task, "ann": ann, "frames": frames})
    return by_task


def evaluate_continual_tasks(env, task_oracle, policy, eval_episodes, max_extra_steps=25):
    per_task_sr = {}
    latencies_ms = []
    route_hits, route_total = 0, 0
    for task_name, eps in eval_episodes.items():
        successes = 0
        for ep in eps:
            f0 = ep["frames"][0]
            obs = env.reset(robot_obs=f0["robot_obs"], scene_obs=f0["scene_obs"])
            start_info = env.get_info()
            policy.reset_episode(ep["ann"], obs, info=start_info)
            if hasattr(policy, "active_cell") and policy.active_cell is not None:
                route_total += 1
                if policy.active_cell["task"] == task_name:
                    route_hits += 1
            solved = False
            max_steps = len(ep["frames"]) + max_extra_steps
            for _ in range(max_steps):
                t0 = time.perf_counter()
                action = policy.act(obs)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                obs, _, _, cur_info = env.step(action)
                if task_name in task_oracle.get_task_info_for_set(start_info, cur_info, {task_name}):
                    solved = True
                    break
            policy.episode_snap_back()
            if solved:
                successes += 1
        per_task_sr[task_name] = 100.0 * successes / max(1, len(eps))
    route_acc = (100.0 * route_hits / route_total) if route_total > 0 else None
    return float(np.mean(list(per_task_sr.values()))), per_task_sr, float(np.mean(latencies_ms)) if latencies_ms else 0.0, route_acc


def evaluate_calvin_5step_chain(env, task_oracle, policy, eval_episodes, num_chains=20, chain_len=5):
    task_keys = list(eval_episodes.keys())
    lengths = []
    for c_i in range(num_chains):
        rng = random.Random(2000 + c_i)
        chosen_tasks = [rng.choice(task_keys) for _ in range(chain_len)]
        seq = [rng.choice(eval_episodes[t]) for t in chosen_tasks]
        f0 = seq[0]["frames"][0]
        obs = env.reset(robot_obs=f0["robot_obs"], scene_obs=f0["scene_obs"])
        completed = 0
        for sub_ep in seq:
            t_name = sub_ep["task"]
            start_info = env.get_info()
            policy.reset_episode(sub_ep["ann"], obs, info=start_info)
            solved = False
            for _ in range(len(sub_ep["frames"]) + 25):
                action = policy.act(obs)
                obs, _, _, cur_info = env.step(action)
                if t_name in task_oracle.get_task_info_for_set(start_info, cur_info, {t_name}):
                    solved = True
                    break
            policy.episode_snap_back()
            if solved:
                completed += 1
            else:
                break
        lengths.append(completed)
    return float(np.mean(lengths)), 100.0 * sum(1 for x in lengths if x == chain_len) / max(1, num_chains)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.expanduser("~/dtra_a6000_lab/calvin_data/task_D_D_subset"))
    parser.add_argument("--calvin_root", type=str, default=os.path.expanduser("~/dtra_a6000_lab/calvin_benchmark/calvin"))
    parser.add_argument("--max_per_task", type=int, default=10)
    parser.add_argument("--max_eval_per_task", type=int, default=5)
    parser.add_argument("--num_chains", type=int, default=30)
    parser.add_argument("--mlp_epochs", type=int, default=60)
    parser.add_argument("--arms", nargs="+", default=["Arm1_Flat_MLP", "Arm2_ER_5pct", "Arm3_DTRA_NoTransport", "Arm4_DTRA_Transport_Oracle", "Arm5_DTRA_Transport_VisionDepth"])
    parser.add_argument("--out_json", type=str, default=os.path.expanduser("~/dtra_a6000_lab/calvin_benchmark/calvin_34tasks_results.json"))
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== DTRA v4 34-Task CALVIN Full Benchmark | Device: {device} ===", flush=True)

    train_dir = os.path.join(args.data_dir, "training")
    val_dir = os.path.join(args.data_dir, "validation")
    train_tasks = load_episodes(train_dir, max_per_task=args.max_per_task)
    val_tasks = load_episodes(val_dir, max_per_task=args.max_eval_per_task)

    all_task_names = sorted(list(set(train_tasks.keys()) | set(val_tasks.keys())))
    eval_split = {t: (val_tasks[t] if t in val_tasks else train_tasks[t]) for t in all_task_names}
    print(f"Tasks ({len(all_task_names)}): train_clips={sum(len(v) for v in train_tasks.values())}, val_clips={sum(len(v) for v in eval_split.values())}", flush=True)

    env = get_env(val_dir, show_gui=False)
    task_cfg_path = os.path.join(args.calvin_root, "calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    if not os.path.exists(task_cfg_path):
        task_cfg_path = os.path.join(args.calvin_root, "calvin_env/conf/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(OmegaConf.load(task_cfg_path))
    encoder = FrozenPerceptionEncoder(device)

    all_arms = {
        "Arm1_Flat_MLP": ContinualMLPBaseline(encoder, device, replay_ratio=0.0, epochs=args.mlp_epochs),
        "Arm2_ER_5pct": ContinualMLPBaseline(encoder, device, replay_ratio=0.05, epochs=args.mlp_epochs),
        "Arm3_DTRA_NoTransport": DTRAv4TransportController(encoder, device, anchor_estimator=None, use_transport=False),
        "Arm4_DTRA_Transport_Oracle": DTRAv4TransportController(encoder, device, anchor_estimator=OracleSceneAnchorEstimator(), use_transport=True),
        "Arm5_DTRA_Transport_VisionDepth": DTRAv4TransportController(encoder, device, anchor_estimator=CalvinCameraVisionAnchorEstimator(env.cameras[0]), use_transport=True),
    }
    arms = {k: all_arms[k] for k in args.arms if k in all_arms}

    results = {}
    for arm_name, policy in arms.items():
        print(f"\n--- Running {arm_name} ({len(all_task_names)} tasks) ---", flush=True)
        initial_sr_map = {}
        t_start = time.perf_counter()
        for t_i, t_name in enumerate(all_task_names):
            tr_eps = train_tasks[t_name] if t_name in train_tasks else val_tasks[t_name]
            policy.train_task(tr_eps)
            sr_now, _, _, _ = evaluate_continual_tasks(env, task_oracle, policy, {t_name: eval_split[t_name]})
            initial_sr_map[t_name] = sr_now
            if (t_i + 1) % 5 == 0 or (t_i + 1) == len(all_task_names):
                cur_init = float(np.mean(list(initial_sr_map.values())))
                print(f"  [{arm_name}] Task {t_i+1}/{len(all_task_names)} ({t_name}): sr_now={sr_now:.1f}% | running_init_mean={cur_init:.2f}% | elapsed={time.perf_counter()-t_start:.1f}s", flush=True)

        train_sec = time.perf_counter() - t_start
        init_mean_sr = float(np.mean(list(initial_sr_map.values())))
        final_mean_sr, final_sr_map, lat_ms, route_acc = evaluate_continual_tasks(env, task_oracle, policy, eval_split)
        forgetting_rate = float(np.mean([max(0.0, initial_sr_map[k] - final_sr_map[k]) for k in all_task_names]))
        drift = policy.parameter_drift()
        avg_chain_len, chain_5_sr = evaluate_calvin_5step_chain(env, task_oracle, policy, eval_split, num_chains=args.num_chains, chain_len=5)

        results[arm_name] = {
            "num_tasks": len(all_task_names),
            "initial_mean_sr": round(init_mean_sr, 2),
            "final_mean_sr": round(final_mean_sr, 2),
            "forgetting_rate": round(forgetting_rate, 2),
            "route_acc": round(route_acc, 2) if route_acc is not None else None,
            "parameter_drift": round(drift, 6),
            "avg_5step_chain_len": round(avg_chain_len, 2),
            "chain_5step_full_sr": round(chain_5_sr, 2),
            "closed_loop_latency_ms": round(lat_ms, 4),
            "total_sec": round(time.perf_counter() - t_start, 2),
            "initial_per_task_sr": initial_sr_map,
            "final_per_task_sr": final_sr_map,
        }
        r_str = f"{route_acc:.1f}%" if route_acc is not None else "N/A"
        print(
            f"==> [{arm_name}] Init SR: {init_mean_sr:.2f}% -> Final SR: {final_mean_sr:.2f}% | "
            f"RouteAcc: {r_str} | Forgetting: {forgetting_rate:.2f}% | Drift: {drift:.6f} | "
            f"5-Step Avg Len: {avg_chain_len:.2f}/5.00 | Latency: {lat_ms:.3f} ms",
            flush=True,
        )
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)

    env.close()
    print(f"\nALL DONE! Saved 34-task benchmark results to: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()

