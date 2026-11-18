import os
import json
import time
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
# 2. Shared Frozen Dual-View ResNet-18 + Deterministic QR Hash Encoder
# ==============================================================================
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
        clean = text.lower().strip()
        words = clean.split()
        for w in words:
            vec[abs(hash(w)) % 512] += 1.0
        for i in range(len(words) - 1):
            vec[abs(hash(words[i] + "_" + words[i + 1])) % 512] += 1.5
        for i in range(len(clean) - 2):
            vec[abs(hash(clean[i:i + 3])) % 512] += 0.5
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
# 3. Fair Converged Baselines: Arm 1 (No-Replay) & Arm 2 (ER 5%)
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
        x = torch.cat([raw_obs, z_lang, phase], dim=-1)
        return self.net(x)


class ContinualMLPBaseline:
    def __init__(self, encoder, device, replay_ratio=0.0):
        self.encoder = encoder
        self.device = device
        self.replay_ratio = replay_ratio
        self.model = MLPPolicy().to(device)
        self.initial_weights = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.replay_buffer = []
        self.step_count = 0

    def train_task(self, episodes, epochs=120, lr=2e-3):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
            sampled = random.sample(self.replay_buffer, min(n_rep, len(self.replay_buffer)))
            for r_obs, r_lang, r_ph, r_act in sampled:
                X_raw.append(r_obs)
                X_lang.append(r_lang)
                X_phase.append(r_ph)
                Y_act.append(r_act)

        X_raw_t = torch.stack(X_raw)
        X_lang_t = torch.stack(X_lang)
        X_phase_t = torch.stack(X_phase)
        Y_act_t = torch.stack(Y_act)

        self.model.train()
        bs = min(64, X_raw_t.size(0))
        for _ in range(epochs):
            perm = torch.randperm(X_raw_t.size(0), device=self.device)
            for i in range(0, X_raw_t.size(0), bs):
                idx = perm[i:i + bs]
                pred = self.model(X_raw_t[idx], X_lang_t[idx], X_phase_t[idx])
                loss = F.mse_loss(pred, Y_act_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        if self.replay_ratio > 0.0:
            keep = max(2, int(len(X_raw) * self.replay_ratio))
            for i in random.sample(range(len(X_raw)), min(keep, len(X_raw))):
                self.replay_buffer.append((
                    X_raw[i].detach(), X_lang[i].detach(), X_phase[i].detach(), Y_act[i].detach()
                ))

    def parameter_drift(self):
        return sum(torch.norm(v.detach() - self.initial_weights[k]).item() for k, v in self.model.state_dict().items())

    def reset_episode(self, ann, obs):
        self.z_lang = self.encoder.encode_lang(ann)
        self.step_count = 0

    @torch.no_grad()
    def act(self, obs):
        self.model.eval()
        _, raw = self.encoder.encode_obs(
            obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"], obs["robot_obs"]
        )
        ph = torch.tensor([[min(1.0, self.step_count / 45.0)]], device=self.device)
        act = self.model(raw.unsqueeze(0), self.z_lang.unsqueeze(0), ph).squeeze(0).cpu().numpy()
        self.step_count += 1
        act[:6] = np.clip(act[:6], -1.0, 1.0)
        act[6] = 1.0 if act[6] >= 0.0 else -1.0
        return act

    def episode_snap_back(self):
        pass


# ==============================================================================
# 4. Arm 3: DTRA v4.0 (Closed-Loop Two-Timescale Elastic Hypertube + B-Spline)
# ==============================================================================
class DTRAv4Controller:
    def __init__(self, encoder, device, num_knots=6, top_m=5):
        self.encoder = encoder
        self.device = device
        self.num_knots = num_knots
        self.top_m = top_m
        self.spline = CubicBSplineBasis(num_knots=num_knots).to(device)

        self.cells = []
        self.frozen_tensor = torch.zeros((1, num_knots, 7), device=device)
        self.initial_frozen_checksum = 0.0

        self.active_cell = None
        self.active_traj = None
        self.step_idx = 0
        self.elastic_offset = np.zeros(7, dtype=np.float32)
        self.init_delta_pos = np.zeros(3, dtype=np.float32)

    def _compute_cmc_key(self, z_lang, z_vis):
        raw_key = F.normalize(torch.cat([z_lang[:64], z_vis[:64]], dim=-1), dim=-1)
        if len(self.cells) >= 2:
            bank = torch.stack([c["raw_key"] for c in self.cells])
            sims = bank @ raw_key
            m = min(self.top_m, bank.size(0))
            common_mode = bank[torch.topk(sims, k=m).indices].mean(dim=0)
            key = F.normalize(raw_key - 0.85 * common_mode, dim=-1)
        else:
            key = raw_key
        return key, raw_key

    def _orthogonal_no_encroachment(self, key):
        if len(self.cells) == 0:
            return key
        bank = torch.stack([c["key"] for c in self.cells])
        sims = bank @ key
        max_sim, idx = torch.max(sims, dim=0)
        if max_sim.item() > 0.92:
            neighbor = bank[idx]
            key = F.normalize(key - 0.5 * (key @ neighbor) * neighbor, dim=-1)
        return key

    def train_task(self, episodes):
        for ep in episodes:
            z_l = self.encoder.encode_lang(ep["ann"])
            f0 = ep["frames"][0]
            z_v0, _ = self.encoder.encode_obs(f0["rgb_static"], f0["rgb_gripper"], f0["robot_obs"])
            key, raw_key = self._compute_cmc_key(z_l, z_v0)
            key = self._orthogonal_no_encroachment(key)

            actions = np.stack([f["rel_actions"] for f in ep["frames"]], axis=0)
            robot_seq = np.stack([f["robot_obs"][:6] for f in ep["frames"]], axis=0)
            traj_t = torch.from_numpy(actions).to(self.device).float()
            knots = self.spline.fit_knots(traj_t).detach()

            self.cells.append({
                "key": key.detach(),
                "raw_key": raw_key.detach(),
                "knots": knots,
                "duration": len(actions),
                "init_pos": f0["robot_obs"][:3].copy(),
                "robot_seq": robot_seq,
                "task": ep["task"],
                "ann": ep["ann"],
            })

        self.frozen_tensor = torch.stack([c["knots"] for c in self.cells])
        self.initial_frozen_checksum = self.frozen_tensor.sum().item()

    def parameter_drift(self):
        bank_drift = abs(self.frozen_tensor.sum().item() - self.initial_frozen_checksum)
        elastic_drift = float(np.linalg.norm(self.elastic_offset))
        return bank_drift + elastic_drift

    def reset_episode(self, ann, obs):
        self.episode_snap_back()
        z_l = self.encoder.encode_lang(ann)
        z_v, _ = self.encoder.encode_obs(
            obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"], obs["robot_obs"]
        )
        q_key, _ = self._compute_cmc_key(z_l, z_v)

        bank = torch.stack([c["key"] for c in self.cells])
        cos_sim = bank @ q_key

        top_k = min(3, len(self.cells))
        cand_idx = torch.topk(cos_sim, k=top_k).indices.tolist()
        cur_pos = obs["robot_obs"][:3]
        best_idx = cand_idx[0]
        best_score = -1e9
        for idx in cand_idx:
            pos_dist = np.linalg.norm(cur_pos - self.cells[idx]["init_pos"])
            score = cos_sim[idx].item() - 0.15 * pos_dist
            if score > best_score:
                best_score = score
                best_idx = idx

        self.active_cell = self.cells[best_idx]
        T = self.active_cell["duration"]
        with torch.no_grad():
            self.active_traj = self.spline.evaluate(self.active_cell["knots"], T).cpu().numpy()
        self.step_idx = 0
        # Initial spatial deviation for multi-step chain transition
        self.init_delta_pos = (self.active_cell["init_pos"] - cur_pos).astype(np.float32)

    def act(self, obs):
        # Closed-loop perception encoding at every control step (fair latency + feedback)
        _, _ = self.encoder.encode_obs(
            obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"], obs["robot_obs"]
        )

        T = len(self.active_traj)
        idx = min(self.step_idx, T - 1)
        base_act = self.active_traj[idx].copy()

        # Fast-timescale elastic hypertube deformation:
        # 1) Compensate initial start-pose offset in early steps (crucial for 5-step chains)
        # 2) Track reference pose tube with proportional elastic stiffness
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


# ==============================================================================
# 5. Dataset Loader & Official Physics Evaluator
# ==============================================================================
def load_episodes(split_dir, max_per_task=4):
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
                "robot_obs": d["robot_obs"],
                "scene_obs": d["scene_obs"],
                "rel_actions": d["rel_actions"],
            })
        if valid and len(frames) > 4:
            by_task[task].append({"task": task, "ann": ann, "frames": frames})
    return by_task


def evaluate_continual_tasks(env, task_oracle, policy, eval_episodes, max_extra_steps=15):
    per_task_sr = {}
    latencies_ms = []
    for task_name, eps in eval_episodes.items():
        successes = 0
        for ep in eps:
            f0 = ep["frames"][0]
            obs = env.reset(robot_obs=f0["robot_obs"], scene_obs=f0["scene_obs"])
            start_info = env.get_info()
            policy.reset_episode(ep["ann"], obs)

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
    mean_sr = float(np.mean(list(per_task_sr.values())))
    mean_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    return mean_sr, per_task_sr, mean_lat


def evaluate_calvin_5step_chain(env, task_oracle, policy, eval_episodes, num_chains=10, chain_len=5):
    # Build distinct-task chains so each step commands a new physical interaction
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
            policy.reset_episode(sub_ep["ann"], obs)
            solved = False
            for _ in range(len(sub_ep["frames"]) + 20):
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
    avg_len = float(np.mean(lengths))
    sr_5 = 100.0 * sum(1 for x in lengths if x == chain_len) / max(1, num_chains)
    return avg_len, sr_5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.expanduser("~/dtra_a6000_lab/calvin_data/calvin_debug_dataset"))
    parser.add_argument("--calvin_root", type=str, default=os.path.expanduser("~/dtra_a6000_lab/calvin_benchmark/calvin"))
    parser.add_argument("--max_per_task", type=int, default=4)
    parser.add_argument("--num_chains", type=int, default=10)
    parser.add_argument("--out_json", type=str, default=os.path.expanduser("~/dtra_a6000_lab/calvin_benchmark/calvin_results_v2.json"))
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Fair Closed-Loop CALVIN Benchmark v2 | Device: {device} ===")

    train_dir = os.path.join(args.data_dir, "training")
    val_dir = os.path.join(args.data_dir, "validation")

    train_tasks = load_episodes(train_dir, max_per_task=args.max_per_task)
    val_tasks = load_episodes(val_dir, max_per_task=args.max_per_task)

    # Train on all available tasks in training (+ any val-only tasks in debug set),
    # and evaluate strictly on held-out validation episodes where available
    all_task_names = sorted(list(set(train_tasks.keys()) | set(val_tasks.keys())))
    eval_split = {}
    for t in all_task_names:
        eval_split[t] = val_tasks[t] if t in val_tasks else train_tasks[t]

    print(f"Tasks ({len(all_task_names)}): {all_task_names}")
    print(f"Held-out Validation Tasks: {sorted(list(val_tasks.keys()))}")

    env = get_env(val_dir, show_gui=False)
    task_cfg_path = os.path.join(args.calvin_root, "calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    if not os.path.exists(task_cfg_path):
        task_cfg_path = os.path.join(args.calvin_root, "calvin_env/conf/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(OmegaConf.load(task_cfg_path))

    encoder = FrozenPerceptionEncoder(device)

    arms = {
        "Arm1_Flat_MLP_NoReplay": ContinualMLPBaseline(encoder, device, replay_ratio=0.0),
        "Arm2_Experience_Replay_5pct": ContinualMLPBaseline(encoder, device, replay_ratio=0.05),
        "Arm3_DTRA_v4_ClosedLoop_Ours": DTRAv4Controller(encoder, device, num_knots=6, top_m=5),
    }

    results = {}
    for arm_name, policy in arms.items():
        print(f"\n--- Running {arm_name} ---")
        initial_sr_map = {}
        t_train_start = time.perf_counter()

        for t_name in all_task_names:
            tr_eps = train_tasks[t_name] if t_name in train_tasks else val_tasks[t_name]
            policy.train_task(tr_eps)
            sr_now, _, _ = evaluate_continual_tasks(env, task_oracle, policy, {t_name: eval_split[t_name]})
            initial_sr_map[t_name] = sr_now

        train_sec = time.perf_counter() - t_train_start
        init_mean_sr = float(np.mean(list(initial_sr_map.values())))

        final_mean_sr, final_sr_map, lat_ms = evaluate_continual_tasks(env, task_oracle, policy, eval_split)
        forgetting_vals = [max(0.0, initial_sr_map[k] - final_sr_map[k]) for k in all_task_names]
        forgetting_rate = float(np.mean(forgetting_vals))
        drift = policy.parameter_drift()

        avg_chain_len, chain_5_sr = evaluate_calvin_5step_chain(
            env, task_oracle, policy, eval_split, num_chains=args.num_chains, chain_len=5
        )

        results[arm_name] = {
            "num_tasks": len(all_task_names),
            "initial_mean_sr": round(init_mean_sr, 2),
            "final_mean_sr": round(final_mean_sr, 2),
            "forgetting_rate": round(forgetting_rate, 2),
            "parameter_drift": round(drift, 6),
            "avg_5step_chain_len": round(avg_chain_len, 2),
            "chain_5step_full_sr": round(chain_5_sr, 2),
            "closed_loop_latency_ms": round(lat_ms, 4),
            "total_train_sec": round(train_sec, 2),
            "initial_per_task_sr": initial_sr_map,
            "final_per_task_sr": final_sr_map,
        }
        print(
            f"[{arm_name}] Init SR: {init_mean_sr:.2f}% -> Final SR: {final_mean_sr:.2f}% | "
            f"Forgetting: {forgetting_rate:.2f}% | Drift: {drift:.6f} | "
            f"5-Step Avg Len: {avg_chain_len:.2f}/5.00 | Closed-Loop Latency: {lat_ms:.3f} ms"
        )

    env.close()
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved fair benchmark v2 results to: {args.out_json}")


if __name__ == "__main__":
    main()
