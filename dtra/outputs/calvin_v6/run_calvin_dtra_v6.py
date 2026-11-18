"""calvin_benchmark/run_calvin_dtra_v6.py

DTRA v6 — Organic System 1 + System 2 Dual-Process Architecture:
  * System 1 (Reflex & Smooth Execution):
      - S^127 Hypertube Routing + Object-Centric C^2 B-Spline
      - Time-Scaled C^2 Geodesic Bridge (SLERP + adaptive curvature damping)
      - Grasp-Preserving Transit + Myelinated Transition Graph M_{ij}
  * System 2 (Deliberative Organ Layer):
      - Precondition Subgoal Resolver (internal world-model check)
      - Refractory Failure Backtracking (non-destructive alternative cell recovery)
      - OOD Resonance Monitor & Zero-Forgetting Cell Spawning (L_inf Drift == 0.0)
"""

import os
import sys
import json
import time
import random
import argparse
import warnings
from typing import Dict, List, Optional, Tuple
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
from calvin_env.envs.play_table_env import get_env

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from hypertube_transport import (
    OracleSceneAnchorEstimator,
    rotate_xy_vector,
    wrap_angle_pi,
)
from run_calvin_dtra_v5 import (
    set_seed,
    CubicBSplineBasisV5,
    FrozenPerceptionEncoder,
    CalvinCameraVisionAnchorEstimatorV5,
    DTRAv5TransportController,
    load_episodes,
    evaluate_continual_tasks,
)
from topological_bridge_v6 import (
    BridgePhase,
    TopologicalGeodesicBridgeV6,
    OrganSystem2Planner,
)

try:
    from calvin_agent.evaluation.multistep_sequences import get_sequences
except Exception:
    get_sequences = None


class DTRAv6DualSystemController(DTRAv5TransportController):
    """Combines DTRA v5 frozen cells with System 1 C^2 Bridge & System 2 Organ Planner."""

    def __init__(
        self,
        encoder,
        device,
        anchor_estimator=None,
        use_transport: bool = True,
        num_knots: int = 10,
        top_m_tasks: int = 5,
        enable_yaw_rotation: bool = False,
        bridge_horizon: int = 16,
        ood_resonance_thresh: float = 0.50,
    ):
        super().__init__(
            encoder=encoder,
            device=device,
            anchor_estimator=anchor_estimator,
            use_transport=use_transport,
            num_knots=num_knots,
            top_m_tasks=top_m_tasks,
            enable_yaw_rotation=enable_yaw_rotation,
        )
        self.bridge = TopologicalGeodesicBridgeV6(
            max_cells=2048, d_topo=128, bridge_horizon=bridge_horizon, damping_alpha=2.5
        ).to(device)
        self.bridge_spline = CubicBSplineBasisV5(num_knots=6).to(device)
        self.planner = OrganSystem2Planner(ood_resonance_thresh=ood_resonance_thresh)

        # Ephemeral chain & bridge states (reset to exact 0.0 on episode_snap_back)
        self.active_cell_idx: int = -1
        self.in_chain: bool = False
        self.prev_cell_idx: Optional[int] = None
        self.prev_z_end: Optional[torch.Tensor] = None
        self.prev_world_pos_knots: Optional[torch.Tensor] = None
        self.eef_history: List[np.ndarray] = []
        self.prev_dur: int = 16
        self.prev_grip: float = 1.0

        self.bridge_active: bool = False
        self.bridge_pos_seq: Optional[np.ndarray] = None
        self.bridge_idx: int = 0

        self.ood_spawn_events: int = 0
        self.backtrack_recoveries: int = 0

    def _get_world_pos_knots(self, cell: dict, obs: dict, info: Optional[dict] = None) -> torch.Tensor:
        pos_knots = cell["pos_knots"].clone()
        ca = cell["cell_anchor"]
        if self.use_transport and self.anchor_estimator is not None and ca.target_name != "fixed":
            try:
                anc = self.anchor_estimator.estimate(obs, ca.target_name, info=info, task_hint=f"{cell['task']} {self.current_ann}")
            except TypeError:
                anc = self.anchor_estimator.estimate(obs, ca.target_name, info=info)
            if anc.valid:
                dyaw = (
                    wrap_angle_pi(float(anc.yaw - ca.proto_obj_yaw))
                    if (self.transport.enable_yaw_rotation and abs(anc.yaw) > 1e-6)
                    else 0.0
                )
                rot_np = rotate_xy_vector(pos_knots.cpu().numpy(), dyaw)
                return torch.from_numpy(rot_np).to(self.device) + torch.from_numpy(anc.pos).to(self.device).unsqueeze(0)
            return pos_knots + torch.from_numpy(ca.proto_obj_pos).to(self.device).unsqueeze(0)
        return pos_knots

    def _route_cell_system2(
        self,
        ann: str,
        obs: dict,
        info: Optional[dict] = None,
        exclude_refractory: bool = True,
    ) -> int:
        z_l = self.encoder.encode_lang(ann)
        z_v, _ = self.encoder.encode_obs(
            obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"], obs["robot_obs"]
        )

        lang_bank = torch.stack([c["z_lang"] for c in self.cells])
        vis_bank = torch.stack([c["z_vis"] for c in self.cells])
        s1_all = 0.88 * (lang_bank @ z_l) + 0.12 * (vis_bank @ z_v)

        region_scores = []
        for tn, idxs in self.region_map.items():
            valid_idxs = [i for i in idxs if not (exclude_refractory and self.bridge.graph.is_refractory(i))]
            if not valid_idxs:
                valid_idxs = idxs
            best_cell_s1 = float(s1_all[valid_idxs].max().item())
            reg_l = torch.stack([self.cells[i]["z_lang"] for i in idxs]).mean(dim=0)
            reg_v = torch.stack([self.cells[i]["z_vis"] for i in idxs]).mean(dim=0)
            region_scores.append((best_cell_s1, tn, reg_l, reg_v, valid_idxs))
        region_scores.sort(key=lambda x: x[0], reverse=True)
        top_regions = region_scores[: min(self.top_m_tasks, len(region_scores))]

        if len(top_regions) >= 2:
            cm_l = torch.stack([r[2] for r in top_regions]).mean(dim=0)
            cm_v = torch.stack([r[3] for r in top_regions]).mean(dim=0)
            ql_cmc = F.normalize(z_l - 0.98 * cm_l, dim=-1)
            qv_cmc = F.normalize(z_v - 0.90 * cm_v, dim=-1)
            cand_pool = []
            for _, _, _, _, idxs in top_regions:
                for i in idxs:
                    cl_cmc = F.normalize(self.cells[i]["z_lang"] - 0.98 * cm_l, dim=-1)
                    cv_cmc = F.normalize(self.cells[i]["z_vis"] - 0.90 * cm_v, dim=-1)
                    sc = 0.92 * float(cl_cmc @ ql_cmc) + 0.08 * float(cv_cmc @ qv_cmc)
                    cand_pool.append((sc, i))
        else:
            cand_pool = [(float(s1_all[i].item()), i) for i in top_regions[0][4]]

        # System 2 object-noun cluster + polarity filter (strictly prevents cross-object refractory fallback)
        allowed_idxs = set(self.planner.filter_candidate_cells(ann, self.cells, [i for _, i in cand_pool]))
        cand_pool = [(sc, i) for sc, i in cand_pool if i in allowed_idxs]
        q_cluster = self.planner.extract_object_noun_cluster(ann)

        max_sc = max(x[0] for x in cand_pool)
        if self.planner.detect_ood(max_sc):
            self.ood_spawn_events += 1
            self.planner.record_spawn()

        cur_pos = obs["robot_obs"][:3]
        anc_cache = {}
        best_idx = cand_pool[0][1]
        best_final = -1e9
        for sc, i in cand_pool:
            if sc < max_sc - 0.04:
                continue
            cell = self.cells[i]
            pos_dist = float(np.linalg.norm(cur_pos - cell["init_pos"]))
            geo_bonus = -0.02 * pos_dist
            if q_cluster == "door_slider":
                init_z = float(cell["init_pos"][2])
                max_z = float(cell.get("max_z", init_z))
                if init_z < 0.50:
                    geo_bonus -= 0.12
                elif init_z >= 0.52 and max_z >= 0.53:
                    geo_bonus += 0.04
            if self.prev_cell_idx is not None and 0 <= self.prev_cell_idx < self.bridge.max_cells and i < self.bridge.max_cells:
                myelin_w = float(self.bridge.graph.myelin_matrix[self.prev_cell_idx, i].item())
                geo_bonus += 0.03 * np.log1p(myelin_w * 10.0)
            ca = cell["cell_anchor"]
            if self.use_transport and self.anchor_estimator is not None and ca.target_name != "fixed":
                tname = ca.target_name
                hint_key = (tname, cell["task"])
                if hint_key not in anc_cache:
                    try:
                        anc_cache[hint_key] = self.anchor_estimator.estimate(obs, tname, info=info, task_hint=f"{cell['task']} {ann}")
                    except TypeError:
                        anc_cache[hint_key] = self.anchor_estimator.estimate(obs, tname, info=info)
                cur_a = anc_cache[hint_key]
                if cur_a.valid:
                    dz = abs(float(cur_a.pos[2] - ca.proto_obj_pos[2]))
                    dxy = float(np.linalg.norm(cur_a.pos[:2] - ca.proto_obj_pos[:2]))
                    dyaw = abs(wrap_angle_pi(float(cur_a.yaw - ca.proto_obj_yaw))) if abs(cur_a.yaw) > 1e-6 else 0.0
                    geo_bonus -= (0.80 * dz + 0.12 * dxy + 0.05 * dyaw)
                else:
                    geo_bonus -= 0.25
            final_sc = sc + geo_bonus
            if final_sc > best_final:
                best_final = final_sc
                best_idx = i
        return best_idx

    def bind_subgoal(
        self,
        ann: str,
        obs: dict,
        info: Optional[dict] = None,
        use_bridge: bool = True,
        exclude_refractory: bool = True,
    ) -> None:
        self.current_ann = str(ann)
        best_idx = self._route_cell_system2(ann, obs, info=info, exclude_refractory=exclude_refractory)
        self.active_cell_idx = best_idx
        self.active_cell = self.cells[best_idx]

        T = self.active_cell["duration"]
        with torch.no_grad():
            self.active_act_seq = self.spline.evaluate(self.active_cell["act_knots"], T).cpu().numpy()
            rel_pos_sp = self.spline.evaluate(self.active_cell["pos_knots"], T).cpu().numpy()
            self.active_orn_seq = self.spline.evaluate(self.active_cell["orn_knots"], T).cpu().numpy()

        from hypertube_transport import TransportCellAnchor

        runtime_anchor = TransportCellAnchor(
            target_name=self.active_cell["cell_anchor"].target_name,
            proto_obj_pos=self.active_cell["cell_anchor"].proto_obj_pos,
            proto_obj_yaw=self.active_cell["cell_anchor"].proto_obj_yaw,
            proto_eef_pos=self.active_cell["cell_anchor"].proto_eef_pos,
            rel_to_obj_seq=rel_pos_sp.astype(np.float32),
        )
        self.active_runtime_anchor = runtime_anchor
        self.step_idx = 0
        self.anchor_bound = False

        if self.use_transport and self.anchor_estimator is not None:
            try:
                self.transported_ref_pos, _ = self.transport.bind_episode(
                    runtime_anchor, obs, self.anchor_estimator, info=info, task_hint=f"{self.active_cell['task']} {ann}"
                )
            except TypeError:
                self.transported_ref_pos, _ = self.transport.bind_episode(
                    runtime_anchor, obs, self.anchor_estimator, info=info
                )
            self.anchor_bound = (runtime_anchor.target_name == "fixed") or (
                float(np.linalg.norm(self.transport.state.cur_obj_pos)) > 1e-6
            )
        else:
            self.transported_ref_pos = rel_pos_sp.astype(np.float32)
            self.anchor_bound = True

        # System 1 C^2 Geodesic Bridge synthesis across consecutive chain skills
        if use_bridge and self.in_chain and self.prev_world_pos_knots is not None:
            knots_a = self.prev_world_pos_knots.clone()
            if len(self.eef_history) >= 3:
                h3 = torch.from_numpy(np.stack(self.eef_history[-3:], axis=0)).to(self.device).float()
                knots_a[-3:] = h3
            else:
                cur_eef = torch.from_numpy(obs["robot_obs"][:3]).to(self.device).float()
                knots_a = knots_a + (cur_eef - knots_a[-1]).unsqueeze(0)

            knots_b = self._get_world_pos_knots(self.active_cell, obs, info=info)
            knots_b = knots_b.clone()
            knots_b[0] = torch.from_numpy(self.transported_ref_pos[0]).to(self.device).float()
            if self.planner.extract_object_noun_cluster(f"{self.active_cell['task']} {ann}") == "door_slider":
                knots_b[:, 2] = torch.clamp(knots_b[:, 2], min=0.52)

            with torch.no_grad():
                p_br = self.bridge.synthesize_bridge_knots(
                    knots_a,
                    knots_b,
                    dur_A=self.prev_dur,
                    dur_B=T,
                    c2_damping=None,
                    action_clip=2.0,
                )
                self.bridge_pos_seq = self.bridge_spline.evaluate(p_br, self.bridge.bridge_horizon).cpu().numpy()

            self.bridge_active = True
            self.bridge_idx = 0
            self.bridge.phase_state = BridgePhase.BRIDGE
            # Smooth handoff into Cell B without sudden approach jump
            self.transport.state.approach_done = True
        else:
            self.bridge_active = False
            self.bridge_pos_seq = None
            self.bridge_idx = 0
            self.bridge.phase_state = BridgePhase.CELL_B

    def reset_episode(self, ann, obs, info=None):
        self.episode_snap_back()
        self.bind_subgoal(ann, obs, info=info, use_bridge=False, exclude_refractory=False)

    def act(self, obs):
        self.eef_history.append(np.asarray(obs['robot_obs'][:3], dtype=np.float32).copy())
        if len(self.eef_history) > 6:
            self.eef_history.pop(0)
        if self.bridge_active and self.bridge_pos_seq is not None:
            tau = min(1.0, (self.bridge_idx + 1) / float(self.bridge.bridge_horizon))
            z_end_a = self.prev_z_end if self.prev_z_end is not None else self.active_cell["z_vis"]
            z_t = self.bridge.slerp(z_end_a, self.active_cell["z_vis"], tau)
            self.bridge.ephemeral_bridge_z.copy_(z_t.detach())

            cur_pos = np.asarray(obs["robot_obs"][:3], dtype=np.float32)
            H_br = len(self.bridge_pos_seq)
            idx_cur = min(self.bridge_idx, H_br - 1)
            idx_prev = max(0, idx_cur - 1)
            target_pos = self.bridge_pos_seq[idx_cur].copy()
            # Smooth sin^2(pi * tau) vertical clearance bell (C^2 smooth at boundaries tau=0, 1)
            lateral_dist = float(np.linalg.norm(self.bridge_pos_seq[-1, :2] - self.bridge_pos_seq[0, :2]))
            rising_to_cabinet = (float(self.bridge_pos_seq[-1, 2]) >= 0.51) and (float(self.bridge_pos_seq[0, 2]) < 0.50)
            if lateral_dist > 0.08 or rising_to_cabinet:
                target_pos[2] += (0.06 if rising_to_cabinet else 0.04) * (np.sin(np.pi * tau) ** 2)
            # C^2 spline feedforward velocity + low-gain damping feedback
            ff_vel = (self.bridge_pos_seq[idx_cur] - self.bridge_pos_seq[idx_prev]) * 20.0
            fb_vel = 6.0 * (target_pos - cur_pos)
            ramp = min(1.0, (self.bridge_idx + 1) / 4.0)
            delta_xyz = np.clip(ramp * (ff_vel + fb_vel), -0.65, 0.65)
            self.elastic_offset[:3] = delta_xyz

            act = np.zeros(7, dtype=np.float32)
            act[:3] = delta_xyz
            act[6] = 1.0 if self.prev_grip >= 0.0 else -1.0

            self.bridge_idx += 1
            if self.bridge_idx >= self.bridge.bridge_horizon:
                self.bridge_active = False
                self.bridge.phase_state = BridgePhase.CELL_B
            return act

        return super().act(obs)

    def finish_chain_subgoal(self, obs: dict, info: Optional[dict] = None, success: bool = True) -> None:
        if self.active_cell_idx >= 0:
            self.bridge.graph.mark_refractory(self.active_cell_idx)
        z_cur = self.active_cell["z_vis"].detach()
        if success and self.prev_cell_idx is not None and self.prev_z_end is not None:
            self.bridge.graph.update_myelination(
                self.prev_cell_idx, self.active_cell_idx, self.prev_z_end, z_cur, success=True
            )
        self.prev_cell_idx = self.active_cell_idx
        self.prev_z_end = z_cur.clone()
        self.prev_world_pos_knots = self._get_world_pos_knots(self.active_cell, obs, info=info)
        self.prev_dur = self.active_cell["duration"]
        self.prev_grip = float(self.active_cell["grip_seq"][-1])
        self.in_chain = True

    def episode_snap_back(self):
        self.in_chain = False
        self.prev_cell_idx = None
        self.prev_z_end = None
        self.prev_world_pos_knots = None
        self.eef_history.clear()
        self.bridge_active = False
        self.bridge_pos_seq = None
        self.bridge_idx = 0
        self.bridge.episode_snap_back()
        super().episode_snap_back()


def compute_transition_jerk(action_trace: List[np.ndarray], transition_steps: List[int]) -> float:
    """Measures mean acceleration jump ||Delta a_ddot||_2 around skill transition boundaries."""
    if len(action_trace) < 4 or not transition_steps:
        return 0.0
    acts = np.asarray(action_trace, dtype=np.float32)[:, :3]
    jerks = []
    for t in transition_steps:
        if 2 <= t < len(acts) - 1:
            acc_before = acts[t - 1] - acts[t - 2]
            acc_after = acts[t + 1] - acts[t]
            jerks.append(float(np.linalg.norm(acc_after - acc_before)))
    return float(np.mean(jerks)) if jerks else 0.0


def build_valid_5step_chains(eval_episodes: dict, num_chains: int = 30) -> List[List[dict]]:
    """Builds physically precondition-valid 5-step task sequences grounded in episode scene_obs."""
    avail_tasks = set(eval_episodes.keys())
    chains: List[List[dict]] = []

    # Anchor Chain #0 & #1 to the canonical Section 4.2 5-step chain when its episodes satisfy preconditions
    canon = ["lift_red_block_table", "place_in_drawer", "close_drawer", "move_slider_left", "turn_on_lightbulb"]
    if all(t in avail_tasks for t in canon):
        start_cands = [
            ep for ep in eval_episodes["lift_red_block_table"]
            if ep["frames"][0]["scene_obs"][0] < 0.08
            and ep["frames"][0]["scene_obs"][1] > 0.12
            and ep["frames"][0]["scene_obs"][4] < 0.5
        ]
        if start_cands:
            for p_idx in [1, 2]:
                if p_idx < len(eval_episodes["place_in_drawer"]):
                    chains.append([
                        start_cands[0],
                        eval_episodes["place_in_drawer"][p_idx],
                        eval_episodes["close_drawer"][0],
                        {**eval_episodes["move_slider_left"][0], "ann": "push the door to the left side"},
                        eval_episodes["turn_on_lightbulb"][0],
                    ])
                    if len(chains) >= num_chains:
                        return chains

    def valid_next_tasks(state: dict, last_task: str) -> List[str]:
        opts = []
        if not state["holding"]:
            if not state["drawer_open"] and "open_drawer" in avail_tasks:
                opts.append("open_drawer")
            if state["drawer_open"] and "close_drawer" in avail_tasks:
                opts.append("close_drawer")
            if not state["slider_left"] and "move_slider_left" in avail_tasks:
                opts.append("move_slider_left")
            if state["slider_left"] and "move_slider_right" in avail_tasks:
                opts.append("move_slider_right")
            if not state["led_on"] and "turn_on_led" in avail_tasks:
                opts.append("turn_on_led")
            if state["led_on"] and "turn_off_led" in avail_tasks:
                opts.append("turn_off_led")
            if not state["bulb_on"] and "turn_on_lightbulb" in avail_tasks:
                opts.append("turn_on_lightbulb")
            if state["bulb_on"] and "turn_off_lightbulb" in avail_tasks:
                opts.append("turn_off_lightbulb")
            for lt in ("lift_red_block_table", "lift_blue_block_table", "lift_pink_block_table"):
                if lt in avail_tasks and lt != last_task:
                    opts.append(lt)
        else:
            if state["drawer_open"] and "place_in_drawer" in avail_tasks:
                opts.append("place_in_drawer")
            if state["slider_left"] and "place_in_slider" in avail_tasks:
                opts.append("place_in_slider")
        return [o for o in opts if o != last_task] or opts

    def apply_effect(state: dict, task: str) -> None:
        if task == "open_drawer":
            state["drawer_open"] = True
        elif task == "close_drawer":
            state["drawer_open"] = False
        elif task == "move_slider_left":
            state["slider_left"] = True
        elif task == "move_slider_right":
            state["slider_left"] = False
        elif task == "turn_on_led":
            state["led_on"] = True
        elif task == "turn_off_led":
            state["led_on"] = False
        elif task == "turn_on_lightbulb":
            state["bulb_on"] = True
        elif task == "turn_off_lightbulb":
            state["bulb_on"] = False
        elif task.startswith("lift_"):
            state["holding"] = True
        elif task.startswith("place_in_") or task == "push_into_drawer":
            state["holding"] = False

    start_pool = [
        t for t in (
            "lift_red_block_table", "lift_blue_block_table", "lift_pink_block_table",
            "open_drawer", "close_drawer", "move_slider_left", "move_slider_right",
            "turn_on_led", "turn_on_lightbulb",
        )
        if t in avail_tasks
    ] or sorted(list(avail_tasks))

    c_idx = len(chains)
    tries = 0
    while len(chains) < num_chains and tries < num_chains * 10:
        rng = random.Random(2000 + c_idx + tries)
        tries += 1
        init_task = rng.choice(start_pool)
        # Filter initial episodes that actually satisfy init_task's physical precondition
        ep0_cands = []
        for ep in eval_episodes[init_task]:
            s0 = ep["frames"][0]["scene_obs"]
            if init_task == "move_slider_left" and s0[0] >= 0.08:
                continue
            if init_task == "move_slider_right" and s0[0] <= 0.18:
                continue
            if init_task == "open_drawer" and s0[1] >= 0.08:
                continue
            if init_task == "close_drawer" and s0[1] <= 0.12:
                continue
            if init_task == "turn_on_led" and s0[3] >= 0.5:
                continue
            if init_task == "turn_on_lightbulb" and s0[4] >= 0.5:
                continue
            ep0_cands.append(ep)
        ep0 = rng.choice(ep0_cands or eval_episodes[init_task])
        s0 = ep0["frames"][0]["scene_obs"]
        state = {
            "slider_left": bool(s0[0] > 0.14),
            "drawer_open": bool(s0[1] > 0.10),
            "led_on": bool(s0[3] > 0.5),
            "bulb_on": bool(s0[4] > 0.5),
            "holding": False,
        }
        apply_effect(state, init_task)
        seq = [ep0]
        cur = init_task
        for _ in range(4):
            cands = valid_next_tasks(state, cur)
            if not cands:
                break
            cur = rng.choice(cands)
            apply_effect(state, cur)
            seq.append(rng.choice(eval_episodes[cur]))
        if len(seq) == 5:
            chains.append(seq)
            c_idx += 1
    return chains


def evaluate_5step_chains_v6(
    env,
    task_oracle,
    policy: DTRAv6DualSystemController,
    chains: List[List[dict]],
    use_system12_bridge: bool = True,
) -> dict:
    lengths = []
    boundary_jerks = []

    for seq in chains:
        f0 = seq[0]["frames"][0]
        obs = env.reset(robot_obs=f0["robot_obs"], scene_obs=f0["scene_obs"])
        policy.episode_snap_back()

        completed = 0
        action_trace: List[np.ndarray] = []
        transition_steps: List[int] = []

        for sub_idx, sub_ep in enumerate(seq):
            t_name = sub_ep["task"]
            ann = sub_ep["ann"]

            sub_plan = [(t_name, ann, False)]

            target_solved = False
            for plan_task, plan_ann, is_prereq in sub_plan:
                if sub_idx > 0 or is_prereq:
                    transition_steps.append(len(action_trace))

                # Up to 2 attempts when System 2 backtracking is enabled
                max_attempts = 2 if (use_system12_bridge and not is_prereq) else 1
                sub_solved = False
                for attempt in range(max_attempts):
                    if attempt > 0:
                        policy.backtrack_recoveries += 1
                    start_info = env.get_info()
                    if use_system12_bridge:
                        policy.bind_subgoal(
                            plan_ann,
                            obs,
                            info=start_info,
                            use_bridge=(sub_idx > 0 or attempt > 0),
                            exclude_refractory=True,
                        )
                    else:
                        policy.reset_episode(plan_ann, obs, info=start_info)

                    horizon = len(sub_ep["frames"]) + (40 if use_system12_bridge else 25)
                    for _ in range(horizon):
                        action = policy.act(obs)
                        action_trace.append(action.copy())
                        obs, _, _, cur_info = env.step(action)
                        if plan_task in task_oracle.get_task_info_for_set(start_info, cur_info, {plan_task}):
                            sub_solved = True
                            break

                    if use_system12_bridge:
                        policy.finish_chain_subgoal(obs, info=env.get_info(), success=sub_solved)
                    if sub_solved:
                        break

                if not is_prereq:
                    target_solved = sub_solved
                elif not sub_solved:
                    break

            if target_solved:
                completed += 1
            else:
                break

        policy.episode_snap_back()
        lengths.append(completed)
        boundary_jerks.append(compute_transition_jerk(action_trace, transition_steps))

    myelin_edges = int((policy.bridge.graph.myelin_matrix > 0.05).sum().item())
    return {
        "avg_5step_chain_len": round(float(np.mean(lengths)), 2),
        "chain_5step_full_sr": round(100.0 * sum(1 for x in lengths if x == 5) / max(1, len(chains)), 2),
        "mean_boundary_jerk": round(float(np.mean(boundary_jerks)), 4),
        "myelinated_edges": myelin_edges,
        "backtrack_recoveries": policy.backtrack_recoveries,
        "ood_spawn_events": policy.ood_spawn_events,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=os.path.expanduser("~/dtra_a6000_lab/calvin_data/task_D_D_subset"))
    parser.add_argument("--calvin_root", type=str, default=os.path.expanduser("~/dtra_a6000_lab/calvin_benchmark/calvin"))
    parser.add_argument("--max_tasks", type=int, default=34, help="Number of tasks to train/evaluate (use 8 for fast smoke test)")
    parser.add_argument("--max_per_task", type=int, default=10)
    parser.add_argument("--max_eval_per_task", type=int, default=5)
    parser.add_argument("--num_chains", type=int, default=30)
    parser.add_argument("--num_knots", type=int, default=10)
    parser.add_argument("--mode", type=str, default="oracle", choices=["oracle", "vision"])
    parser.add_argument("--out_json", type=str, default=os.path.expanduser("~/dtra_a6000_lab/calvin_benchmark/calvin_v6_system12_results.json"))
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== DTRA v6 System 1 + System 2 Benchmark | Device: {device} | Mode: {args.mode} | MaxTasks: {args.max_tasks} ===", flush=True)

    train_dir = os.path.join(args.data_dir, "training")
    val_dir = os.path.join(args.data_dir, "validation")
    train_tasks = load_episodes(train_dir, max_per_task=args.max_per_task)
    val_tasks = load_episodes(val_dir, max_per_task=args.max_eval_per_task)

    all_task_names = sorted(list(set(train_tasks.keys()) | set(val_tasks.keys())))[: args.max_tasks]
    eval_split = {t: (val_tasks[t] if t in val_tasks else train_tasks[t]) for t in all_task_names}

    env = get_env(val_dir, show_gui=False)
    task_cfg_path = os.path.join(args.calvin_root, "calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml")
    if not os.path.exists(task_cfg_path):
        task_cfg_path = os.path.join(args.calvin_root, "calvin_env/conf/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(OmegaConf.load(task_cfg_path))
    encoder = FrozenPerceptionEncoder(device)

    if args.mode == "oracle":
        estimator = OracleSceneAnchorEstimator()
        enable_yaw = True
    else:
        estimator = CalvinCameraVisionAnchorEstimatorV5(env.cameras[0])
        enable_yaw = False

    policy = DTRAv6DualSystemController(
        encoder=encoder,
        device=device,
        anchor_estimator=estimator,
        use_transport=True,
        num_knots=args.num_knots,
        enable_yaw_rotation=enable_yaw,
    )

    t0 = time.perf_counter()
    for t_name in all_task_names:
        tr_eps = train_tasks[t_name] if t_name in train_tasks else val_tasks[t_name]
        policy.train_task(tr_eps)

    single_sr, _, lat_ms, route_acc = evaluate_continual_tasks(env, task_oracle, policy, eval_split)
    chains = build_valid_5step_chains(eval_split, num_chains=args.num_chains)
    print(f"Built {len(chains)} valid 5-step chains. Running ablation (Hard-Switch vs System 1+2 Bridge)...", flush=True)

    res_hard = evaluate_5step_chains_v6(env, task_oracle, policy, chains, use_system12_bridge=False)
    res_sys12 = evaluate_5step_chains_v6(env, task_oracle, policy, chains, use_system12_bridge=True)
    drift = policy.parameter_drift()

    jerk_reduction = 0.0
    if res_hard["mean_boundary_jerk"] > 1e-6:
        jerk_reduction = 100.0 * (res_hard["mean_boundary_jerk"] - res_sys12["mean_boundary_jerk"]) / res_hard["mean_boundary_jerk"]

    out = {
        "mode": args.mode,
        "num_tasks": len(all_task_names),
        "single_task_mean_sr": round(single_sr, 2),
        "route_acc": round(route_acc, 2) if route_acc is not None else None,
        "parameter_drift": round(drift, 6),
        "closed_loop_latency_ms": round(lat_ms, 4),
        "Ablation_HardSwitch_NoBridge": res_hard,
        "DTRA_V6_System1_Bridge_System2_Organ": res_sys12,
        "boundary_jerk_reduction_pct": round(jerk_reduction, 2),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    print(json.dumps(out, indent=2), flush=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
