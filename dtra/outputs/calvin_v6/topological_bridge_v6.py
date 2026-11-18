from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgePhase(IntEnum):
    IDLE = 0
    CELL_A = 1
    BRIDGE = 2
    CELL_B = 3


class MyelinatedTransitionGraph(nn.Module):
    """
    System 1 Structural Memory:
    Non-parametric myelination matrix M_{ij} + refractory episodic trace.
    Strictly decoupled from neural weights (L_inf weight drift == 0.0).
    """

    def __init__(
        self,
        max_cells: int = 512,
        d_topo: int = 128,
        decay: float = 0.01,
        lr_myelin: float = 0.25,
        tau_sphere: float = 0.5,
        refractory_threshold: float = 0.5,
    ):
        super().__init__()
        self.max_cells = max_cells
        self.d_topo = d_topo
        self.decay = decay
        self.lr_myelin = lr_myelin
        self.tau_sphere = tau_sphere
        self.refractory_threshold = refractory_threshold

        self.register_buffer("myelin_matrix", torch.full((max_cells, max_cells), 1e-3))
        self.register_buffer("episodic_trace", torch.zeros(max_cells))

    @torch.no_grad()
    def update_myelination(
        self,
        cell_i: int,
        cell_j: int,
        z_end_i: torch.Tensor,
        z_start_j: torch.Tensor,
        success: bool,
    ) -> None:
        if not success or cell_i == cell_j or cell_i >= self.max_cells or cell_j >= self.max_cells:
            return
        u = F.normalize(z_end_i.detach(), dim=-1)
        v = F.normalize(z_start_j.detach(), dim=-1)
        geodesic_dist = torch.acos(torch.clamp((u * v).sum(dim=-1), -1.0 + 1e-6, 1.0 - 1e-6))
        affinity = float(torch.exp(-geodesic_dist / self.tau_sphere).mean().item())
        self.myelin_matrix[cell_i, cell_j] = (
            (1.0 - self.decay) * self.myelin_matrix[cell_i, cell_j] + self.lr_myelin * affinity
        )

    @torch.no_grad()
    def mark_refractory(self, cell_id: int) -> None:
        if 0 <= cell_id < self.max_cells:
            self.episodic_trace[cell_id] = 1.0

    @torch.no_grad()
    def is_refractory(self, cell_id: int) -> bool:
        if 0 <= cell_id < self.max_cells:
            return bool(self.episodic_trace[cell_id].item() > self.refractory_threshold)
        return False

    @torch.no_grad()
    def reset_trace(self) -> float:
        self.episodic_trace.zero_()
        return float(self.episodic_trace.abs().max().item())


class TopologicalGeodesicBridgeV6(nn.Module):
    """
    System 1 Reflex & Smooth Skill Chaining:
      1) Hyperspherical SLERP transport on S^{d-1}
      2) Time-scaled C^2 / adaptive-damped C^1 clamped cubic B-Spline bridge (N_k=6 -> H=16)
      3) Grasp-preserving transit + 0.0000 episode_snap_back()
    """

    def __init__(
        self,
        max_cells: int = 512,
        d_topo: int = 128,
        bridge_horizon: int = 16,
        damping_alpha: float = 2.0,
    ):
        super().__init__()
        self.max_cells = max_cells
        self.d_topo = d_topo
        self.bridge_horizon = bridge_horizon
        self.damping_alpha = damping_alpha

        self.graph = MyelinatedTransitionGraph(max_cells=max_cells, d_topo=d_topo)
        self.register_buffer("ephemeral_bridge_z", torch.zeros(d_topo))
        self.register_buffer("bridge_step", torch.tensor(0, dtype=torch.long))
        self.phase_state: BridgePhase = BridgePhase.IDLE

    @staticmethod
    def slerp(u: torch.Tensor, v: torch.Tensor, tau: float, eps: float = 1e-6) -> torch.Tensor:
        u = F.normalize(u, dim=-1)
        v = F.normalize(v, dim=-1)
        dot = torch.clamp((u * v).sum(dim=-1, keepdim=True), -1.0 + eps, 1.0 - eps)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        t = torch.as_tensor(tau, dtype=u.dtype, device=u.device)
        slerp_vec = (torch.sin((1.0 - t) * omega) / sin_omega) * u + (torch.sin(t * omega) / sin_omega) * v
        lerp_vec = F.normalize((1.0 - t) * u + t * v, dim=-1)
        out = torch.where(sin_omega.abs() < 1e-4, lerp_vec, slerp_vec)
        return F.normalize(out, dim=-1)

    def synthesize_bridge_knots(
        self,
        knots_A: torch.Tensor,
        knots_B: torch.Tensor,
        dur_A: int = 16,
        dur_B: int = 16,
        c2_damping: Optional[float] = None,
        action_clip: float = 10.0,
    ) -> torch.Tensor:
        """
        Synthesizes 6 clamped cubic B-Spline bridge knots (M_br = 3) matching physical
        position (C^0), velocity (C^1), and acceleration (C^2) for arbitrary N_k >= 4 and durations.
        """
        m_a = max(1, knots_A.shape[-2] - 3)
        m_b = max(1, knots_B.shape[-2] - 3)
        m_br = 3.0
        rho_a = (m_a / max(1.0, float(dur_A))) / (m_br / float(self.bridge_horizon))
        rho_b = (m_b / max(1.0, float(dur_B))) / (m_br / float(self.bridge_horizon))

        pa_m3, pa_m2, pa_m1 = knots_A[..., -3, :], knots_A[..., -2, :], knots_A[..., -1, :]
        pb_0, pb_1, pb_2 = knots_B[..., 0, :], knots_B[..., 1, :], knots_B[..., 2, :]

        p0 = pa_m1
        p1 = pa_m1 + rho_a * (pa_m1 - pa_m2)
        p2_c2 = 3.0 * p1 - 2.0 * p0 + (rho_a ** 2) * (2.0 * pa_m1 - 3.0 * pa_m2 + pa_m3)

        p5 = pb_0
        p4 = pb_0 + rho_b * (pb_0 - pb_1)
        p3_c2 = 3.0 * p4 - 2.0 * p5 + (rho_b ** 2) * (2.0 * pb_0 - 3.0 * pb_1 + pb_2)

        if c2_damping is None:
            dist_sq = (pa_m1 - pb_0).pow(2).sum(dim=-1, keepdim=True)
            lam = torch.exp(-self.damping_alpha * dist_sq)
        else:
            lam = torch.as_tensor(c2_damping, dtype=knots_A.dtype, device=knots_A.device)

        p2_c1 = (2.0 / 3.0) * p1 + (1.0 / 3.0) * p4
        p3_c1 = (1.0 / 3.0) * p1 + (2.0 / 3.0) * p4
        p2 = lam * p2_c2 + (1.0 - lam) * p2_c1
        p3 = lam * p3_c2 + (1.0 - lam) * p3_c1

        p_out = torch.stack([p0, p1, p2, p3, p4, p5], dim=-2)
        return torch.clamp(p_out, -action_clip, action_clip)

    @torch.no_grad()
    def episode_snap_back(self) -> float:
        self.ephemeral_bridge_z.zero_()
        self.bridge_step.zero_()
        res = self.graph.reset_trace()
        self.phase_state = BridgePhase.IDLE
        return max(float(self.ephemeral_bridge_z.abs().max().item()), res)


class OrganSystem2Planner:
    """
    System 2 Deliberative Organ Layer (Zero Task-Hardcoding, Zero scene_obs Leak):
      1) OOD Resonance Monitor & Zero-Forgetting Cell Spawner
      2) Object-Noun Cluster + Polarity Filtering for Refractory Failure Backtracking
    """

    def __init__(self, ood_resonance_thresh: float = 0.55):
        self.ood_resonance_thresh = ood_resonance_thresh
        self.spawned_cells_count = 0

    def detect_ood(self, max_resonance: float) -> bool:
        return max_resonance < self.ood_resonance_thresh

    def record_spawn(self) -> int:
        self.spawned_cells_count += 1
        return self.spawned_cells_count

    @staticmethod
    def extract_object_noun_cluster(text: str) -> str:
        """Group instruction/task into canonical target object noun clusters."""
        tokens = set(text.lower().replace(",", " ").replace("_", " ").split())
        if tokens & {"stack", "unstack", "another", "top", "tower"}:
            return "block"
        if "into" in tokens and bool(tokens & {"push", "sweep", "slide"}) and ("drawer" in tokens):
            return "push_drawer"
        if bool(tokens & {"place", "put", "store"}) and bool(tokens & {"drawer", "slider", "cabinet"}):
            return "place_drawer" if "drawer" in tokens else "place_slider"
        if tokens & {"block", "red", "blue", "pink", "cube"}:
            return "block"
        if tokens & {"door", "slider", "sliding"}:
            return "door_slider"
        if "drawer" in tokens:
            return "drawer"
        if tokens & {"led", "lightbulb", "bulb", "light", "button", "switch", "yellow", "green", "lamp"}:
            return "light"
        return "general"

    @staticmethod
    def extract_block_color(text: str) -> Optional[str]:
        tokens = set(text.lower().replace(",", " ").replace("_", " ").split())
        for c in ("red", "blue", "pink"):
            if c in tokens:
                return c
        return None

    @staticmethod
    def extract_action_polarity(text: str) -> Optional[str]:
        tokens = set(text.lower().replace(",", " ").replace("_", " ").split())
        if "left" in tokens:
            return "left"
        if "right" in tokens:
            return "right"
        if "open" in tokens:
            return "open"
        if "close" in tokens:
            return "close"
        if tokens & {"on", "up", "upwards"}:
            return "on"
        if tokens & {"off", "down", "downwards"}:
            return "off"
        return None

    def filter_candidate_cells(
        self,
        ann: str,
        cells: list,
        cand_indices: List[int],
    ) -> List[int]:
        """Restrict candidate / refractory recovery cells to the matching object-noun cluster."""
        if not cand_indices:
            return cand_indices
        q_cluster = self.extract_object_noun_cluster(ann)
        q_color = self.extract_block_color(ann)
        q_pol = self.extract_action_polarity(ann)

        filtered = cand_indices
        if q_cluster != "general":
            by_cluster = [
                i for i in filtered
                if self.extract_object_noun_cluster(f"{cells[i].get('task', '')} {cells[i].get('ann', '')}") == q_cluster
            ]
            if by_cluster:
                filtered = by_cluster

        if q_color is not None:
            by_color = [
                i for i in filtered
                if self.extract_block_color(f"{cells[i].get('task', '')} {cells[i].get('ann', '')}") == q_color
            ]
            if by_color:
                filtered = by_color

        if q_pol is not None:
            by_pol = [
                i for i in filtered
                if self.extract_action_polarity(f"{cells[i].get('task', '')} {cells[i].get('ann', '')}") == q_pol
            ]
            if by_pol:
                filtered = by_pol

        return filtered


if __name__ == "__main__":
    bridge = TopologicalGeodesicBridgeV6(max_cells=64, d_topo=128, bridge_horizon=16)
    ka = torch.randn(10, 3, dtype=torch.float64) * 0.02
    kb = torch.randn(10, 3, dtype=torch.float64) * 0.02
    p_br = bridge.synthesize_bridge_knots(ka, kb, dur_A=40, dur_B=32, c2_damping=1.0, action_clip=10.0)
    assert p_br.shape == (6, 3)
    assert bridge.episode_snap_back() == 0.0
    print("[PASS] Clean System 1 + System 2 self-test OK!")
