from enum import IntEnum
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgePhase(IntEnum):
    IDLE = 0
    CELL_A = 1
    BRIDGE = 2
    CELL_B = 3
    DONE = 4


class MyelinatedTransitionGraph(nn.Module):
    """
    Non-parametric structural myelination matrix M_{ij} and refractory call-stack.
    Operates strictly outside gradient-updated weights (L_inf weight drift == 0.0).
    """

    def __init__(
        self,
        num_cells: int,
        d_topo: int = 64,
        decay: float = 0.01,
        lr_myelin: float = 0.25,
        tau_sphere: float = 0.5,
        refractory_threshold: float = 0.5,
    ):
        super().__init__()
        self.num_cells = num_cells
        self.d_topo = d_topo
        self.decay = decay
        self.lr_myelin = lr_myelin
        self.tau_sphere = tau_sphere
        self.refractory_threshold = refractory_threshold

        # Structural transition matrix M_{ij} (never registered as nn.Parameter)
        self.register_buffer("myelin_matrix", torch.full((num_cells, num_cells), 1e-3))
        # Canonical entry proto-knots per frozen cell: (K, 3, D_act)
        self.register_buffer("entry_proto_knots", torch.zeros(num_cells, 3, 1))
        # Ephemeral episodic working memory (reset to exact 0.0 on episode_snap_back)
        self.register_buffer("episodic_trace", torch.zeros(num_cells))

    def init_proto_knots(self, action_dim: int) -> None:
        self.entry_proto_knots = torch.zeros(
            self.num_cells, 3, action_dim, device=self.myelin_matrix.device
        )

    @torch.no_grad()
    def register_cell_entry_anchor(self, cell_id: int, head_knots: torch.Tensor) -> None:
        """Caches the canonical first 3 B-Spline knots (p0, p1, p2) of a frozen cell."""
        if self.entry_proto_knots.shape[-1] != head_knots.shape[-1]:
            self.init_proto_knots(head_knots.shape[-1])
        self.entry_proto_knots[cell_id].copy_(head_knots[..., :3, :].detach())

    @torch.no_grad()
    def update_myelination(
        self,
        cell_i: int,
        cell_j: int,
        z_end_i: torch.Tensor,
        z_start_j: torch.Tensor,
        success: bool,
    ) -> None:
        """Hebbian structural myelination update gated by task completion."""
        if not success or cell_i == cell_j:
            return
        u = F.normalize(z_end_i.detach(), dim=-1)
        v = F.normalize(z_start_j.detach(), dim=-1)
        geodesic_dist = torch.acos(torch.clamp((u * v).sum(dim=-1), -1.0 + 1e-6, 1.0 - 1e-6))
        affinity = torch.exp(-geodesic_dist / self.tau_sphere).mean().item()

        self.myelin_matrix[cell_i, cell_j] = (
            (1.0 - self.decay) * self.myelin_matrix[cell_i, cell_j]
            + self.lr_myelin * affinity
        )

    @torch.no_grad()
    def select_next_cell(
        self,
        current_cell: int,
        cell_centers: torch.Tensor,
        goal_embedding: Optional[torch.Tensor] = None,
        goal_beta: float = 1.0,
    ) -> int:
        """Selects next cell j* outside the refractory mask R_t."""
        scores = torch.log(self.myelin_matrix[current_cell] + 1e-8).clone()
        if goal_embedding is not None:
            centers = F.normalize(cell_centers, dim=-1)
            goal = F.normalize(goal_embedding, dim=-1)
            scores = scores + goal_beta * torch.matmul(centers, goal)

        # Apply refractory mask (blocks perseveration / re-firing completed cells)
        refractory_mask = self.episodic_trace > self.refractory_threshold
        refractory_mask[current_cell] = True
        scores = scores.masked_fill(refractory_mask, float("-inf"))
        return int(torch.argmax(scores).item())

    @torch.no_grad()
    def mark_refractory(self, cell_id: int) -> None:
        self.episodic_trace[cell_id] = 1.0

    @torch.no_grad()
    def reset_trace(self) -> float:
        self.episodic_trace.zero_()
        return float(self.episodic_trace.abs().max().item())


class TopologicalGeodesicBridge(nn.Module):
    """
    DTRA Stage-2 Compositional Module (Zero-Parameter-Drift Skill Chaining).
    Connects frozen S^{d-1} hypertube cells via:
      1) Hyperspherical SLERP geodesic state transport
      2) Closed-form C^2 / adaptive-damped-C^1 clamped cubic B-Spline knot stitching (N_k=6)
      3) Proprioceptive event trigger + myelinated refractory FSM + 0.0000 snap-back
    """

    def __init__(
        self,
        num_cells: int,
        d_topo: int = 64,
        n_knots: int = 6,
        horizon: int = 16,
        damping_alpha: float = 1.5,
        phase_trigger_thresh: float = 0.95,
        gripper_delta_thresh: float = 0.15,
    ):
        super().__init__()
        if n_knots != 6:
            raise ValueError("Closed-form C^2 stitch matrix requires n_knots == 6.")
        self.num_cells = num_cells
        self.d_topo = d_topo
        self.n_knots = n_knots
        self.horizon = horizon
        self.damping_alpha = damping_alpha
        self.phase_trigger_thresh = phase_trigger_thresh
        self.gripper_delta_thresh = gripper_delta_thresh

        c2_stitch_matrix = torch.tensor(
            [
                [0.0,  0.0, 1.0, 0.0,  0.0, 0.0],  # p0 = pA_5 (C^0 start)
                [0.0, -1.0, 2.0, 0.0,  0.0, 0.0],  # p1 = 2*pA_5 - pA_4 (C^1 start)
                [1.0, -6.0, 6.0, 0.0,  0.0, 0.0],  # p2 = pA_3 - 6*pA_4 + 6*pA_5 (C^2 start)
                [0.0,  0.0, 0.0, 6.0, -6.0, 1.0],  # p3 = 6*pB_0 - 6*pB_1 + pB_2 (C^2 end)
                [0.0,  0.0, 0.0, 2.0, -1.0, 0.0],  # p4 = 2*pB_0 - pB_1 (C^1 end)
                [0.0,  0.0, 0.0, 1.0,  0.0, 0.0],  # p5 = pB_0 (C^0 end)
            ],
            dtype=torch.float32,
        )
        self.register_buffer("c2_stitch_matrix", c2_stitch_matrix)
        self.graph = MyelinatedTransitionGraph(num_cells=num_cells, d_topo=d_topo)

        # Ephemeral state buffers (strictly zeroed on episode_snap_back)
        self.register_buffer("ephemeral_bridge_z", torch.zeros(d_topo))
        self.register_buffer("bridge_step", torch.tensor(0, dtype=torch.long))
        self.phase_state: BridgePhase = BridgePhase.IDLE
        self.active_cell_a: Optional[int] = None
        self.active_cell_b: Optional[int] = None

    @staticmethod
    def slerp_s63(u: torch.Tensor, v: torch.Tensor, tau: float, eps: float = 1e-6) -> torch.Tensor:
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

    def compute_adaptive_damping(self, p_end_a: torch.Tensor, p_start_b: torch.Tensor) -> torch.Tensor:
        """
        Returns lambda_damp in (0, 1]:
          - ~1.0 when endpoints are close (exact C^2 acceleration continuity)
          - -> 0.0 when endpoints are far apart (minimum-jerk C^1 chord to prevent overshoot)
        """
        dist_sq = (p_end_a - p_start_b).pow(2).sum(dim=-1, keepdim=True).unsqueeze(-2)
        return torch.exp(-self.damping_alpha * dist_sq)

    def synthesize_bridge_knots(
        self,
        knots_A: torch.Tensor,
        knots_B: Optional[torch.Tensor] = None,
        target_cell_b: Optional[int] = None,
        c2_damping: Optional[float] = None,
        action_clip: float = 1.0,
    ) -> torch.Tensor:
        """
        Synthesizes 6 bridge control knots from Cell A tail (p3..p5) and Cell B head (p0..p2).
        Falls back to canonical entry proto-knots of target_cell_b if knots_B is not yet observed.
        """
        tail_a = knots_A[..., 3:6, :]
        if knots_B is not None:
            head_b = knots_B[..., 0:3, :]
        elif target_cell_b is not None:
            head_b = self.graph.entry_proto_knots[target_cell_b].expand_as(tail_a)
        else:
            raise ValueError("Either knots_B or target_cell_b must be provided.")

        boundary = torch.cat([tail_a, head_b], dim=-2)
        stitch_mat = self.c2_stitch_matrix.to(dtype=boundary.dtype, device=boundary.device)
        p_c2 = torch.matmul(stitch_mat, boundary)

        if c2_damping is None:
            lam = self.compute_adaptive_damping(tail_a[..., 2, :], head_b[..., 0, :])
        else:
            lam = torch.as_tensor(c2_damping, dtype=boundary.dtype, device=boundary.device)

        p0, p1, p4, p5 = p_c2[..., 0:1, :], p_c2[..., 1:2, :], p_c2[..., 4:5, :], p_c2[..., 5:6, :]
        p2_c1 = (2.0 / 3.0) * p1 + (1.0 / 3.0) * p4
        p3_c1 = (1.0 / 3.0) * p1 + (2.0 / 3.0) * p4
        p2 = lam * p_c2[..., 2:3, :] + (1.0 - lam) * p2_c1
        p3 = lam * p_c2[..., 3:4, :] + (1.0 - lam) * p3_c1
        p_out = torch.cat([p0, p1, p2, p3, p4, p5], dim=-2)

        return torch.clamp(p_out, -action_clip, action_clip)

    def check_transition_trigger(
        self,
        phase_progress: float,
        gripper_delta: float = 0.0,
        resonance_peak: bool = False,
    ) -> bool:
        """
        Fires transition from CELL_A -> BRIDGE only when Cell A is active,
        not in refractory state, and phase + proprioceptive/resonance condition holds.
        """
        if self.phase_state != BridgePhase.CELL_A or self.active_cell_a is None:
            return False
        if self.graph.episodic_trace[self.active_cell_a] > self.graph.refractory_threshold:
            return False
        proprio_event = (abs(gripper_delta) >= self.gripper_delta_thresh) or resonance_peak
        return (phase_progress >= self.phase_trigger_thresh) and proprio_event

    @torch.no_grad()
    def start_sequence(self, cell_a: int, cell_b: int) -> None:
        self.active_cell_a = cell_a
        self.active_cell_b = cell_b
        self.phase_state = BridgePhase.CELL_A
        self.bridge_step.zero_()

    @torch.no_grad()
    def step_fsm(
        self,
        z_end_A: torch.Tensor,
        z_start_B: torch.Tensor,
        phase_progress: float,
        gripper_delta: float = 0.0,
        resonance_peak: bool = False,
    ) -> Tuple[BridgePhase, torch.Tensor]:
        """Advances the myelinated call-stack FSM by one control step."""
        if self.phase_state == BridgePhase.CELL_A:
            if self.check_transition_trigger(phase_progress, gripper_delta, resonance_peak):
                self.graph.mark_refractory(self.active_cell_a)
                self.phase_state = BridgePhase.BRIDGE
                self.bridge_step.zero_()
            else:
                z_a = F.normalize(z_end_A, dim=-1)
                self.ephemeral_bridge_z.copy_(z_a.detach())
                return self.phase_state, z_a

        if self.phase_state == BridgePhase.BRIDGE:
            step_idx = int(self.bridge_step.item())
            tau = min(1.0, (step_idx + 1) / float(self.horizon))
            z_t = self.slerp_s63(z_end_A, z_start_B, tau)
            self.ephemeral_bridge_z.copy_(z_t.detach())
            self.bridge_step.add_(1)
            if int(self.bridge_step.item()) >= self.horizon:
                self.phase_state = BridgePhase.CELL_B
            return self.phase_state, z_t

        z_b = F.normalize(z_start_B, dim=-1)
        self.ephemeral_bridge_z.copy_(z_b.detach())
        return self.phase_state, z_b

    @torch.no_grad()
    def episode_snap_back(self) -> float:
        """Restores all ephemeral bridge and refractory buffers to exact 0.0000."""
        self.ephemeral_bridge_z.zero_()
        self.bridge_step.zero_()
        trace_residual = self.graph.reset_trace()
        self.phase_state = BridgePhase.IDLE
        self.active_cell_a = None
        self.active_cell_b = None
        return max(float(self.ephemeral_bridge_z.abs().max().item()), trace_residual)