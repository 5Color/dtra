"""dtra/core/hypertube_transport.py

Object-Centric Hypertube Transport module for spatial/SE(2) equivariant
execution of frozen B-Spline visuomotor primitives without weight drift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple
import numpy as np


DEFAULT_BLOCK_SLICES: Dict[str, Tuple[slice, slice]] = {
    "block_red": (slice(6, 9), slice(9, 12)),
    "block_blue": (slice(12, 15), slice(15, 18)),
    "block_pink": (slice(18, 21), slice(21, 24)),
}

# HSV/RGB heuristics tuned only for foreground cube color cues in CALVIN-style scenes
COLOR_RGB_PRIORS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "block_red": (np.array([0.85, 0.15, 0.15], dtype=np.float32), np.array([1.0, -0.5, -0.5], dtype=np.float32)),
    "block_blue": (np.array([0.15, 0.25, 0.85], dtype=np.float32), np.array([-0.5, -0.3, 1.0], dtype=np.float32)),
    "block_pink": (np.array([0.85, 0.35, 0.70], dtype=np.float32), np.array([0.7, -0.8, 0.6], dtype=np.float32)),
}


@dataclass
class ObjectAnchor:
    """Estimated 3D position and yaw orientation of a target scene object."""

    name: str
    pos: np.ndarray
    yaw: float = 0.0
    valid: bool = True


@dataclass
class TransportCellAnchor:
    """Frozen object-centric reference metadata stored per DTRA cell (~24 bytes + traj)."""

    target_name: str
    proto_obj_pos: np.ndarray
    proto_obj_yaw: float
    proto_eef_pos: np.ndarray
    rel_to_obj_seq: np.ndarray  # (T, 3) robot_pos - proto_obj_pos


@dataclass
class TransportState:
    """Fast-timescale ephemeral state cleared via episode_snap_back()."""

    cur_obj_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    delta_yaw: float = 0.0
    approach_steps: int = 0
    approach_done: bool = False
    last_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def reset(self) -> None:
        self.cur_obj_pos.fill(0.0)
        self.delta_yaw = 0.0
        self.approach_steps = 0
        self.approach_done = False
        self.last_offset.fill(0.0)


def infer_target_object_name(
    task_name: str = "",
    ann_text: str = "",
    frames: Optional[Sequence[dict]] = None,
    block_slices: Optional[Dict[str, Tuple[slice, slice]]] = None,
) -> str:
    """Infer target movable object from task/instruction text or motion variance."""
    text = f"{task_name} {ann_text}".lower()
    if "red" in text:
        return "block_red"
    if "blue" in text:
        return "block_blue"
    if "pink" in text:
        return "block_pink"

    # Fixed-fixture tasks stay in world coordinates unless stack/place uses movable block
    fixed_markers = ("lightbulb", "led", "button", "switch", "drawer", "slider")
    if any(m in text for m in fixed_markers) and "block" not in text and "stack" not in text:
        return "fixed"

    if frames and len(frames) >= 2 and "scene_obs" in frames[0]:
        slices = block_slices or DEFAULT_BLOCK_SLICES
        best_name = "fixed"
        best_disp = 0.02  # ignore sub-2cm jitter
        s0 = np.asarray(frames[0]["scene_obs"], dtype=np.float32)
        s1 = np.asarray(frames[-1]["scene_obs"], dtype=np.float32)
        for name, (pos_sl, _) in slices.items():
            disp = float(np.linalg.norm(s1[pos_sl] - s0[pos_sl]))
            if disp > best_disp:
                best_disp = disp
                best_name = name
        return best_name

    return "fixed"


def rotate_xy_vector(vec: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Apply in-plane Z rotation to 2D/3D vector(s) of shape (..., D>=2)."""
    if abs(yaw_rad) < 1e-6:
        return vec
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    out = vec.copy()
    x = vec[..., 0]
    y = vec[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def wrap_angle_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class ObjectAnchorEstimator(ABC):
    """Pluggable interface isolating scene-state estimation from DTRA tube execution."""

    @abstractmethod
    def estimate(
        self,
        obs: dict,
        target_name: str,
        info: Optional[dict] = None,
    ) -> ObjectAnchor:
        raise NotImplementedError


class OracleSceneAnchorEstimator(ObjectAnchorEstimator):
    """Stage-1 estimator reading exact 3D pose from simulator scene_info or scene_obs."""

    def __init__(self, block_slices: Optional[Dict[str, Tuple[slice, slice]]] = None) -> None:
        self.block_slices = block_slices or DEFAULT_BLOCK_SLICES

    def estimate(
        self,
        obs: dict,
        target_name: str,
        info: Optional[dict] = None,
    ) -> ObjectAnchor:
        if target_name == "fixed":
            return ObjectAnchor(name="fixed", pos=np.zeros(3, dtype=np.float32), yaw=0.0, valid=False)

        if info and "scene_info" in info:
            movable = info["scene_info"].get("movable_objects", {})
            if target_name in movable:
                obj = movable[target_name]
                pos = np.asarray(obj["current_pos"], dtype=np.float32)[:3]
                orn = np.asarray(obj.get("current_orn", [0, 0, 0, 1]), dtype=np.float32)
                yaw = 0.0
                if len(orn) == 4:
                    # Quaternion (x, y, z, w) -> yaw around Z
                    x, y, z, w = float(orn[0]), float(orn[1]), float(orn[2]), float(orn[3])
                    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
                return ObjectAnchor(name=target_name, pos=pos, yaw=yaw, valid=True)

        if "scene_obs" in obs and target_name in self.block_slices:
            pos_sl, orn_sl = self.block_slices[target_name]
            sc = np.asarray(obs["scene_obs"], dtype=np.float32)
            pos = sc[pos_sl].copy()
            yaw = float(sc[orn_sl][-1])
            return ObjectAnchor(name=target_name, pos=pos, yaw=yaw, valid=True)

        return ObjectAnchor(name=target_name, pos=np.zeros(3, dtype=np.float32), yaw=0.0, valid=False)


class VisionDepthAnchorEstimator(ObjectAnchorEstimator):
    """Stage-2 vision-only estimator using RGB color-saliency centroid + depth backprojection.

    Fits an affine pixel-depth-to-world regressor from calibration samples (e.g., training
    frames where robot TCP reaches known tabletop depths or camera projection matrix)
    so inference never touches simulator `scene_obs`.
    """

    def __init__(self) -> None:
        # Linear map from [u_norm, v_norm, depth, 1.0] -> world [x, y, z]
        self.uvd_to_xyz: Optional[np.ndarray] = None

    def fit_linear_camera_calibration(
        self,
        uvd_points: np.ndarray,
        world_xyz_points: np.ndarray,
    ) -> float:
        """Fit a 4x3 affine map [u_norm, v_norm, depth, 1] -> [x, y, z] via least squares."""
        uvd = np.asarray(uvd_points, dtype=np.float64)
        xyz = np.asarray(world_xyz_points, dtype=np.float64)
        design = np.concatenate([uvd, np.ones((uvd.shape[0], 1), dtype=np.float64)], axis=1)
        coef, _, _, _ = np.linalg.lstsq(design, xyz, rcond=None)
        self.uvd_to_xyz = coef.astype(np.float32)
        pred = design @ coef
        return float(np.sqrt(np.mean(np.sum((pred - xyz) ** 2, axis=1))))

    @staticmethod
    def detect_block_uvd(
        rgb_static: np.ndarray,
        depth_static: np.ndarray,
        target_name: str,
    ) -> Optional[Tuple[float, float, float]]:
        if target_name not in COLOR_RGB_PRIORS:
            return None
        rgb = np.asarray(rgb_static, dtype=np.float32) / 255.0
        depth = np.asarray(depth_static, dtype=np.float32)
        h, w, _ = rgb.shape

        _, weights = COLOR_RGB_PRIORS[target_name]
        # Simple linear color opponent score across image pixels
        score = rgb @ weights
        # Restrict out extreme table borders where static robot base sits
        mask_region = np.zeros((h, w), dtype=bool)
        mask_region[int(0.18 * h):int(0.88 * h), int(0.12 * w):int(0.88 * w)] = True
        score = np.where(mask_region, score, -1e9)

        thresh = float(np.percentile(score[mask_region], 99.0))
        cand = (score >= max(thresh, 0.20)) & mask_region
        if int(cand.sum()) < 4:
            return None
        ys, xs = np.where(cand)
        u_norm = float(np.median(xs)) / float(w)
        v_norm = float(np.median(ys)) / float(h)
        d_val = float(np.median(depth[ys, xs]))
        return u_norm, v_norm, d_val

    def estimate(
        self,
        obs: dict,
        target_name: str,
        info: Optional[dict] = None,
    ) -> ObjectAnchor:
        if target_name == "fixed" or self.uvd_to_xyz is None:
            return ObjectAnchor(name=target_name, pos=np.zeros(3, dtype=np.float32), yaw=0.0, valid=False)

        rgb_obs = obs.get("rgb_obs", obs)
        depth_obs = obs.get("depth_obs", obs)
        if "rgb_static" not in rgb_obs or "depth_static" not in depth_obs:
            return ObjectAnchor(name=target_name, pos=np.zeros(3, dtype=np.float32), yaw=0.0, valid=False)

        uvd = self.detect_block_uvd(rgb_obs["rgb_static"], depth_obs["depth_static"], target_name)
        if uvd is None:
            return ObjectAnchor(name=target_name, pos=np.zeros(3, dtype=np.float32), yaw=0.0, valid=False)

        feat = np.array([uvd[0], uvd[1], uvd[2], 1.0], dtype=np.float32)
        xyz = feat @ self.uvd_to_xyz
        return ObjectAnchor(name=target_name, pos=xyz.astype(np.float32), yaw=0.0, valid=True)


class ObjectCentricTransportAdapter:
    """Wraps frozen DTRA B-Spline trajectory synthesis with SE(2) object transport."""

    def __init__(
        self,
        approach_tol_m: float = 0.030,
        max_approach_steps: int = 18,
        approach_gain: float = 12.0,
        track_gain: float = 4.5,
        enable_yaw_rotation: bool = False,
        safe_z_lift_m: float = 0.05,
    ) -> None:
        self.approach_tol_m = float(approach_tol_m)
        self.max_approach_steps = int(max_approach_steps)
        self.approach_gain = float(approach_gain)
        self.track_gain = float(track_gain)
        self.enable_yaw_rotation = bool(enable_yaw_rotation)
        self.safe_z_lift_m = float(safe_z_lift_m)
        self.state = TransportState()

    def episode_snap_back(self) -> float:
        self.state.reset()
        return 0.0

    def register_cell_anchor(
        self,
        task_name: str,
        ann_text: str,
        frames: Sequence[dict],
        estimator: ObjectAnchorEstimator,
    ) -> TransportCellAnchor:
        target = infer_target_object_name(task_name=task_name, ann_text=ann_text, frames=frames)
        f0 = frames[0]
        robot_seq = np.stack([np.asarray(f["robot_obs"], dtype=np.float32)[:3] for f in frames], axis=0)
        proto_eef = robot_seq[0].copy()

        anchor = estimator.estimate(f0, target)
        if anchor.valid:
            proto_pos = anchor.pos.copy()
            proto_yaw = float(anchor.yaw)
            rel_seq = (robot_seq - proto_pos[None, :]).astype(np.float32)
        else:
            target = "fixed"
            proto_pos = np.zeros(3, dtype=np.float32)
            proto_yaw = 0.0
            rel_seq = robot_seq.copy()

        return TransportCellAnchor(
            target_name=target,
            proto_obj_pos=proto_pos,
            proto_obj_yaw=proto_yaw,
            proto_eef_pos=proto_eef,
            rel_to_obj_seq=rel_seq,
        )

    def bind_episode(
        self,
        cell_anchor: TransportCellAnchor,
        obs: dict,
        estimator: ObjectAnchorEstimator,
        info: Optional[dict] = None,
    ) -> Tuple[np.ndarray, float]:
        """Compute transported world reference trajectory for the current episode spawn."""
        self.state.reset()
        if cell_anchor.target_name == "fixed":
            self.state.approach_done = False
            return cell_anchor.rel_to_obj_seq.copy(), 0.0

        cur_anchor = estimator.estimate(obs, cell_anchor.target_name, info=info)
        if not cur_anchor.valid:
            return (cell_anchor.proto_obj_pos[None, :] + cell_anchor.rel_to_obj_seq).astype(np.float32), 0.0

        self.state.cur_obj_pos = cur_anchor.pos.copy()
        delta_yaw = wrap_angle_pi(cur_anchor.yaw - cell_anchor.proto_obj_yaw) if self.enable_yaw_rotation else 0.0
        self.state.delta_yaw = delta_yaw

        rotated_rel = rotate_xy_vector(cell_anchor.rel_to_obj_seq, delta_yaw)
        transported_ref_pos = (cur_anchor.pos[None, :] + rotated_rel).astype(np.float32)
        return transported_ref_pos, delta_yaw

    def step_action(
        self,
        cur_eef_pos: np.ndarray,
        base_rel_action: np.ndarray,
        transported_ref_pos: np.ndarray,
        tube_step_idx: int,
        ref_orn_seq: Optional[np.ndarray] = None,
        cur_eef_orn: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """Return (7D action, advance_tube_flag). Keeps Phase-0 approach separate from Phase-1."""
        cur_pos = np.asarray(cur_eef_pos, dtype=np.float32)[:3]
        ref_start = transported_ref_pos[0]

        # Phase 0: Pre-execution alignment to the transported tube entry point
        if not self.state.approach_done and self.state.approach_steps < self.max_approach_steps:
            err_entry = ref_start - cur_pos
            dist_entry = float(np.linalg.norm(err_entry))
            if dist_entry > self.approach_tol_m:
                xy_dist = float(np.linalg.norm(err_entry[:2]))
                target_pt = ref_start.copy()
                if xy_dist > 0.07:
                    # Clear tabletop obstacles smoothly when transitioning from a previous task
                    target_pt[2] = max(float(ref_start[2]) + self.safe_z_lift_m, float(cur_pos[2]))
                err_target = target_pt - cur_pos
                delta_xyz = np.clip(self.approach_gain * err_target, -0.8, 0.8)
                self.state.last_offset = delta_xyz.astype(np.float32)
                self.state.approach_steps += 1

                act = np.zeros(7, dtype=np.float32)
                act[:3] = delta_xyz
                act[6] = 1.0 if base_rel_action[6] >= 0.0 else -1.0
                return act, False
            self.state.approach_done = True

        # Phase 1: Execute frozen B-Spline trajectory rotated/translated into object frame
        idx = min(tube_step_idx, len(transported_ref_pos) - 1)
        act = np.asarray(base_rel_action, dtype=np.float32).copy()
        if abs(self.state.delta_yaw) > 1e-6:
            act[:2] = rotate_xy_vector(act[:2], self.state.delta_yaw)

        tube_err = np.clip(transported_ref_pos[idx] - cur_pos, -0.06, 0.06)
        fb_xyz = np.clip(self.track_gain * tube_err, -0.6, 0.6)
        self.state.last_offset = fb_xyz.astype(np.float32)
        act[:3] = np.clip(act[:3] + fb_xyz, -1.0, 1.0)

        if ref_orn_seq is not None and cur_eef_orn is not None:
            o_idx = min(tube_step_idx, len(ref_orn_seq) - 1)
            orn_err = np.clip(ref_orn_seq[o_idx] - np.asarray(cur_eef_orn, dtype=np.float32)[:3], -0.2, 0.2)
            act[3:6] = np.clip(act[3:6] + 0.15 * orn_err, -1.0, 1.0)

        act[6] = 1.0 if act[6] >= 0.0 else -1.0
        return act, True

