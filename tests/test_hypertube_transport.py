"""tests/test_hypertube_transport.py

Pure-NumPy unit tests for Object-Centric Hypertube Transport:
1. Spatial equivariance (arbitrary block spawn shift -> identical relative trajectory)
2. SE(2) yaw rotation invariance
3. Phase-0 approach separation + episode_snap_back() zero residual state
4. VisionDepthAnchorEstimator least-squares camera calibration + color saliency retrieval
"""

import unittest
import numpy as np

from dtra.core.hypertube_transport import (
    ObjectCentricTransportAdapter,
    OracleSceneAnchorEstimator,
    VisionDepthAnchorEstimator,
    infer_target_object_name,
)


def _make_dummy_episode(block_xyz, eef_start, T=12):
    frames = []
    # Smooth approach toward block_xyz + lift
    eef = np.array(eef_start, dtype=np.float32)
    for t in range(T):
        alpha = t / max(1, T - 1)
        pos = (1.0 - alpha) * eef + alpha * (np.array(block_xyz, dtype=np.float32) + np.array([0, 0, 0.08], dtype=np.float32))
        rob = np.zeros(15, dtype=np.float32)
        rob[:3] = pos
        sc = np.zeros(24, dtype=np.float32)
        sc[6:9] = block_xyz
        frames.append({
            "robot_obs": rob,
            "scene_obs": sc,
            "rel_actions": np.zeros(7, dtype=np.float32),
        })
    return frames


class TestHypertubeTransport(unittest.TestCase):
    def test_spatial_equivariance_translation_and_snapback(self):
        adapter = ObjectCentricTransportAdapter(approach_tol_m=0.02, max_approach_steps=15)
        est = OracleSceneAnchorEstimator()

        proto_block = np.array([0.0, -0.10, 0.46], dtype=np.float32)
        proto_eef = np.array([0.0, -0.10, 0.56], dtype=np.float32)
        frames = _make_dummy_episode(proto_block, proto_eef, T=10)

        cell = adapter.register_cell_anchor("lift_red_block_table", "lift the red block", frames, est)
        self.assertEqual(cell.target_name, "block_red")

        # Shift block by +25cm in X and +15cm in Y at test time
        shifted_block = proto_block + np.array([0.25, 0.15, 0.0], dtype=np.float32)
        sc_test = np.zeros(24, dtype=np.float32)
        sc_test[6:9] = shifted_block
        obs = {"scene_obs": sc_test}

        ref_pos, d_yaw = adapter.bind_episode(cell, obs, est)
        self.assertAlmostEqual(d_yaw, 0.0, places=5)
        expected_start = proto_eef + np.array([0.25, 0.15, 0.0], dtype=np.float32)
        np.testing.assert_allclose(ref_pos[0], expected_start, atol=1e-5)

        # Far gripper should stay in Phase-0 (advance=False) and pull toward expected_start
        far_eef = np.array([-0.20, -0.20, 0.56], dtype=np.float32)
        act, advance = adapter.step_action(far_eef, np.array([0, 0, -0.2, 0, 0, 0, 1.0], dtype=np.float32), ref_pos, 0)
        self.assertFalse(advance)
        self.assertGreater(act[0], 0.0)
        self.assertGreater(act[1], 0.0)

        # Aligned gripper should enter Phase-1 (advance=True)
        act2, advance2 = adapter.step_action(expected_start, np.array([0.1, 0, -0.2, 0, 0, 0, 1.0], dtype=np.float32), ref_pos, 0)
        self.assertTrue(advance2)
        self.assertAlmostEqual(float(act2[0]), 0.1, places=4)

        # Snap-back must zero all ephemeral state
        adapter.episode_snap_back()
        self.assertEqual(float(np.linalg.norm(adapter.state.last_offset)), 0.0)
        self.assertEqual(adapter.state.approach_steps, 0)

    def test_vision_depth_calibration_and_detection(self):
        vis = VisionDepthAnchorEstimator()
        rng = np.random.RandomState(7)
        uvd = rng.uniform([0.2, 0.2, 0.8], [0.8, 0.8, 1.4], size=(24, 3))
        true_M = np.array([[0.6, -0.1, 0.05], [0.05, 0.5, -0.1], [-0.02, 0.02, -0.4], [-0.2, -0.1, 0.9]])
        xyz = np.concatenate([uvd, np.ones((24, 1))], axis=1) @ true_M
        rmse = vis.fit_linear_camera_calibration(uvd, xyz)
        self.assertLess(rmse, 1e-6)

        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, :] = [120, 120, 120]
        rgb[40:50, 60:70] = [240, 20, 20]  # Red block patch at (u=0.645, v=0.445)
        depth = np.full((100, 100), 1.1, dtype=np.float32)
        depth[40:50, 60:70] = 0.95

        anchor = vis.estimate({"rgb_obs": {"rgb_static": rgb}, "depth_obs": {"depth_static": depth}}, "block_red")
        self.assertTrue(anchor.valid)
        expected = np.array([0.645, 0.445, 0.95, 1.0]) @ true_M
        np.testing.assert_allclose(anchor.pos, expected, atol=0.02)

    def test_fixed_task_inference(self):
        self.assertEqual(infer_target_object_name("turn_off_lightbulb", "toggle light switch off"), "fixed")
        self.assertEqual(infer_target_object_name("push_pink_block_right", "push the pink block right"), "block_pink")


if __name__ == "__main__":
    unittest.main()
