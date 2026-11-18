# CALVIN DTRA v6 (System 1 + System 2) Scope and Limitations

This archive is strictly separate from the published `DTRA v3.0.0` release and belongs to the post-v3 embodied continual-learning / System 1+System 2 research track (bridged from the RoboCasa v4 atomic-vs-composite diagnosis).

## Experimental scope

- Dataset and simulator: CALVIN `task_D_D_subset` (34 atomic tasks; `max_per_task=10` training episodes per task -> 340 topological cells; `max_eval_per_task=5` validation episodes per task; `seed=42`).
- Two matched anchor modes under identical zero-leak runtime rules (`CALVIN_EVAL_MODE=vision` uses only `rgb_static`, `depth_static`, `rgb_gripper`, and `robot_obs`, never `scene_obs`; `oracle` reads simulator `scene_obs` only via `OracleSceneAnchorEstimator` as an upper-bound comparison).
- Evaluation suite in `run_calvin_dtra_v6.py`:
  1. Single-task continual evaluation across all 34 tasks (`170` total episodes, no replay).
  2. Multi-step 5-task chain evaluation (`num_chains=30` precondition-valid CALVIN sequences grounded in each chain's initial `scene_obs`, including the canonical Section 4.2 5-step scenario) comparing `Ablation_HardSwitch_NoBridge` against `DTRA_V6_System1_Bridge_System2_Organ`.

## Architecture & v6 Fixes Applied (System 1 + System 2)

1. **Pre-existing System 2 status clarified:** `OrganSystem2Planner` (`topological_bridge_v6.py`) + `_route_cell_system2` (`run_calvin_dtra_v6.py`) operate as a lightweight topological executive layer consisting of OOD resonance detection (`max_sc < 0.50`), episodic refractory memory (`MyelinatedTransitionGraph.episodic_trace`), object-noun cluster + polarity filtering, and a 2-attempt refractory backtracking loop (`max_attempts=2`), rather than a full VLM/symbolic planner.
2. **3D $Z$-Gating in Vision Target Extractor (`run_calvin_dtra_v5.py`, `hypertube_transport.py`):**
   - Added `"door"` and `"cabinet"` to `fixed_markers` (`infer_target_object_name`) and `"door"`, `"handle"`, `"switch"`, `"button"` to `DISCRIMINATIVE_WORDS`.
   - Separated 3D workspace gating in `CalvinCameraVisionAnchorEstimatorV5.resolve_z_bounds`:
     - `door` / `slider` targets: upper cabinet region **$Z \in [0.52, 0.65]\text{m}$**, $Y \in [-0.16, 0.12]\text{m}$.
     - Tabletop `block_*` targets: tabletop plane **$Z \in [0.43, 0.50]\text{m}$** (while drawer block tasks search $Z \in [0.31, 0.43]\text{m}$ and cabinet/slider block tasks search $Z \in [0.50, 0.62]\text{m}$).
     - `drawer` targets: $Z \in [0.31, 0.44]\text{m}$.
   - Refined `block_blue` mask (`b > r + 0.10`) so pink blocks ($r \approx b$) cannot be misidentified as blue blocks.
3. **System 2 Object-Noun Cluster + Polarity Filter (`topological_bridge_v6.py`, `run_calvin_dtra_v6.py`):**
   - Added `OrganSystem2Planner.extract_object_noun_cluster`, `extract_block_color`, `extract_action_polarity`, and `filter_candidate_cells` isolating `door_slider`, `drawer`, `block`, `place_drawer`, `place_slider`, `push_drawer`, and `light` cell clusters plus action direction (`left`/`right`, `open`/`close`, `on`/`off`).
   - Prevented refractory failure recovery (`attempt == 1`) from crossing object categories (eliminating the `CELL #108` `move_slider_left` -> `CELL #155` `push_blue_block_left` failure mode).
   - Added upper-cabinet trajectory prior (`init_z >= 0.52`, `max_z >= 0.53`) and bridge $Z$-clearance ($\ge 0.52\text{m}$) for `door_slider` transitions.
4. **Precondition-Valid Chain Builder & Upgraded HUD (`run_calvin_dtra_v6.py`, `render_v6_hud_demo.py`):**
   - Fixed `build_valid_5step_chains` to track `(slider_left, drawer_open, led_on, bulb_on, holding)` against `ep0["frames"][0]["scene_obs"]`, eliminating physically impossible chains (e.g., `place_in_*` with an empty gripper or pushing a slider already at its joint stop `0.273`).
   - Upgraded `render_v6_hud_demo.py` with a full 5-row subgoal checklist, per-chain vs cumulative System 2 recovery counter, post-completion `5/5 CHAIN SOLVED! (PURE VISION)` banner, and 24-frame (`1.0s`) final solved hold.

## Recorded results and honest limitations

- **34-Task Vision Benchmark (`calvin_v6_vision_34tasks.json`):**
  - `route_acc`: **97.65%** (up from 97.06% pre-patch)
  - `single_task_mean_sr`: **55.88%** (strict tabletop $[0.43, 0.50]\text{m}$ $Z$-gating trades 4.12 pp vs the un-gated `60.0%` baseline in exchange for zero cross-plane confusion between tabletop blocks and the upper cabinet)
  - `DTRA_V6_System1_Bridge_System2_Organ`: `avg_5step_chain_len` = **2.37** (vs `1.90` Hard-Switch baseline), `chain_5step_full_sr` = **16.67%** (vs `6.67%` Hard-Switch baseline), `mean_boundary_jerk` = **0.0752** (`+15.12%` jerk reduction vs `0.0886`), `myelinated_edges` = **22**, `backtrack_recoveries` = **30**, `closed_loop_latency_ms` = **9.3022 ms**, `parameter_drift` = **0.0**.
- **34-Task Oracle Upper-Bound (`calvin_v6_oracle_34tasks.json`):**
  - `route_acc`: **97.65%** (up from 95.88%), `single_task_mean_sr`: **46.47%**, `avg_5step_chain_len`: **2.53** (vs `1.90` Hard-Switch), `chain_5step_full_sr`: **16.67%** (vs `10.00%`), `mean_boundary_jerk`: **0.0684** (`+15.24%` jerk reduction vs `0.0807`), `myelinated_edges` = **25**, `parameter_drift` = **0.0**, `closed_loop_latency_ms` = **8.4917 ms**.
- **HUD Rollout Video (`dtra_v6_pure_vision_chain_hud.mp4`, `.gif`):**
  - `render_v6_hud_demo.py` captured **Chain #1** (`Completed=5/5` consecutive tasks in pure `VISION`, `Leak: ZERO`, `Drift: 0.0000`, `ChainRecov=2` live System 2 recoveries, `603 frames`, `960x540 @ 24 fps`):
    1. `grasp the red block on the table, then lift it up` (`lift_red_block_table`)
    2. `move to the drawer, then place the object` (`place_in_drawer`)
    3. `close the cabinet drawer` (`close_drawer`)
    4. `push the door to the left side` (`move_slider_left`, recovered via `CELL #108 -> CELL #100`)
    5. `move the light switch to turn on the yellow light` (`turn_on_lightbulb`)
- **Limitations:**
  - `parameter_drift` (`0.0`) holds by construction via frozen memory tensors and `episode_snap_back()` ephemeral buffer zeroing.
  - `render_v6_hud_demo.py` showcases the canonical `5/5` sequence (**Chain #1** among 30 precondition-valid chains) rather than claiming 100% 5-step chain completion across all sequences (`16.67%` full 5-step SR / `2.37` mean chain length in Vision mode).
  - Precondition planning (`sub_plan`) remains a lightweight wrapper without full VLM/symbolic precondition decomposition when an arbitrary episode starts with unsatisfied scene state.

## Upstream integrity

- Consult `upstream_git_diff_evidence.txt` for commit hashes, remotes, complete porcelain status, and tracked/index diffs across both `calvin` (`fa03f01...`) and `calvin_env` (`1431a46...`). All tracked, cached, and porcelain states are 100% clean.
