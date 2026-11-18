# Research Lineage: From Drawing-VLA (v1) to DTRA (v2)

This document chronicles the research progression from **1st-Generation B-Spline Action Chunking (`v1_drawing_vla`)** to **2nd-Generation Dynamic Tube Routing Architecture (`dtra`)**, recording the architectural hypotheses, structural failure modes uncovered along the way, and verified empirical milestones across 2D sketch generation, 3D Meta-World manipulation, and 7-DoF LIBERO manipulation.

---

## 1. Chronological Evolution

```
[Gen 1: v1_drawing_vla]
  1.1  Drawing-VLA (v1.1.2)
       - Continuous 2D stroke generation via clamped cubic B-Spline chunking (M=6, H=16, exec=8)
       - Frozen B-Spline basis matrix B guarantees C^2 continuity and 80% temporal compression
         |
         v
  1.2  ABK-VLA (v1.1.3)
       - Added LoRA-adapted frozen visual-prototype routing + DoG 3D geometric keys + Skill-Pool expansion
       - Solved catastrophic forgetting in 2D sketch CL (avg final F1 = 0.856, FR = 0.017 vs Naive FT 0.216)
         |
         +--> Exposed 4 structural ceilings in 3D Continual Manipulation:
              (C1) O(K) Sequential Routing Bottleneck (DoG + Blend CPU latency grows 7.41x: 0.333 -> 2.470 ms)
              (C2) t=0 Visual Collision & Frame Confusion (similar start views across tasks)
              (C3) OOD Brittleness of Fixed Geometric Keys (Arm 1 OOD sigma=0.20 SR drops 82.8% -> 2.2%, Route 34.7%)
              (C4) Static Routing & Linear Blending Interference (FR = 8.6% under v1.1.3 blend on Meta-World)
         |
         v
[Gen 2: dtra (v2.0 -> v2.1)]
  2.1  DTRA Step-1 Mathematical Foundation (`dtra/core/sim_dtra_tube_dynamics.py`)
       - Elliptical Hypertube Rings on S^63 (closed-form atan2 geodesic phase + elliptical radial envelope)
       - Reversible elastic bending (delta_z*, bounded alpha_i in [0,1]) + snap_back() -> 0.0 residual state
         |
         v
  2.2  DTRA 3D Manipulation & Multimodal Resonance (`dtra/experiments/metaworld/`)
       - Multimodal key (frozen DINOv2-style patch tokens + MiniLM-L6-v2 instruction embedding)
       - O(1) GPU tensorized routing (~1.87 ms flat from K=1 to K=8), 0.0% FR, 100.0% RouteAcc across clean/OOD/reverse
         |
         v
  2.3  DTRA-LIBERO 7-DoF Multi-Stage Control (`dtra/experiments/libero/`)
       - Dual-camera (128x128 agentview + eye_in_hand) + proprioception (`ee_pos`, `gripper_qpos`) + M_ij myelination
       - Verified on LIBERO-Goal 5-task pilot (DTRA Elastic 92.0% SR, 0.0% BWT, 100.0% RouteAcc, 3,696 MB VRAM)
```

---

## 2. Structural Transition Matrix (`v1` -> `v2`)

| Dimension | Gen 1: `v1_drawing_vla` (ABK-VLA v1.1.3) | Gen 2: `dtra` (DTRA v2.1) | Empirical Gain / Resolution |
| :--- | :--- | :--- | :--- |
| **Action Decoder** | Clamped cubic B-Spline ($M=6$, $H=16$) | Same frozen B-Spline tensor decoder (`dtra/core/bspline_core.py`) | Preserved $C^2$ smoothness in 2D (`xy`), 3D (`xyz`+grip), and 7-DoF LIBERO (`num_cp=8, H=16`) |
| **Task Memory Topology** | Discrete per-task LoRA adapters + scalar DoG prototypes | Elliptical hypertube rings $\mathcal{T}_i(\theta)$ on unit hypersphere $S^{63}$ | Eliminates $t=0$ state aliasing via closed-form phase $\theta^* = \mathrm{atan2}(\cdot)$ |
| **Routing Complexity ($K=1 \to 8$)** | Sequential LoRA / blend overhead ($O(K)$) | Tensorized cosine + inner-product docking ($O(1)$ on GPU; flat CPU cost) | CPU active time: ABK Blend `0.333 -> 2.470 ms` (7.41x) vs DTRA Full `1.866 -> 1.866 ms` (1.00x) |
| **Query Key Modality** | Vision DoG geometric signature | Normalized vision + frozen `all-MiniLM-L6-v2` language key | Multimodal RouteAcc improves to `100.0%` under both clean and `sigma=0.20` OOD |
| **Inference Adaptation** | Rigid weights ($\Delta z = 0$) | Elastic offset $\delta z^*$ with bounded $\alpha_i \in [0, 1]$ + episode-end `snap_back()` | Meta-World $\sigma=0.20$ SR: ABK `2.2%` -> DTRA Full Multimodal `51.4%`; post-episode residual = `0.0` |
| **Inter-Skill Coupling** | Linear weight/output blending | Symmetric myelination matrix $M_{ij}$ over tube skills | Prevents cross-task drift (`FR = 0.0%` on Meta-World 3-seed & Reverse Order `T7 -> T0`) |

---

## 3. Verified Experimental Milestones

### Milestone 1: Gen-1 Autonomous Sketch Continual Learning (`v1_drawing_vla`)
Source: [`outputs/v1_archive/autonomous_cl_benchmark.json`](outputs/v1_archive/autonomous_cl_benchmark.json) (`v1_drawing_vla/src/experiments/benchmark_autonomous_cl.py`)

| Model | Avg Final F1 | Forgetting (FR) | Backward Transfer (BWT) |
| :--- | :---: | :---: | :---: |
| Model 1: GRU + Naive FT | 0.121 | 0.464 | -0.464 |
| Model 2: ABK-VLA + Naive FT | 0.216 | 0.893 | -0.893 |
| Model 3: ABK-VLA + Exp. Replay | 0.629 | 0.141 | +0.020 |
| Model 4: ABK-VLA + Oracle (Model C) | 0.930 | 0.000 | 0.000 |
| **Model 5: Autonomous ABK-VLA (Model D)** | **0.856** | **0.017** | **-0.017** |

---

### Milestone 2: Gen-2 Meta-World Initial Pilot & 6-Arm 3-Seed Scientific Ablation (`dtra`)
1. **Initial Meta-World Pilot** ([`outputs/metaworld/dtra_3d_metaworld_benchmark.json`](outputs/metaworld/dtra_3d_metaworld_benchmark.json)):
   - `v1.1.3 ABK (Blend)`: Final SR `80.0%`, FR `8.6%`, BWT `-5.7%`
   - `v1.1.3 ABK (Argmax)`: Final SR `82.5%`, FR `0.0%`, BWT `0.0%`
   - `DTRA v2.1 (3D Hypertube O(1))`: Final SR `90.0%`, FR `0.0%`, BWT `0.0%`

2. **Rigorous 6-Arm 3-Seed Ablation (Seeds `42, 100, 2026`, 8 Meta-World Tasks)** ([`outputs/metaworld/ablate_dtra_scientific_results.json`](outputs/metaworld/ablate_dtra_scientific_results.json)):

| Arm | Key Design | Params | Clean SR | FR | RouteAcc | OOD ($\sigma=0.20$) SR | OOD Route | CPU Active $K=1 \to 8$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `arm1_abk_dog_vision` | v1.1.3 DoG Vision Key | 5.25M | 82.8% $\pm$ 2.6% | 0.0% | 99.7% | 2.2% $\pm$ 1.0% | 34.7% | `0.273 -> 0.825 ms` (`blend`: `0.333 -> 2.470 ms`) |
| `arm2_frozen_mlp_vision` | Frozen Slot MLP (Vision) | 5.25M | 45.8% $\pm$ 3.5% | 20.0% $\pm$ 0.8% | 99.2% | 25.0% $\pm$ 3.0% | 94.4% | - |
| `arm3_dtra_full_vision` | DTRA Full (Vision) | 5.25M | 75.8% $\pm$ 5.1% | 0.0% | 99.2% | **52.5% $\pm$ 10.3%** | 95.0% | `1.969 -> 1.873 ms` |
| `arm4_frozen_mlp_multimodal` | Frozen Slot MLP (Multimodal) | 5.25M | **85.6% $\pm$ 3.1%** | 0.0% | **100.0%** | 23.6% $\pm$ 3.9% | **100.0%** | - |
| `arm5_dtra_rigid_alpha0` | DTRA Rigid ($\alpha=0$) | 5.25M | 77.8% $\pm$ 3.5% | 0.0% | **100.0%** | **53.6% $\pm$ 6.8%** | **100.0%** | `1.863 -> 1.872 ms` |
| `arm6_dtra_full_multimodal` | DTRA Full (Multimodal) | 5.25M | 72.2% $\pm$ 6.8% | **0.0%** | **100.0%** | **51.4% $\pm$ 4.4%** | **100.0%** | `1.866 -> 1.866 ms` (`post_snap_buf_abs_max = 0.0`) |

---

### Milestone 3: Supplemental CL Baselines, 9-Regime OOD & Task-Order Invariance (`Seed 42`)
Source: [`outputs/metaworld/ablate_dtra_supplemental_results.json`](outputs/metaworld/ablate_dtra_supplemental_results.json) (`dtra/experiments/metaworld/ablate_dtra_supplemental.py`)

- **Part 1 — Classical Shared-Backbone CL Baselines (8 tasks, Seed 42):**
  - `Naive Fine-Tuning`: Clean SR `12.5%`, FR `98.1%`, OOD ($\sigma=0.20$) SR `5.8%`
  - `EWC`: Clean SR `11.7%`, FR `100.0%`, OOD ($\sigma=0.20$) SR `18.3%`
  - `Experience Replay (5%)`: Clean SR `77.5%`, FR `20.0%`, OOD ($\sigma=0.20$) SR `25.0%`
  - `Arm 6 (DTRA Full Multimodal, Seed 42)`: Clean SR `80.8%`, FR **`0.0%`**, OOD ($\sigma=0.20$) SR **`57.5%`**
- **Part 2 — Zero-Shot 9-Regime OOD Curves (`clean`, Gaussian noise `0.05..0.30`, `cutout 15%/30%`, `bright_high`, `contrast_harsh`):**
  - `arm4_frozen_mlp_multimodal` falls sharply from `90.8%` (clean) -> `56.7%` ($\sigma=0.05$) -> `32.5%` ($\sigma=0.20$) -> `27.5%` (`bright_high`).
  - `arm6_dtra_full_multimodal` retains `81.7%` (clean) -> `73.3%` ($\sigma=0.05$) -> `50.8%` ($\sigma=0.20$) -> `50.8%` ($\sigma=0.30$) -> `51.7%` (`cutout_15`) -> `46.7%` (`cutout_30`) -> `68.3%` (`bright_high`) -> `79.2%` (`contrast_harsh`) at **100.0% RouteAcc** across every regime.
- **Part 3 — Reverse Task Order ($T_7 \to T_0$, `arm6` Seed 42):**
  - Clean SR `74.2%`, Test SR `75.8%`, OOD Severe SR `64.2%`, **FR `0.0%`**, **BWT `0.0%`**, **RouteAcc `100.0%`** (both clean and OOD).

---

### Milestone 4: Gen-2 LIBERO 7-DoF Manipulation (`dtra/experiments/libero/`)
1. **Single-Task Sanity Check (`libero_goal` T0)** ([`outputs/libero/libero_t0_sanity.json`](outputs/libero/libero_t0_sanity.json)):
   - Both `arm_frozen_mlp` and `arm_dtra_elastic` reached `100.0%` SR (`1.0` RouteAcc, `post_snap_norm = 0.0`) on `open_the_middle_drawer_of_the_cabinet`, peaking under `1,283 MB` VRAM (`batch_size=128`).
2. **LIBERO-Goal 5-Task Pilot (`T0..T4`, 4 Arms)** ([`outputs/libero/libero_goal_pilot_4arms.json`](outputs/libero/libero_goal_pilot_4arms.json)):

| Arm | Mean Final SR | Forgetting (FR) | BWT | Mean RouteAcc | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `arm_frozen_mlp` | **100.0%** | **0.0%** | +15.0% | **100.0%** | 3,695 MB |
| `arm_er_5pct` | 88.0% | 10.0% | -5.0% | 20.0% | 3,298 MB |
| `arm_dtra_rigid` | 88.0% | 10.0% | -5.0% | **100.0%** | 3,696 MB |
| `arm_dtra_elastic` | 92.0% | 5.0% | **0.0%** | **100.0%** | 3,696 MB |

---

## 4. Key Documents & Figures

- **Gen-1 Archive & Preprint:** [`v1_drawing_vla/paper_v1/ABK_VLA_Preprint.pdf`](v1_drawing_vla/paper_v1/ABK_VLA_Preprint.pdf)
- **Gen-2 Specification & Preprint:** [`dtra/paper_v2/DTRA_v2_Spec.md`](dtra/paper_v2/DTRA_v2_Spec.md), [`dtra/paper_v2/DTRA_Preprint.pdf`](dtra/paper_v2/DTRA_Preprint.pdf), [`dtra/paper_v2/dtra_v21_preprint.tex`](dtra/paper_v2/dtra_v21_preprint.tex)
- **Publication & XAI Figures (`outputs/figures/`, generated by [`dtra/visualization/visualize_dtra_xai.py`](dtra/visualization/visualize_dtra_xai.py)):**
  - `fig0_dtra_architecture_schematic.png` — End-to-end multimodal docking, $S^{63}$ tube routing & B-Spline pipeline
  - `fig1_ood_perturbation_curves.png` — 9-regime OOD robustness curves
  - `fig2_manifold_elastic_snapback.png` — 3-panel Macro tube layout + 10x Zoom-In elastic stretch & `snap_back()` + stress bar chart
  - `fig3_xai_telemetry_rollout.png` — Live camera frames + 3D B-Spline control points + Route Confidence & $\|\delta z^*\|$ telemetry
  - `fig4_cl_forgetting_heatmap_and_bars.png` — $8 \times 8$ sequential SR heatmap (`Naive FT` / `EWC` / `ER 5%` vs `DTRA Arm 6`) & Clean / FR / OOD bars
  - `fig5_o1_latency_scaling.png` — $K=1 \to K=8$ constant $O(1)$ active control latency (`~1.87 ms`) vs $O(K)$ blending baselines
