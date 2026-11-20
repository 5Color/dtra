# DTRA: Dynamic Topology Ring Architecture

[English](README.md) | [한국어](README_KO.md)

[![Preprint (DTRA v3.0)](https://img.shields.io/badge/Preprint-DTRA__v3.0__Preprint.pdf-b31b1b.svg)](dtra/paper_v2/DTRA_v3.0_Preprint.pdf)
[![arXiv Manuscript](https://img.shields.io/badge/arXiv%20Clean-dtra__preprint.pdf-9e1c20.svg)](dtra/paper_v2/dtra_preprint.pdf)
[![Zenodo v3.0.0 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24297711.svg)](https://doi.org/10.5281/zenodo.24297711)
[![Zenodo Concept DOI](https://img.shields.io/badge/Zenodo%20All%20Versions-10.5281%2Fzenodo.24218482-blue.svg)](https://doi.org/10.5281/zenodo.24218482)
[![Zenodo ABK-VLA](https://img.shields.io/badge/Gen1%20ABK--VLA-10.5281%2Fzenodo.24212130-lightgrey)](https://doi.org/10.5281/zenodo.24212130)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

**Replay-free, Task-ID-free continual visuomotor control with hyperspherical routing and shared motor bases: `87.00 ± 0.86%` final LIBERO-100 success across three training seeds (verified at `85.20 ± 0.35%` over 12,000 zero-retrain rollouts at `n_eval=20`, outperforming matched-encoder `DER++`, `L2P`, and `O-LoRA`).**

> **Repository Separation Note:** This repository tracks the full **ARIN $\to$ Drawing-VLA $\to$ ABK-VLA $\to$ DTRA v3.0–v6.0** research lineage across two generations:
> 1. **Gen 1 — `v1_drawing_vla` (`Drawing-VLA` v1.0.0 & `ABK-VLA` v1.1.3):** Closed-loop ACT sketch chunking ($K=16$, Catmull-Rom brush rendering in `Drawing-VLA`, DOI `10.5281/zenodo.24122154`) followed by clamped cubic B-Spline control-knot compression ($N_k=6 \to H=16$), geometric DoG prototype routing, and continual 2D/3D benchmarks (`ABK-VLA`, DOI `10.5281/zenodo.24212130`).
> 2. **Gen 2 — `dtra` (**DTRA v3.0 — Dynamic Topology Ring Architecture**, corresponding to [`5Color/dtra`](https://github.com/5Color/dtra)):** Resolves Voronoi encroachment, duplicate-instruction routing collapse, and $O(K)$ task-specific head growth across $K=100$ sequential 7-DoF LIBERO tasks via $\mathbb{S}^{63}$ topological ring isolation, Two-Stage Top-$M$ CMC, `SharedBasisBank` ($100 \to 27$ frozen primitive bases, `-73%`), and Two-Timescale Snap-Back (`post_snap_norm = 0.0000`).
> 3. **Post-v3 Embodied Continual Tracks (`v4.0`–`v6.0`):** Strictly excluded from **DTRA v3.0** preprint tables and isolated under `dtra/experiments/{robocasa_v4,calvin_v4,calvin_v5}/` and `dtra/outputs/calvin_v6/` (archived alongside `outputs/{organ_v4,calvin_v4,calvin_v5,calvin_v6,robocasa}/`). Note that `v6.0`'s `OrganSystem2Planner` borrows the `Organ` name from **ARIN** as a lightweight System 2 routing prototype, distinct from the conceptual multi-ring $10^4\text{--}10^8$ `Organ` roadmap.

> **Inference-input scope:** Task-ID-free means no integer task index is passed to the policy at inference. Routing uses dual-camera features (plus depth in pure RGB-D tracks), proprioception, and a language embedding; evaluations that provide language commands are command-conditioned, not observation-only.

For the complete hypothesis-by-hypothesis chronology, see [**RESEARCH_LINEAGE.md**](RESEARCH_LINEAGE.md), [**RELEASE_NOTES_v3.0.0.md**](RELEASE_NOTES_v3.0.0.md), [**dtra/README.md**](dtra/README.md), and [**dtra/PAPER_20P_OUTLINE.md**](dtra/PAPER_20P_OUTLINE.md).

---

## Visual Rollout Demos (Post-v3 Embodied Tracks)

| DTRA v6.0 — CALVIN Pure Vision 5-Step Chain (`System 1 + System 2`) | DTRA v4.0 — RoboCasa Pure RGB-D Kitchen (`Hypertube Transport`) |
| :---: | :---: |
| ![DTRA v6.0 CALVIN Pure Vision 5-Step Chain](dtra/assets/calvin_v6_pure_vision_chain_preview.gif) | ![DTRA v4.0 RoboCasa Pure RGB-D Rollout](dtra/assets/robocasa_v4_pure_rgbd_preview.gif) |
| Canonical `5/5` sequence (`Chain #1`, `Mode: VISION`, `Leak: ZERO`, `Drift: 0.0000`, live System 2 recovery `CELL #108 -> #100`). 34 tasks / 30 chains: **`97.65%` RouteAcc**, **`2.37` mean chain len** (vs `1.90`), **`16.67%` 5/5 SR** (vs `6.67%`), **`+15.12%` jerk reduction**, **`9.30 ms`**. ([Scope Notes](dtra/outputs/calvin_v6/BENCHMARK_SCOPE_AND_LIMITATIONS.md)) | 12 tasks (`8` Atomic + `4` Composite, `0%` replay, no `object-state`, `Drift: 0.000000`): `Arm4_DTRA_v4_Hypertube` reaches **`29.17%` Atomic SR** (`OpenMicrowave 100.0%`), **`100.0%` RouteAcc**, **`1.928 ms`** (`Flat_MLP` & `ER_5%` collapse to `0.00%`). Composite `0.00%` motivated `v5.0`/`v6.0`. ([Results JSON](dtra/experiments/robocasa_v4/robocasa_continual_results_4arms.json)) |

---

## 1. Repository Structure

```text
drawing-vla/
├── README.md                              # Project landing page, 3-seed v3.0 tables & reproduction guide
├── RELEASE_NOTES_v3.0.0.md                # Official DTRA v3.0.0 release notes & Zenodo DOI (10.5281/zenodo.24297711)
├── RESEARCH_LINEAGE.md                    # Gen-1 -> Gen-2 research chronology & milestone ledger
├── requirements.txt                       # Core Python dependencies
│
├── v1_drawing_vla/                        # [Gen 1 Archive] B-Spline Action Chunking & ABK-VLA
│   ├── src/                               # Early B-Spline policies, evaluators & CL experiments
│   ├── handvsai/                          # HandVsAI crowdsourced scoring & sketch templates
│   └── paper_v1/                          # ABK-VLA preprint PDF & v1 architecture specs
│
├── dtra/                                  # [Gen 2 Main Research] Dynamic Topology Ring Architecture
│   ├── assets/                            # Lightweight README preview GIFs (< 8 MB, 640x360 @ 12 fps)
│   ├── PAPER_20P_OUTLINE.md               # Exact 20-page NeurIPS/CoRL 1-col structure & version isolation spec
│   ├── core/
│   │   ├── bspline_core.py                # Frozen clamped C^2 cubic B-Spline basis decoder (N_k=6 [2D/Meta-World] or num_cp=8 [7-DoF LIBERO] -> H=16)
│   │   ├── hypertube_transport.py         # [v4.0+ Track] Object-centric C^2 hypertube transport adapter
│   │   ├── sim_dtra_tube_dynamics.py      # HypertubeRingLayer, TwoStageDTRA & 6 invariant checks
│   │   └── topological_bridge.py          # C^2 geodesic bridge & myelinated transition graph core
│   ├── experiments/
│   │   ├── libero/                        # [v3.0] LIBERO-Goal (K=5) & LIBERO-100 (K=100) harness
│   │   ├── metaworld/                     # [v3.0] 8-task 3D Meta-World 3-seed 6-arm ablation & OOD suite
│   │   ├── calvin_v4/, calvin_v5/         # [Post-v3 Track] CALVIN 34-task System 1 & Z-gating anchors
│   │   └── robocasa_v4/                   # [Post-v3 Track] RoboCasa 12-task pure RGB-D kitchen CL
│   ├── outputs/
│   │   └── calvin_v6/                     # [Post-v3 Track] CALVIN v6 System 1+2 JSONs, HUD & scope notes
│   ├── visualization/
│   │   └── visualize_dtra_xai.py          # Generates 300 DPI architecture, OOD, manifold & telemetry figures
│   ├── README.md                          # DTRA package overview (English)
│   ├── README_KO.md                       # DTRA package overview (Korean)
│   └── paper_v2/
│       ├── DTRA_v3.0_Preprint.pdf         # Exact 20-page DTRA v3.0 preprint (XeTeX build, >=8.5pt tables)
│       ├── dtra_preprint.pdf              # Exact 20-page clean ASCII arXiv build (pdfLaTeX/XeTeX safe)
│       ├── dtra_v3.0_preprint.tex         # XeTeX 20-page LaTeX source
│       ├── dtra_preprint.tex              # Pure ASCII 20-page arXiv LaTeX source
│
├── outputs/
│   ├── libero/                            # Locked LIBERO-100 3-seed JSONs (42, 100, 2026) & LIBERO-Goal pilot
│   ├── metaworld/                         # Meta-World 3-seed & supplemental JSON results + checkpoints
│   ├── figures/                           # 300 DPI publication figures (fig0 .. fig8)
│   ├── calvin_v4/, calvin_v5/, calvin_v6/, organ_v4/, robocasa/  # [Post-v3 Track] Isolated A6000 archives
│   └── v1_archive/                        # Archived Gen-1 benchmark plots & JSON logs
│
├── scripts/
│   ├── build_dtra_v3_paper.py             # Generates 20-page v3.0 LaTeX + compiles both PDFs
│   ├── generate_libero100_figures.py      # Generates 300 DPI fig6, fig7 (3-seed error bars) & fig8
│   ├── summarize_libero_multiseed.py      # Prints finalized K=10..100 multi-seed Mean ± Std summary tables
│   └── sync_remote_dtra.py, check_remote_env.py
│
└── tests/                                 # Automated invariance & v3.0 module verification suite
```

---

## 2. Verified DTRA v3.0 / v3.1 Benchmark Results (`LIBERO-100` & `Meta-World` Only)

### A. LIBERO-100 (`libero_10 + libero_90`, $K=10 \to 100$) Multi-Seed & External CL Summary (Strict Encoder Parity)

All parity arms share identical frozen perception (`ResNet-18` DualCam + proprioception) and `LightweightTextEncoder` with DTRA Elastic peak VRAM of `2,713 ± 77 MB` (`n_eval=5` per task at each milestone). All four core arms cover **complete `N=3` training seeds `{42, 100, 2026}`**, complemented by matched-encoder published continual learning baselines (`DER++`, `L2P`, `O-LoRA`) on Seed `42`:

| Arm (Strict Parity) | $K=10$ SR | $K=30$ SR | $K=50$ SR | $K=70$ SR | **$K=100$ SR $\uparrow$** | FR $\downarrow$ | BWT $\uparrow$ | RouteAcc $\uparrow$ | Dup-Lang Route $\uparrow$ | Bases ($K=100$) | Active Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `arm_frozen_mlp` ($N=3$) | `82.00±4.32%` | `85.11±2.79%` | `79.07±0.50%` | `76.48±0.97%` | `70.33 ± 1.18%` | `2.90±1.05%` | `+0.20±2.29%` | `81.94±0.24%` | `41.76 ± 2.09%` | `100` | `6.46±0.57 ms` |
| `arm_er_5pct` ($N=3$) | `58.67±4.99%` | `56.67±6.69%` | `52.00±4.42%` | `47.43±4.07%` | `39.47 ± 1.48%` | `26.33±3.46%` | `-20.00±2.59%` | `---` | `---` | `1` | `4.93±0.59 ms` |
| `arm_dtra_rigid` ($N=3$) | `83.33±1.89%` | `90.44±0.83%` | `90.67±0.94%` | `87.52±0.71%` | `86.67 ± 0.66%` | `2.56±0.91%` | `+0.67±1.10%` | `99.91±0.13%` | `99.68 ± 0.45%` | `27` | `8.70±0.93 ms` |
| **`arm_dtra_elastic` ($N=3$, Full)** | `80.00±1.63%` | **`91.11±1.37%`** | **`90.67±1.00%`** | **`87.71±0.70%`** | **`87.00 ± 0.86%`** | **`2.69±0.67%`** | **`+0.20±0.72%`** | **`99.90±0.13%`** | **`99.68 ± 0.45%`** | **`27`** | `9.55±0.20 ms` |
| `DER++` (5% Buf. + Distill., Seed 42) | `68.00%` | `60.00%` | `60.40%` | `51.71%` | `38.40%` | `29.70%` | `-16.16%` | `---` | `---` | `1` | `5.49 ms` |
| `L2P` (Prompt Pool, Seed 42) | `8.00%` | `2.00%` | `1.60%` | `1.14%` | `1.20%` | `51.72%` | `-51.72%` | `---` | `---` | `32` | `7.79 ms` |
| `O-LoRA` (Ortho Subspace, Seed 42) | `2.00%` | `1.33%` | `2.00%` | `1.14%` | `1.00%` | `42.63%` | `-42.63%` | `---` | `---` | `100` | `7.05 ms` |

- **Per-Seed K=100 Breakdown (`N=3` Complete):**
  - `arm_dtra_elastic`: Seed 42 **`87.80%`**, Seed 100 **`87.40%`**, Seed 2026 **`85.80%`** $\rightarrow$ Mean **`87.00 ± 0.86%`** (`+16.67%p` SR and `+57.93%p` Dup-Lang routing over `arm_frozen_mlp`, `27` frozen shared bases = **73% motor basis compression**).
  - `arm_dtra_rigid`: Seed 42 `86.20%`, Seed 100 `86.20%`, Seed 2026 `87.60%` $\rightarrow$ Mean `86.67 ± 0.66%`.
  - `arm_frozen_mlp`: Seed 42 `72.00%`, Seed 100 `69.60%`, Seed 2026 `69.40%` $\rightarrow$ Mean `70.33 ± 1.18%`.
  - `arm_er_5pct`: Seed 42 `40.20%`, Seed 100 `40.80%`, Seed 2026 `37.40%` $\rightarrow$ Mean `39.47 ± 1.48%`.

#### Zero-Retrain Extended Initial-State Verification (`n_eval=20`, **12,000 Total Rollouts**)

Evaluating all three saved $K=100$ checkpoints (`Seeds 42, 100, 2026`) across `20` initial states/task (`6,000` rollouts/arm, `12,000` total; [`eval_libero100_extended_neval20.json`](outputs/libero/eval_libero100_extended_neval20.json)) confirms zero overfitting to `init[0:5]`:

| Saved $K=100$ Policy ($N=3$) | `init[0:5]` | **Unseen Held-Out `init[5:20]` [95% Wilson CI]** | **All `init[0:20]` [95% Wilson CI]** | **Dup-Lang SR (Route)** | Unique-Lang SR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `arm_frozen_mlp` (`6,000` rollouts) | `70.00 ± 0.75%` | `68.82 ± 0.74%` `[67.5, 70.2]` | `69.12 ± 0.74%` `[67.9, 70.3]` | `36.33%` (`41.80%`) | `83.17%` |
| **`arm_dtra_elastic` (`6,000` rollouts)** | `86.40 ± 0.86%` | **`84.80 ± 0.36%` `[83.7, 85.8]`** | **`85.20 ± 0.35%` `[84.3, 86.1]`** | **`89.06%` (`99.94%`)** | `83.55%` |
| **Gain ($\Delta$ DTRA $-$ MLP)** | **`+16.40%p`** | **`+15.98%p`** | **`+16.08%p`** | **`+52.72%p` (`+58.1%p`)** | `+0.38%p` |

### B. 3D Meta-World (`8` Tasks, 3-Seed Causal Ablation & External CL) & `LIBERO-Goal` (`5` Tasks)

| Benchmark | Arm / Mode | Clean / Mean SR | FR $\downarrow$ | RouteAcc | Severe OOD ($\sigma=0.20$) / Peak VRAM | Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Meta-World (8 Tasks, 3 Seeds)** | `arm4_frozen_mlp_multimodal` | `85.6 ± 3.1%` | `0.0 ± 0.0%` | `100.0%` | `23.6 ± 3.9%` OOD SR | [`ablate_dtra_scientific_results.json`](outputs/metaworld/ablate_dtra_scientific_results.json) |
| **Meta-World (8 Tasks, 3 Seeds)** | **`arm6_dtra_full_multimodal` (Feat-Reg)** | `72.2 ± 6.8%` | **`0.0 ± 0.0%`** | **`100.0%`** | **`51.4 ± 4.4%` OOD SR** (`1.87 ms` $O(1)$ CPU) | [`ablate_dtra_scientific_results.json`](outputs/metaworld/ablate_dtra_scientific_results.json) |
| **Meta-World Causal (3 Seeds)** | **`DTRA Full (No Feat-Reg, p=0, σ=0)`** | **`83.1 ± 5.1%`** | **`0.0 ± 0.0%`** | **`100.0%`** | **`45.0 ± 5.4%` OOD SR** | [`ablate_dtra_v31_final_results.json`](outputs/metaworld/ablate_dtra_v31_final_results.json) |
| **Meta-World Causal (3 Seeds)** | `Frozen MLP + Conv4 Feat-Reg` | `90.8 ± 3.0%` | `0.0 ± 0.0%` | `100.0%` | `49.2 ± 8.3%` OOD SR | [`ablate_dtra_v31_final_results.json`](outputs/metaworld/ablate_dtra_v31_final_results.json) |
| **Meta-World External CL (3 Seeds)** | `DER++` / `L2P` / `O-LoRA` | `73.3%` / `16.7%` / `7.2%` | `22.9%` / `79.4%` / `63.2%` | `---` | `13.6%` / `17.5%` / `9.4%` OOD SR | [`ablate_dtra_v31_final_results.json`](outputs/metaworld/ablate_dtra_v31_final_results.json) |
| **Meta-World Reverse ($T_7 \to T_0$, Seed 42)** | `arm6_dtra_full_multimodal` | `74.2%` | **`0.0%`** | **`100.0%`** | `64.2%` OOD SR | [`ablate_dtra_supplemental_results.json`](outputs/metaworld/ablate_dtra_supplemental_results.json) |
| **LIBERO-Goal Pilot (5 Tasks, Seed 42)** | **`arm_dtra_elastic`** | **`92.0%`** | `5.0%` | **`100.0%`** | `3,696 MB` VRAM (`post_snap_norm = 0.0`) | [`libero_goal_pilot_4arms.json`](outputs/libero/libero_goal_pilot_4arms.json) |


### Evaluation scope and resource interpretation

All four core LIBERO-100 arms (`Frozen MLP`, `ER 5%`, `DTRA Rigid`, `DTRA Elastic`) cover three training seeds (`42, 100, 2026`); mid-stream checkpoints use `n_eval=5` rollouts/task, while final $K=100$ generalization for `DTRA Elastic` vs. `Frozen MLP` is verified across all three seeds over `n_eval=20` initial states (`12,000` rollouts total, including `15` unseen held-out initial states `init[5:20]` with pooled 95% Wilson CIs). Matched-encoder external baselines (`DER++`, `L2P`, `O-LoRA`) evaluate Seed 42 on LIBERO-100 and all three seeds on Meta-World.

Parameter freezing and temporary state reset are separate: `post_snap_norm = 0.0000` records a reset state, not an independent before/after weight comparison. LIBERO-100 empirical success-rate forgetting is **2.69 ± 0.67%** (`DTRA Elastic`) and **2.56 ± 0.91%** (`DTRA Rigid`).

The **73% reduction concerns motor basis count**, from one basis per task to 27 shared bases at K=100. Recorded policy parameters occupy **183.14 MB** for DTRA versus **151.67 MB** for Frozen MLP, excluding the common frozen backbone.

On Meta-World, our three-seed causal ablation (`ablate_dtra_v31_final.py`) isolates the clean-versus-OOD trade-off: removing Conv4 feature-noise regularization (`p=0, σ=0`) lifts DTRA Full from `72.2 ± 6.8%` to **`83.1 ± 5.1%` clean SR** while preserving **`45.0 ± 5.4%` severe OOD SR** (nearly 2× unregularized Frozen MLP's `23.6 ± 3.9%`). Broad retrieval remains $O(K)$, with a constant-cost selected decoding core (`9.55 ± 0.20 ms` GPU policy inference at $K=100$).

### C. Post-v3 Embodied Continual Tracks (`DTRA v4.0 – v6.0`, Separate from v3.0 Preprint)

| Track / Benchmark | Arm / Mode | Key Metrics | Drift | Source |
| :--- | :--- | :--- | :---: | :--- |
| **v4.0 RoboCasa (`12` Tasks: `8` Atomic + `4` Composite, `0%` Replay)** | `Arm4_DTRA_v4_Hypertube` (vs `Flat_MLP`/`ER_5%` `0.00%`) | Atomic SR **`29.17%`** (`OpenMicrowave 100.0%`), Overall `19.44%`, RouteAcc **`100.0%`**, BWT `+0.00%`, `1.928 ms` *(Composite `0.00%` $\rightarrow$ System 2 motivation)* | `0.000000` | [`robocasa_continual_results_4arms.json`](dtra/experiments/robocasa_v4/robocasa_continual_results_4arms.json) |
| **v5.0 + v6.0 CALVIN (`34` Tasks, `30` 5-Step Chains)** | `Pure Vision` (`Leak: ZERO`) | RouteAcc **`97.65%`**, Single SR `55.88%`, 5-Chain Len **`2.37`** *(vs `1.90`)*, 5/5 SR **`16.67%`** *(vs `6.67%`)*, Jerk **`+15.12%`**, `22` edges, `9.30 ms` | **`0.0000`** | [`calvin_v6_vision_34tasks.json`](dtra/outputs/calvin_v6/calvin_v6_vision_34tasks.json) |
| **v5.0 + v6.0 CALVIN (`34` Tasks, `30` 5-Step Chains)** | `Oracle Anchor` *(Upper-Bound Ref)* | RouteAcc `97.65%`, Single SR `46.47%`, 5-Chain Len `2.53` *(vs `1.90`)*, 5/5 SR `16.67%` *(vs `10.00%`)*, Jerk `+15.24%`, `25` edges, `8.49 ms` | `0.0000` | [`calvin_v6_oracle_34tasks.json`](dtra/outputs/calvin_v6/calvin_v6_oracle_34tasks.json) |

---

## 3. Reproduction and Code Availability

Core DTRA modules, unit tests, figure/paper build scripts, locked result JSONs, and the Meta-World / LIBERO / post-v3 benchmark scripts are included in this repository. Installing `requirements.txt` covers the core modules, unit tests, and figure/LaTeX regeneration scripts; full simulator rollouts additionally require installing the external simulation packages (`metaworld`, `mujoco`, `gymnasium` for Meta-World; `libero`, `robosuite`, `bddl`, `h5py`, and LIBERO HDF5 demos for LIBERO-100). Use `python scripts/check_remote_env.py` to inspect which simulator dependencies are available in your environment.

```bash
# 0. Install core dependencies, inspect environment & run invariant unit tests
pip install -r requirements.txt
python scripts/check_remote_env.py --env dtra
python -m unittest discover -s tests -v
python dtra/tests/test_topological_bridge.py

# 1. Step-1 Hypertube manifold invariants (6 automated checks)
python dtra/core/sim_dtra_tube_dynamics.py

# 2. 3D Meta-World 8-task 3-seed scientific ablation & supplemental stress suite (requires metaworld/mujoco)
python -m dtra.experiments.metaworld.ablate_dtra_scientific --seeds 42 100 2026 --epochs 80
python -m dtra.experiments.metaworld.ablate_dtra_supplemental --seed 42 --epochs 80

# 3. LIBERO-100 (K=100) continual scaling stream (requires `libero` conda env + LIBERO HDF5 demos)
python -m dtra.experiments.libero.benchmark_dtra_libero \
  --suites libero_10,libero_90 --max-tasks 100 \
  --stream-tasks --eval-milestones 10,30,50,70,100 \
  --epochs 25 --batch-size 128 --n-eval 5 --seed 42 \
  --out-json outputs/libero/benchmark_dtra_libero_10_libero_90_k100.json

# 4. Aggregate 3-seed tables, regenerate 300 DPI figures & compile 20-page DTRA v3.0 PDFs
python scripts/summarize_libero_multiseed.py
python scripts/generate_libero100_figures.py
python scripts/build_dtra_v3_paper.py
```

---

## 4. Visual Figures (`outputs/figures/`) & Preprints (`dtra/paper_v2/`)

- **23-Page Final Preprints & arXiv Source Bundle:** [`dtra/paper_v2/DTRA_v3.0_Preprint.pdf`](dtra/paper_v2/DTRA_v3.0_Preprint.pdf) | [`dtra/paper_v2/dtra_preprint.pdf`](dtra/paper_v2/dtra_preprint.pdf) (Clean ASCII arXiv, 23p) | [`dtra/paper_v2/dtra_preprint.tex`](dtra/paper_v2/dtra_preprint.tex) | [`dtra_arxiv_submission.tar.gz`](dtra/paper_v2/dtra_arxiv_submission.tar.gz)
- **v3.0 Scaling & Architecture Figures:** [`fig6_dtra_v3_architecture_schematic.png`](outputs/figures/fig6_dtra_v3_architecture_schematic.png) | [`fig7_libero100_scaling_curves.png`](outputs/figures/fig7_libero100_scaling_curves.png) | [`fig8_libero100_basis_and_callstack.png`](outputs/figures/fig8_libero100_basis_and_callstack.png)
- **Meta-World XAI Figures:** [`fig0`](outputs/figures/fig0_dtra_architecture_schematic.png), [`fig1`](outputs/figures/fig1_ood_perturbation_curves.png), [`fig2`](outputs/figures/fig2_manifold_elastic_snapback.png), [`fig3`](outputs/figures/fig3_xai_telemetry_rollout.png), [`fig4`](outputs/figures/fig4_cl_forgetting_heatmap_and_bars.png), [`fig5`](outputs/figures/fig5_o1_latency_scaling.png)

---

## 5. Citation & License

```bibtex
@misc{son2026dtra,
  author       = {Son, Seungjoon},
  title        = {{DTRA}: Dynamic Topology Ring Architecture for Replay-Free Continual Visuomotor Control via Hyperspherical Isolation and Shared Motor Bases},
  year         = {2026},
  month        = nov,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24297711},
  url          = {https://doi.org/10.5281/zenodo.24297711},
  note         = {v3.0.0 20-page Preprint; Concept DOI: 10.5281/zenodo.24218482; prior v2.1.0: 10.5281/zenodo.24218483; Repository: \url{https://github.com/5Color/dtra}}
}

@misc{son2026abkvla,
  author       = {Son, Seungjoon},
  title        = {{ABK-VLA}: Task-ID-Free Continual Visuomotor Policy Learning via Autonomous B-Spline Knot Compression and Visual Prototype Routing},
  year         = {2026},
  month        = nov,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24212130},
  url          = {https://doi.org/10.5281/zenodo.24212130},
  note         = {v1.1.3; v1.1.1: 10.5281/zenodo.24208545; Concept DOI: 10.5281/zenodo.24208544}
}
```

Released under the [MIT License](LICENSE).
