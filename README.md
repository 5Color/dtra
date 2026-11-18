# DTRA: Dynamic Topology Ring Architecture

[English](README.md) | [한국어](README_KO.md)

[![Preprint (DTRA v3.0)](https://img.shields.io/badge/Preprint-DTRA__v3.0__Preprint.pdf-b31b1b.svg)](dtra/paper_v2/DTRA_v3.0_Preprint.pdf)
[![arXiv Manuscript](https://img.shields.io/badge/arXiv%20Clean-dtra__preprint.pdf-9e1c20.svg)](dtra/paper_v2/dtra_preprint.pdf)
[![Zenodo v3.0 DOI Slot](https://img.shields.io/badge/Zenodo%20v3.0-Pending%20DOI%20Minting-informational.svg)](RELEASE_NOTES_v3.0.0.md)
[![Zenodo DTRA Archive](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24218483-blue)](https://doi.org/10.5281/zenodo.24218483)
[![Zenodo ABK-VLA](https://img.shields.io/badge/Gen1%20DOI-10.5281%2Fzenodo.24208545-lightgrey)](https://doi.org/10.5281/zenodo.24208545)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

**Replay-free, Task-ID-free continual visuomotor control with hyperspherical routing and shared motor bases: 87.00 ± 0.86% final LIBERO-100 success across three training seeds.**

> **Repository Separation Note:** This repository tracks the full **ARIN $\to$ Drawing-VLA $\to$ ABK-VLA $\to$ DTRA v3.0–v6.0** research lineage across two generations:
> 1. **Gen 1 — `v1_drawing_vla` (`Drawing-VLA` v1.1.2 & `ABK-VLA` v1.1.3):** Clamped cubic B-Spline action chunking ($N_k=6, H=16$), geometric DoG prototype routing, and continual 2D stroke benchmarks.
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
├── RELEASE_NOTES_v3.0.0.md                # Official DTRA v3.0.0 release notes & Zenodo DOI slot
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
│   │   ├── bspline_core.py                # Frozen clamped C^2 cubic B-Spline basis decoder (N_k=6 -> H=16)
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

## 2. Verified DTRA v3.0 Benchmark Results (`LIBERO-100` & `Meta-World` Only)

### A. LIBERO-100 (`libero_10 + libero_90`, $K=10 \to 100$) Multi-Seed Summary (Strict Encoder Parity)

All parity arms share identical frozen perception (`ResNet-18` DualCam + proprioception) and `LightweightTextEncoder` with DTRA Elastic peak VRAM of `2,713 ± 77 MB` (`n_eval=5` per task at each milestone; ER 5% peaks at approximately `4,772 MB`). Finalized `_locked.json` checkpoints cover full 4-arm runs on Seeds `{42, 100}` ($N=2$) and 2-arm core parity (`arm_frozen_mlp`, `arm_dtra_elastic`) on Seeds `{42, 100, 2026}` ($N=3$):

| Arm (Strict Parity) | $K=10$ SR | $K=30$ SR | $K=50$ SR | $K=70$ SR | **$K=100$ SR $\uparrow$** | FR $\downarrow$ | BWT $\uparrow$ | RouteAcc $\uparrow$ | Dup-Lang Route $\uparrow$ | Bases ($K=100$) | Active Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `arm_frozen_mlp` ($N=3$) | `82.00±4.32%` | `85.11±2.79%` | `79.07±0.50%` | `76.48±0.97%` | `70.33 ± 1.18%` | `2.90±1.05%` | `+0.20±2.29%` | `81.94±0.24%` | `41.76 ± 2.09%` | `100.0` | `6.46±0.57 ms` |
| `arm_er_5pct` ($N=2$) | `62.00±2.00%` | `59.67±6.33%` | `54.80±2.40%` | `49.14±4.00%` | `40.50 ± 0.30%` | `27.07±4.04%` | `-20.71±2.93%` | `1.00±0.00%` | `0.00 ± 0.00%` | `1.0` | `5.34±0.03 ms` |
| `arm_dtra_rigid` ($N=2$) | `84.00±2.00%` | `90.33±1.00%` | `91.00±1.00%` | `87.14±0.57%` | `86.20 ± 0.00%` | `2.93±0.91%` | `+0.10±0.91%` | `99.86±0.14%` | `99.53 ± 0.47%` | `27.0` | `9.36±0.05 ms` |
| **`arm_dtra_elastic` ($N=3$, Full)** | `80.00±1.63%` | **`91.11±1.37%`** | **`90.67±1.00%`** | **`87.71±0.70%`** | **`87.00 ± 0.86%`** | **`2.69±0.67%`** | **`+0.20±0.72%`** | **`99.90±0.13%`** | **`99.68 ± 0.45%`** | **`27.0`** | `9.55±0.20 ms` |

- **Per-Seed K=100 Breakdown:**
  - `arm_dtra_elastic`: Seed 42 **`87.80%`**, Seed 100 **`87.40%`**, Seed 2026 **`85.80%`** $\rightarrow$ Mean **`87.00 ± 0.86%`** (`+16.67%p` SR and `+57.93%p` Dup-Lang routing over `arm_frozen_mlp`, `27` frozen shared bases = **73% motor basis compression**).
  - `arm_frozen_mlp`: Seed 42 `72.00%`, Seed 100 `69.60%`, Seed 2026 `69.40%` $\rightarrow$ Mean `70.33 ± 1.18%`.

### B. 3D Meta-World (`8` Tasks, 3-Seed Mean ± Std) & `LIBERO-Goal` (`5` Tasks)

| Benchmark | Arm / Mode | Clean / Mean SR | FR $\downarrow$ | RouteAcc | Severe OOD ($\sigma=0.20$) / Peak VRAM | Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Meta-World (8 Tasks, 3 Seeds)** | `arm4_frozen_mlp_multimodal` | `85.6 ± 3.1%` | `0.0 ± 0.0%` | `100.0%` | `23.6 ± 3.9%` OOD SR | [`ablate_dtra_scientific_results.json`](outputs/metaworld/ablate_dtra_scientific_results.json) |
| **Meta-World (8 Tasks, 3 Seeds)** | **`arm6_dtra_full_multimodal`** | `72.2 ± 6.8%` | **`0.0 ± 0.0%`** | **`100.0%`** | **`51.4 ± 4.4%` OOD SR** (`1.87 ms` $O(1)$ CPU) | [`ablate_dtra_scientific_results.json`](outputs/metaworld/ablate_dtra_scientific_results.json) |
| **Meta-World Reverse ($T_7 \to T_0$, Seed 42)** | `arm6_dtra_full_multimodal` | `74.2%` | **`0.0%`** | **`100.0%`** | `64.2%` OOD SR | [`ablate_dtra_supplemental_results.json`](outputs/metaworld/ablate_dtra_supplemental_results.json) |
| **LIBERO-Goal Pilot (5 Tasks, Seed 42)** | **`arm_dtra_elastic`** | **`92.0%`** | `5.0%` | **`100.0%`** | `3,696 MB` VRAM (`post_snap_norm = 0.0`) | [`libero_goal_pilot_4arms.json`](outputs/libero/libero_goal_pilot_4arms.json) |


### Evaluation scope and resource interpretation

Frozen MLP and DTRA Elastic use seeds 42, 100, and 2026; ER 5% and DTRA Rigid use seeds 42 and 100. Each task is evaluated with five rollouts per seed using the same first five provided initial states. Cross-seed standard deviations are not confidence intervals over a broader initial-state distribution.

Parameter freezing and temporary state reset are separate: `post_snap_norm = 0.0000` records a reset state, not an independent before/after weight comparison. LIBERO-100 success-rate forgetting is **2.69 ± 0.67%**. Its sources have not been separated into routing changes and rollout variability.

The **73% reduction concerns motor basis count**, from one basis per task to 27 shared bases at K=100. Recorded policy parameters occupy **183.14 MB** for DTRA versus **151.67 MB** for Frozen MLP, excluding the common frozen backbone. The experiments do not establish an asymptotic memory-growth law or a break-even task count beyond K=100.

On Meta-World, DTRA Full has lower clean success (**72.2 ± 6.8%** vs. **85.6 ± 3.1%**) and higher severe OOD success (**51.4 ± 4.4%** vs. **23.6 ± 3.9%**) than multimodal Frozen MLP. This is an observed trade-off, not a demonstrated causal explanation. Registration involves training; the **9.55 ± 0.20 ms** LIBERO figure measures policy inference. Broad retrieval remains O(K), with a constant-cost selected decoding core.

### C. Post-v3 Embodied Continual Tracks (`DTRA v4.0 – v6.0`, Separate from v3.0 Preprint)

| Track / Benchmark | Arm / Mode | Key Metrics | Drift | Source |
| :--- | :--- | :--- | :---: | :--- |
| **v4.0 RoboCasa (`12` Tasks: `8` Atomic + `4` Composite, `0%` Replay)** | `Arm4_DTRA_v4_Hypertube` (vs `Flat_MLP`/`ER_5%` `0.00%`) | Atomic SR **`29.17%`** (`OpenMicrowave 100.0%`), Overall `19.44%`, RouteAcc **`100.0%`**, BWT `+0.00%`, `1.928 ms` *(Composite `0.00%` $\rightarrow$ System 2 motivation)* | `0.000000` | [`robocasa_continual_results_4arms.json`](dtra/experiments/robocasa_v4/robocasa_continual_results_4arms.json) |
| **v5.0 + v6.0 CALVIN (`34` Tasks, `30` 5-Step Chains)** | `Pure Vision` (`Leak: ZERO`) | RouteAcc **`97.65%`**, Single SR `55.88%`, 5-Chain Len **`2.37`** *(vs `1.90`)*, 5/5 SR **`16.67%`** *(vs `6.67%`)*, Jerk **`+15.12%`**, `22` edges, `9.30 ms` | **`0.0000`** | [`calvin_v6_vision_34tasks.json`](dtra/outputs/calvin_v6/calvin_v6_vision_34tasks.json) |
| **v5.0 + v6.0 CALVIN (`34` Tasks, `30` 5-Step Chains)** | `Oracle Anchor` *(Upper-Bound Ref)* | RouteAcc `97.65%`, Single SR `46.47%`, 5-Chain Len `2.53` *(vs `1.90`)*, 5/5 SR `16.67%` *(vs `10.00%`)*, Jerk `+15.24%`, `25` edges, `8.49 ms` | `0.0000` | [`calvin_v6_oracle_34tasks.json`](dtra/outputs/calvin_v6/calvin_v6_oracle_34tasks.json) |

---

## 3. Reproduction and Code Availability

The public release contains preprints, LaTeX sources, figures, and locked result JSONs. The benchmark harness and reproduction dependencies are not yet fully included. Experimental commands below describe the local research checkout and are not a complete setup recipe for the public checkout.

```bash
# 0. Install dependencies & run invariant unit tests
pip install -r requirements.txt
python -m unittest discover -s tests -v

# 1. Step-1 Hypertube manifold invariants (6 automated checks)
python dtra/core/sim_dtra_tube_dynamics.py

# 2. 3D Meta-World 8-task 3-seed scientific ablation & supplemental stress suite
python -m dtra.experiments.metaworld.ablate_dtra_scientific --seeds 42 100 2026 --epochs 80
python -m dtra.experiments.metaworld.ablate_dtra_supplemental --seed 42 --epochs 80

# 3. LIBERO-100 (K=100) continual scaling stream (inside `libero` conda env)
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

- **20-Page Preprints (`9p + 2p + 9p`):** [`dtra/paper_v2/DTRA_v3.0_Preprint.pdf`](dtra/paper_v2/DTRA_v3.0_Preprint.pdf) | [`dtra/paper_v2/dtra_preprint.pdf`](dtra/paper_v2/dtra_preprint.pdf) (Clean arXiv ASCII, 20p) | [`dtra/paper_v2/dtra_preprint.tex`](dtra/paper_v2/dtra_preprint.tex)
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
  doi          = {10.5281/zenodo.24218483},
  url          = {https://doi.org/10.5281/zenodo.24218483},
  note         = {Repository: \url{https://github.com/5Color/dtra} (v3.0.0 20-page Preprint)}
}

@misc{son2026abkvla,
  author       = {Son, Seungjoon},
  title        = {{ABK-VLA}: Task-ID-Free Continual Visuomotor Policy Learning via Autonomous B-Spline Knot Compression and Visual Prototype Routing},
  year         = {2026},
  month        = nov,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24208545},
  url          = {https://doi.org/10.5281/zenodo.24208545}
}
```

Released under the [MIT License](LICENSE).
