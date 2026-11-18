# DTRA v3.0.0 Release Notes — *Dynamic Topology Ring Architecture*

[![Preprint (20 Pages)](https://img.shields.io/badge/Preprint-DTRA__v3.0__Preprint.pdf-b31b1b.svg)](dtra/paper_v2/DTRA_v3.0_Preprint.pdf)
[![arXiv Build (20 Pages)](https://img.shields.io/badge/arXiv%20Clean-dtra__preprint.pdf-9e1c20.svg)](dtra/paper_v2/dtra_preprint.pdf)
[![Zenodo v3.0 DOI Slot](https://img.shields.io/badge/Zenodo%20v3.0-Pending%20DOI%20Minting-informational.svg)](https://doi.org/10.5281/zenodo.24218483)

This release (`v3.0.0`) publishes the finalized **20-page single-column research preprint** ([`DTRA_v3.0_Preprint.pdf`](dtra/paper_v2/DTRA_v3.0_Preprint.pdf) and clean ASCII [`dtra_preprint.pdf`](dtra/paper_v2/dtra_preprint.pdf)) together with **LIBERO-100 results with three seeds for Frozen MLP and DTRA Elastic and two seeds for ER 5% and DTRA Rigid, plus three-seed Meta-World results** under strict matched-encoder parity.

---

## Highlights & Architectural Contributions

### 1. 20-Page Preprint (`DTRA v3.0`, Exact 20 Pages, Zero Overfulls & Zero Underfulls)
- **Structure (`dtra/PAPER_20P_OUTLINE.md`):**
  - **Pages 1–9 (Main Body):** Introduction (`1–2`), Related Work & Mathematical Preliminaries (`3–4`), Method (`5–6`), Experiments & Continual Scaling Ablations (`7–8`), Discussion, Scope Boundaries & Conclusion (`9`).
  - **Pages 10–11 (References):** Full bibliography (`[1]`–`[50]`).
  - **Pages 12–20 (Technical Appendices A–D, High-Readability 9p):** Appendix A (`12–13`: Zero Residual Drift, $C^2$ B-Spline Smoothness, Isometric Zero-Padding Inner-Product Invariance + Fig. 5 $O(1)$ Latency), Appendix B (`14–16`: Seed Registry, LIBERO-100 $K=10\to100$ Milestones, LIBERO-Goal Pilot, `L10` vs. `L90` Split, FWT & Peak VRAM Resources, Hyperparameters, Dup-Lang/Scene Stratification & `L10` Breakdown), Appendix C (`17–18`: 3-Seed Meta-World Per-Task, Supplemental CL & Reverse Order, 9-Regime OOD Stress, Snap-Back Telemetry + Fig. 6 Heatmaps), Appendix D (`19–20`: Fig. 7 SharedBasisBank Diagnostics, 12-Row White-Box Routing Call Stack Log, Fig. 8 Closed-Loop B-Spline Rollout Telemetry & Surgical Fault Isolation).
- **Layout Verification:** Verified at **exact 20 pages** (`Overfull \hbox = 0`, `Underfull \hbox = 0`, `Undefined = 0`, `\linespread{1.05}`, `\arraystretch{1.2}`, all tables $\ge 8.5\text{pt}$ without `\resizebox` shrinking) on both [`DTRA_v3.0_Preprint.pdf`](dtra/paper_v2/DTRA_v3.0_Preprint.pdf) and clean ASCII arXiv [`dtra_preprint.pdf`](dtra/paper_v2/dtra_preprint.pdf) (`non_ascii_lines=0`).

### 2. Core Algorithmic Mechanisms (`dtra/`)
1. **No-Encroachment Key Constraint:** Caches prior raw anchors $\mu_j$ and floors $s_{jj}$ to project out repulsive updates that invade earlier Voronoi territories on $\mathbb{S}^{63}$.
2. **Two-Stage Top-$M$ Common-Mode Cancellation (CMC):** Stage 1 retrieves Top-$M$ ($M=5$) candidate Regions; Stage 2 subtracts shared language/visual background to boost 3-seed $K=100$ `Dup-Lang` routing from `41.76 ± 2.09%` (`Frozen MLP`) to **`99.68 ± 0.45%`**.
3. **SharedBasisBank + Strict Parameter Freeze ($\nabla_\theta = 0$):** Compresses $K=100$ tasks into **`27` frozen shared cubic B-Spline primitive bases** (**73% basis reduction**) with explicit optimizer exclusion (`0.0` AdamW silent drift).
4. **Two-Timescale Snap-Back:** Decouples intra-episode chunk reset (`chunk_snap_back()`) from episode-end refractory reset (`episode_snap_back()`, exact `post_snap_norm = 0.0000`).

---

## Finalized 3-Seed Benchmark Verification (`v3.0` Scope)

### LIBERO-100 (`libero_10 + libero_90`, $K=100$, Seeds `42, 100, 2026`, `n_eval=5`)

| Arm (Strict Encoder Parity) | Seed 42 | Seed 100 | Seed 2026 | Mean SR $\uparrow$ | FR $\downarrow$ | BWT $\uparrow$ | RouteAcc $\uparrow$ | Dup-Lang $\uparrow$ | Active Bases |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `arm_frozen_mlp` ($N=3$) | `72.00%` | `69.60%` | `69.40%` | `70.33 ± 1.18%` | `2.90 ± 1.05%` | `+0.20 ± 2.29%` | `81.94 ± 0.24%` | `41.76 ± 2.09%` | `100` (`151.67 MB`) |
| `arm_er_5pct` ($N=2$) | `40.20%` | `40.80%` | --- | `40.50 ± 0.30%` | `27.07 ± 4.04%` | `-20.71 ± 2.93%` | `1.00 ± 0.00%` | `0.00 ± 0.00%` | `1` (`1.70 MB`) |
| `arm_dtra_rigid` ($N=2$) | `86.20%` | `86.20%` | --- | `86.20 ± 0.00%` | `2.93 ± 0.91%` | `+0.10 ± 0.91%` | `99.86 ± 0.14%` | `99.53 ± 0.47%` | `27` (`183.14 MB`) |
| **`arm_dtra_elastic` ($N=3$, Full)** | **`87.80%`** | **`87.40%`** | **`85.80%`** | **`87.00 ± 0.86%`** | **`2.69 ± 0.67%`** | **`+0.20 ± 0.72%`** | **`99.90 ± 0.13%`** | **`99.68 ± 0.45%`** | **`27` (`183.14 MB`)** |

* **Milestone Trajectory (`arm_dtra_elastic`, $N=3$):** `80.00±1.63%` ($K=10$, `7` bases) $\to$ **`91.11±1.37%`** ($K=30$, `13` bases) $\to$ **`90.67±1.00%`** ($K=50$, `22` bases) $\to$ **`87.71±0.70%`** ($K=70$, `24` bases) $\to$ **`87.00±0.86%`** ($K=100$, `27` bases; `2,713 ± 77 MB` Peak VRAM; `9.55 ± 0.20 ms` active latency).
* **Meta-World (8 Tasks, 3 Seeds) & LIBERO-Goal (5 Tasks):** Preserves **`0.0 ± 0.0%` forgetting**, **`100.0%` routing**, `O(1)` **`1.87 ms`** CPU latency, and **`51.4 ± 4.4%` severe OOD SR** on Meta-World (vs. `23.6%` Frozen MLP), and **`92.0%` Mean SR** (`100.0%` routing, `post_snap_norm = 0.0`) on LIBERO-Goal.

---

## Strict Version Isolation (`v3.0` vs. `v4.0` Future Work Track)

Only single-ring `Region` implementations (`K=100` `LIBERO-100`, `LIBERO-Goal`, and 3D `Meta-World`) enter the **DTRA v3.0** preprint and release tables. Multi-ring `Organ` scaling (`K=5,000`), `CALVIN` V4/V5 (`ABCD -> D`), and `RoboCasa` kitchen manipulation benchmarks are strictly isolated under future-work archive paths (`outputs/organ_v4/`, `outputs/calvin_v4/`, `outputs/calvin_v5/`, `outputs/robocasa/`).

---

## Verification and Code Availability

The public release contains preprints, LaTeX sources, figures, and locked result JSONs. The benchmark harness and reproduction dependencies are not yet fully included. Experimental commands below describe the local research checkout and are not a complete setup recipe for the public checkout.

```bash
python -m unittest discover -s tests -v
python scripts/summarize_libero_multiseed.py
python scripts/generate_libero100_figures.py
python scripts/build_dtra_v3_paper.py
```

**Release Assets Checklist:**
- Preprint PDFs: `dtra/paper_v2/DTRA_v3.0_Preprint.pdf` (20p), `dtra/paper_v2/dtra_preprint.pdf` (20p arXiv clean)
- Locked Checkpoints: `outputs/libero/benchmark_dtra_libero_10_libero_90_k100_seed{42,100,2026}_locked.json`
- Zenodo DOI Slot (for release minting): Replace `Pending DOI Minting` badge in `README.md` / `dtra/README.md` upon upload to Zenodo (`prior DOI: 10.5281/zenodo.24218483`).
## Editorial revision (2026-11-17)

The title and abstract now describe replay-free continual control, without a SOTA ranking claim or epigraph. Parameter freezing, temporary state reset, and empirical success retention are distinguished. Memory claims refer to observed basis counts and measured parameter footprints; no logarithmic growth or future break-even point is claimed. Clean/OOD trade-offs and the five-rollout evaluation limit are stated explicitly. Both PDF assets are refreshed without changing the benchmark JSONs or figures. The v3.0.0 tag remains at its original release commit; refreshed editorial assets correspond to the current main revision.
