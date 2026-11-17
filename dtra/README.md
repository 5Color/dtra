# DTRA: Dynamic Topology Ring Architecture for Replay-Free Continual Visuomotor Control via Hyperspherical Isolation and Shared Motor Bases

[![Preprint (DTRA v3.0, 20p)](https://img.shields.io/badge/Preprint-DTRA__v3.0__Preprint.pdf-b31b1b.svg)](paper_v2/DTRA_v3.0_Preprint.pdf)
[![arXiv Manuscript (20p)](https://img.shields.io/badge/arXiv%20Clean-dtra__preprint.pdf-9e1c20.svg)](paper_v2/dtra_preprint.pdf)
[![Zenodo v3.0 DOI Slot](https://img.shields.io/badge/Zenodo%20v3.0-Pending%20DOI%20Minting-informational.svg)](../RELEASE_NOTES_v3.0.0.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24218483.svg)](https://doi.org/10.5281/zenodo.24218483)
[![LaTeX Source](https://img.shields.io/badge/LaTeX-dtra__preprint.tex-008080.svg)](paper_v2/dtra_preprint.tex)
[![Prior Work: ABK-VLA](https://img.shields.io/badge/Prior%20Work-5Color%2Fabk--vla-2969d8.svg)](https://github.com/5Color/abk-vla)
[![ABK-VLA DOI](https://img.shields.io/badge/ABK--VLA%20DOI-10.5281%2Fzenodo.24208545-blue.svg)](https://doi.org/10.5281/zenodo.24208545)
[![License: MIT](https://img.shields.io/badge/License-MIT-108a43.svg)](../LICENSE)

Research materials and 20-page preprint manuscripts for **DTRA v3.0** (*Dynamic Topology Ring Architecture*, corresponding to [`5Color/dtra`](https://github.com/5Color/dtra)), a geometry-first, replay-free, and Task-ID-free continual visuomotor control framework.

> **Repository & Scope Separation:** This `dtra/` package houses **Generation 2 (`DTRA v3.0`)** verified on 3-seed 3D Meta-World (`K=8`) and 3-seed 7-DoF LIBERO-100 (`K=100`). Generation 1 (`ABK-VLA`) lives in [`../v1_drawing_vla/`](../v1_drawing_vla/), while subsequent **v4.0 Future Work** research (`Organ` $K=5{,}000$, `CALVIN` V4/V5, `RoboCasa`) is strictly separated in [`PAPER_20P_OUTLINE.md`](PAPER_20P_OUTLINE.md) and isolated under `experiments/{calvin_v4,calvin_v5,robocasa_v4}/`.

---

## 1. Architecture Overview (`v3.0` Single-Ring Region)

![DTRA v3.0 End-to-End Architecture](../outputs/figures/fig6_dtra_v3_architecture_schematic.png)

Scaling continual visuomotor control from $K=8$ Meta-World tasks to $K=100$ sequential 7-DoF LIBERO manipulation tasks (`libero_10 + libero_90`, including 30 tasks across 13 identical-instruction groups) exposes **Voronoi encroachment** and **$O(K)$ action-head parameter growth**. Under strict matched-encoder parity (`Frozen ResNet-18` DualCam + proprioception + `LightweightTextEncoder`), **DTRA v3.0** resolves both via four mechanisms:

1. **No-Encroachment Margin Refinement:** Caches prior task raw query anchors $\mu_j$ and self-similarity floors $s_{jj} = \cos(\mu_j, \mathbf{k}_j)$, projecting out repulsive key updates that would steal routing territory from prior Voronoi cells $0 \dots i-1$ on $\mathbb{S}^{63}$.
2. **Two-Stage Top-$M$ Common-Mode Cancellation (CMC):** Stage 1 retrieves Top-$M$ ($M=5$) candidates via $O(K)$ multimodal cosine scan; Stage 2 subtracts the candidate set's shared language/scene mean to isolate fine DualCam spatial grids ($4 \times 4 / 2 \times 2$ + DoG) and proprioceptive residuals, lifting $K=100$ `Dup-Lang` routing accuracy across 3 seeds from **41.76%** to **99.68 ± 0.45%**.
3. **SharedBasisBank + Zero AdamW Silent Drift:** Factorizes $H=16$ action chunks via clamped $C^2$ cubic B-Spline knots ($N_k=6$) into a dynamically shared library of frozen primitive bases, compressing $K=100$ tasks into **27 bases** (**73% basis reduction**) while enforcing `requires_grad=False` and optimizer exclusion ($L_\infty\text{ drift} = 0.0$).
4. **Two-Timescale Snap-Back:** Separates intra-episode chunk reset (`chunk_snap_back()`, $\Delta \mathbf{z}_i \leftarrow \mathbf{0}$) from episode-end refractory reset (`episode_snap_back()`, `post_snap_norm = 0.0000`) while retaining progress traces to prevent long-horizon perseveration.

---

## 2. Finalized Benchmark Results (`v3.0` Preprint Scope Only)

### A. LIBERO-100 ($K=10 \to 100$) Multi-Seed Continual Scaling (Seeds `42, 100, 2026`, Strict Encoder Parity)

![LIBERO-100 Scaling Curves](../outputs/figures/fig7_libero100_scaling_curves.png)

All comparison arms share identical frozen perception and `LightweightTextEncoder`, streaming tasks with DTRA Elastic peak VRAM of `2,713 ± 77 MB` (`n_eval = 5` rollouts per milestone checkpoint; full 4-arm suite on Seeds `{42, 100}` [$N=2$] and 2-arm core parity [`arm_frozen_mlp`, `arm_dtra_elastic`] on Seeds `{42, 100, 2026}` [$N=3$]):

| Method (Strict Parity) | $K=10$ SR | $K=30$ SR | $K=50$ SR | $K=70$ SR | **$K=100$ SR $\uparrow$** | FR $\downarrow$ | BWT $\uparrow$ | RouteAcc $\uparrow$ | Dup-Lang $\uparrow$ | Bases | Active Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `arm_frozen_mlp` ($N=3$) | `82.00±4.32%` | `85.11±2.79%` | `79.07±0.50%` | `76.48±0.97%` | `70.33 ± 1.18%` | `2.90±1.05%` | `+0.20±2.29%` | `81.94±0.24%` | `41.76 ± 2.09%` | `100.0` | `6.46±0.57 ms` |
| `arm_er_5pct` ($N=2$) | `62.00±2.00%` | `59.67±6.33%` | `54.80±2.40%` | `49.14±4.00%` | `40.50 ± 0.30%` | `27.07±4.04%` | `-20.71±2.93%` | `1.00±0.00%` | `0.00 ± 0.00%` | `1.0` | `5.34±0.03 ms` |
| `arm_dtra_rigid` ($N=2$) | `84.00±2.00%` | `90.33±1.00%` | `91.00±1.00%` | `87.14±0.57%` | `86.20 ± 0.00%` | `2.93±0.91%` | `+0.10±0.91%` | `99.86±0.14%` | `99.53 ± 0.47%` | `27.0` | `9.36±0.05 ms` |
| **`arm_dtra_elastic` ($N=3$, Full)** | `80.00±1.63%` | **`91.11±1.37%`** | **`90.67±1.00%`** | **`87.71±0.70%`** | **`87.00 ± 0.86%`** | **`2.69±0.67%`** | **`+0.20±0.72%`** | **`99.90±0.13%`** | **`99.68 ± 0.45%`** | **`27.0`** | `9.55±0.20 ms` |

* **Per-Seed Final $K=100$ Breakdown:**
  * `arm_dtra_elastic`: Seed 42 **`87.80%`**, Seed 100 **`87.40%`**, Seed 2026 **`85.80%`** $\rightarrow$ **3-Seed Mean `87.00 ± 0.86%`** (`+16.67%p` SR and `+57.93%p` Dup-Lang routing over matched `arm_frozen_mlp`; sub-linear `7 -> 13 -> 22 -> 24 -> 27` frozen shared bases across all seeds).
  * `arm_frozen_mlp`: Seed 42 `72.00%`, Seed 100 `69.60%`, Seed 2026 `69.40%` $\rightarrow$ Mean `70.33 ± 1.18%`.
* **Note on Milestone FR/BWT Under Exact Parameter Freezing:** Weight parameters incur strictly **0.0** gradient drift ($\nabla_\theta = 0$). In `arm_frozen_mlp`, intra-interval Voronoi encroachment degrades accuracy between discrete checkpoints prior to peak capture, collapsing `Mean SR` (`70.33%`) rather than inflating milestone-delta `FR` (`2.90%`), while residual `~2%` FR/BWT reflects $n_{\text{eval}}=5$ binomial simulator variance.

### B. LIBERO-Goal 5-Task Pilot (Seed 42, $n_{\text{eval}}=5$)

| Method (4 Parity Arms) | Mean SR $\uparrow$ | FR $\downarrow$ | BWT $\uparrow$ | Route $\uparrow$ | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Frozen Slot MLP (Baseline) | `100.0%` | `0.0%` | `+15.0%` | `100.0%` | `3695 MB` |
| Experience Replay (5%) | `88.0%` | `10.0%` | `-5.0%` | --- | `3298 MB` |
| DTRA Rigid ($\alpha=0$) | `88.0%` | `10.0%` | `-5.0%` | `100.0%` | `3696 MB` |
| **DTRA v3.0 Elastic (Ours)** | `92.0%` | `5.0%` | `+0.0%` | `100.0%` | `3696 MB` |

### C. 3D Meta-World 8-Task Continual Ablation (3-Seed Mean ± Std, Seeds `42, 100, 2026`)

| Method | Clean SR $\uparrow$ | FR $\downarrow$ | Clean Route $\uparrow$ | Severe OOD SR $\uparrow$ | OOD Route $\uparrow$ | CPU Active ($K=1\to 8$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ABK-VLA v1.1.3 (DoG Vision) | `82.8 ± 2.6%` | `0.0 ± 0.0%` | `99.7%` | `2.2 ± 1.0%` | `34.7%` | `0.27 -> 0.83 ms` |
| Frozen Slot MLP (Multimodal) | `85.6 ± 3.1%` | `0.0 ± 0.0%` | `100.0%` | `23.6 ± 3.9%` | `100.0%` | `0.21 -> 0.20 ms` |
| **DTRA Full (Multimodal, Ours)** | `72.2 ± 6.8%` | **`0.0 ± 0.0%`** | **`100.0%`** | **`51.4 ± 4.4%`** | **`100.0%`** | **`1.87 -> 1.87 ms`** |


### Evaluation scope and resource interpretation

Frozen MLP and DTRA Elastic use seeds 42, 100, and 2026; ER 5% and DTRA Rigid use seeds 42 and 100. Each task is evaluated with five rollouts per seed using the same first five provided initial states. Cross-seed standard deviations are not confidence intervals over a broader initial-state distribution.

Parameter freezing and temporary state reset are separate: `post_snap_norm = 0.0000` records a reset state, not an independent before/after weight comparison. LIBERO-100 success-rate forgetting is **2.69 ± 0.67%**. Its sources have not been separated into routing changes and rollout variability.

The **73% reduction concerns motor basis count**, from one basis per task to 27 shared bases at K=100. Recorded policy parameters occupy **183.14 MB** for DTRA versus **151.67 MB** for Frozen MLP, excluding the common frozen backbone. The experiments do not establish an asymptotic memory-growth law or a break-even task count beyond K=100.

On Meta-World, DTRA Full has lower clean success (**72.2 ± 6.8%** vs. **85.6 ± 3.1%**) and higher severe OOD success (**51.4 ± 4.4%** vs. **23.6 ± 3.9%**) than multimodal Frozen MLP. This is an observed trade-off, not a demonstrated causal explanation. Registration involves training; the **9.55 ± 0.20 ms** LIBERO figure measures policy inference. Broad retrieval remains O(K), with a constant-cost selected decoding core.

---

## 3. Repository Structure (`dtra/`)

```text
dtra/
├── PAPER_20P_OUTLINE.md                 # Exact 20-page NeurIPS/CoRL outline & v3.0/v4.0 boundary ledger
├── core/
│   ├── bspline_core.py                  # C^2 clamped cubic B-Spline trajectory decoder (N_k=6 -> H=16)
│   ├── hypertube_transport.py           # [v4.0 Track] Object-centric hypertube coordinate transport
│   └── sim_dtra_tube_dynamics.py        # Hypertube ring dynamics, elastic attraction & snap-back kernel
├── experiments/
│   ├── libero/
│   │   └── benchmark_dtra_libero.py     # LIBERO-Goal & LIBERO-100 continual suite runner & SharedBasisBank
│   ├── metaworld/
│   │   ├── ablate_dtra_scientific.py    # 3-seed 6-arm 3D Meta-World ablation & CPU O(1) benchmark
│   │   ├── ablate_dtra_supplemental.py  # Naive FT / EWC / ER5% / multi-OOD / reverse-order checks
│   │   └── benchmark_dtra_3d_metaworld.py
│   ├── calvin_v4/, calvin_v5/           # [v4.0 Future Track] CALVIN ABCD->D (excluded from v3.0)
│   └── robocasa_v4/                     # [v4.0 Future Track] RoboCasa kitchen benchmark (excluded from v3.0)
├── visualization/
│   └── visualize_dtra_xai.py            # White-box routing & hypertube XAI diagnostics
└── paper_v2/
    ├── DTRA_v3.0_Preprint.pdf           # GitHub/personal build (DTRA v3.0, 20 pages, >=8.5pt tables)
    ├── dtra_preprint.pdf                # Clean arXiv build (pure ASCII TeX + pdfLaTeX safe, 20 pages)
    ├── dtra_v3.0_preprint.tex           # XeTeX 20-page LaTeX source
    ├── dtra_preprint.tex                # Pure ASCII arXiv 20-page LaTeX source
    ├── DTRA_v2_Spec.md                  # Technical specification archive
    └── README_KO.md                     # Korean paper notes
```

---

## 4. Quickstart (Run from Repository Root)

The public release contains preprints, LaTeX sources, figures, and locked result JSONs. The benchmark harness and reproduction dependencies are not yet fully included. Experimental commands below describe the local research checkout and are not a complete setup recipe for the public checkout.

```bash
# Unit Tests (No-Encroachment, CMC, Two-Timescale Snap-Back, SharedBasisBank Zero-Drift)
python -m unittest discover -s tests -v

# LIBERO-100 (K=100) Continual Benchmark (runs inside `libero` conda env)
python -m dtra.experiments.libero.benchmark_dtra_libero \
  --suites libero_10,libero_90 --max-tasks 100 \
  --stream-tasks --eval-milestones 10,30,50,70,100 \
  --epochs 25 --batch-size 128 --n-eval 5 --seed 42 \
  --out-json outputs/libero/benchmark_dtra_libero_10_libero_90_k100.json

# 3D Meta-World 3-Seed Scientific Ablation
python -m dtra.experiments.metaworld.ablate_dtra_scientific --seeds 42 100 2026 --epochs 80

# Rebuild 3-Seed Tables, Figures & 20-Page Manuscripts
python scripts/summarize_libero_multiseed.py
python scripts/generate_libero100_figures.py
python scripts/build_dtra_v3_paper.py
```

---

## 5. Citation

```bibtex
@misc{son2026dtra,
  author       = {Son, Seungjoon},
  title        = {{DTRA}: Dynamic Topology Ring Architecture for Replay-Free Continual Visuomotor Control via Hyperspherical Isolation and Shared Motor Bases},
  year         = {2026},
  month        = nov,
  doi          = {10.5281/zenodo.24218483},
  url          = {https://doi.org/10.5281/zenodo.24218483}
}

@misc{son2026abkvla,
  author       = {Son, Seungjoon},
  title        = {{ABK-VLA}: Task-ID-Free Continual Visuomotor Policy Learning via Autonomous B-Spline Knot Compression and Visual Prototype Routing},
  year         = {2026},
  month        = nov,
  doi          = {10.5281/zenodo.24208545},
  url          = {https://doi.org/10.5281/zenodo.24208545}
}
```
