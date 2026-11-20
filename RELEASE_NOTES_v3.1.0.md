# DTRA v3.1.0 Release Notes — *Final arXiv Preprint, External CL Parity & 12k-Rollout Verification*

[![Preprint (DTRA v3.0/v3.1, 23p)](https://img.shields.io/badge/Preprint-DTRA__v3.0__Preprint.pdf-b31b1b.svg)](dtra/paper_v2/DTRA_v3.0_Preprint.pdf)
[![arXiv Manuscript (23p)](https://img.shields.io/badge/arXiv%20Clean-dtra__preprint.pdf-9e1c20.svg)](dtra/paper_v2/dtra_preprint.pdf)
[![arXiv Source Bundle](https://img.shields.io/badge/arXiv%20Bundle-.tar.gz-2ea44f.svg)](dtra/paper_v2/dtra_arxiv_submission.tar.gz)
[![Prior Release v3.0.0](https://zenodo.org/badge/DOI/10.5281/zenodo.24297711.svg)](https://doi.org/10.5281/zenodo.24297711)

This release (**`v3.1.0`**) finalizes the **23-page arXiv-ready manuscript** ([`dtra_preprint.pdf`](dtra/paper_v2/dtra_preprint.pdf), standalone-verified [`dtra_arxiv_submission.tar.gz`](dtra/paper_v2/dtra_arxiv_submission.tar.gz), `0` Overfull hboxes, `0` warnings, `0` non-ASCII characters) by resolving every major peer-review vulnerability identified after `v3.0.0` (DOI `10.5281/zenodo.24297711`):

---

## Key Additions Over `v3.0.0`

### 1. Direct Matched-Encoder Comparison Against Published CL Baselines (`DER++`, `L2P`, `O-LoRA`)
* **LIBERO-100 ($K=100$, Seed 42, identical frozen ResNet-18 DualCam parity):**
  * **DTRA Elastic ($N=3$):** **`87.00 ± 0.86%`** (`FR = 2.69 ± 0.67%`, `Dup-Lang Route = 99.68 ± 0.45%`, `27` frozen shared bases)
  * **DER++ (NeurIPS '20, 5% buffer + dark-logit distillation):** `38.40%` (`FR = 29.70%`, `BWT = -16.16%`; collapses to `12.00%` on earliest 10 tasks `libero_10`)
  * **L2P (CVPR '22, replay-free prompt pool):** `1.20%` (`FR = 51.72%`, `BWT = -51.72%`)
  * **O-LoRA (EMNLP '23, orthogonal low-rank adapters):** `1.00%` (`FR = 42.63%`, `BWT = -42.63%`; cross-hardware A6000 replication: `0.40%` / `FR = 49.09%`)
* **3D Meta-World ($K=8$, 3 Seeds `{42, 100, 2026}`):**
  * **DER++:** `73.3 ± 3.6%` Clean / `22.9 ± 2.7%` FR / `13.6 ± 7.3%` Severe OOD
  * **L2P:** `16.7 ± 3.6%` Clean / `79.4 ± 10.9%` FR / `17.5 ± 7.1%` Severe OOD
  * **O-LoRA:** `7.2 ± 1.4%` Clean / `63.2 ± 5.5%` FR / `9.4 ± 5.1%` Severe OOD

### 2. Zero-Retrain Extended Initial-State Verification (`n_eval=20`, **12,000 Total Rollouts**)
Evaluating all three saved $K=100$ checkpoints (`Seeds 42, 100, 2026`) over `20` initial states/task (`6,000` rollouts/arm, `12,000` total; [`eval_libero100_extended_neval20.json`](outputs/libero/eval_libero100_extended_neval20.json)) confirms generalization to **15 completely unseen initial states (`init[5:20]`)**:
* **DTRA Elastic:** **`85.20 ± 0.35%`** overall (`[84.3, 86.1]%` 95% Wilson CI) | Unseen `init[5:20]`: **`84.80 ± 0.36%`** (`[83.7, 85.8]%`) | `Dup-Lang`: **`89.06%`** (`99.94%` routing)
* **Frozen MLP:** `69.12 ± 0.74%` overall (`[67.9, 70.3]%`) | Unseen `init[5:20]`: `68.82 ± 0.74%` (`[67.5, 70.2]%`) | `Dup-Lang`: `36.33%` (`41.80%` routing)
* **Verified Delta:** **`+16.08%p`** overall, **`+15.98%p`** on unseen `init[5:20]`, **`+52.72%p`** on `Dup-Lang` SR.

### 3. Complete `N=3` Multi-Seed Parity Across All Four Core LIBERO-100 Arms
Completed the two remaining `Seed 2026` runs (`arm_dtra_rigid`: `87.60%` SR / `1.82%` FR; `arm_er_5pct`: `37.40%` SR / `24.85%` FR), bringing all four core arms to full `N=3` (`Seeds 42, 100, 2026`):
* `arm_dtra_elastic` ($N=3$): **`87.00 ± 0.86%`** (`FR = 2.69 ± 0.67%`)
* `arm_dtra_rigid` ($N=3$): **`86.67 ± 0.66%`** (`FR = 2.56 ± 0.91%`, `BWT = +0.67 ± 1.10%`)
* `arm_frozen_mlp` ($N=3$): `70.33 ± 1.18%` (`FR = 2.90 ± 1.05%`)
* `arm_er_5pct` ($N=3$): `39.47 ± 1.48%` (`FR = 26.33 ± 3.46%`, `BWT = -20.00 ± 2.59%`)

### 4. Meta-World Clean-vs-OOD Causal Decomposition (3 Seeds)
Controlled 3-seed causal ablation ([`ablate_dtra_v31_final.py`](dtra/experiments/metaworld/ablate_dtra_v31_final.py)) isolates Conv4 feature-noise regularization (`p=0.30, sigma=0.05`) inside `DenoisingFineHead`:
* Disabling feature-noise regularization (`p=0, sigma=0`) lifts **DTRA Full (No Feat-Reg)** clean SR by **`+10.9%p`** (`72.2 ± 6.8%` $\to$ **`83.1 ± 5.1%`**) while retaining **`45.0 ± 5.4%`** severe OOD SR (`0.0%` FR, `100.0%` routing, nearly 2× unregularized Frozen MLP's `23.6 ± 3.9%`).
* Updated Related Work & 54-entry bibliography with 2026 continual VLA/LLM literature (`OrthoSkillVLA`, `SAMBAR`, `R2-WAM`, `ComposeCL`, `On-Policy Replay`).

### 5. Hero Visual Asset & Documentation Polish
* Added high-resolution 4-clip LIBERO-100 closed-loop Hero Rollout GIF ([`dtra/assets/libero100_k100_hero_preview.gif`](dtra/assets/libero100_k100_hero_preview.gif), `2.62 MB`) rendered directly from the frozen `K=100` `DTRA Elastic` policy (`T09` Kitchen Microwave, `T05`/`T87` Dup-Lang identical-instruction pair reusing `Basis #04`, and `T67` Living Room).
* Updated 7-arm scaling curves ([`outputs/figures/fig7_libero100_scaling_curves.png`](outputs/figures/fig7_libero100_scaling_curves.png)) and synced all front-facing READMEs while preserving `RELEASE_NOTES_v3.0.0.md` and `RESEARCH_LINEAGE.md` intact.
