# DTRA v3.1.0 Release Notes — *Final arXiv Preprint, External CL Parity & 12k-Rollout Verification*

[![Preprint (DTRA v3.1 Build, 23p)](https://img.shields.io/badge/Preprint-DTRA__v3.0__Preprint.pdf-b31b1b.svg)](dtra/paper_v2/DTRA_v3.0_Preprint.pdf)
[![arXiv Manuscript (23p)](https://img.shields.io/badge/arXiv%20Clean-dtra__preprint.pdf-9e1c20.svg)](dtra/paper_v2/dtra_preprint.pdf)
[![arXiv Source Bundle](https://img.shields.io/badge/arXiv%20Bundle-.tar.gz-2ea44f.svg)](dtra/paper_v2/dtra_arxiv_submission.tar.gz)
[![Prior Release v3.0.0](https://zenodo.org/badge/DOI/10.5281/zenodo.24297711.svg)](https://doi.org/10.5281/zenodo.24297711)

This release (**`v3.1.0`**) finalizes the **23-page arXiv-ready manuscript** ([`dtra_preprint.pdf`](dtra/paper_v2/dtra_preprint.pdf), standalone-verified [`dtra_arxiv_submission.tar.gz`](dtra/paper_v2/dtra_arxiv_submission.tar.gz), `0` undefined/citation warnings, `0` non-ASCII characters, **57 references** including `LOTUS`, `TAIL`, `BUDS`, `OrthoSkillVLA`, `SAMBAR`, `R2-WAM`, `ComposeCL`, and `On-Policy Replay`). *(Note: `DTRA_v3.0_Preprint.pdf` keeps its legacy filename for link compatibility while containing the latest v3.1.0 compiled manuscript.)*

---

## Key Additions & Code-Accuracy Audit Updates Over `v3.0.0`

### 1. Direct Matched-Encoder Comparison Against Published CL Baselines (`DER++`, `L2P`, `O-LoRA`)
* **LIBERO-100 ($K=100$, Seed 42, identical frozen ResNet-18 DualCam parity; Mean ± Sample Std `ddof=1`):**
  * **DTRA Elastic ($N=3$):** **`87.00 ± 1.06%`** (`FR = 2.69 ± 0.82%`, `Dup-Lang Route = 99.68 ± 0.55%`, `27` frozen shared bases; `183.14 MB` total policy params)
  * **DER++ (NeurIPS '20, 5% buffer + distillation, `1.70 MB`):** `38.40%` (`FR = 29.70%`, `BWT = -16.16%`; collapses to `12.00%` on earliest 10 tasks `libero_10`)
  * **L2P (CVPR '22, prompt pool, `2.09 MB`):** `1.20%` (`FR = 51.72%`, `BWT = -51.72%`)
  * **O-LoRA (EMNLP '23, orthogonal LoRA, `1.70 MB`):** `1.00%` (`FR = 42.63%`, `BWT = -42.63%`; cross-hardware A6000 replication: `0.40%`)
  * *Full methodological transparency:* Section 6.3 explicitly discloses the $88\times\text{--}108\times$ parameter capacity ratio (`183.14 MB` vs `1.70–2.09 MB`) and non-transformer adaptation constraints of `L2P`/`O-LoRA`.
* **3D Meta-World ($K=8$, 3 Seeds `{42, 100, 2026}`):**
  * **DER++:** `73.3 ± 3.6%` Clean / `22.9 ± 2.7%` FR / `13.6 ± 7.3%` Severe OOD
  * **L2P:** `16.7 ± 3.6%` Clean / `79.4 ± 10.9%` FR / `17.5 ± 7.1%` Severe OOD
  * **O-LoRA:** `7.2 ± 1.4%` Clean / `63.2 ± 5.5%` FR / `9.4 ± 5.1%` Severe OOD

### 2. Zero-Retrain Extended Initial-State Verification (`n_eval=20`, **12,000 Total Rollouts**)
Evaluating all three saved $K=100$ checkpoints (`Seeds 42, 100, 2026`) over `20` initial states/task (`6,000` rollouts/arm, `12,000` total; [`eval_libero100_extended_neval20.json`](outputs/libero/eval_libero100_extended_neval20.json)) confirms generalization to **15 completely unseen initial states (`init[5:20]`)**:
* **DTRA Elastic:** **`85.20 ± 0.43%`** overall (`[84.3, 86.1]%` descriptive pooled 95% Wilson CI) | Unseen `init[5:20]`: **`84.80 ± 0.44%`** (`[83.7, 85.8]%`) | `Dup-Lang`: **`89.06%`** (`99.94%` routing)
* **Frozen MLP:** `69.12 ± 0.90%` overall (`[67.9, 70.3]%`) | Unseen `init[5:20]`: `68.82 ± 0.91%` (`[67.5, 70.2]%`) | `Dup-Lang`: `36.33%` (`41.80%` routing)
* **Verified Delta:** **`+16.08%p`** overall, **`+15.98%p`** on unseen `init[5:20]`, **`+52.72%p`** on `Dup-Lang` SR (`Unique-Lang`: `+0.38%p`).

### 3. Complete `N=3` Multi-Seed Parity Across All Four Core LIBERO-100 Arms (`ddof=1`)
Completed the two remaining `Seed 2026` runs (`arm_dtra_rigid`: `87.60%` SR / `1.82%` FR; `arm_er_5pct`: `37.40%` SR / `24.85%` FR), bringing all four core arms to full `N=3` (`Seeds 42, 100, 2026`):
* `arm_dtra_elastic` ($N=3$): **`87.00 ± 1.06%`** (`FR = 2.69 ± 0.82%`)
* `arm_dtra_rigid` ($N=3$): **`86.67 ± 0.81%`** (`FR = 2.56 ± 1.11%`, statistically comparable to Elastic)
* `arm_frozen_mlp` ($N=3$): `70.33 ± 1.45%` (`FR = 2.90 ± 1.29%`)
* `arm_er_5pct` ($N=3$): `39.47 ± 1.81%` (`FR = 26.33 ± 4.24%`, `BWT = -20.00 ± 3.17%`)

### 4. Strict Paper-vs-Code Audit & Meta-World Causal Decomposition
* **100% Paper-vs-Code Alignment:** Separated Meta-World (`HypertubeNode`, `N_k=6`, `epochs=55`, `n_ood_eval=15`) from LIBERO-100 (`LIBEROHypertubeNode`, `N_cp=8`, 2-layer `theta_head`, 3-modality `4×4+DoG` CMC, `_val_queries` 96-vector key-validation exemplars/task, `SharedBasisBank` `73%` basis-slot / `5.33%` total parameter reduction), removed buggy `FWT` column, downgraded `Theorems 1–3` to `Propositions 1–3`, and transparently disclosed Meta-World 3-seed paraphrase (`26.1%` vs `61.1%`) and `ml1.test_tasks` (`75.0%` vs `87.5%`).
* **Meta-World Causal Decomposition (3 Seeds):** Disabling Conv4 feature-noise regularization (`p=0, sigma=0`) lifts **DTRA Full (No Feat-Reg)** clean SR by **`+10.9%p`** (`72.2 ± 6.8%` $\to$ **`83.1 ± 5.1%`**) while retaining **`45.0 ± 5.4%`** severe OOD SR (`0.0%` FR, `100.0%` routing).

### 5. Hero Visual Asset
* Added high-resolution 4-clip LIBERO-100 closed-loop Hero Rollout GIF ([`dtra/assets/libero100_k100_hero_preview.gif`](dtra/assets/libero100_k100_hero_preview.gif), `2.62 MB`) rendered directly from the frozen `K=100` `DTRA Elastic` policy.
