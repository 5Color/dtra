# ABK-VLA v1.1.2 Release Notes

This maintenance and documentation release (`v1.1.2`) updates the preprint manuscript (`paper/ABK_VLA_Preprint.pdf`), synchronizes all verified benchmark figures and metrics between the paper and `README.md`, clarifies latency and zero-`Task-ID` routing terminology, and improves figure legibility across the multi-column layout.

---

## Highlights & Changes

### 1. Preprint Manuscript Polish & Recompilation ([`paper/ABK_VLA_Paper.tex`](paper/ABK_VLA_Paper.tex) → [`paper/ABK_VLA_Preprint.pdf`](paper/ABK_VLA_Preprint.pdf))
- **Typesetting & Math Formatting Fixes:** Fixed inline math/code typography (converting parameter identifiers such as `exec_steps`, `max_iters`, `patience`, and `effective_patience` cleanly to `\texttt{...}` and `NMJ` to `\mathrm{NMJ}`) and ensured clean two-column compilation with zero table or text column overflows.
- **Latency Consistency Reconciliation (Table 1 vs. Table 3):** Added explicit clarification explaining why **Table 1** reports **`464.3 ms`** per episode for Model 5 (`Blend T=0.05`, timed inline within the full 5-model sequential training-and-evaluation runner `experiments/benchmark_autonomous_cl.py`), whereas **Table 3** reports **`387.8 ms`** (`Blend T=0.05`) vs. **`357.5 ms`** (`Hard Argmax`) when re-evaluating `outputs/autonomous_model_d_final.pth` under identical standalone checkpoint profiling (`experiments/eval_argmax_vs_blend.py`). Also clarified that shorter episode durations at `T >= 0.25` (`167.4 ms` / `242.8 ms`) result from early rollout stall termination under high forgetting rather than speedup.
- **Expanded 3D & OOD Diagnostic Analysis:**
  - **Experiment 1 (OOD Perturbations):** Added analysis on the clean decoupling between visual prototype routing (**`100.0%`** accuracy up to ±8 px shift, ±15° rotation, ±15% scale, `σ=0.40` noise, and 20% occlusion) and closed-loop control degradation on unaugmented noisy/shifted target inputs (`65.04%` F1 at 4 px shift, `29.55%` at `σ=0.15` noise).
  - **Experiment 4 (3D Meta-World):** Detailed why `15×15` local average subtraction (DoG-like high-pass residual) + 8 initial-frame exemplars cleanly separate tabletop camera scenes (`max_prior_sim`: `0.8143 → 0.9592 → 0.9792` under threshold `0.996`) and how positive cross-task backward transfer (**`+11.1%` BWT**, `window-open-v3`: `40.0% → 73.3%`) raises `Blend` to **`91.7%` Avg SR / `0.0%` FR** (`86.7%` SR / `0.0%` FR for `Hard Argmax`).

### 2. Figure Readability Improvements ([`outputs/exp2_multiseed_order.png`](outputs/exp2_multiseed_order.png) & Two-Column `figure*` Layout)
- **Error-Bar Label Offset Fix (`Exp 2`):** Adjusted bar value annotation offsets and `y`-axis headroom in [`experiments/benchmark_multiseed_order.py`](experiments/benchmark_multiseed_order.py) (`save_outputs`) and regenerated [`outputs/exp2_multiseed_order.png`](outputs/exp2_multiseed_order.png) at 200 DPI so `mean ± std` labels (`79.50±4.97`, `78.16±5.53`) no longer collide with error-bar caps.
- **Full-Width Two-Column Figures (`Fig. 4–7`):** Promoted Experiments 1–4 figures in `paper/ABK_VLA_Paper.tex` to two-column `figure*` floats for legible multi-panel axis ticks and numerical annotations inside `paper/ABK_VLA_Preprint.pdf`.

### 3. `README.md` Cross-Verification & Terminology Clarity ([`README.md`](README.md))
- **Unambiguous Routing Phrasing:** Replaced ambiguous phrasing (`"100.0% held-out Task-ID routing"`) with **`"100.0% autonomous routing accuracy without test-time Task-ID"`** in both summary prose and the Meta-World benchmark table so readers cannot mistake autonomous visual routing for oracle `Task-ID` supervision.
- **100% Paper–README–JSON Numerical Alignment:** Verified all reported `F1`, `Forgetting Rate (FR)`, `BWT`, `CD`, `NMJ`, `CR`, `Latency`, `Meta-World Success Rate`, parameter counts, and CLI flags against `outputs/autonomous_cl_benchmark.json` and `outputs/exp{1,2,3,4}_*.json`.
- **Reproducible CLI & Path Sync:** Documented `.zenodo.json` (`DOI: 10.5281/zenodo.24208545`), `RELEASE_NOTES_v1.1.2.md`, exact checkpoint inputs (`outputs/autonomous_model_d_final.pth`), and matching CLI flags for Experiments 1–4.

---

## Verified Key Benchmark Summary (Unchanged Ground-Truth Metrics)

| Benchmark | Configuration | Primary Metric | Forgetting Rate (↓) | Autonomous Routing |
| :--- | :--- | :---: | :---: | :---: |
| **2D 4-Task CL (`seed=42`)** | `Model 5` (`Select-then-Blend`, `T=0.05`) | **85.59%** Avg $F_1$ | **1.66%** | **99.98%** diag. (`4/4`) |
| **2D Checkpoint Mode (`Exp 3`)** | `Model 5` (`Hard Argmax`) | **84.39%** Avg $F_1$ | **0.00%** | **99.98%** diag. (`4/4`) |
| **2D 3-Seed Forward (`{42,43,44}`)** | `Blend` / `Hard Argmax` | `85.15±1.06%` / `84.58±1.14%` | `0.61±0.74%` / **0.00%** | **99.98 ± 0.00%** |
| **2D 3-Seed Reverse (`{42,43,44}`)** | `Blend` / `Hard Argmax` | `79.50±4.97%` / `78.16±5.53%` | `0.06±0.06%` / **0.00%** | **99.98 ± 0.00%** |
| **3D Meta-World (`Exp 4`, 4-Task)** | `3D ABK (Blend)` / `3D ABK (Argmax)` | **91.7%** / `86.7%` Avg SR | **0.0%** / **0.0%** | **100.0%** (No `Task-ID`) |

**Archive & Citation:**  
- **Zenodo DOI:** [`10.5281/zenodo.24208545`](https://doi.org/10.5281/zenodo.24208545)  
- **Preprint Asset Attached:** `paper/ABK_VLA_Preprint.pdf`
