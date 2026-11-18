# ABK-VLA v1.1.3 Release Notes

This update release (`v1.1.3`) preserves the existing `v1.1.2` release while publishing an academically strengthened **9-page** preprint ([`paper/ABK_VLA_Preprint.pdf`](paper/ABK_VLA_Preprint.pdf)), OpenType Unicode LaTeX typography (`fontspec`, `unicode-math`), improved full-width benchmark tables, and cleaner Experiment 3 plot headroom.

---

## Highlights & Changes

### 1. Academic & Diagnostic Strengthening ([`paper/ABK_VLA_Paper.tex`](paper/ABK_VLA_Paper.tex) → [`paper/ABK_VLA_Preprint.pdf`](paper/ABK_VLA_Preprint.pdf), 9 pages)
- **Expanded Bibliography (`[1]`–`[29]`):** Expanded the references from 5 to **29 peer-reviewed citations** covering continual learning & parameter isolation, prototype & prompt routing, continual robot manipulation benchmarks (Meta-World, LIBERO), action chunking & multimodal policies, WTA multiple-choice learning, and movement primitives / NURBS B-splines.
- **Scope Clarification on `ABK-VLA`:** Explicitly clarified in the Introduction and Scope section that ABK-VLA denotes a modular **vision-and-proprioception** action-chunking architecture prior to language-backbone integration (containing no language encoder/VLM backbone or physical robot hardware experiments in this release).
- **Optimizer Learning-Rate Rationale (`Section 4.1`):** Explained why full-backbone fine-tuning baselines (`Models 1–3`) use `lr = 3e-4` over all `4.59M` shared visual-backbone parameters for encoder stability, whereas frozen-backbone slot-isolated heads (`Models 4–5`) train compact `0.30M–0.33M` MLP slot heads at `lr = 1e-3` (`5e-3` on `prior_weight`).
- **$O(K)$ Sequential Residual Chaining Analysis (`Exp 3` & `Section 6`):** Clarified that even under `Hard Argmax` (selecting a single slot $s^*$), sequential residual chaining across slots $0 \dots s^*$ via Eqs. (3)–(4) still executes in $O(K)$ time, explaining the modest `7.8%` standalone latency reduction (`387.8 ms` → `357.5 ms`).
- **3D Meta-World Cosine Margin & $t=0$ Identifiability Bounds (`Exp 4` & `Section 6`):** Reported the raw $T_3$ prototype cosine margin of **`0.0168`** (`0.9792` vs. threshold `0.996`, expanded via temperature `T=0.003` to a `5.60` logit gap) and explicitly noted that $t=0$ key caching assumes distinguishable reset scene geometries unless paired with foreground segmentation or goal/language conditioning.

### 2. OpenType Unicode Typesetting, Full-Width Tables & Plot Polish
- **Native OpenType Unicode Pipeline:** Switched [`paper/ABK_VLA_Paper.tex`](paper/ABK_VLA_Paper.tex) from 8-bit `T1`/`lmodern` encoding to XeTeX OpenType (`fontspec`, `amsmath`, `unicode-math`) so arrows (`→`, `↑`, `↓`), math operators (`∈`, `±`, `⊤`), code identifiers (`prior_weight`), and bold Greek symbols (`\symbfit{\mu}`) render cleanly and extract reliably in `/ToUnicode` PDF CMaps.
- **Full-Width Tables 2–4 & Layout Convergence (9 Pages):** Promoted Tables 2–4 to unshrunk two-column `\begin{table*}[t]` (`\small`) without `\resizebox{\columnwidth}{!}` distortion or gutter overlap, promoted the 7-d geometric pointer $\mathbf{p}_{\text{geom}}$ to display math, fixed Abstract/caption spacing, and compacted two-column float/reference placement cleanly into **9 pages** with zero `Overfull \hbox` warnings.
- **Independent Fig. 2 / Fig. 3 & Exp 3 Plot Margin Fix ([`experiments/eval_argmax_vs_blend.py`](experiments/eval_argmax_vs_blend.py) → [`outputs/exp3_argmax_vs_blend.png`](outputs/exp3_argmax_vs_blend.png)):** Separated the retention heatmap (`Fig. 2`) and autonomous routing matrix (`Fig. 3`) into standalone figures and updated `plot_exp3()` (adding `--plot-only` support) to regenerate `outputs/exp3_argmax_vs_blend.png` at 200 DPI with clean bar-label headroom (`84.33%` FR).

---

## Verified Benchmark Consistency (`v1.1.2` Metrics Preserved)

| Benchmark | Configuration | Primary Metric | Forgetting Rate ↓ | Autonomous Routing |
| :--- | :--- | :---: | :---: | :---: |
| **2D 4-Task CL (`seed=42`)** | `Model 5` (`Select-then-Blend`, `T=0.05`) | **85.59%** Avg $F_1$ | **1.66%** | **99.98%** diag. (`4/4`) |
| **2D Checkpoint Mode (`Exp 3`)** | `Model 5` (`Hard Argmax`) | **84.39%** Avg $F_1$ | **0.00%** | **99.98%** diag. (`4/4`) |
| **2D 3-Seed Forward (`{42,43,44}`)** | `Blend` / `Hard Argmax` | `85.15±1.06%` / `84.58±1.14%` | `0.61±0.74%` / **0.00%** | **99.98 ± 0.00%** |
| **2D 3-Seed Reverse (`{42,43,44}`)** | `Blend` / `Hard Argmax` | `79.50±4.97%` / `78.16±5.53%` | `0.06±0.06%` / **0.00%** | **99.98 ± 0.00%** |
| **3D Meta-World (`Exp 4`, 4-Task)** | `3D ABK (Blend)` / `3D ABK (Argmax)` | **91.7%** / `86.7%` Avg SR | **0.0%** / **0.0%** | **100.0%** (No `Task-ID`) |

**Archive & Preprint Asset:**
- **Zenodo DOI:** [`10.5281/zenodo.24208545`](https://doi.org/10.5281/zenodo.24208545)
- **Preprint PDF Attached:** `paper/ABK_VLA_Preprint.pdf` (9 pages)
