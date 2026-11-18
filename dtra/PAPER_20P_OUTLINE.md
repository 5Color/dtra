# DTRA v3.0 — NeurIPS / CoRL 20-Page Master Manuscript Blueprint

> **Document Layout:** Research preprint — **10pt, 1-Column (Single-Column)** (`\linespread{1.05}`, `\arraystretch{1.2}`, Table Font $\ge 8.5\text{pt}$, `Overfull = 0`, `Underfull = 0`)  
> **Total Budget:** **20 Pages** (`9 pages Main Body + 2 pages References + 9 pages Appendix A–D`, restoring `Figure 8` XAI rollout telemetry & all detailed split tables at full readability)  
> **Strict Version Boundary:** 본 20쪽 설계서(`DTRA v3.0`)의 모든 본문·부록 실측 수치, 표, 그래프는 **단일 링 `Region` ($K=100$, LIBERO-100) 및 `Meta-World` / `LIBERO-Goal` 구현체**의 검증된 `_locked.json` 결과만 연결합니다.  
> **멀티 링 `Organ` ($K=5{,}000$)** 및 **`CALVIN v4` / `RoboCasa` (`Object-Centric Hypertube Transport`)** 실험은 **Section 6.1 / Future Work 및 후속 논문(`DTRA v4.0`) 트랙으로 엄격히 분리**하며 본 원고 실측 표에 혼입하지 않습니다.

---

## 0. Page Allocation Overview (Finalized 20-Page Budget Map)

| Page Range | Block | Target Role | Page Budget |
| :--- | :--- | :--- | :---: |
| **Pages 1–2** | **Section 1: Introduction** | 파라미터 간섭 문제 정의, 기하·위상 분리 패러다임, 4대 핵심 기법 및 기여 요약, Fig. 1 (End-to-End Architecture & Scope Boundary) | **2.0 p** |
| **Pages 3–4** | **Section 2 & 3: Related Work & Preliminaries** | 연속 학습 3계열(Regularization / Rehearsal / Isolation), 행동 청킹·B-Spline 운동 원시, 연속 PEFT(`SeqLoRA`/`KeepLoRA`/`OrthoSkillVLA` 도메인·사전학습 전제 분리), $\mathbb{S}^{63}$ 초구면·보로노이 분할·$C^2$ B-Spline 정식화 | **2.0 p** |
| **Pages 5–6** | **Section 4: Method (DTRA v3.0)** | 단일 링 `Region` 내부 메커니즘 4종(No-Encroachment 마진 정제, 2-Stage Top-$M$ CMC, `SharedBasisBank` + AdamW Silent Drift $=0.0$, Two-Timescale Snap-Back) 및 Algorithm 1 Pseudocode | **2.0 p** |
| **Pages 7–8** | **Section 5: Experiments & Ablations** | 엄격 인코더 패리티 4-Arm 프로토콜, 3D Meta-World 8-Task (3-Seed + Supplemental CL/OOD/Reverse), LIBERO-Goal ($K=5$) Pilot, LIBERO-100 ($K=10\to 100$) 스케일링 곡선(Fig. 4) 및 밀도 전이(Regime Crossover) 분석 | **2.0 p** |
| **Page 9** | **Section 6 & 7: Discussion, Scope Boundary & Conclusion** | `Cell -> Region` (Implemented) vs `Organ` (Conceptual Future Work) 경계, White-Box `Routing Call Stack` & 외과적 결함 격리, 등록 학습과 제어 추론 분리, $O(K)$ 검색과 $O(1)$ 선택 커널 비용 구분, 가중치 불변과 상태 리셋의 구분, 한계 및 결론 | **1.0 p** |
| **Pages 10–11** | **References** | 약 **50편** 핵심 문헌 (모놀리식 트랜스포머·VLA, 연속학습 정칙화/리플레이/격리, 운동 원시·B-Spline, 직교 PEFT/부분공간, 초구면/위상 라우팅, 벤치마크·선행연구 `ABK-VLA`) | **2.0 p** |
| **Pages 12–20** | **Appendix A–D (Supplementary Material)** | 수학적 증명 + $O(1)$ 레이턴시(A, `12–13p`), LIBERO-100 상세 마일스톤·스플릿·리소스·하이퍼파라미터·씬별 분할(B, `14–16p`), Meta-World 태스크별 3-Seed·9종 OOD·Snap-Back·Heatmap(C, `17–18p`), SharedBasisBank 진단·12행 Call Stack·Fig. 8 XAI 궤적(D, `19–20p`) | **9.0 p** |

---

## 1. Main Body Detailed Section-by-Section Plan (Pages 1–9)

### Pages 1–2 — Section 1: Introduction (2.0 p)
* **Page 1 (Title, Abstract, Problem Framing):**
  * Title & Author block (`10pt single-column` 표준)
  * **Abstract:** Replay-Free · Task-ID-Free 연속 제어 프레임워크와 4대 구성 요소를 소개하고, LIBERO-100 3-Seed 성공률 `87.00 ± 0.86%`, 라우팅 `99.68 ± 0.45%`, 관측 FR `2.69 ± 0.67%`, 기저 개수와 전체 파라미터 용량의 차이, Meta-World Clean/OOD 트레이드오프를 구분합니다. 다른 평가 프로토콜의 OrthoSkillVLA와 수치 순위를 비교하지 않습니다.
  * **Fig. 1 (`fig6_dtra_v3_architecture_schematic.png`):** 단일 링 `Region`($K=100$) 4대 모듈 파이프라인 + 하단 `Cell -> Region [IMPLEMENTED]` vs `Organ [FUTURE ROADMAP]` 구현 경계 명시도.
* **Page 2 (Research Lineage & 4 Key Bottleneck Solutions):**
  * **Research Lineage (`ABK-VLA` $K=4$ → Meta-World $K=8$ → LIBERO-100 $K=100$):** `ABK-VLA`가 남긴 $O(K)$ 잔차 체이닝 지연 및 DoG 노이즈 취약성 해결과, $K=100$ 고밀도 환경에서 새롭게 부각된 (1) **Voronoi Encroachment (`Dup-Lang` 붕괴)**, (2) **$O(K)$ Head 증가 vs 표현력 압축 딜레마** 정리.
  * **4-Bullet Contributions (1-Column 여유 분량 활용, 수식-직관 매핑 강화):**
    1. No-Encroachment Margin Refinement (이전 태스크 원본 앵커 $\mu_j$ & 자기유사도 하한 $s_{jj}$ 직교 사영 보존)
    2. Two-Stage Top-$M$ Common-Mode Cancellation ($K=100$ `Dup-Lang` 충돌 해소)
    3. `SharedBasisBank` ($K=100 \to 27$ 기저, 73% 압축 + `requires_grad=False` & AdamW Silent Drift $= 0.0$)
    4. Two-Timescale Elastic Snap-Back (`chunk_snap_back()` vs `episode_snap_back()`, `post_snap_norm = 0.0000`)

### Pages 3–4 — Section 2 & 3: Related Work & Preliminaries (2.0 p)
* **Page 3 — Section 2: Related Work:**
  * **2.1 Continual Learning & Parameter Isolation:** Regularization (EWC, RWalk), Rehearsal (GEM, ER), Dynamic Architecture / Isolation (PNN, PackNet, L2P) 비교 표 삽입 (Task-ID 필요 여부, Replay 필요 여부, Gradient Drift $=0$ 보장 여부, $O(1)$ 활성 실행 여부).
  * **2.2 Visuomotor Action Chunking & Continual PEFT:** ACT, Diffusion Policy, OpenVLA, ProMP 대비 $C^2$ Clamped Cubic B-Spline 압축의 장점; 일반 연속 LoRA(`KeepLoRA` — VLM/분류), 확산 모델 직교 LoRA(`SeqLoRA` — text-to-image), 로봇 VLA 부분공간(`OrthoSkillVLA` — VLM 사전학습 + 표준 LIBERO 서브스위트) 간 **적용 도메인 및 비교 전제** 명확히 구분.
* **Page 4 — Section 3: Mathematical Preliminaries:**
  * **3.1 Multimodal Metric Key Construction:** Frozen ResNet-18 DualCam (정적 그리드 + 손목 카메라 + DoG 저주파 패리티) + 고유수용감각 + 결정론적 `LightweightTextEncoder` 결합 및 $\mathbb{S}^{127} \xrightarrow{P_{\text{topo}}} \mathbb{S}^{63}$ 위상 사상.
  * **3.2 Clamped $C^2$ Cubic B-Spline Decoder ($N_k = 6 \to H = 16$):** 매듭 벡터, 기저 행렬 $B \in \mathbb{R}^{16 \times 6}$, 폐루프 실행 스텝(`exec_steps = 8`)에서의 위치·속도·가속도 연속성 조건 정식화.
  * **3.3 Hyperspherical Voronoi Partitioning Problem:** 고밀도 $K=100$ 및 동일 지시어(`Dup-Lang`) 환경에서 보로노이 경계가 무너지는 수학적 조건 도출.

### Pages 5–6 — Section 4: DTRA v3.0 Method Architecture (2.0 p)
* **Page 5 (Core Routing & Encroachment Avoidance):**
  * **4.1 No-Encroachment Margin Refinement:** 제약 조건 $\cos(\mu_j, \mathbf{k}_i) \le s_{jj} - m_{\text{safe}}$ 및 업데이트 직교 투영 연산자 상세 유도.
  * **4.2 Two-Stage Top-$M$ Common-Mode Cancellation (CMC):** Stage 1 $O(K)$ 멀티모달 코사인 스캔 $\to$ Top-$M$ ($M=5$) 후보 추출 $\to$ 후보 공통 성분 $\bar{\mathbf{x}}_{\text{cm}}$ 제거 후 DualCam 미세 공간 그리드 + 고유수용 잔차 재순위화 정식화.
* **Page 6 (Motor Basis Compression, Zero-Drift Snap-Back & Algorithm):**
  * **4.3 `SharedBasisBank` & Optimizer Exclusion:** 신규 태스크 등록 시 기존 기저 재사용(Cosine/잔차 임계치 기반 신규 할당 vs 조합) 및 동결 기저 `AdamW` 파라미터 그룹 완전 배제($L_\infty$ Silent Drift $= 0.0$).
  * **4.4 Two-Timescale Elastic Snap-Back & Episodic Trace:** `chunk_snap_back()` ($\Delta \mathbf{z}_i \leftarrow \mathbf{0}$), 장기 루프 고착 방지용 `episodic_trace + Conf Gate`, 에피소드 종료 시 `episode_snap_back()` (`post_snap_norm = 0.0000`).
  * **Algorithm 1 (`1-Column Full-Width`):** Train Registration + Closed-Loop Eval/Control 통합 의사코드.

### Pages 7–8 — Section 5: Experiments & Ablation Analysis (2.0 p)
* **Page 7 (Experimental Setup + Meta-World 3-Seed + Supplemental + LIBERO-Goal Pilot):**
  * **5.1 Strict-Parity Protocol & Step 1 Unit Invariants:** 4개 대조군(`arm_frozen_mlp`, `arm_er_5pct`, `arm_dtra_rigid`, `arm_dtra_elastic`), 6개 Step 1 동역학 불변량, 5개 v3.0 모듈 단위 테스트 요약.
  * **Table 1 & Fig. 1 (`fig1_ood_perturbation_curves.png`):** Meta-World 8-Task 3-Seed (Seeds 42, 100, 2026) 6-Arm 결과, CPU Active Execution $K=1\to 8$ ($1.87 \to 1.87\,\text{ms}$), Table 2 (Supplemental CL `Naive FT`/`EWC`/`ER 5%` 및 Reverse Order $T_7 \to T_0$), Table 3 (`LIBERO-Goal` $K=5$ Pilot).
* **Page 8 (LIBERO-100 $K=10\to 100$ Scaling & Structural Diagnostics):**
  * **Table 4 & Table 5 & Fig. 7 (`fig7_libero100_scaling_curves.png`):** `outputs/libero/*_locked.json` 자동 파싱 연동 표(마일스톤별 Mean SR, FR, BWT, Overall Route, Dup-Lang Route, Active Bases, Peak VRAM, Latency).
  * **5.2 Interpreting Non-Zero Milestone FR/BWT Under Exact Parameter Freezing:** 마일스톤 간격 내(intra-interval) 보로노이 침범(`arm_frozen_mlp` Mean SR 붕괴 vs 낮은 FR 착시) 및 $n_{\text{eval}}=5$ 이항 시뮬레이터 초기 상태 분산 vs $\nabla_\theta = 0$ 수학적 정리.
  * **5.3 Low-Density ($K \le 8$) vs. High-Density ($K = 100$) Regime Crossover:** 저밀도 대비 고밀도 구간에서의 DTRA $+15.80\%$p 역전 분석.

### Page 9 — Section 6 & 7: Discussion, Scope Boundary & Conclusion (1.0 p)
* **6.1 Implementation Scope Boundary (`Cell -> Region` Implemented vs `Organ` Conceptual Future Work):**
  * v3.0 본 논문의 실증 범위는 **단일 링 `Region` ($K=100$)** 까지임을 명확히 선 긋고, `Organ` 계층($10^4\text{--}10^8$ 태스크), `Isometric Zero-Padding`($[\mathbf{v}; \mathbf{0}]$), 물리 환경 일반화(객체 좌표 변환 기반 하이퍼튜브 전이)는 **Future Work / 후속 논문(`DTRA v4.0`)** 으로 명시.
* **6.2 Structural White-Box Interpretability & Surgical Fault Isolation:** `Region -> Cell ID -> Active Basis -> Stage-2 CMC Margin -> ||Δz*||_2` 감사 로그 및 $C^2$ 기구학 안전 포락선.
* **6.3 Edge Hardware Affinity & Deterministic Real-Time Control:** 비활성 튜브 0 FLOPs, SRAM/DRAM/SSD 계층 정렬, CPU 1.87ms / GPU 9.42ms 기반 100 Hz+ ROS2 제어 가능성.
* **7 Conclusion:** v3.0 핵심 결론 및 요약.

---

## 2. References Plan (Pages 10–11, ~50 References)

* **Goal:** 2쪽 분량 내에 **약 50편**의 정선된 핵심 문헌 배치 (NeurIPS/CoRL 표준 `natbib`/`cite` 형식).
* **Category Breakdown (~50 papers):**
  1. **Foundation VLAs & Monolithic Policy Backbones (~10편):** Attention Is All You Need (Vaswani et al., NeurIPS 2017), RT-1, RT-2, Octo, OpenVLA (Kim et al., 2024), pi0 / ACT (Zhao et al., RSS 2023), Diffusion Policy (Chi et al., RSS 2023), RVT/PerAct 등.
  2. **Continual Learning — Regularization, Rehearsal & Isolation (~14편):** EWC (Kirkpatrick et al., PNAS 2017), RWalk (Chaudhry et al., ECCV 2018), SI, MAS, GEM (Lopez-Paz & Ranzato, NeurIPS 2017), A-GEM, DER++, Progressive Neural Networks (Rusu et al., 2016), PackNet (Mallya & Lazebnik, CVPR 2018), HAT, L2P (Wang et al., CVPR 2022), DualPrompt, CODA-Prompt.
  3. **Continual PEFT, Subspace Adaptation & Movement Primitives (~12편):** LoRA (Hu et al., ICLR 2022), O-LoRA, InfLoRA, SeqLoRA (Parsa et al., arXiv:2605.22743, 2026 — diffusion domain), KeepLoRA (Luo et al., ICLR 2026 — VLM/classification), OrthoSkillVLA (Wang et al., arXiv:2608.19589, 2026 — pretrained VLA continual skill adaptation; direct numerical ranking excluded), ProMP (Paraschos et al., NeurIPS 2013), DMPs (Ijspeert et al., 2013), B-Spline Motion Planning / ViMP.
  4. **Hyperspherical Geometry, Metric Routing & Topological Memory (~8편):** SphereFace/CosFace/ArcFace 초구면 마진 기하학, Hyperdimensional Computing / Vector Symbolic Architectures (Kanerva), Sparse Distributed Memory, Hopfield Networks & Associative Recall, Neural Collapse on Hyperspheres.
  5. **Benchmarks & Prior Lineage (~6편):** Meta-World (Yu et al., CoRL 2020), Continual World (Wołczyk et al., NeurIPS 2021), LIBERO (Liu et al., NeurIPS 2023), Drawing-VLA (Son, Zenodo DOI: 10.5281/zenodo.24122154), ABK-VLA (Son, Zenodo DOI: 10.5281/zenodo.24208545), DTRA v2/v3 Preprint Archive (Zenodo DOI: 10.5281/zenodo.24218483).

---

## 3. Appendix Plan (Pages 12–20, 9.0 Pages, >=8.5pt Readable Layout)

### Appendix A — Mathematical Proofs & Theoretical Guarantees (Pages 12–13, 2.0 p)
* **A.1 Exact Zero Residual & Weight Drift Proof ($\nabla_\theta = 0$, $L_\infty = 0.0$):** Optimizer exclusion (`requires_grad=False`) + `chunk_snap_back()` $\to$ `episode_snap_back()` (`post_snap_norm = 0.0000`).
* **A.2 $C^2$ Clamped Cubic B-Spline Continuity & Knot-Space Compression Bound:** $C^0/C^1/C^2$ continuity + 단일 청크 내부 연속성과 control-point perturbation에 대한 출력 유계성; 청크 간 연속성·하드웨어 jerk/torque 안전성은 보장하지 않음.
* **A.3 Isometric Zero-Padding Inner-Product & Voronoi Margin Invariance:** $\Phi_d^D(\mathbf{v}) = [\mathbf{v}; \mathbf{0}] \in \mathbb{S}^{D-1}$ exact inner-product & Voronoi invariance + **Fig. 5** (`fig5_o1_latency_scaling.png`).

### Appendix B — LIBERO-100 Full Milestones, Splits, Resources & Reproducibility (`_locked.json` Binding) (Pages 14–16, 3.0 p)
* **Strict `_locked.json` Binding Policy:** 본 부록의 모든 표는 아래 확정된 `_locked.json` 파일에서 `scripts/build_dtra_v3_paper.py` 및 `scripts/summarize_libero_multiseed.py`로 자동 파싱된 실측 수치만 기록합니다.

| Seed / Scope | Status (verified) | Bound Source File (Auto-Parsed Only) | Current Notes |
| :--- | :---: | :--- | :--- |
| **Seed 42** (4-Arm Full `K=10..100`) | **COMPLETED** | `outputs/libero/benchmark_dtra_libero_10_libero_90_k100_seed42_locked.json` | 4-Arm 모두 완료 (`arm_dtra_elastic 87.80%`, `arm_dtra_rigid 86.20%`, `arm_frozen_mlp 72.00%`, `arm_er_5pct 40.20%`, Bases `27`) |
| **Seed 100** (4-Arm Full `K=10..100`) | **COMPLETED** | `outputs/libero/benchmark_dtra_libero_10_libero_90_k100_seed100_locked.json` | 4-Arm 모두 완료 (`arm_dtra_elastic 87.40%`, `arm_dtra_rigid 86.20%`, `arm_frozen_mlp 69.60%`, `arm_er_5pct 40.80%`, Bases `27`) |
| **Seed 2026** (2-Arm Core: `arm_frozen_mlp`, `arm_dtra_elastic`) | **COMPLETED** | `outputs/libero/benchmark_dtra_libero_10_libero_90_k100_seed2026_locked.json` | 2-Arm Core 완료 (`arm_dtra_elastic 85.80%`, `arm_frozen_mlp 69.40%`, Bases `27`, Dup-Lang `100.0%`; 3-Seed `N=3` 합산 **`87.00 ± 0.86%`**) |

* **Appendix B Contents:** Seed Provenance Registry, Full 5-Checkpoint Milestones (`K=10..100`, `N=3/N=2` 명시), `LIBERO-Goal` 5-Task Pilot, `libero_10` vs. `libero_90` Split, Resource/FWT/VRAM (`2,686–2,713 MB`) Table, Hyperparameter Schedule, Stratified `Dup-Lang`/Scene Breakdown, `libero_10` Per-Task (`T00–T09`) Table.

### Appendix C — 3D Meta-World Detailed Per-Task Analysis & 9-Regime OOD Curves (Pages 17–18, 2.0 p)
* 8-Task 3-Seed Per-Task Breakdown (`button-press-v3` ~ `coffee-button-v3`), Supplemental CL Baselines & Reverse Ordering ($T_7 \to T_0$), 9-Regime OOD Stress Curves (`Seed 42`), Snap-Back Telemetry (`0.0000`), and **Fig. 6** (`fig4_cl_forgetting_heatmap_and_bars.png`).

### Appendix D — White-Box Routing Call Stack Logs, 27-Basis & Rollout Telemetry (Pages 19–20, 2.0 p)
* **Fig. 7** (`fig8_libero100_basis_and_callstack.png`), 12-Row White-Box Routing & Consolidation Audit Log parsed from `arm_dtra_elastic` (`Seed 42 _locked.json`), **Fig. 8** (`fig3_xai_telemetry_rollout.png`, Closed-Loop B-Spline Safety Envelope), and Surgical Fault Isolation protocol.

---

## 4. Strict Version Isolation: Scope Separation (`DTRA v3.0` vs. `DTRA v4.0 Future Work`)

다음 실험·구현 트랙은 **별도 서버(RTX A6000) 검증이 완료되어 로컬에 안전 백업되어 있으나, 본 20쪽 `DTRA v3.0` 논문의 본문·부록 실측치에는 절대 편입하지 않으며 Section 6.1(Scope Boundary) 및 Future Work 후속 논문(`DTRA v4.0`) 전용으로만 분리 관리**합니다:

| Experiment / Artifact Track | Target Manuscript | Local Archive Path (Isolated from v3.0) | Isolation Rule |
| :--- | :---: | :--- | :--- |
| **Multi-Ring `Organ` Scaling ($K=100\to 5{,}000$)** & 4-Arm Organ Stress Test | **DTRA v4.0 (Future Track)** | `outputs/organ_v4/`, `dtra/experiments/calvin_v4/organ_dtra_benchmark.py`, `four_arm_control_benchmark.py` | v3.0에서는 개념적 로드맵(`Organ`)으로만 서술 (Section 6.1, Proof A.3만 이론 기반 제시) |
| **`CALVIN` 34-Task ($ABCD\to D$) V4/V5 + Object-Centric Hypertube Transport** (`Oracle` & `Pure RGB-D`) | **DTRA v4.0 (Embodied CL Paper 2)** | `outputs/calvin_v4/`, `outputs/calvin_v5/`, `dtra/core/hypertube_transport.py`, `dtra/experiments/calvin_v4/`, `dtra/experiments/calvin_v5/` | v3.0 표·그래프에 미포함; 별도 후속 논문/실물 SO-101 확장 트랙으로 분리 |
| **`RoboCasa` 12-Task (`8` Single-Stage + `4` Multi-Stage) Kitchen Benchmark** (`Pure RGB-D`, Unmodified `_check_success`) | **DTRA v4.0 (Embodied CL Paper 2)** | `outputs/robocasa/`, `dtra/experiments/robocasa_v4/` | v3.0 표·그래프에 미포함; 후속 주방 조작 연속학습 트랙으로 분리 |
| **Single-Ring `Region` ($K=100$ `LIBERO-100`) + `Meta-World` + `LIBERO-Goal`** | **DTRA v3.0 (This 20p Paper)** | `outputs/libero/*_locked.json`, `outputs/metaworld/`, `scripts/build_dtra_v3_paper.py` | v3.0 20쪽 원고 본문 및 Appendix A–D의 유일한 실측 데이터 원천 |
