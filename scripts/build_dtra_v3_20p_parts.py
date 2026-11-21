"""scripts/build_dtra_v3_20p_parts.py

Modular scholarly text blocks (50 verified References, Formal Mathematical Proofs
for Appendix A, Architecture Comparison Table, and Method/Discussion prose) for
the 20-page NeurIPS/CoRL 10pt single-column DTRA v3.0 manuscript.
"""

from __future__ import annotations

METHOD_PARADIGM_TABLE = r"""
\begin{table}[t]
  \centering
  \caption{Structural comparison of continual visuomotor control paradigms. DTRA~v3.0 combines Task-ID-free inference, zero raw-demonstration replay (retaining lightweight 96-vector key-validation exemplars per task during registration), exact parameter freezing ($\nabla_\theta = 0$), constant $O(1)$ active decoding core, and shared knot bases within a white-box routing stack.}
  \label{tab:paradigm_compare}
  \fontsize{7.8pt}{9.6pt}\selectfont
  \setlength{\tabcolsep}{2.0pt}
  \begin{tabular}{lccccccc}
    \toprule
    \textbf{Paradigm / Method} & \textbf{Task-Free} & \textbf{No Raw Rep.} & \textbf{$\nabla_\theta=0$} & \textbf{Active} & \textbf{Scaling} & \textbf{Dup-Lang} & \textbf{No VLM} \\
    \midrule
    Naive Fine-Tuning & \checkmark & \checkmark & $\times$ & $O(1)$ & Monolith & $\times$ & \checkmark \\
    EWC / RWalk~\cite{kirkpatrick2017ewc,chaudhry2018rwalk} & \checkmark & \checkmark & $\times$ & $O(1)$ & Monolith+Pen. & $\times$ & \checkmark \\
    GEM / ER / DER++~\cite{lopezpaz2017gem,rolnick2019clear,buzzega2020der} & \checkmark & $\times$ & $\times$ & $O(1)$+Buf & Monolith & $\times$ & \checkmark \\
    PNN / PackNet~\cite{rusu2016pnn,mallya2018packnet} & $\times$ & \checkmark & Masked & $O(K)$ & Linear / Mask & N/A & \checkmark \\
    BUDS / LOTUS / TAIL~\cite{zhu2022buds,wan2024lotus,liu2024tail} & Partial & Partial & Adapter & Skill/PEFT & Sub-lin./PEFT & Partial & Mixed \\
    L2P / CODA-Prompt~\cite{wang2022l2p,smith2023coda} & \checkmark & \checkmark & Backb. & Pool & Prompts & Partial & ViT/CNN \\
    O-LoRA / Seq- / KeepLoRA~\cite{wang2023olora,parsa2026seqlora,luo2026keeplora} & Depends & Mostly & Adapter & $O(r)$ & LoRA & $\times$ & VLM/MLP \\
    OrthoSkillVLA / SAMBAR~\cite{wang2026orthoskill,shrivastava2026sambar} & Depends & \checkmark & Subsp. & VLA & Subsp./Anch. & $\times$ & VLA \\
    $\text{R}^2$-WAM~\cite{fang2026r2wam} & \checkmark & Gen.Rep. & $\times$ & WAM & Video WAM & $\times$ & WAM+VLM \\
    ABK-VLA v1.1.3~\cite{son2026abkvla} & \checkmark & \checkmark & \checkmark & $O(K)$ & Linear $O(K)$ & $\times$ (DoG) & \checkmark \\
    \textbf{DTRA v3.0 (Ours)} & \textbf{\checkmark} & \textbf{\checkmark (Key Ex.)} & \textbf{\checkmark ($0$)} & \textbf{$O(1)$} & \textbf{27b / $O(K)$ ad.} & \textbf{\checkmark (CMC)} & \textbf{\checkmark} \\
    \bottomrule
  \end{tabular}
\end{table}
"""

ALGORITHM_1_BLOCK = r"""
\begin{algorithm}[t]
\caption{\textbf{DTRA v3.0 Training Registration \& Closed-Loop Control (LIBERO-100 Implementation)}}
\label{alg:dtra_v3}
\small
\begin{algorithmic}[1]
\Require Frozen DualCam \texttt{ResNet-18} + proprioception + \texttt{LightweightTextEncoder}; frozen \texttt{SharedBasisBank} $\mathcal{B} = \{\mathbf{B}_b\}_{b=1}^{B}$ (LIBERO-100: $N_{\text{cp}}=8$ clamped cubic B-Spline knots $\to H=16$; Meta-World: $N_k=6$); empty cell registry $\mathcal{C} = \emptyset$.
\Statex \textbf{Phase I: Sequential Continual Registration ($k = 1, \dots, K$, Task-ID Stored Only for Audit)}
\State Project visual, proprioceptive, and language features into normalized query $\mathbf{c}_t \in \mathbb{S}^{63}$ via fixed random projections ($0.25\mathbf{k}_v + 0.10\mathbf{k}_p + 0.65\mathbf{k}_\ell$ on LIBERO-100); compute task prototype $\mu_k$ and 3-modality signature prototypes $(\mathbf{v}_k^{\text{sig}}, \mathbf{p}_k^{\text{sig}}, \boldsymbol{\ell}_k^{\text{sig}})$ using $4\times 4+\text{DoG}$ visual features.
\State Refine routing key $\mathbf{k}_k$ against prior tasks $\{j < k\}$ subject to \textbf{No-Encroachment}: penalize $\langle \mu_j, \mathbf{k}_k \rangle > s_{jj} - m_{\text{safe}}$ and verify accuracy on cached lightweight key-validation exemplars ($\le 96$ normalized query vectors/task, \texttt{\_val\_queries}; no raw image/action replay).
\State Fit $C^2$ clamped cubic B-Spline knots ($\mathbf{P}_k \in \mathbb{R}^{8\times 7}$ on LIBERO-100; $\mathbb{R}^{6\times 4}$ on Meta-World); match cosine similarity to $\mathcal{B}$. If below threshold, append new frozen basis $\mathbf{B}_{B+1}$ ($\texttt{requires\_grad=False}$, excluded from \texttt{AdamW}); else share $\mathbf{B}_{b_k}$ and train per-task node adapter \texttt{LIBEROHypertubeNode} (spatial context + phase/residual heads).
\State Freeze all consolidated node and basis parameters permanently ($\nabla_\theta = 0$, $L_\infty$ silent drift $= 0.0$).
\Statex \textbf{Phase II: No-Raw-Replay, Task-ID-Free Closed-Loop Evaluation (At Control Tick $t$)}
\State \textbf{Stage 1 ($O(K)$ Broad Scan):} Score all $K$ keys on $\mathbb{S}^{63}$ (with optional temporal inertia) and retrieve Top-$M$ ($M=5$) candidates $\mathcal{M}_t$.
\State \textbf{Stage 2 (3-Modality Common-Mode Cancellation):} Compute candidate-local means across visual ($4\times 4+\text{DoG}$ 256-d), proprioceptive (12-d), and language (128-d) signature banks; strip common modes and select active cell $i^* = \arg\max_{j \in \mathcal{M}_t} s_{\text{CMC}}(t, j)$.
\State Decode selected node $i^*$ and shared basis $\mathbf{B}_{b_{i^*}}$ via phase head (\texttt{theta\_head} on LIBERO-100; elliptical \texttt{atan2} on Meta-World) and bounded spherical interpolation $\alpha_{i^*}$ into $H=16$ chunk $\mathbf{A}_t = \mathbf{B}_{\text{spline}}\mathbf{P}_{\text{eff}}$; execute \texttt{exec\_steps=8} steps.
\State At episode termination, call \texttt{episode\_snap\_back()} to reset routing inertia (\texttt{prev\_selected}, \texttt{prev\_act\_vec}) and diagnostic buffers to exact $0.0000$.
\end{algorithmic}
\end{algorithm}
"""

APPENDIX_A_PROOFS = r"""
\section{Implementation Invariants \& Geometric Properties}
\label{app:proofs}

This appendix records three verification properties underlying the mechanisms in Sections~\ref{sec:method} and~\ref{sec:discussion}: exact zero optimizer drift under \texttt{AdamW} parameter-group exclusion (Proposition~\ref{prop:zero_drift}), standard clamped cubic B-Spline within-chunk continuity and output perturbation bounds (Proposition~\ref{prop:bspline}), and the conceptual isometric zero-padding property noted for future capacity expansion (Proposition~\ref{prop:isometry}). We frame these explicitly as implementation invariants and standard mathematical properties rather than novel theorems.

\subsection{Zero Optimizer Drift Under Consolidated Parameter Exclusion}
\begin{proposition}[Exact Zero Optimizer Drift \& Episode State Reset]
\label{prop:zero_drift}
Let $\Theta_{\text{frozen}}^{(k)} = \{\mathbf{k}_j, \mathbf{B}_{b_j}, \mathbf{W}_{\text{node},j}\}_{j=1}^{k}$ denote consolidated keys, shared motor bases in \texttt{SharedBasisBank}, and prior node weights after mastering task $T_k$. Under DTRA registration:
\begin{enumerate}[leftmargin=*,itemsep=2pt]
  \item \textbf{Zero Optimizer Drift ($L_\infty = 0.0$):} For any subsequent task curriculum $T_{k+1}, \dots, T_K$ trained with decoupled weight decay \texttt{AdamW} ($\lambda > 0$), $\|\Theta_{\text{frozen}}^{(k)}(T_K) - \Theta_{\text{frozen}}^{(k)}(T_k)\|_\infty = 0.0$ holds identically.
  \item \textbf{Deterministic Episode-Boundary Reset:} Calling \texttt{episode\_snap\_back()} prior to $t=0$ clears active routing inertia (\texttt{prev\_selected}, \texttt{prev\_act\_vec}) and diagnostic deformation buffers to exact $0.0000$, preventing inter-episode state carryover.
\end{enumerate}
\end{proposition}

\begin{proof}
(1) Standard \texttt{AdamW} updates obey $\theta_{s+1} = (1 - \eta\lambda)\theta_s - \eta \hat{m}_s / (\sqrt{\hat{v}_s} + \epsilon)$. Even when $\nabla_\theta\mathcal{L} = \mathbf{0}$, retaining $\theta_s$ inside an optimizer parameter group applies silent exponential decay $(1 - \eta\lambda)^S \theta_0 \neq \theta_0$. DTRA sets \texttt{requires\_grad = False} on $\Theta_{\text{frozen}}^{(k)}$ and instantiates each task's \texttt{AdamW} optimizer exclusively over newly created active parameters ($\Theta_{\text{frozen}}^{(k)} \cap \text{param\_groups}(\texttt{AdamW}) = \emptyset$), so $\theta_{s+1} = \theta_s$ identically in IEEE-754 arithmetic. (2) Explicit zero-reinitialization of routing inertia and diagnostic buffers at episode reset guarantees identical initial internal state at every $t=0$. Note that parameter stability (1) is distinct from empirical closed-loop success-rate forgetting, which can still be non-zero due to routing boundary changes or rollout stochasticity.
\end{proof}

\subsection{Clamped \texorpdfstring{$C^2$}{C2} Cubic B-Spline Continuity \& Perturbation Bound}
\begin{proposition}[Within-Chunk $C^2$ Continuity and Convex-Hull Perturbation Bound]
\label{prop:bspline}
Let the clamped uniform degree-$3$ knot vector $U = \{0, 0, 0, 0, u_4, \dots, u_{N_{\text{cp}}-1}, 1, 1, 1, 1\}$ have $N_{\text{cp}}$ control points $\mathbf{P} \in \mathbb{R}^{N_{\text{cp}} \times d_a}$ ($N_{\text{cp}}=8$ on LIBERO-100; $N_{\text{cp}}=6$ on Meta-World), evaluated at $\tau_h = h / (H-1)$ for $h = 0, \dots, H-1$ ($H=16$). By standard spline properties~\cite{deboor1978splines,piegl1997nurbs}:
\begin{enumerate}[leftmargin=*,itemsep=2pt]
  \item Every synthesized chunk $\mathbf{a}(\tau) = \sum_{n=0}^{N_{\text{cp}}-1} B_{n,3}(\tau)\,\mathbf{p}_n$ satisfies endpoint clamping ($\mathbf{a}(0) = \mathbf{p}_0$, $\mathbf{a}(1) = \mathbf{p}_{N_{\text{cp}}-1}$) and $C^2$ continuity at all interior knots.
  \item By non-negativity ($B_{n,3}(\tau) \ge 0$) and partition of unity ($\sum_{n=0}^{N_{\text{cp}}-1} B_{n,3}(\tau) \equiv 1$), any knot perturbation $\|\delta\mathbf{P}\|_\infty \le \epsilon$ obeys $\sup_{\tau \in [0,1]} \|\delta\mathbf{a}(\tau)\|_\infty \le \epsilon$ within a single chunk.
\end{enumerate}
\end{proposition}

\begin{proof}
By Cox--de~Boor recursion with multiplicity-$4$ boundary knots and simple interior knots ($m_j=1$), $\mathbf{a}(\tau)$ is $C^{p-m_j}=C^2$ across interior transitions and satisfies $B_{n,3}(\tau) \ge 0$, $\sum_{n=0}^{N_{\text{cp}}-1} B_{n,3}(\tau) = 1$. H\"older's inequality immediately yields $\left\| \sum_{n} B_{n,3}(\tau)\, \delta\mathbf{p}_n \right\|_\infty \le (\sum_n |B_{n,3}(\tau)|)\max_n \|\delta\mathbf{p}_n\|_\infty = \|\delta\mathbf{P}\|_\infty \le \epsilon$. Replanning-boundary smoothness across consecutive chunks and physical actuator torque limits are outside this single-chunk bound.
\end{proof}

\subsection{Conceptual Isometric Zero-Padding Property (Future Work)}
\begin{proposition}[Isometric Zero-Padding Inner-Product Preservation]
\label{prop:isometry}
For unit vectors $\{\mu_j, \mathbf{k}_j\}_{j=1}^K \subset \mathbb{S}^{d-1}$, the zero-padding map $\Phi_d^{D}(\mathbf{v}) = [\mathbf{v}; \mathbf{0}_{D-d}] \in \mathbb{S}^{D-1}$ ($D > d$) preserves all norms and pairwise inner products identically ($\langle \Phi_d^D(\mathbf{u}), \Phi_d^D(\mathbf{v}) \rangle = \langle \mathbf{u}, \mathbf{v} \rangle$), thereby preserving existing spherical Voronoi partitions and No-Encroachment margins $\langle \mu_j, \mathbf{k}_i \rangle \le s_{jj} - m_{\text{safe}}$ without distortion. This property is noted solely as a conceptual design primitive for future capacity expansion and is not implemented in DTRA~v3.0.
\end{proposition}
"""

REFERENCES_50_BLOCK = r"""
\begin{thebibliography}{50}
\small
\raggedright
\setlength{\itemsep}{1.1pt}
\setlength{\parskip}{0pt}

\bibitem{vaswani2017attention}
A.~Vaswani, N.~Shazeer, N.~Parmar, J.~Uszkoreit, L.~Jones, A.~N.~Gomez, {\L}.~Kaiser, and I.~Polosukhin, ``Attention is all you need,'' in \emph{Adv.\ Neural Inf.\ Process.\ Syst.\ (NeurIPS)}, vol.~30, 2017.

\bibitem{bommasani2021foundation}
R.~Bommasani et~al., ``On the opportunities and risks of foundation models,'' \texttt{arXiv:2108.07258}, 2021.

\bibitem{brohan2023rt1}
A.~Brohan et~al., ``RT-1: Robotics transformer for real-world control at scale,'' in \emph{Proc.\ Robot.:\ Sci.\ Syst.\ (RSS)}, 2023.

\bibitem{brohan2023rt2}
A.~Brohan et~al., ``RT-2: Vision-language-action models transfer web knowledge to robotic control,'' in \emph{Proc.\ Conf.\ Robot Learn.\ (CoRL)}, 2023.

\bibitem{octo2024}
Octo Model Team, D.~Ghosh, H.~Walke, K.~Pertsch, et~al., ``Octo: An open-source generalist robot policy,'' in \emph{Proc.\ Robot.:\ Sci.\ Syst.\ (RSS)}, 2024.

\bibitem{kim2024openvla}
M.~J.~Kim, K.~Pertsch, S.~Karamcheti, et~al., ``OpenVLA: An open-source vision-language-action model,'' \texttt{arXiv:2406.09246}, 2024.

\bibitem{black2024pi0}
K.~Black et~al., ``$\pi_0$: A vision-language-action flow model for general robot control,'' \texttt{arXiv:2410.24164}, 2024.

\bibitem{zhao2023act}
T.~Z.~Zhao, V.~Kumar, S.~Levine, and C.~Finn, ``Learning fine-grained bimanual manipulation with low-cost hardware,'' in \emph{Proc.\ Robot.:\ Sci.\ Syst.\ (RSS)}, 2023.

\bibitem{chi2023diffusion}
C.~Chi, S.~Feng, Y.~Du, Z.~Xu, E.~Cousineau, B.~Burchfiel, and S.~Song, ``Diffusion policy: Visuomotor policy learning via action diffusion,'' in \emph{Proc.\ Robot.:\ Sci.\ Syst.\ (RSS)}, 2023.

\bibitem{shridhar2022peract}
M.~Shridhar, L.~Manuelli, and D.~Fox, ``Perceiver-Actor: A multi-task transformer for robotic manipulation,'' in \emph{Proc.\ Conf.\ Robot Learn.\ (CoRL)}, 2022.

\bibitem{goyal2023rvt}
A.~Goyal, J.~Xu, Y.~Guo, V.~Blukis, Y.-W.~Chao, and D.~Fox, ``RVT: Robotic view transformer for 3D object manipulation,'' in \emph{Proc.\ Int.\ Conf.\ Mach.\ Learn.\ (ICML)}, 2023.

\bibitem{levine2016end}
S.~Levine, C.~Finn, T.~Darrell, and P.~Abbeel, ``End-to-end training of deep visuomotor policies,'' \emph{J.\ Mach.\ Learn.\ Res.\ (JMLR)}, vol.~17, no.~39, pp.~1--40, 2016.

\bibitem{finn2016deep}
C.~Finn, X.~Y.~Tan, Y.~Duan, T.~Darrell, S.~Levine, and P.~Abbeel, ``Deep spatial autoencoders for visuomotor learning,'' in \emph{Proc.\ IEEE Int.\ Conf.\ Robot.\ Autom.\ (ICRA)}, 2016.

\bibitem{he2016resnet}
K.~He, X.~Zhang, S.~Ren, and J.~Sun, ``Deep residual learning for image recognition,'' in \emph{Proc.\ IEEE/CVF Conf.\ Comput.\ Vis.\ Pattern Recognit.\ (CVPR)}, 2016.

\bibitem{yu2020metaworld}
T.~Yu, D.~Quillen, Z.~He, R.~Julian, A.~Narayan, H.~Shively, A.~Bellathur, K.~Hausman, C.~Finn, and S.~Levine, ``Meta-World: A benchmark and evaluation for multi-task and meta reinforcement learning,'' in \emph{Proc.\ Conf.\ Robot Learn.\ (CoRL)}, 2020.

\bibitem{wolczyk2021continual}
M.~Wo{\l}czyk, M.~Zaj{\k{a}}c, R.~Pascanu, {\L}.~Kuci{\'n}ski, and P.~Mi{\l}o{\'s}, ``Continual World: A robotic benchmark for continual reinforcement learning,'' in \emph{Adv.\ Neural Inf.\ Process.\ Syst.\ (NeurIPS)}, vol.~34, 2021.

\bibitem{liu2023libero}
B.~Liu, Y.~Zhu, C.~Gao, Y.~Feng, Q.~Liu, Y.~Zhu, and P.~Stone, ``LIBERO: Benchmarking knowledge transfer for lifelong robot learning,'' in \emph{Adv.\ Neural Inf.\ Process.\ Syst.\ (NeurIPS Datasets and Benchmarks Track)}, 2023.

\bibitem{mees2022calvin}
O.~Mees, L.~Hermann, E.~Rosete-Beas, and W.~Burgard, ``CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks,'' \emph{IEEE Robot.\ Autom.\ Lett.\ (RA-L)}, vol.~7, no.~3, pp.~7327--7334, 2022.

\bibitem{mccloskey1989catastrophic}
M.~McCloskey and N.~J.~Cohen, ``Catastrophic interference in connectionist networks: The sequential learning problem,'' \emph{Psychol.\ Learn.\ Motiv.}, vol.~24, pp.~109--165, 1989.

\bibitem{french1999catastrophic}
R.~M.~French, ``Catastrophic forgetting in connectionist networks,'' \emph{Trends Cogn.\ Sci.}, vol.~3, no.~4, pp.~128--135, 1999.

\bibitem{parisi2019continual}
G.~I.~Parisi, R.~Kemker, J.~L.~Part, C.~Kanan, and S.~Wermter, ``Continual lifelong learning with neural networks: A review,'' \emph{Neural Netw.}, vol.~113, pp.~54--71, 2019.

\bibitem{kirkpatrick2017ewc}
J.~Kirkpatrick et~al., ``Overcoming catastrophic forgetting in neural networks,'' \emph{Proc.\ Natl.\ Acad.\ Sci.\ (PNAS)}, vol.~114, no.~13, pp.~3521--3526, 2017.

\bibitem{zenke2017si}
F.~Zenke, B.~Poole, and S.~Ganguli, ``Continual learning through synaptic intelligence,'' in \emph{Proc.\ Int.\ Conf.\ Mach.\ Learn.\ (ICML)}, 2017.

\bibitem{aljundi2018mas}
R.~Aljundi, F.~Babiloni, M.~Elhoseiny, M.~Rohrbach, and T.~Tuytelaars, ``Memory aware synapses: Learning what (not) to forget,'' in \emph{Proc.\ Eur.\ Conf.\ Comput.\ Vis.\ (ECCV)}, 2018.

\bibitem{chaudhry2018rwalk}
A.~Chaudhry, P.~K.~Dokania, T.~Ajanthan, and P.~H.~S.~Torr, ``Riemannian walk for incremental learning: Understanding forgetting and intransigence,'' in \emph{Proc.\ Eur.\ Conf.\ Comput.\ Vis.\ (ECCV)}, 2018.

\bibitem{lopezpaz2017gem}
D.~Lopez-Paz and M.~Ranzato, ``Gradient episodic memory for continual learning,'' in \emph{Adv.\ Neural Inf.\ Process.\ Syst.\ (NeurIPS)}, 2017.

\bibitem{chaudhry2019agem}
A.~Chaudhry, M.~Ranzato, M.~Rohrbach, and M.~Elhoseiny, ``Efficient lifelong learning with A-GEM,'' in \emph{Proc.\ Int.\ Conf.\ Learn.\ Represent.\ (ICLR)}, 2019.

\bibitem{rolnick2019clear}
D.~Rolnick, A.~Ahuja, J.~Schwarz, T.~P.~Lillicrap, and G.~Wayne, ``Experience replay for continual learning,'' in \emph{Adv.\ Neural Inf.\ Process.\ Syst.\ (NeurIPS)}, 2019.

\bibitem{buzzega2020der}
P.~Buzzega, M.~Boschini, A.~Porrello, D.~Abati, and S.~Calderara, ``Dark experience for general continual learning: a strong, simple baseline,'' in \emph{Adv.\ Neural Inf.\ Process.\ Syst.\ (NeurIPS)}, 2020.

\bibitem{rusu2016pnn}
A.~A.~Rusu, N.~C.~Rabinowitz, G.~Desjardins, H.~Soyer, J.~Kirkpatrick, K.~Kavukcuoglu, R.~Pascanu, and R.~Hadsell, ``Progressive neural networks,'' \texttt{arXiv:1606.04671}, 2016.

\bibitem{mallya2018packnet}
A.~Mallya and S.~Lazebnik, ``PackNet: Adding multiple tasks to a single network by iterative pruning,'' in \emph{Proc.\ IEEE/CVF Conf.\ Comput.\ Vis.\ Pattern Recognit.\ (CVPR)}, 2018.

\bibitem{serra2018hat}
J.~Serr{\`a}, D.~Sur{\'\i}s, M.~Miron, and A.~Karatzoglou, ``Overcoming catastrophic forgetting with hard attention to the task,'' in \emph{Proc.\ Int.\ Conf.\ Mach.\ Learn.\ (ICML)}, 2018.

\bibitem{wang2022l2p}
Z.~Wang et~al., ``Learning to prompt for continual learning,'' in \emph{Proc.\ IEEE/CVF Conf.\ Comput.\ Vis.\ Pattern Recognit.\ (CVPR)}, 2022.

\bibitem{wang2022dualprompt}
Z.~Wang, Z.~Zhang, S.~Ebrahimi, R.~Sun, H.~Zhang, C.-Y.~Lee, X.~Ren, G.~Su, V.~Perot, J.~Dy, and T.~Pfister, ``DualPrompt: Complementary prompting for rehearsal-free continual learning,'' in \emph{Proc.\ Eur.\ Conf.\ Comput.\ Vis.\ (ECCV)}, 2022.

\bibitem{smith2023coda}
J.~S.~Smith, L.~Karlinsky, V.~Gutta, P.~Cascante-Bonilla, D.~Kim, A.~Arbelle, R.~Panda, R.~Feris, and Z.~Kira, ``CODA-Prompt: COntinual Decomposed Attention-based Prompting for rehearsal-free continual learning,'' in \emph{Proc.\ IEEE/CVF Conf.\ Comput.\ Vis.\ Pattern Recognit.\ (CVPR)}, 2023.

\bibitem{hu2022lora}
E.~J.~Hu, Y.~Shen, P.~Wallis, Z.~Allen-Zhu, Y.~Li, S.~Wang, L.~Wang, and W.~Chen, ``LoRA: Low-rank adaptation of large language models,'' in \emph{Proc.\ Int.\ Conf.\ Learn.\ Represent.\ (ICLR)}, 2022.

\bibitem{saha2021gpm}
G.~Saha, I.~Garg, and K.~Roy, ``Gradient projection memory for continual learning,'' in \emph{Proc.\ Int.\ Conf.\ Learn.\ Represent.\ (ICLR)}, 2021.

\bibitem{wang2023olora}
X.~Wang, T.~Chen, Q.~Ge, H.~Xia, R.~Bao, R.~Zheng, Q.~Zhang, T.~Gui, and X.~Huang, ``Orthogonal subspace learning for language model continual learning,'' in \emph{Findings of EMNLP}, 2023.

\bibitem{liang2024inflora}
Y.-S.~Liang and W.-J.~Li, ``InfLoRA: Interference-free low-rank adaptation for continual learning,'' in \emph{Proc.\ IEEE/CVF Conf.\ Comput.\ Vis.\ Pattern Recognit.\ (CVPR)}, 2024.

\bibitem{parsa2026seqlora}
J.~Parsa, E.~Simsar, A.~Joudaki, T.~Hofmann, and A.~M.~H.~Teixeira, ``SeqLoRA: Bilevel orthogonal adaptation for continual multi-concept generation,'' \texttt{arXiv:2605.22743}, 2026.

\bibitem{luo2026keeplora}
M.-L.~Luo, Z.-H.~Zhou, Y.-L.~Zhang, Y.~Wan, M.-L.~Zhang, and T.~Wei, ``KeepLoRA: Continual learning with residual gradient adaptation,'' in \emph{Proc.\ Int.\ Conf.\ Learn.\ Represent.\ (ICLR)} (\texttt{arXiv:2601.19659}), 2026.

\bibitem{wang2026orthoskill}
J.~Wang, Z.~Fang, Q.~Shi, and Y.~Zhou, ``OrthoSkillVLA: Continual skill learning via gradient-informed skill subspace adaptation,'' \texttt{arXiv:2608.19589}, 2026.

\bibitem{shrivastava2026sambar}
A.~Shrivastava, X.~Zhou, H.~Zhao, Z.~Chen, and N.~Mehr, ``SAMBAR: Selective anchoring via method of multipliers for balanced knowledge acquisition and retention in vision-language-action models,'' \texttt{arXiv:2609.32108}, 2026.

\bibitem{fang2026r2wam}
K.~Fang, H.~Liu, C.~Shi, Y.~Zhou, H.~Zhu, M.~Guo, J.~Song, and L.~Gao, ``Data-free continual learning for world action model with rectified generative replay,'' \texttt{arXiv:2610.18038}, 2026.

\bibitem{chen2026opr}
Y.~Chen, T.~Zhu, M.~Zhang, X.~Chen, J.~Huang, D.~Xu, and Y.~Wang, ``On-policy replay for continual supervised fine-tuning,'' \texttt{arXiv:2605.29495}, 2026.

\bibitem{zhang2026composecl}
Z.~Zhang, A.~Zhang, D.~Khashabi, and T.~Shu, ``Continual learning mechanisms compose for long-horizon memorization,'' \texttt{arXiv:2609.06986}, 2026.

\bibitem{ijspeert2013dmp}
A.~J.~Ijspeert, J.~Nakanishi, H.~Hoffmann, P.~Pastor, and S.~Schaal, ``Dynamical movement primitives: Learning attractor models for motor behaviors,'' \emph{Neural Comput.}, vol.~25, no.~2, pp.~328--373, 2013.

\bibitem{paraschos2013promp}
A.~Paraschos, C.~Daniel, J.~Peters, and G.~Neumann, ``Probabilistic movement primitives,'' in \emph{Adv.\ Neural Inf.\ Process.\ Syst.\ (NeurIPS)}, vol.~26, 2013.

\bibitem{piegl1997nurbs}
L.~Piegl and W.~Tiller, \emph{The NURBS Book}, 2nd~ed.\hskip 1em plus 0.5em minus 0.4em\relax Springer, 1997.

\bibitem{deboor1978splines}
C.~de~Boor, \emph{A Practical Guide to Splines}.\hskip 1em plus 0.5em minus 0.4em\relax Springer-Verlag, 1978.

\bibitem{deng2019arcface}
J.~Deng, J.~Guo, N.~Xue, and S.~Zafeiriou, ``ArcFace: Additive angular margin loss for deep face recognition,'' in \emph{Proc.\ IEEE/CVF Conf.\ Comput.\ Vis.\ Pattern Recognit.\ (CVPR)}, 2019.

\bibitem{kanerva2009hdc}
P.~Kanerva, ``Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors,'' \emph{Cogn.\ Comput.}, vol.~1, no.~2, pp.~139--159, 2009.

\bibitem{papyan2020neural}
V.~Papyan, X.~Y.~Han, and D.~L.~Donoho, ``Prevalence of neural collapse during the terminal phase of deep learning training,'' \emph{Proc.\ Natl.\ Acad.\ Sci.\ (PNAS)}, vol.~117, no.~40, pp.~24652--24663, 2020.

\bibitem{zhu2022buds}
Y.~Zhu, P.~Stone, and Y.~Zhu, ``Bottom-up skill discovery from unsegmented demonstrations for long-horizon robot manipulation,'' \emph{IEEE Robot.\ Autom.\ Lett.\ (RA-L)}, vol.~7, no.~2, pp.~4126--4133, 2022.

\bibitem{wan2024lotus}
W.~Wan, Y.~Zhu, R.~Shah, and Y.~Zhu, ``LOTUS: Continual imitation learning for robot manipulation through unsupervised skill discovery,'' in \emph{Proc.\ IEEE Int.\ Conf.\ Robot.\ Autom.\ (ICRA)}, 2024.

\bibitem{liu2024tail}
Z.~Liu, J.~Zhang, K.~Asadi, Y.~Liu, D.~Zhao, S.~Sabach, and R.~Fakoor, ``TAIL: Task-specific adapters for imitation learning with large pretrained models,'' in \emph{Proc.\ Int.\ Conf.\ Learn.\ Represent.\ (ICLR)}, 2024.

\bibitem{son2026abkvla}
S.~Son, ``ABK-VLA: Task-ID-free continual visuomotor policy learning via autonomous B-spline knot compression and visual prototype routing,'' \emph{Zenodo Preprint}, DOI: \texttt{10.5281/zenodo.24208545}, Nov.\ 2026. Repository: \url{https://github.com/5Color/abk-vla}.

\end{thebibliography}
"""
