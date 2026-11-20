"""scripts/build_dtra_v3_20p_parts.py

Modular scholarly text blocks (50 verified References, Formal Mathematical Proofs
for Appendix A, Architecture Comparison Table, and Method/Discussion prose) for
the 20-page NeurIPS/CoRL 10pt single-column DTRA v3.0 manuscript.
"""

from __future__ import annotations

METHOD_PARADIGM_TABLE = r"""
\begin{table}[t]
  \centering
  \caption{Structural comparison of continual visuomotor control paradigms. DTRA~v3.0 unifies Task-ID-free inference, zero rehearsal memory, exact parameter freezing ($\nabla_\theta = 0$), constant $O(1)$ active kernel execution, and sub-linear basis sharing within a white-box hyperspherical routing stack.}
  \label{tab:paradigm_compare}
  \fontsize{8.0pt}{9.8pt}\selectfont
  \setlength{\tabcolsep}{2.2pt}
  \begin{tabular}{lccccccc}
    \toprule
    \textbf{Paradigm / Method} & \textbf{Task-Free} & \textbf{No Rep.} & \textbf{$\nabla_\theta=0$} & \textbf{Active} & \textbf{Scaling} & \textbf{Dup-Lang} & \textbf{No VLM} \\
    \midrule
    Naive Fine-Tuning & \checkmark & \checkmark & $\times$ & $O(1)$ & Monolith & $\times$ & \checkmark \\
    EWC / RWalk~\cite{kirkpatrick2017ewc,chaudhry2018rwalk} & \checkmark & \checkmark & $\times$ & $O(1)$ & Monolith+Pen. & $\times$ & \checkmark \\
    GEM / ER / DER++~\cite{lopezpaz2017gem,rolnick2019clear,buzzega2020der} & \checkmark & $\times$ & $\times$ & $O(1)$+Buf & Monolith & $\times$ & \checkmark \\
    PNN / PackNet~\cite{rusu2016pnn,mallya2018packnet} & $\times$ & \checkmark & Masked & $O(K)$ & Linear $O(K)$ & N/A & \checkmark \\
    L2P / CODA-Prompt~\cite{wang2022l2p,smith2023coda} & \checkmark & \checkmark & Backb. & Pool & Prompts & Partial & ViT/CNN \\
    O-LoRA / Seq- / KeepLoRA~\cite{wang2023olora,parsa2026seqlora,luo2026keeplora} & Depends & Mostly & Adapter & $O(r)$ & LoRA & $\times$ & VLM/MLP \\
    OrthoSkillVLA / SAMBAR~\cite{wang2026orthoskill,shrivastava2026sambar} & Depends & \checkmark & Subsp. & VLA & Subsp./Anch. & $\times$ & VLA \\
    $\text{R}^2$-WAM~\cite{fang2026r2wam} & \checkmark & Gen.Rep. & $\times$ & WAM & Video WAM & $\times$ & WAM+VLM \\
    ABK-VLA v1.1.3~\cite{son2026abkvla} & \checkmark & \checkmark & \checkmark & $O(K)$ & Linear $O(K)$ & $\times$ (DoG) & \checkmark \\
    \textbf{DTRA v3.0 (Ours)} & \textbf{\checkmark} & \textbf{\checkmark} & \textbf{\checkmark ($0$)} & \textbf{$O(1)$} & \textbf{27 ($-73\%$)} & \textbf{\checkmark (CMC)} & \textbf{\checkmark} \\
    \bottomrule
  \end{tabular}
\end{table}
"""

ALGORITHM_1_BLOCK = r"""
\begin{algorithm}[t]
\caption{\textbf{DTRA v3.0 Training Registration \& Dual-Timescale Closed-Loop Control}}
\label{alg:dtra_v3}
\small
\begin{algorithmic}[1]
\Require Frozen perception + \texttt{LightweightTextEncoder} $\to \mathbf{x}_t \in \mathbb{S}^{127}$; frozen \texttt{SharedBasisBank} $\mathcal{B} = \{\mathbf{B}_b\}_{b=1}^{B}$ ($N_k = 6$); empty cell registry $\mathcal{C} = \emptyset$.
\Statex \textbf{Phase I: Sequential Continual Registration ($k = 1, \dots, K$, Task-ID Stored Only for Audit)}
\State Compute task prototype query $\mu_k \leftarrow \mathrm{norm}\!\left(\mathbb{E}_{t}[\mathbf{x}_{k,t}]\right)$ and topological map $\mathbf{k}_k^{(0)} = \mathrm{norm}(P_{\text{topo}}\mu_k) \in \mathbb{S}^{63}$.
\State Optimize $\mathbf{k}_k$ against prior cells $\{j < k\}$ subject to \textbf{No-Encroachment}: $\langle \mu_j, \mathbf{k}_k \rangle \le s_{jj} - m_{\text{safe}}$, caching raw anchor $\mu_k$ and self-similarity floor $s_{kk} = \langle \mu_k, \mathbf{k}_k \rangle$.
\State Fit $C^2$ clamped cubic B-Spline knots $\mathbf{P}_k \in \mathbb{R}^{6 \times d_a}$ on demos; match cosine similarity to $\mathcal{B}$. If below allocation threshold, append new frozen basis $\mathbf{B}_{B+1}$ ($\texttt{requires\_grad=False}$, excluded from \texttt{AdamW}); else share $\mathbf{B}_{b_k}$ with compact cell residual parameters $(\mathbf{r}_k, \alpha_k, \beta_k)$.
\State Freeze cell parameters permanently ($\nabla_\theta = 0$, $L_\infty$ silent drift $= 0.0$).
\Statex \textbf{Phase II: Replay-Free, Task-ID-Free Closed-Loop Evaluation (At Control Tick $t$)}
\State Extract multimodal $\mathbf{x}_t$ and project $\mathbf{c}_t = \mathrm{norm}(P_{\text{topo}}\mathbf{x}_t) \in \mathbb{S}^{63}$.
\State \textbf{Stage 1 (Region Scan):} Score all $K$ keys in $O(K)$ SIMD inner product and retrieve Top-$M$ ($M=5$) candidates $\mathcal{M}_t$.
\State \textbf{Stage 2 (Common-Mode Cancellation):} Compute candidate centroid $\bar{\mathbf{x}}_{\text{cm}} = \frac{1}{M}\sum_{j \in \mathcal{M}_t}\mathbf{x}_j^{\text{proto}}$, form differential residual $\tilde{\mathbf{x}}_t = \mathrm{norm}(\mathbf{x}_t - \gamma_{\text{cm}}\bar{\mathbf{x}}_{\text{cm}})$, and select active cell $i^* = \arg\max_{j \in \mathcal{M}_t} \langle \tilde{\mathbf{x}}_t, \tilde{\mathbf{x}}_j \rangle$ gated by $\tau_{i^*}$ and \texttt{episodic\_trace}.
\State Synthesize elastic hypertube displacement $\Delta \mathbf{z}_t^*$ via closed-form \texttt{atan2} phase projection bounded by $\alpha_{i^*} \in (0, 1)$, decode $H=16$ action chunk $\mathbf{A}_t = \mathbf{B}_{\text{spline}}\mathbf{P}_{\text{eff}}$, and execute \texttt{exec\_steps=8} steps.
\State Execute \texttt{chunk\_snap\_back()} ($\Delta\mathbf{z}_{i^*} \leftarrow \mathbf{0}$) per chunk and \texttt{episode\_snap\_back()} ($\text{\texttt{post\_snap\_norm}} = 0.0000$) upon episode termination.
\end{algorithmic}
\end{algorithm}
"""

APPENDIX_A_PROOFS = r"""
\section{Formal Mathematical Proofs \& Geometric Invariants}
\label{app:proofs}

This appendix provides complete proofs for the three core invariants asserted in Section~\ref{sec:method} and Section~\ref{sec:discussion}: exact zero parameter and residual drift under \texttt{AdamW} exclusion and Two-Timescale Snap-Back (Theorem~\ref{thm:zero_drift}), within-chunk $C^2$ cubic B-Spline continuity and output perturbation bounds (Theorem~\ref{thm:bspline}), and isometric dimension expansion via zero-padding (Theorem~\ref{thm:isometry}).

\subsection{Exact Zero Parameter \& Residual Drift Under Continual Updates}
\begin{theorem}[Zero Weight and Residual Drift]
\label{thm:zero_drift}
Let $\Theta_{\text{frozen}}^{(k)} = \{\mathbf{k}_j, \mathbf{B}_{b_j}, \mathbf{W}_{\text{head},j}\}_{j=1}^{k}$ denote the union of consolidated keys, shared motor bases in \texttt{SharedBasisBank}, and prior cell heads after mastering task $T_k$. Under DTRA registration and Two-Timescale Snap-Back:
\begin{enumerate}[leftmargin=*,itemsep=2pt]
  \item \textbf{Zero Optimizer Drift ($L_\infty = 0.0$):} For any subsequent task curriculum $T_{k+1}, \dots, T_K$ trained with decoupled weight decay \texttt{AdamW} ($\lambda > 0$), $\|\Theta_{\text{frozen}}^{(k)}(T_K) - \Theta_{\text{frozen}}^{(k)}(T_k)\|_\infty = 0.0$ holds identically.
  \item \textbf{Zero Inter-Episode State Contamination:} Across arbitrary rollout length $L$ and elastic compliance $\alpha_i \in (0,1)$, the post-episode ephemeral deformation norm satisfies $\lim_{\text{end}} \|\Delta\mathbf{z}_i\|_2 = 0.0000$ and $\mathbf{e}_t^{\text{trace}} = \mathbf{0}$ prior to every $t=0$ reset.
\end{enumerate}
\end{theorem}

\begin{proof}
(1) Standard \texttt{AdamW} parameter updates obey $\theta_{s+1} = (1 - \eta\lambda)\theta_s - \eta \hat{m}_s / (\sqrt{\hat{v}_s} + \epsilon)$. Even when a frozen parameter's gradient is zero ($\mathbf{g}_s = \nabla_\theta\mathcal{L} = \mathbf{0}$), leaving $\theta_s$ inside an optimizer parameter group induces exponential silent weight decay $(1 - \eta\lambda)^S \theta_0 \neq \theta_0$ over $S$ optimization steps. DTRA enforces two strict structural preconditions upon consolidating task $T_k$: first, setting \texttt{requires\_grad = False} on $\Theta_{\text{frozen}}^{(k)}$; second, constructing a fresh per-task \texttt{AdamW} parameter group containing exclusively newly instantiated active parameters $\Theta_{\text{active}}^{(k+1)}$ ($\Theta_{\text{frozen}}^{(k)} \cap \text{param\_groups}(\texttt{AdamW}) = \emptyset$). Consequently, every step satisfies $\theta_{s+1} = \theta_s$ identically in IEEE-754 floating-point arithmetic, guaranteeing exact $L_\infty = 0.0$.

(2) During control tick $t$, fast-timescale elastic tube displacement is synthesized purely as an ephemeral state tensor $\Delta \mathbf{z}_t^* = \alpha_{i^*} \cdot \Pi_{\mathcal{T}_{\mathbf{c}}(\mathbb{S}^{63})}(\mathbf{c}_t - \mathbf{c}_{\text{ring}, i^*}(\theta_t^*))$ stored outside persistent parameter memory. At the end of each $H=16$ chunk, \texttt{chunk\_snap\_back()} explicitly executes $\Delta\mathbf{z}_i \leftarrow \mathbf{0}$ while preserving the bounded progress counter \texttt{episodic\_trace} to prevent intra-episode perseveration loops. At episode completion, \texttt{episode\_snap\_back()} clears both $\Delta\mathbf{z}_i \leftarrow \mathbf{0}$ and $\mathbf{e}^{\text{trace}} \leftarrow \mathbf{0}$, enforcing an exact $0.0000$ post-episode state norm before the next environment reset.
\end{proof}

\subsection{Clamped \texorpdfstring{$C^2$}{C2} Cubic B-Spline Continuity \& Output Perturbation Bound}
\begin{theorem}[Within-Chunk $C^2$ Continuity and Perturbation Bound]
\label{thm:bspline}
Let $U = \{0, 0, 0, 0, u_4, \dots, u_{N_k-1}, 1, 1, 1, 1\}$ be a clamped uniform knot vector of degree $p=3$ with $N_k = 6$ control points $\mathbf{P} \in \mathbb{R}^{6 \times d_a}$ evaluated over uniform chunk timesteps $\tau_h = h / (H-1)$ for $h = 0, \dots, H-1$ ($H=16$). Then:
\begin{enumerate}[leftmargin=*,itemsep=2pt]
  \item Every synthesized chunk trajectory $\mathbf{a}(\tau) = \sum_{n=0}^{5} B_{n,3}(\tau)\,\mathbf{p}_n$ satisfies exact endpoint clamping ($\mathbf{a}(0) = \mathbf{p}_0$, $\mathbf{a}(1) = \mathbf{p}_5$) and continuous position ($C^0$), velocity ($C^1$), and acceleration ($C^2$) at all interior knots.
  \item Because $\sum_{n=0}^{5} B_{n,3}(\tau) = 1$ with $B_{n,3}(\tau) \ge 0$ (convex-hull property), any bounded perturbation $\|\delta\mathbf{P}\|_\infty \le \epsilon$ induced by visual OOD noise or elastic deformation obeys $\sup_{\tau \in [0,1]} \|\delta\mathbf{a}(\tau)\|_\infty \le \epsilon$, bounding spline-output perturbations within a single chunk.
\end{enumerate}
\end{theorem}

\begin{proof}
By the Cox--de~Boor recursion on multiplicity-$4$ clamped boundary knots $u_0=u_1=u_2=u_3=0$ and $u_6=u_7=u_8=u_9=1$, the basis functions satisfy $B_{0,3}(0) = 1$, $B_{5,3}(1) = 1$, and non-negativity $B_{n,3}(\tau) \ge 0$ with partition of unity $\sum_{n=0}^5 B_{n,3}(\tau) \equiv 1$ for all $\tau \in [0, 1]$. Each interior knot $u_j \in (0,1)$ has multiplicity $m_j = 1$, which guarantees $C^{p - m_j} = C^{3-1} = C^2$ continuity of $\mathbf{a}(\tau)$ across every interior polynomial transition. Finally, by H\"older's inequality and the partition-of-unity property:
\[
\left\| \sum_{n=0}^{5} B_{n,3}(\tau)\, \delta\mathbf{p}_n \right\|_\infty \le \left(\sum_{n=0}^5 |B_{n,3}(\tau)|\right) \max_{0 \le n \le 5} \|\delta\mathbf{p}_n\|_\infty = \|\delta\mathbf{P}\|_\infty \le \epsilon.
\]
This result applies within a single chunk. Replanning-boundary continuity and physical jerk or torque bounds require additional assumptions and are outside its scope.
\end{proof}

\subsection{Isometric Zero-Padding Invariance for Dynamic Manifold Expansion}
\begin{theorem}[Isometric Zero-Padding Boundary Preservation]
\label{thm:isometry}
Suppose an existing Region stores anchored unit keys and query representations $\{\mu_j, \mathbf{k}_j\}_{j=1}^K \subset \mathbb{S}^{d-1}$. Define the isometric zero-padding operator $\Phi_d^{D}: \mathbb{S}^{d-1} \to \mathbb{S}^{D-1}$ ($D > d$) by $\Phi_d^D(\mathbf{v}) = [\mathbf{v}; \mathbf{0}_{D-d}]$. Then for all $\mathbf{u}, \mathbf{v}, \mathbf{k}_j, \mu_j \in \mathbb{S}^{d-1}$:
\begin{enumerate}[leftmargin=*,itemsep=2pt]
  \item Norm and inner-product preservation: $\|\Phi_d^D(\mathbf{v})\|_2 = 1$ and $\langle \Phi_d^D(\mathbf{u}), \Phi_d^D(\mathbf{v}) \rangle = \langle \mathbf{u}, \mathbf{v} \rangle$.
  \item Exact Voronoi partition \& No-Encroachment invariance: every previously valid No-Encroachment inequality $\langle \mu_j, \mathbf{k}_i \rangle \le s_{jj} - m_{\text{safe}}$ remains satisfied identically upon embedding into $\mathbb{S}^{D-1}$, while newly added tasks can allocate orthogonal residual components in the fresh subspace $\mathbf{0}_{D-d}^\perp$.
\end{enumerate}
\end{theorem}

\begin{proof}
For any $\mathbf{u}, \mathbf{v} \in \mathbb{S}^{d-1}$, direct expansion yields
\[
\langle \Phi_d^D(\mathbf{u}), \Phi_d^D(\mathbf{v}) \rangle = \sum_{r=1}^{d} u_r v_r + \sum_{r=d+1}^{D} 0 \cdot 0 = \langle \mathbf{u}, \mathbf{v} \rangle,
\]
so $\|\Phi_d^D(\mathbf{v})\|_2 = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle} = 1$, preserving geodesic distances $d_{\text{geo}} = \arccos\langle \mathbf{u}, \mathbf{v} \rangle$ and spherical chord distances $d_{\text{chord}} = \sqrt{2(1-\langle \mathbf{u},\mathbf{v}\rangle)}$ exactly. Similarly, self-similarity floors $s_{jj}' = \langle \Phi(\mu_j), \Phi(\mathbf{k}_j) \rangle = s_{jj}$ and pairwise routing boundaries $\{ \mathbf{c} \in \mathbb{S}^{D-1} : \langle \mathbf{c}, \Phi(\mathbf{k}_j) \rangle \ge \langle \mathbf{c}, \Phi(\mathbf{k}_i) \rangle \}$ induce the identical spherical Voronoi tessellation on the embedded $(d-1)$-submanifold, while any future key $\mathbf{k}_{\text{new}} = [\alpha \mathbf{v}_{\parallel}; \sqrt{1-\alpha^2}\mathbf{v}_\perp]$ can expand capacity strictly along orthogonal coordinates $\mathbf{v}_\perp$ with tunable interference $\alpha \to 0$.
\end{proof}
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

\bibitem{son2026abkvla}
S.~Son, ``ABK-VLA: Task-ID-free continual visuomotor policy learning via autonomous B-spline knot compression and visual prototype routing,'' \emph{Zenodo Preprint}, DOI: \texttt{10.5281/zenodo.24208545}, Nov.\ 2026. Repository: \url{https://github.com/5Color/abk-vla}.

\end{thebibliography}
"""
