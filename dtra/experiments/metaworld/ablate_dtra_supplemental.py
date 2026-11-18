"""experiments/ablate_dtra_supplemental.py

3-Seed 본 실험 후속 보완 벤치마크 (Seed=42, K=8 Meta-World, 파라미터 일치 ~5.25M):
  1) [전통적 CL 대조군 3종] 동일 Conv4 + Hybrid Key + 6-mode B-Spline 공유 가중치(~5.25M)에
     (a) Naive Fine-Tuning, (b) EWC, (c) Experience Replay (과거 태스크 5% 버퍼) 순차 학습 비교
  2) [다중 OOD 섭동 강건성 곡선] Arm 3, Arm 4, Arm 5, Arm 6 (Seed 42) 체크포인트 저장·재로드 후
     추론 전용으로 Gaussian sigma=[0.05, 0.10, 0.20, 0.30], Cutout=[15%, 30%], 밝기/대비 왜곡 평가
  3) [역순 학습(Reverse Order: T7 -> T0) 불변성 검증] coffee-button-v3(T7) -> button-press-v3(T0)
     역순 시퀀스 학습 시 Arm 6의 FR 0.0% 및 RouteAcc 100.0% 유지 여부 검증

기존 Zenodo/3-seed 스크립트(benchmark_dtra_3d_metaworld.py, ablate_dtra_scientific.py)는 수정하지 않고
모델/데이터로더/평가 유틸을 재사용하는 독립 실행 스크립트입니다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V1_SRC_DIR = PROJECT_ROOT / "v1_drawing_vla" / "src"
for _p in (str(PROJECT_ROOT), str(V1_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from dtra.experiments.metaworld.ablate_dtra_scientific import (
    ControlledDTRAPolicy,
    HybridVisualKeyExtractor,
    MatchedABK3DPolicy,
    MatchedFrozenMLPPolicy,
    MetaWorldEvalRegistry,
    build_arm_factory,
    eval_arm_task,
    train_arm_task_fast,
)
from v1_drawing_vla.src.experiments.benchmark_autonomous_cl import calc_cl_metrics, set_seed
from dtra.experiments.metaworld.benchmark_dtra_3d_metaworld import (
    LightweightTextEncoder,
    OUT_DIR,
    TASK_INSTRUCTIONS,
    TASK_SPECS_8,
    apply_ood_rgb_perturbation,
    pick_device,
    sync,
)
from v1_drawing_vla.src.experiments.benchmark_metaworld_cl import (
    build_rgb_spatial_encoder,
    collect_or_load_task_data,
)
from dtra.core.bspline_core import BSplineDecoder, build_difference_matrices


# ==============================================================================
# 1. 가중치 공유 기반 전통적 지속학습 모델 (~5.25M Parameter-Matched)
# ==============================================================================


class SharedMultimodalPolicy(nn.Module):
    """Naive FT / EWC / Experience Replay 대조군용 가중치 공유 멀티모달 정책 (~5.25M).

    Arm 4 (MatchedFrozenMLPPolicy)와 동일한 Conv4 백본, State Net, Text Encoder,
    HybridVisualKeyExtractor([0.55*vis; 0.835*lang]), 6-mode WTA B-Spline 디코더를
    공유하되, 동결 슬롯 분기 없이 모든 파라미터를 공유하고 순차 갱신합니다.
    hidden_dim=1672 설정 시 전체 파라미터 수가 ~5.25M(5,246,746)으로 일치합니다.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        d_key: int = 128,
        hidden_dim: int = 1672,
        num_cp: int = 6,
        horizon: int = 16,
        action_dim: int = 4,
        num_modes: int = 6,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.d_key = d_key
        self.num_cp = num_cp
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_modes = num_modes

        self.encoder = build_rgb_spatial_encoder(feat_dim=feat_dim)
        self.state_net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.text_encoder = LightweightTextEncoder(d_lang=64)
        self.visual_key_extractor = HybridVisualKeyExtractor()

        in_dim = feat_dim + 64 + d_key  # 448
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.pts_head = nn.Linear(hidden_dim, num_modes * num_cp * action_dim)
        self.gate_head = nn.Linear(hidden_dim, num_modes)

        self.decoder = BSplineDecoder(num_control_points=num_cp, horizon=horizon, degree=3)
        _, d2_np = build_difference_matrices(num_cp)
        self.register_buffer("d2_mat", torch.tensor(d2_np, dtype=torch.float32))

    def reset(self) -> None:
        return None

    def _compose_key(self, vis_64: torch.Tensor, instructions: Sequence[str], device: torch.device) -> torch.Tensor:
        lang_64 = self.text_encoder(list(instructions), device=device)
        return F.normalize(torch.cat([0.55 * vis_64, 0.835 * lang_64], dim=-1), p=2, dim=-1)

    def forward_features(
        self, imgs: torch.Tensor, st: torch.Tensor, instructions: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(imgs)
        f_st = self.state_net(st)
        vis_64 = self.visual_key_extractor(imgs)
        x_mag = self._compose_key(vis_64, instructions, imgs.device)
        return torch.cat([x, f_st, x_mag], dim=-1), x_mag

    def loss(
        self,
        imgs: torch.Tensor,
        st: torch.Tensor,
        tgt_cp: torch.Tensor,
        tgt_act: torch.Tensor,
        instructions: Sequence[str],
    ) -> torch.Tensor:
        bsz = imgs.shape[0]
        feat, _ = self.forward_features(imgs, st, instructions)
        h = self.shared_mlp(feat)
        cand_cps = torch.sigmoid(self.pts_head(h)).view(bsz, self.num_modes, self.num_cp, self.action_dim)
        sel = self.gate_head(h)

        acts = (
            self.decoder(cand_cps.view(bsz * self.num_modes, self.num_cp, self.action_dim)).view(
                bsz, self.num_modes, self.horizon, self.action_dim
            )
            * 2.0
            - 1.0
        )
        l_act = torch.abs(acts - tgt_act[:, None]).mean(dim=(2, 3))
        l_cp = torch.abs(cand_cps - tgt_cp[:, None]).mean(dim=(2, 3))
        d2 = torch.matmul(self.d2_mat, cand_cps.view(bsz * self.num_modes, self.num_cp, self.action_dim)).view(
            bsz, self.num_modes, -1, self.action_dim
        )
        l_sm = (d2 ** 2).sum(dim=-1).mean(dim=-1)
        tot = l_act + 0.5 * l_cp + 0.01 * l_sm
        best = tot.argmin(dim=1)
        b_idx = torch.arange(bsz, device=imgs.device)
        return 0.85 * tot[b_idx, best].mean() + 0.05 * tot.mean() + 0.10 * F.cross_entropy(sel, best)

    @torch.no_grad()
    def predict(self, img: torch.Tensor, st: torch.Tensor, instruction: str) -> Tuple[np.ndarray, Dict[str, int]]:
        bsz = img.shape[0]
        feat, _ = self.forward_features(img, st, [instruction] * bsz)
        h = self.shared_mlp(feat)
        cand_cps = torch.sigmoid(self.pts_head(h)).view(bsz, self.num_modes, self.num_cp, self.action_dim)
        best_mode = self.gate_head(h).argmax(dim=1)
        p_star = cand_cps[torch.arange(bsz, device=img.device), best_mode].clamp(0.0, 1.0)
        act = self.decoder(p_star) * 2.0 - 1.0
        return act[0].detach().cpu().numpy(), {"slot_idx": 0}


def compute_empirical_fisher(
    model: SharedMultimodalPolicy,
    x_i: torch.Tensor,
    x_s: torch.Tensor,
    y_c: torch.Tensor,
    y_a: torch.Tensor,
    instruction: str,
    trainable_names_params: List[Tuple[str, nn.Parameter]],
    batch_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """태스크 완료 시점 경험적 대각 피셔 정보 행렬(empirical diagonal Fisher) 계산."""
    model.eval()
    fisher = {name: torch.zeros_like(p, device="cpu") for name, p in trainable_names_params}
    n_samples = x_i.shape[0]
    n_batches = 0
    for s in range(0, n_samples, batch_size):
        model.zero_grad(set_to_none=True)
        bi = x_i[s : s + batch_size]
        loss = model.loss(bi, x_s[s : s + batch_size], y_c[s : s + batch_size], y_a[s : s + batch_size], [instruction] * len(bi))
        loss.backward()
        for name, p in trainable_names_params:
            if p.grad is not None:
                fisher[name] += (p.grad.detach() ** 2).cpu()
        n_batches += 1
    model.zero_grad(set_to_none=True)
    denom = max(1, n_batches)
    for name in fisher:
        fisher[name] /= denom
    return fisher


def run_traditional_cl_baselines(
    tasks: Sequence[str],
    datasets: Dict[str, Dict],
    registry: MetaWorldEvalRegistry,
    device: torch.device,
    seed: int = 42,
    epochs: int = 55,
    n_eval: int = 15,
    n_ood_eval: int = 15,
    ewc_lambda: float = 500.0,
    er_ratio: float = 0.05,
) -> Dict[str, Dict]:
    """[보완 1] Naive FT / EWC / ER-5% 전통적 공유가중치 CL 대조군 3종 (Seed 42)."""
    methods = [
        ("cl_naive_ft", "Naive Fine-Tuning (Shared ~5.25M)"),
        ("cl_ewc", f"EWC (Shared ~5.25M, lambda={ewc_lambda:.0f})"),
        ("cl_er_5pct", f"Experience Replay {er_ratio*100:.0f}% Buffer (Shared ~5.25M)"),
    ]
    k_total = len(tasks)
    results: Dict[str, Dict] = {}

    for m_key, m_title in methods:
        print(f"\n=== [Part 1: {m_title}] (seed={seed}) ===", flush=True)
        set_seed(seed)
        model = SharedMultimodalPolicy().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        trainable_params = [p for _, p in trainable_named]

        r_mat = np.zeros((k_total, k_total), dtype=np.float32)
        ewc_anchors: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = []
        er_imgs: List[torch.Tensor] = []
        er_states: List[torch.Tensor] = []
        er_cps: List[torch.Tensor] = []
        er_acts: List[torch.Tensor] = []
        er_instrs: List[str] = []
        er_rng = np.random.default_rng(seed + 777)

        for s_idx, t_name in enumerate(tasks):
            ds = datasets[t_name]
            instr = TASK_INSTRUCTIONS[t_name]
            x_i = (torch.from_numpy(ds["imgs"]).permute(0, 3, 1, 2).float() / 255.0).to(device)
            x_s = torch.from_numpy(ds["states"]).to(device)
            y_c = torch.from_numpy(ds["cps"]).to(device)
            y_a = torch.from_numpy(ds["acts"]).to(device)
            n_samples = x_i.shape[0]
            batch_size = 64

            # EWC 페널티용 이전 태스크 앵커를 GPU에 임시 캐싱
            gpu_ewc_anchors = []
            if m_key == "cl_ewc" and len(ewc_anchors) > 0:
                for f_cpu, th_cpu in ewc_anchors:
                    gpu_ewc_anchors.append(
                        (
                            {k: v.to(device) for k, v in f_cpu.items()},
                            {k: v.to(device) for k, v in th_cpu.items()},
                        )
                    )

            # ER 버퍼 텐서 준비
            if m_key == "cl_er_5pct" and len(er_imgs) > 0:
                buf_i = torch.cat(er_imgs, dim=0).to(device)
                buf_s = torch.cat(er_states, dim=0).to(device)
                buf_c = torch.cat(er_cps, dim=0).to(device)
                buf_a = torch.cat(er_acts, dim=0).to(device)
                buf_n = buf_i.shape[0]
            else:
                buf_i = buf_s = buf_c = buf_a = None
                buf_n = 0

            opt = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

            t0 = time.perf_counter()
            for _ in range(epochs):
                model.train()
                perm = torch.randperm(n_samples, device="cpu").to(device)
                for s in range(0, n_samples, batch_size):
                    idx = perm[s : s + batch_size]
                    loss = model.loss(x_i[idx], x_s[idx], y_c[idx], y_a[idx], [instr] * len(idx))

                    if m_key == "cl_ewc" and len(gpu_ewc_anchors) > 0:
                        ewc_pen = torch.zeros((), device=device)
                        for f_map, th_map in gpu_ewc_anchors:
                            for p_name, p_cur in trainable_named:
                                ewc_pen = ewc_pen + torch.sum(f_map[p_name] * (p_cur - th_map[p_name]) ** 2)
                        loss = loss + 0.5 * ewc_lambda * ewc_pen

                    if m_key == "cl_er_5pct" and buf_n > 0:
                        rb_n = min(batch_size, buf_n)
                        rb_idx_cpu = torch.randint(0, buf_n, (rb_n,))
                        rb_idx = rb_idx_cpu.to(device)
                        rb_instr = [er_instrs[int(k)] for k in rb_idx_cpu.tolist()]
                        loss_rep = model.loss(
                            buf_i[rb_idx], buf_s[rb_idx], buf_c[rb_idx], buf_a[rb_idx], rb_instr
                        )
                        loss = 0.5 * loss + 0.5 * loss_rep

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    opt.step()
                sched.step()
            sync(device)
            tr_sec = time.perf_counter() - t0

            if m_key == "cl_ewc" and s_idx < k_total - 1:
                f_cur = compute_empirical_fisher(model, x_i, x_s, y_c, y_a, instr, trainable_named)
                th_cur = {n: p.detach().clone().cpu() for n, p in trainable_named}
                ewc_anchors.append((f_cur, th_cur))
                f_sum = float(sum(v.sum().item() for v in f_cur.values()))
                print(f"  [EWC Fisher T{s_idx}] total_diag_fisher={f_sum:.4f}", flush=True)

            if m_key == "cl_er_5pct" and s_idx < k_total - 1:
                keep_k = max(1, int(round(er_ratio * n_samples)))
                chosen = er_rng.choice(n_samples, size=keep_k, replace=False)
                chosen_t = torch.from_numpy(chosen).long()
                er_imgs.append(x_i[chosen_t.to(device)].cpu())
                er_states.append(x_s[chosen_t.to(device)].cpu())
                er_cps.append(y_c[chosen_t.to(device)].cpu())
                er_acts.append(y_a[chosen_t.to(device)].cpu())
                er_instrs.extend([instr] * keep_k)
                print(f"  [ER Buffer T{s_idx}] +{keep_k} samples (total={len(er_instrs)})", flush=True)

            del gpu_ewc_anchors, buf_i, buf_s, buf_c, buf_a

            for e_idx in range(s_idx + 1):
                e_name = tasks[e_idx]
                ev = eval_arm_task(
                    model, registry, e_name, device, n_eval=n_eval, task_id=None, split="train_heldout", seed=seed
                )
                r_mat[e_idx, s_idx] = ev["success_rate"]
                print(
                    f"    Eval T{e_idx} {e_name:17s} | SR={ev['success_rate']*100:5.1f}% | train={tr_sec:.1f}s",
                    flush=True,
                )

        cl_m = calc_cl_metrics(r_mat)
        ood_srs = []
        for e_idx, e_name in enumerate(tasks):
            ev_o = eval_arm_task(
                model,
                registry,
                e_name,
                device,
                n_eval=n_ood_eval,
                task_id=None,
                split="train_heldout",
                noise_std=0.20,
                seed=seed + 2000,
            )
            ood_srs.append(float(ev_o["success_rate"]))

        results[m_key] = {
            "title": m_title,
            "total_params": int(total_params),
            "R": r_mat.tolist(),
            "clean_avg_sr": float(cl_m["avg_final_f1"]),
            "fr": float(cl_m["avg_forgetting"]),
            "bwt": float(cl_m["avg_bwt"]),
            "clean_per_task": [float(r_mat[i, k_total - 1]) for i in range(k_total)],
            "ood_severe_sr": float(np.mean(ood_srs)),
            "ood_per_task": ood_srs,
        }
        print(
            f"  [Summary {m_key}] Params={total_params:,} | Clean SR={cl_m['avg_final_f1']*100:5.1f}% "
            f"(FR={cl_m['avg_forgetting']*100:4.1f}%, BWT={cl_m['avg_bwt']*100:+5.1f}%) | "
            f"OOD={np.mean(ood_srs)*100:5.1f}%",
            flush=True,
        )
    return results


# ==============================================================================
# 2. 다중 OOD 섭동 강건성 곡선 (체크포인트 저장·재로드 후 순수 추론 평가)
# ==============================================================================


def apply_multi_ood_perturbation(
    tx: torch.Tensor,
    perturbation: str,
    rng: torch.Generator,
) -> torch.Tensor:
    if perturbation == "clean":
        return tx
    if perturbation.startswith("noise_"):
        sigma = float(perturbation.split("_")[1])
        return apply_ood_rgb_perturbation(tx, sigma, 1.0, 0.0, rng)
    if perturbation.startswith("cutout_"):
        ratio = float(perturbation.split("_")[1]) / 100.0
        bsz, _, h, w = tx.shape
        side = max(1, int(round(math.sqrt(ratio) * min(h, w))))
        out = tx.clone()
        for b in range(bsz):
            top = int(torch.randint(0, max(1, h - side + 1), (1,), generator=rng).item())
            left = int(torch.randint(0, max(1, w - side + 1), (1,), generator=rng).item())
            out[b, :, top : top + side, left : left + side] = 0.0
        return out
    if perturbation == "bright_high":
        return apply_ood_rgb_perturbation(tx, 0.0, 0.65, 0.22, rng)
    if perturbation == "contrast_harsh":
        return apply_ood_rgb_perturbation(tx, 0.0, 1.40, -0.18, rng)
    raise ValueError(f"Unknown perturbation: {perturbation}")


def eval_arm_with_perturbation(
    model: nn.Module,
    registry: MetaWorldEvalRegistry,
    task_name: str,
    device: torch.device,
    perturbation: str,
    task_id: int,
    n_eval: int = 15,
    exec_steps: int = 8,
    seed: int = 42,
) -> Tuple[float, float]:
    env, eval_tasks = registry.get_env_and_tasks(task_name, split="train_heldout", n_eval=n_eval)
    model.eval()
    instr = TASK_INSTRUCTIONS[task_name]
    rng = torch.Generator().manual_seed(seed + task_id * 100)

    succs = 0
    matched_slots = 0
    try:
        for t in eval_tasks:
            env.set_task(t)
            obs, _ = env.reset()
            model.reset()
            ep_matched = False
            done = False
            for step_idx in range(0, 150, exec_steps):
                fr = np.flipud(env.render()).copy()
                tx = torch.from_numpy(fr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                tx = apply_multi_ood_perturbation(tx, perturbation, rng)
                ts = torch.from_numpy(obs[:4].astype(np.float32)).unsqueeze(0).to(device)

                if isinstance(model, ControlledDTRAPolicy):
                    chunk, info_route = model.predict(tx, ts, instruction=instr, collect_telemetry=False)
                    if step_idx == 0:
                        ep_matched = int(info_route["fine_idx"]) == task_id
                else:
                    chunk, info_route = model.predict(tx, ts, instruction=instr)
                    if step_idx == 0:
                        ep_matched = int(info_route["slot_idx"]) == task_id

                for a in chunk[:exec_steps]:
                    obs, _, _, _, info = env.step(np.clip(a, -1.0, 1.0))
                    if info.get("success", 0) > 0:
                        done = True
                        break
                if done:
                    break
            matched_slots += int(ep_matched)
            succs += int(done)
    finally:
        env.close()

    n = max(1, len(eval_tasks))
    return float(succs / n), float(matched_slots / n)


def train_and_save_or_load_checkpoint(
    arm_key: str,
    tasks: Sequence[str],
    datasets: Dict[str, Dict],
    ckpt_dir: Path,
    device: torch.device,
    seed: int = 42,
    epochs: int = 55,
) -> nn.Module:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{arm_key}_seed{seed}.pt"

    if not ckpt_path.exists():
        print(f"  [Checkpoint Train] {arm_key} (seed={seed}) -> {ckpt_path.name}", flush=True)
        set_seed(seed)
        model = build_arm_factory(arm_key)().to(device)
        for s_idx, t_name in enumerate(tasks):
            ds = datasets[t_name]
            init_t = torch.from_numpy(ds["init_frames"][:8]).permute(0, 3, 1, 2).float() / 255.0
            model.register_or_expand_task(init_t, TASK_INSTRUCTIONS[t_name])
            train_arm_task_fast(model, ds, TASK_INSTRUCTIONS[t_name], device, epochs=epochs, lr=1e-3, task_idx=s_idx)
            if s_idx == 0:
                model.freeze_prior()
        torch.save(
            {
                "arm_key": arm_key,
                "seed": seed,
                "tasks": list(tasks),
                "state_dict": model.state_dict(),
            },
            ckpt_path,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  [Checkpoint Load] {ckpt_path.name}", flush=True)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    reloaded = build_arm_factory(arm_key)().to(device)
    for s_idx, t_name in enumerate(blob["tasks"]):
        ds = datasets[t_name]
        init_t = torch.from_numpy(ds["init_frames"][:8]).permute(0, 3, 1, 2).float() / 255.0
        reloaded.register_or_expand_task(init_t, TASK_INSTRUCTIONS[t_name])
        if s_idx == 0:
            reloaded.freeze_prior()
    reloaded.load_state_dict(blob["state_dict"], strict=True)
    if hasattr(reloaded, "refresh_eval_caches"):
        reloaded.refresh_eval_caches()
    reloaded.eval()
    return reloaded


def run_multi_ood_curve(
    tasks: Sequence[str],
    datasets: Dict[str, Dict],
    registry: MetaWorldEvalRegistry,
    ckpt_dir: Path,
    device: torch.device,
    seed: int = 42,
    epochs: int = 55,
    n_eval: int = 15,
) -> Dict[str, Dict]:
    target_arms = [
        "arm3_dtra_full_vision",
        "arm4_frozen_mlp_multimodal",
        "arm5_dtra_rigid_alpha0",
        "arm6_dtra_full_multimodal",
    ]
    perturbations = [
        "clean",
        "noise_0.05",
        "noise_0.10",
        "noise_0.20",
        "noise_0.30",
        "cutout_15",
        "cutout_30",
        "bright_high",
        "contrast_harsh",
    ]

    print(f"\n=== [Part 2: Multi-OOD Perturbation Curve on Reloaded Seed {seed} Checkpoints] ===", flush=True)
    curve: Dict[str, Dict] = {}
    for arm_key in target_arms:
        model = train_and_save_or_load_checkpoint(arm_key, tasks, datasets, ckpt_dir, device, seed=seed, epochs=epochs)
        arm_res: Dict[str, Dict] = {}
        for pert in perturbations:
            srs, rts = [], []
            for e_idx, e_name in enumerate(tasks):
                sr, rt = eval_arm_with_perturbation(
                    model, registry, e_name, device, pert, task_id=e_idx, n_eval=n_eval, seed=seed + 2000
                )
                srs.append(sr)
                rts.append(rt)
            arm_res[pert] = {
                "avg_sr": float(np.mean(srs)),
                "avg_route_acc": float(np.mean(rts)),
                "per_task_sr": srs,
                "per_task_route": rts,
            }
            print(
                f"  [{arm_key:26s}] {pert:15s} | SR={np.mean(srs)*100:5.1f}% | RouteAcc={np.mean(rts)*100:5.1f}%",
                flush=True,
            )
        curve[arm_key] = arm_res
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return curve


# ==============================================================================
# 3. 태스크 학습 순서 역전 (Reverse Order: T7 -> T0) 불변성 검증
# ==============================================================================


def run_reverse_order_arm6(
    tasks: Sequence[str],
    datasets: Dict[str, Dict],
    registry: MetaWorldEvalRegistry,
    device: torch.device,
    seed: int = 42,
    epochs: int = 55,
    n_eval: int = 15,
    n_ood_eval: int = 15,
    n_test_eval: int = 15,
) -> Dict[str, object]:
    rev_tasks = list(reversed(list(tasks)))
    k_total = len(rev_tasks)
    print(f"\n=== [Part 3: Reverse Order T7 -> T0 Invariance on Arm 6 (seed={seed})] ===", flush=True)
    print("Sequence:", " -> ".join(rev_tasks), flush=True)

    set_seed(seed)
    model = build_arm_factory("arm6_dtra_full_multimodal")().to(device)
    r_mat = np.zeros((k_total, k_total), dtype=np.float32)
    r_route = np.zeros((k_total, k_total), dtype=np.float32)

    for s_idx, t_name in enumerate(rev_tasks):
        ds = datasets[t_name]
        init_t = torch.from_numpy(ds["init_frames"][:8]).permute(0, 3, 1, 2).float() / 255.0
        info = model.register_or_expand_task(init_t, TASK_INSTRUCTIONS[t_name])
        print(
            f"  [Expand step={s_idx}] {t_name:17s} | C={info['coarse_idx']}(exp={info['coarse_expanded']}, sim={info['coarse_max_sim']:.4f}) | "
            f"F={info['fine_idx']}(exp={info['fine_expanded']}, sim={info['fine_max_sim']:.4f})",
            flush=True,
        )
        _, tr_sec = train_arm_task_fast(
            model, ds, TASK_INSTRUCTIONS[t_name], device, epochs=epochs, lr=1e-3, task_idx=s_idx
        )
        if s_idx == 0:
            model.freeze_prior()

        for e_idx in range(s_idx + 1):
            e_name = rev_tasks[e_idx]
            ev = eval_arm_task(
                model, registry, e_name, device, n_eval=n_eval, task_id=e_idx, split="train_heldout", seed=seed
            )
            r_mat[e_idx, s_idx] = ev["success_rate"]
            r_route[e_idx, s_idx] = ev["routing_top1_acc"] or 0.0
            print(
                f"    Eval step={e_idx} {e_name:17s} | SR={ev['success_rate']*100:5.1f}% | "
                f"Route={ev['routing_top1_acc']*100:5.1f}% | train={tr_sec:.1f}s",
                flush=True,
            )

    cl_m = calc_cl_metrics(r_mat)
    clean_rts = [float(r_route[i, k_total - 1]) for i in range(k_total)]

    test_srs, test_rts = [], []
    ood_srs, ood_rts = [], []
    for e_idx, e_name in enumerate(rev_tasks):
        ev_t = eval_arm_task(
            model, registry, e_name, device, n_eval=n_test_eval, task_id=e_idx, split="official_test", seed=seed + 1000
        )
        test_srs.append(float(ev_t["success_rate"]))
        test_rts.append(float(ev_t["routing_top1_acc"] or 0.0))

        ev_o = eval_arm_task(
            model,
            registry,
            e_name,
            device,
            n_eval=n_ood_eval,
            task_id=e_idx,
            split="train_heldout",
            noise_std=0.20,
            seed=seed + 2000,
        )
        ood_srs.append(float(ev_o["success_rate"]))
        ood_rts.append(float(ev_o["routing_top1_acc"] or 0.0))

    res = {
        "sequence": rev_tasks,
        "R": r_mat.tolist(),
        "R_route": r_route.tolist(),
        "clean_avg_sr": float(cl_m["avg_final_f1"]),
        "fr": float(cl_m["avg_forgetting"]),
        "bwt": float(cl_m["avg_bwt"]),
        "clean_route_acc": float(np.mean(clean_rts)),
        "test_avg_sr": float(np.mean(test_srs)),
        "test_route_acc": float(np.mean(test_rts)),
        "ood_severe_sr": float(np.mean(ood_srs)),
        "ood_route_acc": float(np.mean(ood_rts)),
        "clean_per_task": [float(r_mat[i, k_total - 1]) for i in range(k_total)],
        "ood_per_task": ood_srs,
    }
    print(
        f"  [Summary Reverse T7->T0 Arm 6] Clean SR={cl_m['avg_final_f1']*100:5.1f}% "
        f"(FR={cl_m['avg_forgetting']*100:4.1f}%, BWT={cl_m['avg_bwt']*100:+5.1f}%, Route={np.mean(clean_rts)*100:5.1f}%) | "
        f"Test={np.mean(test_srs)*100:5.1f}% (Route={np.mean(test_rts)*100:5.1f}%) | "
        f"OOD={np.mean(ood_srs)*100:5.1f}% (Route={np.mean(ood_rts)*100:5.1f}%)",
        flush=True,
    )
    return res


def print_markdown_summaries(payload: Dict) -> None:
    print("\n" + "=" * 100)
    if "part1_traditional_cl" in payload:
        print("### Supplemental Table S1: Shared-Weight Traditional CL Baselines vs Frozen/DTRA (Seed=42, ~5.25M Params)")
        print("| Method | Params | Clean Avg SR | FR | BWT | OOD Severe SR (sigma=0.20) |")
        print("|---|---:|---:|---:|---:|---:|")
        for _, v in payload["part1_traditional_cl"].items():
            print(
                f"| {v['title']} | {v['total_params']/1e6:.2f}M | {v['clean_avg_sr']*100:5.1f}% | "
                f"{v['fr']*100:4.1f}% | {v['bwt']*100:+5.1f}% | {v['ood_severe_sr']*100:5.1f}% |"
            )

    if "part2_multi_ood_curve" in payload:
        print("\n### Supplemental Table S2: Multi-OOD Perturbation Robustness Curve (Seed=42 Checkpoints, SR / RouteAcc)")
        perts = ["clean", "noise_0.05", "noise_0.10", "noise_0.20", "noise_0.30", "cutout_15", "cutout_30", "bright_high", "contrast_harsh"]
        header = "| Arm | " + " | ".join(perts) + " |"
        sep = "|---|" + "|".join(["---:"] * len(perts)) + "|"
        print(header)
        print(sep)
        for ak, pd in payload["part2_multi_ood_curve"].items():
            cells = [f"{pd[p]['avg_sr']*100:4.1f}% ({pd[p]['avg_route_acc']*100:4.1f}%)" for p in perts]
            print(f"| {ak} | " + " | ".join(cells) + " |")

    if "part3_reverse_order_arm6" in payload:
        r3 = payload["part3_reverse_order_arm6"]
        print("\n### Supplemental Table S3: Task Order Invariance Check (Arm 6, Reverse Order T7 -> T0, Seed=42)")
        print("| Order | Clean Avg SR | FR | BWT | Clean RouteAcc | Official Test SR | OOD Severe SR | OOD RouteAcc |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        print(
            f"| Reverse (T7 -> T0) | {r3['clean_avg_sr']*100:5.1f}% | {r3['fr']*100:4.1f}% | {r3['bwt']*100:+5.1f}% | "
            f"{r3['clean_route_acc']*100:5.1f}% | {r3['test_avg_sr']*100:5.1f}% | {r3['ood_severe_sr']*100:5.1f}% | "
            f"{r3['ood_route_acc']*100:5.1f}% |"
        )
    print("=" * 100 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="DTRA Supplemental Benchmarks (Traditional CL, Multi-OOD, Reverse Order)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--n-demos", type=int, default=30)
    parser.add_argument("--n-eval", type=int, default=15)
    parser.add_argument("--n-ood-eval", type=int, default=15)
    parser.add_argument("--n-test-eval", type=int, default=15)
    parser.add_argument("--ewc-lambda", type=float, default=500.0)
    parser.add_argument("--er-ratio", type=float, default=0.05)
    parser.add_argument(
        "--parts",
        nargs="+",
        default=["part1", "part2", "part3"],
        choices=["part1", "part2", "part3"],
    )
    parser.add_argument("--ckpt-dir", type=str, default=str(OUT_DIR / "checkpoints_seed42"))
    parser.add_argument("--out-json", type=str, default=str(OUT_DIR / "ablate_dtra_supplemental_results.json"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    device = pick_device()
    tasks = [name for name, _ in TASK_SPECS_8]

    print("=" * 88)
    print(f"DTRA Supplemental Benchmarks | seed={args.seed} | epochs={args.epochs} | parts={args.parts}")
    print(f"[device] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else device}")
    print("=" * 88)

    datasets = {
        name: collect_or_load_task_data(name, pcls, n_episodes=args.n_demos, H=16, Nk=6)
        for name, pcls in TASK_SPECS_8
    }

    registry = MetaWorldEvalRegistry(tasks, ml1_seed=42)
    payload: Dict = {
        "seed": args.seed,
        "epochs": args.epochs,
        "tasks": tasks,
        "checkpoint_note": (
            "ablate_dtra_scientific.py는 .pth 체크포인트를 디스크에 저장하지 않았으므로, "
            "Part 2에서는 Seed 42 모델(Arm 3/4/5/6)을 1회 학습·저장(checkpoints_seed42/*.pt)한 뒤 "
            "재로드하여 다중 OOD 섭동 조건을 순수 추론으로 평가함."
        ),
    }

    try:
        registry.prepare_all()

        def flush_json() -> None:
            out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        if "part1" in args.parts:
            payload["part1_traditional_cl"] = run_traditional_cl_baselines(
                tasks,
                datasets,
                registry,
                device,
                seed=args.seed,
                epochs=args.epochs,
                n_eval=args.n_eval,
                n_ood_eval=args.n_ood_eval,
                ewc_lambda=args.ewc_lambda,
                er_ratio=args.er_ratio,
            )
            flush_json()

        if "part2" in args.parts:
            payload["part2_multi_ood_curve"] = run_multi_ood_curve(
                tasks,
                datasets,
                registry,
                ckpt_dir,
                device,
                seed=args.seed,
                epochs=args.epochs,
                n_eval=args.n_eval,
            )
            flush_json()

        if "part3" in args.parts:
            payload["part3_reverse_order_arm6"] = run_reverse_order_arm6(
                tasks,
                datasets,
                registry,
                device,
                seed=args.seed,
                epochs=args.epochs,
                n_eval=args.n_eval,
                n_ood_eval=args.n_ood_eval,
                n_test_eval=args.n_test_eval,
            )
            flush_json()

        print_markdown_summaries(payload)
        flush_json()
        print(f"[SAVED] Supplemental results -> {out_json}", flush=True)
    finally:
        pass


if __name__ == "__main__":
    main()
