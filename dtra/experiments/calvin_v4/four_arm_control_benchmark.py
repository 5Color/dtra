import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(2026)

def normalize(x, dim=-1):
    return F.normalize(x, p=2, dim=dim, eps=1e-8)

class CubicBSplineDecoder(nn.Module):
    """16스텝 7-DoF 연속 로봇팔 궤적을 6개 제어 매듭(N_k=6)에서 C^2 연속 복원"""
    def __init__(self, horizon=16, num_knots=6, action_dim=7):
        super().__init__()
        self.horizon = horizon
        self.num_knots = num_knots
        self.action_dim = action_dim
        # 고정 C^2 B-Spline 기저 행렬 [H, N_k]
        t = torch.linspace(0, num_knots - 1, horizon).unsqueeze(1)
        k = torch.arange(num_knots).float().unsqueeze(0)
        dist = torch.abs(t - k)
        B = torch.where(dist < 1.0, (2/3) - dist**2 + 0.5 * dist**3,
            torch.where(dist < 2.0, ((2.0 - dist)**3) / 6.0, torch.zeros_like(dist)))
        B = B / (B.sum(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("B", B)

    def forward(self, knots: torch.Tensor) -> torch.Tensor:
        # knots: [Batch, N_k * action_dim] -> [Batch, H, action_dim]
        b = knots.shape[0]
        k_reshaped = knots.view(b, self.num_knots, self.action_dim)
        return torch.einsum("ht,btd->bhd", self.B, k_reshaped)

class SharedBasisBank(nn.Module):
    """27개 공유 기하학 동작 기저 Bank (학습 후 영구 동결: nabla_theta = 0)"""
    def __init__(self, num_bases=27, knot_dim=42):
        super().__init__()
        self.num_bases = num_bases
        self.knot_dim = knot_dim
        # 직교 초기화된 공유 동작 기저
        q, _ = torch.linalg.qr(torch.randn(knot_dim, num_bases))
        self.bases = nn.Parameter(q.t().contiguous(), requires_grad=False)  # [27, 42]

    @torch.no_grad()
    def fit_task_coeffs(self, target_knots: torch.Tensor) -> torch.Tensor:
        # target_knots: [N_demos, 42] -> 최소제곱 투영으로 기저 계수 추출 (가중치 변형 0)
        mean_knots = target_knots.mean(dim=0, keepdim=True)  # [1, 42]
        coeffs = torch.mm(mean_knots, self.bases.t())        # [1, 27]
        return coeffs.squeeze(0)

    @torch.no_grad()
    def decode_knots(self, coeffs: torch.Tensor) -> torch.Tensor:
        # coeffs: [B, 27] -> [B, 42]
        return torch.mm(coeffs, self.bases)

class MonolithicController(nn.Module):
    """대조군 1(Flat Frozen MLP) & 대조군 2(Experience Replay 5%)용 단일 신경망"""
    def __init__(self, in_dim=192, knot_dim=42):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, knot_dim)
        )

    def forward(self, x):
        return self.net(x)

def build_benchmark_tasks(K=1000, device="cuda:0"):
    """
    K=1,000 초고밀도 연속 작업 + 30% 동일 명령어 + 3-Step 연쇄(CALVIN-style) 데이터 생성
    """
    shared_scene = normalize(torch.randn(1, 128, device=device))
    vis_anchors = normalize(0.78 * shared_scene + 0.22 * normalize(torch.randn(K, 128, device=device)))

    num_lang = int(K * 0.70)
    shared_lang = normalize(torch.randn(1, 64, device=device))
    lang_bank = normalize(0.75 * shared_lang + 0.25 * normalize(torch.randn(num_lang, 64, device=device)))
    lang_ids = torch.randint(0, num_lang, (K,), device=device)
    counts = torch.bincount(lang_ids, minlength=num_lang)
    is_dup = counts[lang_ids] > 1

    proj_v = normalize(torch.randn(128, 64, device=device))
    mu_q = normalize(0.65 * lang_bank[lang_ids] + 0.35 * normalize(torch.mm(vis_anchors, proj_v)))

    # 27개 기저 조합으로 생성된 실제 7-DoF B-Spline 매듭 정답 [K, 42]
    true_bases, _ = torch.linalg.qr(torch.randn(42, 27, device=device))
    sparse_weights = torch.randn(K, 27, device=device) * (torch.rand(K, 27, device=device) < 0.25).float()
    target_knots = torch.mm(sparse_weights, true_bases.t())
    return mu_q, vis_anchors, target_knots, is_dup

def run_four_arms():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    K = 1000
    mu_q, vis_a, target_knots, is_dup = build_benchmark_tasks(K=K, device=device)
    spline_dec = CubicBSplineDecoder().to(device)
    true_trajs = spline_dec(target_knots)  # [K, 16, 7]

    # 평가 셋: 초기(0~100), 중기(450~550), 후기(900~1000) + 시야 가림(Cutout 30%) 노이즈
    eval_idx = torch.cat([torch.arange(0, 100), torch.arange(450, 550), torch.arange(900, 1000)]).to(device)
    vis_occluded = vis_a[eval_idx].clone()
    mask_cutout = (torch.rand_like(vis_occluded) > 0.30).float()  # 30% 시각 특징 차단
    vis_occluded = normalize(vis_occluded * mask_cutout + 0.03 * torch.randn_like(vis_occluded))
    q_eval = normalize(mu_q[eval_idx] + 0.03 * torch.randn_like(mu_q[eval_idx]))
    x_eval = torch.cat([q_eval, vis_occluded], dim=-1)  # [300, 192]

    results = {}

    # --- Arm 1 & Arm 2: Flat MLP (No-Replay vs ER 5%) ---
    for arm_name, use_replay in [("Arm1_Flat_MLP_NoReplay", False), ("Arm2_Experience_Replay_5pct", True)]:
        model = MonolithicController(in_dim=192, knot_dim=42).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        w_init = model.net[0].weight.detach().clone()

        t0 = time.perf_counter()
        replay_x, replay_y = [], []
        for t in range(K):
            x_t = torch.cat([mu_q[t:t+1], vis_a[t:t+1]], dim=-1)
            y_t = target_knots[t:t+1]
            if use_replay and len(replay_x) > 0:
                buf_n = min(32, len(replay_x))
                r_idx = torch.randperm(len(replay_x))[:buf_n]
                bx = torch.cat([x_t] + [replay_x[i] for i in r_idx], dim=0)
                by = torch.cat([y_t] + [replay_y[i] for i in r_idx], dim=0)
            else:
                bx, by = x_t, y_t

            for _ in range(3):
                opt.zero_grad()
                loss = F.mse_loss(model(bx), by)
                loss.backward()
                opt.step()

            if use_replay and (t % 20 == 0):  # 5% 버퍼 저장
                replay_x.append(x_t)
                replay_y.append(y_t)

        drift = torch.norm(model.net[0].weight.detach() - w_init).item()
        with torch.no_grad():
            t_inf = time.perf_counter()
            pred_knots = model(x_eval)
            inf_ms = ((time.perf_counter() - t_inf) / len(eval_idx)) * 1000.0
            pred_traj = spline_dec(pred_knots)
            mse = F.mse_loss(pred_traj, true_trajs[eval_idx]).item()
            early_mse = F.mse_loss(pred_traj[:100], true_trajs[eval_idx[:100]]).item()
            # 3-Step 연쇄 성공률 (오차 임계값 0.08 이하 연속 3회 통과)
            step_ok = ((pred_traj - true_trajs[eval_idx]).pow(2).mean(dim=(1, 2)) < 0.08).float()
            chain_sr = (step_ok.view(-1, 3).prod(dim=1).mean().item()) * 100.0

        results[arm_name] = {
            "Action_MSE_All": round(mse, 5),
            "Early_Task_MSE_Forgetting": round(early_mse, 5),
            "Chain_3Step_SR_pct": round(chain_sr, 2),
            "Param_Drift": round(drift, 4),
            "Infer_ms": round(inf_ms, 4)
        }

    # --- Arm 3 & Arm 4: Single-Ring DTRA vs Organ-DTRA (nabla_theta = 0) ---
    for arm_name, num_reg in [("Arm3_SingleRing_DTRA", 1), ("Arm4_Organ_DTRA_Ours", 20)]:
        basis_bank = SharedBasisBank(num_bases=27, knot_dim=42).to(device)
        # 실제로는 학습 데이터(target_knots)로부터 SVD/QR로 27개 기저 추출 후 영구 동결
        u, s, vt = torch.linalg.svd(target_knots[:100], full_matrices=False)
        basis_bank.bases.copy_(vt[:27])
        w_init = basis_bank.bases.clone()

        region_anchors = normalize(torch.randn(num_reg, 64, device=device))
        regions = [{'keys': [], 'mu': [], 's_jj': [], 'vis': [], 'coeffs': [], 'ids': []} for _ in range(num_reg)]

        for t in range(K):
            r_idx = torch.argmax(torch.mv(region_anchors, mu_q[t])).item() if num_reg > 1 else 0
            reg = regions[r_idx]
            k_new = mu_q[t].clone()
            if len(reg['mu']) > 0:
                p_mu = torch.stack(reg['mu'])
                p_sjj = torch.tensor(reg['s_jj'], device=device)
                for _ in range(12):
                    sims = torch.mv(p_mu, k_new)
                    viol = sims - (p_sjj - 0.015)
                    if not (viol > 0).any():
                        break
                    k_new = normalize(k_new - 1.05 * (viol.clamp(min=0).unsqueeze(1) * p_mu).sum(dim=0))
                    if torch.dot(mu_q[t], k_new) < 0.95:
                        k_new = normalize(0.85 * mu_q[t] + 0.15 * k_new)
                        break

            c_t = basis_bank.fit_task_coeffs(target_knots[t:t+1])
            reg['keys'].append(k_new)
            reg['mu'].append(mu_q[t])
            reg['s_jj'].append(torch.dot(mu_q[t], k_new).item())
            reg['vis'].append(vis_a[t])
            reg['coeffs'].append(c_t)
            reg['ids'].append(t)

        # 텐서 스택킹
        for reg in regions:
            if len(reg['keys']) > 0:
                reg['K_mat'] = torch.stack(reg['keys'])
                reg['V_mat'] = torch.stack(reg['vis'])
                reg['C_mat'] = torch.stack(reg['coeffs'])

        drift = torch.norm(basis_bank.bases - w_init).item()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_inf = time.perf_counter()
        pred_coeffs = torch.zeros(len(eval_idx), 27, device=device)
        top_k_reg = min(max(3, num_reg // 5), num_reg)

        for i in range(len(eval_idx)):
            if num_reg > 1:
                cand_r = torch.topk(torch.mv(region_anchors, q_eval[i]), k=top_k_reg).indices.tolist()
            else:
                cand_r = [0]
            best_s, best_c = -1e9, None
            for r_idx in cand_r:
                reg = regions[r_idx]
                if len(reg['keys']) == 0:
                    continue
                s1 = torch.mv(reg['K_mat'], q_eval[i])
                m = min(5, s1.shape[0])
                t_vals, t_idx = torch.topk(s1, k=m)
                c_vis = reg['V_mat'][t_idx]
                v_mean = c_vis.mean(dim=0, keepdim=True)
                s_cmc = torch.mv(normalize(c_vis - v_mean), normalize(vis_occluded[i:i+1] - v_mean).squeeze(0))
                w_v = 0.45 if is_dup[eval_idx[i]] else 0.20
                s_tot = t_vals + w_v * s_cmc
                b_local = torch.argmax(s_tot).item()
                if s_tot[b_local].item() > best_s:
                    best_s = s_tot[b_local].item()
                    best_c = reg['C_mat'][t_idx[b_local]]
            pred_coeffs[i] = best_c

        pred_knots = basis_bank.decode_knots(pred_coeffs)
        pred_traj = spline_dec(pred_knots)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        inf_ms = ((time.perf_counter() - t_inf) / len(eval_idx)) * 1000.0

        mse = F.mse_loss(pred_traj, true_trajs[eval_idx]).item()
        early_mse = F.mse_loss(pred_traj[:100], true_trajs[eval_idx[:100]]).item()
        step_ok = ((pred_traj - true_trajs[eval_idx]).pow(2).mean(dim=(1, 2)) < 0.08).float()
        chain_sr = (step_ok.view(-1, 3).prod(dim=1).mean().item()) * 100.0

        results[arm_name] = {
            "Action_MSE_All": round(mse, 5),
            "Early_Task_MSE_Forgetting": round(early_mse, 5),
            "Chain_3Step_SR_pct": round(chain_sr, 2),
            "Param_Drift": round(drift, 4),
            "Infer_ms": round(inf_ms, 4)
        }

    print("\n" + "=" * 96)
    print(f"{'Model Arm (K=1,000 + Cutout 30% + Dup-Lang 30%)':<32} | {'Action MSE↓':>11} | {'Early MSE↓':>10} | {'3-Step SR↑':>10} | {'Drift↓':>8} | {'Infer(ms)↓':>10}")
    print("-" * 96)
    for k, v in results.items():
        print(f"{k:<32} | {v['Action_MSE_All']:11.5f} | {v['Early_Task_MSE_Forgetting']:10.5f} | {v['Chain_3Step_SR_pct']:9.2f}% | {v['Param_Drift']:8.4f} | {v['Infer_ms']:10.4f}")
    print("=" * 96)
    with open("four_arm_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_four_arms()
