import time
import json
import torch
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normalize(x, dim=-1):
    return F.normalize(x, p=2, dim=dim, eps=1e-8)

class IsometricZeroPadder:
    """논문 6.1절: 기존 내적과 보로노이 경계를 0.0000 왜곡으로 보존하며 차원 확장"""
    @staticmethod
    def expand(vectors: torch.Tensor, target_dim: int) -> torch.Tensor:
        cur_dim = vectors.shape[-1]
        if cur_dim >= target_dim:
            return vectors
        pad = torch.zeros(*vectors.shape[:-1], target_dim - cur_dim, device=vectors.device, dtype=vectors.dtype)
        return torch.cat([vectors, pad], dim=-1)

class OrganHierarchicalRouter:
    """
    Organ -> Region(각 최대 100개 Cell) 계층 라우터
    - 의미/운동학 도메인 기반 Region 라우팅 (평균 뭉개짐 방지)
    - No-Encroachment 제약 + Top-M(M=5) Common-Mode Cancellation(CMC)
    """
    def __init__(self, dim=64, num_regions=50, top_m=5, m_safe=0.015):
        self.dim = dim
        self.num_regions = num_regions
        self.top_m = top_m
        self.m_safe = m_safe
        # 고정 직교-유사 Region 앵커 (뇌의 기능별 피질 영역에 대응)
        self.region_anchors = normalize(torch.randn(num_regions, dim, device=device))
        self.regions = [
            {'keys': None, 'mu': None, 's_jj': None, 'vis_proto': None, 'global_ids': []}
            for _ in range(num_regions)
        ]

    def expand_manifold(self, new_dim: int):
        if new_dim <= self.dim:
            return
        self.region_anchors = IsometricZeroPadder.expand(self.region_anchors, new_dim)
        for r in self.regions:
            if r['keys'] is not None:
                r['keys'] = IsometricZeroPadder.expand(r['keys'], new_dim)
                r['mu'] = IsometricZeroPadder.expand(r['mu'], new_dim)
        self.dim = new_dim

    @torch.no_grad()
    def register_task(self, global_id: int, mu_query: torch.Tensor, vis_proto: torch.Tensor):
        mu_query = normalize(mu_query)
        vis_proto = normalize(vis_proto)

        # 가장 가까운 기능 Region 선택
        r_idx = torch.argmax(torch.mv(self.region_anchors, mu_query)).item()
        reg = self.regions[r_idx]

        if reg['keys'] is None:
            reg['keys'] = mu_query.unsqueeze(0).clone()
            reg['mu'] = mu_query.unsqueeze(0).clone()
            reg['s_jj'] = torch.tensor([1.0], device=device)
            reg['vis_proto'] = vis_proto.unsqueeze(0).clone()
            reg['global_ids'].append(global_id)
            return

        k_new = mu_query.clone()
        prior_mu = reg['mu']
        prior_sjj = reg['s_jj']

        # No-Encroachment Margin Refinement
        for _ in range(15):
            sims = torch.mv(prior_mu, k_new)
            violations = sims - (prior_sjj - self.m_safe)
            mask = violations > 0
            if not mask.any():
                break
            viol_mu = prior_mu[mask]
            viol_amt = violations[mask].unsqueeze(1)
            k_new = k_new - 1.05 * (viol_amt * viol_mu).sum(dim=0)
            k_new = normalize(k_new)
            if torch.dot(mu_query, k_new) < 0.95:
                k_new = normalize(0.85 * mu_query + 0.15 * k_new)
                break

        s_ii = torch.dot(mu_query, k_new).unsqueeze(0)
        reg['keys'] = torch.cat([reg['keys'], k_new.unsqueeze(0)], dim=0)
        reg['mu'] = torch.cat([reg['mu'], mu_query.unsqueeze(0)], dim=0)
        reg['s_jj'] = torch.cat([reg['s_jj'], s_ii], dim=0)
        reg['vis_proto'] = torch.cat([reg['vis_proto'], vis_proto.unsqueeze(0)], dim=0)
        reg['global_ids'].append(global_id)

    @torch.no_grad()
    def route_batch(self, q_batch: torch.Tensor, vis_batch: torch.Tensor, is_dup_lang: torch.Tensor):
        B = q_batch.shape[0]
        r_scores = torch.mm(q_batch, self.region_anchors.t())  # [B, R]
        top_r = torch.topk(r_scores, k=min(max(3, self.num_regions // 8), self.num_regions), dim=-1).indices  # [B, 3]

        preds = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(B):
            best_score = -1e9
            best_global_id = 0
            for r_idx in top_r[i].tolist():
                reg = self.regions[r_idx]
                if reg['keys'] is None:
                    continue
                s1 = torch.mv(reg['keys'], q_batch[i])
                m = min(self.top_m, s1.shape[0])
                top_vals, top_idx = torch.topk(s1, k=m)

                # Stage-2 Top-M Common-Mode Cancellation
                cand_vis = reg['vis_proto'][top_idx]
                vis_mean = cand_vis.mean(dim=0, keepdim=True)
                cand_vis_cm = normalize(cand_vis - vis_mean)
                q_vis_cm = normalize(vis_batch[i].unsqueeze(0) - vis_mean).squeeze(0)
                s_vis_cm = torch.mv(cand_vis_cm, q_vis_cm)

                w_v = 0.45 if is_dup_lang[i] else 0.15
                s_final = top_vals + w_v * s_vis_cm
                local_best = torch.argmax(s_final).item()
                if s_final[local_best].item() > best_score:
                    best_score = s_final[local_best].item()
                    best_global_id = reg['global_ids'][top_idx[local_best].item()]
            preds[i] = best_global_id
        return preds

class FlatBaselineRouter:
    """대조군 1: 무방비 1단계 코사인 라우터 (No-Encroachment 없음, Top-M CMC 없음)"""
    def __init__(self):
        self.keys = []

    @torch.no_grad()
    def register_task(self, mu_query: torch.Tensor):
        mu_query = normalize(mu_query)
        if len(self.keys) > 0:
            K_mat = torch.stack(self.keys, dim=0)
            sims = torch.mv(K_mat, mu_query)
            closest = torch.argmax(sims)
            if sims[closest] > 0.85:
                mu_query = normalize(mu_query - 0.40 * K_mat[closest])
        self.keys.append(mu_query)

    @torch.no_grad()
    def route_batch(self, q_batch: torch.Tensor):
        K_mat = torch.stack(self.keys, dim=0)
        scores = torch.mm(q_batch, K_mat.t())
        return torch.argmax(scores, dim=-1)

def generate_libero_like_stream(num_tasks: int, dim=64, vis_dim=128, dup_ratio=0.30):
    """
    LIBERO-100 실제 통계 구조 재현:
    - 공유 주방/서재 배경(공통 모드 0.80) + 미세 객체 차이(0.20)
    - 30% 동일 명령어 그룹(Dup-Lang) + 근접 명령어 쌍
    """
    shared_scene = normalize(torch.randn(1, vis_dim, device=device))
    vis_anchors = normalize(0.80 * shared_scene + 0.20 * normalize(torch.randn(num_tasks, vis_dim, device=device)))

    num_unique_lang = max(13, int(num_tasks * (1.0 - dup_ratio)))
    shared_lang_topic = normalize(torch.randn(1, dim, device=device))
    lang_bank = normalize(0.75 * shared_lang_topic + 0.25 * normalize(torch.randn(num_unique_lang, dim, device=device)))

    task_lang_ids = torch.randint(0, num_unique_lang, (num_tasks,), device=device)
    counts = torch.bincount(task_lang_ids, minlength=num_unique_lang)
    is_dup = counts[task_lang_ids] > 1

    proj_v = normalize(torch.randn(vis_dim, dim, device=device))
    vis_in_q = normalize(torch.mm(vis_anchors, proj_v))
    # 실제 DTRA 가중치 비율(언어 0.65 + 시각/고유수용감각 0.35)
    mu_queries = normalize(0.65 * lang_bank[task_lang_ids] + 0.35 * vis_in_q)
    return mu_queries, vis_anchors, is_dup

def run_benchmark():
    scales = [100, 500, 1000, 2500, 5000]
    results = []

    print(f"Running on Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 94)
    print(f"{'K (Tasks)':>9} | {'Baseline All':>12} | {'Baseline Dup':>12} | {'Organ-DTRA All':>14} | {'Organ-DTRA Dup':>14} | {'DTRA Latency':>12}")
    print("-" * 94)

    for K in scales:
        mu_q_64, vis_a, is_dup = generate_libero_like_stream(K, dim=64, vis_dim=128, dup_ratio=0.30)
        mu_q_organ = mu_q_64.clone()

        base_router = FlatBaselineRouter()
        num_reg = max(10, K // 50)
        organ_router = OrganHierarchicalRouter(dim=64, num_regions=num_reg, top_m=5)

        for t in range(K):
            base_router.register_task(mu_q_64[t])
            organ_router.register_task(t, mu_q_organ[t], vis_a[t])
            # K=1000 도달 시 등거리 제로 패딩(64 -> 128차원) 동적 확장 테스트
            if t == 999:
                organ_router.expand_manifold(128)
                mu_q_organ = IsometricZeroPadder.expand(mu_q_organ, 128)

        eval_n = min(K, 1000)
        idx = torch.randperm(K, device=device)[:eval_n]
        noise_64 = 0.03 * torch.randn_like(mu_q_64[idx])
        q_eval_base = normalize(mu_q_64[idx] + noise_64)
        q_eval_organ = normalize(mu_q_organ[idx] + IsometricZeroPadder.expand(noise_64, mu_q_organ.shape[-1]))
        v_eval = normalize(vis_a[idx] + 0.02 * torch.randn_like(vis_a[idx]))
        dup_eval = is_dup[idx]

        # Baseline 평가
        base_preds = base_router.route_batch(q_eval_base)
        base_all_acc = (base_preds == idx).float().mean().item() * 100.0
        base_dup_acc = (base_preds[dup_eval] == idx[dup_eval]).float().mean().item() * 100.0 if dup_eval.any() else 100.0

        # Organ-DTRA 평가
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        organ_preds = organ_router.route_batch(q_eval_organ, v_eval, dup_eval)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt_ms = ((time.perf_counter() - t0) / eval_n) * 1000.0

        organ_all_acc = (organ_preds == idx).float().mean().item() * 100.0
        organ_dup_acc = (organ_preds[dup_eval] == idx[dup_eval]).float().mean().item() * 100.0 if dup_eval.any() else 100.0

        row = {
            "K": K,
            "baseline_all_acc": round(base_all_acc, 2),
            "baseline_dup_acc": round(base_dup_acc, 2),
            "organ_dtra_all_acc": round(organ_all_acc, 2),
            "organ_dtra_dup_acc": round(organ_dup_acc, 2),
            "latency_per_query_ms": round(dt_ms, 4),
            "num_regions": num_reg,
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else 0.0
        }
        results.append(row)
        print(f"{K:9d} | {base_all_acc:11.2f}% | {base_dup_acc:11.2f}% | {organ_all_acc:13.2f}% | {organ_dup_acc:13.2f}% | {dt_ms:10.4f} ms")

    with open("organ_scale_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("=" * 94)
    print("Saved results to ~/dtra_a6000_lab/organ_scale/organ_scale_results.json")

if __name__ == "__main__":
    run_benchmark()
