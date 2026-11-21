"""scripts/summarize_libero_multiseed.py

DTRA v3.0 LIBERO-100 (libero_10 + libero_90, K=100) 다중 시드(Multi-Seed) 결과 집계 및
논문용 Mean ± Std 마일스톤(K=10, 30, 50, 70, 100) 표 생성 스크립트.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def fmt_mean_std(vals: List[float], scale: float = 100.0, decimals: int = 2, signed: bool = False) -> str:
    if not vals:
        return "-"
    arr = np.array(vals, dtype=np.float64) * scale
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(vals) > 1 else 0.0
    sign_fmt = "+" if signed else ""
    if len(vals) == 1:
        return f"{mean:{sign_fmt}.{decimals}f}"
    return f"{mean:{sign_fmt}.{decimals}f} ± {std:.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Multi-Seed DTRA v3.0 K=100 Results (Sample Std ddof=1)")
    parser.add_argument(
        "--jsons",
        nargs="+",
        default=[
            "outputs/libero/benchmark_dtra_libero_10_libero_90_k100_seed42_locked.json",
            "outputs/libero/benchmark_dtra_libero_10_libero_90_k100_seed100_locked.json",
            "outputs/libero/benchmark_dtra_libero_10_libero_90_k100_seed2026_locked.json",
        ],
    )
    args = parser.parse_args()

    loaded: List[Dict] = []
    for p in args.jsons:
        path = Path(p)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Non-destructively merge finalized Seed 2026 supplemental arms (rigid / er5)
            if "seed2026" in path.name and isinstance(data.get("arms"), dict):
                data = dict(data)
                data["arms"] = dict(data["arms"])
                for arm_k, supp_fn in [
                    ("arm_dtra_rigid", "benchmark_dtra_libero_k100_seed2026_rigid.json"),
                    ("arm_er_5pct", "benchmark_dtra_libero_k100_seed2026_er5.json"),
                ]:
                    supp_p = path.parent / supp_fn
                    if arm_k not in data["arms"] and supp_p.exists():
                        with open(supp_p, "r", encoding="utf-8") as sf:
                            sd = json.load(sf)
                        av = sd.get("arms", {}).get(arm_k, {})
                        if av.get("status") == "completed" or "100" in av.get("milestones", {}):
                            data["arms"][arm_k] = av
            loaded.append(data)

    if not loaded:
        print("No JSON files found.")
        return

    arms_order = ["arm_frozen_mlp", "arm_er_5pct", "arm_dtra_rigid", "arm_dtra_elastic"]
    milestones = ["10", "30", "50", "70", "100"]

    print(f"=== DTRA LIBERO-100 K=100 Multi-Seed Summary (N={len(loaded)} seeds, Sample Std ddof=1) ===\n")
    for mk in milestones:
        print(f"### Milestone K={mk}")
        print(
            "| Arm | N | Mean SR (%) | Forgetting (%) | BWT (%) | RouteAcc (%) | Dup-Lang Route (%) | Active Bases | Params (MB) | Latency (ms) |"
        )
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for arm in arms_order:
            sr_l, fr_l, bwt_l, rt_l, dup_l, bs_l, mb_l, lat_l = [], [], [], [], [], [], [], []
            for d in loaded:
                a_obj = d.get("arms", {}).get(arm, {})
                m_obj = a_obj.get("milestones", {}).get(mk)
                if not m_obj:
                    continue
                tl = a_obj.get("train_logs", [])
                k_int = int(mk)
                true_b = tl[k_int - 1].get("active_basis_count") if isinstance(tl, list) and len(tl) >= k_int else None
                sr_l.append(m_obj["mean_sr"])
                fr_l.append(m_obj["forgetting_rate"])
                bwt_l.append(m_obj["bwt"])
                rt_l.append(m_obj["mean_route_acc"])
                dup_l.append(m_obj.get("dup_lang_route_acc", m_obj["mean_route_acc"]))
                bs_l.append(float(true_b if true_b is not None else m_obj.get("active_basis_count", k_int)))
                mb_l.append(float(m_obj.get("total_params_mb", 0.0)))
                if "active_latency_ms" in m_obj:
                    lat_l.append(float(m_obj["active_latency_ms"]))
            if not sr_l:
                continue
            rt_s = "---" if arm == "arm_er_5pct" else fmt_mean_std(rt_l)
            dup_s = "---" if arm == "arm_er_5pct" else fmt_mean_std(dup_l)
            lat_s = fmt_mean_std(lat_l, scale=1.0, decimals=2) if lat_l else "---"
            print(
                f"| `{arm}` | {len(sr_l)} | {fmt_mean_std(sr_l)} | {fmt_mean_std(fr_l)} | "
                f"{fmt_mean_std(bwt_l, signed=True)} | {rt_s} | {dup_s} | "
                f"{fmt_mean_std(bs_l, scale=1.0, decimals=1)} | {fmt_mean_std(mb_l, scale=1.0, decimals=2)} | {lat_s} |"
            )
        print()


if __name__ == "__main__":
    main()
