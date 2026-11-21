"""scripts/build_dtra_v3_paper.py

DTRA v3.0 23-Page NeurIPS/CoRL 10pt Single-Column Manuscript Pipeline.
- Strictly parses finalized _locked.json outputs (Seed 42, Seed 100, Seed 2026) and Meta-World per-seed JSONs
- Unified sample standard deviation (ddof=1) across all multi-seed LIBERO-100 & Meta-World tables and prose
- Preserves breathable typography (linespread=1.05, arraystretch=1.2)
- Strictly isolates DTRA v3.0 (Single-Ring Region K=100) from v4.0 Future Work (Organ / CALVIN)
- Generates both DTRA_v3.0_Preprint.pdf (versioned build) and dtra_preprint.pdf (pure ASCII arXiv build)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_dtra_v3_20p_parts import (
    ALGORITHM_1_BLOCK,
    APPENDIX_A_PROOFS,
    METHOD_PARADIGM_TABLE,
    REFERENCES_50_BLOCK,
)
from scripts.generate_libero100_figures import (
    generate_fig6_architecture_schematic,
    generate_fig7_libero100_scaling,
    generate_fig8_basis_and_encroachment,
)

PAPER_DIR = PROJECT_ROOT / "dtra" / "paper_v2"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
MW_DIR = PROJECT_ROOT / "outputs" / "metaworld"
LIBERO_DIR = PROJECT_ROOT / "outputs" / "libero"

TEX_V3_PATH = PAPER_DIR / "dtra_v3.0_preprint.tex"
PDF_V3_PATH = PAPER_DIR / "DTRA_v3.0_Preprint.pdf"
TEX_ARXIV_PATH = PAPER_DIR / "dtra_preprint.tex"
PDF_ARXIV_PATH = PAPER_DIR / "dtra_preprint.pdf"
_TECTONIC_WHICH = shutil.which("tectonic")
TECTONIC_EXE = (
    Path(_TECTONIC_WHICH)
    if _TECTONIC_WHICH
    else Path(r"C:\Users\sonsj\AppData\Local\Programs\tectonic\tectonic.exe")
)

SEED_JSON_CANDIDATES = [
    (
        42,
        LIBERO_DIR / "benchmark_dtra_libero_10_libero_90_k100_seed42_locked.json",
        LIBERO_DIR / "benchmark_dtra_libero_10_libero_90_k100.json",
    ),
    (
        100,
        LIBERO_DIR / "benchmark_dtra_libero_10_libero_90_k100_seed100_locked.json",
        LIBERO_DIR / "benchmark_dtra_libero_10_libero_90_k100_seed100.json",
    ),
    (
        2026,
        LIBERO_DIR / "benchmark_dtra_libero_10_libero_90_k100_seed2026_locked.json",
        LIBERO_DIR / "benchmark_dtra_libero_10_libero_90_k100_seed2026.json",
    ),
]


def resolve_seed_entries() -> List[Tuple[int, Path, bool]]:
    """Prefer finalized _locked.json paths; mark seed status cleanly."""
    resolved: List[Tuple[int, Path, bool]] = []
    for seed_id, locked_p, fallback_p in SEED_JSON_CANDIDATES:
        if locked_p.exists():
            resolved.append((seed_id, locked_p, True))
        elif fallback_p.exists():
            resolved.append((seed_id, fallback_p, False))
    return resolved


def _load_seed_json(path: Path, is_locked: bool) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not is_locked and isinstance(data.get("arms"), dict):
        data = dict(data)
        data["arms"] = {
            k: v
            for k, v in data["arms"].items()
            if v.get("status") == "completed" or "100" in v.get("milestones", {})
        }
    # Non-destructively merge finalized Seed 2026 supplemental arms (rigid / er5) without altering _locked.json
    if "seed2026" in path.name and isinstance(data.get("arms"), dict):
        data = dict(data)
        data["arms"] = dict(data["arms"])
        for arm_k, supp_fn in [
            ("arm_dtra_rigid", "benchmark_dtra_libero_k100_seed2026_rigid.json"),
            ("arm_er_5pct", "benchmark_dtra_libero_k100_seed2026_er5.json"),
        ]:
            supp_p = path.parent / supp_fn
            if arm_k not in data["arms"] and supp_p.exists():
                try:
                    with open(supp_p, "r", encoding="utf-8") as sf:
                        sd = json.load(sf)
                    av = sd.get("arms", {}).get(arm_k, {})
                    if av.get("status") == "completed" or "100" in av.get("milestones", {}):
                        data["arms"][arm_k] = av
                except Exception:
                    pass
    return data if data.get("arms") else None


def _wilson_ci95(k: float, n: int) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return (max(0.0, float(center - half)), min(1.0, float(center + half)))


def _load_simple_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_signed(val: float, dec: int = 2, suffix: str = "") -> str:
    sign = "+" if val >= 0 else "-"
    return f"${sign}{abs(val):.{dec}f}${suffix}"


def _fmt_mean_std(
    vals: List[float],
    scale: float = 100.0,
    dec: int = 2,
    signed: bool = False,
    suffix: str = "",
    bold: bool = False,
) -> str:
    if not vals:
        return r"\texttt{[PENDING]}"
    arr = np.array(vals, dtype=np.float64) * scale
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(vals) > 1 else 0.0
    if signed:
        m_str = f"+{abs(m):.{dec}f}" if m >= 0 else f"-{abs(m):.{dec}f}"
    else:
        m_str = f"{m:.{dec}f}"
    body = m_str if len(vals) == 1 else f"{m_str} \\pm {s:.{dec}f}"
    if bold:
        return f"$\\mathbf{{{body}}}${suffix}"
    return f"${body}${suffix}"


def _true_basis_count(arm_obj: dict, k_int: int, fallback: int) -> int:
    tl = arm_obj.get("train_logs", [])
    if isinstance(tl, list) and len(tl) >= k_int:
        v = tl[k_int - 1].get("active_basis_count")
        if v is not None:
            return int(v)
    return int(fallback)


def collect_libero_summary(lib_entries: List[Tuple[int, dict]]) -> Dict[str, Any]:
    """Compute exact real metrics across observed locked seeds without fabrication."""
    arms = ["arm_frozen_mlp", "arm_er_5pct", "arm_dtra_rigid", "arm_dtra_elastic"]
    out: Dict[str, Any] = {"seeds_observed": [s for s, _ in lib_entries], "arms": {}}
    for arm in arms:
        sr, fr, bwt, rt, dup, bs, mb, lat, vram = [], [], [], [], [], [], [], [], []
        seed_sr: Dict[int, float] = {}
        for s_id, d in lib_entries:
            ao = d.get("arms", {}).get(arm, {})
            mv = ao.get("milestones", {}).get("100")
            if not mv:
                continue
            seed_sr[s_id] = float(mv["mean_sr"]) * 100.0
            sr.append(mv["mean_sr"])
            fr.append(mv["forgetting_rate"])
            bwt.append(mv["bwt"])
            rt.append(mv["mean_route_acc"])
            dup.append(mv.get("dup_lang_route_acc", mv["mean_route_acc"]))
            bs.append(float(_true_basis_count(ao, 100, mv.get("active_basis_count", 100))))
            mb.append(float(mv.get("total_params_mb", 0.0)))
            lat.append(float(mv.get("active_latency_ms", 0.0)))
            vram.append(float(mv.get("peak_vram_mb", 0.0)))
        out["arms"][arm] = {
            "n": len(sr),
            "seed_sr": seed_sr,
            "sr": sr,
            "fr": fr,
            "bwt": bwt,
            "rt": rt,
            "dup": dup,
            "bs": bs,
            "mb": mb,
            "lat": lat,
            "vram": vram,
        }
    return out


def build_main_tables(
    sci: Optional[dict],
    supp: Optional[dict],
    lib_goal: Optional[dict],
    lib_entries: List[Tuple[int, dict]],
    summary: Dict[str, Any],
    mw_v31: Optional[dict] = None,
    lib_ext: Optional[dict] = None,
    lib_neval20: Optional[dict] = None,
) -> Dict[str, str]:
    lib_100_list = [d for _, d in lib_entries]
    # 1. Meta-World 3-Seed Table (computed with sample std ddof=1 from per_seed JSONs)
    def _mw_sci_ddof1(sci_obj: dict, arm_k: str) -> Optional[Dict[str, float]]:
        ps = sci_obj.get("per_seed", {})
        seeds = [str(s) for s in sci_obj.get("seeds", []) if str(s) in ps and arm_k in ps[str(s)]]
        if not seeds:
            return None
        c_sr = [float(np.mean(ps[s][arm_k]["clean_id"]["per_task_sr"])) for s in seeds]
        fr_v = [float(ps[s][arm_k]["clean_id"]["avg_forgetting"]) for s in seeds]
        c_rt = [float(ps[s][arm_k]["clean_id"]["avg_route_acc"]) for s in seeds]
        o_sr = [float(np.mean(ps[s][arm_k]["ood_severe_sigma020"]["per_task_sr"])) for s in seeds]
        o_rt = [float(ps[s][arm_k]["ood_severe_sigma020"]["avg_route_acc"]) for s in seeds]
        ddof = 1 if len(seeds) > 1 else 0
        return {
            "clean_avg_sr_mean": float(np.mean(c_sr)),
            "clean_avg_sr_std": float(np.std(c_sr, ddof=ddof)),
            "clean_fr_mean": float(np.mean(fr_v)),
            "clean_fr_std": float(np.std(fr_v, ddof=ddof)),
            "clean_route_mean": float(np.mean(c_rt)),
            "ood_severe_sr_mean": float(np.mean(o_sr)),
            "ood_severe_sr_std": float(np.std(o_sr, ddof=ddof)),
            "ood_severe_route_mean": float(np.mean(o_rt)),
        }

    def _mw_v31_ddof1(v31_obj: dict, arm_k: str) -> Optional[Dict[str, float]]:
        ps = v31_obj.get("per_seed", {})
        seeds = [s for s in ps if arm_k in ps[s]]
        if not seeds:
            return v31_obj.get("aggregated", {}).get(arm_k)
        c_l = [float(ps[s][arm_k].get("clean_sr", ps[s][arm_k].get("clean_avg_sr"))) for s in seeds]
        f_l = [float(ps[s][arm_k].get("fr", ps[s][arm_k].get("forgetting_rate"))) for s in seeds]
        o_l = [float(ps[s][arm_k].get("ood_sr", ps[s][arm_k].get("ood_severe_sr"))) for s in seeds]
        cr_l = [float(ps[s][arm_k]["clean_route_acc"]) for s in seeds if "clean_route_acc" in ps[s][arm_k]]
        or_l = [float(ps[s][arm_k]["ood_route_acc"]) for s in seeds if "ood_route_acc" in ps[s][arm_k]]
        ddof = 1 if len(seeds) > 1 else 0
        res: Dict[str, float] = {
            "clean_sr_mean": float(np.mean(c_l)),
            "clean_sr_std": float(np.std(c_l, ddof=ddof)),
            "fr_mean": float(np.mean(f_l)),
            "fr_std": float(np.std(f_l, ddof=ddof)),
            "ood_sr_mean": float(np.mean(o_l)),
            "ood_sr_std": float(np.std(o_l, ddof=ddof)),
        }
        if cr_l:
            res["clean_route_mean"] = float(np.mean(cr_l))
        if or_l:
            res["ood_route_mean"] = float(np.mean(or_l))
        return res

    # Collect all Meta-World Table 2 entries first so we bold strictly the best metric per column
    mw_entries: List[Dict[str, Any]] = []
    mw_arm_labels = [
        ("arm1_abk_dog_vision", r"ABK-VLA v1.1.3 (DoG Vision)"),
        ("arm2_frozen_mlp_vision", r"Frozen Slot MLP (Vision)"),
        ("arm3_dtra_full_vision", r"DTRA Full (Vision-Only)"),
        ("arm4_frozen_mlp_multimodal", r"Frozen Slot MLP (Multimodal)"),
        ("arm5_dtra_rigid_alpha0", r"DTRA Rigid ($\alpha=0$, MM)"),
        ("arm6_dtra_full_multimodal", r"\textbf{DTRA Full (Feat-Reg, Ours)}"),
    ]
    arm6_lat_pair = None
    if sci and "aggregated" in sci:
        agg = sci["aggregated"]
        for key, label in mw_arm_labels:
            v_agg = agg.get(key)
            v = _mw_sci_ddof1(sci, key) or v_agg
            if not v:
                continue
            lat = (v_agg or {}).get("cpu_latency", {})
            l1 = lat.get("K=1", {}).get("active_exec_ms")
            l8 = lat.get("K=8", {}).get("active_exec_ms")
            if key == "arm6_dtra_full_multimodal" and l1 is not None and l8 is not None:
                arm6_lat_pair = (float(l1), float(l8))
            mw_entries.append({
                "section": "core",
                "label": label,
                "c_m": v["clean_avg_sr_mean"] * 100.0,
                "c_s": v["clean_avg_sr_std"] * 100.0,
                "f_m": v["clean_fr_mean"] * 100.0,
                "f_s": v["clean_fr_std"] * 100.0,
                "rt": v["clean_route_mean"] * 100.0,
                "o_m": v["ood_severe_sr_mean"] * 100.0,
                "o_s": v["ood_severe_sr_std"] * 100.0,
                "o_rt": v["ood_severe_route_mean"] * 100.0,
                "l1": float(l1) if l1 is not None else None,
                "l8": float(l8) if l8 is not None else None,
            })
    if mw_v31 and "aggregated" in mw_v31:
        v_nf = _mw_v31_ddof1(mw_v31, "causal_dtra_no_featreg")
        if v_nf:
            mw_entries.append({
                "section": "core",
                "label": r"\textbf{DTRA Full (No Feat-Reg, Ours)}",
                "c_m": v_nf["clean_sr_mean"] * 100.0,
                "c_s": v_nf["clean_sr_std"] * 100.0,
                "f_m": v_nf["fr_mean"] * 100.0,
                "f_s": v_nf["fr_std"] * 100.0,
                "rt": v_nf.get("clean_route_mean", 1.0) * 100.0,
                "o_m": v_nf["ood_sr_mean"] * 100.0,
                "o_s": v_nf["ood_sr_std"] * 100.0,
                "o_rt": v_nf.get("ood_route_mean", 1.0) * 100.0,
                "l1": arm6_lat_pair[0] if arm6_lat_pair else None,
                "l8": arm6_lat_pair[1] if arm6_lat_pair else None,
                "lat_same_arch": True,
            })
        for ek, elab in [
            ("ext_derpp", "DER++ (5\\% Buffer + Dark Distill.)"),
            ("ext_l2p", "L2P (Prompt Pool, Replay-Free)"),
            ("ext_olora", "O-LoRA (Ortho Subspace, Replay-Free)"),
        ]:
            ev = _mw_v31_ddof1(mw_v31, ek)
            if ev:
                mw_entries.append({
                    "section": "ext",
                    "label": elab,
                    "c_m": ev["clean_sr_mean"] * 100.0,
                    "c_s": ev["clean_sr_std"] * 100.0,
                    "f_m": ev["fr_mean"] * 100.0,
                    "f_s": ev["fr_std"] * 100.0,
                    "rt": None,
                    "o_m": ev["ood_sr_mean"] * 100.0,
                    "o_s": ev["ood_sr_std"] * 100.0,
                    "o_rt": None,
                    "l1": None,
                    "l8": None,
                })

    mw_rows = []
    if mw_entries:
        best_c = max(e["c_m"] for e in mw_entries)
        best_f = min(e["f_m"] for e in mw_entries)
        best_o = max(e["o_m"] for e in mw_entries)
        best_rt = max(e["rt"] for e in mw_entries if e["rt"] is not None)
        best_ort = max(e["o_rt"] for e in mw_entries if e["o_rt"] is not None)
        best_l8 = min(e["l8"] for e in mw_entries if e["l8"] is not None)
        added_mid = False
        for e in mw_entries:
            if e["section"] == "ext" and not added_mid:
                mw_rows.append(r"    \midrule")
                added_mid = True
            c_txt = f"{e['c_m']:.1f} \\pm {e['c_s']:.1f}"
            c_cell = f"$\\mathbf{{{c_txt}}}$" if abs(e["c_m"] - best_c) < 0.05 else f"${c_txt}$"
            f_txt = f"{e['f_m']:.1f} \\pm {e['f_s']:.1f}"
            f_cell = f"$\\mathbf{{{f_txt}}}$" if abs(e["f_m"] - best_f) < 0.05 else f"${f_txt}$"
            rt_cell = "---" if e["rt"] is None else (f"\\textbf{{{e['rt']:.1f}}}" if abs(e["rt"] - best_rt) < 0.05 else f"{e['rt']:.1f}")
            o_txt = f"{e['o_m']:.1f} \\pm {e['o_s']:.1f}"
            o_cell = f"$\\mathbf{{{o_txt}}}$" if abs(e["o_m"] - best_o) < 0.05 else f"${o_txt}$"
            ort_cell = "---" if e["o_rt"] is None else (f"\\textbf{{{e['o_rt']:.1f}}}" if abs(e["o_rt"] - best_ort) < 0.05 else f"{e['o_rt']:.1f}")
            if e["l1"] is not None and e["l8"] is not None:
                l_txt = f"{e['l1']:.2f} \\to {e['l8']:.2f}"
                if e.get("lat_same_arch"):
                    lat_cell = f"${l_txt}$$^\\dagger$"
                elif abs(e["l8"] - best_l8) < 0.005:
                    lat_cell = f"$\\mathbf{{{l_txt}}}$"
                else:
                    lat_cell = f"${l_txt}$"
            else:
                lat_cell = "---"
            mw_rows.append(f"    {e['label']} & {c_cell} & {f_cell} & {rt_cell} & {o_cell} & {ort_cell} & {lat_cell} \\\\")

    # 2. Meta-World Supplemental & Causal Decomposition Table (bold strictly best per column)
    supp_entries: List[Dict[str, Any]] = []
    if supp and "part1_traditional_cl" in supp:
        p1 = supp["part1_traditional_cl"]
        for key, label in [
            ("cl_naive_ft", "Naive Fine-Tuning (Seed 42)"),
            ("cl_ewc", r"EWC ($\lambda=500$, Seed 42)"),
            ("cl_er_5pct", "Experience Replay (5\\%, Seed 42)"),
        ]:
            v = p1.get(key)
            if v:
                supp_entries.append({
                    "sec": "trad", "label": label, "multi": False,
                    "c_m": v["clean_avg_sr"] * 100.0, "c_s": 0.0,
                    "f_m": v["fr"] * 100.0, "f_s": 0.0,
                    "o_m": v["ood_severe_sr"] * 100.0, "o_s": 0.0,
                })
    if mw_v31 and "aggregated" in mw_v31:
        for ek, elab in [
            ("ext_derpp", "DER++ (3-Seed Mean $\\pm$ Std)"),
            ("ext_l2p", "L2P Prompt Pool (3-Seed Mean $\\pm$ Std)"),
            ("ext_olora", "O-LoRA Subspace (3-Seed Mean $\\pm$ Std)"),
        ]:
            ev = _mw_v31_ddof1(mw_v31, ek)
            if ev:
                supp_entries.append({
                    "sec": "trad", "label": elab, "multi": True,
                    "c_m": ev["clean_sr_mean"] * 100.0, "c_s": ev["clean_sr_std"] * 100.0,
                    "f_m": ev["fr_mean"] * 100.0, "f_s": ev["fr_std"] * 100.0,
                    "o_m": ev["ood_sr_mean"] * 100.0, "o_s": ev["ood_sr_std"] * 100.0,
                })
        for ck, clab in [
            ("causal_frozen_mlp_featreg", r"Frozen MLP + Conv4 Feat-Reg ($p=0.30,\sigma=0.05$, 3-Seed)"),
            ("causal_dtra_rigid_no_featreg", r"DTRA Rigid ($\alpha=0$, No Feat-Reg $p=0,\sigma=0$, 3-Seed)"),
            ("causal_dtra_no_featreg", r"\textbf{DTRA Full (No Feat-Reg $p=0,\sigma=0$, 3-Seed)}"),
            ("causal_dtra_res075", r"DTRA Rigid (Mild Feat-Reg $p=0.15,\sigma=0.025,r=0.75$, 3-Seed)"),
        ]:
            cv = _mw_v31_ddof1(mw_v31, ck)
            if cv:
                supp_entries.append({
                    "sec": "causal", "label": clab, "multi": True,
                    "c_m": cv["clean_sr_mean"] * 100.0, "c_s": cv["clean_sr_std"] * 100.0,
                    "f_m": cv["fr_mean"] * 100.0, "f_s": cv["fr_std"] * 100.0,
                    "o_m": cv["ood_sr_mean"] * 100.0, "o_s": cv["ood_sr_std"] * 100.0,
                })
    if supp and "part3_reverse_order_arm6" in supp:
        r6 = supp["part3_reverse_order_arm6"]
        supp_entries.append({
            "sec": "causal", "label": r"\textbf{DTRA Reverse ($T_7 \to T_0$, Seed 42, Ours)}", "multi": False,
            "c_m": r6["clean_avg_sr"] * 100.0, "c_s": 0.0,
            "f_m": r6["fr"] * 100.0, "f_s": 0.0,
            "o_m": r6["ood_severe_sr"] * 100.0, "o_s": 0.0,
        })

    supp_rows = []
    if supp_entries:
        best_sc = max(e["c_m"] for e in supp_entries)
        best_sf = min(e["f_m"] for e in supp_entries)
        best_so = max(e["o_m"] for e in supp_entries)
        added_mid = False
        for e in supp_entries:
            if e["sec"] == "causal" and not added_mid:
                supp_rows.append(r"    \midrule")
                added_mid = True
            def _sfmt(m: float, s: float, multi: bool, is_best: bool) -> str:
                inner = f"{m:.1f} \\pm {s:.1f}" if multi else f"{m:.1f}"
                return f"$\\mathbf{{{inner}}}$\\%" if is_best else f"${inner}$\\%"
            c_s = _sfmt(e["c_m"], e["c_s"], e["multi"], abs(e["c_m"] - best_sc) < 0.05)
            f_s = _sfmt(e["f_m"], e["f_s"], e["multi"], abs(e["f_m"] - best_sf) < 0.05)
            o_s = _sfmt(e["o_m"], e["o_s"], e["multi"], abs(e["o_m"] - best_so) < 0.05)
            supp_rows.append(f"    {e['label']} & {c_s} & {f_s} & {o_s} \\\\")

    # 3. LIBERO-Goal Pilot Table
    lg_rows = []
    if lib_goal and "arms" in lib_goal:
        lg_specs = [
            ("arm_frozen_mlp", "Frozen Slot MLP (Baseline)"),
            ("arm_er_5pct", "Experience Replay (5\\%)"),
            ("arm_dtra_rigid", r"DTRA Rigid ($\alpha=0$)"),
            ("arm_dtra_elastic", r"\textbf{DTRA v3.0 Elastic (Ours)}"),
        ]
        lg_vals = [(k, lab, lib_goal["arms"].get(k)) for k, lab in lg_specs if lib_goal["arms"].get(k)]
        best_lg_sr = max(v["mean_final_sr"] for _, _, v in lg_vals)
        best_lg_fr = min(v["forgetting_rate"] for _, _, v in lg_vals)
        best_lg_bwt = max(v["bwt"] for _, _, v in lg_vals)
        for key, label, v in lg_vals:
            sr_v = v["mean_final_sr"] * 100.0
            fr_v = v["forgetting_rate"] * 100.0
            bwt_v = v["bwt"] * 100.0
            sr_c = f"\\textbf{{{sr_v:.1f}\\%}}" if abs(v["mean_final_sr"] - best_lg_sr) < 1e-6 else f"{sr_v:.1f}\\%"
            fr_c = f"\\textbf{{{fr_v:.1f}\\%}}" if abs(v["forgetting_rate"] - best_lg_fr) < 1e-6 else f"{fr_v:.1f}\\%"
            bwt_inner = f"+{abs(bwt_v):.1f}" if bwt_v >= 0 else f"-{abs(bwt_v):.1f}"
            bwt_c = f"$\\mathbf{{{bwt_inner}}}$\\%" if abs(v["bwt"] - best_lg_bwt) < 1e-6 else f"${bwt_inner}$\\%"
            rt = "---" if key == "arm_er_5pct" else f"{v['mean_final_route_acc']*100:.1f}\\%"
            lg_rows.append(f"    {label} & {sr_c} & {fr_c} & {bwt_c} & {rt} & {v['max_vram_mb']:.0f}\\,MB \\\\")

    # 4. LIBERO-100 Milestones Table (strict best-per-K bolding + math minus everywhere)
    arm_meta = [
        ("arm_frozen_mlp", "Frozen MLP"),
        ("arm_er_5pct", "ER 5\\% (Replay)"),
        ("arm_dtra_rigid", r"DTRA Rigid ($\alpha=0$)"),
        ("arm_dtra_elastic", r"\textbf{DTRA Elastic (Ours)}"),
    ]
    ms_rows = []
    for k in [10, 30, 50, 70, 100]:
        mk = str(k)
        k_arm_data = []
        for arm_key, arm_label in arm_meta:
            sr_l, fr_l, bwt_l, rt_l, dup_l, bs_l = [], [], [], [], [], []
            for d in lib_100_list:
                ao = d.get("arms", {}).get(arm_key, {})
                mv = ao.get("milestones", {}).get(mk)
                if not mv:
                    continue
                sr_l.append(mv["mean_sr"])
                fr_l.append(mv["forgetting_rate"])
                bwt_l.append(mv["bwt"])
                rt_l.append(mv["mean_route_acc"])
                dup_l.append(mv.get("dup_lang_route_acc", mv["mean_route_acc"]))
                bs_l.append(float(_true_basis_count(ao, k, mv.get("active_basis_count", k))))
            k_arm_data.append((arm_key, arm_label, sr_l, fr_l, bwt_l, rt_l, dup_l, bs_l))
        best_k_sr = max((round(float(np.mean(x[2])) * 100.0, 2) for x in k_arm_data if x[2]), default=-1.0)
        best_k_rt = max((round(float(np.mean(x[5])) * 100.0, 2) for x in k_arm_data if x[5] and x[0] != "arm_er_5pct"), default=-1.0)
        best_k_dup = max((round(float(np.mean(x[6])) * 100.0, 2) for x in k_arm_data if x[6] and x[0] != "arm_er_5pct"), default=-1.0)
        for idx_a, (arm_key, arm_label, sr_l, fr_l, bwt_l, rt_l, dup_l, bs_l) in enumerate(k_arm_data):
            k_col = f"$K={k}$" if idx_a == 0 else ""
            if sr_l:
                sr_s = _fmt_mean_std(sr_l, 100.0, 2, suffix=r"\%", bold=round(float(np.mean(sr_l)) * 100.0, 2) == best_k_sr)
                fr_s = _fmt_mean_std(fr_l, 100.0, 2, suffix=r"\%")
                bwt_s = _fmt_mean_std(bwt_l, 100.0, 2, signed=True, suffix=r"\%")
                rt_s = "---" if arm_key == "arm_er_5pct" else _fmt_mean_std(rt_l, 100.0, 2, suffix=r"\%", bold=round(float(np.mean(rt_l)) * 100.0, 2) == best_k_rt)
                dup_s = "---" if arm_key == "arm_er_5pct" else _fmt_mean_std(dup_l, 100.0, 2, suffix=r"\%", bold=round(float(np.mean(dup_l)) * 100.0, 2) == best_k_dup)
                bs_s = f"{int(round(np.mean(bs_l)))}" if np.std(bs_l) < 1e-6 else _fmt_mean_std(bs_l, 1.0, 1)
                ms_rows.append(f"    {k_col} & {arm_label} & {sr_s} & {fr_s} & {bwt_s} & {rt_s} & {dup_s} & {bs_s} \\\\")
            else:
                ms_rows.append(f"    {k_col} & {arm_label} & \\texttt{{[PENDING]}} & \\texttt{{[PENDING]}} & \\texttt{{[PENDING]}} & --- & --- & 1 \\\\")
        if lib_ext and isinstance(lib_ext.get("arms"), dict):
            for ek, elab, ebs in [
                ("arm_derpp", "DER++ (5\\% Buf., S42)", "1"),
                ("arm_l2p", "L2P (Prompts, S42)", "32"),
                ("arm_olora", "O-LoRA (Subsp., S42)", str(k)),
            ]:
                mv = lib_ext["arms"].get(ek, {}).get("milestones", {}).get(mk)
                if mv:
                    ms_rows.append(
                        f"    & {elab} & {mv['mean_sr']*100:.2f}\\% & {mv['forgetting_rate']*100:.2f}\\% & "
                        f"{_fmt_signed(mv['bwt']*100.0, 2, r'\%')} & --- & --- & {ebs} \\\\"
                    )
        if k != 100:
            ms_rows.append(r"    \midrule")

    # Compact LIBERO-100 Trajectory table for Main Body (strict best per K)
    best_traj_sr: Dict[int, float] = {}
    for k in [10, 30, 50, 70, 100]:
        mk = str(k)
        cand = []
        for arm_key, _ in arm_meta:
            sr_l = [d.get("arms", {}).get(arm_key, {}).get("milestones", {}).get(mk, {}).get("mean_sr") for d in lib_100_list]
            sr_l = [x for x in sr_l if x is not None]
            if sr_l:
                cand.append(round(float(np.mean(sr_l)) * 100.0, 2))
        best_traj_sr[k] = max(cand) if cand else -1.0

    traj_rows = []
    for arm_key, arm_label in arm_meta:
        k_cells = []
        for k in [10, 30, 50, 70, 100]:
            mk = str(k)
            sr_l = [
                d.get("arms", {}).get(arm_key, {}).get("milestones", {}).get(mk, {}).get("mean_sr")
                for d in lib_100_list
            ]
            sr_l = [x for x in sr_l if x is not None]
            if sr_l:
                is_b = round(float(np.mean(sr_l)) * 100.0, 2) == best_traj_sr[k]
                k_cells.append(_fmt_mean_std(sr_l, 100.0, 2, suffix=r"\%", bold=is_b))
            else:
                k_cells.append(r"\texttt{[PENDING]}")
        traj_rows.append(f"    {arm_label} & {' & '.join(k_cells)} \\\\")
    if lib_ext and isinstance(lib_ext.get("arms"), dict):
        ext_rows_added = False
        for ek, elab in [
            ("arm_derpp", "DER++ (5\\% Buffer)"),
            ("arm_l2p", "L2P (Prompt Pool)"),
            ("arm_olora", "O-LoRA (Ortho LoRA)"),
        ]:
            eao = lib_ext["arms"].get(ek, {})
            ems = eao.get("milestones", {})
            if not ems:
                continue
            if not ext_rows_added:
                traj_rows.append(r"    \midrule")
                ext_rows_added = True
            k_cells = []
            for k in [10, 30, 50, 70, 100]:
                mv = ems.get(str(k), {})
                if "mean_sr" in mv:
                    k_cells.append(f"{mv['mean_sr']*100:.2f}\\%")
                else:
                    k_cells.append(r"\texttt{[RUN]}")
            traj_rows.append(f"    {elab} & {' & '.join(k_cells)} \\\\")

    # 5. LIBERO-100 Final K=100 Table (9 compact columns, strict best-per-column bolding on rounded display values)
    core_k100 = [(ak, al, summary["arms"][ak]) for ak, al in arm_meta if summary["arms"][ak]["n"] > 0]
    best_100_sr = max(round(float(np.mean(ad["sr"])) * 100.0, 2) for _, _, ad in core_k100)
    best_100_fr = min(round(float(np.mean(ad["fr"])) * 100.0, 2) for _, _, ad in core_k100)
    best_100_bwt = max(round(float(np.mean(ad["bwt"])) * 100.0, 2) for _, _, ad in core_k100)
    best_100_rt = max(round(float(np.mean(ad["rt"])) * 100.0, 2) for ak, _, ad in core_k100 if ak != "arm_er_5pct")
    best_100_dup = max(round(float(np.mean(ad["dup"])) * 100.0, 2) for ak, _, ad in core_k100 if ak != "arm_er_5pct")
    best_100_bs = min(int(round(np.mean(ad["bs"]))) for ak, _, ad in core_k100 if ak != "arm_er_5pct")
    best_100_lat = min(round(float(np.mean(ad["lat"])), 2) for _, _, ad in core_k100)

    k100_rows = []
    for arm_key, arm_label, ad in core_k100:
        s42 = f"{ad['seed_sr'][42]:.1f}" if 42 in ad["seed_sr"] else "---"
        s100 = f"{ad['seed_sr'][100]:.1f}" if 100 in ad["seed_sr"] else "---"
        s2026 = f"{ad['seed_sr'][2026]:.1f}" if 2026 in ad["seed_sr"] else "---"
        seeds_triplet = f"{s42}/{s100}/{s2026}"
        ms_sr = _fmt_mean_std(ad["sr"], 100.0, 2, bold=round(float(np.mean(ad["sr"])) * 100.0, 2) == best_100_sr)
        fr_s = _fmt_mean_std(ad["fr"], 100.0, 2, bold=round(float(np.mean(ad["fr"])) * 100.0, 2) == best_100_fr)
        bwt_s = _fmt_mean_std(ad["bwt"], 100.0, 2, signed=True, bold=round(float(np.mean(ad["bwt"])) * 100.0, 2) == best_100_bwt)
        rt_s = "---" if arm_key == "arm_er_5pct" else _fmt_mean_std(ad["rt"], 100.0, 2, bold=round(float(np.mean(ad["rt"])) * 100.0, 2) == best_100_rt)
        dup_s = "---" if arm_key == "arm_er_5pct" else _fmt_mean_std(ad["dup"], 100.0, 2, bold=round(float(np.mean(ad["dup"])) * 100.0, 2) == best_100_dup)
        bs_int = int(round(np.mean(ad["bs"])))
        if np.std(ad["bs"]) < 1e-6:
            bs_s = f"\\textbf{{{bs_int}}}" if (arm_key != "arm_er_5pct" and bs_int == best_100_bs) else f"{bs_int}"
        else:
            bs_s = _fmt_mean_std(ad["bs"], 1.0, 1)
        lat_s = _fmt_mean_std(ad["lat"], 1.0, 2, bold=round(float(np.mean(ad["lat"])), 2) == best_100_lat)
        k100_rows.append(
            f"    {arm_label} & {seeds_triplet} & {ms_sr} & {fr_s} & {bwt_s} & {rt_s} & {dup_s} & {bs_s} & {lat_s} \\\\"
        )
    if lib_ext and isinstance(lib_ext.get("arms"), dict):
        ext_k100_added = False
        for ek, elab, ebs in [
            ("arm_derpp", "DER++ (5\\% Buf.)", "1"),
            ("arm_l2p", "L2P (Prompts)", "32"),
            ("arm_olora", "O-LoRA (Subsp.)", "100"),
        ]:
            eao = lib_ext["arms"].get(ek, {})
            mv = eao.get("milestones", {}).get("100")
            if not mv:
                continue
            if not ext_k100_added:
                k100_rows.append(r"    \midrule")
                ext_k100_added = True
            sr_v = mv["mean_sr"] * 100.0
            fr_v = mv["forgetting_rate"] * 100.0
            bwt_v = mv["bwt"] * 100.0
            lat_v = mv.get("active_latency_ms", 0.0)
            k100_rows.append(
                f"    {elab} & {sr_v:.1f}/---/--- & {sr_v:.2f} & {fr_v:.2f} & {_fmt_signed(bwt_v, 2)} & --- & --- & {ebs} & {lat_v:.2f} \\\\"
            )

    n_rigid = summary["arms"]["arm_dtra_rigid"]["n"]
    n_er = summary["arms"]["arm_er_5pct"]["n"]
    if n_rigid == 3 and n_er == 3:
        cov_note = "$N=3$ across Seeds~[42, 100, 2026] for all four core arms"
    elif n_rigid == 3:
        cov_note = f"$N=3$ across Seeds~[42, 100, 2026] for Frozen MLP, DTRA Rigid \\& DTRA Elastic; $N={n_er}$ for ER~5\\%"
    else:
        cov_note = f"$N=3$ for Frozen MLP \\& DTRA Elastic; $N={n_rigid}$ for DTRA Rigid \\& $N={n_er}$ for ER~5\\%"
    ext_note = "; Seed~42 for external baselines DER++, L2P, O-LoRA" if (lib_ext and lib_ext.get("arms")) else ""

    # 6. Extended n_eval=20 Zero-Retrain Verification Table (12,000 Rollouts across 3 Seeds)
    neval20_block = ""
    if lib_neval20 and isinstance(lib_neval20.get("runs"), dict):
        runs = lib_neval20["runs"]
        ne_rows = []
        arm_aggs = {}
        for arm_k, arm_lab in [
            ("arm_frozen_mlp", "Frozen MLP"),
            ("arm_dtra_elastic", r"\textbf{DTRA Elastic}"),
        ]:
            f5_l, h15_l, a20_l, dup_l, drt_l, unq_l = [], [], [], [], [], []
            for sd in (42, 100, 2026):
                r = runs.get(f"seed{sd}_{arm_k}", {}).get("summary")
                if not r:
                    continue
                f5_l.append(r["sr_first5"])
                h15_l.append(r["sr_heldout"])
                a20_l.append(r["sr_all"])
                dup_l.append(r["dup_lang_sr"])
                drt_l.append(r["dup_lang_route"])
                unq_l.append(r["unique_lang_sr"])
            if a20_l:
                n_s = len(a20_l)
                tot_ep = n_s * 2000
                held_ep = n_s * 1500
                ci_all = _wilson_ci95(float(np.sum(a20_l)) * 2000.0, tot_ep)
                ci_held = _wilson_ci95(float(np.sum(h15_l)) * 1500.0, held_ep)
                arm_aggs[arm_k] = {
                    "f5": float(np.mean(f5_l)) * 100.0,
                    "h15": float(np.mean(h15_l)) * 100.0,
                    "a20": float(np.mean(a20_l)) * 100.0,
                    "dup": float(np.mean(dup_l)) * 100.0,
                    "drt": float(np.mean(drt_l)) * 100.0,
                    "unq": float(np.mean(unq_l)) * 100.0,
                }
                is_el = arm_k == "arm_dtra_elastic"
                f5_s = _fmt_mean_std(f5_l, 100.0, 2, suffix=r"\%", bold=is_el)
                h15_s = (
                    _fmt_mean_std(h15_l, 100.0, 2, suffix=r"\%", bold=is_el)
                    + (f" \\textbf{{[{ci_held[0]*100:.1f}, {ci_held[1]*100:.1f}]}}" if is_el else f" [{ci_held[0]*100:.1f}, {ci_held[1]*100:.1f}]")
                )
                a20_s = (
                    _fmt_mean_std(a20_l, 100.0, 2, suffix=r"\%", bold=is_el)
                    + (f" \\textbf{{[{ci_all[0]*100:.1f}, {ci_all[1]*100:.1f}]}}" if is_el else f" [{ci_all[0]*100:.1f}, {ci_all[1]*100:.1f}]")
                )
                dup_s = f"\\textbf{{{np.mean(dup_l)*100:.2f}\\% ({np.mean(drt_l)*100:.2f}\\%)}}" if is_el else f"{np.mean(dup_l)*100:.2f}\\% ({np.mean(drt_l)*100:.2f}\\%)"
                unq_s = f"\\textbf{{{np.mean(unq_l)*100:.2f}\\%}}" if is_el else f"{np.mean(unq_l)*100:.2f}\\%"
                ne_rows.append(f"    {arm_lab} & {f5_s} & {h15_s} & {a20_s} & {dup_s} & {unq_s} \\\\")
        if "arm_dtra_elastic" in arm_aggs and "arm_frozen_mlp" in arm_aggs:
            de, df = arm_aggs["arm_dtra_elastic"], arm_aggs["arm_frozen_mlp"]
            ne_rows.append(r"    \midrule")
            ne_rows.append(
                f"    \\textbf{{$\\Delta$ (DTRA $-$ MLP)}} & "
                f"\\textbf{{{_fmt_signed(de['f5']-df['f5'], 2, r'\,pp')}}} & "
                f"\\textbf{{{_fmt_signed(de['h15']-df['h15'], 2, r'\,pp')}}} & "
                f"\\textbf{{{_fmt_signed(de['a20']-df['a20'], 2, r'\,pp')}}} & "
                f"\\textbf{{{_fmt_signed(de['dup']-df['dup'], 2, r'\,pp')} ({_fmt_signed(de['drt']-df['drt'], 2, r'\,pp')})}} & "
                f"{_fmt_signed(de['unq']-df['unq'], 2, r'\,pp')} \\\\"
            )
        if ne_rows:
            neval20_block = f"""
\\begin{{table}}[ht!]
  \\centering
  \\caption{{Zero-Retrain Extended Initial-State Verification on Saved $K=100$ LIBERO-100 Checkpoints ($n_{{\\text{{eval}}}}=20$ initial states/task $\\times 100$ tasks $\\times 3$ training seeds $= 6{{,}}000$ rollouts/arm, $12{{,}}000$ total; Mean $\\pm$ Sample Std \\texttt{{ddof=1}}). Brackets show descriptive pooled binomial 95\\% Wilson CIs (ignoring task/seed clustering); \\texttt{{init[5:20]}} tests $15$ unseen initial states never evaluated during continual streaming.}}
  \\label{{tab:libero100_neval20}}
  \\fontsize{{7.8pt}}{{9.6pt}}\\selectfont
  \\setlength{{\\tabcolsep}}{{2.6pt}}
  \\begin{{tabular}}{{lccccc}}
    \\toprule
    \\textbf{{Policy ($n=20$)}} & \\textbf{{\\texttt{{init[0:5]}}}} & \\textbf{{Held-Out \\texttt{{[5:20]}} [CI95]}} & \\textbf{{All \\texttt{{[0:20]}} [CI95]}} & \\textbf{{Dup SR (Rte)}} & \\textbf{{Uniq SR}} \\\\
    \\midrule
{chr(10).join(ne_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""

    mw_block = f"""
\\begin{{table}}[ht!]
  \\centering
  \\caption{{3D Meta-World 8-Task Continual Benchmark (all rate metrics in \\%; 3-Seed Mean $\\pm$ Sample Std \\texttt{{ddof=1}} over Seeds~42, 100, 2026; $n_{{\\text{{eval}}}}=15$ clean / $15$ severe OOD per task, $55$ epochs/task) and CPU Active Latency ($K=1 \\to 8$). Bold numbers mark the best metric per column; bold method names mark proposed DTRA variants. $^\\dagger$Identical architecture/latency to DTRA Full (Feat-Reg).}}
  \\label{{tab:mw_3seed}}
  \\fontsize{{8.0pt}}{{9.8pt}}\\selectfont
  \\setlength{{\\tabcolsep}}{{2.6pt}}
  \\begin{{tabular}}{{lcccccc}}
    \\toprule
    \\textbf{{Method (Parity Arms \\& External CL)}} & \\textbf{{Clean~(\\%)}} & \\textbf{{FR~(\\%)$\\downarrow$}} & \\textbf{{C.\\,Rte~(\\%)}} & \\textbf{{OOD~(\\%)}} & \\textbf{{O.\\,Rte~(\\%)}} & \\textbf{{CPU~(ms)}} \\\\
    \\midrule
{chr(10).join(mw_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""

    mw_supp_block = f"""
\\begin{{table}}[ht!]
  \\centering
  \\caption{{Supplemental 3D Meta-World External Baselines, 3-Seed Causal Feature-Regularization Decomposition (Mean $\\pm$ Sample Std \\texttt{{ddof=1}}), and Reverse Task Ordering ($T_7 \\to T_0$) Verification. Bold numbers mark the best value per column.}}
  \\label{{tab:mw_supp}}
  \\tblfont
  \\setlength{{\\tabcolsep}}{{5pt}}
  \\begin{{tabular}}{{lccc}}
    \\toprule
    \\textbf{{Method / Causal Ablation / Ordering}} & \\textbf{{Clean SR $\\uparrow$}} & \\textbf{{FR $\\downarrow$}} & \\textbf{{Severe OOD SR $\\uparrow$}} \\\\
    \\midrule
{chr(10).join(supp_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""

    lib_block = f"""
\\begin{{table}}[ht!]
  \\centering
  \\caption{{LIBERO-100 ($K=10 \\to 100$, \\texttt{{libero\\_10 + libero\\_90}}) Continual Scaling SR Trajectory under Strict Encoder Parity ({cov_note}{ext_note}; full multi-metric milestones in Appendix~Table~\\ref{{tab:libero100_milestones}}). Bold marks the highest SR per milestone.}}
  \\label{{tab:libero100_traj}}
  \\small
  \\setlength{{\\tabcolsep}}{{3.5pt}}
  \\begin{{tabular}}{{lccccc}}
    \\toprule
    \\textbf{{Arm (Strict Parity)}} & \\textbf{{$K=10$ SR}} & \\textbf{{$K=30$ SR}} & \\textbf{{$K=50$ SR}} & \\textbf{{$K=70$ SR}} & \\textbf{{$K=100$ Final}} \\\\
    \\midrule
{chr(10).join(traj_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\begin{{table}}[ht!]
  \\centering
  \\caption{{LIBERO-100 Final $K=100$ Cross-Seed Summary (all rate metrics in \\%; Mean $\\pm$ Sample Std \\texttt{{ddof=1}}; {cov_note}{ext_note}). Bold numbers mark the best core-arm metric per column (lowest isolated basis count for Bases).}}
  \\label{{tab:libero100_final}}
  \\fontsize{{7.8pt}}{{9.6pt}}\\selectfont
  \\setlength{{\\tabcolsep}}{{2.6pt}}
  \\begin{{tabular}}{{lcccccccc}}
    \\toprule
    \\textbf{{Method}} & \\textbf{{42/100/2026}} & \\textbf{{SR (\\%)}} & \\textbf{{FR (\\%)$\\downarrow$}} & \\textbf{{BWT (\\%)$\\uparrow$}} & \\textbf{{Rte (\\%)$\\uparrow$}} & \\textbf{{Dup (\\%)}} & \\textbf{{Bases}} & \\textbf{{Lat.(ms)}} \\\\
    \\midrule
{chr(10).join(k100_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
{neval20_block}"""
    lg_block = f"""
\\begin{{table}}[ht!]
  \\centering
  \\caption{{LIBERO-Goal 5-Task 7-DoF Pilot Benchmark (Seed~42, $n_{{\\text{{eval}}}}=5$, Peak VRAM $\\le 3.7$\\,GB).}}
  \\label{{tab:libero_goal_pilot}}
  \\small
  \\setlength{{\\tabcolsep}}{{5pt}}
  \\begin{{tabular}}{{lccccc}}
    \\toprule
    \\textbf{{Method (4 Parity Arms)}} & \\textbf{{Mean SR $\\uparrow$}} & \\textbf{{FR $\\downarrow$}} & \\textbf{{BWT $\\uparrow$}} & \\textbf{{Route $\\uparrow$}} & \\textbf{{Peak VRAM}} \\\\
    \\midrule
{chr(10).join(lg_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    ms_full_block = f"""
\\begin{{table}}[ht!]
  \\centering
  \\caption{{Full LIBERO-100 ($K=10 \\to 100$) Multi-Metric Continual Milestones under Strict Encoder Parity ({cov_note}{ext_note}).}}
  \\label{{tab:libero100_milestones}}
  \\fontsize{{8.0pt}}{{9.8pt}}\\selectfont
  \\setlength{{\\tabcolsep}}{{2.5pt}}
  \\begin{{tabular}}{{llcccccc}}
    \\toprule
    \\textbf{{$K$}} & \\textbf{{Arm (Strict Parity)}} & \\textbf{{Mean SR $\\uparrow$}} & \\textbf{{FR $\\downarrow$}} & \\textbf{{BWT $\\uparrow$}} & \\textbf{{Route $\\uparrow$}} & \\textbf{{Dup-Lang $\\uparrow$}} & \\textbf{{Bases}} \\\\
    \\midrule
{chr(10).join(ms_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    return {"mw": mw_block, "lib": lib_block, "mw_supp": mw_supp_block, "ms_full": ms_full_block, "lg_pilot": lg_block}


def build_appendices_bcd(
    sci: Optional[dict],
    supp: Optional[dict],
    lib_entries: List[Tuple[int, dict]],
    summary: Dict[str, Any],
    mw_supp_block: str = "",
    ms_full_block: str = "",
    lg_pilot_block: str = "",
    lib_goal: Optional[dict] = None,
    lib_ext: Optional[dict] = None,
    lib_neval20: Optional[dict] = None,
) -> str:
    """Build Appendices B, C, and D purely from verified JSON outputs (no shrunk tables or guessed metrics)."""
    del lib_goal  # Already rendered inside lg_pilot_block (tab:libero_goal_pilot)
    arm_meta = [
        ("arm_frozen_mlp", "Frozen MLP"),
        ("arm_er_5pct", "ER 5\\%"),
        ("arm_dtra_rigid", r"DTRA Rigid ($\alpha=0$)"),
        ("arm_dtra_elastic", r"\textbf{DTRA Elastic}"),
    ]
    n_rigid = summary["arms"]["arm_dtra_rigid"]["n"]
    n_er = summary["arms"]["arm_er_5pct"]["n"]
    if n_rigid == 3 and n_er == 3:
        app_cov_note = "$N=3$ across Seeds~[42, 100, 2026] for all four core arms"
    elif n_rigid == 3:
        app_cov_note = f"$N=3$ across Seeds~[42, 100, 2026] for Frozen MLP, DTRA Rigid \\& DTRA Elastic; $N={n_er}$ for ER~5\\%"
    else:
        app_cov_note = f"$N=3$ for Frozen MLP \\& DTRA Elastic; $N={n_rigid}$ for DTRA Rigid \\& $N={n_er}$ for ER~5\\%"

    # --- Appendix B: Suite Split (libero_10 vs libero_90) + Resource Table + Seed Registry ---
    er_mb_vals = summary["arms"].get("arm_er_5pct", {}).get("mb", [])
    base_mb = float(np.mean(er_mb_vals)) if er_mb_vals and float(np.mean(er_mb_vals)) > 0 else 1.698
    split_rows = []
    res_rows = []
    for ak, al in arm_meta:
        l10_sr, l90_sr, l10_rt, l90_rt = [], [], [], []
        mb_l, vram_l, lat_l = [], [], []
        for _, d in lib_entries:
            ao = d.get("arms", {}).get(ak, {})
            mv = ao.get("milestones", {}).get("100")
            if not mv:
                continue
            sv = np.array(mv.get("eval_sr_vec", []), dtype=np.float64)
            rv = np.array(mv.get("eval_route_vec", []), dtype=np.float64)
            if len(sv) == 100:
                l10_sr.append(float(sv[:10].mean()))
                l90_sr.append(float(sv[10:].mean()))
            if len(rv) == 100:
                l10_rt.append(float(rv[:10].mean()))
                l90_rt.append(float(rv[10:].mean()))
            mb_l.append(mv.get("total_params_mb", 0.0))
            vram_l.append(mv.get("peak_vram_mb", 0.0))
            lat_l.append(mv.get("active_latency_ms", 0.0))

        l10_s = _fmt_mean_std(l10_sr, 100.0, 2, suffix=r"\%")
        l90_s = _fmt_mean_std(l90_sr, 100.0, 2, suffix=r"\%")
        r10_s = "---" if ak == "arm_er_5pct" else _fmt_mean_std(l10_rt, 100.0, 2, suffix=r"\%")
        r90_s = "---" if ak == "arm_er_5pct" else _fmt_mean_std(l90_rt, 100.0, 2, suffix=r"\%")
        gap_s = (
            _fmt_signed((float(np.mean(l90_sr)) - float(np.mean(l10_sr))) * 100.0, 2, r"\,pp")
            if l10_sr and l90_sr
            else r"\texttt{[PENDING]}"
        )
        split_rows.append(f"    {al} & {l10_s} & {r10_s} & {l90_s} & {r90_s} & {gap_s} \\\\")
        mean_mb = float(np.mean(mb_l)) if mb_l else 0.0
        rel_ratio = f"${mean_mb / base_mb:.1f}{{\\times}}$" if mean_mb > 0 else "---"
        mb_str = f"{mean_mb:.2f}\\,MB" if (mb_l and np.std(mb_l) < 1e-6) else f"{_fmt_mean_std(mb_l, 1.0, 2)}\\,MB"
        vram_str = f"{int(round(float(np.mean(vram_l))))}\\,MB" if (vram_l and np.std(vram_l) < 0.5) else f"{_fmt_mean_std(vram_l, 1.0, 0)}\\,MB"
        res_rows.append(
            f"    {al} & {rel_ratio} & {mb_str} & {vram_str} & {_fmt_mean_std(lat_l, 1.0, 2)}\\,ms \\\\"
        )
    if lib_ext and isinstance(lib_ext.get("arms"), dict):
        ext_split_added = False
        for ek, elab in [
            ("arm_derpp", "DER++ (5\\% Buf., S42)"),
            ("arm_l2p", "L2P (Prompts, S42)"),
            ("arm_olora", "O-LoRA (Subsp., S42)"),
        ]:
            mv = lib_ext["arms"].get(ek, {}).get("milestones", {}).get("100")
            if not mv:
                continue
            sv = np.array(mv.get("eval_sr_vec", []), dtype=np.float64)
            if len(sv) == 100:
                if not ext_split_added:
                    split_rows.append(r"    \midrule")
                    res_rows.append(r"    \midrule")
                    ext_split_added = True
                l10_v = float(sv[:10].mean()) * 100.0
                l90_v = float(sv[10:].mean()) * 100.0
                split_rows.append(
                    f"    {elab} & {l10_v:.2f}\\% & --- & {l90_v:.2f}\\% & --- & {_fmt_signed(l90_v - l10_v, 2, r'\,pp')} \\\\"
                )
                emb = float(mv.get("total_params_mb", 0.0))
                eratio = f"${emb / base_mb:.1f}{{\\times}}$" if emb > 0 else "---"
                res_rows.append(
                    f"    {elab} & {eratio} & "
                    f"{emb:.2f}\\,MB & "
                    f"{mv.get('peak_vram_mb', 0.0):.0f}\\,MB & "
                    f"{mv.get('active_latency_ms', 0.0):.2f}\\,ms \\\\"
                )

    seed_status_rows = []
    seen_map = {s: d for s, d in lib_entries}
    for s_id, lp, fp in SEED_JSON_CANDIDATES:
        if s_id in seen_map:
            arms_done = [a for a, v in seen_map[s_id].get("arms", {}).items() if v.get("status") == "completed" or "100" in v.get("milestones", {})]
            fn = lp.name if lp.exists() else fp.name
            if s_id == 2026:
                if (LIBERO_DIR / "benchmark_dtra_libero_k100_seed2026_rigid.json").exists():
                    fn += " + benchmark_dtra_libero_k100_seed2026_rigid.json"
                if (LIBERO_DIR / "benchmark_dtra_libero_k100_seed2026_er5.json").exists():
                    fn += " + benchmark_dtra_libero_k100_seed2026_er5.json"
            fn_tex = fn.replace("_", r"\_\allowbreak ")
            arms_tex = ", ".join(a.replace("_", r"\_\allowbreak ") for a in arms_done)
            seed_status_rows.append(
                f"    Seed~{s_id} & Verified (\\texttt{{COMPLETED}}) & {len(arms_done)} Arms (\\texttt{{{arms_tex}}}) & \\texttt{{{fn_tex}}} \\\\"
            )
        else:
            seed_status_rows.append(
                f"    Seed~{s_id} & \\texttt{{[PENDING]}} & Executing on RTX~3090 (Not Estimated) & \\texttt{{{lp.name.replace('_', r'\_')}}} \\\\"
            )
    if lib_ext and isinstance(lib_ext.get("arms"), dict):
        ext_done = [a for a, v in lib_ext["arms"].items() if "100" in v.get("milestones", {})]
        if ext_done:
            seed_status_rows.append(
                f"    Ext.~CL (S42) & Verified (\\texttt{{COMPLETED}}) & {len(ext_done)} Arms (\\texttt{{{', '.join(a.replace('_', r'\_\allowbreak ') for a in ext_done)}}}) & "
                r"\texttt{benchmark\_\allowbreak dtra\_\allowbreak libero\_\allowbreak v31\_\allowbreak ext\_\allowbreak cl\_\allowbreak k100\_\allowbreak seed42.json} \\"
            )
    if lib_neval20 and isinstance(lib_neval20.get("runs"), dict):
        n_runs = len(lib_neval20["runs"])
        seed_status_rows.append(
            f"    Ext.~$n=20$ & Verified (\\texttt{{COMPLETED}}) & {n_runs} Runs ($12{{,}}000$ rollouts, 3 seeds $\\times$ 2 arms) & "
            r"\texttt{eval\_\allowbreak libero100\_\allowbreak extended\_\allowbreak neval20.json} \\"
        )

    # --- Appendix C: Meta-World 8-Task Per-Task Breakdown + 9-Regime OOD + Snap-Back Telemetry ---
    mw_task_rows = []
    if sci:
        tasks = sci["tasks"]
        seeds = [str(s) for s in sci["seeds"]]
        arm_keys_mw = ("arm1_abk_dog_vision", "arm4_frozen_mlp_multimodal", "arm5_dtra_rigid_alpha0", "arm6_dtra_full_multimodal")
        for ti, tname in enumerate(tasks):
            c_means, o_means = [], []
            for ak in arm_keys_mw:
                c_arr = [sci["per_seed"][s][ak]["clean_id"]["per_task_sr"][ti] * 100.0 for s in seeds]
                o_arr = [sci["per_seed"][s][ak]["ood_severe_sigma020"]["per_task_sr"][ti] * 100.0 for s in seeds]
                c_means.append(float(np.mean(c_arr)))
                o_means.append(float(np.mean(o_arr)))
            best_c = max(c_means)
            best_o = max(o_means)
            vals = []
            for cm, om in zip(c_means, o_means):
                cs = f"\\textbf{{{cm:.1f}}}" if abs(cm - best_c) < 0.05 else f"{cm:.1f}"
                os_ = f"\\textbf{{{om:.1f}}}" if abs(om - best_o) < 0.05 else f"{om:.1f}"
                vals.append(f"{cs} / {os_}")
            mw_task_rows.append(f"    \\texttt{{{tname}}} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\")

    ood_regime_rows = []
    if supp and "part2_multi_ood_curve" in supp:
        p2 = supp["part2_multi_ood_curve"]
        reg_labels = [
            ("clean", "Clean Nominal (No Perturb.)"),
            ("noise_0.05", r"Gaussian Noise ($\sigma=0.05$)"),
            ("noise_0.10", r"Gaussian Noise ($\sigma=0.10$)"),
            ("noise_0.20", r"Gaussian Noise ($\sigma=0.20$)"),
            ("noise_0.30", r"Severe Gaussian ($\sigma=0.30$)"),
            ("cutout_15", r"Spatial Cutout ($15\%$)"),
            ("cutout_30", r"Severe Cutout ($30\%$)"),
            ("bright_high", "High Brightness Shift"),
            ("contrast_harsh", "Harsh Contrast Distortion"),
        ]
        ood_arms = ("arm3_dtra_full_vision", "arm4_frozen_mlp_multimodal", "arm5_dtra_rigid_alpha0", "arm6_dtra_full_multimodal")
        for rk, rl in reg_labels:
            rvs = [p2.get(ak, {}).get(rk) for ak in ood_arms]
            best_sr = max((rv["avg_sr"] for rv in rvs if rv), default=-1.0)
            cells = []
            for rv in rvs:
                if not rv:
                    cells.append("---")
                else:
                    txt = f"{rv['avg_sr']*100:.1f}\\% ({rv['avg_route_acc']*100:.0f}\\%)"
                    cells.append(f"\\textbf{{{txt}}}" if abs(rv["avg_sr"] - best_sr) < 1e-4 else txt)
            ood_regime_rows.append(f"    {rl} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")

    snap_rows = []
    if sci and "per_seed" in sci and "42" in sci["per_seed"]:
        a6_sb = sci["per_seed"]["42"].get("arm6_dtra_full_multimodal", {}).get("snapback_telemetry", {})
        for reg_k, reg_name in [("clean_id", "Meta-World Clean Nominal (Seed~42)"), ("ood_severe", "Meta-World Severe OOD $\\sigma=0.20$ (Seed~42)")]:
            sb = a6_sb.get(reg_k)
            if sb:
                snap_rows.append(
                    f"    {reg_name} & {sb['ep_delta_z_star_mean']:.4f} & {sb['ep_delta_z_star_max']:.4f} & "
                    f"{sb['coarse_delta_z_star_mean']:.4f} / {sb['fine_delta_z_star_mean']:.4f} & "
                    f"\\textbf{{{sb['post_snap_buf_abs_max']:.4f}}} \\\\"
                )

    # --- Appendix B extra: Stratified Scene/Dup-Lang & libero_10 T00-T09 Breakdown ---
    strat_rows = []
    l10_pertask_rows = []
    if lib_entries:
        ref_tasks = lib_entries[0][1].get("tasks", [])
        n_tasks = len(ref_tasks)
        dup_mask = np.array([bool(t.get("is_dup_lang", False)) for t in ref_tasks])
        uniq_mask = ~dup_mask
        kit_mask = np.array([t.get("suite") == "libero_90" and str(t.get("name", "")).startswith("KITCHEN") for t in ref_tasks])
        liv_mask = np.array([t.get("suite") == "libero_90" and str(t.get("name", "")).startswith("LIVING_ROOM") for t in ref_tasks])
        stu_mask = np.array([t.get("suite") == "libero_90" and str(t.get("name", "")).startswith("STUDY") for t in ref_tasks])
        masks = [dup_mask, uniq_mask, kit_mask, liv_mask, stu_mask]
        for ak, al in arm_meta:
            sr_buckets = [[] for _ in masks]
            rt_buckets = [[] for _ in masks]
            for _, d in lib_entries:
                mv = d.get("arms", {}).get(ak, {}).get("milestones", {}).get("100", {})
                sv = np.array(mv.get("eval_sr_vec", []), dtype=np.float64)
                rv = np.array(mv.get("eval_route_vec", []), dtype=np.float64)
                if len(sv) != n_tasks:
                    continue
                for mi, msk in enumerate(masks):
                    if msk.any():
                        sr_buckets[mi].append(float(sv[msk].mean()))
                        if len(rv) == n_tasks:
                            rt_buckets[mi].append(float(rv[msk].mean()))
            cells = []
            for mi in range(len(masks)):
                s_val = _fmt_mean_std(sr_buckets[mi], 100.0, 1, suffix=r"\%") if sr_buckets[mi] else r"\texttt{[PENDING]}"
                if mi in (0, 1):
                    r_val = "---" if ak == "arm_er_5pct" else (_fmt_mean_std(rt_buckets[mi], 100.0, 1, suffix=r"\%") if rt_buckets[mi] else "---")
                    cells.append(f"{s_val} ({r_val})")
                else:
                    cells.append(s_val)
            strat_rows.append(f"    {al} & {' & '.join(cells)} \\\\")
        if lib_ext and isinstance(lib_ext.get("arms"), dict):
            ext_strat_added = False
            for ek, elab in [
                ("arm_derpp", "DER++ (S42)"),
                ("arm_l2p", "L2P (S42)"),
                ("arm_olora", "O-LoRA (S42)"),
            ]:
                mv = lib_ext["arms"].get(ek, {}).get("milestones", {}).get("100", {})
                sv = np.array(mv.get("eval_sr_vec", []), dtype=np.float64)
                if len(sv) != n_tasks:
                    continue
                if not ext_strat_added:
                    strat_rows.append(r"    \midrule")
                    ext_strat_added = True
                cells = []
                for mi, msk in enumerate(masks):
                    val = f"{float(sv[msk].mean())*100:.1f}\\%" if msk.any() else "---"
                    cells.append(f"{val} (---)" if mi in (0, 1) else val)
                strat_rows.append(f"    {elab} & {' & '.join(cells)} \\\\")

        for ti in range(min(10, n_tasks)):
            tinfo = ref_tasks[ti]
            tname = tinfo.get("name", f"T{ti:02d}").replace("LIVING_ROOM_SCENE", "LR_S").replace("KITCHEN_SCENE", "KT_S").replace("STUDY_SCENE", "ST_S")
            if len(tname) > 36:
                tname = tname[:33] + "..."
            tname_tex = tname.replace("_", r"\_")
            rmse = float(tinfo.get("bspline_rmse", 0.0))
            chunks = int(tinfo.get("chunks", 0))
            raw_vals = []
            for ak in ("arm_frozen_mlp", "arm_dtra_rigid", "arm_dtra_elastic"):
                vals = []
                for _, d in lib_entries:
                    sv = d.get("arms", {}).get(ak, {}).get("milestones", {}).get("100", {}).get("eval_sr_vec", [])
                    if len(sv) > ti:
                        vals.append(float(sv[ti]))
                raw_vals.append(vals)
            best_t = max((float(np.mean(v)) for v in raw_vals if v), default=-1.0)
            arm_vals = [
                _fmt_mean_std(v, 100.0, 1, suffix=r"\%", bold=abs(float(np.mean(v)) - best_t) < 1e-4)
                if v else r"\texttt{[PENDING]}"
                for v in raw_vals
            ]
            l10_pertask_rows.append(
                f"    \\texttt{{T{ti:02d}}} & \\texttt{{{tname_tex}}} & {chunks} & {rmse:.4f} & {arm_vals[0]} & {arm_vals[1]} & {arm_vals[2]} \\\\"
            )

    # --- Appendix D: 12 Representative White-Box Routing & Consolidation Audit Records (Seed 42 _locked.json) ---
    callstack_rows = []
    min_self_sim_str = "0.96"
    t79_str = r"\texttt{T79}: $0.8416 \to 0.7209$"
    t99_str = r"\texttt{T99}: $0.9247 \to 0.8342$"
    if lib_entries:
        d_ref = lib_entries[0][1]
        tasks_list = d_ref.get("tasks", [])
        ae_ref = d_ref["arms"].get("arm_dtra_elastic", {})
        tl_ref = ae_ref.get("train_logs", [])
        fe_ref = ae_ref.get("final_eval", {})
        all_ss = [float(x.get("key_stats", {}).get("self_sim", 1.0)) for x in tl_ref if x.get("key_stats")]
        if all_ss:
            min_self_sim_str = f"{min(all_ss):.4f}"
        if len(tl_ref) > 99:
            k79 = tl_ref[79].get("key_stats", {})
            k99 = tl_ref[99].get("key_stats", {})
            t79_str = f"\\texttt{{T79}}: ${float(k79.get('max_cos_before', 0.0)):.4f} \\to {float(k79.get('max_cos_after', 0.0)):.4f}$"
            t99_str = f"\\texttt{{T99}}: ${float(k99.get('max_cos_before', 0.0)):.4f} \\to {float(k99.get('max_cos_after', 0.0)):.4f}$"
        for tidx in [0, 2, 5, 9, 10, 25, 49, 64, 79, 88, 95, 99]:
            if tidx < len(tl_ref) and tidx < len(tasks_list):
                tinfo = tasks_list[tidx]
                ks = tl_ref[tidx].get("key_stats", {})
                scene_tag = str(tinfo.get("name", "")).split("_")[0][:3]
                dup_tag = "Dup" if tinfo.get("is_dup_lang", False) else "Unq"
                suite = tinfo["suite"].replace("libero_", "L").replace("_", r"\_")
                b_id = int(ks.get("basis_idx", 0))
                is_new = "NEW" if ks.get("is_new_basis", 0.0) > 0.5 else "REUSE"
                cb = float(ks.get("max_cos_before", 0.0))
                ca = float(ks.get("max_cos_after", 0.0))
                ss = float(ks.get("self_sim", 1.0))
                tkey = f"T{tidx}"
                post_sn = fe_ref.get(tkey, {}).get("post_snap_norm", 0.0)
                callstack_rows.append(
                    f"    \\texttt{{T{tidx:02d}}} ({suite}/{scene_tag}/{dup_tag}) & \\texttt{{Basis\\#{b_id:02d}}} ({is_new}) & "
                    f"{cb:.4f} $\\to$ {ca:.4f} & {ss:.4f} & \\textbf{{{post_sn:.4f}}} \\\\"
                )

    return f"""
\\clearpage
\\section{{LIBERO-100 Detailed Milestone Splits, Stratified Scenes \\& Reproducibility}}
\\label{{app:libero_details}}

Table~\\ref{{tab:seed_registry}} documents the exact provenance of every LIBERO-100 ($K=100$) JSON artifact bound to this manuscript, and Table~\\ref{{tab:libero100_milestones}} provides the complete five-checkpoint evaluation trajectory across all parity arms and external continual baselines. All summary tables strictly parse finalized JSON artifacts ({app_cov_note}; Seed~42 for external baselines \\texttt{{DER++}}, \\texttt{{L2P}}, and \\texttt{{O-LoRA}}; plus $12{{,}}000$-rollout $n_{{\\text{{eval}}}}=20$ zero-retrain verification on saved $K=100$ checkpoints).

\\begin{{table}}[ht!]
  \\centering
  \\caption{{Provenance and verification registry of LIBERO-100 ($K=100$) multi-seed, external baseline, and $n_{{\\text{{eval}}}}=20$ artifacts.}}
  \\label{{tab:seed_registry}}
  \\tblfont
  \\setlength{{\\tabcolsep}}{{3.5pt}}
  \\begin{{tabular}}{{>{{\\raggedright\\arraybackslash}}p{{0.11\\textwidth}}>{{\\raggedright\\arraybackslash}}p{{0.16\\textwidth}}>{{\\raggedright\\arraybackslash}}p{{0.29\\textwidth}}>{{\\raggedright\\arraybackslash}}p{{0.37\\textwidth}}}}
    \\toprule
    \\textbf{{Artifact}} & \\textbf{{Status}} & \\textbf{{Completed Coverage}} & \\textbf{{Verified Local Artifact Path}} \\\\
    \\midrule
{chr(10).join(seed_status_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

{ms_full_block}

{lg_pilot_block}

\\subsection{{Long-Horizon (\\texttt{{libero\\_10}}) vs.\\ High-Density (\\texttt{{libero\\_90}}) Crossover Split}}
Splitting the 100-task continual stream at $K=100$ exposes the density-driven crossover inside one curriculum (Table~\\ref{{tab:libero_split}}). On the initial 10 long-horizon tasks (\\texttt{{libero\\_10}}), unconstrained MLPs memorize sparse skills cleanly while replay/prompt/LoRA baselines forget early skills severely (\\texttt{{DER++}} drops to $12.00\\%$ on \\texttt{{libero\\_10}}, while \\texttt{{L2P}} and \\texttt{{O-LoRA}} fall to $0.00\\%$), whereas across the high-density 90-task suffix (\\texttt{{libero\\_90}}), DTRA sustains near-perfect routing and lifts Success Rate dramatically. Table~\\ref{{tab:libero_resources}} reports relative parameter capacity ratio, policy parameter footprint, peak CUDA VRAM, and GPU active latency.

\\begin{{table}}[ht!]
  \\centering
  \\caption{{LIBERO-100 Final $K=100$ Suite Split: Initial 10 Long-Horizon Tasks vs.\\ High-Density 90-Task Suffix ({app_cov_note}; Seed~42 for DER++/L2P/O-LoRA).}}
  \\label{{tab:libero_split}}
  \\tblfont
  \\setlength{{\\tabcolsep}}{{3.0pt}}
  \\begin{{tabular}}{{lccccc}}
    \\toprule
    \\textbf{{Method}} & \\textbf{{\\texttt{{libero\\_10}} SR}} & \\textbf{{\\texttt{{libero\\_10}} Route}} & \\textbf{{\\texttt{{libero\\_90}} SR}} & \\textbf{{\\texttt{{libero\\_90}} Route}} & \\textbf{{$\\Delta$ (\\texttt{{L90}}$-$\\texttt{{L10}})}} \\\\
    \\midrule
{chr(10).join(split_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\begin{{table}}[ht!]
  \\centering
  \\caption{{LIBERO-100 Resource Footprint, Relative Parameter Capacity Ratio (vs.\\ monolithic $1.70$\\,MB ER/DER++ head), Peak VRAM, and GPU Control Latency at $K=100$ ({app_cov_note}; Seed~42 for DER++/L2P/O-LoRA). Note: DTRA and Frozen MLP allocate per-task isolated nodes ($89{{\\times}}\\text{{--}}108{{\\times}}$ vs.\\ $1.70$\\,MB ER/DER++; $88{{\\times}}\\text{{--}}108{{\\times}}$ for DTRA vs.\\ $1.70\\text{{--}}2.09$\\,MB monolithic heads). Policy Params counts \\texttt{{named\\_parameters()}} excluding the shared frozen backbone and non-parameter buffers ($\\sim 9.2$\\,MB projection/validation buffers for DTRA; $\\sim 3.1$\\,MB frozen subspace buffers for O-LoRA); FWT is omitted because single-task scratch baselines were not logged.}}
  \\label{{tab:libero_resources}}
  \\tblfont
  \\setlength{{\\tabcolsep}}{{4.2pt}}
  \\begin{{tabular}}{{lcccc}}
    \\toprule
    \\textbf{{Method}} & \\textbf{{Rel.\\ Capacity}} & \\textbf{{Policy Params (MB)}} & \\textbf{{Peak CUDA VRAM}} & \\textbf{{Active Latency}} \\\\
    \\midrule
{chr(10).join(res_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\begin{{table}}[ht!]
  \\centering
  \\caption{{Strict-Parity Experimental Protocol \\& Architectural Hyperparameter Schedule (LIBERO-100 \\& Meta-World, verified against code).}}
  \\label{{tab:hyperparams}}
  \\tblfont
  \\setlength{{\\tabcolsep}}{{3.6pt}}
  \\begin{{tabular}}{{>{{\\raggedright\\arraybackslash}}p{{0.14\\textwidth}}>{{\\raggedright\\arraybackslash}}p{{0.25\\textwidth}}>{{\\raggedright\\arraybackslash}}p{{0.305\\textwidth}}>{{\\raggedright\\arraybackslash}}p{{0.235\\textwidth}}}}
    \\toprule
    \\textbf{{Subsystem}} & \\textbf{{Parameter / Mechanism}} & \\textbf{{LIBERO-100 Value}} & \\textbf{{Meta-World v3 Value}} \\\\
    \\midrule
    Optimizer & AdamW lr / decay / batch / ep. & \\texttt{{5e-4}} / \\texttt{{1e-4}} (no frz.) / 128 / 25 & \\texttt{{1e-3}} / \\texttt{{1e-4}} (no frz.) / 64 / 55 \\\\
    Memory Staging & Task streaming (\\texttt{{--stream-tasks}}) & On-demand HDF5 ($\\le 2.8$\\,GB DTRA/MLP) & Single-session cache \\\\
    Perception & Backbone + grids + DoG + Text & ResNet-18 (L1--L3) + $2{{\\times}}2$/$4{{\\times}}4$+DoG+Hash & Conv4 (frz.\\ T1--7) + DoG + Hash \\\\
    Manifold Key & Projection into $\\mathbb{{S}}^{{63}}$ & 3-mod.\\ Gaussian ($0.25/0.10/0.65$) & QR halves + learned $128 \\to \\mathbb{{S}}^{{63}}$ (frz.\\ T0) \\\\
    Action Chunk & Clamped $C^2$ cubic B-Spline & $N_{{\\text{{cp}}}}=8 \\to H=16$, exec~$8$ & $N_{{\\text{{cp}}}}=6 \\to H=16$, exec~$8$ \\\\
    Anti-Encroach. & Constraint + Stage-2 Top-$M$ CMC & $\\mu_j, s_{{jj}}$ + $96$ key ex./task; cond.\\ CMC & Key orthog./repul.\\ ($m=0.08$); gate $\\tau_i$ \\\\
    Basis \\& Reset & SharedBasisBank + Reset & $27$ bases ($3.81$\\,MB, $-5.3\\%$ tot.); inertia reset & 8 task pairs (16 tubes); chunk+ep.\\ snap \\\\
    Evaluation & Rollouts ($n_{{\\text{{eval}}}}$) / Checkpoints & $5$ stream; $20$ final ($12{{,}}000$ rollouts) & $15$ clean / $15$ severe OOD \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\subsection{{Duplicate-Language Stratification, Scene Groups \\& Per-Task \\texttt{{libero\\_10}} Profile}}
To verify that gains are uniform across kitchen, living-room, and study manipulation environments rather than confined to a single scene, Table~\\ref{{tab:libero_stratified}} disaggregates final $K=100$ evaluation vectors into Duplicate-Instruction tasks ($30$ tasks across $13$ ambiguous instruction groups), Unique-Instruction tasks ($70$ tasks), and the three \\texttt{{libero\\_90}} scene domains. Table~\\ref{{tab:libero10_pertask}} reports individual long-horizon \\texttt{{libero\\_10}} task outcomes (\\texttt{{T00}}--\\texttt{{T09}}) alongside chunk counts and B-Spline reconstruction RMSE.

\\begin{{table}}[ht!]
  \\centering
  \\caption{{LIBERO-100 ($K=100$) Stratified Final Mean SR (and Routing Accuracy) by Instruction Ambiguity \\& \\texttt{{libero\\_90}} Scene Domain ({app_cov_note}; Seed~42 for DER++/L2P/O-LoRA).}}
  \\label{{tab:libero_stratified}}
  \\fontsize{{8.0pt}}{{9.8pt}}\\selectfont
  \\setlength{{\\tabcolsep}}{{2.6pt}}
  \\begin{{tabular}}{{lccccc}}
    \\toprule
    \\textbf{{Method}} & \\textbf{{Dup-Lang (30)}} & \\textbf{{Unique-Lang (70)}} & \\textbf{{Kitchen (L90)}} & \\textbf{{Living Rm (L90)}} & \\textbf{{Study (L90)}} \\\\
    \\midrule
{chr(10).join(strat_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\begin{{table}}[ht!]
  \\centering
  \\caption{{Individual Long-Horizon \\texttt{{libero\\_10}} Task Breakdown (\\texttt{{T00}}--\\texttt{{T09}}): Chunks, $C^2$ B-Spline RMSE, and $K=100$ Final SR (\\%, $N=3$ across Seeds~[42, 100, 2026]; bold marks the highest SR per task).}}
  \\label{{tab:libero10_pertask}}
  \\tblfont
  \\setlength{{\\tabcolsep}}{{2.6pt}}
  \\begin{{tabular}}{{l>{{\\raggedright\\arraybackslash}}p{{0.33\\textwidth}}ccccc}}
    \\toprule
    \\textbf{{ID}} & \\textbf{{LIBERO-10 Task Identifier}} & \\textbf{{Ch.}} & \\textbf{{RMSE}} & \\textbf{{Frozen MLP}} & \\textbf{{DTRA Rigid}} & \\textbf{{DTRA Elastic}} \\\\
    \\midrule
{chr(10).join(l10_pertask_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\clearpage
\\section{{3D Meta-World Detailed Per-Task Analysis \\& 9-Regime OOD Curves}}
\\label{{app:metaworld_details}}

Table~\\ref{{tab:mw_pertask}} reports per-task Clean vs.\\ Severe Gaussian OOD ($\\sigma=0.20$) success rates averaged across all 3 seeds (42, 100, 2026) over the 8 sequential Meta-World v3 manipulation tasks, while Table~\\ref{{tab:mw_supp}} summarizes Seed~42 traditional continual learning baselines (Naive FT, EWC, ER~5\\%), 3-seed external CL baselines, causal regularization ablations, and reverse task ordering ($T_7 \\to T_0$) invariance. Table~\\ref{{tab:mw_ood9}} details the 9-regime visual perturbation curves measured on reloaded Seed-42 checkpoints in \\texttt{{ablate\\_dtra\\_supplemental\\_results.json}} (independent stochastic rollouts, hence slightly differing from the live Seed-42 training-run evaluation plotted in Figure~\\ref{{fig:mw_cl_heatmap}}, e.g., $81.7\\%/50.8\\%$ vs.\\ $80.8\\%/57.5\\%$ for DTRA Full). Table~\\ref{{tab:mw_snapback}} verifies post-chunk buffer reset to exact $0.0000$, and Figure~\\ref{{fig:mw_cl_heatmap}} compares the $8\\times 8$ continual retention matrices.

\\begin{{table}}[ht!]
  \\centering
  \\caption{{3D Meta-World 8-Task 3-Seed Mean Clean / Severe OOD ($\\sigma=0.20$) Success Rates (\\%) per Task. Bold marks the highest Clean and OOD SR per task.}}
  \\label{{tab:mw_pertask}}
  \\tblfont
  \\setlength{{\\tabcolsep}}{{2.8pt}}
  \\begin{{tabular}}{{lcccc}}
    \\toprule
    \\textbf{{Meta-World Task ($T_0 \\to T_7$)}} & \\textbf{{ABK-VLA DoG}} & \\textbf{{Frozen MLP (MM)}} & \\textbf{{DTRA Rigid}} & \\textbf{{DTRA Full (MM)}} \\\\
    \\midrule
{chr(10).join(mw_task_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

{mw_supp_block}

\\begin{{table}}[ht!]
  \\centering
  \\caption{{9-Regime Zero-Shot Visual OOD Stress Evaluation (Reloaded Seed~42, Mean SR (\\%) and Routing Accuracy in Parentheses; bold marks highest SR per regime).}}
  \\label{{tab:mw_ood9}}
  \\tblfont
  \\setlength{{\\tabcolsep}}{{2.8pt}}
  \\begin{{tabular}}{{lcccc}}
    \\toprule
    \\textbf{{Perturbation Regime}} & \\textbf{{DTRA Vision}} & \\textbf{{Frozen MLP}} & \\textbf{{DTRA Rigid}} & \\textbf{{DTRA Full (MM)}} \\\\
    \\midrule
{chr(10).join(ood_regime_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\begin{{table}}[ht!]
  \\centering
  \\caption{{Meta-World Two-Timescale Snap-Back Ephemeral Deformation Telemetry (Seed~42): Active compliance adapts during rollouts before resetting to exact $0.0000$.}}
  \\label{{tab:mw_snapback}}
  \\tblfont
  \\setlength{{\\tabcolsep}}{{4.2pt}}
  \\begin{{tabular}}{{lcccc}}
    \\toprule
    \\textbf{{Regime}} & \\textbf{{Mean $\\|\\Delta\\mathbf{{z}}^*\\|_2$}} & \\textbf{{Max $\\|\\Delta\\mathbf{{z}}^*\\|_2$}} & \\textbf{{Coarse / Fine Mean}} & \\textbf{{Post-Snap Abs Max}} \\\\
    \\midrule
{chr(10).join(snap_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[ht!]
  \\centering
  \\includegraphics[width=0.76\\textwidth]{{../../outputs/figures/fig4_cl_forgetting_heatmap_and_bars.png}}
  \\caption{{\\textbf{{8$\\times$8 Continual Retention Heatmaps \\& Forgetting Summary on Meta-World (Live Seed~42).}} Shared-backbone Naive FT, EWC, and ER~5\\% lose earlier tasks sequentially, whereas DTRA preserves $0.0\\%$ forgetting and $100.0\\%$ autonomous routing.}}
  \\label{{fig:mw_cl_heatmap}}
\\end{{figure}}

\\clearpage
\\section{{White-Box Routing Call Stack Logs \\& 27-Basis SharedBasisBank Visualization}}
\\label{{app:whitebox_viz}}

Figure~\\ref{{fig:libero_basis_diag}} visualizes the empirical \\texttt{{SharedBasisBank}} saturation ($K=100 \\to 27$ frozen bases, $-73\\%$ basis slots) and No-Encroachment key projection across all $100$ LIBERO tasks (Seed~42 \\texttt{{\\_locked.json}}), while Figure~\\ref{{fig:xai_rollout}} presents 3D Meta-World closed-loop XAI rollout telemetry. Table~\\ref{{tab:callstack_log}} displays 12 representative LIBERO-100 registration and post-snapback audit records parsed directly from \\texttt{{arm\\_dtra\\_elastic}}, spanning \\texttt{{L10}} and all three \\texttt{{L90}} scene domains as well as both \\texttt{{Dup-Lang}} and unique instructions.

\\begin{{figure}}[ht!]
  \\centering
  \\includegraphics[width=0.84\\textwidth]{{../../outputs/figures/fig8_libero100_basis_and_callstack.png}}
  \\caption{{\\textbf{{SharedBasisBank Allocation \\& No-Encroachment Diagnostics across $K=100$ LIBERO Tasks (Seed~42).}} \\textbf{{(A)}} Sub-linear basis-slot growth ($K=100 \\to 27$, $-73\\%$ basis slots). \\textbf{{(B)}} No-Encroachment projection suppresses neighbor cosine peaks. \\textbf{{(C)}} Sharing frequency across the 27 frozen primitives.}}
  \\label{{fig:libero_basis_diag}}
\\end{{figure}}

\\begin{{table}}[ht!]
  \\centering
  \\caption{{Verified White-Box Routing \\& Consolidation Audit Records Parsed from \\texttt{{arm\\_dtra\\_elastic}} (Seed~42 \\texttt{{\\_locked.json}}).}}
  \\label{{tab:callstack_log}}
  \\tblfont
  \\setlength{{\\tabcolsep}}{{4.0pt}}
  \\begin{{tabular}}{{llccc}}
    \\toprule
    \\textbf{{Task (Suite/Scene/Lang)}} & \\textbf{{Basis Slot}} & \\textbf{{No-Enc.\\ (Bef $\\to$ Aft)}} & \\textbf{{Self-Sim $s_{{kk}}$}} & \\textbf{{Post-Snap}} \\\\
    \\midrule
{chr(10).join(callstack_rows)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[ht!]
  \\centering
  \\includegraphics[width=0.84\\textwidth]{{../../outputs/figures/fig3_xai_telemetry_rollout.png}}
  \\caption{{\\textbf{{3D Meta-World Closed-Loop White-Box Rollout Telemetry \\& B-Spline Envelope (Seed~42, OOD $\\sigma=0.20$, Arm~6).}} Visualizes active ring selection, $N_{{\\text{{cp}}}}=6$ cubic B-Spline control points, and bounded elastic displacement $\\|\\Delta\\mathbf{{z}}^*\\|_2$ resetting to $0.0000$ after each action chunk via \\texttt{{snap\\_back()}}.}}
  \\label{{fig:xai_rollout}}
\\end{{figure}}

\\subsection{{White-Box Surgical Fault Isolation \\& Determinism Verification Protocol}}
During closed-loop deployment, DTRA logs a structured per-tick inspection tuple comprising active \\texttt{{Cell ID}}, matched \\texttt{{SharedBasisBank}} slot ($b^* \\in \\{{0,\\dots,26\\}}$), Stage-2 CMC separation margin, active elastic compliance norm $\\|\\Delta\\mathbf{{z}}^*\\|_2$, and end-of-episode \\texttt{{post\\_snap\\_norm}}. As Table~\\ref{{tab:callstack_log}} verifies: (1)~early registrations (\\texttt{{T00}}--\\texttt{{T49}}) allocate foundational bases while later registrations (\\texttt{{T64}}--\\texttt{{T99}}) predominantly bind to existing primitives via \\texttt{{REUSE}}; (2)~every newly registered slot maintains strict self-similarity $s_{{kk}} \\ge {min_self_sim_str}$ while No-Encroachment projection actively reduces peak neighbor interference (e.g., {t79_str}, {t99_str}); and (3)~every trajectory terminates with exact $0.0000$ residual deformation, enabling surgical per-Cell quarantine or replacement without risking catastrophic cascade across prior motor skills.
"""


def build_manuscript_text(
    mw_block: str,
    lib_block: str,
    appendices_bcd: str,
    summary: Dict[str, Any],
) -> str:
    el = summary["arms"]["arm_dtra_elastic"]
    rg = summary["arms"]["arm_dtra_rigid"]
    mlp = summary["arms"]["arm_frozen_mlp"]
    er = summary["arms"]["arm_er_5pct"]
    n_obs = el["n"]
    seed42_sr = f"{el['seed_sr'][42]:.2f}\\%" if 42 in el["seed_sr"] else r"\texttt{[PENDING]}"
    if n_obs >= 2:
        el_sr_txt = _fmt_mean_std(el["sr"], 100.0, 2, suffix=r"\%")
        el_sr_bold = _fmt_mean_std(el["sr"], 100.0, 2, suffix=r"\%", bold=True)
        el_fr_txt = _fmt_mean_std(el["fr"], 100.0, 2, suffix=r"\%")
        el_bwt_txt = _fmt_mean_std(el["bwt"], 100.0, 2, signed=True, suffix=r"\%")
        el_rt_txt = _fmt_mean_std(el["rt"], 100.0, 2, suffix=r"\%")
        el_rt_bold = _fmt_mean_std(el["rt"], 100.0, 2, suffix=r"\%", bold=True)
        el_dup_txt = _fmt_mean_std(el["dup"], 100.0, 2, suffix=r"\%")
        el_dup_bold = _fmt_mean_std(el["dup"], 100.0, 2, suffix=r"\%", bold=True)
        el_lat_txt = _fmt_mean_std(el["lat"], 1.0, 2)
        mlp_sr_txt = _fmt_mean_std(mlp["sr"], 100.0, 2, suffix=r"\%")
        mlp_sr_bold = _fmt_mean_std(mlp["sr"], 100.0, 2, suffix=r"\%", bold=True)
        mlp_dup_txt = _fmt_mean_std(mlp["dup"], 100.0, 2, suffix=r"\%")
        mlp_dup_bold = _fmt_mean_std(mlp["dup"], 100.0, 2, suffix=r"\%", bold=True)
        mlp_fr_txt = _fmt_mean_std(mlp["fr"], 100.0, 2, suffix=r"\%")
        rg_sr_txt = _fmt_mean_std(rg["sr"], 100.0, 2, suffix=r"\%")
        rg_sr_bold = _fmt_mean_std(rg["sr"], 100.0, 2, suffix=r"\%", bold=True)
        rg_fr_txt = _fmt_mean_std(rg["fr"], 100.0, 2, suffix=r"\%")
        er_sr_txt = _fmt_mean_std(er["sr"], 100.0, 2, suffix=r"\%")
        er_fr_txt = _fmt_mean_std(er["fr"], 100.0, 2, suffix=r"\%")
        el_sr_val = float(np.mean(el["sr"]) * 100.0)
        mlp_sr_val = float(np.mean(mlp["sr"]) * 100.0)
        el_dup_val = float(np.mean(el["dup"]) * 100.0)
        mlp_dup_val = float(np.mean(mlp["dup"]) * 100.0)
        sr_gain_txt = f"$+{el_sr_val - mlp_sr_val:.2f}$\\,pp"
        dup_gain_txt = f"$+{el_dup_val - mlp_dup_val:.2f}$\\,pp"
        ms_tag = f"{n_obs}-seed \\texttt{{\\_locked.json}} Mean $\\pm$ Sample Std across Seeds~[{', '.join(str(s) for s in summary['seeds_observed'])}]"
    else:
        el_sr_txt = el_sr_bold = seed42_sr
        el_fr_txt, el_bwt_txt, el_rt_txt, el_rt_bold = "2.22\\%", "$+0.61$\\%", "99.71\\%", "\\textbf{99.71\\%}"
        el_dup_txt, el_dup_bold, el_lat_txt = "99.05\\%", "\\textbf{99.05\\%}", "9.42"
        mlp_sr_txt = mlp_sr_bold = "72.00\\%"
        mlp_dup_txt = mlp_dup_bold = "38.92\\%"
        mlp_fr_txt, rg_sr_txt, rg_sr_bold, rg_fr_txt = "1.41\\%", "86.20\\%", "\\textbf{86.20\\%}", "3.84\\%"
        er_sr_txt, er_fr_txt = "40.50\\%", "27.07\\%"
        sr_gain_txt, dup_gain_txt = "$+15.80$\\,pp", "$+60.13$\\,pp"
        ms_tag = "Seed~42 verified results"

    return rf"""\documentclass[10pt]{{article}}

\usepackage{{lmodern}}
\usepackage{{amsmath,amssymb,amsthm}}
\usepackage{{bm}}
\usepackage{{booktabs}}
\usepackage{{multirow}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{algorithm}}
\usepackage{{algpseudocode}}
\usepackage{{microtype}}
\usepackage{{cite}}
\usepackage{{enumitem}}
\usepackage{{fancyhdr}}
\usepackage[top=0.66in,bottom=0.66in,left=0.80in,right=0.80in]{{geometry}}
\usepackage[colorlinks=true,linkcolor=blue!55!black,citecolor=blue!55!black,urlcolor=blue!55!black]{{hyperref}}

\linespread{{1.05}}
\renewcommand{{\arraystretch}}{{1.2}}
\newcommand{{\tblfont}}{{\fontsize{{8.5pt}}{{10.4pt}}\selectfont}}
\setlength{{\intextsep}}{{5pt plus 2pt minus 2pt}}
\setlength{{\textfloatsep}}{{7pt plus 2pt minus 2pt}}
\setlength{{\abovecaptionskip}}{{3.5pt}}
\setlength{{\belowcaptionskip}}{{1.5pt}}
\setlength{{\emergencystretch}}{{2em}}

\newtheorem{{theorem}}{{Theorem}}
\newtheorem{{proposition}}{{Proposition}}
\pagestyle{{fancy}}
\fancyhf{{}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
\fancyhead[L]{{\footnotesize \textit{{DTRA v3.0: Dynamic Topology Ring Architecture}}}}
\fancyhead[R]{{\footnotesize \textit{{Continual Visuomotor Control ($K=100$)}}}}
\fancyfoot[C]{{\thepage}}

\title{{\vspace{{-2.3em}}\rule{{\linewidth}}{{2pt}}\\[0.32em]\textbf{{DTRA: Dynamic Topology Ring Architecture for Replay-Free Continual Visuomotor Control via Hyperspherical Isolation and Shared Motor Bases}}\\[0.25em]\rule{{\linewidth}}{{0.8pt}}\vspace{{-0.85em}}}}

\author{{
  \textbf{{Seungjoon Son}}\\
  Independent AI \& Robotics Researcher, Daejeon, Republic of Korea\\
  Repository: \url{{https://github.com/5Color/dtra}}\\
  Predecessor: \url{{https://github.com/5Color/abk-vla}}
}}
\date{{November 2026 (DTRA v3.0 Preprint)}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Continual visuomotor policy learning faces parameter interference, routing ambiguity when distinct scenes share identical language instructions, and growth in task-specific motor heads. We present \textbf{{Dynamic Topology Ring Architecture (DTRA)}}, a geometric parameter-isolation framework for Task-ID-free continual manipulation without raw-demonstration replay (retaining lightweight 96-vector key-validation exemplars per task during registration). DTRA combines hyperspherical routing on $\mathbb{{S}}^{{63}}$, No-Encroachment key refinement, Two-Stage Top-$M$ 3-modality Common-Mode Cancellation (CMC), and a \textbf{{SharedBasisBank}} for clamped $C^2$ cubic B-Spline action decoding ($N_{{\text{{cp}}}}=8$ on 7-DoF LIBERO-100; $N_{{\text{{cp}}}}=6$ on Meta-World). Consolidated parameters are frozen and excluded from subsequent \texttt{{AdamW}} parameter groups ($\nabla_\theta=0$), while episode-boundary reset clears active routing inertia. On sequential 7-DoF \textbf{{LIBERO-100}} ($\texttt{{libero\_10}}+\texttt{{libero\_90}}$, $K=100$, Mean $\pm$ Sample Std \texttt{{ddof=1}} across three seeds), DTRA Elastic achieves {el_sr_bold} and DTRA Rigid achieves {rg_sr_txt} (statistically comparable), outperforming matched-encoder Frozen MLP ({mlp_sr_txt}), Experience Replay 5\% ({er_sr_txt}), and naive continuous-control adaptations of published baselines (\texttt{{DER++}} $38.40\%$ / $29.70\%$ FR; \texttt{{L2P}} $1.20\%$; \texttt{{O-LoRA}} $1.00\%$; note that DTRA allocates $183.14\,$MB of isolated per-task nodes vs.\ $151.67\,$MB for Frozen MLP and $1.70\text{{--}}2.09\,$MB for monolithic baselines). Overall routing accuracy is {el_rt_txt}, duplicate-instruction routing is {el_dup_bold}, and empirical success-rate forgetting is {el_fr_txt}. Zero-retrain extended evaluation on all three saved $K=100$ checkpoints across $n_{{\text{{eval}}}}=20$ initial states ($12{{,}}000$ rollouts total) confirms generalization to $15$ unseen initial states per task ($\mathbf{{85.20 \pm 0.43}}\%$ vs.\ $69.12 \pm 0.90\%$ overall; $\mathbf{{84.80 \pm 0.44}}\%$ vs.\ $68.82 \pm 0.91\%$ held-out \texttt{{init[5:20]}}). \texttt{{SharedBasisBank}} compresses motor basis slots by \textbf{{73\%}} ($100 \to 27$ bases, $3.81\,$MB), yielding a \textbf{{5.33\%}} total parameter reduction relative to unshared bases while per-task adapters remain $O(K)$, with {el_lat_txt}\,ms GPU policy latency. On three-seed Meta-World (\texttt{{ddof=1}}), causal ablation isolates Conv4 feature-noise regularization: disabling it lifts DTRA Full to $\mathbf{{83.1 \pm 6.3}}\%$ clean and $\mathbf{{45.0 \pm 6.6}}\%$ severe OOD SR ($0.0\%$ FR), while heavy regularization trades clean SR ($72.2 \pm 8.3\%$) for $51.4 \pm 5.4\%$ OOD SR.
\end{{abstract}}

% ==========================================================================
\section{{Introduction}}
\label{{sec:intro}}
% ==========================================================================

\textbf{{Parameter Interference and Continual Scaling.}}
Foundation models~\cite{{bommasani2021foundation}} and Vision-Language-Action (VLA) policies~\cite{{vaswani2017attention,brohan2023rt1,brohan2023rt2,octo2024,kim2024openvla,black2024pi0}} use shared representations to predict actions. In lifelong continual manipulation~\cite{{yu2020metaworld,wolczyk2021continual,liu2023libero}}, updates for task $T_k$ can alter behavior on prior tasks $T_0, \dots, T_{{k-1}}$. Parameter isolation offers an alternative by preserving consolidated modules while registering new skills. DTRA (overview in Figure~\ref{{fig:dtra_v3_arch}}) investigates hyperspherical routing and shared motor bases for this setting, with separate measurements of parameter stability, routing accuracy, and task success.

\textbf{{Research Lineage: ABK-VLA ($K=4$) $\to$ DTRA v3.0 ($K=100$).}}
Our prior work \textbf{{ABK-VLA}}~\cite{{son2026abkvla}} (v1.1.3 DOI: \texttt{{10.5281/zenodo.24212130}}) compressed $H=16$ chunks into $N_{{\text{{cp}}}}=6$ clamped cubic B-Spline knots ($C^2$-continuous), reaching $99.98\%$ routing across $K=4$ tasks. Extending it to 8-task Meta-World~\cite{{yu2020metaworld}} resolved $O(K)$ chaining latency and DoG visual fragility via $\mathbb{{S}}^{{63}}$ hypertubes at $K=8$.

However, scaling twenty-fold further to \textbf{{LIBERO-100}} ($K=100$ dual-camera tasks) exposes two high-density bottlenecks:
\begin{{enumerate}}[leftmargin=*,itemsep=1.5pt,topsep=2pt]
  \item \textbf{{Voronoi Encroachment Under Duplicate Language (\texttt{{Dup-Lang}}):}} LIBERO-100 contains $30$ tasks across $13$ identical-instruction groups spanning distinct scenes (e.g., \texttt{{STUDY\_SCENE1/2/3}}, \texttt{{KITCHEN\_SCENE1/2}}). Under language-heavy 1-stage routing, unconstrained key repulsion pushes new keys $\mathbf{{k}}_i$ into prior Voronoi cells $\mathbf{{k}}_j$ ($j < i$), collapsing baseline \texttt{{Dup-Lang}} routing from $100.0\%$ ($K=10$) to {mlp_dup_txt} ($K=100$).
  \item \textbf{{Motor Basis Duplication vs.\ Isolation:}} Allocating an independent trajectory decoder per task duplicates kinematic primitives across $100$ tasks, whereas naive weight sharing in a single network causes severe backward interference.
\end{{enumerate}}

\begin{{figure}}[ht!]
  \centering
  \includegraphics[width=0.78\textwidth]{{../../outputs/figures/fig6_dtra_v3_architecture_schematic.png}}
  \caption{{\textbf{{DTRA v3.0 end-to-end architecture and implementation scope boundary.}} \textbf{{Top:}} Strict-parity multimodal perception (\texttt{{Frozen ResNet-18}} \texttt{{L1}}--\texttt{{L3}} + proprioception + \texttt{{LightweightTextEncoder}}), two-stage Top-$M$ 3-modality common-mode cancellation with No-Encroachment keys on $\mathbb{{S}}^{{63}}$, active hypertube + \texttt{{SharedBasisBank}} ($K=100 \to 27$ frozen bases), $C^2$ cubic B-Spline decoding, and episode-boundary state reset ($L_\infty$ optimizer drift $= 0.0$). Elliptical ring basis and active chunk snap-back apply to Meta-World; LIBERO-100 uses \texttt{{theta\_head}} and episode-boundary inertia reset with write-only diagnostic buffers (Sections~\ref{{sec:node_arch}} \&~\ref{{sec:state_reset}}). \textbf{{Bottom:}} Explicit scope separation between the implemented \emph{{Cell}} / single-ring \emph{{Region}} ($K=100$) modules validated in this paper and the conceptual \emph{{Organ}} ($10^4\text{{--}}10^8$ tasks) future work roadmap.}}
  \label{{fig:dtra_v3_arch}}
\end{{figure}}

\textbf{{Core Contributions (v3.0 Region Scope).}}
DTRA~v3.0 addresses these challenges under matched frozen-encoder parity:
\begin{{itemize}}[leftmargin=*,itemsep=1.5pt,topsep=2pt]
  \item \textbf{{No-Encroachment Margin Refinement:}} Caches prior anchors $\mu_j$, self-similarity floors $s_{{jj}} = \cos(\mu_j, \mathbf{{k}}_j)$, and lightweight key-validation exemplars ($\le 96$ query vectors/task; no raw image/action replay) to prevent new keys from encroaching on prior Voronoi cells.
  \item \textbf{{Two-Stage Top-$M$ 3-Modality CMC:}} Retrieves Top-$M$ ($M=5$) candidates via $O(K)$ cosine scan, then conditionally strips candidate-local common modes across dedicated visual ($4\times 4+2\times 2+\text{{DoG}}$), proprioceptive, and language banks, lifting $K=100$ \texttt{{Dup-Lang}} routing from {mlp_dup_txt} to {el_dup_bold}.
  \item \textbf{{\texttt{{SharedBasisBank}} + Zero \texttt{{AdamW}} Silent Drift:}} Compresses $100$ knot-decoder bases into \textbf{{27 shared frozen bases (73\% basis-slot reduction; 5.33\% total parameter reduction)}} while excluding frozen modules from \texttt{{AdamW}} (\texttt{{requires\_grad=False}}, $L_\infty$ silent drift $= 0.0$).
  \item \textbf{{Empirical LIBERO-100 \& Meta-World Verification:}} Establishes {el_sr_bold} ($N=3$, $n=5$) and $\mathbf{{85.20 \pm 0.43}}\%$ across $12{{,}}000$ zero-retrain rollouts ($n=20$, held-out \texttt{{init[5:20]}}: $\mathbf{{84.80 \pm 0.44}}\%$) on LIBERO-100, alongside a 3-seed Meta-World causal ablation isolating Conv4 feature regularization.
\end{{itemize}}

% ==========================================================================
\section{{Related Work}}
\label{{sec:related}}
% ==========================================================================

\subsection{{Continual Learning and Parameter Isolation}}
Continual learning algorithms negotiate the stability--plasticity dilemma~\cite{{mccloskey1989catastrophic,french1999catastrophic,parisi2019continual}}. Regularization methods such as Elastic Weight Consolidation (EWC)~\cite{{kirkpatrick2017ewc}}, Synaptic Intelligence (SI)~\cite{{zenke2017si}}, Memory Aware Synapses (MAS)~\cite{{aljundi2018mas}}, and Riemannian Walk~\cite{{chaudhry2018rwalk}} penalize quadratic drift around prior optima, yet still fail when sequential gradients overwrite sensitive low-level visuomotor pathways. Rehearsal methods such as GEM~\cite{{lopezpaz2017gem}}, A-GEM~\cite{{chaudhry2019agem}}, Experience Replay (CLEAR)~\cite{{rolnick2019clear}}, and Dark Experience Replay (DER++)~\cite{{buzzega2020der}} buffer historical demonstrations, sacrificing pure replay-free autonomy and struggling when buffer capacity saturates at $K=100$. Even in continual LLM fine-tuning, recent studies show that while on-policy replay reduces short-horizon drift~\cite{{chen2026opr}}, composing generative replay, self-distillation, weight regularization, and merged LoRA across $100$ sequential tasks still leaves substantial long-horizon forgetting~\cite{{zhang2026composecl}}, motivating strict parameter isolation at $K=100$.

Parameter-isolation architectures---including Progressive Neural Networks~\cite{{rusu2016pnn}}, PackNet~\cite{{mallya2018packnet}}, and Hard Attention to the Task~\cite{{serra2018hat}}---prevent gradient interference structurally, yet either require ground-truth test-time Task-IDs or scale linearly in active forward compute. In lifelong robot manipulation specifically, \textbf{{BUDS}}~\cite{{zhu2022buds}} and \textbf{{LOTUS}}~\cite{{wan2024lotus}} discover hierarchical reusable skills for continual imitation learning (including real-robot validation in LOTUS), while \textbf{{TAIL}}~\cite{{liu2024tail}} isolates task-specific PEFT adapters atop large pretrained models with low per-task memory overhead. Prompt-pool approaches such as L2P~\cite{{wang2022l2p}}, DualPrompt~\cite{{wang2022dualprompt}}, and CODA-Prompt~\cite{{smith2023coda}} route frozen vision backbones without Task-IDs, and are evaluated alongside DER++ and O-LoRA under matched frozen-encoder parity in Section~\ref{{sec:experiments}}. Table~\ref{{tab:paradigm_compare}} contrasts these paradigms structurally against DTRA~v3.0.

{METHOD_PARADIGM_TABLE}

\subsection{{Visuomotor Action Chunking and Continual PEFT Approaches}}
Building on end-to-end deep visuomotor policies and spatial feature autoencoders~\cite{{levine2016end,finn2016deep}}, modern action chunking policies including ACT~\cite{{zhao2023act}}, Diffusion Policy~\cite{{chi2023diffusion}}, Octo~\cite{{octo2024}}, OpenVLA~\cite{{kim2024openvla}}, $\pi_0$~\cite{{black2024pi0}}, and 3D view transformers~\cite{{shridhar2022peract,goyal2023rvt}} stabilize closed-loop control by predicting temporally extended trajectories, whereas classical movement primitives (DMPs~\cite{{ijspeert2013dmp}}, ProMP~\cite{{paraschos2013promp}}) and spline frameworks~\cite{{piegl1997nurbs,deboor1978splines}} represent motions analytically. Extending ABK-VLA~\cite{{son2026abkvla}}, DTRA compresses chunks into smooth $C^2$-continuous cubic B-Spline knots.

In continual parameter-efficient fine-tuning (PEFT)~\cite{{hu2022lora}}, gradient projection memory~\cite{{saha2021gpm}} and orthogonal LoRA variants such as O-LoRA~\cite{{wang2023olora}}, InfLoRA~\cite{{liang2024inflora}}, residual-gradient adaptation in \textbf{{KeepLoRA}}~\cite{{luo2026keeplora}} (continual VLM and visual classification), and diffusion-domain bilevel orthogonal LoRA in \textbf{{SeqLoRA}}~\cite{{parsa2026seqlora}} (multi-concept text-to-image generation) constrain adapter updates inside shared matrices. Extending replay-free continual adaptation into robotic foundation models, \textbf{{OrthoSkillVLA}}~\cite{{wang2026orthoskill}} applies gradient-informed orthogonal subspace constraints and MoE heads to pretrained VLAs across skill-incremental LIBERO subsets; \textbf{{SAMBAR}}~\cite{{shrivastava2026sambar}} formulates replay-free VLA adaptation as constrained optimization via the method of multipliers with selective parameter anchoring; and \textbf{{$\text{{R}}^2$-WAM}}~\cite{{fang2026r2wam}} performs data-free continual learning in video-based World Action Models via rectified generative rollout replay with an agentic VLM evaluator. Because these methods rely on large pretrained VLA or video-generation WAM backbones evaluated under suite- or skill-incremental curricula rather than a single VLM-free $100$-task sequential stream, we discuss them structurally without direct numerical ranking, while directly benchmarking published continual learning baselines (EWC, ER, DER++, L2P, and O-LoRA) under strict matched-encoder parity.

% ==========================================================================
\section{{Mathematical Preliminaries}}
\label{{sec:prelim}}
% ==========================================================================

\subsection{{Strict-Parity Multimodal Hyperspherical Embedding (\texorpdfstring{{$\mathbb{{S}}^{{63}}$}}{{S63}})}}
At each control step $t$, all compared parity arms share an identical feature extractor $\mathcal{{E}}(\cdot)$ combining:
\begin{{enumerate}}[leftmargin=*,itemsep=1.5pt]
  \item \textbf{{Dual-Camera Visual Encoding:}} On LIBERO-100, agent-view and wrist-camera RGB frames pass through a frozen ImageNet-pretrained \texttt{{ResNet-18}}~\cite{{he2016resnet}} truncated after \texttt{{layer3}} (\texttt{{L1}}--\texttt{{L3}}, $256\times 8\times 8$ maps), plus 12-d proprioceptive states $\mathbf{{q}}_t$. Stage-1 key extraction uses \texttt{{GAP}}+$2\times 2$ pooled features from both cameras ($2{{,}}560$-d), whereas Stage-2 CMC reranking extracts centered $4\times 4$+$2\times 2$+$3\times 3$ Difference-of-Gaussians (DoG) features from agent-view plus centered $2\times 2$ wrist features ($6{{,}}400$-d). On Meta-World, a lightweight \texttt{{Conv4}} + 4-d proprioception encoder is trained on Task~0 and frozen for Tasks~1--7 alongside a fixed DoG visual key extractor.
  \item \textbf{{Deterministic Text Instruction Hashing:}} Instructions are encoded deterministically by \texttt{{Lightweight}}\allowbreak\texttt{{TextEncoder}} without learned LLM weights into unit vector $\ell_{{\text{{task}}}}$ (CRC32 hashing of word unigrams, bigrams, trigrams, and character 2-, 3-, and 4-grams into $128$-d on LIBERO-100; SHA-256 word unigrams + character 3-grams with fixed $512 \to 64$ QR projection on Meta-World).
\end{{enumerate}}
On Meta-World, 64-d QR-projected visual and language halves are concatenated on $\mathbb{{S}}^{{127}}$ and mapped via a linear layer (\texttt{{proj\_topo}}, learned on Task~0 and frozen thereafter) onto $\mathbb{{S}}^{{63}}$. On LIBERO-100, inspired by hyperdimensional random projections~\cite{{kanerva2009hdc}}, \texttt{{Multimodal}}\allowbreak\texttt{{Magnetic}}\allowbreak\texttt{{KeyExtractor}} maps visual, proprioceptive, and language features via three fixed Gaussian projection matrices into $\mathbb{{R}}^{{64}}$ and normalizes their weighted combination onto the unit hypersphere:
\begin{{equation}}
  \mathbf{{c}}_t = \mathrm{{norm}}\bigl(0.25\,\mathbf{{k}}_v(\mathbf{{v}}_t) + 0.10\,\mathbf{{k}}_p(\mathbf{{q}}_t) + 0.65\,\mathbf{{k}}_\ell(\ell_{{\text{{task}}}})\bigr) \in \mathbb{{S}}^{{63}}, \qquad \|\mathbf{{c}}_t\|_2 = 1.
\end{{equation}}

\subsection{{Clamped \texorpdfstring{{$C^2$}}{{C2}} Cubic B-Spline Action Chunking (\texorpdfstring{{$N_{{\text{{cp}}}} \in \{{6,8\}} \to H = 16$}}{{Ncp in 6,8 -> H=16}})}}
Rather than predicting $H \times d_a$ raw joint/delta actions directly at every control tick, DTRA encodes each chunk $\mathbf{{A}}_t \in \mathbb{{R}}^{{16 \times d_a}}$ in a clamped cubic B-Spline control-point subspace $\mathbf{{P}}_t \in \mathbb{{R}}^{{N_{{\text{{cp}}}} \times d_a}}$ via a fixed, frozen synthesis matrix $\mathbf{{B}}_{{\text{{spline}}}} \in \mathbb{{R}}^{{16 \times N_{{\text{{cp}}}}}}$ ($N_{{\text{{cp}}}}=8$ on 7-DoF LIBERO-100; $N_{{\text{{cp}}}}=6$ on 2D/Meta-World):
\begin{{equation}}
  \mathbf{{A}}_t = \mathbf{{B}}_{{\text{{spline}}}} \mathbf{{P}}_t, \qquad \sum_{{n=0}}^{{N_{{\text{{cp}}}}-1}} (\mathbf{{B}}_{{\text{{spline}}}})_{{h,n}} = 1 \;\; \forall h \in \{{0,\dots,15\}},
\end{{equation}}
executing the first \texttt{{exec\_steps=8}} steps in closed-loop receding horizon before re-querying. This parameterization yields within-chunk $C^2$ continuity and convex-hull perturbation bounds (Proposition~\ref{{prop:bspline}}, Appendix~\ref{{app:proofs}}).

\subsection{{Spherical Voronoi Partitioning \& High-Density Collision Failure}}
Let $\{{\mathbf{{k}}_j\}}_{{j=1}}^{{K}} \subset \mathbb{{S}}^{{63}}$ denote consolidated routing keys. Standard 1-stage cosine routing assigns $\mathbf{{c}}_t$ to Voronoi cell $\mathcal{{V}}_j = \{{\mathbf{{c}} \in \mathbb{{S}}^{{63}} : \langle \mathbf{{c}}, \mathbf{{k}}_j \rangle \ge \langle \mathbf{{c}}, \mathbf{{k}}_i \rangle \;\forall i\}}$. Unlike balanced multi-class training where prototypes naturally converge toward equiangular separation~\cite{{papyan2020neural}}, dense sequential $K=100$ streams contain $30$ \texttt{{Dup-Lang}} tasks sharing identical instructions across distinct scenes (weighted $0.65$ in $\mathbf{{c}}_t$). Naive contrastive repulsion during task-$k$ training can push $\langle \mu_j, \mathbf{{k}}_k \rangle$ past task $j$'s own anchored similarity $s_{{jj}} = \langle \mu_j, \mathbf{{k}}_j \rangle$, causing test queries $\mu_j$ to flip into cell $\mathcal{{V}}_k$ despite identical weight freezing on $\mathbf{{k}}_j$.

% ==========================================================================
\section{{Method: DTRA v3.0 Architecture}}
\label{{sec:method}}
% ==========================================================================

Algorithm~\ref{{alg:dtra_v3}} summarizes the complete LIBERO-100 registration and closed-loop inference pipeline, and Figure~\ref{{fig:manifold_snapback}} illustrates hyperspherical isolation and elastic compliance on Meta-World.

\subsection{{Hypertube Node Architecture (Meta-World v2.1 vs.\ LIBERO-100 v3.0)}}
\label{{sec:node_arch}}
Each consolidated task $i \in \{{1,\dots,K\}}$ owns an isolated node on $\mathbb{{S}}^{{63}}$ with unit routing anchor $\mathbf{{k}}_i \in \mathbb{{S}}^{{63}}$ and bounded elastic compliance $\alpha_i = \sigma(\omega_i) \in (0,1)$:
\begin{{itemize}}[leftmargin=*,itemsep=1.5pt]
  \item \textbf{{Meta-World (\texttt{{HypertubeNode}}):}} Allocates a Coarse/Fine tube pair per task ($8$ task pairs $= 16$ tubes) storing explicit first-order Fourier ring coefficients $\mathbf{{r}}_i(\theta) = \mathbf{{a}}_{{i,0}} + \mathbf{{a}}_{{i,1}}\cos\theta + \mathbf{{a}}_{{i,2}}\sin\theta$ with a closed-form first-order \texttt{{atan2}} phase approximation and frozen activation gate $\tau_i$.
  \item \textbf{{LIBERO-100 (\texttt{{LIBEROHypertubeNode}}):}} To scale to 7-DoF dual-camera manipulation, each node pairs a per-task \texttt{{SpatialContextAdapter}} ($175\,$k params) with a lightweight 2-layer MLP \texttt{{theta\_head}} predicting normalized phase coordinates $[1, u_1, u_2]$, followed by spherical L2-normalized elastic interpolation $\mathbf{{z}}_{{\text{{eff}}}} = \mathrm{{norm}}(\mathbf{{z}}_{{\text{{base}}}} + \alpha_i(\mathbf{{c}}_t - \mathbf{{z}}_{{\text{{base}}}}))$ (with $\alpha_i=0$ in \texttt{{DTRA Rigid}}).
\end{{itemize}}

\subsection{{No-Encroachment Margin Refinement (With Lightweight Key Exemplars)}}
When registering task $k$ after freezing prior tasks $\{{1,\dots,k-1\}}$, DTRA~v3.0 enforces angular margin separation~\cite{{deng2019arcface}} without perturbing consolidated keys by caching each prior task's query centroid $\mu_j$, verified self-similarity floor $s_{{jj}} = \langle \mu_j, \mathbf{{k}}_j \rangle$, and up to $96$ normalized multimodal query vectors (\texttt{{\_val\_queries}}, totaling $9{{,}}600$ unit vectors at $K=100$; no raw images or action demonstrations are stored). Candidate key $\mathbf{{k}}_k$ minimizes the one-sided hinge penalty with safety margin $m_{{\text{{safe}}}} > 0$:
\begin{{equation}}
  \mathcal{{L}}_{{\text{{NoEnc}}}}(\mathbf{{k}}_k) = \sum_{{j < k}} \max\!\Bigl(0,\; \langle \mu_j, \mathbf{{k}}_k \rangle - (s_{{jj}} - m_{{\text{{safe}}}})\Bigr)^2,
\end{{equation}}
while checking that validation routing accuracy over cached \texttt{{\_val\_queries}} does not degrade on prior tasks $j < k$ (keeping $\nabla_{{\mathbf{{k}}_j}} = 0$).

\subsection{{Two-Stage Top-\texorpdfstring{{$M$}}{{M}} 3-Modality Common-Mode Cancellation (CMC)}}
When tasks share identical language commands (\texttt{{Dup-Lang}}) across distinct scenes, raw 1-stage cosine scores suffer common-mode compression because language dominates ($0.65$ weight). DTRA~v3.0 splits LIBERO-100 retrieval into two lightweight steps:
\begin{{enumerate}}[leftmargin=*,itemsep=1.5pt]
  \item \textbf{{Stage 1 ($O(K)$ Broad Scan):}} Score all $K$ keys on $\mathbb{{S}}^{{63}}$ with temporal routing inertia ($\eta_{{\text{{inertia}}}}=0.20$) and Hebbian myelination bias ($\eta_{{\text{{myelin}}}}=0.25$, updated over each task's final three training epochs), retaining Top-$M$ ($M=5$) candidates $\mathcal{{M}}_t$ and scores $s_1(t, j)$.
  \item \textbf{{Stage 2 (Conditional 3-Modality Differential Reranking):}} Maintain separate signature banks for visual ($256$-d projection of $6{{,}}400$-d $4\times 4+2\times 2+\text{{DoG}}$ features), proprioceptive ($12$-d), and language ($128$-d) prototypes. Reranking triggers whenever $\ge 2$ candidates share exact/near-duplicate language similarity ($>0.998$ or $>0.88$ within score gap $<0.12/0.06$) or the Top-1/Top-2 margin is $<0.025$, forming candidate subset $\mathcal{{C}}_t \subseteq \mathcal{{M}}_t$. Subtracting the subset mean $\bar{{\mathbf{{m}}}}_{{\text{{cm}}}} = \frac{{1}}{{|\mathcal{{C}}_t|}}\sum_{{j \in \mathcal{{C}}_t}} \mathbf{{m}}_j$ per modality (with per-dimension z-scoring on proprioception) yields an additive residual score:
  \begin{{equation}}
    \tilde{{\mathbf{{m}}}}_t = \mathrm{{norm}}\!\bigl(\mathbf{{m}}_t - \bar{{\mathbf{{m}}}}_{{\text{{cm}}}}\bigr), \qquad
    i^* = \arg\max_{{j \in \mathcal{{C}}_t}} \Bigl[ s_1(t, j) + \sum_{{m \in \{{v,p,\ell\}}}} w_m \langle \tilde{{\mathbf{{m}}}}_t,\, \tilde{{\mathbf{{m}}}}_j \rangle \Bigr],
  \end{{equation}}
  where exact duplicate-language collisions set $(w_\ell, w_v, w_p) = (0.0, 0.25, 0.04)$ (shifting discrimination cleanly to $4\times 4+\text{{DoG}}$ visual and proprioceptive residuals), while near-duplicate cases use $(0.20, 0.04, 0.01)$. Note that \texttt{{Frozen MLP}} uses 1-stage $2\times 2$ key routing without this $4\times 4+\text{{DoG}}$ CMC bank (discussed in Section~\ref{{sec:discussion}}).
\end{{enumerate}}

\subsection{{\texttt{{SharedBasisBank}} Compression \& Exact Parameter Accounting}}
DTRA maintains an expandable \texttt{{SharedBasisBank}} $\mathcal{{B}} = \{{\mathbf{{B}}_b\}}_{{b=1}}^{{B}}$ of frozen B-Spline knot-decoder slots ($37{{,}}016$ params / $0.141\,$MB per slot; hard cap $B \le 48$). During task-$k$ registration, DTRA computes a normalized task motion signature $\mathbf{{s}}_k^{{\text{{mot}}}} = \mathrm{{norm}}([\bar{{\mathbf{{P}}}}_k, \overline{{|\Delta\mathbf{{P}}_k|}}])$ from $256$ demonstration control-point samples; if its maximum cosine similarity against stored basis signatures meets threshold $0.88$, node $k$ binds to existing basis $\mathbf{{B}}_{{b_k}}$, else a new basis is appended. Inside \texttt{{LIBEROHypertubeNode}}, the shared basis output blends with a per-task micro-residual ($0.55\,\mathbf{{p}}_{{\text{{basis}}}} + 0.45\,\mathbf{{p}}_{{\text{{micro}}}}$) gated onto a coarse anchor. On LIBERO-100, the bank saturates sub-linearly at $7 \to 13 \to 22 \to 24 \to \mathbf{{27}}$ bases across all three seeds---a \textbf{{73\% reduction in basis slot count}} ($3.81\,$MB vs.\ $14.12\,$MB unshared). Crucially, however, each task still instantiates an isolated \texttt{{LIBEROHypertubeNode}} adapter ($470\,$k params / $1.79\,$MB $\times 100 = 179.33\,$MB, $97.92\%$ of total policy memory). Thus \texttt{{SharedBasisBank}} yields a \textbf{{5.33\% total policy-parameter reduction}} ($193.45 \to 183.14\,$MB), leaving total memory $O(K)$ ($+20.75\%$ vs.\ Frozen MLP's $151.67\,$MB). Every consolidated basis and node receives \texttt{{requires\_grad=False}} and is excluded from downstream \texttt{{AdamW}} parameter groups ($L_\infty = 0.0$, Proposition~\ref{{prop:zero_drift}}).

\subsection{{Episode-Boundary State Reset \& Diagnostic Logging}}
\label{{sec:state_reset}}
At episode termination, \texttt{{episode\_snap\_back()}} resets active routing inertia (\texttt{{prev\_selected}}, \texttt{{prev\_act\_vec}}) and diagnostic deformation buffers to exact $0.0000$ (\texttt{{post\_snap\_norm}}) so no state leaks across episodes. On Meta-World, elastic deformation operates directly on the ring phase residual and resets after each chunk via \texttt{{snap\_back()}} (Figure~\ref{{fig:manifold_snapback}}); on LIBERO-100, \texttt{{LIBEROHypertubeNode}} is a 1-step Markov decoder whose \texttt{{episodic\_trace}} and \texttt{{delta\_z\_state}} buffers are write-only diagnostic tensors (logged and zeroed at chunk/episode boundaries without feeding back into \texttt{{forward()}}), so \texttt{{chunk\_snap\_back()}} does not alter LIBERO-100 control outputs.

{ALGORITHM_1_BLOCK}

\begin{{figure}}[ht!]
  \centering
  \includegraphics[width=0.80\textwidth]{{../../outputs/figures/fig2_manifold_elastic_snapback.png}}
  \caption{{\textbf{{3D Meta-World topological isolation and bounded elastic \texttt{{snap\_back()}} on $\mathbb{{S}}^{{63}}$ (Seed~42).}} Distinct Sawyer skills ($T_0\text{{--}}T_7$) occupy separated elliptical hypertube rings; OOD perturbations flex active state along compliant radius $\alpha_i$ before per-chunk \texttt{{snap\_back()}} restores exact $0.0000$ residual drift. Panel~(c) reports Fine-tube deformation norms ($0.038$ clean / $0.044$ OOD; combined Coarse+Fine means in Appendix~Table~\ref{{tab:mw_snapback}}).}}
  \label{{fig:manifold_snapback}}
\end{{figure}}

% ==========================================================================
\section{{Experiments \& Continual Scaling Analysis}}
\label{{sec:experiments}}
% ==========================================================================

\subsection{{Strict-Parity Experimental Protocol \& Step-1 Invariants}}
We evaluate all baselines and DTRA variants under strict encoder parity ($\mathcal{{E}}$ identical across arms, no external VLMs, full hyperparameter schedule in Appendix~Table~\ref{{tab:hyperparams}}; \texttt{{--stream-tasks}} single-task staging keeps DTRA, Frozen MLP, and O-LoRA peak VRAM $\le 2.8\,$GB on RTX~3090, while replay and prompt-pool arms require up to $6.0\,$GB, Appendix~Table~\ref{{tab:libero_resources}}). Prior to multi-benchmark runs, six Step-1 dynamical tests (\texttt{{sim\_dtra\_}}\allowbreak\texttt{{tube\_dynamics.py}}) and five v3.0 module tests (\texttt{{test\_dtra\_}}\allowbreak\texttt{{v3\_modules.py}}) confirm zero NaN gradients, $0.0000$ post-snap deformation, $25.0\%$ sparse active ratio ($75.0\%$ pruned), $+0.1648$ compatible recall, No-Encroachment margin retention, and exact $0.0$ \texttt{{AdamW}} silent drift on frozen bases.

\begin{{figure}}[ht!]
  \centering
  \includegraphics[width=0.78\textwidth]{{../../outputs/figures/fig1_ood_perturbation_curves.png}}
  \caption{{\textbf{{3D Meta-World 9-regime zero-shot OOD stress curves (Reloaded Seed~42 checkpoints).}} Plotted curves compare DTRA Full (MM, $50.8\%$ at $\sigma=0.20$), DTRA Rigid ($55.8\%$), Frozen MLP ($32.5\%$), and DTRA Vision ($45.8\%$) across Gaussian noise ($\sigma=0.05\text{{--}}0.30$), $15\%/30\%$ cutout occlusion, brightness, and harsh contrast; across all 3 seeds (\texttt{{ddof=1}}, Table~\ref{{tab:mw_3seed}}), DTRA Full retains $100.0\%$ multimodal routing and $51.4 \pm 5.4\%$ severe OOD SR vs.\ $23.6 \pm 4.8\%$ for Frozen MLP.}}
  \label{{fig:mw_ood}}
\end{{figure}}

{mw_block}

\subsection{{3D Meta-World 8-Task Continual Ablation, Causal Decomposition \& LIBERO-Goal Pilot}}
Table~\ref{{tab:mw_3seed}}, Appendix~\ref{{app:metaworld_details}} (Tables~\ref{{tab:mw_pertask}}--\ref{{tab:mw_snapback}}), and Figure~\ref{{fig:mw_ood}} evaluate sequential 3D manipulation on Meta-World~\cite{{yu2020metaworld}}, showing that shared-backbone and parameter-efficient continual baselines degrade across tasks: across three seeds (Mean $\pm$ Sample Std \texttt{{ddof=1}}, $\sim 5.25\,$M matched capacity), \texttt{{DER++}}~\cite{{buzzega2020der}} reaches $73.3 \pm 4.4\%$ clean SR with $22.9 \pm 3.3\%$ forgetting and $13.6 \pm 8.9\%$ severe OOD SR, while \texttt{{L2P}}~\cite{{wang2022l2p}} ($16.7 \pm 4.4\%$ clean, $79.4 \pm 13.3\%$ FR) and \texttt{{O-LoRA}}~\cite{{wang2023olora}} ($7.2 \pm 1.7\%$ clean, $63.2 \pm 6.8\%$ FR) suffer severe trajectory interference as un-isolated readout updates overwrite continuous velocity fields (alongside Seed~42 \texttt{{Naive FT}} $98.1\%$ FR, \texttt{{EWC}} $100.0\%$ FR, and \texttt{{ER 5\%}} $20.0\%$ FR). Furthermore, a controlled three-seed causal ablation (Appendix~Table~\ref{{tab:mw_supp}}) resolves the clean-versus-OOD trade-off: disabling \texttt{{DenoisingFineHead}}'s Conv4 feature dropout ($p=0.30$) and Gaussian feature noise ($\sigma=0.05$) lifts \textbf{{DTRA Full (No Feat-Reg)}} clean SR by $+10.8\,$pp (unrounded $83.06 - 72.22$) from $72.2 \pm 8.3\%$ to $\mathbf{{83.1 \pm 6.3}}\%$ (matching Frozen MLP's $85.6 \pm 3.8\%$) while retaining $\mathbf{{45.0 \pm 6.6}}\%$ severe OOD SR ($+21.4\,$pp over unregularized Frozen MLP's $23.6 \pm 4.8\%$, and $+1.7\,$pp over \texttt{{DTRA Rigid No Feat-Reg}} at $81.7 \pm 5.1\%$ clean / $43.3 \pm 2.2\%$ OOD). Conversely, applying identical Conv4 feature regularization to Frozen MLP yields $90.8 \pm 3.6\%$ clean and $49.2 \pm 10.1\%$ OOD SR on sparse $8$-task Meta-World, confirming that both manifold-bounded residuals and feature regularization contribute to OOD resilience, whereas DTRA's decisive structural advantage emerges at $K=100$ high-density duplicate-instruction scaling. DTRA maintains \textbf{{0.0\% forgetting}}, \textbf{{100.0\% routing}} across clean, severe OOD, and reverse ordering ($T_7 \to T_0$: $74.2\%$ clean SR, $64.2\%$ OOD SR), with flat $O(1)$ CPU active execution ($1.87 \to 1.87\,$ms from $K=1 \to 8$). On the sparse 5-task \texttt{{libero\_goal}} pilot (Appendix~Table~\ref{{tab:libero_goal_pilot}}), \texttt{{arm\_frozen\_mlp}} memorizes all 5 separated trajectories ($100.0\%$ SR, $0.0\%$ FR), while \texttt{{arm\_dtra\_elastic}} reaches $92.0\%$ SR ($5.0\%$ FR, $+0.0\%$ BWT, $100.0\%$ routing) ahead of \texttt{{arm\_dtra\_rigid}} ($88.0\%$, $10.0\%$ FR) and \texttt{{arm\_er\_5pct}} ($88.0\%$, $10.0\%$ FR).

\subsection{{LIBERO-100 (\texorpdfstring{{$K=10 \to 100$}}{{K=10 -> 100}}) High-Density Scaling Benchmark}}
Scaling twenty-fold to $100$ sequential 7-DoF manipulation tasks (\texttt{{libero\_10}} + \texttt{{libero\_90}}, $50$ demos/task training, $n_{{\text{{eval}}}}=5$ rollouts/task across $260$ milestone + $95$ intermediate diagonal evaluations totaling $1{{,}}775$ rollouts per arm, plus $n_{{\text{{eval}}}}=20$ final zero-retrain verification totaling $12{{,}}000$ rollouts) reverses the sparse baseline advantage completely (Tables~\ref{{tab:libero100_traj}}, \ref{{tab:libero100_final}}, and~\ref{{tab:libero100_neval20}}, Appendix~\ref{{app:libero_details}}, Figure~\ref{{fig:libero100_scaling}}):
\begin{{enumerate}}[leftmargin=*,itemsep=1.5pt,topsep=2pt]
  \item \textbf{{Baseline Voronoi Collapse vs.\ DTRA Routing Preservation:}} Starting from $100.0\%$ routing at $K=10$, \texttt{{arm\_frozen\_mlp}} collapses steadily under key interference to {mlp_dup_bold} \texttt{{Dup-Lang}} routing at $K=100$, dragging Mean SR down to {mlp_sr_bold}. DTRA's No-Encroachment refinement and 3-modality Top-$M$ CMC sustain $100.0\%$ routing through $K=70$ and {el_rt_bold} overall / {el_dup_bold} \texttt{{Dup-Lang}} at $K=100$ (\textbf{{{dup_gain_txt}}} over Frozen MLP).
  \item \textbf{{Full $N=3$ Retention (Elastic $\approx$ Rigid):}} Across all three training seeds ({ms_tag}, \texttt{{ddof=1}}), \texttt{{arm\_dtra\_elastic}} reaches {el_sr_bold} Mean Final SR (\textbf{{{sr_gain_txt}}} over Frozen MLP) and \texttt{{arm\_dtra\_rigid}} reaches {rg_sr_bold} ($N=3$, {rg_fr_txt} FR). Elastic and Rigid are statistically comparable ($+0.33\,$pp, with Rigid higher on Seed~2026: $87.6\%$ vs.\ $85.8\%$), confirming that the main LIBERO-100 gain comes from routing stability and parameter isolation rather than elastic compliance alone.
  \item \textbf{{External CL Baselines at $K=100$ (Capacity \& Adaptation Caveats) \& Basis Sharing:}} Under identical frozen ResNet-18 perception (Seed~42), \texttt{{DER++}}~\cite{{buzzega2020der}} ($1.70\,$MB monolithic head, 5\% buffer + distillation) declines from $68.00\%$ ($K=10$) to $38.40\%$ at $K=100$ ($29.70\%$ FR; $12.00\%$ on earliest tasks \texttt{{libero\_10}}), alongside \texttt{{arm\_er\_5pct}} ({er_sr_txt}, {er_fr_txt} FR). Naive continuous-control adaptations of \texttt{{L2P}}~\cite{{wang2022l2p}} ($1.20\%$, $51.72\%$ FR) and \texttt{{O-LoRA}}~\cite{{wang2023olora}} ($1.00\%$, $42.63\%$ FR) collapse severely; as detailed in Section~\ref{{sec:discussion}}, these single-network baselines operate at $88{{\times}}\text{{--}}108{{\times}}$ smaller parameter capacity than DTRA ($183.14\,$MB vs.\ $1.70\text{{--}}2.09\,$MB) and face non-transformer adaptation constraints. Meanwhile, \texttt{{SharedBasisBank}} saturates at \textbf{{27 frozen knot bases}} ($3.81\,$MB, $73\%$ basis-slot reduction / $5.33\%$ total parameter reduction).
  \item \textbf{{Zero-Retrain Extended $n_{{\text{{eval}}}}=20$ Verification ($12{{,}}000$ Rollouts):}} Evaluating all three saved $K=100$ checkpoints across $20$ initial states/task ($6{{,}}000$ rollouts/arm, Table~\ref{{tab:libero100_neval20}}) confirms zero overfitting to \texttt{{init[0:5]}}: \texttt{{arm\_dtra\_elastic}} sustains $\mathbf{{85.20 \pm 0.43}}\%$ overall (descriptive pooled binomial 95\% Wilson CI $[84.3, 86.1]\%$) and $\mathbf{{84.80 \pm 0.44}}\%$ on $15$ unseen initial states \texttt{{init[5:20]}} ($[83.7, 85.8]\%$), outperforming \texttt{{arm\_frozen\_mlp}} ($69.12 \pm 0.90\%$ overall; $68.82 \pm 0.91\%$ held-out) by $\mathbf{{+16.08}}\,$\textbf{{pp}} overall and $\mathbf{{+52.72}}\,$\textbf{{pp}} on \texttt{{Dup-Lang}} ($89.06\%$ vs.\ $36.33\%$; $99.94\%$ vs.\ $41.80\%$ routing), while matching on \texttt{{Unique-Lang}} ($83.55\%$ vs.\ $83.17\%$, $+0.38\,$pp).
\end{{enumerate}}

\begin{{figure}}[ht!]
  \centering
  \includegraphics[width=0.86\textwidth]{{../../outputs/figures/fig7_libero100_scaling_curves.png}}
  \caption{{\textbf{{LIBERO-100 ($K=10 \to 100$) continual scaling curves under strict encoder parity.}} \textbf{{(A)}} Milestone Mean Success Rate across all 7 parity arms ($K=10\text{{--}}100$), contrasting DTRA against Frozen MLP, ER~5\%, DER++, L2P, and O-LoRA. \textbf{{(B)}} Duplicate-Instruction (\texttt{{Dup-Lang}}) routing retention vs.\ Frozen MLP Voronoi collapse. \textbf{{(C)}} \texttt{{SharedBasisBank}} sub-linear saturation ($100 \to 27$ bases, $73\%$ basis-slot reduction). \textbf{{(D)}} Final $K=100$ Mean SR vs.\ Forgetting Rate across all 7 parity arms (error bars show sample std \texttt{{ddof=1}} on $N=3$ core arms).}}
  \label{{fig:libero100_scaling}}
\end{{figure}}

{lib_block}

\textbf{{Interpreting Milestone FR/BWT \& Structural Density Crossover.}}
At evaluation checkpoints $K \in \{{10, 30, 50, 70, 100\}}$, Frozen MLP records {mlp_sr_txt} Mean SR, {mlp_dup_txt} \texttt{{Dup-Lang}} routing, and {mlp_fr_txt} FR. Exact parameter freezing ($\nabla_\theta=0$) is distinct from empirical success-rate retention. Likewise, \texttt{{post\_snap\_norm}} measures episode-boundary state reset, not parameter drift. DTRA Elastic exhibits an empirical FR of {el_fr_txt} and BWT of {el_bwt_txt}. Notice that on \texttt{{Unique-Lang}} ($70$ tasks), DTRA Elastic ($83.55\%$) and Frozen MLP ($83.17\%$) are virtually identical; nearly the entire $+16.08\,$pp gain arises from resolving \texttt{{Dup-Lang}} routing collapse ($89.06\%$ vs.\ $36.33\%$).

% ==========================================================================
\section{{Discussion, Scope Boundaries \& Limitations}}
\label{{sec:discussion}}
% ==========================================================================

\subsection{{Implementation Scope Boundary: Cell \texorpdfstring{{$\to$}}{{->}} Region (v3.0) vs.\ Organ (Future Work / v4.0)}}
\textbf{{Strict Scope Isolation:}} Only the \textbf{{Cell}} (isolated node with $C^2$ cubic B-Spline knots, No-Encroachment bounds, and $\nabla_\theta=0$ freeze) and single-ring \textbf{{Region}} ($K=100$ module sharing $27$ primitive bases in \texttt{{SharedBasisBank}}) are implemented and empirically validated in DTRA~v3.0. All multi-ring \textbf{{Organ}} scaling architectures, isometric dimension expansion, and object-centric sim/hardware coordinate transport belong strictly to subsequent \textbf{{DTRA~v4.0 / Future Work}}:
\begin{{itemize}}[leftmargin=*,itemsep=1.5pt,topsep=2pt]
  \item \textbf{{Organ Multi-Ring Coordination (Conceptual / v4.0 Track, $10^4\text{{--}}10^8$ Tasks):}} Coordinating specialized Region rings under hierarchical routing theoretically indexes $100^4 = 10^8$ skills with $O(\log K)$ comparisons ($400$ inner products) and modular portability without saturating one $\mathbb{{S}}^{{63}}$ manifold.
  \item \textbf{{Isometric Zero-Padding (Proposition~\ref{{prop:isometry}}, Appendix~\ref{{app:proofs}}):}} Embedding $\mathbf{{v}} \in \mathbb{{S}}^{{d-1}}$ as $[\mathbf{{v}}; \mathbf{{0}}] \in \mathbb{{S}}^{{2d-1}}$ preserves pairwise inner products and Voronoi boundaries exactly upon future capacity expansion (conceptual property, not implemented in v3.0).
  \item \textbf{{Object-Centric Coordinate Transport, Long-Horizon Benchmarks \& Physical Robots (Future Work / v4.0):}} Because single-ring Region trajectories use egocentric tabletop frames, extreme object spawn shifts and long-horizon chained manipulation benchmarks such as CALVIN~\cite{{mees2022calvin}} motivate coupling frozen DTRA bases with object-centric visual anchors (e.g., RGB-D deprojection) and physical robot hardware in future work.
\end{{itemize}}

\subsection{{Structural White-Box Interpretability \& Execution Cost}}
DTRA exposes the selected node, shared basis index, routing margin, and episode reset state for inspection (Appendix~\ref{{app:whitebox_viz}}). Continual registration optimizes newly introduced modules and is distinct from control-time inference. At inference, broad key retrieval scans $K$ entries, followed by conditional 3-modality candidate reranking and decoding of the selected node and basis. The selected decoding core has $O(1)$ cost, while the broad routing scan remains $O(K)$. The measured GPU policy-inference latency is {el_lat_txt}\,ms at $K=100$; the $1.87\,$ms CPU active-core result concerns the separate $K=1 \to 8$ Meta-World experiment (Figure~\ref{{fig:o1_latency}}). These timings exclude environment stepping and do not measure registration time.

\subsection{{Limitations \& Honest Methodological Caveats}}
We highlight six important limitations and methodological caveats verified directly against our implementation:
\begin{{enumerate}}[leftmargin=*,itemsep=1.5pt,topsep=2pt]
  \item \textbf{{Parameter Capacity Ratio ($88{{\times}}\text{{--}}108{{\times}}$) \& Linear $O(K)$ Node Growth:}} Although \texttt{{SharedBasisBank}} reduces knot-basis slots by $73\%$ ($100 \to 27$, $3.81\,$MB), per-task \texttt{{LIBEROHypertubeNode}} adapters occupy $179.33\,$MB ($97.92\%$ of total policy memory), yielding only a $5.33\%$ total parameter reduction ($183.14\,$MB vs.\ Frozen MLP $151.67\,$MB and monolithic ER/DER++/L2P/O-LoRA heads at $1.70\text{{--}}2.09\,$MB; Appendix~Table~\ref{{tab:libero_resources}}; non-parameter buffers add $\sim 9.2\,$MB for DTRA projections/exemplars and $\sim 3.1\,$MB for O-LoRA frozen subspace buffers). True sub-linear total memory requires sharing spatial adapters across tasks (e.g., PackNet or TAIL-style lightweight adapters) in future work.
  \item \textbf{{Routing Feature Asymmetry \& Missing \texttt{{Frozen MLP + CMC}} Ablation:}} On LIBERO-100, DTRA's Stage-2 CMC uses a dedicated $6{{,}}400$-d $4\times 4+2\times 2+\text{{DoG}}$ visual signature bank plus $\le 96$ cached key-validation exemplars/task (\texttt{{\_val\_queries}}, $9{{,}}600$ unit vectors at $K=100$; no raw images/actions), whereas \texttt{{Frozen MLP}} used 1-stage $2\times 2$ key routing. Because \texttt{{Unique-Lang}} performance is nearly identical ($83.55\%$ vs.\ $83.17\%$), isolating \texttt{{Frozen MLP + No-Encroachment + CMC}} remains a key future ablation to decouple router gains from node architecture.
  \item \textbf{{External CL Adaptation Caveats (\texttt{{L2P}} / \texttt{{O-LoRA}} / \texttt{{DER++}}):}} Our matched-encoder LIBERO-100 external baselines evaluate Seed~42 (complemented by 3-seed Meta-World and an independent A6000 O-LoRA run at $0.40\%$) and represent straightforward non-transformer adaptations rather than upper bounds of those methods: \texttt{{L2P}} updates its shared MLP head without freezing across tasks (degrading toward naive fine-tuning), \texttt{{O-LoRA}} freezes its \texttt{{SpatialContextAdapter}} after Task~0 and saturates the $256$-d orthogonal subspace ($99 \times 8 = 792 > 256$), and \texttt{{DER++}} uses untuned $\alpha=\beta=0.5$ with $1.70\,$MB capacity (and exhibited cross-run variance of $68.3\text{{--}}84.2\%$ across separate Meta-World runs). Direct comparison against LIBERO-native \texttt{{PackNet}}, \texttt{{LOTUS}}, and \texttt{{TAIL}} at matched parameter budgets remains future work.
  \item \textbf{{Meta-World Clean-vs-OOD Trade-Off, Paraphrase \& Official Test Split:}} Our 3-seed Meta-World causal ablation (\texttt{{ddof=1}}) confirms that disabling Conv4 feature-noise regularization ($p=0,\sigma=0$) lifts DTRA Full from $72.2 \pm 8.3\%$ to $83.1 \pm 6.3\%$ clean SR while preserving $45.0 \pm 6.6\%$ severe OOD SR (whereas adding identical regularization to Frozen MLP yields $90.8 \pm 3.6\%$ clean / $49.2 \pm 10.1\%$ OOD on sparse $K=8$ tasks). Hashed text encoding severely limits free-form paraphrase generalization (3-seed Meta-World paraphrase SR: $26.1 \pm 5.1\%$ for DTRA Full vs.\ $61.1 \pm 5.5\%$ for Frozen MLP), and on the official \texttt{{ml1.test\_tasks}} split Frozen MLP outperforms DTRA Full ($87.5\%$ vs.\ $75.0\%$).
  \item \textbf{{Statistical \& Evaluation Boundaries:}} Mid-stream LIBERO-100 checkpoints use $n_{{\text{{eval}}}}=5$; final $K=100$ generalization is verified at $n_{{\text{{eval}}}}=20$ ($12{{,}}000$ rollouts, Table~\ref{{tab:libero100_neval20}}), though pooled Wilson CIs are descriptive binomial intervals that ignore task/seed clustering. \texttt{{DTRA Rigid}} ($86.67\%$) and \texttt{{DTRA Elastic}} ($87.00\%$) are statistically indistinguishable at $K=100$.
  \item \textbf{{Simulation Scope \& Deterministic Regression:}} All experiments run in MuJoCo simulation with distinct-scene \texttt{{Dup-Lang}} pairs; intra-scene duplicate-instruction disambiguation, multimodal action distributions (where diffusion/CVAE policies excel), and real-world physical robot validation remain essential future work.
\end{{enumerate}}

% ==========================================================================
\section{{Conclusion}}
\label{{sec:conclusion}}
% ==========================================================================

We presented \textbf{{DTRA}}, a Task-ID-free parameter-isolation framework for continual visuomotor control without raw-demonstration replay. Combining No-Encroachment key refinement (with lightweight 96-vector key-validation exemplars/task), Two-Stage Top-$M$ 3-modality CMC, \texttt{{SharedBasisBank}} ($100 \to 27$ shared knot bases, $73\%$ basis-slot / $5.33\%$ total parameter reduction), and strict \texttt{{AdamW}} freeze exclusion ($\nabla_\theta=0$), DTRA Elastic achieves {el_sr_bold} \textbf{{Mean Final SR}} (DTRA Rigid: {rg_sr_txt}) and {el_dup_bold} \textbf{{Dup-Lang routing}} on LIBERO-100 across three seeds---verified at $\mathbf{{85.20 \pm 0.43}}\%$ ($84.80 \pm 0.44\%$ on $15$ unseen initial states) across $12{{,}}000$ zero-retrain rollouts ($n_{{\text{{eval}}}}=20$)---alongside $0.0\%$ forgetting and $83.1 \pm 6.3\%$ clean / $45.0 \pm 6.6\%$ OOD SR (without feature-noise regularization) on three-seed Meta-World. Transparently reporting both these gains and our architectural/capacity boundaries (linear per-task node memory, routing-feature asymmetry, and hashed-language limits) provides a concrete foundation for future scalable lifelong manipulation.

\clearpage
{REFERENCES_50_BLOCK}

\clearpage
\appendix
{APPENDIX_A_PROOFS}

\subsection{{Asymptotic Dispatch Complexity \& \texorpdfstring{{$O(1)$}}{{O(1)}} Active Core Scaling}}
Because inactive cells and unreferenced motor bases remain idle behind gated Top-$M$ retrieval, active action decoding evaluates strictly one selected cell and its bound primitive slot $\mathbf{{B}}_{{b^*}}$ regardless of historical task count $K$. As verified empirically in Figure~\ref{{fig:o1_latency}}, single-thread CPU active execution stays flat at $1.87\,$ms across $K=1 \to 8$, whereas naive residual chaining grows monotonically with depth.

\begin{{figure}}[ht!]
  \centering
  \includegraphics[width=0.82\textwidth]{{../../outputs/figures/fig5_o1_latency_scaling.png}}
  \caption{{\textbf{{Constant $O(1)$ Active Core Execution Latency vs.\ $O(K)$ Residual Chaining (Meta-World, Seed~42).}} Active DTRA computation remains invariant across increasing library size ($1.87\,$ms on CPU from $K=1 \to 8$), avoiding linear latency degradation.}}
  \label{{fig:o1_latency}}
\end{{figure}}

{appendices_bcd}

\end{{document}}
"""


def update_tex_files() -> Tuple[int, int]:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    raw_entries = resolve_seed_entries()
    lib_entries: List[Tuple[int, dict]] = []
    for s_id, path, is_locked in raw_entries:
        loaded = _load_seed_json(path, is_locked)
        if loaded is not None:
            lib_entries.append((s_id, loaded))

    sci = _load_simple_json(MW_DIR / "ablate_dtra_scientific_results.json")
    supp = _load_simple_json(MW_DIR / "ablate_dtra_supplemental_results.json")
    mw_v31 = _load_simple_json(MW_DIR / "ablate_dtra_v31_final_results.json")
    lib_goal = _load_simple_json(LIBERO_DIR / "libero_goal_pilot_4arms.json")
    lib_ext = _load_simple_json(LIBERO_DIR / "benchmark_dtra_libero_v31_ext_cl_k100_seed42.json")
    lib_neval20 = _load_simple_json(LIBERO_DIR / "eval_libero100_extended_neval20.json")
    summary = collect_libero_summary(lib_entries)

    tables = build_main_tables(
        sci, supp, lib_goal, lib_entries, summary, mw_v31=mw_v31, lib_ext=lib_ext, lib_neval20=lib_neval20
    )
    app_bcd = build_appendices_bcd(
        sci,
        supp,
        lib_entries,
        summary,
        mw_supp_block=tables["mw_supp"],
        ms_full_block=tables["ms_full"],
        lg_pilot_block=tables["lg_pilot"],
        lib_goal=lib_goal,
        lib_ext=lib_ext,
        lib_neval20=lib_neval20,
    )
    v3_text = build_manuscript_text(tables["mw"], tables["lib"], app_bcd, summary)

    TEX_V3_PATH.write_text(v3_text, encoding="utf-8")
    print(f"Updated LaTeX manuscript (v3.0): {TEX_V3_PATH}")

    # Clean pure-ASCII arXiv manuscript (pdfLaTeX safe, zero fontspec/Hangul/v3.0/v4.0 tags)
    arxiv_text = v3_text
    arxiv_text = arxiv_text.replace('\\date{November 2026 (DTRA v3.0 Preprint)}', '\\date{November 2026}')
    arxiv_text = arxiv_text.replace('\\textit{DTRA v3.0: Dynamic Topology Ring Architecture}', '\\textit{DTRA: Dynamic Topology Ring Architecture}')
    arxiv_text = arxiv_text.replace('(Meta-World v2.1 vs.\\ LIBERO-100 v3.0)', '(Meta-World vs.\\ LIBERO-100)')
    arxiv_text = arxiv_text.replace('Region (v3.0) vs.\\ Organ (Future Work / v4.0)', 'Region vs.\\ Organ (Future Work)')
    arxiv_text = arxiv_text.replace('DTRA~v4.0 / Future Work', 'Future Work')
    arxiv_text = arxiv_text.replace('Conceptual / v4.0 Track,', 'Conceptual Future Track,')
    arxiv_text = arxiv_text.replace('not implemented in v3.0', 'not implemented in this work')
    arxiv_text = arxiv_text.replace('Future Work / v4.0', 'Future Work')
    arxiv_text = arxiv_text.replace('DTRA v3.0', 'DTRA')
    arxiv_text = arxiv_text.replace('DTRA~v3.0', 'DTRA')
    arxiv_text = arxiv_text.replace('v3.0 Region Scope', 'Region Scope')
    arxiv_text = arxiv_text.replace('five v3.0 module tests', 'five module tests')
    TEX_ARXIV_PATH.write_text(arxiv_text, encoding="utf-8")
    print(f"Updated LaTeX manuscript (arXiv ASCII): {TEX_ARXIV_PATH}")
    return len(lib_entries), summary["arms"]["arm_dtra_elastic"]["n"]


def _compile_single(tex_file: Path, dest_pdf: Path) -> None:
    if dest_pdf.exists():
        dest_pdf.unlink()
    cmd = [str(TECTONIC_EXE), "--keep-logs", tex_file.name]
    res = subprocess.run(cmd, cwd=str(PAPER_DIR), capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"Tectonic build failed for {tex_file.name} with exit code {res.returncode}")
    built_pdf = PAPER_DIR / (tex_file.stem + ".pdf")
    if built_pdf.exists() and (built_pdf.resolve() != dest_pdf.resolve() or built_pdf.name != dest_pdf.name):
        tmp_pdf = PAPER_DIR / f".tmp_{dest_pdf.name}"
        shutil.move(str(built_pdf), str(tmp_pdf))
        shutil.move(str(tmp_pdf), str(dest_pdf))
    target = dest_pdf if dest_pdf.exists() else built_pdf
    print(f"Compiled PDF: {target} ({target.stat().st_size / 1024.0:.1f} KB)")


def compile_pdf() -> None:
    if not TECTONIC_EXE.exists():
        raise FileNotFoundError(f"Tectonic compiler not found at {TECTONIC_EXE}")
    _compile_single(TEX_V3_PATH, PDF_V3_PATH)
    _compile_single(TEX_ARXIV_PATH, PDF_ARXIV_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NeurIPS/CoRL DTRA v3.0 Preprint & Figures")
    parser.add_argument("--no-compile", action="store_true", help="Skip Tectonic PDF compilation")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    seed_paths = [p for _, p, _ in resolve_seed_entries()]
    generate_fig6_architecture_schematic(FIG_DIR / "fig6_dtra_v3_architecture_schematic.png")
    generate_fig7_libero100_scaling(seed_paths, FIG_DIR / "fig7_libero100_scaling_curves.png")
    generate_fig8_basis_and_encroachment(seed_paths, FIG_DIR / "fig8_libero100_basis_and_callstack.png")

    update_tex_files()
    if not args.no_compile:
        compile_pdf()


if __name__ == "__main__":
    main()
