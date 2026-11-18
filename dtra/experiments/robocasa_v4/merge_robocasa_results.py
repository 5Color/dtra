"""Merge GPU0 Baseline and GPU1 DTRA RoboCasa JSON results and summarize table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge RoboCasa GPU0 + GPU1 continual benchmark JSONs")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--out",
        required=True,
    )
    args = parser.parse_args()

    merged: Dict[str, Any] = {}
    for p in args.inputs:
        path = Path(p)
        if path.exists():
            merged.update(json.loads(path.read_text(encoding="utf-8")))

    order = ["Arm1_Flat_MLP", "Arm2_ER_5pct", "Arm3_DTRA_v4_NoTransport", "Arm4_DTRA_v4_Hypertube"]
    ordered = {k: merged[k] for k in order if k in merged}
    for k, v in merged.items():
        if k not in ordered:
            ordered[k] = v

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    print(f"Merged {len(ordered)} arms -> {out_path}\n")
    print("| Arm | Replay | Init SR (%) | Final SR (%) | Atomic (%) | Composite (%) | BWT (%) | RouteAcc (%) | Weight Drift |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for arm, d in ordered.items():
        rt = f"{d['route_acc']:.2f}%" if d.get("route_acc") is not None else "N/A"
        print(
            f"| `{arm}` | {d['replay']} | {d['initial_mean_sr']:.2f}% | {d['final_mean_sr']:.2f}% | "
            f"{d['atomic_final_sr']:.2f}% | {d['composite_final_sr']:.2f}% | {d['bwt']:+.2f}% | "
            f"{rt} | {d['parameter_drift']:.6f} |"
        )


if __name__ == "__main__":
    main()
