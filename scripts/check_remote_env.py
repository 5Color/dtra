"""scripts/check_remote_env.py

로컬 또는 원격 RTX 3090(`C:\\Users\\sonsj\\dtra`)의 패키지 임포트, CUDA GPU,
Meta-World 및 LIBERO 벤치마크 실행 환경을 한 번에 진단하는 유틸리티.

사용 예시:
  python scripts/check_remote_env.py
  python scripts/check_remote_env.py --remote --env dtra
  python scripts/check_remote_env.py --remote --env libero
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_local(env_name: str = "dtra") -> int:
    print(f"[Env Check] Repo root: {PROJECT_ROOT} (profile={env_name})")
    print(f"[Env Check] Python: {sys.executable} ({sys.version.split()[0]})")

    try:
        import torch

        print(f"[PyTorch] version={torch.__version__} | cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"[GPU] {torch.cuda.get_device_name(0)} | VRAM={props.total_memory / 1e9:.2f} GB")
    except Exception as exc:
        print(f"[PyTorch] IMPORT FAILED: {exc}")

    common_mods = [
        "dtra.core.bspline_core",
        "dtra.core.sim_dtra_tube_dynamics",
        "gymnasium",
    ]
    metaworld_mods = [
        "dtra.experiments.metaworld.benchmark_dtra_3d_metaworld",
        "dtra.experiments.metaworld.ablate_dtra_scientific",
        "dtra.experiments.metaworld.ablate_dtra_supplemental",
        "dtra.visualization.visualize_dtra_xai",
        "metaworld",
    ]
    libero_mods = [
        "dtra.experiments.libero.benchmark_dtra_libero",
        "libero.libero",
        "robosuite",
        "h5py",
    ]
    optional = set(libero_mods if env_name == "dtra" else metaworld_mods)
    modules = common_mods + metaworld_mods + libero_mods
    failed_required = 0
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"  [OK]   {mod}")
        except Exception as exc:
            tag = "OPT" if mod in optional else "FAIL"
            if tag == "FAIL":
                failed_required += 1
            print(f"  [{tag}] {mod} -> {type(exc).__name__}: {exc}")

    paths = [
        PROJECT_ROOT / "outputs" / "metaworld" / "ablate_dtra_scientific_results.json",
        PROJECT_ROOT / "outputs" / "metaworld" / "ablate_dtra_supplemental_results.json",
        PROJECT_ROOT / "outputs" / "libero" / "libero_goal_pilot_4arms.json",
        PROJECT_ROOT / "outputs" / "figures" / "fig0_dtra_architecture_schematic.png",
    ]
    print("[Artifact Check]")
    for p in paths:
        status = f"OK ({p.stat().st_size // 1024} KB)" if p.exists() else "MISSING"
        print(f"  {p.relative_to(PROJECT_ROOT)} -> {status}")

    return 1 if failed_required else 0


def check_remote(host: str, remote_dir: str, env_name: str) -> int:
    py_exe = rf"C:\Users\sonsj\miniconda3\envs\{env_name}\python.exe"
    cmd = f'powershell -Command "cd \'{remote_dir}\'; & \'{py_exe}\' scripts/check_remote_env.py --env {env_name}"'
    print(f"[Remote Check] host={host} env={env_name} dir={remote_dir}")
    res = subprocess.run(["ssh", host, cmd])
    return int(res.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose local or remote RTX 3090 DTRA/LIBERO environment")
    parser.add_argument("--remote", action="store_true", help="Execute check on remote RTX 3090 via SSH")
    parser.add_argument("--host", default="sonsj@100.100.160.99", help="SSH host")
    parser.add_argument("--remote-dir", default=r"C:\Users\sonsj\dtra", help="Remote repo directory")
    parser.add_argument("--env", default="dtra", choices=["dtra", "libero"], help="Remote conda env name")
    args = parser.parse_args()

    if args.remote:
        return check_remote(args.host, args.remote_dir, args.env)
    return check_local(args.env)


if __name__ == "__main__":
    raise SystemExit(main())
