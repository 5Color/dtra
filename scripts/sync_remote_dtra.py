"""scripts/sync_remote_dtra.py

로컬 `drawing-vla` 저장소의 새 디렉터리 구조(`dtra/`, `v1_drawing_vla/src/`, `tests/`, `scripts/`)를
원격 RTX 3090 호스트(`C:\\Users\\sonsj\\dtra`)와 안전하게 동기화하는 공용 유틸리티.

기존 `outputs/` 결과물·체크포인트·캐시는 덮어쓰지 않고 코드/테스트만 아카이브로 반영하며,
`--pull-outputs` 옵션으로 원격 `outputs/metaworld`, `outputs/libero` JSON 결과를 로컬로 회수할 수 있습니다.

사용 예시:
  python scripts/sync_remote_dtra.py --dry-run
  python scripts/sync_remote_dtra.py
  python scripts/sync_remote_dtra.py --pull-outputs
  python scripts/sync_remote_dtra.py --run-env libero --run "python dtra/experiments/libero/benchmark_dtra_libero.py --help"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "sonsj@100.100.160.99"
DEFAULT_REMOTE_DIR = r"C:\Users\sonsj\dtra"
SYNC_DIRS = ("dtra", "v1_drawing_vla/src", "v1_drawing_vla/handvsai", "tests", "scripts")
SYNC_FILES = ("requirements.txt", "RESEARCH_LINEAGE.md", "README.md")
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".pt", ".pth", ".npz", ".sqlite", ".log", ".pdf", ".zip")
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git"}


def _should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    if "handvsai" in path.parts:
        if "tools" in path.parts:
            return True
        if path.name not in ("__init__.py", "scoring.py", "targets_128.json", "templates.json", "README.md"):
            return True
    if path.name == "convert_dataset_to_bspline.py":
        return True
    return False


def build_sync_archive(archive_path: Path) -> int:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with tarfile.open(archive_path, "w:gz") as tar:
        for rel_dir in SYNC_DIRS:
            src_dir = PROJECT_ROOT / rel_dir
            if not src_dir.exists():
                continue
            for fp in sorted(src_dir.rglob("*")):
                if fp.is_dir() or _should_exclude(fp):
                    continue
                arcname = fp.relative_to(PROJECT_ROOT).as_posix()
                tar.add(fp, arcname=arcname)
                added += 1
        for rel_file in SYNC_FILES:
            fp = PROJECT_ROOT / rel_file
            if fp.exists():
                tar.add(fp, arcname=rel_file)
                added += 1
    return added


def pull_remote_outputs(host: str, remote_dir: str) -> None:
    remote_unix = remote_dir.replace("\\", "/")
    for sub in ("metaworld", "libero"):
        local_dir = PROJECT_ROOT / "outputs" / sub
        local_dir.mkdir(parents=True, exist_ok=True)
        cmd = f'powershell -Command "Get-ChildItem -Path \'{remote_dir}\\outputs\\{sub}\\*.json\' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name"'
        res = subprocess.run(["ssh", host, cmd], capture_output=True, text=True)
        names = [line.strip() for line in res.stdout.splitlines() if line.strip().endswith(".json")]
        for name in names:
            src = f"{host}:{remote_unix}/outputs/{sub}/{name}"
            dst = str(local_dir / name)
            print(f"[Pull] {src} -> {dst}")
            subprocess.run(["scp", src, dst], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync reorganized drawing-vla/DTRA code to remote RTX 3090")
    parser.add_argument("--host", default=DEFAULT_HOST, help="SSH host (default: sonsj@100.100.160.99)")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help="Remote repo root (default: C:\\Users\\sonsj\\dtra)")
    parser.add_argument("--dry-run", action="store_true", help="Build archive and inspect file count without SSH transfer")
    parser.add_argument("--pull-outputs", action="store_true", help="Fetch outputs/{metaworld,libero}/*.json from remote")
    parser.add_argument("--run", default="", help="Optional command to execute inside remote repo after sync")
    parser.add_argument("--run-env", default="dtra", help="Conda env name when --run is provided (dtra | libero)")
    args = parser.parse_args()

    tmp_archive = PROJECT_ROOT / "outputs" / ".dtra_sync_bundle.tar.gz"
    count = build_sync_archive(tmp_archive)
    size_kb = tmp_archive.stat().st_size / 1024.0
    print(f"[Sync Bundle] {count} files packaged -> {tmp_archive} ({size_kb:.1f} KB)")

    if args.dry_run:
        tmp_archive.unlink(missing_ok=True)
        print("[Dry-Run] Archive check completed cleanly without remote transfer.")
        return 0

    remote_tar = f"{args.remote_dir.rstrip(os.sep)}\\.dtra_sync_bundle.tar.gz"
    remote_tar_scp = remote_tar.replace("\\", "/")
    print(f"[Upload] {tmp_archive} -> {args.host}:{remote_tar_scp}")
    subprocess.run(["scp", str(tmp_archive), f"{args.host}:{remote_tar_scp}"], check=True)
    tmp_archive.unlink(missing_ok=True)

    extract_cmd = (
        f'powershell -Command "New-Item -ItemType Directory -Force -Path \'{args.remote_dir}\' | Out-Null; '
        f"cd '{args.remote_dir}'; tar -xzf '.dtra_sync_bundle.tar.gz'; Remove-Item -Force '.dtra_sync_bundle.tar.gz'\""
    )
    subprocess.run(["ssh", args.host, extract_cmd], check=True)
    print(f"[Synced] Codebase unpacked into {args.host}:{args.remote_dir}")

    if args.pull_outputs:
        pull_remote_outputs(args.host, args.remote_dir)

    if args.run:
        py_exe = rf"C:\Users\sonsj\miniconda3\envs\{args.run_env}\python.exe"
        escaped = args.run.replace("python ", f"& '{py_exe}' ", 1) if args.run.startswith("python ") else args.run
        remote_run_cmd = f'powershell -Command "cd \'{args.remote_dir}\'; {escaped}"'
        print(f"[Remote Run ({args.run_env})] {args.run}")
        subprocess.run(["ssh", args.host, remote_run_cmd], check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
