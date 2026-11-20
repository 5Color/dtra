@echo off
setlocal
cd /d C:\Users\sonsj\dtra
set PY=C:\Users\sonsj\miniconda3\envs\libero\python.exe
set LOG=C:\Users\sonsj\dtra\outputs\libero\v31_remote_queue.log

echo ============================================================ >> "%LOG%"
echo [START %DATE% %TIME%] DTRA v3.1 Remote RTX 3090 Queue >> "%LOG%"
echo ============================================================ >> "%LOG%"

echo [Stage 1] External Published CL Baselines (DER++, L2P, O-LoRA) on LIBERO-100 Seed 42 >> "%LOG%"
:stage1
"%PY%" -u -m dtra.experiments.libero.benchmark_dtra_libero_v31_ext ^
  --mode external_cl ^
  --suites libero_10,libero_90 --max-tasks 100 ^
  --eval-milestones 10,30,50,70,100 ^
  --arms arm_derpp arm_l2p arm_olora ^
  --seed 42 --epochs 25 --batch-size 128 --n-eval 5 --resume ^
  --out-json C:\Users\sonsj\dtra\outputs\libero\benchmark_dtra_libero_v31_ext_cl_k100_seed42.json >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [Stage 1 Retry %TIME%] exit=%ERRORLEVEL% >> "%LOG%"
  timeout /t 5 /nobreak >nul
  goto stage1
)

echo [Stage 2] Extended n_eval=20 Evaluation (Zero Retrain) on Saved K=100 Checkpoints (Seeds 42, 100, 2026) >> "%LOG%"
:stage2
"%PY%" -u -m dtra.experiments.libero.benchmark_dtra_libero_v31_ext ^
  --mode extended_eval ^
  --suites libero_10,libero_90 --max-tasks 100 ^
  --n-eval 20 --eval-seeds 42,100,2026 ^
  --arms arm_dtra_elastic arm_frozen_mlp ^
  --ckpt-dir C:\Users\sonsj\dtra\outputs\libero --resume ^
  --out-json C:\Users\sonsj\dtra\outputs\libero\eval_libero100_extended_neval20.json >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [Stage 2 Retry %TIME%] exit=%ERRORLEVEL% >> "%LOG%"
  timeout /t 5 /nobreak >nul
  goto stage2
)

echo [Stage 3] Complete Seed 2026 Missing Arms (arm_dtra_rigid, arm_er_5pct) for Full N=3 >> "%LOG%"
if not exist C:\Users\sonsj\dtra\outputs\libero\benchmark_dtra_libero_10_libero_90_k100_seed2026_full4.json (
  copy /Y C:\Users\sonsj\dtra\outputs\libero\benchmark_dtra_libero_10_libero_90_k100_seed2026.json C:\Users\sonsj\dtra\outputs\libero\benchmark_dtra_libero_10_libero_90_k100_seed2026_full4.json >> "%LOG%" 2>&1
)
:stage3
"%PY%" -u -m dtra.experiments.libero.benchmark_dtra_libero ^
  --suites libero_10,libero_90 --max-tasks 100 ^
  --eval-milestones 10,30,50,70,100 ^
  --arms arm_frozen_mlp arm_dtra_elastic arm_dtra_rigid arm_er_5pct ^
  --seed 2026 --stream-tasks --epochs 25 --batch-size 128 --n-eval 5 --resume ^
  --out-json C:\Users\sonsj\dtra\outputs\libero\benchmark_dtra_libero_10_libero_90_k100_seed2026_full4.json >> "%LOG%" 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [Stage 3 Retry %TIME%] exit=%ERRORLEVEL% >> "%LOG%"
  timeout /t 5 /nobreak >nul
  goto stage3
)

echo [ALL STAGES COMPLETED %DATE% %TIME%] >> "%LOG%"
endlocal
