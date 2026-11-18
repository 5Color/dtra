# DTRA: Dynamic Topology Ring Architecture

[English](README.md) | **한국어**

[![프리프린트 (DTRA v3.0)](https://img.shields.io/badge/Preprint-DTRA__v3.0__Preprint.pdf-b31b1b.svg)](dtra/paper_v2/DTRA_v3.0_Preprint.pdf)
[![arXiv 원고](https://img.shields.io/badge/arXiv%20Clean-dtra__preprint.pdf-9e1c20.svg)](dtra/paper_v2/dtra_preprint.pdf)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24218483.svg)](https://doi.org/10.5281/zenodo.24218483)
[![MIT 라이선스](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**초구면 라우팅과 공유 운동 기저를 이용한 재생 없는 지속 시각운동 제어.** DTRA v3.0은 3개 학습 시드에서 LIBERO-100 최종 성공률 `87.00 ± 0.86%`를 보고합니다.

> **연구 세대와 범위:** 이 저장소는 Gen 1 Drawing-VLA/ABK-VLA와 Gen 2 DTRA v3.0을 함께 보관합니다. 본 README의 DTRA 실험 수치는 LIBERO-100, LIBERO-Goal 파일럿, 8태스크 Meta-World 범위입니다. CALVIN v4/v5, RoboCasa, Organ은 후속 v4.0 연구 트랙이며 v3.0 결과에 포함하지 않습니다. 연구 이력은 [RESEARCH_LINEAGE.md](RESEARCH_LINEAGE.md), 버전 경계는 [DTRA 논문 개요](dtra/PAPER_20P_OUTLINE.md)를 참고하세요.

## 아키텍처 개요

Meta-World 8태스크에서 LIBERO의 순차 100태스크로 확장하면 라우팅 영역 침범과 태스크별 행동 헤드의 선형 증가가 문제가 됩니다. DTRA v3.0은 동결된 ResNet-18 DualCam 인코더, 고유수용성 입력, `LightweightTextEncoder`를 공통 인식 경로로 사용하고 네 요소를 결합합니다.

1. **라우팅 마진 보정:** 이전 태스크의 쿼리 기준점과 자기 유사도 하한을 보존하고, 이전 라우팅 영역을 침범하는 키 업데이트를 투영해 제거합니다.
2. **2단계 Top-M 공통 성분 제거(CMC):** 전체 키 코사인 스캔으로 후보 5개를 고른 뒤, 후보들이 공유하는 언어·장면 성분을 제거하고 세부 카메라 특징과 고유수용성 잔차로 재순위화합니다. 동일 지시어 태스크의 LIBERO-100 라우팅 정확도는 `41.76%`에서 `99.68 ± 0.45%`로 보고되었습니다.
3. **공유 운동 기저(SharedBasisBank):** 길이 16의 행동 청크를 6개 제어점의 클램프드 `C²` 큐빅 B-스플라인으로 나타내고, 100개 태스크에서 27개 동결 기저를 공유합니다. 73% 감소는 기저 개수 기준이며 전체 모델 메모리 감소를 뜻하지 않습니다.
4. **두 시간척도의 Snap-Back:** 청크 사이 임시 변형 상태를 되돌리는 `chunk_snap_back()`과 에피소드 종료 시 `episode_snap_back()`을 구분합니다. `post_snap_norm = 0.0000`은 상태 초기화를 뜻하며 파라미터 보존을 독립적으로 입증하는 수치는 아닙니다.

### 입력 조건

여기서 **Task-ID-free**는 추론 때 정수형 태스크 인덱스를 정책에 직접 전달하지 않는다는 뜻입니다. 라우팅 입력에는 두 카메라 특징, 고유수용성 상태와 언어 임베딩이 포함됩니다. 언어 지시를 제공하는 평가는 순수 시각·고유수용성 관측만 사용하는 평가와 구분해야 합니다.

## 벤치마크 결과

### LIBERO-100 (K=100)

각 마일스톤에서 시드당 5회 롤아웃을 평가했습니다. 전체 4개 비교군은 2개 시드, `Frozen MLP`와 `DTRA Elastic` 핵심 비교는 3개 시드를 사용했습니다.

| 방법 | 최종 성공률 | 망각률 | BWT | 라우팅 정확도 | 동일 지시어 라우팅 | 기저 수 | 활성 추론 지연 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arm_frozen_mlp` (N=3) | 70.33 ± 1.18% | 2.90 ± 1.05% | +0.20 ± 2.29% | 81.94 ± 0.24% | 41.76 ± 2.09% | 100 | 6.46 ± 0.57 ms |
| `arm_er_5pct` (N=2) | 40.50 ± 0.30% | 27.07 ± 4.04% | −20.71 ± 2.93% | 1.00 ± 0.00% | 0.00 ± 0.00% | 1 | 5.34 ± 0.03 ms |
| `arm_dtra_rigid` (N=2) | 86.20 ± 0.00% | 2.93 ± 0.91% | +0.10 ± 0.91% | 99.86 ± 0.14% | 99.53 ± 0.47% | 27 | 9.36 ± 0.05 ms |
| **`arm_dtra_elastic` (N=3)** | **87.00 ± 0.86%** | **2.69 ± 0.67%** | **+0.20 ± 0.72%** | **99.90 ± 0.13%** | **99.68 ± 0.45%** | **27** | 9.55 ± 0.20 ms |

`arm_dtra_elastic`의 시드별 최종 성공률은 42번 시드 `87.80%`, 100번 시드 `87.40%`, 2026번 시드 `85.80%`입니다.

### Meta-World 및 LIBERO-Goal

| 벤치마크 | 방법 | 성공률 | 망각률 | 라우팅 | OOD 성공률 / VRAM |
| :--- | :--- | ---: | ---: | ---: | :--- |
| Meta-World 8태스크 (3개 시드) | Frozen MLP 멀티모달 | 85.6 ± 3.1% | 0.0 ± 0.0% | 100.0% | 심한 OOD 23.6 ± 3.9% |
| Meta-World 8태스크 (3개 시드) | **DTRA Full 멀티모달** | **72.2 ± 6.8%** | **0.0 ± 0.0%** | **100.0%** | **심한 OOD 51.4 ± 4.4%** |
| Meta-World 역순 (T7→T0, 시드 42) | DTRA Full 멀티모달 | 74.2% | 0.0% | 100.0% | OOD 64.2% |
| LIBERO-Goal 5태스크 파일럿 (시드 42) | DTRA Elastic | 92.0% | 5.0% | 100.0% | VRAM 3,696 MB |

### 평가 범위와 수치 해석

- 각 태스크는 시드별로 동일한 첫 5개 초기 상태를 사용해 5회 평가했습니다. 시드 간 표준편차는 더 넓은 초기 상태 분포에 대한 신뢰구간이 아닙니다.
- LIBERO-100의 성공률 기준 망각률은 `2.69 ± 0.67%`입니다. 파라미터 동결과 임시 상태 초기화는 별개이며, 망각률에서 라우팅 변화와 롤아웃 변동의 기여는 분리되지 않았습니다.
- 73% 감소는 K=100에서 기저 수를 100개에서 27개로 줄인 값입니다. 공통 동결 백본을 제외한 정책 파라미터 저장 공간은 DTRA `183.14 MB`, Frozen MLP `151.67 MB`였습니다. K=100 이후의 점근적 메모리 성장률이나 손익분기점은 입증하지 않았습니다.
- Meta-World에서 DTRA는 Frozen MLP보다 정상 조건 성공률이 낮고 심한 OOD 성공률이 높았습니다. 이는 관측된 성능 절충이며 인과적 설명이 입증된 것은 아닙니다.
- LIBERO의 `9.55 ± 0.20 ms`는 정책 추론 측정입니다. 전체 후보 검색은 `O(K)`이며, 선택 후보의 디코딩 코어가 상수 비용입니다. 이 수치를 종단 간 로봇 제어 지연으로 해석하면 안 됩니다.

## 저장소 구조

```text
v1_drawing_vla/          # Gen 1 Drawing-VLA 및 ABK-VLA 보관
dtra/                    # DTRA v3.0 핵심 코드와 실험
  core/                  # B-스플라인 디코더 및 링 동역학
  experiments/libero/    # LIBERO-Goal 및 LIBERO-100
  experiments/metaworld/ # Meta-World 평가와 절제 실험
  experiments/calvin_v4/, calvin_v5/, robocasa_v4/  # v4.0 후속 연구
  paper_v2/              # 프리프린트 PDF와 LaTeX 소스
outputs/                 # 고정 결과 JSON, 체크포인트와 그림
scripts/                 # 결과 집계, 그림 및 프리프린트 생성
tests/                   # 모듈 및 불변량 테스트
```

## 재현과 코드 공개 범위

공개 릴리스에는 프리프린트, LaTeX 소스, 그림과 고정 결과 JSON이 포함됩니다. 벤치마크 실행 코드와 재현 의존성은 공개 체크아웃에 아직 모두 포함되어 있지 않습니다. 아래 명령은 로컬 연구 체크아웃 기준이며 완전한 빈 환경 설치 절차가 아닙니다.

```bash
# 핵심 테스트
pip install -r requirements.txt
python -m unittest discover -s tests -v

# LIBERO-100 (libero conda 환경에서 실행)
python -m dtra.experiments.libero.benchmark_dtra_libero \
  --suites libero_10,libero_90 --max-tasks 100 \
  --stream-tasks --eval-milestones 10,30,50,70,100 \
  --epochs 25 --batch-size 128 --n-eval 5 --seed 42 \
  --out-json outputs/libero/benchmark_dtra_libero_10_libero_90_k100.json

# Meta-World 3개 시드 절제 실험
python -m dtra.experiments.metaworld.ablate_dtra_scientific \
  --seeds 42 100 2026 --epochs 80

# 결과 요약, 그림 및 프리프린트 다시 생성
python scripts/summarize_libero_multiseed.py
python scripts/generate_libero100_figures.py
python scripts/build_dtra_v3_paper.py
```

## 그림과 프리프린트

- [DTRA v3.0 프리프린트 PDF](dtra/paper_v2/DTRA_v3.0_Preprint.pdf) · [arXiv용 PDF](dtra/paper_v2/dtra_preprint.pdf)
- [아키텍처 그림](outputs/figures/fig6_dtra_v3_architecture_schematic.png) · [LIBERO-100 확장 그래프](outputs/figures/fig7_libero100_scaling_curves.png) · [기저 및 호출 경로](outputs/figures/fig8_libero100_basis_and_callstack.png)

## 인용

```bibtex
@misc{son2026dtra,
  author = {Son, Seungjoon},
  title  = {{DTRA}: Dynamic Topology Ring Architecture for Replay-Free Continual Visuomotor Control via Hyperspherical Isolation and Shared Motor Bases},
  year   = {2026},
  month  = nov,
  doi    = {10.5281/zenodo.24218483},
  url    = {https://doi.org/10.5281/zenodo.24218483}
}
```

[MIT License](LICENSE)
