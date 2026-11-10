# 합성 미세구조 경도 예측 — 재현 패키지 (v24a 구성, 대회 중 작성)

> **이 문서는 대회 중에 만든 재현 패키지 설명입니다.** `build_package.py`가 만드는 패키지와 `run_final.py`는 v24a 구성(Public 9.898)이고,
> 아래 소요 시간·파이프라인도 v24a 기준입니다. **최종 제출(v32b) 재현은 [루트 README 9장](../README.md#9-실행-방법)을 따르세요.**
> `run_final_v*.py`는 구성이 바뀔 때마다 만든 드라이버로, 파일 이름의 버전이 [제출 기록](../docs/submission_log.csv)의 제출 버전입니다.

## 1. 환경
| 항목 | 값 |
|---|---|
| OS | Windows 11 (Linux에서도 동작, 경로는 모두 상대경로) |
| Python | 3.11.9 |
| GPU | NVIDIA RTX 5060 Ti 16GB (CUDA 12.8 빌드 PyTorch 필요) |
| 라이브러리 | `requirements.txt` (torch/torchvision은 `pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0`) |

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux: .venv/bin/python
```

## 2. 데이터 배치
```
data/train.csv  data/sample_submission.csv  data/train/*.png  data/test/*.png
```

## 3. 실행
```bash
python run_final.py --mode infer   # 저장된 디노이저 가중치로 추론 (약 3시간 15분) → submission/submission.csv
python run_final.py --mode train   # 디노이저까지 처음부터 학습 (추가 약 35분)
```
- 단계별 결과가 있으면 건너뜁니다(중단 후 재개 가능). `--force`로 전부 다시 계산합니다.
- DINOv2·TIPSv2·CNN 사전학습 가중치는 처음 실행 시 Hugging Face Hub(timm)에서 내려받습니다.
- 학습하는 모델은 자체 디노이저(U-Net)뿐입니다. 공개 디노이저 DRUNet은 가중치를 그대로 추론에만 씁니다. 경도 회귀는 모두 고정(frozen) 사전학습 특징 + Ridge입니다.

## 4. 파이프라인
| 단계 | 내용 | 스크립트 |
|---|---|---|
| 1 | 원본 특징 캐시 (이미지별 노이즈 σ 추정) | `src/raw_features.py`, `src/features.py` |
| 2 | 미세구조 디노이저: train 저노이즈(σ<7) 이미지에 합성 가우시안 노이즈를 더해 U-Net 학습 → 1차 출력을 정답으로 재학습(부트스트랩) → 전체 이미지 디노이즈 (D4 평균) | `src/train_denoiser.py`, `src/denoise_with.py` |
| 2c | 두 번째 디노이즈 경로: 공개 사전학습 DRUNet(gray)에 이미지별 추정 노이즈 σ를 넣어 디노이즈 (D4 평균). 결정립계를 지어내지 않는 충실한 복원 | `src/drunet_denoise.py`, `src/drunet_net.py` |
| 3 | 저품질 촬영 모드(σ≥10)만 조명 불균일 보정: 가우시안 저역(σ 32/48/64/96) 제거 (두 디노이즈 경로 모두) | `src/flatfield_cache.py` |
| 4 | 고정(frozen) 사전학습 모델의 특징맵 풀링 × 보정 4종. 풀링은 공통: 평균·표준편차·영역 불균일성(hetero3) + 상 조건부 풀링(토큰을 2상/기지 비율로 가중한 평균·표준편차) [+ 혼합 법칙 기여량(2상 분율 × 2상 평균)]. 풀링 통계가 이미지를 넣는 방향·격자 위치(보기)에 따라 흔들리므로 여러 보기의 통계를 평균한다. **4a** DINOv2-L 448 입력 9~21층, 4보기 / **4b** 같은 모델 16보기(D4 8방향 × 격자 이동 2) / **4e** 같은 16보기를 DRUNet 경로 이미지에서 / **4c** TIPSv2-So400m 448 입력 8~14층, 16보기 / **4d** ImageNet CNN 4종(DenseNet-121·흐림 풀링 DenseNet-121 denseblock2, ResNet-50 A1·A2 layer2)의 stride-8 특징맵, 512 입력, 8보기 | `src/probe_models.py` (`--pool phase2` / `--pool phase2m --tta 8 --shifts 2` / `--model cnn:<timm 이름>@2 --pool phase2m --tta 8`) |
| 5 | 모델·보정별 Ridge (5폴드 OOF로 검증, test는 전체 500장 재적합): hetero3 + 상 조건부 풀링(상대 분산 0.25) [+ 기여량(0.05)]. 벌점은 작은 고정값(`--alpha-rel 0.01`, 중심화한 특징의 행별 제곱합 평균 대비) — 구성원별로 LOO가 고르는 벌점은 블렌드에는 과하다. DINOv2 16보기는 U-Net 경로와 DRUNet 경로 특징을 한 Ridge에 결합. 고정 CNN은 4개 모델의 특징을 한 Ridge에 결합 | `src/multi_ridge.py` |
| 6 | σ48 보정 이미지에서 측정 특징: 상(phase) 분율(상대 어둡기 CDF·분위수·multi-Otsu), 수작업 특징(ph_/pore/gr/sp), σ7~10 가우시안 평활 이미지의 기지 대비 어둡기 CDF(결정립 규모 2상 분포) | `src/phase_features.py`, `src/features_from_cache.py`, `src/smooth_cdf_features.py` |
| 7 | σ48의 4보기 DINO 특징(hetero3 + 상 조건부 풀링) + 상분율(0.4) + 수작업(0.15) + 평활 CDF(0.2) 결합 Ridge (전체 재적합) | `src/multi_ridge.py` |
| 8 | 최종 = 0.1765 × DINO+측정 특징 + 0.8235 × [0.45 × DINOv2(4보기 0.275 : 두 경로 16보기 0.425) + 0.25 × TIPSv2 + 0.30 × 고정 CNN 4종 결합] → 예측 폭 ×1.03 (train 경도 평균 기준의 고정 배율, `--scale 1.03`) | `src/final_blend.py` |

## 5. 규칙 준수
- 외부 데이터 미사용: 학습에는 제공된 train 500장만 사용합니다. 디노이저도 train 이미지와 합성 노이즈만으로 학습합니다.
- test 데이터는 추론에만 사용합니다. pseudo-label, test 통계 기반 정규화·적응은 하지 않습니다.
  - 조명 보정 대상 판별(σ≥10)은 이미지별 고정 규칙입니다.
  - 특징 표준화는 train 통계로 합니다.
- 사전학습 모델: 모두 **대회 시작일(2026-09-29) 이전에 가중치가 공식 공개된 오픈소스**입니다 (timm 표기 라이선스 / Hugging Face 저장소 공개일).

| 모델 | 용도 | 라이선스 | 공개일 |
|---|---|---|---|
| `vit_large_patch14_dinov2.lvd142m` | 고정 특징 (4a·4b) | Apache-2.0 | 2023-05 |
| `vit_so400m_patch14_reg1_tipsv2.webli` | 고정 특징 (4c) | Apache-2.0 | 2026-04-09 (Google 공식), 2026-05-27 (timm) |
| `densenet121.tv_in1k` | 고정 특징 (4d), ImageNet 사전학습 | Apache-2.0 (timm 표기; 원 가중치 torchvision BSD-3) | 2023-04 (timm 저장소; 원 torchvision 가중치는 2017) |
| `densenetblur121d.ra_in1k` | 고정 특징 (4d), ImageNet 사전학습 | Apache-2.0 | 2023-04 |
| `resnet50.a1_in1k`, `resnet50.a2_in1k` | 고정 특징 (4d), ImageNet 사전학습 | Apache-2.0 | 2023-04 |
| DRUNet gray (`drunet_gray.pth`, KAIR) | 두 번째 디노이즈 경로 (2c), 추론만 | MIT | 2021-09 (KAIR v1.0 릴리스) |

## 6. 재현성 참고
- `infer` 모드는 저장된 디노이저 가중치를 사용합니다. Ridge와 특징 계산은 결정적입니다. GPU bf16 연산 차이로 예측이 소수점 첫째~둘째 자리 수준에서 달라질 수 있습니다 (자체 검증: 패키지 재실행 결과와 제출 파일의 차이 RMS 약 0.2).
- `train` 모드는 GPU 비결정성(cuDNN 알고리즘 선택, 무작위 크롭) 때문에 디노이저가 매번 조금씩 달라집니다.
- 폴드 분할은 `outputs/folds.csv`(경도 10분위 층화, seed 42)로 고정했습니다 (Ridge의 교차검증 점수 계산용; test 예측은 전체 500장 재적합).
