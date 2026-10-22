# 구조 특징과 CNN 개선 실험 결과

작성일: 2026-10-04. 모든 성능 비교는 제공된 train 500장으로 수행했다.
2026-10-05 사용자 확인: `sub_v4_big512.csv`의 Public 점수는
12.3168181193이다. 이 보고서의 OOF 수치와 Public 점수는 별도로 취급한다.

## 결론

- 새로운 구조 특징은 특징 기반 Ridge의 반복 분할 평균 RMSE를 13.5982에서
  13.3278로 개선했다. 세 분할 모두 개선되지만 고노이즈 구간의 개선은 작다.
- 새 CNN 입력/헤드는 ResNet-18 대조군 대비 일부 개선되지만 기존 512 모델
  앙상블을 개선하지 못했다. 이번 설정은 최종 앙상블에 채택하지 않는다.
- 기존 v4 앙상블의 Ridge만 새 특징 Ridge로 교체하면 OOF 12.771755에서
  12.769344가 된다. 차이가 0.0024에 불과해 실질적인 개선으로 판단하지 않는다.
- 검토용 `submissions/sub_structure_v1_fused.csv`를 생성했다. 온라인 제출은
  수행하지 않았으며 Public/Private 성능은 알 수 없다. 9점대 목표는 미달성이다.

## 1. 규정과 데이터 사용

공식 규정: https://daker.ai/public/hackathons/synthetic-microstructure-property-prediction-ai

- `eda/run_eda.py::analyze_shift`에서 train/test를 섞어 도메인 분류기를 학습하던
  절차를 제거했다. StandardScaler와 PCA도 train에만 적합한다. 남아 있는 KS와
  test 투영은 기술 통계이며 이번 모델 선택에는 사용하지 않았다.
- 기존 보고서의 adversarial AUC는 과거 절차의 산출물이다. 새 실험의 근거로
  재사용하지 않았다. 과거 기록 자체는 지우지 않았다.
- 구조 특징의 필터/분할 기준은 고정된 이미지별 연산이다. test 특징 캐시는
  추론 입력 준비용으로만 생성했고, 회귀 계수, 정규화, 하이퍼파라미터, 앙상블
  가중치를 test로 학습하거나 조정하지 않았다.
- CNN 실험은 `resnet18.a1_in1k`의 공개 가중치를 사용했다. 모델 카드에
  Apache-2.0으로 표시되어 있다: https://huggingface.co/timm/resnet18.a1_in1k
- 외부 이미지/라벨, 원격 예측 API, test pseudo-label, test-time adaptation은
  사용하지 않았다.

## 2. 특징 실험

`src/structure_features.py`에 NLM 노이즈 제거, h-minima watershed, 결정립
크기 분포, 개수/면적 기준 상 비율, 경계 밀도, variogram, 다중 스케일 정렬도
279개 특징을 구현했다. 원본 특징 103개와 합치면 382개다.

각 outer split은 경도 10분위 층화 5-fold다. seed 42는 기존 folds.csv를 유지하고
142/242는 새 분할을 생성했다. Ridge alpha는 outer train 내부 4-fold에서 고르며,
표준화도 각 inner train에서만 적합한다. 비교 대상 모두 같은 평가법을 사용했다.
따라서 이번 baseline 13.4858은 이전 RidgeCV 구현의 13.5219와 약간 다르다.

| 특징 구성 | split 42 | split 142 | split 242 | 평균 |
|---|---:|---:|---:|---:|
| 기존 103개 | 13.4858 | 13.6756 | 13.6332 | 13.5982 |
| 새 구조 279개 | 13.3049 | 13.5878 | 13.3478 | 13.4135 |
| 기존 + 새 결정립 특징 | 13.4406 | 13.6361 | 13.6976 | 13.5915 |
| 기존 + 새 특징 전체 | **13.2551** | **13.4268** | **13.3016** | **13.3278** |

split 42에서 기존→전체 결합의 저노이즈 RMSE는 12.0134→11.6016,
고노이즈는 15.3018→15.2619다. 결정립 특징만 추가한 경우의 개선은 일관되지
않아, 분할 정확도가 충분히 해결되었다고 결론 내릴 수 없다.

추가로 RBF-SVR를 outer train 내부에서 C/gamma를 탐색하여 비교했다.
전체 결합 특징의 세 분할 RMSE는 13.7264/13.8855/13.7139로 Ridge보다 나빴다.

원본 결과: `outputs/structure_v1/evaluation/results.json`,
`outputs/structure_v1/evaluation_svr/results.json`.

## 3. CNN 통제 실험

`src/train_structure_cnn.py`: ResNet-18, 256 입력, 30 epochs, batch 32,
AdamW lr 1e-3, weight decay 0.05, 1 epoch warmup + cosine,
MSE, seed 42, 기존 5-fold, D4 TTA. 최종 epoch를 사용하고 검증 점수로
checkpoint를 골라내지 않았다. 학습 손실 >0.8인 경우만 재시도한다.

| 구성 | OOF RMSE | 저노이즈 | 고노이즈 | 5-fold 시간(분) |
|---|---:|---:|---:|---:|
| raw: 원본 1채널 대조군 | 13.6006 | 12.3472 | 15.1743 | 1.61 |
| views: 원본/NLM/경계 3채널 | 13.4944 | 12.0130 | 15.3203 | 1.59 |
| stats: 4개 stage 평균/표준편차 + MLP | 13.5137 | 12.1844 | 15.1719 | 1.76 |
| fine_stats: stats + 첫 conv stride 1 | 13.6546 | 12.4899 | 15.1276 | 5.25 |

views의 기하 변환은 채널에 공통으로 적용하고, 밝기/대비는 raw/NLM에,
추가 노이즈는 raw에 적용했다. stats는 통계 헤드와 dropout을 함께 바꾼
아키텍처 실험이므로 효과를 특정 한 연산에만 귀속시키지 않는다.
실행 시간은 하드웨어/다른 작업의 부하에 따라 달라진다.

기존 v4에 새 CNN을 10%/25%/50% 섞는 사전 지정 비교에서 모두 악화됐다.
예를 들어 25% 혼합 OOF는 views 12.8658, stats 12.8598,
fine_stats 12.8968이었다. 대형 모델 확장/추가 시드 반복 대상으로 채택하지 않았다.

이는 이 설정들이 효과가 작다는 결과이며, 모든 NLM/통계 헤드 방식이나
결정립 정보 자체가 무효하다는 결론은 아니다.

## 4. 제출 후보와 검증

`src/build_structure_submission.py`로 각 25%의 고정 가중치를 사용했다.

- ConvNeXt-T 512 (`s512_convnext_tiny_s42`)
- EfficientNet-B2 512 (`s512_effnet_b2_s42`)
- ResNet-50 512 (`s512_resnet50_s42`)
- 전체 구조 특징 Ridge, split 42의 5개 fold 모델 평균

OOF 12.7693435275. 1000행, `ID,hardness`, sample_submission ID 순서,
유한한 예측값을 확인했다. 파일 생성과 온라인 제출은 별개이며, 생성만 완료했다.
출처 파일 해시는 `outputs/structure_v1/submission_manifest.json`에 저장했다.

`tools/verify_structure.py`는 다섯 회귀 모델의 scaler가 각각의 학습 fold에만
맞춰졌는지, checkpoint로 OOF가 재현되는지, 캐시 ID/형태, 제출 형식/해시를
검증한다. 신경망 4종의 forward/backward 유한성도 확인했다.
상수 영상 검사를 통해 noise 추정의 NaN 예외 처리를 추가했다.

BF16 CNN checkpoint 재추론은 같은 cuDNN benchmark 설정에서 raw/stats의
fold 0을 확인했다. 최대 예측 차이는 각각 0.00426/0.00636 HV,
fold RMSE 차이는 0.000010/0.000055였다. GPU 연산의 미세한 수치 차이가
남으며 비트 단위 동일성을 주장하지 않는다. 새 CNN은 최종 후보에 포함하지 않았다.

## 5. 해석과 다음 실험의 기준

반복 분할은 같은 500장을 사용하므로 독립적인 외부 검증은 아니다.
기존 CNN은 이전 실험에서 선택됐고 앙상블도 OOF로 비교했으므로 모델 선택
편향이 남아 있다. 특히 0.0024의 개선을 새로운 Public 최고점으로 해석하면 안 된다.

이번 실험은 저노이즈에서 특징 측정 개선을 확인했으나, 고노이즈 오차와
기존 모델들이 공유하는 잔차를 충분히 줄이지 못했다. 같은 설정의 모델 수를
더 늘리기보다 train의 실패 사례에서 측정 불가능한 구조/누락된 신호를
검증하는 별도의 실험 설계가 필요하다.

## 재현

프로젝트 루트에서 `experiments/structure_v1.ps1`을 실행한다. 기존 로컬
사전학습 가중치를 사용하며 신규 CNN checkpoint는 `--resume`으로 이어간다.
별도 모델 설정을 바꿀 때는 출력 실험 경로를 분리해야 한다.

검토용 제출 파일 재생성:

```powershell
.venv/Scripts/python src/structure_features.py --split test --workers 4
.venv/Scripts/python src/evaluate_structure.py --predict-test fused
.venv/Scripts/python src/build_structure_submission.py
.venv/Scripts/python tools/verify_structure.py
```
