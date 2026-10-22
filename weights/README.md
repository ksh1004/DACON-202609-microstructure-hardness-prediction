# 가중치

GitHub 파일 크기 제한(100MB)과 각 가중치의 라이선스 때문에 저장소에는 넣지 않았습니다.

| 파일 | 내용 | 얻는 방법 |
|---|---|---|
| `pretrained_denoiser/drunet_gray.pth` | 공개 사전학습 DRUNet (gray), MIT | [KAIR v1.0 릴리스](https://github.com/cszn/KAIR/releases/tag/v1.0)에서 내려받기 |
| `denoiser_cm7_am22_l1_i10000_boot.pt` | 자체 U-Net 디노이저 (부트스트랩, U 경로) | `src/train_denoiser.py` (train 이미지 + 합성 가우시안 노이즈로 학습) |
| `denoiser_drt10b_i10000.pt` | 자체 U-Net 증류 디노이저 (DRT 경로) | `src/train_denoiser.py --target-src ...` 2단계 (`reproduce/run_final_v29b.py`의 2d 단계) |
| `pretrained_denoiser/drunetft_ft1b.pt` | DRUNet 미세조정 2차 (F 경로) | `src/drunet_finetune.py` 2단계 (`reproduce/run_final_v29b.py`의 2e 단계) |

특징 추출용 사전학습 모델(DINOv2, Depth-Anything-V2, TIPSv2, ImageNet CNN)은 처음 실행할 때 Hugging Face Hub에서 자동으로 내려받습니다.
