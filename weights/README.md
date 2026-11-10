# 가중치

가중치 파일은 저장소에 넣지 않았고(GitHub 파일 크기 제한 100MB), 아래 위치에서 내려받습니다.

| 파일 (이 폴더 기준 위치) | 내용 | 받는 곳 | 크기 |
|---|---|---|---|
| `pretrained_denoiser/drunet_gray.pth` | 공개 사전학습 DRUNet (gray), MIT | [KAIR v1.0 릴리스](https://github.com/cszn/KAIR/releases/tag/v1.0) | 약 125MB |
| `denoiser_cm7_am22_l1_i10000_boot.pt` | 자체 U-Net 디노이저 (부트스트랩, U 경로) | [이 저장소 Releases `weights-v1`](https://github.com/ksh1004/DACON-202609-microstructure-hardness-prediction/releases/tag/weights-v1) | 41MB |
| `denoiser_drt10b_i10000.pt` | 자체 U-Net 증류 디노이저 (X 경로) | 위와 같음 | 41MB |
| `pretrained_denoiser/drunetft_ft1b.pt` | DRUNet 미세조정 2차 (F 경로) | 위와 같음 | 125MB |

저장소 루트에서 실행합니다 (Windows는 Git Bash, `curl`은 Windows 10 이상에 기본 포함).

```bash
REL=https://github.com/ksh1004/DACON-202609-microstructure-hardness-prediction/releases/download/weights-v1
mkdir -p weights/pretrained_denoiser
curl -L -o weights/denoiser_cm7_am22_l1_i10000_boot.pt    $REL/denoiser_cm7_am22_l1_i10000_boot.pt
curl -L -o weights/denoiser_drt10b_i10000.pt              $REL/denoiser_drt10b_i10000.pt
curl -L -o weights/pretrained_denoiser/drunetft_ft1b.pt   $REL/drunetft_ft1b.pt
curl -L -o weights/pretrained_denoiser/drunet_gray.pth    https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth
```

SHA256 (내려받은 파일 확인용):

```
ce70055a9d8ad7b67a8c633964b5cd269c2edf1034fe84039555fe635cf42fee  denoiser_cm7_am22_l1_i10000_boot.pt
2212670dc23c7bf5b3973e3986bbf0bfd1f7a72e1078d7fb99d6882930d1c007  denoiser_drt10b_i10000.pt
8c809537786a97d813719a9f1c2a892a09525f44e4a9b6b35b59eb0fb1a6c03a  drunetft_ft1b.pt
```

직접 학습하려면 `run_final.py --mode train`을 쓰면 됩니다([README 9-4](../README.md#9-4-실행-순서)). 자체 U-Net은 `src/train_denoiser.py`, DRUNet 미세조정은 `src/drunet_finetune.py`로 학습합니다.
디노이저 학습에는 무작위성이 있어, 다시 학습한 가중치는 공개한 파일과 같지 않습니다.

**라이선스** — 자체 U-Net 가중치는 이 저장소와 같은 MIT입니다. `drunetft_ft1b.pt`는 KAIR의 DRUNet(MIT, Copyright (c) 2019 Kai Zhang)을 미세조정한 것이라 원본 고지를 함께 따릅니다([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).
특징 추출용 사전학습 모델(DINOv2, Depth-Anything-V2, TIPSv2, ImageNet CNN)은 처음 실행할 때 Hugging Face Hub에서 자동으로 내려받습니다.
