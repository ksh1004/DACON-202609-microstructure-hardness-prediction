#!/usr/bin/env bash
# 실제 2모드 열화(저품질 = 대비 0.65~0.95 + 블러 0.6~1.4px + 노이즈 σ9~22)로 디노이저 학습 (부트스트랩 가중치·정답에서 시작) → DINO 프로브
# 근거: 파워스펙트럼상 σ≥10 이미지는 블러 σb≈1.0px·대비≈0.77배 (σ<10은 블러 없음). 비교: 부트스트랩 L11-17 11.517, σ≥10 13.42
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM|정답|PSNR 입력 [0-9.]+ → D4|it[0-9]+000 "
$PY -W ignore src/train_denoiser.py --iters 10000 --realistic --target-src outputs/denoised/unet_cm7_am22_l1_i6000.npz --init weights/denoiser_cm7_am22_l1_i10000_boot.pt --tag real_boot_i10000 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_real_boot_i10000.npz 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"
