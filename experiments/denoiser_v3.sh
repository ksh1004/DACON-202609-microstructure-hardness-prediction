#!/usr/bin/env bash
# 디노이저 3차: 부트스트랩 반복(2차 부트스트랩 출력을 정답으로) → DINO 프로브, 이어서 부트스트랩 디노이즈 입력 고정 모델 재선별 (GPU 단독)
# 기준: 1차 11.742 / 부트스트랩 11.593 (L9-21), L11-17 11.697 / 11.517
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM|PSNR 입력 [0-9.]+ → D4"
$PY -W ignore src/train_denoiser.py --iters 10000 --target-src outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz --tag boot2_i10000 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_boot2_i10000.npz 2>&1 | grep --line-buffered -E "$F"
bash experiments/screen_denoised.sh outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz
