#!/usr/bin/env bash
# 특징 일치 디노이저 미세조정(부트스트랩에서 시작, λ=1, 2000it) → DINO-L het3@448 L9-21 프로브. 비교: 부트스트랩 L11-17 11.517, σ≥10 13.42
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
$PY -W ignore src/finetune_denoiser_feat.py --init weights/denoiser_cm7_am22_l1_i10000_boot.pt --iters 2000 --lam 1 2>&1 | grep --line-buffered -vE "^\s*$|HF_TOKEN|Warning"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_featft_lam1_i2000.npz 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"
