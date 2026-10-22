#!/usr/bin/env bash
# 디노이저 4차: 정답 범위 확대(σ<10, 부트스트랩 정답) + 폭 64 + 12k → DINO-L het3@448 L9-21 프로브 (기준: 부트스트랩 11.593 / L11-17 11.517)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM|정답|PSNR 입력 [0-9.]+ → D4"
$PY -W ignore src/train_denoiser.py --iters 12000 --clean-max 10 --width 64 --target-src outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz --tag boot_cm10_w64_i12000 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_boot_cm10_w64_i12000.npz 2>&1 | grep --line-buffered -E "$F"
