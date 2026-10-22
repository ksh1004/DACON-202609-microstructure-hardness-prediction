#!/usr/bin/env bash
# 암기 효과 진단: σ<7 정답 후보의 절반만 정답으로 1차(6k) → 부트스트랩(10k) 디노이저 학습 → DINO-L het3@448 프로브.
# 제외된 절반에서 '전체 정답 부트스트랩(v11b, 그 이미지를 봄)' vs '절반 부트스트랩(안 봄)' 오차 비교.
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM|정답|PSNR 입력 [0-9.]+ → D4"
$PY -W ignore src/train_denoiser.py --iters 6000 --target-frac 0.5 --tag half_v1 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/train_denoiser.py --iters 10000 --target-frac 0.5 --target-src outputs/denoised/unet_half_v1.npz --tag half_boot 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_half_boot.npz 2>&1 | grep --line-buffered -E "$F"
