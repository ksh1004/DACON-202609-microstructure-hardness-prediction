#!/usr/bin/env bash
# 디노이저 5차 (v11b 부트스트랩과 같은 정답 = 1차 디노이저 출력, σ<7): (a) non-blind σ 입력 (b) 256 전체 이미지 학습 → 각 DINO-L het3@448 프로브
# 비교는 정답으로 안 쓴 이미지(σ≥7) 기준. 부트스트랩 기준: L9-21 11.593
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM|정답|PSNR 입력 [0-9.]+ → D4"
T=outputs/denoised/unet_cm7_am22_l1_i6000.npz
probe(){ $PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_$1.npz 2>&1 | grep --line-buffered -E "$F"; }
$PY -W ignore src/train_denoiser.py --iters 10000 --target-src $T --nonblind 2>&1 | grep --line-buffered -E "$F"
probe cm7_am22_l1_i10000_boot_nb
$PY -W ignore src/train_denoiser.py --iters 10000 --target-src $T --crop 256 --batch 8 2>&1 | grep --line-buffered -E "$F"
probe cm7_am22_l1_i10000_boot_c256
