#!/usr/bin/env bash
# 디노이저 2차: (1) 부트스트랩 정답(1차 디노이즈 결과를 정답으로, 잔여 σ0 제거) 10k / (2) 원 정답 15k
# 각 디노이저 → DINOv2-L hetero3@448 L9-21 프로브 (1차 기준: 11.742, L11-17 11.697). GPU 단독.
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM|PSNR 입력 [0-9.]+ → D4"
probe(){ $PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_$1.npz 2>&1 | grep --line-buffered -E "$F"; }
$PY -W ignore src/train_denoiser.py --iters 10000 --target-src outputs/denoised/unet_cm7_am22_l1_i6000.npz 2>&1 | grep --line-buffered -E "$F|정답"
probe cm7_am22_l1_i10000_boot
$PY -W ignore src/train_denoiser.py --iters 15000 2>&1 | grep --line-buffered -E "$F|정답"
probe cm7_am22_l1_i15000
