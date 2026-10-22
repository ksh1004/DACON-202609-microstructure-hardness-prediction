#!/usr/bin/env bash
# 정답 확대 디노이저: σ<10 이미지(291장)를 부트스트랩 정답으로 (폭 48 그대로, 10k). 1차 디노이저 출력이 정답.
# 평가: 정답으로 안 쓴 σ≥10 이미지(209장)에서만 boot(σ<7 정답)와 비교 — DINO 단독 / DINO+상분율
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM|정답|PSNR 입력 [0-9.]+ → D4"
T=boot_cm10_i10000
$PY -W ignore src/train_denoiser.py --iters 10000 --clean-max 10 --target-src outputs/denoised/unet_cm7_am22_l1_i6000.npz --tag $T 2>&1 | grep --line-buffered -E "$F"
$PY src/flatfield_cache.py --src outputs/denoised/unet_$T.npz --mode lowpass --sigma 48 --only-low | grep "^\["
S=unet_${T}_fflowpass48_low
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/$S.npz 2>&1 | grep -E "^\[|Error|Traceback|OOM"
$PY -W ignore src/phase_features.py --src outputs/denoised/$S.npz --workers 6 | tail -1
P=outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_S$S
$PY src/block_ridge.py --src $P --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dn_${T}_ff48_L11-17 | grep "^\["
$PY src/block_ridge.py --src $P --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dnpf_${T}_ff48_L11-17 --extra 0.4 outputs/features_cache/phase_$S::pf_ | grep "^\["
