#!/usr/bin/env bash
# NAFNet 디노이저 (ECCV 2022, 폭 32, 11.6M) — 부트스트랩 디노이저와 같은 정답(1차 출력, σ<7)·노이즈(0~22)·L1·10k
# 기준(부트스트랩 U-Net): DINO 11.337 / +pf 10.849 / +pf+ph+pgs 10.723
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM|정답|모델|PSNR 입력 [0-9.]+ → D4|it[0-9]*000 "
T=naf_boot_i10000
$PY -W ignore src/train_denoiser.py --arch nafnet --width 32 --iters 10000 --target-src outputs/denoised/unet_cm7_am22_l1_i6000.npz --tag $T 2>&1 | grep --line-buffered -E "$F"
$PY src/flatfield_cache.py --src outputs/denoised/unet_$T.npz --mode lowpass --sigma 48 --only-low | grep "^\["
S=unet_${T}_fflowpass48_low
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/$S.npz 2>&1 | grep -E "^\[|Error|Traceback|OOM"
$PY -W ignore src/phase_features.py --src outputs/denoised/$S.npz --workers 6 | tail -1
P=outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_S$S
$PY src/block_ridge.py --src $P --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dn_${T}_ff48_L11-17 --full | grep "^\["
$PY src/block_ridge.py --src $P --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dnpf_${T}_ff48_L11-17 --extra 0.4 outputs/features_cache/phase_$S::pf_ --full | grep "^\["
