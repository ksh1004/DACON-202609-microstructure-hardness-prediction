#!/usr/bin/env bash
# 디노이저 4종(boot/boot2/c256/nb) 출력 픽셀 평균 → 저품질 σ48 조명보정 → DINO-L het3@448 + 상분율 특징 (v13a 설계)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
S=unet_avg4_fflowpass48_low
$PY src/flatfield_cache.py --src outputs/denoised/unet_avg4.npz --mode lowpass --sigma 48 --only-low | grep "^\["
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/$S.npz 2>&1 | grep -E "^\[|Error|Traceback|OOM"
$PY -W ignore src/phase_features.py --src outputs/denoised/$S.npz --workers 6 | tail -1
$PY src/block_ridge.py --src outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_S$S --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dn_avg4_ff48_L11-17 --full | grep "^\["
$PY src/block_ridge.py --src outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_S$S --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dnpf_avg4_ff48_L11-17 --extra 0.4 outputs/features_cache/phase_$S::pf_ --full | grep "^\["
