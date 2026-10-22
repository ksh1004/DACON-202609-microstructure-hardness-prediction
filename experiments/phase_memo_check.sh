#!/usr/bin/env bash
# 위상 특징 암기 효과 점검: σ<7 정답 절반만 쓴 디노이저(half_boot)로 같은 파이프라인(σ48 저품질 조명보정 → DINO-L het3@448 + 위상 특징)
# → 정답에서 제외된 절반에서 boot(봄) vs half(안 봄) 오차 비교
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
$PY src/flatfield_cache.py --src outputs/denoised/unet_half_boot.npz --mode lowpass --sigma 48 --only-low | grep "^\["
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_half_boot_fflowpass48_low.npz 2>&1 | grep -E "^\[|Error|Traceback|OOM"
$PY -W ignore src/phase_features.py --src outputs/denoised/unet_half_boot_fflowpass48_low.npz --workers 6 | tail -1
S=unet_half_boot_fflowpass48_low
$PY src/block_ridge.py --src outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_S$S --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dnpf_half_ff48_L11-17 --extra 0.4 outputs/features_cache/phase_$S::pf_ | grep "^\["
