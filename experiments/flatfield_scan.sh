#!/usr/bin/env bash
# 조명 불균일 보정 스캔 (부트스트랩 디노이즈 입력, DINO-L het3@448 L9-21 → block_ridge L11-17). 기준: 저품질만 σ48 = 11.337
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
S=outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz
run(){ v=$1; shift; $PY src/flatfield_cache.py --src $S "$@" | grep "^\["
  $PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_cm7_am22_l1_i10000_boot_$v.npz 2>&1 | grep -E "^\[|Error|Traceback"
  $PY src/block_ridge.py --src outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_Sunet_cm7_am22_l1_i10000_boot_$v --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dn2b_${v}_L11-17; }
run fflowpass32_low --mode lowpass --sigma 32 --only-low
run fflowpass64_low --mode lowpass --sigma 64 --only-low
run fflowpass96_low --mode lowpass --sigma 96 --only-low
run fflowpass48 --mode lowpass --sigma 48
run fflowpass48_low_pall --mode lowpass --sigma 48 --only-low --plane-all
