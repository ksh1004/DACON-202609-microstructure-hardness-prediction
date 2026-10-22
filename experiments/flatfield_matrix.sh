#!/usr/bin/env bash
# 기지상 기반 조명 추정 보정 (그레이 클로징 창 W → 가우시안 σ 평활) — 저품질만 / 전체. 기준: 저역 σ48 저품질만 = 11.337
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
S=outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz
run(){ v=$1; shift; $PY src/flatfield_cache.py --src $S "$@" | grep "^\["
  $PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_cm7_am22_l1_i10000_boot_$v.npz 2>&1 | grep -E "^\[|Error|Traceback"
  $PY src/block_ridge.py --src outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_Sunet_cm7_am22_l1_i10000_boot_$v --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dn2b_${v}_L11-17; }
run ffmatrix24w31_low --mode matrix --win 31 --sigma 24 --only-low
run ffmatrix24w31 --mode matrix --win 31 --sigma 24
run ffmatrix32w21_low --mode matrix --win 21 --sigma 32 --only-low
run ffmatrix16w41_low --mode matrix --win 41 --sigma 16 --only-low
