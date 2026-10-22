#!/usr/bin/env bash
# DINO 입력 표현: 전역 대비 증폭(×2, ×3) / bicubic 확대 — 부트스트랩 디노이즈 + 저품질 σ48 조명보정 이미지
# 기준: DINO L11-17 11.337 / + 상분율(pf_ 0.4) 10.849
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
S=unet_cm7_am22_l1_i10000_boot_fflowpass48_low
T=vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_S$S
for opt in "--gain 2:_g2" "--gain 3:_g3" "--interp bicubic:_bicubic"; do
  args=${opt%%:*}; suf=${opt##*:}
  $PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/$S.npz $args 2>&1 | grep -E "^\[|Error|Traceback|OOM"
  $PY src/block_ridge.py --src outputs/probe2/$T$suf --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dn_ff48${suf}_L11-17 | grep "^\["
  $PY src/block_ridge.py --src outputs/probe2/$T$suf --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dnpf_ff48${suf}_L11-17 --extra 0.4 outputs/features_cache/phase_$S::pf_ | grep "^\["
done
