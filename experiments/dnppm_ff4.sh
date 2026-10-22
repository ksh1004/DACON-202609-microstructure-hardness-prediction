#!/usr/bin/env bash
# DINO hetero3 + 상 조건부 풀링 + 혼합 법칙 기여량(phase2m) — 조명보정 σ 4종
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8
for s in 32 48 64 96; do
  S=unet_cm7_am22_l1_i10000_boot_fflowpass${s}_low
  E=outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2m_S$S
  [ -f $E/emb.npz ] || $PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool phase2m --bs 8 --src outputs/denoised/$S.npz 2>&1 | grep -E "probe OOF|Error"
  for g in 0.05 0.15; do
    $PY -W ignore src/multi_ridge.py --name dnppm${g}_ff$s --group "emb:$E:1:1,2,3,4:1024:11:0,1,2,3,4,5" --group "emb:$E:0.25:1,2,3,4:1024:11:6,7,8,9" --group "emb:$E:$g:1,2,3,4:1024:11:10" | tail -1
  done
done
echo DONE
