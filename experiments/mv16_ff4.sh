#!/usr/bin/env bash
# 16보기(D4 8방향 × 격자 이동 2) 특징 평균으로 DINO hetero3 + 상 조건부 풀링 + 혼합 기여량(phase2m) 재추출 — 조명보정 σ 4종
# 근거(2026-10-13): 풀링 특징이 보기에 따라 크게 흔들림(단일 보기 11.80 / 4보기 평균 11.02±0.13 / 8보기 10.90 / 16~32보기 10.75~10.79)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8
M=vit_large_patch14_dinov2.lvd142m
for s in 48 32 64 96; do
  S=unet_cm7_am22_l1_i10000_boot_fflowpass${s}_low
  E=outputs/probe2/${M}_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${S}_sh2
  [ -f $E/emb.npz ] || $PY -W ignore src/probe_models.py --model $M --size 448 --layers 9 11 13 15 17 19 21 --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/$S.npz 2>&1 | grep -E "probe OOF|Error"
  $PY -W ignore src/multi_ridge.py --name mv16_dnpp0.25_ff$s --group "emb:$E:1:1,2,3,4:1024:11:0,1,2,3,4,5" --group "emb:$E:0.25:1,2,3,4:1024:11:6,7,8,9" | tail -1
  $PY -W ignore src/multi_ridge.py --name mv16_dnppm0.05_ff$s --group "emb:$E:1:1,2,3,4:1024:11:0,1,2,3,4,5" --group "emb:$E:0.25:1,2,3,4:1024:11:6,7,8,9" --group "emb:$E:0.05:1,2,3,4:1024:11:10" | tail -1
done
echo DONE
