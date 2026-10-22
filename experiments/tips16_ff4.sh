#!/usr/bin/env bash
# TIPSv2-So400m (Apache-2.0, 448 기본 해상도) 고정 특징: 16보기(D4 8방향 × 격자 이동 2) hetero3 + 상 조건부 풀링 + 혼합 기여량 — 조명보정 σ 4종
# 근거(2026-10-13, train 8보기): 층 8~14 단독 11.07 (DINOv2-L 10.90), 잔차상관 0.936, 예측 블렌드 L 0.7 : TIPS 0.3 → 10.80 (5폴드 중 4)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8
M=vit_so400m_patch14_reg1_tipsv2.webli
for s in 48 32 64 96; do
  S=unet_cm7_am22_l1_i10000_boot_fflowpass${s}_low
  E=outputs/probe2/${M}_448_L8-10-12-14_Pphase2m_T8_S${S}_sh2
  [ -f $E/emb.npz ] || $PY -W ignore src/probe_models.py --model $M --size 448 --layers 8 10 12 14 --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/$S.npz 2>&1 | grep -E "probe OOF|Error"
  $PY -W ignore src/multi_ridge.py --name tips16_dnppm0.05_ff$s --group "emb:$E:1:0,1,2,3:1152:11:0,1,2,3,4,5" --group "emb:$E:0.25:0,1,2,3:1152:11:6,7,8,9" --group "emb:$E:0.05:0,1,2,3:1152:11:10" | tail -1
done
echo DONE
