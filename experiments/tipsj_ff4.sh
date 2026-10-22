#!/usr/bin/env bash
# TIPSv2: 기존 U-Net 경로 + DRT 경로 특징을 한 Ridge에 (예측 평균 Tm 대신), 조명보정 4종
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
T=vit_so400m_patch14_reg1_tipsv2.webli; TAG=drt10b_i10000
oneT() { echo "--group emb:$1:1:0,1,2,3:1152:11:0,1,2,3,4,5 --group emb:$1:0.25:0,1,2,3:1152:11:6,7,8,9 --group emb:$1:0.05:0,1,2,3:1152:11:10"; }
for s in 48 32 64 96; do
  EU=outputs/probe2/${T}_448_L8-10-12-14_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass${s}_low_sh2
  EX=outputs/probe2/${T}_448_L8-10-12-14_Pphase2m_T8_Sunet_${TAG}_fflowpass${s}_low_sh2
  [ -f outputs/emb_reg/tipsj_${TAG}_ff${s}_a0.01/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name tipsj_${TAG}_ff${s}_a0.01 $(oneT $EU) $(oneT $EX) 2>&1 | tail -1
done
echo ALLDONE
