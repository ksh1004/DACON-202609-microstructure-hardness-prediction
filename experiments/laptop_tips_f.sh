#!/usr/bin/env bash
# [노트북 10/19 새벽] TIPSv2-So400m F 경로 16보기 phase2m 층 6~14 재추출(USB 선별본에 없음) → J3 + TIPS 결합 Ridge (TIPS 비중 0.5, 층 8~14 / 6~14)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
T=vit_so400m_patch14_reg1_tipsv2.webli; P2=outputs/probe2
one() { echo "--group emb:$1:$3:$2:$4:11:0,1,2,3,4,5 --group emb:$1:$(awk "BEGIN{print 0.25*$3}"):$2:$4:11:6,7,8,9 --group emb:$1:$(awk "BEGIN{print 0.05*$3}"):$2:$4:11:10"; }
for s in 48 32 64 96; do
  F=drunetft_ft1b_fflowpass${s}_low
  ET=$P2/${T}_448_L6-8-10-12-14_Pphase2m_T8_S${F}_sh2
  [ -f $ET/emb.npz ] || $PY -W ignore src/probe_models.py --model $T --size 448 --layers 6 8 10 12 14 --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/$F.npz 2>&1 | grep -E "probe OOF|Error|OOM"
  ED=$P2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${F}_sh2; EA=$P2/depth-anything__Depth-Anything-V2-Large-hf_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${F}_sh2; EN=$P2/da_neck_$F
  J3="$(one $ED 0,1,2,3,4 1 1024) $(one $EA 0,1,2,3,4 1 1024) $(one $EN 0,1,2,3 0.3 256)"
  for spec in "1,2,3,4:t814" "0,1,2,3,4:t614"; do LI=${spec%%:*}; NM=${spec##*:}
    n=joint3${NM}_ff${s}_a0.01; [ -f outputs/emb_reg/$n/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name $n $J3 $(one $ET $LI 0.5 1152) | tail -1
  done
  echo "[σ$s 완료 $(date +%H:%M)]"
done
echo ALLDONE
