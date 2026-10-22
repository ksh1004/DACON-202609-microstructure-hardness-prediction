#!/usr/bin/env bash
# [노트북 10/19 새벽] DA 깊이 디코더 특징 16보기(D4 8 × 격자 이동 2) — F 경로 4보정 → J3(16보기 디코더) 결합 Ridge
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
one() { echo "--group emb:$1:$3:$2:$4:11:0,1,2,3,4,5 --group emb:$1:$(awk "BEGIN{print 0.25*$3}"):$2:$4:11:6,7,8,9 --group emb:$1:$(awk "BEGIN{print 0.05*$3}"):$2:$4:11:10"; }
P2=outputs/probe2
for s in 48 32 64 96; do
  F=drunetft_ft1b_fflowpass${s}_low
  [ -f $P2/da_neck_${F}_sh2/emb.npz ] || $PY -W ignore analysis/da_neck_embed.py --shifts 2 --src outputs/denoised/$F.npz --bs 4 2>&1 | grep -E "DONE|rror|Traceback|OOM"
  ED=$P2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${F}_sh2; EA=$P2/depth-anything__Depth-Anything-V2-Large-hf_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${F}_sh2; EN=$P2/da_neck_${F}_sh2
  n=joint3n16_ff${s}_a0.01; [ -f outputs/emb_reg/$n/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name $n $(one $ED 0,1,2,3,4 1 1024) $(one $EA 0,1,2,3,4 1 1024) $(one $EN 0,1,2,3 0.3 256) | tail -1
  echo "[σ$s 완료 $(date +%H:%M)]"
done
echo ALLDONE
