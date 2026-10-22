#!/usr/bin/env bash
# [노트북 10/17] F 경로(ft1b) DINOv2-L 16보기 phase2m를 층 3~17(8개)로 다시 추출 (조명보정 σ 4종) → 결합 Ridge(λ 0.01)
# 근거(σ48·8보기): 층 11~17 단독 10.398 → 9~17 10.267 → 3~17 10.167, 3계열 블렌드 10.007 → 9.955
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
D=vit_large_patch14_dinov2.lvd142m; L="3 5 7 9 11 13 15 17"
one() { echo "--group emb:$1:1:$2:1024:11:0,1,2,3,4,5 --group emb:$1:0.25:$2:1024:11:6,7,8,9 --group emb:$1:0.05:$2:1024:11:10"; }
for s in 48 32 64 96; do
  F=drunetft_ft1b_fflowpass${s}_low
  EF=outputs/probe2/${D}_448_L3-5-7-9-11-13-15-17_Pphase2m_T8_S${F}_sh2
  [ -f $EF/emb.npz ] || $PY -W ignore src/probe_models.py --model $D --size 448 --layers $L --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/$F.npz 2>&1 | grep -E "probe OOF|Error|OOM"
  for spec in "0,1,2,3,4,5,6,7:L3to17" "2,3,4,5,6,7:L7to17"; do
    LI=${spec%%:*}; NM=${spec##*:}
    [ -f outputs/emb_reg/mv16f_ft1b_${NM}_ff${s}_a0.01/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name mv16f_ft1b_${NM}_ff${s}_a0.01 $(one $EF $LI) | tail -1
  done
  echo "[σ$s 완료 $(date +%H:%M)]"
done
echo ALLDONE
