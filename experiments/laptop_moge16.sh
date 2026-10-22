#!/usr/bin/env bash
# [노트북 10/19 새벽] MoGe 백본(MIT) F 경로 16보기 phase2m 층 9~21, 조명보정 σ 4종 → 결합 Ridge(λ 0.01), 층 9~17
# 사용: bash experiments/laptop_moge16.sh moge:Ruicheng/moge-2-vitl moge2
M=${1:-moge:Ruicheng/moge-2-vitl}; NM=${2:-moge2}; MT=$(echo "$M" | sed 's/:/_/; s#/#__#')
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
L="9 11 13 15 17 19 21"
one() { echo "--group emb:$1:1:$2:1024:11:0,1,2,3,4,5 --group emb:$1:0.25:$2:1024:11:6,7,8,9 --group emb:$1:0.05:$2:1024:11:10"; }
for s in 48 32 64 96; do
  F=drunetft_ft1b_fflowpass${s}_low
  E=outputs/probe2/${MT}_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${F}_sh2
  [ -f $E/emb.npz ] || $PY -W ignore src/probe_models.py --model $M --size 448 --layers $L --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/$F.npz 2>&1 | grep -E "probe OOF|Error|OOM"
  n=${NM}16f_ft1b_L01234_ff${s}_a0.01; [ -f outputs/emb_reg/$n/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name $n $(one $E 0,1,2,3,4) | tail -1
  echo "[σ$s 완료 $(date +%H:%M)]"
done
echo ALLDONE
