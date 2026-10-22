#!/usr/bin/env bash
# [노트북 10/17] 고정 CNN 4종 구성원에 미세조정 DRUNet 경로(F, ft1b) 추가: 8보기 phase2m 특징 (조명보정 σ 4종) → 결합 Ridge(λ 0.01)
# 블렌드에서는 기존 U·DRT 경로 예측과 평균(TIPSv2의 Tm3와 같은 방식). 큐 대기: laptop_ft_robust 종료 후
TAG=${1:-ft1b}; AR=0.01; PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
until grep -q "^ALLDONE" logs/laptop_ft_robust.log 2>/dev/null; do sleep 30; done
oneC() { echo "--group emb:$1:1:0:512:11:0,1,2,3,4,5 --group emb:$1:0.25:0:512:11:6,7,8,9 --group emb:$1:0.05:0:512:11:10"; }
for s in 48 32 64 96; do
  X=drunetft_${TAG}_fflowpass${s}_low
  [ -f outputs/denoised/$X.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/drunetft_${TAG}.npz --mode lowpass --sigma $s --only-low | tail -1
  CS=""
  for M in densenet121.tv_in1k@2 resnet50.a1_in1k@2 resnet50.a2_in1k@2 densenetblur121d.ra_in1k@2; do
    EC=outputs/probe2/cnn_${M/@/_s}_512_Pphase2m_T8_S$X
    [ -f $EC/emb.npz ] || $PY -W ignore src/probe_models.py --model cnn:$M --size 512 --pool phase2m --tta 8 --bs 16 --src outputs/denoised/$X.npz 2>&1 | grep -E "probe OOF|Error|OOM"
    CS="$CS $(oneC $EC)"
  done
  [ -f outputs/emb_reg/cnnfj4_${TAG}_ff${s}_a$AR/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name cnnfj4_${TAG}_ff${s}_a$AR $CS | tail -1
done
echo ALLDONE
