#!/usr/bin/env bash
# DPN-92·DPN-98(ImageNet 사전학습, timm dpn92.mx_in1k / dpn98.mx_in1k, Apache-2.0 표기)의 stride-8·stride-16 두 단계 특징맵을 고정 특징으로:
# 512 입력, D4 8보기, phase2m 풀링 → 두 모델 × 두 단계를 한 Ridge에 결합 (고정 벌점). 조명보정 σ 4종.
# 근거(2026-10-14, train σ48): 단독 10.81~10.89 (고정 CNN 4종 결합 10.75~10.83과 동급), 블렌드에 10% 더하면 −0.034~−0.039 (폴드 5/5).
#   기존 고정 CNN 결합 Ridge 안에 넣으면 이득이 없고(포화), 별도 구성원으로 예측을 평균할 때만 이득.
AR=${1:-0.01}; PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
emb() { echo outputs/probe2/cnn_${1/@/_s}_512_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass${2}_low; }
one() { echo "--group emb:$1:1:0:$2:11:0,1,2,3,4,5 --group emb:$1:0.25:0:$2:11:6,7,8,9 --group emb:$1:0.05:0:$2:11:10"; }
for s in 48 32 64 96; do
  S=unet_cm7_am22_l1_i10000_boot_fflowpass${s}_low
  for M in dpn92.mx_in1k@2 dpn92.mx_in1k@3 dpn98.mx_in1k@2 dpn98.mx_in1k@3; do
    E=$(emb $M $s); [ -f $E/emb.npz ] || $PY -W ignore src/probe_models.py --model cnn:$M --size 512 --pool phase2m --tta 8 --bs 16 --src outputs/denoised/$S.npz 2>&1 | grep -E "probe OOF|Error"
  done
  $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name dpnj_ff${s}_a$AR $(one $(emb dpn92.mx_in1k@2 $s) 704) $(one $(emb dpn92.mx_in1k@3 $s) 1552) $(one $(emb dpn98.mx_in1k@2 $s) 768) $(one $(emb dpn98.mx_in1k@3 $s) 1728) | tail -1
done
echo DONE
