#!/usr/bin/env bash
# 고정 CNN 계열을 '모델별 Ridge 예측 평균' 대신 '여러 CNN의 특징을 한 Ridge에 결합'으로 (특징 추출은 cnnf_ff4.sh와 같음).
# 근거(2026-10-14, train σ48, analysis/cnn_joint.py·cnn_joint_check.py): 기존 3개 모델 예측 평균 11.001 → 결합 10.874,
#   + densenetblur121d(단독 11.14, 폴드 밖 선택으로도 뽑힘) 4개 결합 10.750. 3계열 블렌드 10.567 → 10.488 / 10.444 (폴드 5/5).
#   ViT 계열(DINOv2+TIPSv2)은 결합해도 이득 없음(특징 수가 이미 충분) → CNN 계열에만 적용.
# 사용: bash experiments/cnnfj_ff4.sh
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8
emb() { echo outputs/probe2/cnn_${1/@/_s}_512_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass${2}_low; }
grp() { echo "--group emb:$1:1:0:512:11:0,1,2,3,4,5 --group emb:$1:0.25:0:512:11:6,7,8,9 --group emb:$1:0.05:0:512:11:10"; }
for s in 48 32 64 96; do
  S=unet_cm7_am22_l1_i10000_boot_fflowpass${s}_low
  for M in densenet121.tv_in1k@2 resnet50.a1_in1k@2 resnet50.a2_in1k@2 densenetblur121d.ra_in1k@2; do
    E=$(emb $M $s)
    [ -f $E/emb.npz ] || $PY -W ignore src/probe_models.py --model cnn:$M --size 512 --pool phase2m --tta 8 --bs 16 --src outputs/denoised/$S.npz 2>&1 | grep -E "probe OOF|Error"
  done
  A=$(emb densenet121.tv_in1k@2 $s); B=$(emb resnet50.a1_in1k@2 $s); C=$(emb resnet50.a2_in1k@2 $s); D=$(emb densenetblur121d.ra_in1k@2 $s)
  $PY -W ignore src/multi_ridge.py --name cnnfj3_ff$s $(grp $A) $(grp $B) $(grp $C) | tail -1
  $PY -W ignore src/multi_ridge.py --name cnnfj4_ff$s $(grp $A) $(grp $B) $(grp $C) $(grp $D) | tail -1
  $PY -W ignore src/multi_ridge.py --name cnnf_dnb121_ff$s $(grp $D) | tail -1
done
echo DONE
