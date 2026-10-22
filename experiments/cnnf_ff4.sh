#!/usr/bin/env bash
# 고정 CNN 특징 구성원: ImageNet 사전학습 CNN의 stride-8 단계 특징맵(512 입력 → 64×64)에 DINO와 같은 풀링(hetero3 + 상 조건부 + 기여량),
# D4 8보기 평균 → Ridge. 조명보정 σ 4종. 근거(2026-10-13, train 8보기): 단독 11.15~11.42, DINOv2-L과 잔차상관 0.87~0.88,
# DINOv2-L .45 + TIPSv2 .25 + 상위 CNN .30 → 10.62~10.64 (DINOv2+TIPSv2 10.80), 폴드 밖 선택으로도 같은 모델이 뽑힘.
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8
for spec in "densenet121.tv_in1k@2 dn121" "resnet50.a1_in1k@2 r50a1" "resnet50.a2_in1k@2 r50a2"; do
  set -- $spec; M=$1; N=$2
  for s in 48 32 64 96; do
    S=unet_cm7_am22_l1_i10000_boot_fflowpass${s}_low
    E=outputs/probe2/cnn_${M/@/_s}_512_Pphase2m_T8_S${S}
    [ -f $E/emb.npz ] || $PY -W ignore src/probe_models.py --model cnn:$M --size 512 --pool phase2m --tta 8 --bs 16 --src outputs/denoised/$S.npz 2>&1 | grep -E "probe OOF|Error"
    $PY -W ignore src/multi_ridge.py --name cnnf_${N}_ff$s --group "emb:$E:1:0:512:11:0,1,2,3,4,5" --group "emb:$E:0.25:0:512:11:6,7,8,9" --group "emb:$E:0.05:0:512:11:10" | tail -1
  done
done
echo DONE
