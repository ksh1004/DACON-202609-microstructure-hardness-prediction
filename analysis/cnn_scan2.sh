#!/usr/bin/env bash
# 고정 CNN 구성원 2차 스캔 (2026-10-14): 1차에서 DenseNet-121·흐림 풀링 DenseNet이 최상위 → 조밀 연결·안티에일리어싱·다른 학습법 계열을 추가로 확인.
# train σ48, D4 8보기, 단계별 단독 OOF와 DINOv2-L 블렌드 (cnn_scan.py). 결과 특징은 cnn_joint.py가 결합 커널로 다시 평가.
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
for M in densenet121.ra_in1k resnetblur50.bt_in1k resnetaa50.a1h_in1k resnetaa50d.sw_in12k_ft_in1k ese_vovnet39b.ra_in1k cspresnet50.ra_in1k cspdarknet53.ra_in1k \
         darknetaa53.c2ns_in1k dpn92.mx_in1k dpn68b.ra_in1k xception65.ra3_in1k dm_nfnet_f0.dm_in1k nf_resnet50.ra2_in1k resnest50d.in1k repvgg_b3.rvgg_in1k \
         vgg16_bn.tv_in1k tresnet_m.miil_in21k_ft_in1k resnet50.tv2_in1k resnet50.fb_swsl_ig1b_ft_in1k seresnet50.a1_in1k ecaresnet50d.miil_in1k regnetz_c16.ra3_in1k \
         convnextv2_tiny.fcmae inception_resnet_v2.tf_in1k; do
  $PY -W ignore analysis/cnn_scan.py $M 2>&1 | grep -v "sklearnex\|Warning"
done
echo ALLDONE
