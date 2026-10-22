#!/usr/bin/env bash
# 조명 보정 원본 입력 CNN (저품질 모드만 σ48 저역 제거, 디노이즈 없음): ConvNeXt-T → (개선 시) EffNet-B2 / ResNet-50, 512, s42
# 기준: 원본 입력 s512_convnext_tiny_s42 13.07 / effnet_b2 / resnet50 (v7 CNN 9모델 평균 12.92)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
S=outputs/denoised/raw_fflowpass48_low.npz
AUG="--rot-p 0.5 --noise 16 --img-size 512 --grad-ckpt"
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --save-weights --exp "$name" --src $S "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
run ff_s512_convnext_tiny_s42 --model convnext_tiny.fb_in22k_ft_in1k --lr 3e-4 $AUG --seed 42
run ff_s512_effnet_b2_s42     --model efficientnet_b2.ra_in1k --lr 1e-3 --warmup 3 $AUG --seed 42
run ff_s512_resnet50_s42      --model resnet50.a1_in1k --lr 1e-3 $AUG --seed 42
