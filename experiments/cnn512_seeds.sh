#!/usr/bin/env bash
# v6e의 CNN 절반(512 입력) 시드 추가: ConvNeXt-T / EfficientNet-B2 / ResNet-50 × 시드 43, 44 (가중치 저장)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
AUG="--rot-p 0.5 --noise 16 --img-size 512 --grad-ckpt"
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --save-weights --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
for s in 43 44; do
  run s512_convnext_tiny_s$s --model convnext_tiny.fb_in22k_ft_in1k --lr 3e-4 $AUG --seed $s
  run s512_effnet_b2_s$s     --model efficientnet_b2.ra_in1k --lr 1e-3 --warmup 3 $AUG --seed $s
  run s512_resnet50_s$s      --model resnet50.a1_in1k --lr 1e-3 $AUG --seed $s
done
