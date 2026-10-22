#!/usr/bin/env bash
# 512 입력 위에서 학습량·모델 용량 확대 효과 확인 (증강 레시피 동일: --rot-p 0.5 --noise 16)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
AUG="--rot-p 0.5 --noise 16 --img-size 512"
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --eval-every 10 --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
run s512_convnext_nano_s43 --epochs 30 --model convnext_nano.in12k_ft_in1k --lr 3e-4 $AUG --seed 43 --grad-ckpt
run s512_resnet18_ep60_s42 --epochs 60 --model resnet18.a1_in1k --lr 1e-3 $AUG --seed 42
run s512_convnext_tiny_s42 --epochs 30 --model convnext_tiny.fb_in22k_ft_in1k --lr 3e-4 $AUG --seed 42 --grad-ckpt
run s512_effnet_b2_s42     --epochs 30 --model efficientnet_b2.ra_in1k --lr 1e-3 --warmup 3 $AUG --seed 42 --grad-ckpt
run s512_resnet50_s42      --epochs 30 --model resnet50.a1_in1k --lr 1e-3 $AUG --seed 42 --grad-ckpt
