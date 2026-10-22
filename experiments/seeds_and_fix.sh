#!/usr/bin/env bash
# ① 시드 앙상블 효과 측정 (resnet18 × 시드 43, 44) ② EfficientNet-B0 발산 수정 (lr 5e-4)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
run scr_resnet18_s43 --model resnet18.a1_in1k --lr 1e-3 --seed 43
run scr_resnet18_s44 --model resnet18.a1_in1k --lr 1e-3 --seed 44
run scr_efficientnet_b0_lr5e4 --model efficientnet_b0.ra_in1k --lr 5e-4
