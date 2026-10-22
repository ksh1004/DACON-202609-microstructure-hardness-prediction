#!/usr/bin/env bash
# OOM으로 실패한 ConvNeXt-T 512 시드 44 재학습 — DINOv2 3차 평가 종료 후 (GPU 단독)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
until grep -q "vit_giant" "logs/bv13xbfl5.output" 2>/dev/null; do sleep 30; done
mv outputs/logs/s512_convnext_tiny_s44.log outputs/logs/s512_convnext_tiny_s44_oom.log
$PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --save-weights --exp s512_convnext_tiny_s44 --model convnext_tiny.fb_in22k_ft_in1k --lr 3e-4 --rot-p 0.5 --noise 16 --img-size 512 --grad-ckpt --seed 44 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > outputs/logs/s512_convnext_tiny_s44.log; tail -1 outputs/logs/s512_convnext_tiny_s44.log
