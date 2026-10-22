#!/usr/bin/env bash
# DINOv2-Base 도메인 적응(자기지도, train만) → 12층 특징 Ridge. ConvNeXt s44 재학습 종료 후 실행 (GPU 단독)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
until grep -q "OOF RMSE" outputs/logs/s512_convnext_tiny_s44.log 2>/dev/null; do sleep 30; done
$PY -W ignore src/dino_adapt.py --model vit_base_patch14_dinov2.lvd142m --layers 0 1 2 3 4 5 6 7 8 9 10 11 --exp dinoB_adapt_e40 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN" > outputs/logs/dinoB_adapt_e40.log; tail -1 outputs/logs/dinoB_adapt_e40.log
