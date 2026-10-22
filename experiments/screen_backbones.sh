#!/usr/bin/env bash
# 백본 비교 (5-Fold 전체, 공통 레시피: 30ep, 품질증강, D4 TTA) — 모델 선택용
# 사용: bash experiments/screen_backbones.sh [model:lr ...]
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
EP=30
SPECS=${@:-"resnet18.a1_in1k:1e-3 resnet50.a1_in1k:1e-3 convnext_nano.in12k_ft_in1k:3e-4 efficientnet_b0.ra_in1k:1e-3"}
for spec in $SPECS; do
  m=${spec%%:*}; lr=${spec##*:}
  name="scr_$(echo "$m" | cut -d. -f1)"
  $PY -W ignore src/train_cnn.py --model "$m" --epochs "$EP" --lr "$lr" --eval-every 10 --exp "$name" 2>&1 \
    | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"
  tail -1 "outputs/logs/$name.log"
done
