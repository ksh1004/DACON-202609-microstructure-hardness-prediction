#!/usr/bin/env bash
# 레시피 탐색 (resnet18, fold 0,1) — 백본 비교 전 공통 학습 설정 결정용
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --model resnet18.a1_in1k --folds 0 1 --eval-every 10 --exp "rcp_$name" "$@" 2>&1 | grep -v Warning > "outputs/logs/rcp_$name.log"; tail -1 "outputs/logs/rcp_$name.log"; }
run lr1e3_ep25  --lr 1e-3 --epochs 25
run lr1e3_ep60  --lr 1e-3 --epochs 60
run lr5e4_ep60  --lr 5e-4 --epochs 60
run lr1e3_ep60_noaugq --lr 1e-3 --epochs 60 --noise 0 --blur-p 0 --contrast 0
run lr2e3_ep60  --lr 2e-3 --epochs 60
