#!/usr/bin/env bash
# 타일(부분 영역) 학습: 학습 시 무작위 c×c 크롭(확대 없음), 추론은 전체 이미지 + D4 TTA
# 기준: final_resnet18_s42 (256, 크롭 없음, 30ep) = 13.58
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
AUG="--rot-p 0.5 --noise 16"
R18="--model resnet18.a1_in1k --lr 1e-3 --seed 42"
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --eval-every 10 --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
run crop128_ep30_r18 $R18 $AUG --crop 128 --epochs 30
run crop128_ep60_r18 $R18 $AUG --crop 128 --epochs 60
run crop160_ep60_r18 $R18 $AUG --crop 160 --epochs 60
run crop192_ep60_r18 $R18 $AUG --crop 192 --epochs 60
