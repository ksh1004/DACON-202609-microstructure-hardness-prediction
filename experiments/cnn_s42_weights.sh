#!/usr/bin/env bash
# 재현 패키지용: 가중치 미저장이던 s512_*_s42 3개를 같은 레시피로 재학습 (--save-weights). 기존 결과는 outputs/cnn/_archive_s42_noweights/
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
AUG="--rot-p 0.5 --noise 16 --img-size 512 --grad-ckpt"
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --save-weights --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
run s512_convnext_tiny_s42 --model convnext_tiny.fb_in22k_ft_in1k --lr 3e-4 $AUG --seed 42
run s512_effnet_b2_s42     --model efficientnet_b2.ra_in1k --lr 1e-3 --warmup 3 $AUG --seed 42
run s512_resnet50_s42      --model resnet50.a1_in1k --lr 1e-3 $AUG --seed 42
