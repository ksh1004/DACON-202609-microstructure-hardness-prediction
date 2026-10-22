#!/usr/bin/env bash
# 증강 2차: ① 회전+노이즈16 조합 (ResNet-18 3시드, 가중치 저장 → 채택 시 최종 블록) ② 노이즈 σ24 추세 확인
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
TAG=final bash experiments/final_cnn_seeds.sh "--rot-p 0.5 --noise 16" resnet18
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
for s in 42 43; do run aug_noise24_s$s --model resnet18.a1_in1k --lr 1e-3 --seed $s --noise 24; done
