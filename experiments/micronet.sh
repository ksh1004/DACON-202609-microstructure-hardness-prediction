#!/usr/bin/env bash
# NASA MicroNet(현미경 이미지 10만+ 장, MIT 라이선스) 사전학습 가중치 vs ImageNet 가중치 비교 — 같은 레시피·시드
# 가중치 출처: https://huggingface.co/jstuckner (weights/micronet/ 에 저장)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
AUG="--rot-p 0.5 --noise 16"
W=weights/micronet
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --save-weights --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
run mn_resnet18_256_s42 --model resnet18 --init-weights $W/resnet18_imagenet-micronet_weights.pth --lr 1e-3 $AUG --seed 42
run mn_resnet18_512_s42 --model resnet18 --init-weights $W/resnet18_imagenet-micronet_weights.pth --lr 1e-3 $AUG --seed 42 --img-size 512
run mn_effb2_512_s42 --model tf_efficientnet_b2 --init-weights $W/efficientnet-b2_imagenet-micronet_weights.pth --lr 1e-3 --warmup 3 $AUG --seed 42 --img-size 512 --grad-ckpt
run mn_resnet50_512_s42 --model resnet50 --init-weights $W/resnet50_imagenet-micronet_weights.pth --lr 1e-3 $AUG --seed 42 --img-size 512 --grad-ckpt
