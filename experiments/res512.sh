#!/usr/bin/env bash
# 해상도 가설 검증: 입력 2배 업샘플(512) vs 기존 256 (같은 시드·레시피로 쌍대 비교)
# 비교 기준 (256): final_resnet18_s42/s43 = 13.58/13.48, final_effnet_s42 = 13.66
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
AUG="--rot-p 0.5 --noise 16"
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
run res512_resnet18_s42 --model resnet18.a1_in1k --lr 1e-3 $AUG --seed 42 --img-size 512
run res512_resnet18_s43 --model resnet18.a1_in1k --lr 1e-3 $AUG --seed 43 --img-size 512
run res512_effnet_s42   --model efficientnet_b0.ra_in1k --lr 1e-3 --warmup 3 $AUG --seed 42 --img-size 512
