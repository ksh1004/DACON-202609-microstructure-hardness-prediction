#!/usr/bin/env bash
# CNN 블록 최종 학습 (3시드, 가중치 저장) — 증강 실험으로 정한 레시피 사용
# 사용: bash experiments/final_cnn_seeds.sh "<증강 옵션>" [블록 ...]
#   예) bash experiments/final_cnn_seeds.sh "--rot-p 0.5 --noise 16" resnet18 convnext effnet
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
AUG=${1:-"--rot-p 0.5 --noise 16"}; shift
BLOCKS=${@:-"resnet18 convnext effnet"}
TAG=${TAG:-final}
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --save-weights --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
for b in $BLOCKS; do
  case $b in
    resnet18) M="--model resnet18.a1_in1k --lr 1e-3" ;;
    convnext) M="--model convnext_nano.in12k_ft_in1k --lr 3e-4" ;;
    effnet)   M="--model efficientnet_b0.ra_in1k --lr 1e-3 --warmup 3" ;;
  esac
  for s in 42 43 44; do
    run "${TAG}_${b}_s$s" $M $AUG --seed $s
  done
done
