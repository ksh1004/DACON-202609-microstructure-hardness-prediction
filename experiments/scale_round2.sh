#!/usr/bin/env bash
# 큰 모델 방향 2차: ① 크기 vs 해상도 분리 ② ConvNeXt-T·EffNet-B2 512 시드 추가 ③ 더 큰 모델
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
AUG="--rot-p 0.5 --noise 16"
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --save-weights --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
CT="--model convnext_tiny.fb_in22k_ft_in1k --lr 3e-4"
EB2="--model efficientnet_b2.ra_in1k --lr 1e-3 --warmup 3"
run s256_convnext_tiny_s42 $CT $AUG --seed 42
for s in 43 44; do
  run s512_convnext_tiny_s$s $CT $AUG --seed $s --img-size 512 --grad-ckpt
  run s512_effnet_b2_s$s     $EB2 $AUG --seed $s --img-size 512 --grad-ckpt
done
run s512_convnext_small_s42 --model convnext_small.fb_in22k_ft_in1k --lr 2e-4 $AUG --seed 42 --img-size 512 --grad-ckpt
run s512_effnet_b3_s42      --model efficientnet_b3.ra2_in1k --lr 1e-3 --warmup 3 $AUG --seed 42 --img-size 512 --grad-ckpt
