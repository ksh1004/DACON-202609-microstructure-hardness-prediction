#!/usr/bin/env bash
# 디노이즈(U-Net) 입력 CNN: ConvNeXt-T 512 (s42) — 노이즈 증강 σ4 vs σ16 비교 (원본 s512_convnext_tiny_s42 = 13.07)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
S=outputs/denoised/unet_cm7_am22_l1_i6000.npz
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --save-weights --exp "$name" --src $S "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
run dn_s512_convnext_tiny_n4_s42  --model convnext_tiny.fb_in22k_ft_in1k --lr 3e-4 --rot-p 0.5 --noise 4  --img-size 512 --grad-ckpt --seed 42
run dn_s512_convnext_tiny_n16_s42 --model convnext_tiny.fb_in22k_ft_in1k --lr 3e-4 --rot-p 0.5 --noise 16 --img-size 512 --grad-ckpt --seed 42
