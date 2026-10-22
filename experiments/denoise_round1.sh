#!/usr/bin/env bash
# 디노이즈(U-Net) 입력으로 DINO 구성 확장: 해상도 재탐색(hetero3 392·518), L 12층 meanstd@518, B 12층 hetero3@448 (GPU 단독)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM"
S=outputs/denoised/unet_cm7_am22_l1_i6000.npz
L=vit_large_patch14_dinov2.lvd142m
$PY -W ignore src/probe_models.py --model vit_base_patch14_dinov2.lvd142m --size 448 --layers 0 1 2 3 4 5 6 7 8 9 10 11 --pool hetero3 --bs 16 --src $S 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model $L --size 518 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src $S 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model $L --size 392 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src $S 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model $L --size 518 --layers 1 3 5 7 9 11 13 15 17 19 21 23 --pool meanstd --bs 8 --src $S 2>&1 | grep --line-buffered -E "$F"
