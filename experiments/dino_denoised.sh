#!/usr/bin/env bash
# 디노이즈 입력으로 DINOv2-L 특징 (노이즈 큰 이미지의 오차 개선 목적)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run(){ $PY -W ignore src/probe_models.py "$@" 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"; }
L="1 3 5 7 9 11 13 15 17 19 21 23"
for s in tv_c0.6 nlm_c0.8 tv_c1.0; do
  run --model vit_large_patch14_dinov2.lvd142m --size 518 --layers $L --bs 8 --src outputs/denoised/$s.npz
done
