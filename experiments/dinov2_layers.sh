#!/usr/bin/env bash
# DINOv2 층 선택·해상도 조정 (고정 특징 + Ridge). ViT-B는 12블록(0~11), ViT-L은 24블록(0~23)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run(){ $PY -W ignore src/probe_models.py "$@" 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"; }
B=vit_base_patch14_dinov2.lvd142m
run --model $B --size 518 --layers 2 5 8 11
run --model $B --size 518 --layers 12
run --model $B --size 518 --layers 0 1 2 3 4 5
run --model vit_base_patch14_reg4_dinov2.lvd142m --size 518 --layers 2 5 8 11
run --model $B --size 742 --layers 2 5 8 11 --bs 8
run --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 5 11 17 23 --bs 8
