#!/usr/bin/env bash
# 448 hetero3를 DINO 그룹 전체로 확장: L 12층(홀수층 전체, block_ridge로 부분집합 선택) · B 12층 전체 (GPU 단독)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM"
$PY -W ignore src/probe_models.py --model vit_base_patch14_dinov2.lvd142m --size 448 --layers 0 1 2 3 4 5 6 7 8 9 10 11 --pool hetero3 --bs 16 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 1 3 5 7 9 11 13 15 17 19 21 23 --pool hetero3 --bs 8 2>&1 | grep --line-buffered -E "$F"
