#!/usr/bin/env bash
# 입력 해상도 스캔: hetero3는 448(12.28)이 518(12.41)보다 좋음 → 더 낮은 해상도·층 범위 확장, meanstd도 저해상도 확인 (GPU 단독)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM"
M=vit_large_patch14_dinov2.lvd142m
$PY -W ignore src/probe_models.py --model $M --size 392 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model $M --size 336 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model $M --size 448 --layers 1 3 5 7 9 11 13 15 17 19 21 23 --pool meanstd --bs 8 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model $M --size 280 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 2>&1 | grep --line-buffered -E "$F"
