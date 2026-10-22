#!/usr/bin/env bash
# [재개용] 해상도 스캔 남은 작업 (2026-10-05 밤 중단 지점). 완료: hetero3 @448 12.276(최고) / 518 12.407 / 392 12.47 / 336 12.57
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM"
M=vit_large_patch14_dinov2.lvd142m
# 1) 기존 meanstd DINO 블록도 448에서 좋아지는지 (518 L12 T8 = 12.699)
$PY -W ignore src/probe_models.py --model $M --size 448 --layers 1 3 5 7 9 11 13 15 17 19 21 23 --pool meanstd --bs 8 2>&1 | grep --line-buffered -E "$F"
# 2) 448 주변 해상도 (hetero3, 층 9~21 → block_ridge.py로 층 부분집합 선택)
$PY -W ignore src/probe_models.py --model $M --size 420 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model $M --size 476 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 2>&1 | grep --line-buffered -E "$F"
