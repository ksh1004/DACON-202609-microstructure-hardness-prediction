#!/usr/bin/env bash
# DINOv2 2차: 8방향 TTA + 분위수 요약 + Large 층 확장 (특징 저장 → 회귀 비교용)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
OUTF="logs/bjv83eo7u.output"
until grep -q "samvit" "$OUTF" 2>/dev/null || ! tasklist 2>/dev/null | grep -qi python; do sleep 15; done   # SAM 평가 종료 대기 (GPU 단독 사용)
run(){ $PY -W ignore src/probe_models.py "$@" 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"; }
B=vit_base_patch14_dinov2.lvd142m; L=vit_large_patch14_dinov2.lvd142m
run --model $B --size 518 --layers 12 --tta 8 --bs 16
run --model $B --size 518 --layers 12 --tta 8 --pool q --bs 16
run --model $L --size 518 --layers 1 3 5 7 9 11 13 15 17 19 21 23 --tta 8 --bs 8
run --model $L --size 518 --layers 1 3 5 7 9 11 13 15 17 19 21 23 --tta 8 --pool q --bs 8
