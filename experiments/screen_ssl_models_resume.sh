#!/usr/bin/env bash
# 자기지도 모델 선별 — 2026-10-05 중단 지점 이후 남은 5종 (완료: DINO v1 ViT-S/8 = 13.61, outputs/probe2_results.csv)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run(){ $PY -W ignore src/probe_models.py "$@" 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"; }
A12="0 1 2 3 4 5 6 7 8 9 10 11"
run --model vit_base_patch8_224.dino --size 512 --layers $A12 --bs 4
run --model vit_base_patch16_224.dino --size 512 --layers $A12 --bs 16
run --model vit_base_patch16_224.mae --size 512 --layers $A12 --bs 16
run --model vit_large_patch16_224.mae --size 512 --layers 1 3 5 7 9 11 13 15 17 19 21 23 --bs 8
run --model vit_huge_patch14_gap_224.in22k_ijepa --size 518 --layers 2 5 8 11 14 17 20 23 26 29 31 --bs 4 --bf16-weights
