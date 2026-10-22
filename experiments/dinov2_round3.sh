#!/usr/bin/env bash
# DINOv2 3차: 그람(공분산) 요약 + Giant 모델 — CNN 시드 학습 종료 후 실행 (GPU 단독)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
until grep -q "OOF RMSE" outputs/logs/s512_resnet50_s44.log 2>/dev/null; do sleep 30; done
run(){ $PY -W ignore src/probe_models.py "$@" 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"; }
run --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 13 15 17 19 21 --pool gram --bs 8
run --model vit_base_patch14_dinov2.lvd142m --size 518 --layers 12 --pool gram --bs 16
run --model vit_giant_patch14_dinov2.lvd142m --size 518 --layers 3 7 11 15 19 23 27 31 35 39 --bs 4 --bf16-weights
