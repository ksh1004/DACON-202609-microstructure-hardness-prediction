#!/usr/bin/env bash
# DINOv2 특징 읽기 방식 개선: VLAD(패치 유형 분포) / 공간 불균일성 요약 — 디노이즈 실험 종료 후 (GPU 단독)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
OUTF="logs/bot5rkq4z.output"
until grep -q "Stv_c1.0" "$OUTF" 2>/dev/null; do sleep 20; done
$PY -W ignore src/probe_vlad.py --model vit_large_patch14_dinov2.lvd142m --layers 13 17 21 --K 32 --pca 64 2>&1 | grep --line-buffered -E "^\[|codebook|Error|Traceback|OOM"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 13 15 17 19 --pool hetero --bs 8 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"
