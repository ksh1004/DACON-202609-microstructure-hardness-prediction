#!/usr/bin/env bash
# hetero3 해상도·TTA 변형 (hetac 큐 종료 후, GPU 단독)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
OUTF="logs/bbdvqbwet.output"
until [ "$(grep -c '^\[vit' "$OUTF" 2>/dev/null)" -ge 2 ] || grep -qE "Traceback|Error" "$OUTF" 2>/dev/null; do sleep 20; done
sleep 10
F="^\[|Error|Traceback|OOM"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 13 15 17 19 --pool hetero3 --bs 8 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 616 --layers 13 15 17 19 --pool hetero3 --bs 4 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 13 15 17 19 --pool hetero3 --tta 8 --bs 8 2>&1 | grep --line-buffered -E "$F"
