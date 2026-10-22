#!/usr/bin/env bash
# 공간 불균일성(hetero) 풀링 확장 — 노이즈 증강 실험 종료 후 (GPU 단독)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
OUTF="logs/bntn1cnkj.output"
until [ "$(grep -c '^\[aug_' "$OUTF" 2>/dev/null)" -ge 2 ] || grep -qE "Traceback|Error" "$OUTF" 2>/dev/null; do sleep 20; done
sleep 10
F="^\[|Error|Traceback|OOM"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 13 15 17 19 --pool hetero3 --bs 8 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_base_patch14_dinov2.lvd142m --size 518 --layers 0 1 2 3 4 5 6 7 8 9 10 11 --pool hetero --bs 16 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 1 3 5 7 9 11 13 15 17 19 21 23 --pool hetero --bs 8 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 17 19 --pool hetero --bs 8 2>&1 | grep --line-buffered -E "$F"
