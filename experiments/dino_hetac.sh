#!/usr/bin/env bash
# 공간 자기상관(hetac) 풀링: hetero3 + 채널별 자기상관(지연 1·2·4·8 토큰) — GPU 단독
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 13 15 17 19 --pool hetac --bs 8 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 17 19 --pool hetac --bs 8 2>&1 | grep --line-buffered -E "$F"
