#!/usr/bin/env bash
# 특징 수준 노이즈 증강 Ridge (train 이미지 + 노이즈 사본, 같은 정답) — hetero 프로브 종료 후 (GPU 단독)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
OUTF="logs/bxyzjzdkw.output"
until grep -qE "Phetero|Traceback|Error" "$OUTF" 2>/dev/null; do sleep 20; done
sleep 10
$PY -W ignore src/probe_aug.py --layers 13 15 17 19 --copies 2 --lo 4 --hi 20 2>&1 | grep --line-buffered -E "^\[|증강 없음|ready|Error|Traceback|OOM"
$PY -W ignore src/probe_aug.py --layers 13 15 17 19 --copies 2 --lo 8 --hi 28 --blur-p 0.3 --copy-weight 0.5 2>&1 | grep --line-buffered -E "^\[|증강 없음|ready|Error|Traceback|OOM"
