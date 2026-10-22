#!/usr/bin/env bash
# 공개 대형 모델 선별 (나머지 5종) — 층 분산 방식 (DINOv2에서 마지막 4층보다 좋았음)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run(){ $PY -W ignore src/probe_models.py "$@" 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"; }
run --model vit_base_patch16_siglip_256.webli --size 512 --layers 2 5 8 11 --bs 16
run --model eva02_large_patch14_448.mim_m38m_ft_in22k_in1k --size 448 --layers 5 11 17 23 --bs 4
run --model tf_efficientnetv2_m.in21k_ft_in1k --size 512 --bs 16
run --model swinv2_base_window12to16_192to256.ms_in22k_ft_in1k --size 256 --bs 16
run --model samvit_base_patch16.sa1b --size 1024 --layers 2 5 8 11 --bs 2
