#!/usr/bin/env bash
# 자기지도(SSL) 사전학습 모델 선별 — DINOv2처럼 정답 없이 학습된 모델들. DINOv2 미세조정 종료 후 실행 (GPU 단독)
# 라이선스: DINO v1 Apache-2.0, MAE·I-JEPA CC-BY-NC-4.0 (대회 허용). DINOv3는 자체 라이선스라 제외
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
until grep -q "5폴드 병합\|FT RMSE" outputs/logs/dinoB_ft.log 2>/dev/null; do sleep 30; done
run(){ $PY -W ignore src/probe_models.py "$@" 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"; }
A12="0 1 2 3 4 5 6 7 8 9 10 11"
run --model vit_small_patch8_224.dino --size 512 --layers $A12 --bs 8
run --model vit_base_patch8_224.dino --size 512 --layers $A12 --bs 4
run --model vit_base_patch16_224.dino --size 512 --layers $A12 --bs 16
run --model vit_base_patch16_224.mae --size 512 --layers $A12 --bs 16
run --model vit_large_patch16_224.mae --size 512 --layers 1 3 5 7 9 11 13 15 17 19 21 23 --bs 8
run --model vit_huge_patch14_gap_224.in22k_ijepa --size 518 --layers 2 5 8 11 14 17 20 23 26 29 31 --bs 4 --bf16-weights
