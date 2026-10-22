#!/usr/bin/env bash
# 디노이즈 입력에서 다른 고정 모델 재선별 (원본 기준 OOF: MAE-B 13.50, DINO v1 B/16 13.60, MAE-L 13.58, EVA02-L 13.55, DINOv2-g 12.84)
# 원본에선 DINOv2만 우수했지만, 노이즈에 약한 고정 특징이 디노이즈로 살아나면 오차 패턴이 다른 블록이 생길 수 있음. 사용: bash experiments/screen_denoised.sh <디노이즈 npz>
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
S=${1:-outputs/denoised/unet_cm7_am22_l1_i6000.npz}
run(){ $PY -W ignore src/probe_models.py --src $S --pool hetero3 "$@" 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"; }
A12="0 1 2 3 4 5 6 7 8 9 10 11"
run --model vit_base_patch16_224.mae --size 512 --layers $A12 --bs 16
run --model vit_base_patch16_224.dino --size 512 --layers $A12 --bs 16
run --model vit_large_patch16_224.mae --size 512 --layers 5 9 13 17 21 --bs 8
run --model eva02_large_patch14_448.mim_m38m_ft_in22k_in1k --size 448 --layers 5 9 13 17 21 --bs 8
run --model vit_giant_patch14_dinov2.lvd142m --size 448 --layers 18 22 26 30 --bs 4 --bf16-weights
