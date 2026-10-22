#!/usr/bin/env bash
# 부트스트랩 디노이즈 입력에서 풀링 변형 재시험 (원본에선 hetero3 외 무익): gram(공분산), hetac(자기상관), q(분위수) — L11-17 @448
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
S=outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz
for P in hetac gram q; do
  $PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 11 13 15 17 --pool $P --bs 8 --src $S 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"
done
