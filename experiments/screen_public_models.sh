#!/usr/bin/env bash
# 공개 대형 사전학습 모델 선별 (고정 다층 특징 + Ridge). 라이선스: Apache-2.0 / MIT / CC-BY-NC-4.0 (대회 허용)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run(){ $PY -W ignore src/probe_models.py "$@" 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"; }
run --model vit_base_patch14_dinov2.lvd142m --size 518 --bs 16
run --model vit_large_patch14_dinov2.lvd142m --size 518 --bs 8
run --model eva02_base_patch14_448.mim_in22k_ft_in22k_in1k --size 448 --bs 16
run --model convnext_base.fb_in22k_ft_in1k --size 512 --bs 16
run --model convnextv2_base.fcmae_ft_in22k_in1k --size 512 --bs 16
run --model tf_efficientnetv2_m.in21k_ft_in1k --size 512 --bs 16
run --model vit_base_patch16_siglip_256.webli --size 512 --bs 16
run --model swinv2_base_window12to16_192to256.ms_in22k_ft_in1k --size 256 --bs 16
run --model eva02_large_patch14_448.mim_m38m_ft_in22k_in1k --size 448 --bs 4
run --model samvit_base_patch16.sa1b --size 1024 --bs 2
