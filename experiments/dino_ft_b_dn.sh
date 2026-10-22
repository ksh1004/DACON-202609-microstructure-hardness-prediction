#!/usr/bin/env bash
# NAFNet 평가 종료 대기 후: DINOv2-B 미세조정 (디노이즈+σ48 조명보정 입력, 448, hetero3 헤드 LP-FT, LLRD 0.75, lr_bb 5e-5, 20ep)
# 이전 시도(원본 입력, lr 1e-5, ≤10ep)는 고정특징과 동일 → 다른 오차를 가진 강한 모델이 목표 (앙상블 다양성)
cd "$(dirname "$0")/.."
until grep -q "dnpf_naf_boot" outputs/logs/denoiser_nafnet.log 2>/dev/null || grep -qE "Traceback|Error" outputs/logs/denoiser_nafnet.log 2>/dev/null; do sleep 20; done
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
$PY -W ignore src/dino_finetune.py --model vit_base_patch14_dinov2.lvd142m --layers 0 1 2 3 4 5 6 7 8 9 10 11 --size 448 --pool hetero3 \
  --src outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz --noise 0 --blur-p 0 --contrast-aug 0.05 \
  --lr-bb 5e-5 --lr-head 5e-4 --llrd 0.75 --epochs 20 --batch 16 --exp ftB_dnff48_het3 2>&1 | grep --line-buffered -E "RMSE|\[|Error|Traceback|OOM|ep(5|10|15|20) "
