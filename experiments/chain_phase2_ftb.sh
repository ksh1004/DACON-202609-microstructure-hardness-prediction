#!/usr/bin/env bash
# NAFNet 평가 종료 대기 → (1) 상 조건부 풀링 DINO-L 추출 (부트스트랩 ff48) (2) DINOv2-B 미세조정 (디노이즈 ff48 입력)
cd "$(dirname "$0")/.."
until grep -q "dnpf_naf_boot" outputs/logs/denoiser_nafnet.log 2>/dev/null || grep -qE "Traceback|Error" outputs/logs/denoiser_nafnet.log 2>/dev/null; do sleep 20; done
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
S=unet_cm7_am22_l1_i10000_boot_fflowpass48_low
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool phase2 --bs 8 --src outputs/denoised/$S.npz 2>&1 | grep -E "^\[|Error|Traceback|OOM"
echo "PHASE2_DONE"
$PY -W ignore src/dino_finetune.py --model vit_base_patch14_dinov2.lvd142m --layers 0 1 2 3 4 5 6 7 8 9 10 11 --size 448 --pool hetero3 \
  --src outputs/denoised/$S.npz --noise 0 --blur-p 0 --contrast-aug 0.05 \
  --lr-bb 5e-5 --lr-head 5e-4 --llrd 0.75 --epochs 20 --batch 16 --exp ftB_dnff48_het3 2>&1 | grep --line-buffered -E "RMSE|\[|Error|Traceback|OOM|ep(5|10|15|20) "
