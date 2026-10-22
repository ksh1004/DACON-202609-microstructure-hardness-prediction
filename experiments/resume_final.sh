#!/usr/bin/env bash
# 2026-10-04 중단 지점부터 최종 CNN 블록 학습 재개
# 남은 작업: ① ConvNeXt 시드 42 재학습 (기존 실행은 fold4 학습 실패, 손실 0.94 → 자동 재시도 기능으로 다시)
#           ② EfficientNet 시드 43 (중단됨, fold0까지만 저장) ③ EfficientNet 시드 44
# 실행: bash experiments/resume_final.sh   (노트북 기준 약 1~1.5시간, 화면 꺼짐/절전 '안 함' 권장)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
AUG="--rot-p 0.5 --noise 16"
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --save-weights --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }

if [ ! -f outputs/logs/final_convnext_s42_failed_fold4.log ]; then
  mv outputs/logs/final_convnext_s42.log outputs/logs/final_convnext_s42_failed_fold4.log
  mv outputs/cnn/final_convnext_s42 outputs/cnn/_failed_final_convnext_s42
fi
run final_convnext_s42 --model convnext_nano.in12k_ft_in1k --lr 3e-4 $AUG --seed 42
run final_effnet_s43   --model efficientnet_b0.ra_in1k --lr 1e-3 --warmup 3 $AUG --seed 43
run final_effnet_s44   --model efficientnet_b0.ra_in1k --lr 1e-3 --warmup 3 $AUG --seed 44
