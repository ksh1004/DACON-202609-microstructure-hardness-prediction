#!/usr/bin/env bash
# 3단계 증강 실험 (ResNet-18 테스트베드, 5-Fold × 2시드) + EfficientNet 발산 수정 검증
# 기준선: scr_resnet18 / _s43 / _s44 = 13.73 / 13.84 / 13.84 (D4 + 노이즈σ≤8 + 블러 + 대비)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; tail -1 "outputs/logs/$name.log"; }
R18="--model resnet18.a1_in1k --lr 1e-3"
run scr_efficientnet_b0_wu3 --model efficientnet_b0.ra_in1k --lr 1e-3 --warmup 3
for s in 42 43; do
  run aug_rot50_s$s   $R18 --seed $s --rot-p 0.5
  run aug_noise16_s$s $R18 --seed $s --noise 16
  run aug_nod4_s$s    $R18 --seed $s --no-d4
done
