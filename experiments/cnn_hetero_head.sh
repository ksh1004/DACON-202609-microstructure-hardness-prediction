#!/usr/bin/env bash
# 불균일성 후속 큐 (GPU 단독, DINO hetero 큐 종료 후):
#  1) 영역별 예측 분포(probe_regions) 2) DINO hetero5 3) CNN 헤드를 불균일성 풀링으로 교체(s512 레시피, 시드 42)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
OUTF="logs/bd4h6h8p3.output"
until [ "$(grep -c '^\[vit' "$OUTF" 2>/dev/null)" -ge 4 ] || grep -qE "Traceback|Error" "$OUTF" 2>/dev/null; do sleep 20; done
sleep 10
F="^\[|fold|region|Error|Traceback|OOM"
$PY -W ignore src/probe_regions.py --layers 13 15 17 19 --g 4 --views 0 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 13 15 17 19 --pool hetero5 --bs 8 2>&1 | grep --line-buffered -E "$F"
AUG="--rot-p 0.5 --noise 16 --img-size 512 --grad-ckpt --head hetero"
run(){ name=$1; shift; $PY -W ignore src/train_cnn.py --epochs 30 --eval-every 10 --save-weights --exp "$name" "$@" 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > "outputs/logs/$name.log"; echo "[$name] $(tail -1 "outputs/logs/$name.log")"; }
run hh512_convnext_tiny_s42 --model convnext_tiny.fb_in22k_ft_in1k --lr 3e-4 $AUG --seed 42
run hh512_effnet_b2_s42     --model efficientnet_b2.ra_in1k --lr 1e-3 --warmup 3 $AUG --seed 42
run hh512_resnet50_s42      --model resnet50.a1_in1k --lr 1e-3 $AUG --seed 42
# 공간 자기상관 추가 풀링 (CNN 헤드 실험 뒤)
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 13 15 17 19 --pool hetac --bs 8 2>&1 | grep --line-buffered -E "$F"
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 518 --layers 17 19 --pool hetac --bs 8 2>&1 | grep --line-buffered -E "$F"
