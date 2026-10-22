#!/usr/bin/env bash
# 자기지도 2차: ① 낮은 lr 미세조정 ② ResNet-18 500에폭 ③ ResNet-50 200에폭 — 각 사전학습 후 다중스케일 프로브(OOF+test)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
filt(){ grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys"; }
$PY -W ignore src/train_cnn.py --model resnet18 --init-weights "outputs/ssl/ssl_r18_ep200/fold{k}.pt" --lr 3e-4 --rot-p 0.5 --noise 16 --seed 42 --epochs 30 --eval-every 10 --exp ft_ssl_r18_lr3e4_s42 2>&1 | filt > outputs/logs/ft_ssl_r18_lr3e4_s42.log; tail -1 outputs/logs/ft_ssl_r18_lr3e4_s42.log
$PY -W ignore src/ssl_pretrain.py --model resnet18.a1_in1k --epochs 500 --exp ssl_r18_ep500 2>&1 | filt > outputs/logs/ssl_r18_ep500.log; tail -1 outputs/logs/ssl_r18_ep500.log
$PY -W ignore src/ssl_probe.py --exp ssl_r18_ep500 --model resnet18 2>&1 | filt | tail -1
$PY -W ignore src/ssl_pretrain.py --model resnet50.a1_in1k --epochs 200 --batch 64 --grad-ckpt --exp ssl_r50_ep200 2>&1 | filt > outputs/logs/ssl_r50_ep200.log; tail -1 outputs/logs/ssl_r50_ep200.log
$PY -W ignore src/ssl_probe.py --exp ssl_r50_ep200 --model resnet50 2>&1 | filt | tail -1
