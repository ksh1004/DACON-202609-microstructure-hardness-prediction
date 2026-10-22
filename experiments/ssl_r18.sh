#!/usr/bin/env bash
# 자기지도 사전학습(SimCLR, train 이미지만) 효과 검증 — ResNet-18
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="grep --line-buffered -v -i warning\|HF_TOKEN\|unexpected keys"
$PY -W ignore src/ssl_pretrain.py --model resnet18.a1_in1k --epochs 0   --exp ssl_r18_ep0   2>&1 | grep --line-buffered -v -i "warning\|HF_TOKEN" > outputs/logs/ssl_r18_ep0.log;   tail -1 outputs/logs/ssl_r18_ep0.log
$PY -W ignore src/ssl_pretrain.py --model resnet18.a1_in1k --epochs 200 --exp ssl_r18_ep200 2>&1 | grep --line-buffered -v -i "warning\|HF_TOKEN" > outputs/logs/ssl_r18_ep200.log; tail -1 outputs/logs/ssl_r18_ep200.log
$PY -W ignore src/train_cnn.py --model resnet18 --init-weights "outputs/ssl/ssl_r18_ep200/fold{k}.pt" --lr 1e-3 --rot-p 0.5 --noise 16 --seed 42 --epochs 30 --eval-every 10 --exp ft_ssl_r18_s42 2>&1 \
  | grep --line-buffered -v -i "warning\|HF_TOKEN\|unexpected keys" > outputs/logs/ft_ssl_r18_s42.log; tail -1 outputs/logs/ft_ssl_r18_s42.log
