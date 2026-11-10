#!/usr/bin/env bash
# DA 깊이 디코더 특징 Ridge (σ48, 8보기): 4개 해상도 전부 / 저해상도 2개 / 고해상도 2개
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
E=outputs/probe2/da_neck_drunetft_ft1b_fflowpass48_low
one() { echo "--group emb:$E:1:$1:256:11:0,1,2,3,4,5 --group emb:$E:0.25:$1:256:11:6,7,8,9 --group emb:$E:0.05:$1:256:11:10"; }
for spec in "0,1,2,3:all" "0,1:lo" "2,3:hi" "1,2:mid"; do L=${spec%%:*}; N=${spec##*:}
  $PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name daneck_${N}_ff48_a0.01 $(one $L) | tail -1
done
