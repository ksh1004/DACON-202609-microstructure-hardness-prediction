#!/usr/bin/env bash
# [노트북 10/17] TIPSv2 앞쪽 층 효과: F 경로 σ48·8보기(train) 층 2~14 → layer_scan_tips.py (laptop_fo_l3 종료 후)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
# (대기 조건 제거: 10/17 밤 수동 시작)
$PY -W ignore analysis/view_embed.py --model vit_so400m_patch14_reg1_tipsv2.webli --src outputs/denoised/drunetft_ft1b_fflowpass48_low.npz --splits train --shifts 0 --bs 8 --layers 2 4 6 8 10 12 14 --suffix _L2to14 2>&1 | grep -E "DONE|rror"
$PY -W ignore analysis/layer_scan_tips.py 2>&1 | tail -8
echo ALLDONE
