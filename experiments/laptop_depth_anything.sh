#!/usr/bin/env bash
# [노트북 10/18] 새 정보원 후보: Depth-Anything-V2-L 백본(CC-BY-NC-4.0) — F 경로 σ48·8보기(train) 층 3~21 → layer_scan_da.py
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
$PY -W ignore analysis/view_embed.py --model hf:depth-anything/Depth-Anything-V2-Large-hf --src outputs/denoised/drunetft_ft1b_fflowpass48_low.npz --splits train --shifts 0 --bs 8 --layers 3 5 7 9 11 13 15 17 19 21 --suffix _L3to21 2>&1 | grep -E "DONE|rror"
$PY -W ignore analysis/layer_scan_da.py 2>&1 | tail -8
echo ALLDONE
