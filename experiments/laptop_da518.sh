#!/usr/bin/env bash
# [노트북 10/18] Depth-Anything-V2-L 입력 해상도 518(원 학습 해상도) vs 448: F 경로 σ48·8보기(train) 층 9~17
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
$PY -W ignore analysis/view_embed.py --model hf:depth-anything/Depth-Anything-V2-Large-hf --size 518 --src outputs/denoised/drunetft_ft1b_fflowpass48_low.npz --splits train --shifts 0 --bs 4 --layers 9 11 13 15 17 --suffix _L9to17 2>&1 | grep -E "DONE|rror|Traceback|OOM"
echo ALLDONE
