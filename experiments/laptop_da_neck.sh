#!/usr/bin/env bash
# [노트북 10/18] Depth-Anything-V2-L 깊이 디코더(neck) 특징 + 예측 깊이 지도 통계: F 경로 σ48, 8보기, train+test (518 시험 종료 후)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
until grep -q "^ALLDONE" logs/laptop_da518.log 2>/dev/null; do sleep 15; done
$PY -W ignore analysis/da_neck_embed.py --src outputs/denoised/drunetft_ft1b_fflowpass48_low.npz --bs 4 2>&1 | grep -E "da_neck|DONE|rror|Traceback|OOM"
echo ALLDONE
