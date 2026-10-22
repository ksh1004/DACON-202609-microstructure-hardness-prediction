#!/usr/bin/env bash
# [노트북 10/18] MiDaS v3 DPT-Large(Intel, Apache-2.0, 2021) 백본: F 경로 σ48·8보기(train) 층 9~17 → scan_dafam.py (DRT 디코더 추출 종료 후)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
until grep -q "^ALLDONE" logs/laptop_neck_x.log 2>/dev/null; do sleep 15; done
$PY -W ignore analysis/view_embed.py --model hf:Intel/dpt-large --src outputs/denoised/drunetft_ft1b_fflowpass48_low.npz --splits train --shifts 0 --bs 8 --layers 9 11 13 15 17 --suffix _L9to17 2>&1 | grep -E "DONE|rror|Traceback"
$PY -W ignore analysis/scan_dafam.py 2>&1 | tail -9
echo ALLDONE
