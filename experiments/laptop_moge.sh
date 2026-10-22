#!/usr/bin/env bash
# [노트북 10/18 밤] MoGe-1·MoGe-2(Microsoft, MIT; DINOv2-L을 3차원 형상 추정으로 재학습) 백본: F 경로 σ48·8보기(train) 층 9~17 → scan_dafam.py
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
for M in moge:Ruicheng/moge-vitl moge:Ruicheng/moge-2-vitl; do
  $PY -W ignore analysis/view_embed.py --model $M --src outputs/denoised/drunetft_ft1b_fflowpass48_low.npz --splits train --shifts 0 --bs 8 --layers 9 11 13 15 17 --suffix _L9to17 2>&1 | grep -E "DONE|rror|Traceback"
  echo "[$M 완료 $(date +%H:%M)]"
done
$PY -W ignore analysis/scan_dafam.py 2>&1 | tail -10
echo ALLDONE
