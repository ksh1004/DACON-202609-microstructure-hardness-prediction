#!/usr/bin/env bash
# [노트북 10/18] DINOv2 재학습 계열 후보 σ48·8보기(train, 층 9~17) 추출 → scan_dafam.py
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
for M in LiheYoung/depth-anything-large-hf depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf; do
  $PY -W ignore analysis/view_embed.py --model hf:$M --src outputs/denoised/drunetft_ft1b_fflowpass48_low.npz --splits train --shifts 0 --bs 8 --layers 9 11 13 15 17 --suffix _L9to17 2>&1 | grep -E "DONE|rror|Traceback"
  echo "[$M 완료 $(date +%H:%M)]"
done
$PY -W ignore analysis/scan_dafam.py 2>&1 | tail -8
echo ALLDONE
