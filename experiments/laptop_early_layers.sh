#!/usr/bin/env bash
# [노트북 10/17] F 경로 DINOv2에서 앞쪽 층(3·5·7)도 도움이 되는지: σ48·8보기(train)만 층 3~17 추출 → analysis/layer_scan.py
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
until grep -q "^ALLDONE" logs/laptop_scunet.log 2>/dev/null; do sleep 30; done
$PY -W ignore analysis/view_embed.py --src outputs/denoised/drunetft_ft1b_fflowpass48_low.npz --splits train --shifts 0 --bs 8 --layers 3 5 7 9 11 13 15 17 --suffix _L3to17 2>&1 | grep -E "DONE|rror"
$PY -W ignore analysis/layer_scan.py 2>&1 | tail -12
echo ALLDONE
