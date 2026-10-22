#!/usr/bin/env bash
# DRUNet 미세조정 실험이 끝나면 방향별 디노이즈 특징 실험 (GPU 순차 실행)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
until grep -q "^ALLDONE" logs/drunetft.log 2>/dev/null; do sleep 30; done
$PY -W ignore analysis/perview.py --weights weights/denoiser_drt10b_i10000.pt --name unet_drt10b_i10000_pv 2>&1 | grep --line-buffered -E "k7|DONE|rror"
$PY -W ignore analysis/perview.py --weights weights/denoiser_cm7_am22_l1_i10000_boot.pt --name unet_boot_pv 2>&1 | grep --line-buffered -E "k7|DONE|rror"
$PY -W ignore analysis/perview.py --drunet --name drunet_pv 2>&1 | grep --line-buffered -E "k7|DONE|rror"
$PY -W ignore analysis/pathx2.py unet_drt10b_i10000_pv_fflowpass48_low "drt10b+unet_drt10b_i10000_pv_fflowpass48_low" 2>&1 | grep -v sklearnex
$PY -W ignore analysis/perview_eval.py 2>&1 | grep -v sklearnex
echo ALLDONE
