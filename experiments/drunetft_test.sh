#!/usr/bin/env bash
# DRUNet 미세조정 디노이저 → σ48·8보기 DINOv2 경로 비교 (analysis/pathx2.py)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
for T in "$@"; do
  case $T in
    ft1) ARGS="--tag ft1" ;;
    ft1b) ARGS="--tag ft1b --target-src outputs/denoised/drunetft_ft1.npz" ;;
    ftlr3) ARGS="--tag ftlr3 --lr 3e-4" ;;
    *) echo "unknown $T"; continue ;;
  esac
  [ -f outputs/denoised/drunetft_$T.npz ] || $PY -W ignore analysis/drunet_ft.py $ARGS 2>&1 | grep -v sklearnex
  [ -f outputs/denoised/drunetft_${T}_fflowpass48_low.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/drunetft_$T.npz --mode lowpass --sigma 48 --only-low | tail -1
  [ -f outputs/probe2_views/drunetft_${T}_fflowpass48_low/train_s0_k7.npy ] || $PY -W ignore analysis/view_embed.py --src outputs/denoised/drunetft_${T}_fflowpass48_low.npz --splits train --shifts 0 --bs 8 2>&1 | grep -E "DONE|rror"
  $PY -W ignore analysis/pathx2.py drunetft_${T}_fflowpass48_low "drt10b+drunetft_${T}_fflowpass48_low" 2>&1 | grep -v sklearnex
done
echo ALLDONE
