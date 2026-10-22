#!/usr/bin/env bash
# 측정 특징(상분율·수작업·평활 CDF)을 DRT·DRUNet 경로 σ48 이미지에서 계산 (CPU)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=2
for S in unet_drt10b_i10000_fflowpass48_low drunet_fflowpass48_low; do
  [ -f outputs/features_cache/phase_${S}_test.csv ] || $PY -W ignore src/phase_features.py --src outputs/denoised/$S.npz --workers 6 2>&1 | tail -1
  [ -f outputs/features_cache/${S}_test.csv ] || $PY -W ignore src/features_from_cache.py --src outputs/denoised/$S.npz 2>&1 | tail -1
  [ -f outputs/features_cache/smcdf3_${S}_test.csv ] || $PY -W ignore src/smooth_cdf_features.py --src outputs/denoised/$S.npz --workers 4 2>&1 | tail -1
done
echo ALLDONE
