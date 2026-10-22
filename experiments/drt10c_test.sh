#!/usr/bin/env bash
# 증류 한 번 더: drt10b 출력(고품질 291장)을 정답으로 다시 학습 → σ48·8보기 DINOv2 경로 비교
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4
T=drt10c_i10000
[ -f outputs/denoised/unet_$T.npz ] || $PY -W ignore src/train_denoiser.py --iters 10000 --clean-max 10 --target-src outputs/denoised/unet_drt10b_i10000.npz --tag $T 2>&1 | grep -E "it10000|저장"
[ -f outputs/denoised/unet_${T}_fflowpass48_low.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/unet_$T.npz --mode lowpass --sigma 48 --only-low | tail -1
[ -f outputs/probe2_views/unet_${T}_fflowpass48_low/train_s0_k7.npy ] || $PY -W ignore analysis/view_embed.py --src outputs/denoised/unet_${T}_fflowpass48_low.npz --splits train --shifts 0 --bs 8 2>&1 | grep -E "DONE|rror"
$PY -W ignore analysis/pathx.py unet_drt10b_i10000_fflowpass48_low unet_${T}_fflowpass48_low 2>&1 | grep -v sklearnex
echo ALLDONE
