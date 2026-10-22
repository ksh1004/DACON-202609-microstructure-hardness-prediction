#!/usr/bin/env bash
# 방향별 특징 실험이 끝나면: ft1b 전체 4보정 추출 → 3차 증류 ft1c와 다른 시드(ft1s1 → ft1s1b) σ48 시험
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
until grep -q "^ALLDONE" logs/perview.log 2>/dev/null; do sleep 30; done
bash experiments/ft_ff4.sh ft1b
for spec in "ft1c --target-src outputs/denoised/drunetft_ft1b.npz" "ft1s1 --seed 1" "ft1s1b --seed 1 --target-src outputs/denoised/drunetft_ft1s1.npz"; do
  T=${spec%% *}
  [ -f outputs/denoised/drunetft_$T.npz ] || $PY -W ignore analysis/drunet_ft.py --tag $spec 2>&1 | grep --line-buffered -E "정답|it6000|저장|rror"
  [ -f outputs/denoised/drunetft_${T}_fflowpass48_low.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/drunetft_$T.npz --mode lowpass --sigma 48 --only-low | tail -1
  [ -f outputs/probe2_views/drunetft_${T}_fflowpass48_low/train_s0_k7.npy ] || $PY -W ignore analysis/view_embed.py --src outputs/denoised/drunetft_${T}_fflowpass48_low.npz --splits train --shifts 0 --bs 8 2>&1 | grep -E "DONE|rror"
done
echo ALLDONE
