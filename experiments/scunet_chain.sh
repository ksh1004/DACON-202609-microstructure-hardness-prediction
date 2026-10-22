#!/usr/bin/env bash
# ft_chain이 끝나면: 공개 SCUNet(gray σ25, Apache-2.0)을 같은 방식으로 미세조정 sc1 → 2차 증류 sc1b → σ48·8보기 특징
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
until grep -q "^ALLDONE" logs/ft_chain.log 2>/dev/null; do sleep 30; done
for spec in "sc1" "sc1b --target-src outputs/denoised/scunetft_sc1.npz"; do
  T=${spec%% *}
  [ -f outputs/denoised/scunetft_$T.npz ] || $PY -W ignore analysis/drunet_ft.py --arch scunet --tag $spec 2>&1 | grep --line-buffered -E "정답|시작|it[0-9]*000 |저장|rror"
  [ -f outputs/denoised/scunetft_${T}_fflowpass48_low.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/scunetft_$T.npz --mode lowpass --sigma 48 --only-low | tail -1
  [ -f outputs/probe2_views/scunetft_${T}_fflowpass48_low/train_s0_k7.npy ] || $PY -W ignore analysis/view_embed.py --src outputs/denoised/scunetft_${T}_fflowpass48_low.npz --splits train --shifts 0 --bs 8 2>&1 | grep -E "DONE|rror"
done
echo ALLDONE
