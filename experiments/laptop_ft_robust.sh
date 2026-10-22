#!/usr/bin/env bash
# [노트북 10/17] 미세조정 DRUNet 경로 견고성: ft1c(3차 증류, 가중치 있음 → 추론만), 다른 시드 ft1s1 → ft1s1b
# 각 경로: σ48 조명보정 → DINOv2 8보기(train) → analysis/pathsets.py 로 비교
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
post() {  # $1 = tag
  T=$1
  [ -f outputs/denoised/drunetft_${T}_fflowpass48_low.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/drunetft_$T.npz --mode lowpass --sigma 48 --only-low | tail -1
  [ -f outputs/probe2_views/drunetft_${T}_fflowpass48_low/train_s0_k7.npy ] || $PY -W ignore analysis/view_embed.py --src outputs/denoised/drunetft_${T}_fflowpass48_low.npz --splits train --shifts 0 --bs 8 2>&1 | grep -E "DONE|rror"
  echo "[$T] σ48 8보기 완료 $(date +%H:%M)"
}
[ -f outputs/denoised/drunetft_ft1c.npz ] || $PY -W ignore src/drunet_denoise.py --weights weights/pretrained_denoiser/drunetft_ft1c.pt --tag ft1c 2>&1 | grep -E "\[drunet\]|rror"
post ft1c
$PY -W ignore analysis/pathsets.py Fb Fc D+X+Fb D+X+Fc 2>&1 | tail -5
for spec in "ft1s1 --seed 1" "ft1s1b --seed 1 --target-src outputs/denoised/drunetft_ft1s1.npz"; do
  T=${spec%% *}
  [ -f outputs/denoised/drunetft_$T.npz ] || $PY -W ignore analysis/drunet_ft.py --tag $spec 2>&1 | grep --line-buffered -E "정답|it[0-9]*000 |저장|rror|OOM"
  post $T
done
$PY -W ignore analysis/pathsets.py Fb Fc Fs Fsb D+X+Fb D+X+Fsb 2>&1 | tail -8
echo ALLDONE
