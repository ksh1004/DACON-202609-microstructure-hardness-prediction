#!/usr/bin/env bash
# [노트북 10/18] F 경로 디노이즈에 격자 이동 TTA 추가 (ft1b 가중치 그대로, 이동 4 × D4 8 = 32회 평균) → σ48·8보기 → 경로 비교 (Fb = 기존 D4만)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
until grep -q "^ALLDONE" logs/laptop_tips_layers.log 2>/dev/null; do sleep 20; done
T=ft1b_s4
[ -f outputs/denoised/drunetft_$T.npz ] || $PY -W ignore src/drunet_denoise.py --weights weights/pretrained_denoiser/drunetft_ft1b.pt --tag $T --shifts 4 2>&1 | grep -E "\[drunet\]|rror"
[ -f outputs/denoised/drunetft_${T}_fflowpass48_low.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/drunetft_$T.npz --mode lowpass --sigma 48 --only-low | tail -1
[ -f outputs/probe2_views/drunetft_${T}_fflowpass48_low/train_s0_k7.npy ] || $PY -W ignore analysis/view_embed.py --src outputs/denoised/drunetft_${T}_fflowpass48_low.npz --splits train --shifts 0 --bs 8 2>&1 | grep -E "DONE|rror"
$PY -W ignore analysis/pathsets.py Fb Ft4 Fb+Ft4 D+X+Fb D+X+Ft4 2>&1 | tail -6
echo ALLDONE
