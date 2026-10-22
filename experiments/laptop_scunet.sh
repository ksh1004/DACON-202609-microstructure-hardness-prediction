#!/usr/bin/env bash
# [노트북 10/17] 공개 SCUNet(gray σ25, Apache-2.0) 미세조정 sc1 → 2차 sc1b (DRUNet ft1b와 같은 방식, 8GB라 배치 8) → σ48·8보기 → 경로 비교
# 큐 대기: laptop_cnn_ft 종료 후
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
until grep -q "^ALLDONE" logs/laptop_cnn_ft.log 2>/dev/null; do sleep 30; done
for spec in "sc1" "sc1b --target-src outputs/denoised/scunetft_sc1.npz"; do
  T=${spec%% *}
  [ -f outputs/denoised/scunetft_$T.npz ] || $PY -W ignore analysis/drunet_ft.py --arch scunet --batch 8 --tag $spec 2>&1 | grep --line-buffered -E "정답|시작|it[0-9]*000 |저장|rror|OOM"
  [ -f outputs/denoised/scunetft_${T}_fflowpass48_low.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/scunetft_$T.npz --mode lowpass --sigma 48 --only-low | tail -1
  [ -f outputs/probe2_views/scunetft_${T}_fflowpass48_low/train_s0_k7.npy ] || $PY -W ignore analysis/view_embed.py --src outputs/denoised/scunetft_${T}_fflowpass48_low.npz --splits train --shifts 0 --bs 8 2>&1 | grep -E "DONE|rror"
  echo "[$T] σ48 8보기 완료 $(date +%H:%M)"
  $PY -W ignore analysis/pathsets.py Fb S Sb Fb+S Fb+Sb D+X+Fb D+X+Fb+S D+X+Fb+Sb 2>&1 | tail -9
done
echo ALLDONE
