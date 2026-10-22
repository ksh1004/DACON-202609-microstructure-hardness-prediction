#!/usr/bin/env bash
# [노트북 10/18] Depth-Anything-V2-L을 다른 곳에: (1) DRT·U 경로 σ48·8보기(train) (2) U 경로 σ48 4보기 phase2(train+test) → 측정 블록 P의 DINO 부분 교체
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
M=hf:depth-anything/Depth-Anything-V2-Large-hf
for S in unet_drt10b_i10000_fflowpass48_low unet_cm7_am22_l1_i10000_boot_fflowpass48_low; do
  $PY -W ignore analysis/view_embed.py --model $M --src outputs/denoised/$S.npz --splits train --shifts 0 --bs 8 --layers 9 11 13 15 17 --suffix _L9to17 2>&1 | grep -E "DONE|rror|Traceback"
  echo "[DA 8보기 $S 완료 $(date +%H:%M)]"
done
B=unet_cm7_am22_l1_i10000_boot_fflowpass48_low
$PY -W ignore src/probe_models.py --model $M --size 448 --layers 9 11 13 15 17 19 21 --pool phase2 --bs 8 --src outputs/denoised/$B.npz 2>&1 | grep -E "probe OOF|Error|OOM"
echo ALLDONE
