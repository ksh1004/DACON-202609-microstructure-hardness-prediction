#!/usr/bin/env bash
# [노트북 10/18] DA 깊이 디코더 특징 + 깊이 지도 통계를 F 경로 σ32·64·96에서도 (σ48은 완료)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
for s in 32 64 96; do
  [ -f outputs/probe2/da_neck_drunetft_ft1b_fflowpass${s}_low/emb.npz ] || $PY -W ignore analysis/da_neck_embed.py --src outputs/denoised/drunetft_ft1b_fflowpass${s}_low.npz --bs 4 2>&1 | grep -E "DONE|rror|Traceback|OOM"
  E=outputs/probe2/da_neck_drunetft_ft1b_fflowpass${s}_low
  $PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name daneck_all_ff${s}_a0.01 --group emb:$E:1:0,1,2,3:256:11:0,1,2,3,4,5 --group emb:$E:0.25:0,1,2,3:256:11:6,7,8,9 --group emb:$E:0.05:0,1,2,3:256:11:10 | tail -1
  echo "[σ$s 완료 $(date +%H:%M)]"
done
echo ALLDONE
