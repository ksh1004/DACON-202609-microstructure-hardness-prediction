#!/usr/bin/env bash
# 조명 보정 대상(저품질 모드) 판정 기준 σ≥th 변경: 9 / 11 / 12 (기준 10). σ 9~12에서 두 모드가 겹침.
# 기준(th10): DINO L11-17 11.337 / + 상분율 pf_ 10.849
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
B=unet_cm7_am22_l1_i10000_boot
for th in 9 11 12; do
  S=${B}_fflowpass48_low_th$th
  $PY src/flatfield_cache.py --src outputs/denoised/$B.npz --mode lowpass --sigma 48 --only-low --sig-th $th | grep "^\["
  $PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/$S.npz 2>&1 | grep -E "^\[|Error|Traceback|OOM"
  $PY -W ignore src/phase_features.py --src outputs/denoised/$S.npz --workers 6 | tail -1
  T=vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_S$S
  $PY src/block_ridge.py --src outputs/probe2/$T --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dn_ff48_th${th}_L11-17 | grep "^\["
  $PY src/block_ridge.py --src outputs/probe2/$T --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dnpf_ff48_th${th}_L11-17 --extra 0.4 outputs/features_cache/phase_$S::pf_ | grep "^\["
done
