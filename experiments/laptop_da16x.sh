#!/usr/bin/env bash
# [노트북 10/18] 새 정보원: Depth-Anything-V2-L 백본(CC-BY-NC-4.0, 2024-06 공개) — DRT 경로(drt10b) 16보기 phase2m, 층 9~21, 조명보정 σ 4종 → 결합 Ridge(λ 0.01)
# 근거(σ48·8보기): 같은 설정 DINOv2 F 경로 층 9~17 10.267 → DA 10.059 (고품질 8.63→8.36, 저품질 12.18→12.03)
until grep -q "^ALLDONE" logs/laptop_da_more.log 2>/dev/null; do sleep 15; done
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
M=hf:depth-anything/Depth-Anything-V2-Large-hf; MT=depth-anything__Depth-Anything-V2-Large-hf; L="9 11 13 15 17 19 21"
one() { echo "--group emb:$1:1:$2:1024:11:0,1,2,3,4,5 --group emb:$1:0.25:$2:1024:11:6,7,8,9 --group emb:$1:0.05:$2:1024:11:10"; }
for s in 48 32 64 96; do
  F=unet_drt10b_i10000_fflowpass${s}_low
  E=outputs/probe2/${MT}_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${F}_sh2
  [ -f $E/emb.npz ] || $PY -W ignore src/probe_models.py --model $M --size 448 --layers $L --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/$F.npz 2>&1 | grep -E "probe OOF|Error|OOM"
  for spec in "0,1,2,3,4:L01234"; do LI=${spec%%:*}; NM=${spec##*:}
    n=da16x_drt10b_${NM}_ff${s}_a0.01; [ -f outputs/emb_reg/$n/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name $n $(one $E $LI) | tail -1
  done
  echo "[σ$s 완료 $(date +%H:%M)]"
done
echo ALLDONE
