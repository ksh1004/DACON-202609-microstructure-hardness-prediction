#!/usr/bin/env bash
# 측정 특징 블록(mr_v14_pp_sc)을 조명보정 σ32/64/96에서도 만들어 4종 평균 (DINO ff4와 같은 분산 감소). 수작업 ph_/pgs는 σ48 것 공유.
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8
S48=unet_cm7_am22_l1_i10000_boot_fflowpass48_low
for s in 32 64 96; do
  S=unet_cm7_am22_l1_i10000_boot_fflowpass${s}_low
  $PY -W ignore src/phase_features.py --src outputs/denoised/$S.npz | tail -1
  $PY -W ignore src/smooth_cdf_features.py --src outputs/denoised/$S.npz --workers 8 | tail -1
  $PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool phase2 --bs 8 --src outputs/denoised/$S.npz 2>&1 | grep "probe OOF"
  E=outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2_S$S
  $PY -W ignore src/multi_ridge.py --name mr_v14_pp_sc_ff$s --group "emb:$E:1:1,2,3,4:1024:10:0,1,2,3,4,5" --group "emb:$E:0.25:1,2,3,4:1024:10:6,7,8,9" \
    --group "csv:outputs/features_cache/phase_$S:0.381:pf_" --group "csv:outputs/features_cache/$S48:0.019:ph_" \
    --group "csv:outputs/features_cache/$S48:0.15:(pore|gr|sp)_" --group "csv:outputs/features_cache/smcdf3_$S:0.2:sc" | tail -1
done
echo DONE
