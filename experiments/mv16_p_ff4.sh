#!/usr/bin/env bash
# 측정 블록(P)을 16보기 특징으로: σ48(주) + σ32/64/96 (σ 선택 운을 평균으로 없앤 변형용). 수작업 ph_/pgs는 σ48 것 공유(mr_ff4.sh와 동일).
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8
M=vit_large_patch14_dinov2.lvd142m
S48=unet_cm7_am22_l1_i10000_boot_fflowpass48_low
for s in 48 32 64 96; do
  S=unet_cm7_am22_l1_i10000_boot_fflowpass${s}_low
  E=outputs/probe2/${M}_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${S}_sh2
  $PY -W ignore src/multi_ridge.py --name mv16_mr_pp_sc_ff$s --group "emb:$E:1:1,2,3,4:1024:11:0,1,2,3,4,5" --group "emb:$E:0.25:1,2,3,4:1024:11:6,7,8,9" \
    --group "csv:outputs/features_cache/phase_$S:0.381:pf_" --group "csv:outputs/features_cache/$S48:0.019:ph_" \
    --group "csv:outputs/features_cache/$S48:0.15:(pore|gr|sp)_" --group "csv:outputs/features_cache/smcdf3_$S:0.2:sc" | tail -1
done
echo DONE
