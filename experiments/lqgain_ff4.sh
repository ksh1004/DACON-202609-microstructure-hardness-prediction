#!/usr/bin/env bash
# 저품질 특징 배율(multi_ridge --lq-gain): 모든 고정 특징 계열 × 조명보정 σ 4종을 배율 g로 재적합 (CPU만 사용, 특징 재추출 없음)
# 근거(2026-10-13 밤): 저품질 모드는 대비 압축(≈0.77배)으로 특징 편차가 줄어 예측이 눌림(기울기 1.08~1.13) → 학습 전에 저품질 특징 ×g
#   σ48 블록: D16 10.809→10.650(g1.3), TIPS 10.934→10.770, DenseNet 11.148→11.081, 3계열 블렌드 10.563→10.415 (폴드 5/5)
# 사용: bash experiments/lqgain_ff4.sh <g>   → emb_reg/<기존 이름>_g<g>
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8; G=${1:-1.3}
D=vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21; T=vit_so400m_patch14_reg1_tipsv2.webli_448_L8-10-12-14
for s in 48 32 64 96; do
  S=unet_cm7_am22_l1_i10000_boot_fflowpass${s}_low
  E4=outputs/probe2/${D}_Pphase2_S$S; E16=outputs/probe2/${D}_Pphase2m_T8_S${S}_sh2; ET=outputs/probe2/${T}_Pphase2m_T8_S${S}_sh2
  $PY -W ignore src/multi_ridge.py --lq-gain $G --name dnpp0.25_ff${s}_g$G --group "emb:$E4:1:1,2,3,4:1024:10:0,1,2,3,4,5" --group "emb:$E4:0.25:1,2,3,4:1024:10:6,7,8,9" | tail -1
  $PY -W ignore src/multi_ridge.py --lq-gain $G --name mv16_dnppm0.05_ff${s}_g$G --group "emb:$E16:1:1,2,3,4:1024:11:0,1,2,3,4,5" --group "emb:$E16:0.25:1,2,3,4:1024:11:6,7,8,9" --group "emb:$E16:0.05:1,2,3,4:1024:11:10" | tail -1
  $PY -W ignore src/multi_ridge.py --lq-gain $G --name tips16_dnppm0.05_ff${s}_g$G --group "emb:$ET:1:0,1,2,3:1152:11:0,1,2,3,4,5" --group "emb:$ET:0.25:0,1,2,3:1152:11:6,7,8,9" --group "emb:$ET:0.05:0,1,2,3:1152:11:10" | tail -1
  for mn in densenet121.tv_in1k:dn121 resnet50.a1_in1k:r50a1 resnet50.a2_in1k:r50a2; do
    EC=outputs/probe2/cnn_${mn%%:*}_s2_512_Pphase2m_T8_S$S
    $PY -W ignore src/multi_ridge.py --lq-gain $G --name cnnf_${mn##*:}_ff${s}_g$G --group "emb:$EC:1:0:512:11:0,1,2,3,4,5" --group "emb:$EC:0.25:0:512:11:6,7,8,9" --group "emb:$EC:0.05:0:512:11:10" | tail -1
  done
done
S=unet_cm7_am22_l1_i10000_boot_fflowpass48_low; E4=outputs/probe2/${D}_Pphase2_S$S
PG=(--group "emb:$E4:1:1,2,3,4:1024:10:0,1,2,3,4,5" --group "emb:$E4:0.25:1,2,3,4:1024:10:6,7,8,9" --group "csv:outputs/features_cache/phase_$S:0.381:pf_" --group "csv:outputs/features_cache/$S:0.019:ph_" --group "csv:outputs/features_cache/$S:0.15:(pore|gr|sp)_" --group "csv:outputs/features_cache/smcdf3_$S:0.2:sc")
$PY -W ignore src/multi_ridge.py --lq-gain $G --name mr_v14_pp_sc_g$G "${PG[@]}" | tail -1
$PY -W ignore src/multi_ridge.py --lq-gain $G --lq-gain-csv $G --name mr_v14_pp_sc_gc$G "${PG[@]}" | tail -1
echo DONE
