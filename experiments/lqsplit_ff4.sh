#!/usr/bin/env bash
# 저품질 특징 배율 + 모드 분리(multi_ridge --lq-gain G --lq-split): 고품질 이미지는 기존 모델, 저품질 이미지는 저품질 특징을 G배 한 모델로 예측.
# 근거(2026-10-13 밤, analysis/lq_gain2.py·lq_gain3.py): 3계열 σ48 블렌드 OOF 10.563 → 10.325 (G=1.5, Δ제곱 −4.96±1.99, 폴드 5/5),
#   저품질 12.39 → 11.91, 고품질은 그대로. 모드 전용 가중치(다중 과제)는 악화 → 두 모드는 같은 관계를 공유하고 저품질만 신호가 눌려 있음.
# 사용: bash experiments/lqsplit_ff4.sh <G>   → emb_reg/<기존 이름>_m<G>   (CPU만 사용, 특징 재추출 없음)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8; G=${1:-1.5}
D=vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21; T=vit_so400m_patch14_reg1_tipsv2.webli_448_L8-10-12-14
run() { n=$1; shift; [ -f outputs/emb_reg/$n/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --lq-gain $G --lq-split --name $n "$@" | tail -1; }
for s in 48 32 64 96; do
  S=unet_cm7_am22_l1_i10000_boot_fflowpass${s}_low
  E4=outputs/probe2/${D}_Pphase2_S$S; E16=outputs/probe2/${D}_Pphase2m_T8_S${S}_sh2; ET=outputs/probe2/${T}_Pphase2m_T8_S${S}_sh2
  run mv16_dnppm0.05_ff${s}_m$G --group "emb:$E16:1:1,2,3,4:1024:11:0,1,2,3,4,5" --group "emb:$E16:0.25:1,2,3,4:1024:11:6,7,8,9" --group "emb:$E16:0.05:1,2,3,4:1024:11:10"
  run tips16_dnppm0.05_ff${s}_m$G --group "emb:$ET:1:0,1,2,3:1152:11:0,1,2,3,4,5" --group "emb:$ET:0.25:0,1,2,3:1152:11:6,7,8,9" --group "emb:$ET:0.05:0,1,2,3:1152:11:10"
  for mn in densenet121.tv_in1k:dn121 resnet50.a1_in1k:r50a1 resnet50.a2_in1k:r50a2; do
    EC=outputs/probe2/cnn_${mn%%:*}_s2_512_Pphase2m_T8_S$S
    run cnnf_${mn##*:}_ff${s}_m$G --group "emb:$EC:1:0:512:11:0,1,2,3,4,5" --group "emb:$EC:0.25:0:512:11:6,7,8,9" --group "emb:$EC:0.05:0:512:11:10"
  done
  run dnpp0.25_ff${s}_m$G --group "emb:$E4:1:1,2,3,4:1024:10:0,1,2,3,4,5" --group "emb:$E4:0.25:1,2,3,4:1024:10:6,7,8,9"
done
S=unet_cm7_am22_l1_i10000_boot_fflowpass48_low; E4=outputs/probe2/${D}_Pphase2_S$S
PG=(--group "emb:$E4:1:1,2,3,4:1024:10:0,1,2,3,4,5" --group "emb:$E4:0.25:1,2,3,4:1024:10:6,7,8,9" --group "csv:outputs/features_cache/phase_$S:0.381:pf_" --group "csv:outputs/features_cache/$S:0.019:ph_" --group "csv:outputs/features_cache/$S:0.15:(pore|gr|sp)_" --group "csv:outputs/features_cache/smcdf3_$S:0.2:sc")
run mr_v14_pp_sc_m$G "${PG[@]}"
[ -f outputs/emb_reg/mr_v14_pp_sc_mc$G/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --lq-gain $G --lq-gain-csv $G --lq-split --name mr_v14_pp_sc_mc$G "${PG[@]}" | tail -1
echo DONE
