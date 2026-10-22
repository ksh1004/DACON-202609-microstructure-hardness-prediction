#!/usr/bin/env bash
# 측정 블록 P(DINO 4보기 + 상분율·수작업·평활 CDF)의 측정 특징을 다른 디노이즈 경로에서 계산한 변형 (λ 0.01)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
E="outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2_Sunet_cm7_am22_l1_i10000_boot_fflowpass48_low"
DG="--group emb:$E:1:1,2,3,4:1024:10:0,1,2,3,4,5 --group emb:$E:0.25:1,2,3,4:1024:10:6,7,8,9"
U=unet_cm7_am22_l1_i10000_boot_fflowpass48_low; X=unet_drt10b_i10000_fflowpass48_low; D=drunet_fflowpass48_low
meas() { local s=$1 w=$2; echo "--group csv:outputs/features_cache/phase_$s:$(awk "BEGIN{print 0.381*$w}"):pf_ --group csv:outputs/features_cache/$s:$(awk "BEGIN{print 0.019*$w}"):ph_ --group csv:outputs/features_cache/$s:$(awk "BEGIN{print 0.15*$w}"):(pore|gr|sp)_ --group csv:outputs/features_cache/smcdf3_$s:$(awk "BEGIN{print 0.2*$w}"):sc"; }
run() { local n=$1; shift; [ -f outputs/emb_reg/$n/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name $n $DG "$@" 2>&1 | grep -E "OOF|oof" | tail -2; echo "[$n]"; }
run mrP_X_a0.01 $(meas $X 1)
run mrP_D_a0.01 $(meas $D 1)
run mrP_UX_a0.01 $(meas $U 0.5) $(meas $X 0.5)
run mrP_UXD_a0.01 $(meas $U 0.333) $(meas $X 0.333) $(meas $D 0.333)
run mrP_UX1_a0.01 $(meas $U 1) $(meas $X 1)
echo ALLDONE
