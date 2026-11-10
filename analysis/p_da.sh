#!/usr/bin/env bash
# 측정 블록 P(mr_v14_pp_sc)의 DINO 4보기 부분을 Depth-Anything 4보기로 교체 / 둘 다
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
B=unet_cm7_am22_l1_i10000_boot_fflowpass48_low
E4=outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2_S$B
EA=outputs/probe2/depth-anything__Depth-Anything-V2-Large-hf_448_L9-11-13-15-17-19-21_Pphase2_S$B
MEAS="--group csv:outputs/features_cache/phase_$B:0.381:pf_ --group csv:outputs/features_cache/$B:0.019:ph_ --group csv:outputs/features_cache/$B:(0.15):(pore|gr|sp)_ --group csv:outputs/features_cache/smcdf3_$B:0.2:sc"
MEAS=${MEAS//(0.15)/0.15}
dg() { echo "--group emb:$1:$3:$2:1024:10:0,1,2,3,4,5 --group emb:$1:$(awk "BEGIN{print 0.25*$3}"):$2:1024:10:6,7,8,9"; }
$PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name mr_v14_pp_sc_da_a0.01 $(dg $EA 1,2,3,4 1) $MEAS | tail -1
$PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name mr_v14_pp_sc_da01234_a0.01 $(dg $EA 0,1,2,3,4 1) $MEAS | tail -1
$PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name mr_v14_pp_sc_dd_a0.01 $(dg $E4 1,2,3,4 0.5) $(dg $EA 1,2,3,4 0.5) $MEAS | tail -1
