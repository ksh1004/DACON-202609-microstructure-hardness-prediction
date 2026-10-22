#!/usr/bin/env bash
# 모든 Ridge 구성원을 고정 벌점(--alpha-rel, 기본 0.01)으로 다시 적합 (특징 재추출 없음, CPU만). 이름 접미사 _a<값>.
# 근거(2026-10-14 밤, analysis/lam_sweep.py): 구성원별 LOO α 대신 작은 고정 α를 쓰면 3계열 블렌드 OOF 10.444 → 10.30
#   (폭 보정 뒤 비교 10.41 → 10.28, 폴드 5/5, 폴드 밖 선택도 가장 작은 값), 구성원 간 잔차상관 0.873 → 0.83.
# 사용: bash experiments/ridge_alpha_ff4.sh [0.01]
AR=${1:-0.01}; PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
D=vit_large_patch14_dinov2.lvd142m; T=vit_so400m_patch14_reg1_tipsv2.webli; BOOT=unet_cm7_am22_l1_i10000_boot
embC() { echo outputs/probe2/cnn_${1/@/_s}_512_Pphase2m_T8_S${BOOT}_fflowpass${2}_low; }
one() { echo "--group emb:$1:1:$2:$3:11:0,1,2,3,4,5 --group emb:$1:0.25:$2:$3:11:6,7,8,9 --group emb:$1:0.05:$2:$3:11:10"; }
for s in 48 32 64 96; do
  S=${BOOT}_fflowpass${s}_low
  E4=outputs/probe2/${D}_448_L9-11-13-15-17-19-21_Pphase2_S$S; E16=outputs/probe2/${D}_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${S}_sh2; ET=outputs/probe2/${T}_448_L8-10-12-14_Pphase2m_T8_S${S}_sh2
  $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name dnpp0.25_ff${s}_a$AR --group "emb:$E4:1:1,2,3,4:1024:10:0,1,2,3,4,5" --group "emb:$E4:0.25:1,2,3,4:1024:10:6,7,8,9" | tail -1
  $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name mv16_dnppm0.05_ff${s}_a$AR $(one $E16 1,2,3,4 1024) | tail -1
  $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name tips16_dnppm0.05_ff${s}_a$AR $(one $ET 0,1,2,3 1152) | tail -1
  $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name cnnfj4_ff${s}_a$AR $(for M in densenet121.tv_in1k@2 resnet50.a1_in1k@2 resnet50.a2_in1k@2 densenetblur121d.ra_in1k@2; do one $(embC $M $s) 0 512; done) | tail -1
done
S48=${BOOT}_fflowpass48_low; E4=outputs/probe2/${D}_448_L9-11-13-15-17-19-21_Pphase2_S$S48
$PY -W ignore src/multi_ridge.py --alpha-rel $AR --name mr_v14_pp_sc_a$AR --group "emb:$E4:1:1,2,3,4:1024:10:0,1,2,3,4,5" --group "emb:$E4:0.25:1,2,3,4:1024:10:6,7,8,9" \
  --group "csv:outputs/features_cache/phase_$S48:0.381:pf_" --group "csv:outputs/features_cache/$S48:0.019:ph_" --group "csv:outputs/features_cache/$S48:0.15:(pore|gr|sp)_" --group "csv:outputs/features_cache/smcdf3_$S48:0.2:sc" | tail -1
echo DONE
