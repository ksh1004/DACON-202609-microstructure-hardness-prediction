#!/usr/bin/env bash
# 최종 제출(v32b) 구성원 J3 생성 + 블렌드 — 대회 마지막 날 실제로 실행한 명령을 한 파일로 정리한 것.
#
# 선행 조건 (각 스크립트가 만드는 산출물):
#   - F 경로 디노이즈·조명보정 이미지   : outputs/denoised/drunetft_ft1b_fflowpass{32,48,64,96}_low.npz   (experiments/ft_chain.sh, ft_ff4.sh)
#   - DINOv2-L F 경로 16보기 특징       : outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-..-21_Pphase2m_T8_S<F>_sh2  (experiments/ft_ff4.sh)
#   - 나머지 구성원 Ridge 결과(P·A·CNN·TIPSv2): outputs/emb_reg/...  (experiments/ridge_alpha_ff4.sh, drt_ff4.sh, ft_ff4.sh)
# 이 스크립트가 하는 일:
#   1) Depth-Anything-V2-L 백본 16보기 특징 (F 경로, 4보정)      → outputs/probe2/depth-anything__..._sh2
#   2) Depth-Anything-V2-L 깊이 디코더(neck) 특징 8보기 (4보정)   → outputs/probe2/da_neck_<F>
#   3) J3 = DINOv2 + DA 백본 + DA 디코더(상대 분산 0.3) 한 Ridge  → outputs/emb_reg/joint3_g0.3_ff{s}_a0.01
#   4) 최종 블렌드 (src/final_blend.py)                          → submission/sub_v32b_k103.csv
#      (작업 중에는 analysis/b48_joint3.py --save 로 만들었고, 이 블렌드와 OOF 9.619로 같음 — 차이는 가중치 반올림, 최대 0.016)
PY=${PY:-./.venv/Scripts/python.exe}; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
DA=hf:depth-anything/Depth-Anything-V2-Large-hf; DAT=depth-anything__Depth-Anything-V2-Large-hf
D=vit_large_patch14_dinov2.lvd142m; L="9 11 13 15 17 19 21"; P2=outputs/probe2
one() { echo "--group emb:$1:$3:$2:$4:11:0,1,2,3,4,5 --group emb:$1:$(awk "BEGIN{print 0.25*$3}"):$2:$4:11:6,7,8,9 --group emb:$1:$(awk "BEGIN{print 0.05*$3}"):$2:$4:11:10"; }
for s in 48 32 64 96; do
  F=drunetft_ft1b_fflowpass${s}_low
  ED=$P2/${D}_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${F}_sh2
  EA=$P2/${DAT}_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${F}_sh2
  EN=$P2/da_neck_$F
  [ -f $EA/emb.npz ] || $PY -W ignore src/probe_models.py --model $DA --size 448 --layers $L --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/$F.npz
  [ -f $EN/emb.npz ] || $PY -W ignore analysis/da_neck_embed.py --src outputs/denoised/$F.npz --bs 4
  n=joint3_g0.3_ff${s}_a0.01
  [ -f outputs/emb_reg/$n/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel 0.01 --name $n \
      $(one $ED 0,1,2,3,4 1 1024) $(one $EA 0,1,2,3,4 1 1024) $(one $EN 0,1,2,3 0.3 256)
done
# 최종 블렌드: 측정 블록 0.1765 / DINOv2 4보기 0.0182 / 고정 CNN 결합 0.2171 / TIPSv2 0.063 / J3 0.5252, 예측 폭 ×1.03 (×1.06판은 --scale만 다름)
# 그룹 안의 폴더는 같은 가중으로 평균 (TIPSv2 = U·DRT·F 세 경로, CNN = U·DRT 두 경로, 각각 조명보정 4종)
AR=0.01; FF="32 48 64 96"; mkdir -p submission
$PY -W ignore src/final_blend.py --out submission/sub_v32b_k103.csv --weights 0.1765 0.0182 0.2171 0.063 0.5252 --scale 1.03 \
  --group phase outputs/emb_reg/mr_v14_pp_sc_a$AR \
  --group dino4 $(for s in $FF; do echo -n "outputs/emb_reg/dnpp0.25_ff${s}_a$AR "; done) \
  --group cnn $(for p in cnnfj4 cnnfj4_drt10b_i10000; do for s in $FF; do echo -n "outputs/emb_reg/${p}_ff${s}_a$AR "; done; done) \
  --group tips $(for p in tips16_dnppm0.05 tips16_drt10b_i10000 tips16_ftft1b; do for s in $FF; do echo -n "outputs/emb_reg/${p}_ff${s}_a$AR "; done; done) \
  --group joint3 $(for s in $FF; do echo -n "outputs/emb_reg/joint3_g0.3_ff${s}_a$AR "; done)
