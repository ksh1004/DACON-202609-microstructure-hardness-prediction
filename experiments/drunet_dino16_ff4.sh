#!/usr/bin/env bash
# 두 번째 디노이즈 경로: 공개 사전학습 DRUNet(gray, KAIR, MIT, 2021 공개; 이미지별 추정 σ를 넣는 비블라인드)으로 원본을 디노이즈
# → 저품질만 조명 보정(σ 32/48/64/96) → DINOv2-L 16보기 phase2m 특징 → 기존 U-Net 경로의 같은 특징과 한 Ridge에 결합(고정 벌점).
# 근거(2026-10-14 밤, analysis/twopath.py, σ48·8보기): DINOv2 계열 단독 10.85 → 결합 10.55, 3계열 블렌드 10.336 → 10.187
#   (폴드 4/5, 가운데 80%에서도 개선, 고·저품질 모두 개선). TIPSv2·고정 CNN 계열은 결합 이득이 작음(−0.03 이하).
AR=${1:-0.01}; PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
M=vit_large_patch14_dinov2.lvd142m; L="9 11 13 15 17 19 21"
[ -f outputs/denoised/drunet.npz ] || $PY -W ignore analysis/drunet_denoise.py
one() { echo "--group emb:$1:1:1,2,3,4:1024:11:0,1,2,3,4,5 --group emb:$1:0.25:1,2,3,4:1024:11:6,7,8,9 --group emb:$1:0.05:1,2,3,4:1024:11:10"; }
for s in 48 32 64 96; do
  [ -f outputs/denoised/drunet_fflowpass${s}_low.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/drunet.npz --mode lowpass --sigma $s --only-low | tail -1
  EU=outputs/probe2/${M}_448_L9-11-13-15-17-19-21_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass${s}_low_sh2
  ED=outputs/probe2/${M}_448_L9-11-13-15-17-19-21_Pphase2m_T8_Sdrunet_fflowpass${s}_low_sh2
  [ -f $ED/emb.npz ] || $PY -W ignore src/probe_models.py --model $M --size 448 --layers $L --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/drunet_fflowpass${s}_low.npz 2>&1 | grep -E "probe OOF|Error"
  $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name mv16dr_ff${s}_a$AR $(one $ED) | tail -1
  $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name mv16j_ff${s}_a$AR $(one $EU) $(one $ED) | tail -1
done
echo DONE
