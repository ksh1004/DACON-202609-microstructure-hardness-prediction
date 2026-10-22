#!/usr/bin/env bash
# 세 번째 이미지 경로 'DRT': 자체 U-Net 디노이저를 공개 DRUNet이 정리한 고품질 train 이미지(σ<7)를 정답으로 다시 학습
# (src/train_denoiser.py --target-src outputs/denoised/drunet.npz --tag drt_i10000; 입력은 정답 + 합성 가우시안 노이즈, test·라벨 미사용).
# 고품질 이미지의 잔여 노이즈를 지우면서 저품질에서는 자체 U-Net의 결정립 사전 지식을 유지한다.
# → DINOv2·TIPSv2 16보기, 고정 CNN 4종 8보기 특징 (조명보정 σ 4종) → Ridge(고정 벌점):
#   DINOv2: U-Net 부트스트랩 + DRUNet + DRT 세 경로를 한 Ridge에 (mv16j3), TIPSv2·CNN: DRT 경로 단독(블렌드에서 기존 경로와 예측 평균).
# 근거(2026-10-14 밤, σ48·8보기): 3계열 블렌드 10.187 → 10.093 (pw13 8.802 → 8.692, 폴드 4/5).
TAG=${1:-drt_i10000}; AR=${2:-0.01}; PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
D=vit_large_patch14_dinov2.lvd142m; T=vit_so400m_patch14_reg1_tipsv2.webli; L="9 11 13 15 17 19 21"
oneD() { echo "--group emb:$1:1:1,2,3,4:1024:11:0,1,2,3,4,5 --group emb:$1:0.25:1,2,3,4:1024:11:6,7,8,9 --group emb:$1:0.05:1,2,3,4:1024:11:10"; }
oneT() { echo "--group emb:$1:1:0,1,2,3:1152:11:0,1,2,3,4,5 --group emb:$1:0.25:0,1,2,3:1152:11:6,7,8,9 --group emb:$1:0.05:0,1,2,3:1152:11:10"; }
oneC() { echo "--group emb:$1:1:0:512:11:0,1,2,3,4,5 --group emb:$1:0.25:0:512:11:6,7,8,9 --group emb:$1:0.05:0:512:11:10"; }
for s in 48 32 64 96; do
  X=unet_${TAG}_fflowpass${s}_low
  [ -f outputs/denoised/$X.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/unet_${TAG}.npz --mode lowpass --sigma $s --only-low | tail -1
  EX=outputs/probe2/${D}_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${X}_sh2
  [ -f $EX/emb.npz ] || $PY -W ignore src/probe_models.py --model $D --size 448 --layers $L --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/$X.npz 2>&1 | grep -E "probe OOF|Error"
  ET=outputs/probe2/${T}_448_L8-10-12-14_Pphase2m_T8_S${X}_sh2
  [ -f $ET/emb.npz ] || $PY -W ignore src/probe_models.py --model $T --size 448 --layers 8 10 12 14 --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/$X.npz 2>&1 | grep -E "probe OOF|Error"
  CS=""
  for M in densenet121.tv_in1k@2 resnet50.a1_in1k@2 resnet50.a2_in1k@2 densenetblur121d.ra_in1k@2; do
    EC=outputs/probe2/cnn_${M/@/_s}_512_Pphase2m_T8_S$X
    [ -f $EC/emb.npz ] || $PY -W ignore src/probe_models.py --model cnn:$M --size 512 --pool phase2m --tta 8 --bs 16 --src outputs/denoised/$X.npz 2>&1 | grep -E "probe OOF|Error"
    CS="$CS $(oneC $EC)"
  done
  EU=outputs/probe2/${D}_448_L9-11-13-15-17-19-21_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass${s}_low_sh2
  ED=outputs/probe2/${D}_448_L9-11-13-15-17-19-21_Pphase2m_T8_Sdrunet_fflowpass${s}_low_sh2
  $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name mv16j3_${TAG}_ff${s}_a$AR $(oneD $EU) $(oneD $ED) $(oneD $EX) | tail -1
  $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name tips16_${TAG}_ff${s}_a$AR $(oneT $ET) | tail -1
  $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name cnnfj4_${TAG}_ff${s}_a$AR $CS | tail -1
done
echo DONE
