#!/usr/bin/env bash
# 네 번째 이미지 경로 'DRUNet 미세조정'(analysis/drunet_ft.py → outputs/denoised/drunetft_<tag>.npz):
# 공개 DRUNet(gray)을 DRUNet이 정리한 고품질 train 이미지(σ<10, 291장)를 정답으로 미세조정(입력 = 정답 + 합성 가우시안 노이즈, 노이즈 지도 = 실제 σ).
# → DINOv2·TIPSv2 16보기 (조명보정 σ 4종) → Ridge(고정 벌점): DINOv2는 U-Net 부트스트랩 + DRUNet + DRT + 미세조정 네 경로 결합(mv16j4),
#   TIPSv2는 미세조정 경로 단독(블렌드에서 다른 경로와 예측 평균).
# 근거(10/15, σ48·8보기): U+D+X 결합 대비 U+D+X+F 블렌드 −0.027 (4/5), F 단독 10.570 (DRT 10.518, DRUNet 11.111).
TAG=${1:-ft1b}; AR=0.01; XT=drt10b_i10000; PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
D=vit_large_patch14_dinov2.lvd142m; T=vit_so400m_patch14_reg1_tipsv2.webli; L="9 11 13 15 17 19 21"
oneD() { echo "--group emb:$1:1:1,2,3,4:1024:11:0,1,2,3,4,5 --group emb:$1:0.25:1,2,3,4:1024:11:6,7,8,9 --group emb:$1:0.05:1,2,3,4:1024:11:10"; }
oneT() { echo "--group emb:$1:1:0,1,2,3:1152:11:0,1,2,3,4,5 --group emb:$1:0.25:0,1,2,3:1152:11:6,7,8,9 --group emb:$1:0.05:0,1,2,3:1152:11:10"; }
for s in 48 32 64 96; do
  F=drunetft_${TAG}_fflowpass${s}_low
  [ -f outputs/denoised/$F.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/drunetft_${TAG}.npz --mode lowpass --sigma $s --only-low | tail -1
  EF=outputs/probe2/${D}_448_L9-11-13-15-17-19-21_Pphase2m_T8_S${F}_sh2
  [ -f $EF/emb.npz ] || $PY -W ignore src/probe_models.py --model $D --size 448 --layers $L --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/$F.npz 2>&1 | grep -E "probe OOF|Error"
  ET=outputs/probe2/${T}_448_L8-10-12-14_Pphase2m_T8_S${F}_sh2
  [ -f $ET/emb.npz ] || $PY -W ignore src/probe_models.py --model $T --size 448 --layers 8 10 12 14 --pool phase2m --tta 8 --shifts 2 --bs 8 --src outputs/denoised/$F.npz 2>&1 | grep -E "probe OOF|Error"
  EU=outputs/probe2/${D}_448_L9-11-13-15-17-19-21_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass${s}_low_sh2
  ED=outputs/probe2/${D}_448_L9-11-13-15-17-19-21_Pphase2m_T8_Sdrunet_fflowpass${s}_low_sh2
  EX=outputs/probe2/${D}_448_L9-11-13-15-17-19-21_Pphase2m_T8_Sunet_${XT}_fflowpass${s}_low_sh2
  # DINOv2 결합 구성 4종: 네 경로 / U-Net 부트스트랩 경로 제외(DRUNet+DRT+미세조정) / DRUNet+미세조정 / 미세조정 단독
  # (σ48 시험: U 경로를 빼면 다른 구성원(TIPSv2·CNN·측정 블록, 모두 U 경로)과 오차가 덜 겹쳐 블렌드가 좋아짐)
  R() { local n=$1; shift; [ -f outputs/emb_reg/$n/test_pred_full.csv ] || $PY -W ignore src/multi_ridge.py --alpha-rel $AR --name $n "$@" | tail -1; }
  R mv16j4_${TAG}_ff${s}_a$AR $(oneD $EU) $(oneD $ED) $(oneD $EX) $(oneD $EF)
  R mv16dxf_${TAG}_ff${s}_a$AR $(oneD $ED) $(oneD $EX) $(oneD $EF)
  R mv16df_${TAG}_ff${s}_a$AR $(oneD $ED) $(oneD $EF)
  R mv16f_${TAG}_ff${s}_a$AR $(oneD $EF)
  R tips16_ft${TAG}_ff${s}_a$AR $(oneT $ET)
done
echo ALLDONE
