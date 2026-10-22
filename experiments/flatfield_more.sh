#!/usr/bin/env bash
# (1) 디노이즈 전 원본 저품질 이미지 조명 보정(σ48) → 부트스트랩 디노이저 (2) 다른 디노이저 3종 출력에 저품질 σ48 보정 → 각 DINO 프로브 + L11-17
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
probe(){ $PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_$1.npz 2>&1 | grep -E "^\[|Error|Traceback"
  $PY src/block_ridge.py --src outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_Sunet_$1 --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dn_$1_L11-17; }
$PY src/flatfield_cache.py --src outputs/denoised/raw.npz --mode lowpass --sigma 48 --only-low | grep "^\["
$PY -W ignore src/denoise_with.py --weights weights/denoiser_cm7_am22_l1_i10000_boot.pt --src outputs/denoised/raw_fflowpass48_low.npz --tag boot_on_rawff48low
probe boot_on_rawff48low
for d in boot2_i10000 cm7_am22_l1_i10000_boot_c256 cm7_am22_l1_i10000_boot_nb; do
  $PY src/flatfield_cache.py --src outputs/denoised/unet_$d.npz --mode lowpass --sigma 48 --only-low | grep "^\["
  probe ${d}_fflowpass48_low
done
