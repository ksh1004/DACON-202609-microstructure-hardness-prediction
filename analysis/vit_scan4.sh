#!/usr/bin/env bash
# 남은 대형 ViT 후보를 지금 풀링(phase2m, σ48, D4 8보기)으로 재확인 (2026-10-14): 같은 계열의 더 큰 모델·레지스터 판, 조밀 특징 정렬 모델, MIM·JEPA 계열
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
S=outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz
run() {  # 모델 크기 채널 배치 층...
  M=$1; SZ=$2; C=$3; BS=$4; shift 4
  R=""; [ "$SZ" != "448" ] && R="_r$SZ"
  TAG="${R}_m$(echo $M | sed 's/vit_//; s/\./-/g')"
  $PY -W ignore analysis/view_embed.py --src $S --splits train --shifts 0 --model $M --size $SZ --layers "$@" --bs $BS 2>&1 | grep -E "Error|Traceback|rror"
  echo "##### $M @$SZ"
  $PY -W ignore analysis/model_eval.py "$TAG" $# $C "$@" 2>&1 | grep -v "Warning\|기준\|sklearnex"
}
run vit_large_patch14_reg4_dinov2.lvd142m 448 1024 8 7 9 11 13 15 17 19 21
run vit_giant_patch14_dinov2.lvd142m 448 1536 4 16 19 22 25 28 31 34 37
run vit_pe_spatial_large_patch14_448.fb 448 1024 8 7 9 11 13 15 17 19 21
run eva02_large_patch14_224.mim_m38m 448 1024 8 7 9 11 13 15 17 19 21
run vit_huge_patch14_gap_224.in22k_ijepa 448 1280 4 10 13 16 19 22 25 28 31
echo ALLDONE
