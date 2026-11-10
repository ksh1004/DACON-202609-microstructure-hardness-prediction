#!/usr/bin/env bash
# 다른 계열 대형 ViT의 고정 특징 구성원 탐색 (train, σ48, D4 8보기, 8개 층 → model_eval로 층 구간·블렌드 평가)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
S=outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz
run() {  # 모델 크기 채널 층...
  M=$1; SZ=$2; C=$3; shift 3
  R=""; [ "$SZ" != "448" ] && R="_r$SZ"
  TAG="${R}_m$(echo $M | sed 's/vit_//; s/\./-/g')"
  $PY -W ignore analysis/view_embed.py --src $S --splits train --shifts 0 --model $M --size $SZ --layers "$@" --bs 8 2>&1 | grep -E "Error|Traceback" 
  echo "##### $M @$SZ"
  $PY -W ignore analysis/model_eval.py "$TAG" $# $C "$@" 2>&1 | grep -v "Warning\|기준"
}
run vit_so400m_patch16_siglip_512.v2_webli 512 1152 6 8 10 12 14 16 18 20
run vit_large_patch14_clip_336.openai 448 1024 7 9 11 13 15 17 19 21
run eva02_large_patch14_clip_336.merged2b 448 1024 7 9 11 13 15 17 19 21
run vit_pe_core_large_patch14_336.fb 448 1024 7 9 11 13 15 17 19 21
run vit_large_patch16_384.augreg_in21k_ft_in1k 512 1024 7 9 11 13 15 17 19 21
run deit3_large_patch16_384.fb_in22k_ft_in1k 512 1024 7 9 11 13 15 17 19 21
run vit_large_patch16_siglip_512.v2_webli 512 1024 7 9 11 13 15 17 19 21
echo ALLDONE
