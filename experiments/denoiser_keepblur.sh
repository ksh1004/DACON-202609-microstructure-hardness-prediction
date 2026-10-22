#!/usr/bin/env bash
# 블러 보존 디노이저: 저품질 열화(대비↓+블러+노이즈) 입력 → 노이즈만 뺀 (대비↓·블러) 정답. 부트스트랩 가중치·정답에서 시작.
# 근거: 기존 디노이저는 블러 입력에 결정립계를 지어내고(--realistic 디블러는 악화), 이 변형은 지어내지 않고 노이즈만 제거.
# 기준(v12a 블록): 부트스트랩 + 저품질 σ48 조명보정 → DINO-L het3@448 L11-17 = 11.337
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
F="^\[|Error|Traceback|OOM|정답|PSNR 입력 [0-9.]+ → D4|it[0-9]+000 "
T=kb_boot_i10000
$PY -W ignore src/train_denoiser.py --iters 10000 --realistic --keep-blur --target-src outputs/denoised/unet_cm7_am22_l1_i6000.npz --init weights/denoiser_cm7_am22_l1_i10000_boot.pt --tag $T 2>&1 | grep --line-buffered -E "$F"
$PY src/flatfield_cache.py --src outputs/denoised/unet_$T.npz --mode lowpass --sigma 48 --only-low | grep "^\["
for v in "" _fflowpass48_low; do
  $PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/unet_$T$v.npz 2>&1 | grep --line-buffered -E "^\[|Error|Traceback|OOM"
  $PY src/block_ridge.py --src outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_Sunet_$T$v --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dn_$T${v}_L11-17
done
