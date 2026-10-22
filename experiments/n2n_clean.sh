#!/usr/bin/env bash
# 고품질(σ<10) 잔여 노이즈 Noisier2Noise 보정 → σ48 저품질 조명보정 → DINO + 상분율. 기준: DINO 11.337 / +pf 10.849 (고품질 9.41)
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
$PY -W ignore src/n2n_clean.py --th 10 --k 8 2>&1 | grep -vi warn
B=cm7_am22_l1_i10000_boot_n2n10
$PY src/flatfield_cache.py --src outputs/denoised/unet_$B.npz --mode lowpass --sigma 48 --only-low | grep "^\["
S=unet_${B}_fflowpass48_low
$PY -W ignore src/probe_models.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 9 11 13 15 17 19 21 --pool hetero3 --bs 8 --src outputs/denoised/$S.npz 2>&1 | grep -E "^\[|Error|Traceback|OOM"
$PY -W ignore src/phase_features.py --src outputs/denoised/$S.npz --workers 6 | tail -1
P=outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_S$S
$PY src/block_ridge.py --src $P --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dn_n2n10_ff48_L11-17 | grep "^\["
$PY src/block_ridge.py --src $P --nb 6 --mult 1 1 1 1 1 1 --layers 1 2 3 4 --name dnpf_n2n10_ff48_L11-17 --extra 0.4 outputs/features_cache/phase_$S::pf_ | grep "^\["
