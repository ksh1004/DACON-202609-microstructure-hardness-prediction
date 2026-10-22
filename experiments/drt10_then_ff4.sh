#!/usr/bin/env bash
# 1) 정답을 σ<10(291장)으로 늘린 증류 디노이저 2종(drt10, drt10b) 학습 → σ48·8보기 DINOv2 경로 비교(drt2와)
# 2) 가장 나은 디노이저로 전체 4보정 추출(experiments/drt_ff4.sh)
PY=./.venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=8
[ -f outputs/denoised/unet_drt10_i10000.npz ] || $PY -W ignore src/train_denoiser.py --iters 10000 --clean-max 10 --target-src outputs/denoised/drunet.npz --tag drt10_i10000 2>&1 | grep -E "정답|it10000|저장"
[ -f outputs/denoised/unet_drt10b_i10000.npz ] || $PY -W ignore src/train_denoiser.py --iters 10000 --clean-max 10 --target-src outputs/denoised/unet_drt10_i10000.npz --tag drt10b_i10000 2>&1 | grep -E "it10000|저장"
for T in drt10_i10000 drt10b_i10000; do
  [ -f outputs/denoised/unet_${T}_fflowpass48_low.npz ] || $PY -W ignore src/flatfield_cache.py --src outputs/denoised/unet_$T.npz --mode lowpass --sigma 48 --only-low | tail -1
  [ -f outputs/probe2_views/unet_${T}_fflowpass48_low/train_s0_k7.npy ] || $PY -W ignore analysis/view_embed.py --src outputs/denoised/unet_${T}_fflowpass48_low.npz --splits train --shifts 0 --bs 8 2>&1 | grep -E "DONE|rror"
done
$PY -W ignore analysis/pathx.py unet_drt2_i10000_fflowpass48_low unet_drt10_i10000_fflowpass48_low unet_drt10b_i10000_fflowpass48_low 2>&1 | grep -v sklearnex | tee logs/drt10_cmp.txt
BEST=$($PY - <<'PY'
import re
t = open("logs/drt10_cmp.txt", encoding="utf-8").read(); best = None
for blk in re.split(r"\n  \[", t)[1:]:
    name = blk.split("]")[0]; m = re.search(r"U\+D\+X\s+결합 [\d.]+ .*?→ 블렌드 ([\d.]+)", blk)
    if m and (best is None or float(m.group(1)) < best[0]): best = (float(m.group(1)), name)
print(best[1].replace("unet_", "").replace("_fflowpass48_low", ""))
PY
)
echo "선택한 디노이저: $BEST"
bash experiments/drt_ff4.sh $BEST 0.01
echo ALLDONE
