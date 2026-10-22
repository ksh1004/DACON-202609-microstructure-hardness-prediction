"""train/test 이미지 디노이즈 캐시 생성 (특징 추출 입력용, 학습 없음·이미지별 독립 처리).

노이즈 σ를 이미지마다 추정(웨이블릿)해 강도를 σ에 비례시킨다 → 깨끗한 이미지는 거의 그대로, 노이즈가 큰 이미지만 강하게 처리.
- tv:  Total Variation (결정립 = 구간별 상수 → 경계 보존), weight = c·σ
- nlm: Non-local means, h = c·σ

실행: python src/denoise_cache.py --method tv --c 0.6
산출물: outputs/denoised/<method>_c<c>.npz  (train: N×256×256 uint8, test: M×256×256 uint8)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.restoration import denoise_nl_means, denoise_tv_chambolle, estimate_sigma

ROOT = Path(__file__).resolve().parents[1]


def _one(args):
    split, id_, method, c = args
    x = np.array(Image.open(ROOT / "data" / split / f"{id_}.png")).astype(np.float64) / 255.0
    s = float(estimate_sigma(x))
    if method == "tv":
        d = denoise_tv_chambolle(x, weight=max(1e-3, c * s))
    else:
        d = denoise_nl_means(x, h=max(1e-3, c * s), sigma=s, patch_size=5, patch_distance=6, fast_mode=True)
    return np.clip(np.round(d * 255), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["tv", "nlm"], default="tv")
    ap.add_argument("--c", type=float, default=0.6)
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data" / "train.csv").ID
    te = pd.read_csv(ROOT / "data" / "sample_submission.csv").ID
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 4)) as ex:
        A = np.stack(list(ex.map(_one, [("train", i, a.method, a.c) for i in tr], chunksize=8)))
        B = np.stack(list(ex.map(_one, [("test", i, a.method, a.c) for i in te], chunksize=8)))
    out = ROOT / "outputs" / "denoised"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{a.method}_c{a.c}.npz", train=A, test=B)
    print(f"[{a.method}_c{a.c}] train {A.shape} test {B.shape} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
