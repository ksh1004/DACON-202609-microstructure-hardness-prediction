"""결정립 규모 평활 밝기 CDF — 디노이즈(+조명보정) 이미지 캐시(npz)에서 이미지별 고정 연산 (학습·정답 미사용).

가우시안 σ(7·8·9·10) 평활 이미지 g의 기지 밝기(최빈값) 대비 상대 어둡기 rel=(m−g)/m 의 누적 분포(−0.2~0.6, 0.025 간격)와 분위.
σ≈결정립 크기 수준에서 본 2상 분포 (2026-10-07: 블록 OOF 10.720→10.564, σ7~10·디노이저 3종 모두 일관 개선.
평균·중앙값 기준이나 σ1~6·24, 중앙값 필터는 무익 → 기지(최빈) 기준이 핵심).

실행: python src/smooth_cdf_features.py --src outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz
산출물: outputs/features_cache/smcdf3_<stem>_{train,test}.csv ('sc{σ}_' 열)
"""
from __future__ import annotations

import argparse
from functools import partial
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_features import matrix_level  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T = np.round(np.arange(-0.2, 0.601, 0.025), 3)
SC = (7, 8, 9, 10)


def _one(img: np.ndarray, matrix: str = "mode") -> dict:
    x = img.astype(float)
    f = {}
    for s in SC:
        g = ndi.gaussian_filter(x, s)
        m = matrix_level(g, matrix)
        rel = (m - g) / m
        for t in T:
            f[f"sc{s}_gt{t:+.3f}"] = float((rel > t).mean())
        for q in (5, 25, 50, 75, 95):
            f[f"sc{s}_q{q}"] = float(np.percentile(rel, q))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--matrix", choices=["mode", "bright"], default="mode")
    a = ap.parse_args()
    stem = Path(a.src).stem + ("_mb" if a.matrix == "bright" else "")
    z = np.load(ROOT / a.src)
    (ROOT / "outputs/features_cache").mkdir(parents=True, exist_ok=True)
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        ids = pd.read_csv(ROOT / "data" / csv).ID
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(partial(_one, matrix=a.matrix), list(z[split]), chunksize=16))
        pd.DataFrame(rows).assign(ID=ids.values).to_csv(ROOT / "outputs/features_cache" / f"smcdf3_{stem}_{split}.csv", index=False)
        print(f"[smcdf3_{stem}] {split} {len(rows)}장", flush=True)


if __name__ == "__main__":
    main()
