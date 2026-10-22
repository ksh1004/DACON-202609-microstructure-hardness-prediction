"""형태학 측정 특징 — 디노이즈(+조명보정) 이미지 캐시(npz)에서 이미지별 고정 연산 (학습·정답 미사용).

phase_features.py(상분율)와 같은 발상: DINO가 정밀하게 재지 못하는 물리량을 디노이즈 이미지에서 직접 측정.
- bd_ : 결정립계(얇은 어두운 선) — black top-hat(r=1·2·3) 응답의 임계값별 면적분율 (≈ 결정립계 길이밀도 → 결정립 크기)
- gs_ : 그래뉼로메트리 — opening/closing(disk r=1…12) 후 평균 밝기 변화(기지 밝기로 정규화) = 크기별 구조 스펙트럼
- po_ : 기공·2상 덩어리 — 상대어둡기 rel>t 연결성분의 개수·평균/최대 크기·면적 분위 (t = 0.15·0.25·0.35·0.45)
- ht_ : 상분율 공간 불균일성 — 4×4·8×8 칸별 rel>t 분율의 표준편차·범위

실행: python src/morph_features.py --src outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz
산출물: outputs/features_cache/morph_<stem>_{train,test}.csv
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import morphology

from phase_features import matrix_level

ROOT = Path(__file__).resolve().parents[1]
BD_T = (1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25)
GS_R = (1, 2, 3, 4, 5, 6, 8, 10, 12)
PO_T = (0.15, 0.25, 0.35, 0.45)
HT_T = (0.04, 0.08, 0.15, 0.25)


def _one(img: np.ndarray) -> dict:
    x = img.astype(np.float64)
    f = {}
    m = matrix_level(morphology.closing(x, morphology.disk(2)))
    for r in (1, 2, 3):
        bth = morphology.black_tophat(x, morphology.disk(r))
        for t in BD_T:
            f[f"bd_r{r}_gt{t}"] = float((bth > t).mean())
        f[f"bd_r{r}_mean"] = float(bth.mean() / m)
    base = x.mean()
    for r in GS_R:
        se = morphology.disk(r)
        f[f"gs_open{r}"] = float((base - morphology.opening(x, se).mean()) / m)
        f[f"gs_close{r}"] = float((morphology.closing(x, se).mean() - base) / m)
    c2 = morphology.closing(x, morphology.disk(2))
    rel = (m - c2) / m
    for t in PO_T:
        lab, n = ndi.label(rel > t)
        sz = np.bincount(lab.ravel())[1:] if n else np.zeros(1)
        f[f"po_t{t}_n"] = float(n)
        f[f"po_t{t}_mean"] = float(sz.mean())
        f[f"po_t{t}_max"] = float(sz.max())
        for q in (50, 90):
            f[f"po_t{t}_q{q}"] = float(np.percentile(sz, q))
        f[f"po_t{t}_n_ge20"] = float((sz >= 20).sum())
    for t in HT_T:
        b = (rel > t).astype(np.float64)
        for g in (4, 8):
            s = 256 // g
            tile = b.reshape(g, s, g, s).mean((1, 3))
            f[f"ht_t{t}_g{g}_std"] = float(tile.std())
            f[f"ht_t{t}_g{g}_rng"] = float(tile.max() - tile.min())
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/sample_submission.csv")
    z = np.load(ROOT / a.src)
    stem = Path(a.src).stem
    fd = ROOT / "outputs/features_cache"
    t0 = time.time()
    for split, ids in (("train", tr.ID), ("test", te.ID)):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(_one, list(z[split]), chunksize=16))
        df = pd.DataFrame(rows).assign(ID=ids.values)
        df.to_csv(fd / f"morph_{stem}_{split}.csv", index=False)
        print(f"[morph_{stem}] {split} {df.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
